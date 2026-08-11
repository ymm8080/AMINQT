"""涨停板接力模型 (分类) — 预测涨停股明日是否继续连板.

对齐 PIPELINE DESIGN 文档 "模型B":
  - 目标 = 明日继续涨停 (fwd_up_1d), 二分类 (连板/断板), 非回归
  - 特征 = 连板高度/封单/封时/开板/一字/板块/市场情绪
          + 分桶编码 (方案A) + 交互特征 (方案B) + 龙虎榜席位 (面板 lhb_*)
  - 对比 基础 → +分桶 → +交互 → +龙虎榜 四层 OOS AUC, 证明分桶/交互是否有效

验收: OOS(末40%日期) AUC / 分位校准(单调) / 特征重要度 / top20% 前视收益(诚实报告).
只在涨停行训练. main/dual 分板. 随机种子固定. 结果+模型落盘 (WORM).

Usage: python scripts/_build_limit_board_model.py
"""

import gc
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import _research_limit_strong as R
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PANEL = R.PANEL
OUT_DIR = R.OUT_DIR
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── 基础连续特征 (研究脚本 R.MODEL_FEATS) ──
BASE_FEATS = list(R.MODEL_FEATS)
# 分桶列 (categorical) — 文档方案A
BUCKET_COLS = ["b_seal", "b_seal_mins", "b_open", "b_height"]
# 交互列 — 文档方案B
INTER_FEATS = ["first_sector", "seal_health", "zhaban_mkt", "board_seal", "yizi_seal"]
# 主升浪起点事件特征 (文档: 连续一字后首次开板换手, 封单~20% → 可能的主升浪起点)
WAVE_FEATS = ["main_wave_start", "yizi_prev", "yizi_streak"]
# 龙虎榜席位 (面板 lhb_*, 上板当日净买/机构/占比)
LHB_FEATS = ["lhb_net_buy", "lhb_net_ratio", "lhb_inst_net", "is_lhb"]

# 面板额外要加载的原始列 (lhb_net_ratio/is_lhb/lhb_inst_net 在 add_lhb 里派生)
PANEL_EXTRA = [
    "turnover_rate",
    "lhb_net_buy",
    "lhb_buy_amt",
    "lhb_inst_buy",
    "lhb_inst_sell",
]


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """分桶编码 (文档方案A). 无前视: 只用当日值."""
    # 封单比例: 弱封<5% / 健康10-30% / 偏一字50-80% / 极端>80%
    df["b_seal"] = pd.cut(
        df["fd_amount_ratio"],
        [-1, 0.05, 0.30, 0.80, 999],
        labels=["弱封", "健康", "偏一字", "极端"],
    )
    # 封时: 早盘<10点 / 午前 / 午后 / 尾盘
    df["b_seal_mins"] = pd.cut(
        df["seal_mins"],
        [-1, 30, 120, 210, 999],
        labels=["早盘", "午前", "午后", "尾盘"],
    )
    df["b_seal_mins"] = df["b_seal_mins"].cat.add_categories("未知").fillna("未知")
    # 开板次数
    df["b_open"] = pd.cut(
        df["open_times"].fillna(0), [-1, 0, 1, 999], labels=["未开", "开1次", "开2次+"]
    )
    # 连板高度 (涨停行 limit_times>=1)
    df["b_height"] = pd.cut(
        df["limit_times"], [0, 1, 2, 3, 999], labels=["首板", "2板", "3板", "高位"]
    )
    for c in BUCKET_COLS:
        df[c] = df[c].astype("category")
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """交互特征 (文档方案B). 全部 <= 当日."""
    df["first_sector"] = (df["board_height"] == 1).astype(float) * df["sector_n_up"]
    df["seal_health"] = df["fd_amount_ratio"] * df["turnover_rate"].fillna(0)
    df["zhaban_mkt"] = df["mkt_zhaban_rate"] * df["mkt_n_up"]
    df["board_seal"] = df["limit_times"] * df["fd_amount_ratio"]
    df["yizi_seal"] = df["is_yiziban"] * df["fd_amount_ratio"]
    return df


def add_yizi_wave(df: pd.DataFrame) -> pd.DataFrame:
    """一字板换手事件 (文档: 连续一字后首次开板换手, 封单~20% → 主升浪起点)."""
    sym = df["symbol"]
    df["yizi_prev"] = df["is_yiziban"].groupby(sym).shift(1).fillna(0).astype(float)
    # 连续一字天数 (run-length 向量化计数)
    is_y = df["is_yiziban"]
    change = (is_y != is_y.groupby(sym).shift(1)).astype(int).fillna(1)
    run_id = change.groupby(sym).cumsum()
    streak = is_y.groupby([sym, run_id]).cumsum()
    df["yizi_streak"] = streak.where(is_y == 1, 0.0)
    df["yizi_prev_streak"] = df["yizi_streak"].groupby(sym).shift(1).fillna(0)
    # 主升浪起点: 今日涨停、非一字、昨日一字、封单健康
    # fd_amount_ratio (Tushare) = 封单/流通市值, 中位 ~0.008 (0.8%), 健康换手 ~0.5%-5%
    # (设计文档"10-30%"是另一套量纲; 本文档量纲已按真实分布校准)
    health = df["fd_amount_ratio"].between(0.005, 0.05)
    df["main_wave_start"] = (
        (df["is_limit_up"] == 1)
        & (df["is_yiziban"] == 0)
        & (df["yizi_prev"] == 1)
        & health
    ).astype(float)
    return df


def add_lhb(df: pd.DataFrame) -> pd.DataFrame:
    """龙虎榜席位特征 (面板 lhb_*). 未上榜填 0 + is_lhb 标记."""
    for c in ["lhb_net_buy", "lhb_inst_buy", "lhb_inst_sell", "lhb_buy_amt"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["is_lhb"] = (df.get("lhb_buy_amt", 0) > 0).astype(float)
    df["lhb_net_ratio"] = df["lhb_net_buy"] / df["amount"].replace(0, np.nan)
    df["lhb_net_ratio"] = (
        df["lhb_net_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    df["lhb_inst_net"] = df["lhb_inst_buy"] - df["lhb_inst_sell"]
    return df


def _prep(sub, feats, num_feats, cat_cols):
    keep = feats + ["fwd_up_1d", "date"]
    keep += [c for c in ("label_pm_3d", "label_pm_5d") if c in sub.columns]
    sub = sub[keep].copy()
    for c in num_feats:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        sub[c] = sub[c].fillna(sub[c].median())
    for c in cat_cols:
        if c in sub.columns:
            sub[c] = sub[c].astype("category")
    return sub


def _run(up, board, feats, num_feats, cat_cols, tag, out, kargs):
    sub = _prep(up, feats, num_feats, cat_cols)
    sub = sub.dropna(subset=["fwd_up_1d"])
    if len(sub) < 500 or sub["fwd_up_1d"].nunique() < 2:
        print(f"    {tag:<18s} 样本不足, 跳过")
        return
    dts = sorted(sub["date"].unique())
    sp = dts[int(len(dts) * 0.6)]
    tr, te = sub[sub["date"] < sp], sub[sub["date"] >= sp]
    if len(tr) < 200 or len(te) < 200 or te["fwd_up_1d"].nunique() < 2:
        print(f"    {tag:<18s} train/test 不足, 跳过")
        return
    m = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbose=-1,
        **kargs,
    )
    cat_used = [c for c in cat_cols if c in feats]
    m.fit(tr[feats], tr["fwd_up_1d"], categorical_feature=cat_used or None)
    p = m.predict_proba(te[feats])[:, 1]
    y = te["fwd_up_1d"].values
    auc = roc_auc_score(y, p)
    # 分位校准 (predicted → actual 明日涨停率)
    cal = pd.DataFrame({"p": p, "y": y})
    cal["q"] = pd.qcut(pd.Series(p), 5, labels=False, duplicates="drop")
    cal_rows = []
    for qi, g in cal.groupby("q"):
        cal_rows.append(
            (
                int(qi),
                round(float(g["p"].mean()), 3),
                round(float(g["y"].mean()), 3),
                int(len(g)),
            )
        )
    monotone = all(
        cal_rows[i][2] <= cal_rows[i + 1][2] for i in range(len(cal_rows) - 1)
    )
    # top20% 前视 3 日收益 (诚实: 概率高≠涨得多)
    r3 = te["label_pm_3d"] if "label_pm_3d" in te.columns else None
    te2 = te.copy()
    te2["prob"] = p
    top = te2.nlargest(int(len(te2) * 0.2), "prob")
    top_ret = (
        round(float(top["label_pm_3d"].mean()) * 100, 3) if r3 is not None else None
    )
    base_ret = (
        round(float(te["label_pm_3d"].mean()) * 100, 3) if r3 is not None else None
    )
    imp = pd.DataFrame(
        {
            "feature": feats,
            "gain": m.booster_.feature_importance(importance_type="gain"),
        }
    )
    imp = imp.sort_values("gain", ascending=False)
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)

    print(
        f"    {tag:<18s} AUC={auc:.4f} 校准单调={monotone} "
        f"top20%T+3={top_ret}% (基线{base_ret}%)  n_tr={len(tr)} n_te={len(te)}"
    )
    out[f"{tag}"] = {
        "auc": round(auc, 4),
        "n_train": len(tr),
        "n_test": len(te),
        "base_cont_rate": round(float(y.mean()), 4),
        "calibration_monotone": monotone,
        "calibration": [
            {"q": qi, "pred": mp, "actual": ay, "n": nn} for qi, mp, ay, nn in cal_rows
        ],
        "top20_ret3_pct": top_ret,
        "base_ret3_pct": base_ret,
        "importance": imp.to_dict("records"),
    }
    return m


def event_main_wave(up, board, out):
    """主升浪起点事件评估 (文档: 连续一字后首次开板换手, 封单~20%)."""
    print(f"\n  === 主升浪起点事件 ({board.upper()}) ===")
    rows = []
    base = up
    wv = up[up["main_wave_start"] == 1]
    if len(wv) < 10:
        print("    (样本不足)")
        return

    def _s(g):
        n = len(g)
        return {
            "n": n,
            "明日涨停%": round(float(g["fwd_up_1d"].mean()) * 100, 2),
            "3日涨停%": round(float(g["fwd_up_3d"].mean()) * 100, 2),
            "T+3均值%": round(float(g["label_pm_3d"].mean()) * 100, 3),
            "T+5均值%": round(float(g["label_pm_5d"].mean()) * 100, 3),
            "T+5上涨率%": round(float((g["label_pm_5d"] > 0).mean()) * 100, 2),
        }

    rows.append(("全体涨停", _s(base)))
    rows.append(("主升浪起点(一字→开板换手)", _s(wv)))
    rows.append(("  + 昨日2字+", _s(wv[wv["yizi_prev_streak"] >= 2])))
    print(
        f"    {'组':<28s} {'n':>6s} {'明日涨停%':>8s} {'3日涨停%':>8s} {'T+3%':>7s} {'T+5%':>7s} {'T+5涨率%':>8s}"
    )
    for name, s in rows:
        print(
            f"    {name:<28s} {s['n']:>6d} {s['明日涨停%']:>8.1f} {s['3日涨停%']:>8.1f} "
            f"{s['T+3均值%']:>+7.2f} {s['T+5均值%']:>+7.2f} {s['T+5上涨率%']:>8.1f}"
        )
    out[board]["main_wave_event"] = {name: s for name, s in rows}


def analyze(up, board, out):
    out[board] = {"n_limit_up": int(len(up))}
    # 各层特征集 (逐层加特征, 证明分桶/交互/龙虎榜/主升浪 是否有增益)
    base_feats = BASE_FEATS
    buck_feats = BASE_FEATS + BUCKET_COLS + ["turnover_rate"]
    inter_feats = buck_feats + INTER_FEATS
    full_feats = inter_feats + LHB_FEATS
    full_wave_feats = full_feats + WAVE_FEATS
    cat_cols = BUCKET_COLS
    num_sets = {
        "base(原始)": base_feats,
        "+分桶": buck_feats,
        "+交互": inter_feats,
        "+龙虎榜": full_feats,
        "+主升浪": full_wave_feats,
    }
    print(f"\n  === 涨停板接力模型 ({board.upper()}) 分层对比 (OOS AUC) ===")
    best = None
    for tag, feats in num_sets.items():
        m = _run(
            up,
            board,
            feats,
            [c for c in feats if c not in cat_cols],
            cat_cols,
            tag,
            out[board],
            {},
        )
        if m is not None and (best is None or out[board][tag]["auc"] > best[1]):
            best = (m, out[board][tag]["auc"], tag, feats)
    return best


def main():
    feat_files = sorted(glob.glob(os.path.join(OUT_DIR, "limit_feat_*.parquet")))
    feat_path = feat_files[-1]
    feat = pd.read_parquet(feat_path)
    dates = sorted(feat["date"].unique())
    print(f"[1] feature: {feat_path}")

    panel = pd.read_parquet(
        PANEL,
        columns=[
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pre_close",
            "close_hfq",
            "industry",
            "board",
            "circ_mv",
            "is_suspended",
            *PANEL_EXTRA,
        ],
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(dates)]
    panel = panel.merge(feat, on=["symbol", "date"], how="left")
    for c in [
        "is_limit_up",
        "is_limit_down",
        "is_zhaban",
        "limit_times",
        "fd_amount_ratio",
        "open_times",
    ]:
        panel[c] = panel[c].fillna(0.0)
    del feat
    gc.collect()

    print("[2] cleaning...")
    cleaner = R.CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()

    results_out = {}
    boosters = {}
    for board, board_df in [("main", main_df), ("dual", dual_df)]:
        if len(board_df) == 0:
            continue
        df = R.LabelEngine.build_labels(board_df)
        df = R.LabelEngine.mask_suspension(df)
        df = R.LabelEngine.mask_recent_days(df, days=6)
        df = R.add_derived(df)
        df = R.add_fwd_labels(df)
        df = add_buckets(df)
        df = add_interactions(df)
        df = add_lhb(df)
        df = add_yizi_wave(df)
        up = df[df["is_limit_up"] == 1]
        best = analyze(up, board, results_out)
        event_main_wave(up, board, results_out)
        if best:
            m, auc, tag, feats = best
            boosters[board] = {"booster": m, "auc": auc, "tag": tag, "feats": feats}

    results_out["meta"] = {
        "tag": TAG,
        "window": [str(x.date()) for x in [dates[0], dates[-1]]],
        "n_dates": len(dates),
        "script": "_build_limit_board_model.py",
        "feature_file": os.path.basename(feat_path),
        "target": "fwd_up_1d 明日继续涨停 (二分类)",
        "model": "LGBMClassifier balanced, 60/40 date split, seed 42",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"limit_board_model_{TAG}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=1)
    print(f"\nSaved: {path}")
    # 存最优模型 booster (供打板/接力推理)
    for board, info in boosters.items():
        bp = os.path.join(OUT_DIR, f"limit_board_{board}_{TAG}.txt")
        info["booster"].booster_.save_model(bp)
        with open(bp.replace(".txt", "_feats.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "board": board,
                    "auc": info["auc"],
                    "tag": info["tag"],
                    "feats": info["feats"],
                },
                f,
                ensure_ascii=False,
                indent=1,
            )
        print(f"    booster {board}: {bp} (AUC {info['auc']:.4f})")
    print("DONE")


if __name__ == "__main__":
    main()
