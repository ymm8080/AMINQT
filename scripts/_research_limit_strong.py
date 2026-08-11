"""强势股 / 连板延续研究 — 如何消费 limit_list_d 数据抓强势股.

数据:
  - data/factor_registry/limit_feat_<end>.parquet  (_build_limit_features.py 生成)
  - 全量面板 (OHLC / industry / circ_mv / close_hfq / board)

研究内容 (分 main/dual 板):
  H1 回调后再板   涨停→回调→再涨停 (二波启动) 是否更易延续 / 更强势
  H2 二板动量     近 5 日第 2/3+ 次涨停 vs 首板 (近期上涨惯性)
  H3 连板特性     连板高度递增的延续率 + 封单/封时/开板/情绪/板块 区分度
  M  连板延续模型 P(明日继续涨停 | 今日涨停), OOS AUC + 分位校准 + 特征重要度
  S  强势股判别   模型概率能否排名未来收益 (label_pm_3d) + top 分位相对基线

验收口径: label_pm_* (LabelEngine PM 执行口径), fwd_up_* = 未来是否继续涨停.
结果落盘 JSON (WORM).

Usage: python scripts/_research_limit_strong.py
"""

import gc
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.label_engine import LabelEngine
from config import settings

PANEL = settings.PANEL_V3_PATH
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = "data/factor_registry"

MODEL_FEATS = [
    "limit_times",
    "fd_amount_ratio",
    "open_times",
    "seal_mins",
    "is_yiziban",
    "n_up_5d",
    "n_up_10d",
    "days_since_up",
    "ret_since_prev_up",
    "mkt_zhaban_rate",
    "mkt_n_up",
    "sector_n_up",
    "ln_circ_mv",
]
PCT = {"label_pm_2d": "2d", "label_pm_3d": "3d", "label_pm_5d": "5d"}

# 回调续涨模型特征 (非涨停日也能算, 全部 <= 当日)
PULL_FEATS = [
    "days_since_up",
    "ret_since_prev_up",
    "last_height",
    "ret_1d",
    "ret_5d",
    "mkt_zhaban_rate",
    "mkt_n_up",
    "sector_n_up",
    "ln_circ_mv",
]


# ────────────────────────── 特征构建 (向量化) ──────────────────────────
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = df["symbol"]
    up = df["is_limit_up"]

    # 连板高度 (自算连续涨停计数; 与 Tushare limit_times 交叉验证)
    prev_up = up.groupby(sym).shift(1, fill_value=0)
    run_id = ((up == 1) & (prev_up != 1)).cumsum()
    df["board_height"] = up.groupby(run_id).cumsum().where(up == 1, 0.0)

    # 近 5/10/20 日涨停次数 (滚动计数, 用 cumsum 差分避免 rolling 慢)
    cs = up.groupby(sym).cumsum()
    df["n_up_5d"] = cs - cs.groupby(sym).shift(5).fillna(0)
    df["n_up_10d"] = cs - cs.groupby(sym).shift(10).fillna(0)
    df["n_up_20d"] = cs - cs.groupby(sym).shift(20).fillna(0)

    # 距上次涨停交易日数 + 上次涨停收盘 → 当前收盘涨幅
    pos = df.groupby(sym).cumcount()
    last_up_pos = pos.where(up == 1).groupby(sym).ffill()
    df["days_since_up"] = (pos - last_up_pos).fillna(999)
    prev_up_close = (
        df["close"].where(up == 1).groupby(sym).ffill().groupby(sym).shift(1)
    )
    df["ret_since_prev_up"] = df["close"] / prev_up_close - 1

    # 上次涨停的连板高度 (非涨停日 carry-forward) + 短中期动量
    df["last_height"] = df["board_height"].where(up == 1).groupby(sym).ffill().fillna(0)
    df["ret_1d"] = df["close"] / df["pre_close"] - 1
    df["ret_5d"] = df["close"] / df.groupby(sym)["close"].shift(5) - 1

    # 一字板 (OHLC 全等)
    ohlc_eq = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )
    df["is_yiziban"] = (ohlc_eq & (up == 1)).astype(float)

    # 市场情绪 (日度) / 板块效应 (date×industry)
    m = df.groupby("date")
    df["mkt_n_up"] = m["is_limit_up"].transform("sum")
    df["mkt_n_zhaban"] = m["is_zhaban"].transform("sum")
    df["mkt_zhaban_rate"] = (
        df["mkt_n_zhaban"] / (df["mkt_n_up"] + df["mkt_n_zhaban"]).replace(0, np.nan)
    ).fillna(0)
    df["sector_n_up"] = df.groupby(["date", "industry"])["is_limit_up"].transform("sum")
    df["ln_circ_mv"] = np.log(df["circ_mv"].clip(lower=1e6))
    return df


def add_fwd_labels(df: pd.DataFrame) -> pd.DataFrame:
    sym = df["symbol"]
    up = df["is_limit_up"]
    df["fwd_up_1d"] = up.groupby(sym).shift(-1).fillna(0).astype(float)
    f1 = up.groupby(sym).shift(-1).fillna(0)
    f2 = up.groupby(sym).shift(-2).fillna(0)
    f3 = up.groupby(sym).shift(-3).fillna(0)
    df["fwd_up_3d"] = (
        ((f1 > 0) | (f2 > 0) | (f3 > 0)) & (df["is_limit_up"] == 1)
    ).astype(float)
    return df


# ────────────────────────── 统计工具 ──────────────────────────
def _stats(sub, col):
    v = sub[col].dropna()
    if len(v) < 10:
        return None
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 5),
        "hit": round(float((v > 0).mean()), 4),
    }


def _row(sub, ret_col, up_col="fwd_up_1d"):
    """一行: 样本量 + 未来收益均值/上涨率 + 未来涨停延续率."""
    if len(sub) < 10:
        return None
    r = _stats(sub, ret_col)
    return {
        "n": int(len(sub)),
        "mean_pct": round(float(sub[ret_col].mean()) * 100, 3) if r else np.nan,
        "hit_pct": round(float((sub[ret_col] > 0).mean()) * 100, 2) if r else np.nan,
        "fwd_up_1d_pct": round(float(sub[up_col].mean()) * 100, 2),
        "fwd_up_3d_pct": round(float(sub["fwd_up_3d"].mean()) * 100, 2),
    }


def _print_rows(rows, ret_col):
    hdr = f"    {'组':<22s} {'n':>6s} {'均值%':>7s} {'上涨率%':>7s} {'明日涨停%':>8s} {'3日涨停%':>8s}"
    print(hdr)
    print("    " + "-" * 62)
    for name, s in rows:
        if s is None:
            print(f"    {name:<22s} (样本不足)")
            continue
        print(
            f"    {name:<22s} {s['n']:>6d} {s['mean_pct']:>+7.2f} {s['hit_pct']:>6.1f} "
            f"{s['fwd_up_1d_pct']:>7.1f} {s['fwd_up_3d_pct']:>8.1f}"
        )


def _split_half(df):
    dts = sorted(df["date"].unique())
    mid = dts[len(dts) // 2]
    return df[df["date"] < mid], df[df["date"] >= mid]


# ────────────────────────── 假设分析 ──────────────────────────
def hypothesis(df, board, out):
    up = df[df["is_limit_up"] == 1]
    ret_col = "label_pm_3d"
    print(
        f"\n[+] {board.upper()} 涨停事件样本: {len(up)}  (日期 {up['date'].min().date()} .. {up['date'].max().date()})"
    )

    # H1 回调后再板
    print("\n  === H1 回调后再板 (二波启动) ===")
    cond = (
        (up["n_up_20d"] >= 2)
        & (up["ret_since_prev_up"] <= -0.03)
        & (up["board_height"] == 1)
    )
    h1 = [
        ("全部涨停", _row(up, ret_col)),
        ("首板(近20日无涨停)", _row(up[up["n_up_20d"] <= 1], ret_col)),
        ("回调后再板(近20日有+回调>=3%)", _row(up[cond], ret_col)),
        ("连续板(board_height>=2)", _row(up[up["board_height"] >= 2], ret_col)),
    ]
    _print_rows(h1, ret_col)
    out[board]["h1"] = {k: v for k, v in h1 if v}

    # H2 二板动量
    print("\n  === H2 近5日涨停次数 (动量) ===")
    h2 = [
        ("首板 (5日内第1次)", _row(up[up["n_up_5d"] == 1], ret_col)),
        ("二板 (5日内第2次)", _row(up[up["n_up_5d"] == 2], ret_col)),
        ("三板 (5日内第3次)", _row(up[up["n_up_5d"] == 3], ret_col)),
        ("4次+ (5日内)", _row(up[up["n_up_5d"] >= 4], ret_col)),
    ]
    _print_rows(h2, ret_col)
    out[board]["h2"] = {k: v for k, v in h2 if v}

    # H3a 连板高度递增
    print("\n  === H3a 连板高度 → 延续率 ===")
    h3a = [
        (f"{h}连板", _row(up[up["board_height"] == h], ret_col))
        for h in [1, 2, 3, 4, 5]
    ]
    h3a.append(("6连板+", _row(up[up["board_height"] >= 6], ret_col)))
    _print_rows(h3a, ret_col)
    out[board]["h3a"] = {k: v for k, v in h3a if v}

    # H3b 板内特征区分度 (在 2-4 连板中是否继续)
    print("\n  === H3b 涨停组内特征 → 明日延续率 (全体涨停) ===")
    if len(up) > 40:
        rows = []
        up["fwd_up_1d"].mean()
        rows.append(("基线 全体涨停", _row(up, ret_col)))
        if up["seal_mins"].notna().sum() > 20:
            med = up["seal_mins"].median()
            rows.append(("早封板(<=中位)", _row(up[up["seal_mins"] <= med], ret_col)))
            rows.append(("晚封板(>中位)", _row(up[up["seal_mins"] > med], ret_col)))
        if up["fd_amount_ratio"].notna().sum() > 20:
            med = up["fd_amount_ratio"].median()
            rows.append(
                ("强封单(>=中位)", _row(up[up["fd_amount_ratio"] >= med], ret_col))
            )
            rows.append(
                ("弱封单(<中位)", _row(up[up["fd_amount_ratio"] < med], ret_col))
            )
        if (up["is_yiziban"] == 1).sum() > 20:
            rows.append(("一字板", _row(up[up["is_yiziban"] == 1], ret_col)))
            rows.append(("非一字板", _row(up[up["is_yiziban"] == 0], ret_col)))
        if (up["open_times"] >= 2).sum() > 20:
            rows.append(("开板>=2次", _row(up[up["open_times"] >= 2], ret_col)))
            rows.append(("未开板", _row(up[up["open_times"] < 2], ret_col)))
        med_zr = up["mkt_zhaban_rate"].median()
        rows.append(
            ("市场炸板率高(>=中位)", _row(up[up["mkt_zhaban_rate"] >= med_zr], ret_col))
        )
        rows.append(
            ("市场炸板率低(<中位)", _row(up[up["mkt_zhaban_rate"] < med_zr], ret_col))
        )
        rows.append(("板块内>=2只涨停", _row(up[up["sector_n_up"] >= 2], ret_col)))
        rows.append(("板块内仅1只涨停", _row(up[up["sector_n_up"] < 2], ret_col)))
        _print_rows(rows, ret_col)
        out[board]["h3b"] = {k: v for k, v in rows if v}

    # H4 涨停后回调(半月~一月) → 继续上涨 (用户观察: 上板后回调一段时间可能继续涨)
    print("\n  === H4 涨停后回调(半月~一月) → 继续上涨 ===")
    up24 = (df["days_since_up"] >= 8) & (df["days_since_up"] <= 28)  # ~半月到一月
    pb = up24 & (df["ret_since_prev_up"] <= -0.02)
    h4_groups = [
        ("全池基线", df),
        ("涨停后未回调", df[up24 & (df["ret_since_prev_up"] > -0.02)]),
        ("涨停后回调(8~28日,<-2%)", df[pb]),
        ("  回调 8~14日", df[pb & (df["days_since_up"] < 15)]),
        (
            "  回调 15~21日",
            df[pb & (df["days_since_up"] >= 15) & (df["days_since_up"] < 22)],
        ),
        ("  回调 22~28日", df[pb & (df["days_since_up"] >= 22)]),
        ("  回调 2~5%", df[pb & (df["ret_since_prev_up"] > -0.05)]),
        (
            "  回调 5~10%",
            df[
                pb
                & (df["ret_since_prev_up"] <= -0.05)
                & (df["ret_since_prev_up"] > -0.10)
            ],
        ),
        ("  深回调 >10%", df[pb & (df["ret_since_prev_up"] <= -0.10)]),
        ("  前次3连板+", df[pb & (df["last_height"] >= 3)]),
    ]
    for horizon in ["label_pm_3d", "label_pm_5d"]:
        sfx = horizon.replace("label_pm_", "T+")
        print(f"  --- 前视 {sfx} (PM 执行口径) ---")
        _print_rows([(n, _row(g, horizon)) for n, g in h4_groups], horizon)
    out[board]["h4"] = {
        horizon: {n: _row(g, horizon) for n, g in h4_groups if len(g) >= 10}
        for horizon in ["label_pm_3d", "label_pm_5d"]
    }
    return up


# ────────────────────────── 连板延续模型 + 强势股判别 ──────────────────────────
def model_eval(up, board, out):
    ret_col = "label_pm_3d"
    sub = up[MODEL_FEATS + ["fwd_up_1d", ret_col, "date"]].copy()
    for c in MODEL_FEATS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        sub[c] = sub[c].fillna(sub[c].median())
    sub = sub.dropna(subset=["fwd_up_1d", ret_col])
    if len(sub) < 500 or sub["fwd_up_1d"].nunique() < 2:
        print(f"  {board.upper()} 样本不足, 跳过模型")
        return
    dts = sorted(sub["date"].unique())
    sp = dts[int(len(dts) * 0.6)]
    tr = sub[sub["date"] < sp].copy()
    te = sub[sub["date"] >= sp].copy()
    if len(tr) < 200 or len(te) < 200 or te["fwd_up_1d"].nunique() < 2:
        print(f"  {board.upper()} train/test 样本不足, 跳过模型")
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
    )
    m.fit(tr[MODEL_FEATS], tr["fwd_up_1d"])
    p = m.predict_proba(te[MODEL_FEATS])[:, 1]
    y = te["fwd_up_1d"].values
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y, p)
    # 分位校准
    q = pd.qcut(pd.Series(p), 5, labels=False, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": y, "q": q})
    cal_rows = []
    for qi, g in cal.groupby("q"):
        cal_rows.append(
            (
                int(qi),
                round(float(g["p"].mean()), 4),
                round(float(g["y"].mean()), 4),
                int(len(g)),
            )
        )
    # 强势股判别: 模型概率排名未来收益 (test 涨停子集)
    r_ic, _ = spearmanr(p, te[ret_col].values)
    te2 = te.copy()
    te2["prob"] = p
    top = te2.nlargest(int(len(te2) * 0.2), "prob")
    base_mean = float(te[ret_col].mean())
    base_hit = float((te[ret_col] > 0).mean())
    top_mean = float(top[ret_col].mean())
    top_hit = float((top[ret_col] > 0).mean())
    imp = pd.DataFrame(
        {
            "feature": MODEL_FEATS,
            "gain": m.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)

    print(f"\n  === M/S 连板延续模型 ({board.upper()}) ===")
    print(
        f"    train {len(tr)} (fwd_up率 {tr['fwd_up_1d'].mean() * 100:.1f}%) | "
        f"test {len(te)} ({te['fwd_up_1d'].mean() * 100:.1f}%) | 基率 {te['fwd_up_1d'].mean() * 100:.1f}%"
    )
    print(f"    OOS AUC = {auc:.4f}  |  概率 RankIC vs label_pm_3d = {r_ic:+.4f}")
    print(
        f"    top20% 概率组: 均值 {top_mean * 100:+.2f}% (基线 {base_mean * 100:+.2f}%) | "
        f"上涨率 {top_hit * 100:.1f}% (基线 {base_hit * 100:.1f}%)"
    )
    print("    --- 分位校准 (predicted → actual 明日涨停率) ---")
    for qi, mp, ay, nn in cal_rows:
        print(f"      Q{qi + 1}: pred={mp:.3f} actual={ay:.3f} n={nn}")
    print("    --- 特征重要度 (gain%) ---")
    for _, r in imp.iterrows():
        print(f"      {r['feature']:<16s} {r['gain_pct']:.1f}%")

    out[board]["model"] = {
        "auc": round(auc, 4),
        "train_n": len(tr),
        "test_n": len(te),
        "base_cont_rate": round(float(te["fwd_up_1d"].mean()), 4),
        "prob_rankic_vs_pm3d": round(float(r_ic), 4),
        "top20_mean_pct": round(top_mean * 100, 3),
        "base_mean_pct": round(base_mean * 100, 3),
        "top20_hit_pct": round(top_hit * 100, 2),
        "base_hit_pct": round(base_hit * 100, 2),
        "calibration": [
            {"q": qi, "pred": mp, "actual": ay, "n": nn} for qi, mp, ay, nn in cal_rows
        ],
        "importance": imp.to_dict("records"),
    }


def model_pullback(df, board, out):
    """P 回调续涨模型 — 在涨停后回调(半月~一月)的股票里, 排名未来 5 日收益.

    检验用户观察 "涨停回调半月~一月 → 可能继续涨" 是否能被特征排序利用:
    在回调窗口样本上回归 label_pm_5d, 看 test 集 RankIC + top20% 相对基线.
    """
    ret_col = "label_pm_5d"
    sub = df[
        (df["days_since_up"] >= 8)
        & (df["days_since_up"] <= 28)
        & (df["ret_since_prev_up"] <= -0.02)
    ]
    sub = sub[PULL_FEATS + [ret_col, "date"]].copy()
    for c in PULL_FEATS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        sub[c] = sub[c].fillna(sub[c].median())
    sub = sub.dropna(subset=[ret_col])
    if len(sub) < 800:
        print(f"  {board.upper()} 回调样本不足, 跳过回调续涨模型")
        return
    dts = sorted(sub["date"].unique())
    sp = dts[int(len(dts) * 0.6)]
    tr = sub[sub["date"] < sp].copy()
    te = sub[sub["date"] >= sp].copy()
    if len(tr) < 400 or len(te) < 400:
        print(f"  {board.upper()} 回调 train/test 样本不足, 跳过回调续涨模型")
        return

    m = lgb.LGBMRegressor(
        n_estimators=400,
        max_depth=5,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(tr[PULL_FEATS], tr[ret_col])
    p = m.predict(te[PULL_FEATS])
    y = te[ret_col].values
    r_ic, _ = spearmanr(p, y)
    te2 = te.copy()
    te2["pred"] = p
    top = te2.nlargest(int(len(te2) * 0.2), "pred")
    base_mean = float(y.mean())
    base_hit = float((y > 0).mean())
    top_mean = float(top[ret_col].mean())
    top_hit = float((top[ret_col] > 0).mean())
    imp = pd.DataFrame(
        {
            "feature": PULL_FEATS,
            "gain": m.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)

    print(f"\n  === P 回调续涨模型 ({board.upper()}) 样本={len(sub)} ===")
    print(
        f"    train {len(tr)} / test {len(te)} (回调窗口 8~28日, ret_since_prev_up<=-2%)"
    )
    print(f"    预测 label_pm_5d RankIC = {r_ic:+.4f}")
    print(
        f"    top20% 实际均值 {top_mean * 100:+.2f}% (test 基线 {base_mean * 100:+.2f}%) | "
        f"上涨率 {top_hit * 100:.1f}% (基线 {base_hit * 100:.1f}%)"
    )
    print("    --- 特征重要度 (gain%) ---")
    for _, r in imp.iterrows():
        print(f"      {r['feature']:<16s} {r['gain_pct']:.1f}%")

    out[board]["pullback_model"] = {
        "n": len(sub),
        "train_n": len(tr),
        "test_n": len(te),
        "rankic_vs_pm5d": round(float(r_ic), 4),
        "top20_mean_pct": round(top_mean * 100, 3),
        "base_mean_pct": round(base_mean * 100, 3),
        "top20_hit_pct": round(top_hit * 100, 2),
        "base_hit_pct": round(base_hit * 100, 2),
        "importance": imp.to_dict("records"),
    }


def analyze(board_df, board, out):
    df = LabelEngine.build_labels(board_df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)
    df = add_derived(df)
    df = add_fwd_labels(df)
    n_all = int(len(df))
    out[board] = {"n_rows": n_all}
    print(f"\n{'=' * 74}\n[{board.upper()}] rows={n_all}\n{'=' * 74}")
    up = hypothesis(df, board, out)
    # 半窗稳定性: 连板高度延续率 (board_height 1 vs 2+) 两半窗
    dts = sorted(df["date"].unique())
    if len(dts) >= 20:
        half1, half2 = _split_half(df)
        stab = {}
        for hn, hd in [("first", half1), ("second", half2)]:
            hu = hd[hd["is_limit_up"] == 1]
            for hh in [1, 2, 3]:
                s = hu[hu["board_height"] == hh]
                if len(s) >= 10:
                    stab.setdefault(hn, {})[hh] = round(float(s["fwd_up_1d"].mean()), 4)
        if stab.get("first") and stab.get("second"):
            print("\n  === 半窗稳定性: X连板 → 明日延续率 ===")
            for hn in ["first", "second"]:
                parts = [
                    f"  {h}板={stab[hn].get(h, float('nan')):.3f}" for h in [1, 2, 3]
                ]
                print(f"    {hn:8s}" + "".join(parts))
            out[board]["cont_stability"] = stab
    model_eval(up, board, out)
    model_pullback(df, board, out)


def main():
    feat_files = sorted(glob.glob(os.path.join(OUT_DIR, "limit_feat_*.parquet")))
    if not feat_files:
        print("FATAL: no limit_feat_*.parquet found — 先跑 _build_limit_features.py")
        sys.exit(1)
    feat_path = feat_files[-1]
    print(f"[1] feature file: {feat_path}")
    feat = pd.read_parquet(feat_path)
    print(
        f"    feat rows: {len(feat)}, dates: {feat['date'].nunique()}, "
        f"up: {(feat['is_limit_up'] == 1).sum()}, zhaban: {(feat['is_zhaban'] == 1).sum()}"
    )
    dates = sorted(feat["date"].unique())

    print("[2] loading panel window + merging...")
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
            "turnover_rate",
            "is_suspended",
        ],
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(dates)]
    print(f"    panel window: {len(panel)} rows, {panel['date'].nunique()} dates")
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

    print("[3] cleaning...")
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()

    results_out = {}
    for board, board_df in [("main", main_df), ("dual", dual_df)]:
        if len(board_df) == 0:
            continue
        analyze(board_df, board, results_out)

    results_out["meta"] = {
        "tag": TAG,
        "window": [str(x.date()) for x in [dates[0], dates[-1]]],
        "n_dates": len(dates),
        "script": "_research_limit_strong.py",
        "feature_file": os.path.basename(feat_path),
        "source": "Tushare limit_list_d + 面板",
        "labels": "label_pm_* PM 执行口径; fwd_up_* = 未来继续涨停",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"limit_strong_research_{TAG}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=1)
    print(f"\nSaved: {path}")
    print("DONE")


if __name__ == "__main__":
    main()
