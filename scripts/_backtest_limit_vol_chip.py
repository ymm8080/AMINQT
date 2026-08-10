"""回测: 连板 × 量能 × 筹码 × 价格 → 信号密度 + 带成本事件回测.

用户三连问:
  1. 直觉 "涨停回调半月~一月可能继续涨" 有没有数据支持 (严格分桶验证)
  2. 连板信息 联合 量能变化 + 筹码变化 (+价格) 是否更有信号密度
  3. 回测确认 (带交易成本, 铁律)

设计:
  A. 分桶检验 — 主板涨停后回调 8~40 日 × T+3/T+5 均值/上涨率, 找支持窗口
  B. 条件化信号密度 — 在回调满月窗口上叠加 缩量/放量 · 筹码集中 · 获利盘 · 价格分档,
     量能/筹码/价格 阈值全部 无前视 (缩量=固定 ma_vol_ratio_5_20<1.0;
     筹码/获利盘=当日横截面中位; 价格=固定分档)
  C. 模型密度提升 — 连板延续模型 + 回调续涨模型 各加 vol/chip/price 特征, 对比
     base AUC / RankIC 是否提升
  D. 带成本事件回测 — 每周期首次满足条件日买入, 等权, 持有5日,
     净收益 = label_pm_5d - roundtrip_cost; 按日聚合时间序列 mean/std/Sharpe

成本 (settings): 佣金万2.5×2 + 印花0.05%卖 + 滑点0.10%×2 ≈ 单边往返 0.30%.
结果落盘 JSON (WORM).

Usage: python scripts/_backtest_limit_vol_chip.py
"""
import gc
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

import _research_limit_strong as R  # 复用 add_derived / add_fwd_labels / 清洗 / 标签
from app.core.config_loader import load_config

PANEL = R.PANEL
OUT_DIR = R.OUT_DIR
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

# 往返成本 (铁律: 回测必须含交易成本) — 来自 backtest_config.yaml, 不硬编码
_bt_cfg = load_config("backtest_config").get("backtest", {})
COST_RT = (
    2 * _bt_cfg.get("commission_rate", 0.00025)
    + _bt_cfg.get("stamp_tax_rate", 0.001)
    + (_bt_cfg.get("slippage_buy_bp", 10) + _bt_cfg.get("slippage_sell_moo_bp", 10)) / 10000
)

# 面板里现成的 量能/筹码/价格 列 (额外加载, 研究脚本没带)
VOL_CHIP_COLS = [
    "volume_ratio", "ma_vol_ratio_5_20", "winner_ratio",
    "chip_entropy", "chip_gini", "cost_50pct", "cost_95pct", "cost_bias",
    "pct_90_high", "pct_90_con",
]

# 模型特征集 (在 R.MODEL_FEATS / R.PULL_FEATS 基础上追加)
EXTRA_FEATS = ["ma_vol_ratio_5_20", "winner_ratio", "chip_gini", "cost_bias", "close"]
LIMIT_MODEL_FEATS = R.MODEL_FEATS + EXTRA_FEATS
PULL_MODEL_FEATS = R.PULL_FEATS + EXTRA_FEATS


def _clean_fill(sub, feats):
    for c in feats:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        sub[c] = sub[c].fillna(sub[c].median())
    return sub


def add_vol_chip_price(df: pd.DataFrame) -> pd.DataFrame:
    """追加 缩量回调 特征 (量能 vs 上次涨停日量) — 其余量能/筹码/价格直接用面板列."""
    sym = df["symbol"]
    up = df["is_limit_up"]
    last_up_vol = df["volume"].where(up == 1).groupby(sym).ffill()
    df["vol_vs_upday"] = df["volume"] / last_up_vol.replace(0, np.nan)
    df["vol_vs_upday"] = df["vol_vs_upday"].replace([np.inf, -np.inf], np.nan)
    return df


def _median_stat(ser):
    ser = ser.dropna()
    if len(ser) < 20:
        return None
    return {"n": len(ser), "mean_pct": round(float(ser.mean()) * 100, 3),
            "hit_pct": round(float((ser > 0).mean()) * 100, 2)}


# ── A. 分桶检验: 涨停后回调 8~40 日, 找支持窗口 ──
def check_h4(df, board, out):
    up = df["is_limit_up"]
    print(f"\n  === A. 涨停后回调 8~40 日 × T+3/T+5 (主板全部交易日) ===")
    rows = []
    for lo, hi in [(8, 14), (15, 21), (22, 28), (29, 35), (36, 40)]:
        m = (df["days_since_up"] >= lo) & (df["days_since_up"] <= hi) & (df["ret_since_prev_up"] <= -0.02)
        rows.append((f"回调{lo}~{hi}日", df[m]))
    for horizon in ["label_pm_3d", "label_pm_5d"]:
        print(f"  --- {horizon} (净额 = 均值 - 成本{COST_RT*100:.2f}%) ---")
        print(f"    {'组':<14s} {'n':>7s} {'均值%':>7s} {'上涨率%':>7s}")
        for name, g in rows:
            s = _median_stat(g[horizon])
            if s is None:
                print(f"    {name:<14s} (样本不足)")
                continue
            print(f"    {name:<14s} {s['n']:>7d} {s['mean_pct']:>+7.2f} {s['hit_pct']:>7.1f}")
    out[board]["h4_buckets"] = {h: {name: _median_stat(g[h]) for name, g in rows}
                                for h in ["label_pm_3d", "label_pm_5d"]}


# ── B. 条件化信号密度: 回调满月窗口 + 量/筹码/价格 ──
def cond_density(df, board, out):
    print(f"\n  === B. 回调满月(22~28日)窗口 条件化信号密度 (T+5) ===")
    base = (df["days_since_up"] >= 22) & (df["days_since_up"] <= 28) & (df["ret_since_prev_up"] <= -0.02)
    pb = df[base].copy()
    # 无前视条件: 缩量=固定阈值; 筹码/获利盘=当日截面中位; 价格=固定分档
    pb["d_shrink"] = pb["ma_vol_ratio_5_20"] < 1.0
    pb["d_gini_hi"] = pb["chip_gini"] >= pb.groupby("date")["chip_gini"].transform("median")
    pb["d_winner_hi"] = pb["winner_ratio"] >= pb.groupby("date")["winner_ratio"].transform("median")
    pb["d_price"] = pd.cut(pb["close"], [0, 10, 30, np.inf], labels=["低价<10", "中价10-30", "高价>30"])
    conds = [
        ("回调满月(基线)", pd.Series(True, index=pb.index)),
        ("  + 缩量(量能<1.0)", pb["d_shrink"]),
        ("  + 放量(>=1.0)", ~pb["d_shrink"]),
        ("  + 筹码集中(高gini)", pb["d_gini_hi"]),
        ("  + 筹码分散(低gini)", ~pb["d_gini_hi"]),
        ("  + 获利盘高", pb["d_winner_hi"]),
        ("  + 获利盘低", ~pb["d_winner_hi"]),
        ("  + 缩量&筹码集中", pb["d_shrink"] & pb["d_gini_hi"]),
    ]
    rows = []
    print(f"    {'条件':<22s} {'n':>7s} {'均值%':>7s} {'上涨率%':>7s} {'净额%':>7s}")
    for name, mask in conds:
        s = _median_stat(pb[mask]["label_pm_5d"])
        if s is None:
            print(f"    {name:<22s} (样本不足)")
            continue
        net = s["mean_pct"] - COST_RT * 100
        rows.append((name, s, net))
        print(f"    {name:<22s} {s['n']:>7d} {s['mean_pct']:>+7.2f} {s['hit_pct']:>7.1f} {net:>+7.2f}")
    out[board]["cond_density"] = [{"cond": n, **s, "net_pct": round(net, 3)} for n, s, net in rows]

    # 价格分档 (在回调满月内)
    print(f"    --- 价格分档 ---")
    price_rows = []
    for pl in ["低价<10", "中价10-30", "高价>30"]:
        s = _median_stat(pb[pb["d_price"] == pl]["label_pm_5d"])
        if s is None:
            continue
        net = s["mean_pct"] - COST_RT * 100
        price_rows.append({"price": pl, **s, "net_pct": round(net, 3)})
        print(f"    {pl:<14s} {s['n']:>7d} {s['mean_pct']:>+7.2f} {s['hit_pct']:>7.1f} {net:>+7.2f}")
    out[board]["price_bucket"] = price_rows
    return pb


# ── C. 模型密度提升 (base vs +量/筹码/价格) ──
def model_density(df, board, out):
    up = df[df["is_limit_up"] == 1]
    print(f"\n  === C. 模型信号密度: 连板延续 + 量/筹码/价格 ===")
    # C1 连板延续 (预测明日继续涨停)
    for tag, feats in [("base", R.MODEL_FEATS), ("+vol_chip_price", LIMIT_MODEL_FEATS)]:
        sub = _clean_fill(up[feats + ["fwd_up_1d", "date"]].copy(), feats)
        sub = sub.dropna(subset=["fwd_up_1d"])
        if len(sub) < 500 or sub["fwd_up_1d"].nunique() < 2:
            continue
        dts = sorted(sub["date"].unique())
        sp = dts[int(len(dts) * 0.6)]
        tr, te = sub[sub["date"] < sp], sub[sub["date"] >= sp]
        if len(te) < 200 or te["fwd_up_1d"].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                               class_weight="balanced", verbose=-1)
        m.fit(tr[feats], tr["fwd_up_1d"])
        p = m.predict_proba(te[feats])[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(te["fwd_up_1d"], p)
        print(f"    连板延续 AUC {tag:<16s} = {auc:.4f}")
        out[board].setdefault("model_density", {})[f"cont_{tag}"] = {"auc": round(auc, 4), "n_te": len(te)}

    # C2 回调续涨 (预测未来5日收益排名)
    pb_cond = (df["days_since_up"] >= 22) & (df["days_since_up"] <= 28) & (df["ret_since_prev_up"] <= -0.02)
    pb = df[pb_cond].copy()
    for tag, feats in [("base", R.PULL_FEATS), ("+vol_chip_price", PULL_MODEL_FEATS)]:
        sub = _clean_fill(pb[feats + ["label_pm_5d", "date"]].copy(), feats)
        sub = sub.dropna(subset=["label_pm_5d"])
        if len(sub) < 800:
            continue
        dts = sorted(sub["date"].unique())
        sp = dts[int(len(dts) * 0.6)]
        tr, te = sub[sub["date"] < sp], sub[sub["date"] >= sp]
        if len(te) < 400:
            continue
        m = lgb.LGBMRegressor(n_estimators=400, max_depth=5, num_leaves=31, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(tr[feats], tr["label_pm_5d"])
        p = m.predict(te[feats])
        r_ic, _ = spearmanr(p, te["label_pm_5d"])
        te2 = te.copy(); te2["pred"] = p
        top = te2.nlargest(int(len(te2) * 0.2), "pred")
        print(f"    回调续涨 RankIC {tag:<16s} = {r_ic:+.4f} | top20% {top['label_pm_5d'].mean()*100:+.2f}% (基线 {te['label_pm_5d'].mean()*100:+.2f}%)")
        out[board].setdefault("model_density", {})[f"pull_{tag}"] = {
            "rankic": round(float(r_ic), 4),
            "top20_mean_pct": round(float(top["label_pm_5d"].mean()) * 100, 3),
            "base_mean_pct": round(float(te["label_pm_5d"].mean()) * 100, 3),
            "n_te": len(te),
        }


# ── D. 带成本事件回测: 每周期首次满足条件日买入, 持有5日 ──
def backtest(df, board, out):
    print(f"\n  === D. 带成本事件回测 (持有5日, 往返成本 {COST_RT*100:.2f}%) ===")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = df["symbol"]
    pull_win = (df["days_since_up"] >= 22) & (df["days_since_up"] <= 28) & (df["ret_since_prev_up"] <= -0.02)
    shrink = df["ma_vol_ratio_5_20"] < 1.0
    gini_hi = df["chip_gini"] >= df.groupby("date")["chip_gini"].transform("median")
    # 每周期首次进入条件日 (去重, 避免同一只连续多日重复入场)
    first = pull_win & (~pull_win.groupby(sym).shift(1).fillna(False))

    def _run(mask, label):
        rows = df[mask & first & df["label_pm_5d"].notna()]
        if len(rows) == 0:
            print(f"    {label:<24s} (无样本)")
            return
        rows = rows.copy()
        rows["net"] = rows["label_pm_5d"] - COST_RT
        # 按入场日等权组合 → 时间序列
        daily = rows.groupby("date")["net"].mean()
        s = daily.std(ddof=1)
        sharpe = float(daily.mean() / s * np.sqrt(250 / 5)) if s > 0 else 0.0
        per_trade = _median_stat(rows["net"])
        print(f"    {label:<24s} n={len(rows):>6d} 组合均值{float(daily.mean())*100:>+6.2f}% "
              f"Sharpe{sharpe:>6.2f} 交易均值{per_trade['mean_pct']:>+6.2f}% "
              f"交易上涨率{per_trade['hit_pct']:>5.1f}% 覆盖{len(daily):>4d}日")
        return {"n_trades": len(rows), "n_days": len(daily),
                "port_mean_pct": round(float(daily.mean()) * 100, 3),
                "port_sharpe": round(sharpe, 3),
                "trade_mean_pct": per_trade["mean_pct"], "trade_hit_pct": per_trade["hit_pct"]}

    # 基线: 全池 每周期无关 → 直接逐日等权 (不限制为首次)
    base = df[df["label_pm_5d"].notna()].copy()
    base["net"] = base["label_pm_5d"] - COST_RT
    bdaily = base.groupby("date")["net"].mean()
    bs = bdaily.std(ddof=1)
    bsharpe = float(bdaily.mean() / bs * np.sqrt(250 / 5)) if bs > 0 else 0.0
    b_per = _median_stat(base["net"])
    print(f"    {'全池基线(逐日等权)':<24s} 组合均值{float(bdaily.mean())*100:>+6.2f}% "
          f"Sharpe{bsharpe:>6.2f} 交易均值{b_per['mean_pct']:>+6.2f}% "
          f"交易上涨率{b_per['hit_pct']:>5.1f}%")
    res = {"base": {"port_mean_pct": round(float(bdaily.mean()) * 100, 3),
                    "port_sharpe": round(bsharpe, 3),
                    "trade_mean_pct": b_per["mean_pct"], "trade_hit_pct": b_per["hit_pct"]},
           "cost_rt_pct": round(COST_RT * 100, 3),
           "strategies": {}}
    res["strategies"]["回调满月"] = _run(pull_win, "回调满月")
    res["strategies"]["回调满月+缩量"] = _run(pull_win & shrink, "回调满月+缩量")
    res["strategies"]["回调满月+筹码集中"] = _run(pull_win & gini_hi, "回调满月+筹码集中")
    res["strategies"]["回调满月+缩量+筹码集中"] = _run(pull_win & shrink & gini_hi, "回调满月+缩量+筹码集中")
    # 价格分档
    for pl, lo, hi in [("低价<10", 0, 10), ("中价10-30", 10, 30), ("高价>30", 30, np.inf)]:
        res["strategies"][f"回调满月+{pl}"] = _run(pull_win & df["close"].between(lo, hi, inclusive="left"), f"回调满月+{pl}")
    out[board]["backtest"] = res


def analyze(board_df, board, out):
    df = R.LabelEngine.build_labels(board_df)
    df = R.LabelEngine.mask_suspension(df)
    df = R.LabelEngine.mask_recent_days(df, days=6)
    df = R.add_derived(df)
    df = R.add_fwd_labels(df)
    df = add_vol_chip_price(df)
    out[board] = {"n_rows": int(len(df)), "cost_rt_pct": round(COST_RT * 100, 3)}
    print(f"\n{'='*74}\n[{board.upper()}] rows={len(df)}\n{'='*74}")
    check_h4(df, board, out)
    cond_density(df, board, out)
    model_density(df, board, out)
    backtest(df, board, out)


def main():
    feat_files = sorted(glob.glob(os.path.join(OUT_DIR, "limit_feat_*.parquet")))
    if not feat_files:
        print("FATAL: no limit_feat_*.parquet — 先跑 _build_limit_features.py")
        sys.exit(1)
    feat_path = feat_files[-1]
    print(f"[1] feature file: {feat_path}")
    feat = pd.read_parquet(feat_path)
    dates = sorted(feat["date"].unique())

    print("[2] loading panel window + merging (含 量能/筹码/价格 列)...")
    panel = pd.read_parquet(PANEL, columns=[
        "symbol", "date", "open", "high", "low", "close", "volume", "amount",
        "pre_close", "close_hfq", "industry", "board", "circ_mv",
        "turnover_rate", "is_suspended", *VOL_CHIP_COLS,
    ])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(dates)]
    panel = panel.merge(feat, on=["symbol", "date"], how="left")
    for c in ["is_limit_up", "is_limit_down", "is_zhaban", "limit_times",
              "fd_amount_ratio", "open_times"]:
        panel[c] = panel[c].fillna(0.0)
    del feat
    gc.collect()

    print("[3] cleaning...")
    cleaner = R.CleaningPipeline()
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
        "script": "_backtest_limit_vol_chip.py",
        "feature_file": os.path.basename(feat_path),
        "source": "Tushare limit_list_d + 面板(量能/筹码/价格)",
        "cost": f"roundtrip {COST_RT*100:.2f}% (佣金万2.5×2+印花0.05%卖+滑点0.1%×2, settings)",
        "labels": "label_pm_* PM 执行口径; fwd_up_* = 未来继续涨停",
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"limit_volchip_backtest_{TAG}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=1)
    print(f"\nSaved: {path}")
    print("DONE")


if __name__ == "__main__":
    main()
