"""快速核查: 主板 涨停回调满月+低价<10 标签的收益分布 (上涨概率 + 赔率形状).

回答用户: "价格因子标签预测上涨概率高嘛" — 需要 命中率 + 中位数/分位/头部贡献,
不能只看均值 (均值会被少数大赚票拉高).
"""
import gc
import glob
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd

import _research_limit_strong as R
from app.core.config_loader import load_config

_bt_cfg = load_config("backtest_config").get("backtest", {})
COST_RT = (
    2 * _bt_cfg.get("commission_rate", 0.00025)
    + _bt_cfg.get("stamp_tax_rate", 0.001)
    + (_bt_cfg.get("slippage_buy_bp", 10) + _bt_cfg.get("slippage_sell_moo_bp", 10)) / 10000
)


def _dist(ser, label):
    ser = ser.dropna()
    n = len(ser)
    if n < 10:
        print(f"    {label:<20s} (样本不足)")
        return
    hit = (ser > 0).mean() * 100
    p = ser.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).values
    top10_mean = ser.nlargest(int(n * 0.1)).mean()
    rest_mean = ser.nsmallest(int(n * 0.9)).mean()
    print(f"    {label:<20s} n={n:>6d} 上涨率={hit:5.1f}% 均值={ser.mean()*100:+6.2f}% "
          f"中位={p[2]*100:+6.2f}% p10={p[0]*100:+6.2f}% p90={p[4]*100:+6.2f}%")
    print(f"                        max赚={ser.max()*100:+6.1f}% max亏={ser.min()*100:+6.1f}% "
          f"top10%均值={top10_mean*100:+6.2f}% (其余90%均值={rest_mean*100:+6.2f}%)")
    return {"n": n, "hit": round(hit, 2), "mean": round(ser.mean() * 100, 3),
            "median": round(p[2] * 100, 3), "p10": round(p[0] * 100, 3), "p90": round(p[4] * 100, 3),
            "top10_mean": round(top10_mean * 100, 3), "rest90_mean": round(rest_mean * 100, 3)}


def main():
    feat_files = sorted(glob.glob(os.path.join(R.OUT_DIR, "limit_feat_*.parquet")))
    feat = pd.read_parquet(feat_files[-1])
    dates = sorted(feat["date"].unique())
    panel = pd.read_parquet(R.PANEL, columns=[
        "symbol", "date", "open", "high", "low", "close", "volume", "amount",
        "pre_close", "close_hfq", "industry", "board", "circ_mv",
        "turnover_rate", "is_suspended", "ma_vol_ratio_5_20", "winner_ratio",
        "chip_entropy", "chip_gini", "cost_50pct", "cost_95pct", "cost_bias",
        "pct_90_high", "pct_90_con",
    ])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(dates)]
    panel = panel.merge(feat, on=["symbol", "date"], how="left")
    for c in ["is_limit_up", "is_limit_down", "is_zhaban", "limit_times", "fd_amount_ratio", "open_times"]:
        panel[c] = panel[c].fillna(0.0)
    del feat; gc.collect()

    cleaner = R.CleaningPipeline()
    main_df, _ = cleaner.run_train(panel)
    del panel; gc.collect()

    df = R.LabelEngine.build_labels(main_df)
    df = R.LabelEngine.mask_suspension(df)
    df = R.LabelEngine.mask_recent_days(df, days=6)
    df = R.add_derived(df)
    df = R.add_fwd_labels(df)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = df["symbol"]
    pull_win = (df["days_since_up"] >= 22) & (df["days_since_up"] <= 28) & (df["ret_since_prev_up"] <= -0.02)
    first = pull_win & (~pull_win.groupby(sym).shift(1).fillna(False))
    low = pull_win & df["close"].between(0, 10, inclusive="left")

    print(f"往返成本 {COST_RT*100:.2f}% | 持有5日 (label_pm_5d, 净=毛-成本)")
    print("=" * 78)
    print("  [主板] 交易分布")
    print(f"  --- 条件: 涨停后回调满月(22~28日,<-2%), 每周期首次入场 ---")
    base = df[df["label_pm_5d"].notna()].copy()
    base["net"] = base["label_pm_5d"] - COST_RT
    _dist(base["net"], "全池基线")
    pb = df[pull_win & first & df["label_pm_5d"].notna()].copy()
    pb["net"] = pb["label_pm_5d"] - COST_RT
    _dist(pb["net"], "回调满月")
    pb_low = df[pull_win & first & low & df["label_pm_5d"].notna()].copy()
    pb_low["net"] = pb_low["label_pm_5d"] - COST_RT
    _dist(pb_low["net"], "回调满月+低价<10")

    print("\n  --- 纯价格因子对照 (全池 不限涨停, 看低价是否有普遍性) ---")
    all_df = df[df["label_pm_5d"].notna()].copy()
    all_df["net"] = all_df["label_pm_5d"] - COST_RT
    for lo, hi, lab in [(0, 10, "全池低价<10"), (10, 30, "全池中价10-30"), (30, np.inf, "全池高价>30")]:
        _dist(all_df[all_df["close"].between(lo, hi, inclusive="left")]["net"], lab)
    print("DONE")


if __name__ == "__main__":
    main()
