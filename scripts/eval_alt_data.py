# -*- coding: utf-8 -*-
"""
替代数据源独立训练IC评估 (自包含)
===============================
1. 从 panel_full 取子集 (采样或全量)
2. 逐数据源拉取 + merge (Tushare优先 → AKShare降级)
3. 跑 FeatureEngineV35 全部 26 dims
4. 按数据源分组计算 IC → 输出判决

用法:
  python scripts/eval_alt_data.py                    # 默认: 采样 200 股, 3 年
  python scripts/eval_alt_data.py --sample 0        # 全量 (慢)
  python scripts/eval_alt_data.py --sources fina_indicator,holdernumber,margin
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── IC 计算 ────────────────────────────────────────────────

def daily_rank_ic(df: pd.DataFrame, factor: str, label: str) -> float:
    """日频 Rank IC 均值 (abs)."""
    sub = df[[factor, "date", label]].dropna()
    if len(sub) < 100:
        return 0.0
    ics = []
    for _, g in sub.groupby("date"):
        if len(g) < 10:
            continue
        ic = g[[factor, label]].corr(method="spearman").iloc[0, 1]
        ics.append(ic if not np.isnan(ic) else 0.0)
    return float(np.mean(np.abs(ics))) if ics else 0.0


def rolling_ic_stats(df: pd.DataFrame, factor: str, label: str, window: int = 60):
    """60日滚动 IC 均值 + 正值比例."""
    sub = df[[factor, "date", label]].dropna()
    dates = sorted(sub["date"].unique())
    if len(dates) < window:
        return 0.0, 0.0
    daily_ics = {}
    for d in dates:
        g = sub[sub["date"] == d]
        if len(g) < 10:
            continue
        ic = g[[factor, label]].corr(method="spearman").iloc[0, 1]
        daily_ics[d] = ic if not np.isnan(ic) else 0.0
    s = pd.Series(daily_ics).sort_index().dropna()
    if len(s) < window:
        return 0.0, 0.0
    rolls = [s.iloc[i:i + window].mean() for i in range(len(s) - window)]
    rolls_abs = [abs(r) for r in rolls]
    return float(np.mean(rolls_abs)), float((s.abs() > 0).mean())


# ── 数据源定义 ─────────────────────────────────────────────

# 每个数据源的：上游列前缀 → 产出因子前缀 → 标签
DATA_SOURCE_GROUPS = {
    "fundamental_pit": {
        "name": "基本面PIT",
        "upstream": ["roe", "roa", "gross_margin", "net_margin", "eps_yoy",
                      "rev_yoy", "profit_yoy", "debt_ratio", "current_ratio"],
        "features": ["roe_zscore", "roa_zscore", "margin_composite", "growth_composite",
                      "quality_score", "solvency_zscore", "efficiency_score",
                      "roe_stability", "margin_trend"],
        "freq": "季频",
        "coverage": "98%",
    },
    "shareholder_structure": {
        "name": "股东户数+户均持股",
        "upstream": ["holder_count", "avg_shares_per_holder"],
        "features": ["holder_count_log", "holder_count_qoq", "holder_count_yoy",
                      "holder_qoq_accel", "avg_shares_log", "avg_shares_qoq",
                      "avg_shares_yoy", "holder_concentration_zscore"],
        "freq": "季频",
        "coverage": "95%",
    },
    "margin": {
        "name": "融资融券",
        "upstream": ["margin_balance", "short_balance", "margin_buy_amt"],
        "features": ["margin_balance_chg_1d", "margin_balance_chg_5d",
                      "short_balance_ratio", "margin_buy_ratio",
                      "margin_balance_ma20_dev", "margin_balance_yoy",
                      "margin_pressure_score"],
        "freq": "日频",
        "coverage": "35%",
    },
    "northbound": {
        "name": "北向资金",
        "upstream": ["north_net_buy"],
        "features": ["north_net_buy_5d", "north_net_buy_20d", "north_net_buy_streak",
                      "north_buy_ratio", "north_sh_sz_divergence",
                      "north_momentum_5d", "north_flow_zscore"],
        "freq": "日频",
        "coverage": "15%",
    },
    "lhb": {
        "name": "龙虎榜增强",
        "upstream": ["lhb_net_buy", "lhb_institutional_net_buy"],
        "features": ["lhb_inst_net_buy_5d", "lhb_inst_net_buy_20d",
                      "lhb_inst_count_5d", "lhb_inst_buy_ratio",
                      "lhb_abnormal_score"],
        "freq": "日频(稀疏)",
        "coverage": "5%",
    },
    "industry_flow": {
        "name": "行业资金流聚合",
        "upstream": ["ind_north_flow_5d", "ind_margin_balance_chg",
                      "ind_holder_qoq_mean", "ind_lhb_net_buy_5d"],
        "features": ["ind_north_flow_rank", "ind_margin_rank", "ind_holder_rank",
                      "ind_lhb_activity", "ind_flow_composite"],
        "freq": "日频",
        "coverage": "100%",
    },
    "sector_index": {
        "name": "申万行业指数",
        "upstream": ["sw_ret_1d", "sw_ret_5d", "sw_ret_20d"],
        "features": ["sw_ret_5d", "sw_ret_20d", "sw_vol_20d",
                      "sw_relative_strength", "sw_rotation_position",
                      "sw_momentum_accel", "sw_turnover_anomaly"],
        "freq": "日频",
        "coverage": "100%",
    },
}


# ── 主流程 ─────────────────────────────────────────────────

def load_panel(panel_path: str, n_sample: int = 200) -> pd.DataFrame:
    """加载面板, 可选采样."""
    logger.info("Loading %s", panel_path)
    df = pd.read_parquet(panel_path)
    if n_sample > 0 and n_sample < df["symbol"].nunique():
        rng = np.random.RandomState(42)
        syms = rng.choice(df["symbol"].unique(), n_sample, replace=False)
        df = df[df["symbol"].isin(syms)]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info("Panel: %d rows, %d stocks, %d dates",
                len(df), df["symbol"].nunique(), df["date"].nunique())
    return df


def fetch_and_merge_alt_data(
    panel: pd.DataFrame,
    sources: list[str],
    cache_dir: str = "data/supply_cache",
) -> pd.DataFrame:
    """逐数据源拉取 + merge 到面板."""
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain(cache_dir=cache_dir)
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    symbols = sorted(panel["symbol"].unique().tolist())
    logger.info("Fetching alt data: %s, %d stocks, %s - %s",
                sources, len(symbols), start, end)

    for src in sources:
        t0 = time.time()
        try:
            if src == "fina_indicator":
                # Tushare 全量财务指标 — 单次调用获取全部
                df = supply.fetch_fina_indicator(start_date=start, end_date=end)
                if len(df) and "announce_date" in df.columns:
                    fin_cols = [c for c in df.columns
                               if c not in ("symbol", "report_period", "_ts_code", "announce_date")]
                    cols_avail = ["symbol", "announce_date"] + [c for c in fin_cols if c in df.columns]
                    f = df[cols_avail].sort_values("announce_date")
                    panel = panel.sort_values("date")
                    panel = pd.merge_asof(panel, f,
                        left_on="date", right_on="announce_date",
                        by="symbol", direction="backward")
                    logger.info("  fina_indicator: %d rows, %d cols, %.1fs",
                                len(df), len(fin_cols), time.time() - t0)

            elif src == "holdernumber":
                # 逐股拉取 (Tushare stk_holdernumber 暂无全量API)
                frames = []
                for i, sym in enumerate(symbols[:50]):  # 限制 50 股
                    try:
                        ts_code = f"{sym}.{'SZ' if sym.startswith(('0','3','1')) else 'SH'}"
                        df_one = supply.fetch_holdernumber(ts_code=ts_code,
                            start_date=start, end_date=end)
                        if len(df_one):
                            frames.append(df_one)
                    except Exception:
                        pass
                    if i % 10 == 9:
                        time.sleep(0.05)
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    if "announce_date" in df.columns:
                        hn_cols = [c for c in df.columns
                                  if c not in ("symbol", "date", "_ts_code")]
                        f = df[["symbol", "announce_date"] + hn_cols].sort_values("announce_date")
                        panel = panel.sort_values("date")
                        panel = pd.merge_asof(panel, f,
                            left_on="date", right_on="announce_date",
                            by="symbol", direction="backward")
                        logger.info("  holdernumber: %d stocks, %.1fs",
                                    len(frames), time.time() - t0)

            elif src == "margin":
                # Tushare margin_detail 全量
                df = supply.fetch_margin(start_date=start, end_date=end)
                if len(df):
                    merge_cols = ["symbol", "date"]
                    avail = [c for c in df.columns
                            if c not in merge_cols and not c.startswith("_")]
                    panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
                    logger.info("  margin: %d rows, %.1fs", len(df), time.time() - t0)

            elif src == "northbound":
                df = supply.fetch_northbound(start_date=start, end_date=end)
                if len(df):
                    merge_cols = ["symbol", "date"]
                    avail = [c for c in df.columns
                            if c not in merge_cols and not c.startswith("_")]
                    panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
                    logger.info("  northbound: %d rows, %.1fs", len(df), time.time() - t0)

            elif src == "lhb":
                df = supply.fetch_lhb(start_date=start, end_date=end)
                if len(df):
                    merge_cols = ["symbol", "date"]
                    avail = [c for c in df.columns
                            if c not in merge_cols and not c.startswith("_")]
                    panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
                    logger.info("  lhb: %d rows, %.1fs", len(df), time.time() - t0)

            elif src == "sector_index":
                df = supply.fetch_sector_index(start_date=start, end_date=end)
                if len(df) and "industry" in panel.columns:
                    # 行业映射
                    sw_names = dict(zip(df["index_name"], df["index_code"]))
                    ind_map = {}
                    for ind in panel["industry"].dropna().unique():
                        for sw in sw_names:
                            if ind in sw or sw in ind:
                                ind_map[ind] = sw
                                break
                    if ind_map:
                        panel["_sw"] = panel["industry"].map(ind_map)
                        si = df[["index_name", "date", "ret_pct", "close"]].copy()
                        si = si.rename(columns={"ret_pct": "sw_ret_1d", "close": "sw_close"})
                        panel = panel.merge(si,
                            left_on=["_sw", "date"], right_on=["index_name", "date"],
                            how="left")
                        panel = panel.drop(columns=["_sw", "index_name"], errors="ignore")
                        logger.info("  sector_index: %d indices, %.1fs",
                                    len(sw_names), time.time() - t0)

        except Exception as exc:
            logger.warning("  %s failed: %s", src, exc)

    return panel


def evaluate_and_report(panel: pd.DataFrame, output_dir: str = "data/factor_registry"):
    """构建特征 → 计算 IC → 输出报告."""
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine

    logger.info("Building features + labels...")
    t0 = time.time()

    # 补齐必要列
    from app.pipeline1.cleaning_pipeline import board_of
    if "board" not in panel.columns:
        panel["board"] = panel["symbol"].map(board_of)
    if "industry" not in panel.columns:
        panel["industry"] = "UNKNOWN"
    if "is_st" not in panel.columns:
        panel["is_st"] = False
    if "is_suspended" not in panel.columns:
        panel["is_suspended"] = False
    if "list_days" not in panel.columns:
        panel["list_days"] = panel.groupby("symbol").cumcount() + 1
    if "limit_pct" not in panel.columns:
        from app.pipeline1.cleaning_pipeline import get_limit_pct
        panel["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(panel["board"], panel["date"])]

    fe = FeatureEngineV35()
    df = fe.build(panel)
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)
    logger.info("Features built: %d cols in %.1fs", len(df.columns), time.time() - t0)

    # 确定标签
    label_1d = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    label_3d = "label_3d_net" if "label_3d_net" in df.columns else "label_3d"
    label_5d = "label_5d_net" if "label_5d_net" in df.columns else "label_5d"

    # 逐数据源评估
    results = {}
    print("\n" + "=" * 110)
    print(f"{'数据源':<20s} {'因子数':<6s} {'IC_1d':<8s} {'IC_3d':<8s} {'IC_5d':<8s} {'Roll60':<8s} {'Pos%':<7s} {'NaN%':<7s} {'判决':<10s} {'最优因子'}")
    print("-" * 110)

    THRESHOLDS = {"纳入": 0.02, "条件纳入": 0.01, "暂缓": 0.0}

    for src_id, cfg in DATA_SOURCE_GROUPS.items():
        feats = [c for c in cfg["features"] if c in df.columns]
        if not feats:
            results[src_id] = {
                "name": cfg["name"], "n_features": 0, "ic_1d": 0, "ic_3d": 0,
                "ic_5d": 0, "rolling_mean": 0, "rolling_pos": 0,
                "nan_rate": 1.0, "verdict": "暂缓", "best_factor": "N/A",
                "coverage": cfg["coverage"], "freq": cfg["freq"],
            }
            print(f"{cfg['name']:<20s} {'0':<6s} {'-':<8s} {'-':<8s} {'-':<8s} {'-':<8s} {'-':<7s} {'100%':<7s} {'无数据':<10s} -")
            continue

        best_ic1, best_ic3, best_ic5 = 0, 0, 0
        best_rm, best_rp = 0, 0
        best_f = ""
        best_nan = 0.0

        for f in feats:
            ic1 = daily_rank_ic(df, f, label_1d)
            ic3 = daily_rank_ic(df, f, label_3d)
            ic5 = daily_rank_ic(df, f, label_5d)
            rm, rp = rolling_ic_stats(df, f, label_1d)
            ic_best = max(ic1, ic3, ic5)
            if ic_best > best_ic1:
                best_ic1, best_ic3, best_ic5 = ic1, ic3, ic5
                best_rm, best_rp = rm, rp
                best_f = f
                best_nan = df[f].isna().mean()

        # 判决
        max_ic = max(best_ic1, best_ic3, best_ic5)
        if max_ic >= THRESHOLDS["纳入"] and best_rm >= 0.01 and best_rp >= 0.50:
            verdict = "纳入"
        elif max_ic >= THRESHOLDS["条件纳入"]:
            verdict = "条件纳入"
        else:
            verdict = "暂缓"

        results[src_id] = {
            "name": cfg["name"], "n_features": len(feats),
            "ic_1d": round(best_ic1, 5), "ic_3d": round(best_ic3, 5),
            "ic_5d": round(best_ic5, 5),
            "rolling_mean": round(best_rm, 5), "rolling_pos": round(best_rp, 3),
            "nan_rate": round(best_nan, 3), "verdict": verdict,
            "best_factor": best_f, "coverage": cfg["coverage"], "freq": cfg["freq"],
        }
        print(f"{cfg['name']:<20s} {len(feats):<6d} {best_ic1:<8.4f} {best_ic3:<8.4f} "
              f"{best_ic5:<8.4f} {best_rm:<8.4f} {best_rp:<7.2f} "
              f"{best_nan*100:<7.1f} {verdict:<10s} {best_f}")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"alt_data_eval_{datetime.now():%Y%m%d_%H%M}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Panel: {df['symbol'].nunique()} stocks x {df['date'].nunique()} dates = {len(df)} rows")

    # 汇总
    included = [k for k, v in results.items() if v["verdict"] == "纳入"]
    pending = [k for k, v in results.items() if v["verdict"] == "条件纳入"]
    skipped = [k for k, v in results.items() if v["verdict"] in ("暂缓", "无数据")]
    print(f"\n纳入: {included or '无'}")
    print(f"待定: {pending or '无'}")
    print(f"暂缓: {skipped or '无'}")

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/panel_full_enriched.parquet")
    p.add_argument("--sample", type=int, default=200, help="采样股数, 0=全量")
    p.add_argument("--sources", default="fina_indicator,holdernumber,margin,northbound,lhb,sector_index")
    p.add_argument("--output-dir", default="data/factor_registry")
    p.add_argument("--skip-fetch", action="store_true", help="跳过拉取, 面板已有 alt 列")
    args = p.parse_args()

    df = load_panel(args.panel, args.sample)

    if not args.skip_fetch:
        sources = [s.strip() for s in args.sources.split(",")]
        df = fetch_and_merge_alt_data(df, sources)

    evaluate_and_report(df, args.output_dir)


if __name__ == "__main__":
    main()
