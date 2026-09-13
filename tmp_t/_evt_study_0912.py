# -*- coding: utf-8 -*-
"""0912 全池事件研究: 压缩 × 筹码 × S-L持续 三条件 → 后续上涨条件概率表.

研究问题 (用户 0911 定): 三条件交叉分桶后, 后续 3d/5d/10d 上涨条件概率 vs 全池无条件基线.
全池 = 3y 全宇宙 (board="main" 即全宇宙, MAIN-DUAL 是行子集), 不分板不分行情.

条件列:
  压缩 = "带20"        20日高低带宽度 (high.max20/low.min20 - 1), 越小越压缩 (dim09)
  筹码 = "winner_ratio" 获利比例 (Tushare cyq_perf, dim21)。注意: 不用 dim09 "获利盘" —
        ChipDistribution 全帧网格端点前视 (chip_distribution.py:52, 合成验证历史行
        max 漂移 58pp, 2026-09-12 夜确认), winner_ratio 独立数据源同概念干净替代,
        且两板 pin 在册覆盖完整 (nan≈2e-05)。越大套牢越少。
  持续 = "SL多头持续天数" 短期线在长期线上方连续天数 (sl>0 run-length, dim09)

分桶 (描述性研究非预测入模 → 边界用全池分位, 非 train 段):
  带20 / winner_ratio: 全池 rank(pct) 五分位 Q1..Q5 (Q1=值最小; 并列值按平均名次分摊)
  SL多头持续天数: 离散桶 [0] / [1-5] / [6-20] / [>20]

输出 (WORM, data_others_path("diag")):
  evt_study_0912_{ts}.csv            长表 5×5×4×3视界 = 300 行
  evt_study_0912_{ts}_heatmap_p_up.csv  降维表A: 每 horizon 取持续最优档的 P(label>0) 5×5
  evt_study_0912_{ts}_heatmap_mean.csv  降维表B: 同口径 mean(label) 5×5
  evt_study_0912_{ts}_summary.json    桶边界 + 基线 + 最优持续档 + 落盘路径
每格: n, P(label>0), mean(label), 无条件基线及差值 (基线=三条件全非空的分析总体).

用法: python tmp_t/_evt_study_0912.py  (自写日志 tmp_t/_evt_study_0912.log)
重活: 哨兵注册由主会话做 (本文件只带 find_conflicts 并发守卫, 有冲突 rc=3 拒跑).
"""

import gc
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from config.settings import (
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

TAG = "evt_study_0912"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_evt_study_0912.log")
REPORT_DIR = data_others_path("diag")

COND_COLS = ["带20", "winner_ratio", "SL多头持续天数"]
LABELS = ["label_pm_3d_net", "label_pm_5d_net", "label_pm_10d_net"]
N_Q = 5
Q_LABELS = list(range(1, N_Q + 1))  # Q1=值最小 (带20 Q1=最压缩)
DUR_BINS = [-np.inf, 0, 5, 20, np.inf]  # 右闭: (..0]=0, (0,5]=1-5, (5,20]=6-20, (20,..]=>20
DUR_LABELS = ["0", "1-5", "6-20", ">20"]

log = logging.getLogger(TAG)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def build_frame() -> pd.DataFrame:
    """3y 面板 → 清洗(main=全宇宙) → 特征帧 → 立刻瘦身只留条件+标签列 (15.8GB 双驻留陷阱)."""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.train_runner import prepare_board_frame

    t0 = time.time()
    panel = load_panel_v3(path=PANEL_V3_PATH)
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel, board="main")
    del dual_df, panel
    gc.collect()
    df = prepare_board_frame(
        main_df,
        FeatureEngineV35(),
        None,
        cross_sectional_rank=False,
        registry=None,
        label_excess=False,
    )
    del main_df
    gc.collect()

    keep = ["symbol", "date"] + COND_COLS + LABELS
    missing = [c for c in keep if c not in df.columns]
    if missing:  # 失败要大声: 条件/标签列缺 = 研究无法进行, 不静默降级
        raise SystemExit(f"[FATAL] 帧缺列: {missing} (期望 {keep})")
    df = df[keep]
    gc.collect()
    log.info(
        "[frame] 构建完成 %d 行, 瘦身后 %d 列; 条件列空值率 %s (%.0fs)",
        len(df), df.shape[1],
        {c: round(float(df[c].isna().mean()), 4) for c in COND_COLS},
        time.time() - t0,
    )
    return df


def _quintile(s: pd.Series) -> pd.Series:
    """全池 rank(pct) 五分位 (并列值平均名次分摊); 返回 categorical 保 5×5×4 笛卡尔完整."""
    q = np.ceil(s.rank(pct=True, method="average") * N_Q)
    q = q.where(s.notna())
    return pd.Categorical(q, categories=Q_LABELS)


def add_buckets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """向量化分桶; 返回 (带桶列且三条件非空的分析总体, 边界快照)."""
    edges = {}
    df = df.copy()
    df["q_comp"] = _quintile(df["带20"])        # Q1 = 最压缩 (带最小)
    df["q_chip"] = _quintile(df["winner_ratio"])  # Q1 = 获利盘最低 (套牢多)
    df["dur"] = pd.cut(df["SL多头持续天数"], bins=DUR_BINS, labels=DUR_LABELS)
    # rank 分位是唯一桶准则; 分位数值边界仅供 summary 可读性参考
    for c, name in (("带20", "q_comp"), ("winner_ratio", "q_chip")):
        qs = np.nanquantile(df[c], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        edges[name] = [round(float(v), 6) for v in qs]
    edges["dur_bins"] = {"0": "=0", "1-5": "1..5", "6-20": "6..20", ">20": ">=21"}
    n0 = len(df)
    df = df.dropna(subset=["q_comp", "q_chip", "dur"])
    log.info("[bucket] 分析总体 %d/%d 行 (剔三条件空值 %d), 边界快照已取", len(df), n0, n0 - len(df))
    return df, edges


def cell_table(pop: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """长表: 5×5×4×3 视界, 每格 n/p_up/mean + 基线差值; 及持续档边际表 (选最优档用).

    全 groupby 向量化, 循环只跑 3 个视界 (禁 for 遍历股票).
    observed=False + categorical → 空格也保留 (n=0, 统计 NaN), 100 格完整.
    """
    keys = ["q_comp", "q_chip", "dur"]
    out, marg = [], []
    for hz in LABELS:
        sub = pop[keys + [hz]].dropna(subset=[hz])
        g = sub.groupby(keys, observed=False)
        t = g[hz].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "mean_label"})
        t["p_up"] = sub[hz].gt(0).astype(float).groupby([sub[k] for k in keys], observed=False).mean()
        base_p, base_m = float(sub[hz].gt(0).mean()), float(sub[hz].mean())
        t["horizon"], t["base_p_up"], t["base_mean"] = hz, base_p, base_m
        t["diff_p_up"], t["diff_mean"] = t["p_up"] - base_p, t["mean_label"] - base_m
        out.append(t.reset_index())
        # 持续档边际 (按 n 加权的真实边际, 非格均值): 每 horizon 选 P(label>0) 最高档
        gm = sub.groupby("dur", observed=False)[hz]
        m = pd.DataFrame({"n": gm.size(), "p_up": sub[hz].gt(0).astype(float)
                          .groupby(sub["dur"], observed=False).mean()})
        m["horizon"] = hz
        marg.append(m.reset_index())
    long = pd.concat(out, ignore_index=True)
    marginal = pd.concat(marg, ignore_index=True)
    log.info("[cells] 长表 %d 行 (含空格), 基线 %s",
             len(long), {r.horizon: round(r.base_p_up, 4) for r in long.itertuples() if r.dur == DUR_LABELS[0]})
    return long, marginal


def heatmap_table(long: pd.DataFrame, marginal: pd.DataFrame, value: str) -> pd.DataFrame:
    """降维 5×5 表: 每 horizon 固定持续=最优档 (边际 P(label>0) 最高), 压缩×筹码格."""
    blocks = []
    for hz in LABELS:
        mg = marginal[marginal["horizon"] == hz]
        best = str(mg.loc[mg["p_up"].idxmax(), "dur"])
        cell = long[(long["horizon"] == hz) & (long["dur"] == best)]
        grid = (cell.pivot(index="q_comp", columns="q_chip", values=value)
                .reindex(index=Q_LABELS, columns=Q_LABELS))
        grid.columns = [f"chip_Q{c}" for c in grid.columns]
        grid = grid.reset_index()
        grid.insert(0, "dur_best", best)
        grid.insert(0, "horizon", hz)
        blocks.append(grid)
    return pd.concat(blocks, ignore_index=True)


def _write_csv(df: pd.DataFrame, path) -> str:
    df.to_csv(path, index=False, encoding="utf-8-sig")  # Excel 打开中文列名不乱码
    log.info("[WORM] %s", path)
    return str(path)


def main() -> int:
    _setup_logging()
    from scripts._run_guard import find_conflicts

    conflicts = find_conflicts()
    if conflicts:
        for c in conflicts:
            log.error("[guard] 冲突: %s (PID %s)", c["sentinel"], c["pid"])
        return 3
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    ts = time.strftime("%Y%m%d_%H%M%S")
    df = build_frame()
    pop, edges = add_buckets(df)
    del df
    gc.collect()
    long, marginal = cell_table(pop)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p_long = _write_csv(long, REPORT_DIR / f"{TAG}_{ts}.csv")
    p_pup = _write_csv(heatmap_table(long, marginal, "p_up"),
                       REPORT_DIR / f"{TAG}_{ts}_heatmap_p_up.csv")
    p_mean = _write_csv(heatmap_table(long, marginal, "mean_label"),
                        REPORT_DIR / f"{TAG}_{ts}_heatmap_mean.csv")

    best = {hz: str(marginal[marginal["horizon"] == hz].loc[
        marginal[marginal["horizon"] == hz]["p_up"].idxmax(), "dur"]) for hz in LABELS}
    bl_rows = long[long["dur"] == DUR_LABELS[0]][["horizon", "base_p_up", "base_mean"]].drop_duplicates("horizon")
    baselines = {r.horizon: {"p_up": round(float(r.base_p_up), 6), "mean": round(float(r.base_mean), 6)}
                 for r in bl_rows.itertuples(index=False)}
    summary = {
        "tag": TAG, "ts": ts, "n_pop": int(len(pop)),
        "bucket_note": "带20/winner_ratio=全池rank五分位(描述性研究, 非train段); Q1=值最小(带20 Q1=最压缩, "
                       "winner_ratio Q1=获利盘最低); 筹码轴用 dim21 winner_ratio 规避 dim09 ChipDistribution 网格前视",
        "quantile_edges_ref": edges, "dur_best_by_horizon": best,
        "baselines_analysis_pop": baselines,
        "dur_marginal": marginal.to_dict(orient="records"),
        "outputs": {"long": p_long, "heatmap_p_up": p_pup, "heatmap_mean": p_mean},
    }
    spath = REPORT_DIR / f"{TAG}_{ts}_summary.json"
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", spath)
    log.info("[DONE] 三条件×三视界事件研究完成: 300格长表 + 2张5x5降维表 + summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
