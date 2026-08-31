"""LEGACY 推理叠加模块 (做法一, 2026-08-05).

把并行 PIPELINE 今日合并短名单 (狙击TOP-5 ∪ 融合TOP-10, 共现优先) 的 symbol
→ 现成 LEGACY (V35Predictor) 推理 prob_up / pred_ret / composite_score
→ 与池分合成最终分再排, 作为确认/再排层 (只推理, 不重训).

原则:
  - 不改 LEGACY 训练逻辑, 不改并行系统选股逻辑 (隔离原则部分让渡给刻意的推理耦合)
  - 面板已含模型 feature_cols (main 312/312, dual 75/75) → 直接推理, 无需重建特征
  - 无前瞻: 只用当日截面特征 + 当日已训好的模型

用法:
    python -m app.pipeline_parallel.legacy_overlay
    python -m app.pipeline_parallel.legacy_overlay --board main --w-pool 0.6 --w-prob 0.4
    python -m app.pipeline_parallel.legacy_overlay --out data/_legacy_overlay_<ts>.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.pipeline_parallel.config import MAG10D_CAL, OVERLAY_WEIGHTS

MODEL_DIR = "models/pipeline1"

logger = logging.getLogger(__name__)
CHECKPOINTS = {
    "main": os.path.join("data", "_diag_stage_main_3y.parquet"),
    "dual": os.path.join("data", "_diag_stage_dual_3y.parquet"),
}
# rerank 输出的 prob/pred 列优先级 (prob_up_3d 匹配狙击主视界 T+3)
_PROB_COL_PRIORITY = ("prob_up_3d", "prob_up_5d", "prob_up_10d", "prob_up")
_LEGACY_COLS = (
    "prob_up",
    "prob_up_3d",
    "prob_up_5d",
    "prob_up_10d",
    "pred_ret_3d",
    "pred_ret_5d",
    "pred_ret_10d",
    "composite_score",
    "rank_score",
)


def cross_section(board: str, date=None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """读该板块检查点决策日历史窗 (含 label_pm_10d_net + board 列), 供 mag_10d 校准.

    2026-08-07: build_merged_shortlist 改按 mag_10d (score→label_pm_10d_net) 排名,
    需决策日前 cal_n+realized_drop 个交易日的已实现标签做拟合 → 返回历史窗而非单日.
    窗: [决策日 - (cal_n + realized_drop + 40 交易日), 决策日]. date=None → 最新交易日.
    返回 (df, 该日 date); df 多日, 调用方以 df[df["date"]==day] 取当日截面.
    """
    d = pd.read_parquet(CHECKPOINTS[board], columns=["date"])
    day = pd.Timestamp(date) if date is not None else d["date"].max()
    dates = np.sort(d["date"].unique())
    di = int(np.searchsorted(dates, np.datetime64(day), side="right"))
    _rd = int(MAG10D_CAL["buy_lag"]) + int(MAG10D_CAL["label_horizon"])
    _win = int(MAG10D_CAL["cal_n"]) + _rd + 40
    lo = dates[max(0, di - _win)]
    df = pd.read_parquet(
        CHECKPOINTS[board], filters=[("date", ">=", lo), ("date", "<=", day)]
    )
    df = df.copy()
    df["board"] = board
    return df, day


def _pick_prob_col(res: pd.DataFrame) -> str:
    """按优先级取有真实数据 (非全 NaN 占位) 的概率列; 无 → ''."""
    return next(
        (c for c in _PROB_COL_PRIORITY if c in res.columns and res[c].notna().any()), ""
    )


def overlay_weights(board: str) -> tuple[float, float]:
    """该板块叠加权重 (config.OVERLAY_WEIGHTS); 未知板块 → 对半 (0.5, 0.5)."""
    w = OVERLAY_WEIGHTS.get(board, {"w_pool": 0.5, "w_prob": 0.5})
    return float(w["w_pool"]), float(w["w_prob"])


# 前向跟踪快照: 每交付日截取叠加再排结果, 供数日后 join 已实现 MFE 验证改动正误.
# 归一化: 实际命中的 prob 列 → 稳定列名 prob_up (与具体 prob_col 无关);
# 附加该次实际应用的 w_pool/w_prob (config 板块默认或 CLI 覆盖) + 来源 prob_col.
SNAPSHOT_COLS = (
    "board",
    "date",
    "symbol",
    "systems",
    "co_occur",
    "rk_pool",
    "rk_final",
    "score",
    "prob_up",
    "final_score",
    "pred_ret_3d",
    "composite_score",
    "w_pool",
    "w_prob",
    "prob_col",
)


def overlay_snapshot_frame(
    res: pd.DataFrame, board: str, date, w_pool: float, w_prob: float
) -> pd.DataFrame:
    """把 rerank 输出截成前向跟踪快照帧 (纯函数).

    列 = SNAPSHOT_COLS 固定顺序; prob 值取 res.attrs["prob_col"] 命中的列 → prob_up;
    无 prob 数据 → prob_col='', prob_up 全 NaN. board/date/w_pool/w_prob/prob_col
    由调用方传入并 stamp, 保证日后回看能还原当次口径.
    """
    prob_col = res.attrs.get("prob_col", "") or ""
    if prob_col and prob_col in res.columns and prob_col != "prob_up":
        # rerank 恒留 prob_up 占位列 (可能全 NaN 或非本次命中的信号) → 先删避免 rename 撞列
        out = res.copy().drop(columns="prob_up", errors="ignore")
        out = out.rename(columns={prob_col: "prob_up"})
    elif "prob_up" in res.columns:
        out = res.copy()  # prob_up 已存在 (全 NaN 占位或本来就是命中列)
    else:
        out = res.assign(prob_up=np.nan)
    out = out.reindex(columns=SNAPSHOT_COLS)
    out["board"] = board
    out["date"] = str(pd.Timestamp(date).date())
    out["w_pool"] = float(w_pool)
    out["w_prob"] = float(w_prob)
    out["prob_col"] = prob_col
    return out


def rerank(
    shortlist: pd.DataFrame,
    legacy: pd.DataFrame,
    w_pool: float = 0.5,
    w_prob: float = 0.5,
    prob_col: str = "prob_up_3d",
) -> pd.DataFrame:
    """池分与 LEGACY 概率合成最终分再排 (纯函数).

    final_score = w_pool * score + w_prob * prob (prob 缺失 → 只用池分项).
    排序: co_occur 优先 (双系统一致 = 大仓, 仓位纪律), 组内 final_score 降序.
    保留 rk_pool (纯池分次序) 与 rk_final (叠加后次序) 供对比.

    shortlist: build_merged_shortlist 输出 (date/symbol/systems/co_occur/score/rk).
    legacy: V35Predictor.predict 输出 (symbol + prob/pred/composite 列), 可为空.
    """
    res = shortlist.copy()
    res["rk_pool"] = res["rk"]
    if not legacy.empty:
        avail = [c for c in _LEGACY_COLS if c in legacy.columns]
        if avail:
            res = res.merge(legacy[["symbol"] + avail], on="symbol", how="left")
    for c in _LEGACY_COLS:
        if c not in res.columns:
            res[c] = np.nan
    if prob_col not in res.columns or res[prob_col].isna().all():
        prob_col = _pick_prob_col(res)
    prob = res[prob_col] if prob_col else pd.Series(np.nan, index=res.index)
    res["final_score"] = w_pool * res["score"] + w_prob * prob.fillna(0.0)
    res = res.sort_values(["co_occur", "final_score"], ascending=[False, False])
    res["rk_final"] = np.arange(1, len(res) + 1)
    res.attrs["prob_col"] = prob_col
    return res


def legacy_predict(
    df: pd.DataFrame, board: str, predictor, symbols: set[str]
) -> pd.DataFrame:
    """对决策日 symbols 推理 (predictor 取每 symbol 最新行 = 决策日).

    df: 含历史的窗口帧 (predictor 内部 tail(1) 取决策日; 多日历史使
    brute 推理补齐可算滚动窗). symbols 为空 → 空表.
    """
    day = df[df["symbol"].isin(symbols)]
    if day.empty:
        return pd.DataFrame()
    return predictor.predict(day, board)


def _print_overlay(
    res: pd.DataFrame, board: str, date, w_pool: float, w_prob: float, prob_col: str
) -> None:
    logger.info(
        f"\n=== 板块 [{board}] | {pd.Timestamp(date).date()} "
        f"| w_pool={w_pool:.2f} w_prob={w_prob:.2f} ({prob_col or '无 prob'}) ==="
    )
    cols = [
        "rk_pool",
        "rk_final",
        "symbol",
        "systems",
        "co_occur",
        "score",
        prob_col if prob_col else "",
        "pred_ret_3d",
        "composite_score",
        "final_score",
    ]
    cols = [c for c in cols if c]
    show = res[cols].copy()
    show[prob_col] = show[prob_col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    show["pred_ret_3d"] = show["pred_ret_3d"].map(
        lambda v: f"{v:+.1%}" if pd.notna(v) else "-"
    )
    show["composite_score"] = show["composite_score"].map(
        lambda v: f"{v:+.1%}" if pd.notna(v) else "-"
    )
    show["score"] = show["score"].map(lambda v: f"{v:.3f}")
    show["final_score"] = show["final_score"].map(lambda v: f"{v:.3f}")
    show["co_occur"] = show["co_occur"].map(lambda v: "*" if v else "")
    logger.info("\n" + show.to_string(index=False))


def _write_snapshot(snap: pd.DataFrame, out_dir, module: str | None = None) -> None:
    """WORM 写前向跟踪快照 (按 date 分文件; 同名存在则跳过不覆盖).

    文件名 overlay_track_{date}__{module}.csv (module 来自 current_meta tag,
    与 _shortlist_t5_t10/_deliver_legacy_list 命名惯例一致). 未来验证器按
    overlay_track_*.csv 收集后 join 已实现 MFE. module=None → 运行期解析.
    """
    if module is None:
        from app.pipeline1.model_meta import load_modules, module_id

        module = module_id(load_modules())
    suffix = f"__{module}" if module != "na" else ""
    for day, grp in snap.groupby("date", sort=True):
        stamp = str(day).replace("-", "")
        path = Path(out_dir) / f"overlay_track_{stamp}{suffix}.csv"
        if path.exists():
            logger.warning(f"快照已存在, 跳过覆盖 (WORM): {path.name}")
            continue
        grp.to_csv(path, index=False)
        logger.info(f"[snapshot] {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LEGACY 推理叠加: 今日合并短名单 → prob → 再排"
    )
    ap.add_argument("--board", default=None, help="main/dual, 默认两者")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD, 默认每板块最新交易日")
    ap.add_argument(
        "--w-pool",
        type=float,
        default=None,
        help="池分权重覆盖 (默认按板块 config.OVERLAY_WEIGHTS: main 0.2 / dual 0.5)",
    )
    ap.add_argument(
        "--w-prob",
        type=float,
        default=None,
        help="LEGACY prob 权重覆盖 (默认按板块 config.OVERLAY_WEIGHTS: main 0.8 / dual 0.5)",
    )
    ap.add_argument("--out", default=None, help="WORM CSV 落盘路径")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from app.pipeline1.predict_runner import resolve_current_bundles
    from app.pipeline1.predictor import V35Predictor
    from app.pipeline_parallel.backtest import build_merged_shortlist
    from config.settings import STOCK_LIST_DIR

    bundles = resolve_current_bundles(MODEL_DIR)
    if not bundles:
        logger.error(f"无可用模型包: {MODEL_DIR}")
        return 1
    predictor = V35Predictor(bundles)
    logger.info(f"模型包: { {b: os.path.basename(p) for b, p in bundles.items()} }")

    boards = [args.board] if args.board else ["main", "dual"]
    frames: list[pd.DataFrame] = []
    snaps: list[pd.DataFrame] = []
    for board in boards:
        if board not in bundles:
            logger.info(f"[{board}] 无模型包, 跳过")
            continue
        df, day = cross_section(board, date=args.date)
        if df.empty:
            logger.info(f"[{board}] {pd.Timestamp(day).date()} 截面为空")
            continue
        logger.info(f"[{board}] 决策日 {day.date()} 历史窗行数 {len(df):,}")
        # 2026-08-07: build_merged_shortlist 需完整历史窗做 mag_10d 校准,
        # 输出全窗逐日短名单 → 只取决策日.
        sl = build_merged_shortlist(df, top_n=10)
        sl = sl[sl["date"] == day] if not sl.empty else sl
        if sl.empty:
            logger.info(f"[{board}] 合并短名单为空")
            continue
        leg = legacy_predict(df, board, predictor, set(sl["symbol"]))
        w_pool, w_prob = overlay_weights(board)
        if args.w_pool is not None:
            w_pool = args.w_pool
        if args.w_prob is not None:
            w_prob = args.w_prob
        res = rerank(sl, leg, w_pool, w_prob)
        prob_col = res.attrs.get("prob_col", "")
        # build_merged_shortlist 现输出 board 列 (跨板块历史窗) → 去重后规范置首
        if "board" in res.columns:
            res = res.drop(columns=["board"])
        res.insert(0, "board", board)
        res["date"] = str(pd.Timestamp(day).date())
        frames.append(res)
        snaps.append(overlay_snapshot_frame(res, board, day, w_pool, w_prob))
        _print_overlay(res, board, day, w_pool, w_prob, prob_col)

    if frames and args.out:
        pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
        logger.info(f"\nWORM 落盘: {args.out}")
    if snaps:
        _write_snapshot(pd.concat(snaps, ignore_index=True), STOCK_LIST_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
