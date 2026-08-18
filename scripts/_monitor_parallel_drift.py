"""_monitor_parallel_drift.py — parallel dual 幅度漂移监控 (2026-08-18).

与 legacy 漂移监控同构 (08-17 定案: 修漂移=重训, 监控=每日全池预测均值 vs 实现偏差),
但数据源为并行系统: dual 特征冻结后无每周重选, 漂移监控成为季度重选前唯一的漂移信号.

数据源 (全部 WORM, 无新落盘):
  preds    = BACKTEST_RESULT_DIR/<ts>/last_*_days_picks_dual.csv (每日短名单,
             (date,symbol) 去重 keep-first = 决策日当天生成的那份, as-predicted 口径)
  realized = data/_diag_stage_dual_3y.parquet 的 label_pm_10d_net (刷新后检查点,
             与校准目标同口径 — 无 lag/cost 歧义, 仅限 dual)

输出: 成熟日数/最新偏差/滚动偏差 vs 阈值 + 告警; WORM 报告
  DATA OTHERS/diag/parallel_drift_<ts>.json (含逐日明细). 退出码恒 0 (告警不进自动化失败).
用法: python scripts/_monitor_parallel_drift.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from app.pipeline1.drift_monitor import (
    accumulate_parallel_picks,
    check_drift,
    daily_bias,
    rolling_bias,
)
from config.settings import (
    DATA_DIR,
    PARALLEL_DRIFT_MONITOR,
    data_others_path,
)


def _fmt_pct(v) -> str:
    """None/NaN → '—' (积累期滚动偏差未成形), 否则 +%.2%."""
    if v is None:
        return "—"
    try:
        return f"{float(v):+.2%}"
    except (TypeError, ValueError):
        return "—"


def main() -> int:
    cfg = PARALLEL_DRIFT_MONITOR
    window_days = int(cfg["window_days"])
    min_matured = int(cfg["min_matured_days"])
    thresholds = cfg["bias_threshold"]

    preds = accumulate_parallel_picks(data_others_path(cfg["run_root"]))
    if len(preds):
        preds["date"] = pd.to_datetime(preds["date"])
    print(
        f"[preds] {len(preds):,} 票 / {preds['date'].nunique() if len(preds) else 0} 日",
        flush=True,
    )
    if not len(preds):
        print("[warn] 无 last_*_days_picks_dual.csv (并行短名单未落盘), 退出", flush=True)
        return 0

    ckpt = pd.read_parquet(
        DATA_DIR / cfg["checkpoint_dual"],
        columns=["date", "symbol", "label_pm_10d_net"],
    )
    ckpt["symbol"] = ckpt["symbol"].astype(str)
    realized = ckpt.rename(columns={"label_pm_10d_net": "realized_net"}).dropna(
        subset=["realized_net"]
    )
    print(
        f"[realized] 检查点成熟 {len(realized):,} 票, "
        f"最新 {pd.Timestamp(ckpt['date'].max()).date()}",
        flush=True,
    )

    daily = daily_bias(preds, realized)
    rolling = rolling_bias(daily, window_days, min_matured)
    alerts = check_drift(rolling, thresholds)

    print(
        f"===== parallel dual 幅度漂移 (滚动窗 {window_days} 交易日, 成熟≥{min_matured} 日) =====",
        flush=True,
    )
    for _, r in rolling.iterrows():
        th = thresholds.get(r["board"])
        line = (
            f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日 | 最新日 "
            f"{pd.Timestamp(r['latest_date']).date()} 偏差 {_fmt_pct(r['latest_bias'])} | "
            f"滚动偏差 {_fmt_pct(r['bias'])}"
        )
        if th is not None:
            line += f" vs 阈值 {th:+.2%}"
        if any(a["board"] == r["board"] for a in alerts):
            line += "  [DRIFT-ALERT] 考虑提前重训"
        print(line, flush=True)
    if not len(rolling):
        print(
            "[info] 无成熟日 — 积累期 (首个成熟日约在首个短名单后 10 交易日)",
            flush=True,
        )
    if not alerts:
        print("[ok] 无板块滚动偏差超阈值", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    report = {
        "ts": ts,
        "config": cfg,
        "n_pred_rows": int(len(preds)),
        "pred_dates": [str(d) for d in sorted(preds["date"].unique())],
        "checkpoint_last": str(pd.Timestamp(ckpt["date"].max()).date()),
        "daily": daily.to_dict("records"),
        "rolling": rolling.to_dict("records"),
        "alerts": alerts,
    }
    (out_dir / f"parallel_drift_{ts}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/parallel_drift_{ts}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
