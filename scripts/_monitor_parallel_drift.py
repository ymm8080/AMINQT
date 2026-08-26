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
08-26 增补 ECE 校准节: STOCK LIST 短名单 pred_prob_10d vs 盘中 MFE 实现事件
  (net MFE > mfe_threshold, 口径同 backtest.add_mfe_labels), 滚动 ECE 超阈值告警.
用法: python scripts/_monitor_parallel_drift.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from app.pipeline1.drift_monitor import (
    accumulate_parallel_picks,
    check_calibration,
    check_drift,
    compute_realized_mfe,
    daily_bias,
    rolling_bias,
    rolling_calibration_mfe,
)
from config.settings import (
    DATA_DIR,
    PANEL_V3_PATH,
    PARALLEL_DRIFT_MONITOR,
    STOCK_LIST_DIR,
    data_others_path,
)


def _load_shortlist_probs() -> pd.DataFrame:
    """STOCK LIST 的 parallel_shortlist_*.csv → (date, board, symbol, pred_prob_10d).

    文件名含交付日+生成戳 (r/r2/r3 修订版), 按文件名排序拼接后 (date,symbol)
    keep-last = 每个交付日的最新一代. 清单是 top10 板内子集 (非全池), ECE 度量
    的是"交付给用户的概率数字"的校准.
    """
    cols = ["date", "board", "symbol", "pred_prob_10d"]
    parts = []
    for f in sorted(glob.glob(str(STOCK_LIST_DIR / "parallel_shortlist_*.csv"))):
        df = pd.read_csv(f, dtype={"symbol": str})
        if not set(cols).issubset(df.columns):
            continue
        parts.append(df[cols])
    parts = [p for p in parts if len(p)]  # 0 行文件 (历史空转) 避开 concat FutureWarning
    if not parts:
        return pd.DataFrame(columns=cols)
    df = pd.concat(parts, ignore_index=True)
    df["symbol"] = df["symbol"].str.zfill(6)
    return df.drop_duplicates(subset=["date", "symbol"], keep="last")


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
        print(
            "[warn] 无 last_*_days_picks_dual.csv (并行短名单未落盘), 退出", flush=True
        )
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

    cal_cfg = cfg.get("calibration", {})
    cal, cal_alerts, cal_panel_last = [], [], None
    if cal_cfg.get("enable"):
        prob = _load_shortlist_probs()
        prob["date"] = pd.to_datetime(prob["date"])
        print(
            f"\n[calib-preds] 短名单概率 {len(prob):,} 票 / "
            f"{prob['date'].nunique() if len(prob) else 0} 日 (top10 子集)",
            flush=True,
        )
        if len(prob):
            lo = prob["date"].min() - pd.Timedelta(days=15)
            cal_panel = pd.read_parquet(
                str(PANEL_V3_PATH),
                columns=["symbol", "date", "high_hfq", "close_hfq"],
                filters=[("date", ">=", lo)],
            )
            cal_panel["symbol"] = cal_panel["symbol"].astype(str)
            cal_panel_last = str(pd.Timestamp(cal_panel["date"].max()).date())
            realized_mfe = compute_realized_mfe(
                cal_panel,
                prob["date"],
                horizon=int(cal_cfg.get("horizon", 10)),
                cost=float(cal_cfg.get("cost", 0.0030)),
            )
            print(f"[calib-realized] 成熟 {len(realized_mfe):,} 票", flush=True)
            cal = rolling_calibration_mfe(
                prob,
                realized_mfe,
                thr=float(cal_cfg.get("mfe_threshold", 0.06)),
                n_bins=int(cal_cfg.get("n_bins", 5)),
                window_days=window_days,
                min_matured_days=min_matured,
            )
            cal_alerts = check_calibration(cal, cal_cfg.get("ece_threshold", {}))

            print(
                f"===== pred_prob_10d 校准 (滚动窗 {window_days} 交易日, "
                f"事件 = net MFE(盘中) > {float(cal_cfg.get('mfe_threshold', 0.06)):.0%}) =====",
                flush=True,
            )
            for _, r in cal.iterrows():
                th = cal_cfg.get("ece_threshold", {}).get(r["board"])
                if r["ece"] is None:
                    line = f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日 (积累期, 无 ECE)"
                else:
                    line = (
                        f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日 / "
                        f"{r['n_rows']:>6,} 票 | 滚动 ECE {r['ece']:+.2%}"
                    )
                    if th is not None:
                        line += f" vs 阈值 {th:+.2%}"
                    if any(a["board"] == r["board"] for a in cal_alerts):
                        line += "  [CALIB-ALERT] 概率校准漂移"
                print(line, flush=True)
                for b in r["bins"]:
                    print(
                        f"      {b['bin']:>14}: n={b['n']:>4}  pred {b['mean_prob']:.2f} "
                        f"real {b['realized']:.2f} gap {b['gap']:+.2f}",
                        flush=True,
                    )
            if not cal_alerts:
                print("[ok] 无板块校准 ECE 超阈值", flush=True)
        else:
            print("[warn] 无 parallel_shortlist_*.csv (短名单未落盘), 跳过校准", flush=True)

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
        "calibration": {
            "panel_last": cal_panel_last,
            "rolling": cal.to_dict("records"),
            "alerts": cal_alerts,
        },
    }
    (out_dir / f"parallel_drift_{ts}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/parallel_drift_{ts}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
