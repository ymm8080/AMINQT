"""_monitor_legacy_drift.py — legacy 幅度漂移监控 (2026-08-17).

用户定案 (08-17): 幅度模型漂移 (pred_ret_10d 系统高估, 偏差随时间扩大) 修法 = 重训,
监控 = 每日全池预测均值 vs T+10 净实现均值偏差, 滚动窗超阈值 → 告警 (提醒提前重训).

数据源 (全部 WORM, 无新落盘):
  preds   = data/lists/candidates_<YYYYMMDD>.parquet (每日全池原始预测, daily_pipeline)
  realized= V3 面板 close_hfq 现算 (口径与诊断回放逐字一致: buy=决策日后第 buy_lag
            交易日收盘, sell=第 sell_lag 交易日收盘, ps/pb-1-COST, 停牌 ffill)
参照线 = 最近一次 legacy_prob_head_replay WORM CSV 尾窗偏差 (当前 bundle 的诊断基线,
        仅参照非告警; 模型换代后该参照过时, 由报告内 bundle 标签人工识别).

输出: 逐板块成熟日数/最新偏差/滚动偏差 vs 阈值 + 告警; WORM 报告
  DATA OTHERS/diag/legacy_drift_<ts>.json (含逐日明细). 退出码恒 0 (告警不进自动化失败).
用法: python scripts/_monitor_legacy_drift.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.drift_monitor import (
    board_of,
    check_drift,
    compute_realized,
    daily_bias,
    rolling_bias,
)
from config.settings import DRIFT_MONITOR, PANEL_V3_PATH, data_others_path

LISTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lists")


def _load_preds() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(LISTS_DIR, "candidates_*.parquet")))
    if not files:
        return pd.DataFrame(columns=["date", "symbol", "board", "pred_ret_10d"])
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if "pred_ret_10d" not in df.columns:  # 08-06 早期文件无 10d 模型列
            continue
        d = pd.Timestamp(os.path.basename(f)[11:19])
        df = df[["symbol", "board", "pred_ret_10d"]]
        df["symbol"] = df["symbol"].astype(str)
        df["board"] = df["board"].map(board_of)
        df["date"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _replay_reference(window_days: int) -> dict:
    """诊断回放 CSV 尾窗偏差参照 (当前 bundle 基线, 仅参照)."""
    files = sorted(
        glob.glob(str(data_others_path("diag") / "legacy_prob_head_replay_*.csv"))
    )
    if not files:
        return {}
    r = pd.read_csv(files[-1], dtype={"symbol": str})
    r["date"] = pd.to_datetime(r["date"])
    out = {}
    for board in ("main", "dual"):
        g = r[r["board"] == board].dropna(subset=["pred_ret_10d", "realized_net"])
        if not len(g):
            continue
        g = g.assign(bias=g["pred_ret_10d"] - g["realized_net"]).groupby("date")["bias"].mean()
        tail = g.tail(window_days)
        if len(tail):
            out[board] = {
                "window_days": len(tail),
                "bias": float(tail.mean()),
                "source": os.path.basename(files[-1]),
            }
    return out


def main() -> int:
    cfg = DRIFT_MONITOR
    window_days = int(cfg["window_days"])
    min_matured = int(cfg["min_matured_days"])
    thresholds = cfg["bias_threshold"]

    preds = _load_preds()
    print(f"[preds] {len(preds):,} 票 / {preds['date'].nunique() if len(preds) else 0} 日", flush=True)
    if not len(preds):
        print("[warn] 无 candidates_*.parquet (每日全池预测未落盘), 退出", flush=True)
        return 0

    last_d = preds["date"].max()
    bound = np.datetime64(last_d) + np.timedelta64(15, "D")  # 11 交易日 ≈ ≤15 自然日
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq"],
        filters=[("date", "<=", bound)],
    )
    panel["symbol"] = panel["symbol"].astype(str)
    print(
        f"[panel] {len(panel):,} 行 (date ≤ {pd.Timestamp(bound).date()}), "
        f"最新 {pd.Timestamp(panel['date'].max()).date()}",
        flush=True,
    )

    realized = compute_realized(
        panel,
        preds["date"],
        buy_lag=int(cfg["buy_lag"]),
        sell_lag=int(cfg["sell_lag"]),
        cost=float(cfg["cost"]),
    )
    print(f"[realized] 成熟 {len(realized):,} 票", flush=True)

    daily = daily_bias(preds, realized)
    rolling = rolling_bias(daily, window_days, min_matured)
    alerts = check_drift(rolling, thresholds)
    ref = _replay_reference(window_days)

    print(f"===== legacy 幅度漂移 (滚动窗 {window_days} 交易日, 成熟≥{min_matured} 日) =====", flush=True)
    for _, r in rolling.iterrows():
        th = thresholds.get(r["board"])
        line = (
            f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日 | 最新日 "
            f"{pd.Timestamp(r['latest_date']).date()} 偏差 {r['latest_bias']:+.2%} | "
            f"滚动偏差 {r['bias']:+.2%}"
        )
        if th is not None:
            line += f" vs 阈值 {th:+.2%}"
        if any(a["board"] == r["board"] for a in alerts):
            line += "  [DRIFT-ALERT] 考虑提前重训"
        print(line, flush=True)
    if not len(rolling):
        print("[info] 无成熟日 — 积累期 (首个成熟日约在首个候选日后 11 交易日)", flush=True)
    for board in ("main", "dual"):
        rr = ref.get(board)
        if rr:
            base = f"     参照[{board}](回放 {rr['source']} 尾 {rr['window_days']} 日): {rr['bias']:+.2%}"
            th = thresholds.get(board)
            if th is not None:
                base += f" — 当前 bundle 诊断基线, 超阈值 {th:+.2%} 则强烈提示重训"
            print(base, flush=True)
    if not alerts:
        print("[ok] 无板块滚动偏差超阈值", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    report = {
        "ts": ts,
        "config": cfg,
        "n_pred_rows": int(len(preds)),
        "pred_dates": [str(d) for d in sorted(preds["date"].unique())],
        "panel_last": str(pd.Timestamp(panel["date"].max()).date()),
        "daily": daily.to_dict("records"),
        "rolling": rolling.to_dict("records"),
        "alerts": alerts,
        "replay_reference": ref,
    }
    (out_dir / f"legacy_drift_{ts}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/legacy_drift_{ts}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
