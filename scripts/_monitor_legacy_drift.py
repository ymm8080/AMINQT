"""_monitor_legacy_drift.py — legacy 幅度/校准/赢家判别漂移监控 (2026-08-17 / 08-24 加 ECE / 09-03 加赢家AUC).

用户定案 (08-17): 幅度模型漂移 (pred_ret_10d 系统高估, 偏差随时间扩大) 修法 = 重训,
监控 = 每日全池预测均值 vs T+10 净实现均值偏差, 滚动窗超阈值 → 告警 (提醒提前重训).
08-24: 加 p_reg 校准检查 — prob_up_10d 建模事件 = gross_10d > 0.5% (label_engine
CLS_THRESHOLD), ECE 事件 = gross_cc > 0.5% ⟺ realized_net > 0.5% − cost (与幅度同
realized, 防 cost 偏差假触发); 全池分位桶滚动窗 ECE 超阈值 → 校准漂移告警.
09-03: 加排名键赢家判别 AUC 月度节 (winner-leak 复盘: 6 月断裂 0.66→0.52 抛硬币
且未恢复; 连续 2 个成熟月 <0.55 → 报警). candidates 无 base_rate, 用全池口径,
绝对值与复盘 E7 回放不可比, 只看趋势; 配置见 DRIFT_MONITOR["winner_auc"].

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
    bin_calibration,
    board_of,
    check_calibration,
    check_drift,
    check_winner_auc,
    compute_realized,
    daily_bias,
    daily_winner_auc,
    monthly_winner_auc,
    rolling_bias,
    rolling_calibration,
)
from config.settings import DRIFT_MONITOR, PANEL_V3_PATH, data_others_path

LISTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lists"
)


def _load_preds() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(LISTS_DIR, "candidates_*.parquet")))
    if not files:
        return pd.DataFrame(
            columns=["date", "symbol", "board", "pred_ret_10d", "prob_up_10d"]
        )
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if "pred_ret_10d" not in df.columns:  # 08-06 早期文件无 10d 模型列
            continue
        d = pd.Timestamp(os.path.basename(f)[11:19])
        df["prob_up_10d"] = df["prob_up_10d"] if "prob_up_10d" in df.columns else np.nan
        df = df[["symbol", "board", "pred_ret_10d", "prob_up_10d"]]
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
        g = (
            g.assign(bias=g["pred_ret_10d"] - g["realized_net"])
            .groupby("date")["bias"]
            .mean()
        )
        tail = g.tail(window_days)
        if len(tail):
            out[board] = {
                "window_days": len(tail),
                "bias": float(tail.mean()),
                "source": os.path.basename(files[-1]),
            }
    return out


def _replay_calibration_reference(thr: float, n_bins: int, cost: float) -> dict:
    """诊断回放 CSV 的校准参照 (当前 bundle 基线, 仅参照非告警).

    事件口径与 live 一致: gross_cc = realized_net + cost > thr. 回放明细只含
    通过旧生产闸的 picks (子集), 与 live 全池绝对 ECE 不可直接比 → 只作方向参照.
    """
    files = sorted(
        glob.glob(str(data_others_path("diag") / "legacy_prob_head_replay_*.csv"))
    )
    if not files:
        return {}
    r = pd.read_csv(files[-1], dtype={"symbol": str})
    out = {}
    for board in ("main", "dual"):
        g = r[r["board"] == board].dropna(subset=["prob", "realized_net"])
        if not len(g):
            continue
        tab, ece = bin_calibration(
            g["prob"], (g["realized_net"] + cost > thr).astype("int8"), n_bins
        )
        if tab.empty:
            continue
        out[board] = {
            "ece": ece,
            "n_rows": int(len(g)),
            "source": os.path.basename(files[-1]),
            "bins": (
                tab.reset_index()[["bin", "n", "mean_prob", "realized", "gap"]]
                .assign(bin=lambda t: t["bin"].astype(str))
                .to_dict("records")
            ),
        }
    return out


def main() -> int:
    cfg = DRIFT_MONITOR
    window_days = int(cfg["window_days"])
    min_matured = int(cfg["min_matured_days"])
    thresholds = cfg["bias_threshold"]

    preds = _load_preds()
    print(
        f"[preds] {len(preds):,} 票 / {preds['date'].nunique() if len(preds) else 0} 日",
        flush=True,
    )
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

    cal_cfg = cfg.get("calibration", {})
    cal_thr = float(cal_cfg.get("cls_threshold", 0.005))
    cal_bins = int(cal_cfg.get("n_bins", 5))
    cal = rolling_calibration(
        preds,
        realized,
        cost=float(cfg["cost"]),
        thr=cal_thr,
        n_bins=cal_bins,
        window_days=window_days,
        min_matured_days=min_matured,
    )
    cal_alerts = check_calibration(cal, cal_cfg.get("ece_threshold", {}))
    cal_ref = _replay_calibration_reference(cal_thr, cal_bins, float(cfg["cost"]))

    wauc_cfg = cfg.get("winner_auc", {})
    wauc = daily_winner_auc(preds, realized, win_t=float(wauc_cfg.get("win_t", 0.05)))
    wauc_monthly = monthly_winner_auc(wauc, min_days=int(wauc_cfg.get("min_days", 8)))
    wauc_alerts = check_winner_auc(
        wauc_monthly,
        threshold=float(wauc_cfg.get("threshold", 0.55)),
        consecutive_months=int(wauc_cfg.get("consecutive_months", 2)),
    )

    print(
        f"===== legacy 幅度漂移 (滚动窗 {window_days} 交易日, 成熟≥{min_matured} 日) =====",
        flush=True,
    )
    for _, r in rolling.iterrows():
        th = thresholds.get(r["board"])
        line = f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日"
        if r["latest_bias"] is not None:
            line += (
                f" | 最新日 {pd.Timestamp(r['latest_date']).date()} "
                f"偏差 {r['latest_bias']:+.2%}"
            )
        line += (
            f" | 滚动偏差 {r['bias']:+.2%}" if r["bias"] is not None else " | 积累期"
        )
        if th is not None:
            line += f" vs 阈值 {th:+.2%}"
        if any(a["board"] == r["board"] for a in alerts):
            line += "  [DRIFT-ALERT] 考虑提前重训"
        print(line, flush=True)
    if not len(rolling):
        print(
            "[info] 无成熟日 — 积累期 (首个成熟日约在首个候选日后 11 交易日)",
            flush=True,
        )
    for board in ("main", "dual"):
        rr = ref.get(board)
        if rr:
            base = f"     参照[{board}](回放 {rr['source']} 尾 {rr['window_days']} 日): {rr['bias']:+.2%}"
            th = thresholds.get(board)
            if th is not None:
                base += f" — 当前 bundle 诊断基线, 超阈值 {th:+.2%} 则强烈提示重训"
            print(base, flush=True)

    print(
        f"\n===== p_reg 校准 (滚动窗 {window_days} 交易日, 事件 = gross > {cal_thr:.1%}) =====",
        flush=True,
    )
    for _, r in cal.iterrows():
        th = cal_cfg.get("ece_threshold", {}).get(r["board"])
        line = (
            f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日 / {r['n_rows']:>6,} 票 | "
            f"滚动 ECE {r['ece']:+.2%}"
            if r["ece"] is not None
            else f"[{r['board']:>4}] 成熟 {r['n_days']:>3} 日 (积累期, 无 ECE)"
        )
        if th is not None and r["ece"] is not None:
            line += f" vs 阈值 {th:+.2%}"
        if any(a["board"] == r["board"] for a in cal_alerts):
            line += "  [CALIB-ALERT] 概率校准漂移, 考虑提前重训"
        print(line, flush=True)
        if r["ece"] is not None:
            for b in r["bins"]:
                print(
                    f"      {b['bin']:>14}: n={b['n']:>4}  pred {b['mean_prob']:.2f} "
                    f"real {b['realized']:.2f} gap {b['gap']:+.2f}",
                    flush=True,
                )
    for board in ("main", "dual"):
        cr = cal_ref.get(board)
        if cr:
            print(
                f"     参照[{board}](回放 {cr['source']}): ECE {cr['ece']:+.2%} "
                f"({cr['n_rows']:,} 票, 仅 picks 子集)",
                flush=True,
            )

    print(
        f"\n===== 排名键赢家 AUC (月度, 全池, 赢家 = T+10 净 ≥ {wauc_cfg.get('win_t', 0.05):.0%}) =====",
        flush=True,
    )
    if wauc_monthly.empty:
        print("[info] 无成熟 AUC 日 — 积累期", flush=True)
    for _, r in wauc_monthly.iterrows():
        line = f"[{r['board']:>4}] {r['month']}  成熟 {r['n_auc_days']:>2}/{r['n_days']:>2} 日"
        if r["auc"] is not None:
            line += f" | AUC {r['auc']:.3f}"
            if any(
                a["board"] == r["board"] and r["month"] in a["months"]
                for a in wauc_alerts
            ):
                line += f"  [AUC-ALERT] 判别衰减 (连续{wauc_cfg.get('consecutive_months', 2)}月 <{wauc_cfg.get('threshold', 0.55):.2f})"
        else:
            line += " | 无可判别日"
        print(line, flush=True)
    if not wauc_alerts:
        print(
            f"[ok] 无板块连续{wauc_cfg.get('consecutive_months', 2)}月 AUC <{wauc_cfg.get('threshold', 0.55):.2f}",
            flush=True,
        )
    if not cal_alerts:
        print("[ok] 无板块校准 ECE 超阈值", flush=True)
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
        "calibration": {
            "rolling": cal.to_dict("records"),
            "alerts": cal_alerts,
            "replay_reference": cal_ref,
        },
        "winner_auc": {
            "monthly": wauc_monthly.to_dict("records"),
            "alerts": wauc_alerts,
        },
    }
    (out_dir / f"legacy_drift_{ts}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/legacy_drift_{ts}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
