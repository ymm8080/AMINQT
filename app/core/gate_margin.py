"""概率闸自适应 margin 引擎 — legacy/PARALLEL 两线共享 (2026-09-08).

抽自 app/pipeline1/prob_head.py (2026-08-31 legacy 首部署, 生产验证一个多月);
PARALLEL PROB_GATE 移植 (用户令硬指标全自适应化) 复用同一引擎, 各传自己的 cfg:

margin_t = clip(Q_q(近 spread_lookback_days 日参与闸逐股 spread=pred_prob-base_rate), min, max)
- 无历史 → 当日截面 bootstrap (top-q% 语义, 无未来数据)
- 连续 3 决策日 (n_total>0) 100% 剔除 → 熔断放开至 margin_min + 大声告警
- 无历史且无当日截面 → 回退静态 cfg["margin"] (fail-open 保守)
- margin_mode="fixed" → 静态档 (一键回退)
no-lookahead: 池化历史严格 < today (文件名日期比较).

WORM 状态 (调用方传入 cfg["gate_margin_dir"], 两线目录隔离, spread 分布不互池):
  spreads_{board}_{date}.csv   当日逐股 symbol/pred_prob/spread (当日重跑覆盖)
  margin_{board}_{date}.json   当日 margin 决策 (mode/margin/thr/base_rate/n_total/n_kept)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def gate_margin_dir(cfg: dict) -> Path:
    return Path(cfg["gate_margin_dir"])


def history_dates(board: str, today: str, limit: int, cfg: dict) -> list[str]:
    """spread 历史日期 (严格 < today, 降序, 最多 limit 个) — no-lookahead 边界."""
    d = gate_margin_dir(cfg)
    dates = []
    for fp in d.glob(f"spreads_{board}_*.csv"):
        date = fp.stem.removeprefix(f"spreads_{board}_")
        if date.isdigit() and date < today:
            dates.append(date)
    return sorted(dates, reverse=True)[:limit]


def breaker_tripped(board: str, today: str, cfg: dict) -> bool:
    """连续 3 个决策日 (n_total>0) 全部 0 保留 → True (闸退化成板禁时强制放开)."""
    d = gate_margin_dir(cfg)
    dates = []
    for fp in d.glob(f"margin_{board}_*.json"):
        date = fp.stem.removeprefix(f"margin_{board}_")
        if date.isdigit() and date < today:
            dates.append(date)
    n_checked = 0
    for date in sorted(dates, reverse=True):
        try:
            with open(d / f"margin_{board}_{date}.json", encoding="utf-8") as fh:
                dec = json.load(fh)
        except Exception:
            continue
        if int(dec.get("n_total", 0)) <= 0:
            continue  # 无参与者的日子不算决策日
        n_checked += 1
        if int(dec.get("n_kept", 0)) > 0:
            return False
        if n_checked >= 3:
            return True
    return False


def compute_adaptive_margin(
    board: str, today_spreads: pd.Series | None, today: str, cfg: dict
) -> tuple[float, str]:
    """自适应 margin: 滚动分位数 / 熔断 / bootstrap / 静态回退.

    Returns:
        (margin, mode): mode ∈ fixed | breaker | rolling_q | bootstrap | fixed_fallback
    """
    if cfg.get("margin_mode", "fixed") == "fixed":
        return float(cfg["margin"]), "fixed"
    q = float(cfg["margin_q"])
    lo = float(cfg["margin_min"])
    hi = float(cfg["margin_max"])
    if breaker_tripped(board, today, cfg):
        print(
            f"[prob_gate] {board} 熔断: 连续 3 决策日 100% 剔除 → margin 放开至地板 "
            f"{lo} (检查概率头/分布漂移!)",
            flush=True,
        )
        return lo, "breaker"
    vals: list[np.ndarray] = []
    for date in history_dates(board, today, int(cfg["spread_lookback_days"]), cfg):
        try:
            df = pd.read_csv(gate_margin_dir(cfg) / f"spreads_{board}_{date}.csv")
            v = pd.to_numeric(df["spread"], errors="coerce").dropna().to_numpy(float)
        except Exception:
            continue
        if len(v):
            vals.append(v[np.isfinite(v)])
    if vals:
        pooled = np.concatenate(vals)
        return float(np.clip(np.quantile(pooled, q), lo, hi)), "rolling_q"
    if today_spreads is not None:
        s = pd.Series(today_spreads).dropna()
        s = s[np.isfinite(s.astype(float))].astype(float)
        if len(s):
            return float(np.clip(np.quantile(s.to_numpy(), q), lo, hi)), "bootstrap"
    print(
        f"[prob_gate] {board} 无 spread 历史/当日截面 → 回退静态 margin "
        f"{cfg['margin']} (fail-open 保守)",
        flush=True,
    )
    return float(cfg["margin"]), "fixed_fallback"


def persist_gate_state(
    board: str,
    today: str,
    symbols: pd.Index,
    p: pd.Series,
    base: float,
    margin: float,
    mode: str,
    thr: float,
    keep: pd.Series,
    cfg: dict,
) -> None:
    """当日 spread 逐股 + margin 决策落盘 (WORM 逐日文件; 当日重跑覆盖当日文件)."""
    d = gate_margin_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    hist_dates = history_dates(
        board, today, int(cfg["spread_lookback_days"]), cfg
    )
    try:
        sp = pd.DataFrame(
            {
                "symbol": p.index.astype(str).tolist(),
                "pred_prob": p.to_numpy(float),
                "spread": (p - base).to_numpy(float),
            }
        )
        sp.to_csv(d / f"spreads_{board}_{today}.csv", index=False)
        n_total = int(len(symbols))
        n_kept = int(keep.sum())
        with open(d / f"margin_{board}_{today}.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "date": today,
                    "board": board,
                    "mode": mode,
                    "margin": margin,
                    "thr": thr,
                    "base_rate": base,
                    "q": cfg["margin_q"],
                    "n_total": n_total,
                    "n_kept": n_kept,
                    "n_history_days": len(hist_dates),
                },
                fh,
                ensure_ascii=False,
                indent=1,
            )
    except Exception as exc:  # 状态落盘失败不影响闸 (非阻塞)
        print(f"[prob_gate] {board} 状态落盘失败 (非阻塞): {exc}", flush=True)
