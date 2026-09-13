"""PARALLEL 排名键自适应 (0912 夜用户令) — mag/prob/blend 三键评估与解析.

与 LEGACY 动态头选择 (LEGACY_PROB_SOURCE="auto" 重训 argmax) 对齐的 parallel 版:
每次概率头重训后评估三键 — mag (幅度头≈LEGACY reg) / prob (概率头≈LEGACY cls) /
blend (mag×prob, 2026-08-15 A/B 定案现状键) — argmax 加权 IC 自选 (平局→blend);
serving (scripts/_shortlist_t5_t10.rank_and_truncate) 按最新 WORM json 换板级
排序键。fail-open 铁则: json 缺失/过旧/内容异常 → "blend" (不杀清单)。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = ("3d", "5d", "10d")
# 选择聚合权重: 多视界加权 (用户令 3d/5d/10d 都要看); 10d = 服务排名键口径权重最高
HORIZON_WEIGHTS = {"3d": 0.25, "5d": 0.35, "10d": 0.40}
RANK_KEYS = ("mag", "prob", "blend")
DEFAULT_KEY = "blend"


def daily_rank_metrics(df: pd.DataFrame, key: str, label: str, top_n: int = 10) -> dict:
    """逐日 (决策日截面) Spearman IC + TOP-n 实得均值, 再对日聚合.

    df 需含 date/key/label 列; NaN 行剔除 (标签未实现的尾段日期自然出局)。
    逐日分组 (≤百组) 非逐股循环。
    """
    sub = df[["date", key, label]].dropna()
    if sub.empty or sub[key].nunique() < 2 or sub[label].nunique() < 2:
        return {"ic": None, "top10_net": None, "n_days": 0}

    def _spear(g: pd.DataFrame):
        if len(g) < 3:
            return np.nan
        return float(g[key].rank().corr(g[label].rank()))

    ic_series = sub.groupby("date")[[key, label]].apply(_spear).dropna()
    top = sub.sort_values(key, ascending=False).groupby("date", sort=False).head(top_n)
    top10_net = top.groupby("date")[label].mean()
    n_days = int(len(ic_series))
    return {
        "ic": float(ic_series.mean()) if n_days else None,
        "top10_net": float(top10_net.mean()) if len(top10_net) else None,
        "n_days": n_days,
    }


def weighted_ic_from_horizons(per_h: dict) -> float | None:
    """逐视界 IC 按 HORIZON_WEIGHTS 加权; 缺失视界剔除后权重归一; 全缺 → None."""
    pairs = [
        (HORIZON_WEIGHTS[h], v)
        for h in HORIZONS
        for v in [per_h.get(h)]
        if v is not None and np.isfinite(v)
    ]
    if not pairs:
        return None
    wsum = sum(w for w, _ in pairs)
    return float(sum(w * v for w, v in pairs) / wsum)


def evaluate_keys(frame: pd.DataFrame, eval_days: int = 60, top_n: int = 10) -> dict:
    """三键 × 三视界评估 → {key: {"per_horizon": {h: 指标}, "weighted_ic": x}}.

    frame 列: symbol / date / mag / prob / label_pm_{3,5,10}d_net。
    每视界取 trailing eval_days 个标签已实现决策日 (无前瞻); blend = mag×prob。
    """
    f = frame.copy()
    f["blend"] = f["mag"] * f["prob"]
    out: dict = {}
    for key in RANK_KEYS:
        per: dict = {}
        for h in HORIZONS:
            lab = f"label_pm_{h}_net"
            if lab not in f.columns:
                per[h] = {"ic": None, "top10_net": None, "n_days": 0}
                continue
            sub = f[["date", "symbol", key, lab]].dropna()
            if sub.empty:
                per[h] = {"ic": None, "top10_net": None, "n_days": 0}
                continue
            keep = sorted(sub["date"].unique())[-int(eval_days) :]
            per[h] = daily_rank_metrics(sub[sub["date"].isin(keep)], key, lab, top_n)
        out[key] = {
            "per_horizon": per,
            "weighted_ic": weighted_ic_from_horizons(
                {h: m["ic"] for h, m in per.items()}
            ),
        }
    return out


def choose_rank_key(evaluation: dict) -> str:
    """argmax weighted_ic; 平局 (含 blend 与他键并列) / 全缺 → blend (默认键)."""
    valid = {
        k: v["weighted_ic"]
        for k, v in evaluation.items()
        if isinstance(v, dict)
        and v.get("weighted_ic") is not None
        and np.isfinite(v["weighted_ic"])
    }
    if not valid:
        return DEFAULT_KEY
    best = max(valid.values())
    winners = [k for k, v in valid.items() if v >= best]
    return winners[0] if len(winners) == 1 else DEFAULT_KEY


def rank_source_dir(directory=None) -> Path:
    if directory is not None:
        return Path(directory)
    from app.pipeline_parallel.prob_head import bundle_dir

    return bundle_dir()


def save_rank_source(board: str, payload: dict, directory=None, ts=None) -> Path:
    """WORM json 落盘 data/prob_head/{board}_rank_source_{ts}.json; 同 ts 不覆盖."""
    d = rank_source_dir(directory)
    d.mkdir(parents=True, exist_ok=True)
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    path = d / f"{board}_rank_source_{ts}.json"
    n = 1
    while path.exists():
        path = d / f"{board}_rank_source_{ts}_{n}.json"
        n += 1
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_latest_rank_source(board: str, directory=None) -> dict | None:
    """最新一份评估 json; 无 / 损坏 → None (不 raise, fail-open 由调用方处理)."""
    cands = sorted(rank_source_dir(directory).glob(f"{board}_rank_source_*.json"))
    if not cands:
        return None
    try:
        rec = json.loads(cands[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return rec if isinstance(rec, dict) else None


def resolve_rank_key(
    board: str,
    as_of=None,
    knob=None,
    directory=None,
    max_stale_days=None,
) -> str:
    """解析该板 serving 排名键: 显式旋钮 > auto(最新 json, 新鲜度闸内) > blend.

    knob: None → PARALLEL_RANK_SOURCE[board] (默认 "auto"); "mag"/"prob"/"blend"
    显式强制 (回退旋钮)。auto 路径任何缺失/过旧/异常 → blend (fail-open, 大声打印)。
    """
    from app.pipeline_parallel.config import (
        PARALLEL_RANK_SOURCE,
        RANK_SOURCE_MAX_STALE_DAYS,
    )

    chosen_knob = PARALLEL_RANK_SOURCE.get(board, "auto") if knob is None else knob
    if chosen_knob in RANK_KEYS:
        return chosen_knob
    try:
        rec = load_latest_rank_source(board, directory)
        if rec is None:
            return DEFAULT_KEY
        chosen = rec.get("chosen")
        if chosen not in RANK_KEYS:
            return DEFAULT_KEY
        trained_through = rec.get("trained_through")
        if trained_through is None:
            return DEFAULT_KEY
        stale = (
            RANK_SOURCE_MAX_STALE_DAYS
            if max_stale_days is None
            else int(max_stale_days)
        )
        as_of_ts = (
            pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
        )
        gap = abs((as_of_ts - pd.Timestamp(trained_through)).days)
        if gap > stale:
            print(
                f"[rank_source] {board} 评估 json 过旧 (trained_through="
                f"{trained_through}, 距今 {gap} 日 > {stale}) → 回退 {DEFAULT_KEY}",
                flush=True,
            )
            return DEFAULT_KEY
        return chosen
    except Exception as exc:
        print(
            f"[rank_source] {board} 解析异常 ({exc}) → 回退 {DEFAULT_KEY}",
            flush=True,
        )
        return DEFAULT_KEY
