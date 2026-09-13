"""LEGACY 超参 json 旋钮源 (0913 #8 自适应再扫) — WORM + 新鲜度闸 + fail-open.

机制 = app.pipeline_parallel.rank_source 同款三件套 (机器周末自动换 incumbent
须免 commit):
  - WORM: data/others/param_source/{board}_param_source_{ts}.json 只增不覆盖
    (同 ts 重存加序号); 删最新 json 即回上一版 (回退第二层).
  - 新鲜度: trained_through 距今 > LEGACY_PARAM_MAX_STALE_DAYS → 弃用 json.
  - fail-open: 无 json / 过旧 / 内容异常 → {} (代码表 NUM_LEAVES_OVERRIDE /
    PARAMS_OVERRIDE 兜底, 0808/0817 扫描定案; 回退第三层).
显式旋钮 LEGACY_PARAM_SOURCE[board]="code" 压过 json (回退第一层).

payload: {
  "board", "trained_through" ("YYYY-MM-DD"),
  "overrides": {kind: {LGBM 参数增量}},   # 只叠 model_params 产出之上
  "evidence": {"sweep": ..., "judge": "TOP10 实净", ...},  # 判词证据链
  "ts": "..."
}
kind 命名同 model_params: "3d_cls".."10d_cls"/"3d_reg".."10d_reg"/"pain".
半衰期等非 LGBM 覆盖键后续版本再加 (须有已过闸扫描证据才允许生效).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from config.settings import data_others_path


def param_source_dir(directory=None) -> Path:
    if directory is not None:
        return Path(directory)
    return Path(data_others_path("param_source"))


def save_param_source(board: str, payload: dict, directory=None, ts=None) -> Path:
    """WORM json 落盘; 同 ts 不覆盖 (加序号)."""
    d = param_source_dir(directory)
    d.mkdir(parents=True, exist_ok=True)
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    path = d / f"{board}_param_source_{ts}.json"
    n = 1
    while path.exists():
        path = d / f"{board}_param_source_{ts}_{n}.json"
        n += 1
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_latest_param_source(board: str, directory=None) -> dict | None:
    """最新一份超参 json; 无 / 损坏 → None (不 raise)."""
    cands = sorted(param_source_dir(directory).glob(f"{board}_param_source_*.json"))
    if not cands:
        return None
    try:
        rec = json.loads(cands[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return rec if isinstance(rec, dict) else None


def resolve_param_override(
    board: str,
    kind: str,
    as_of=None,
    knob=None,
    directory=None,
    max_stale_days=None,
) -> dict:
    """解析该 (board, kind) 的 LGBM 参数增量覆盖 → dict (空 = 代码表现状).

    显式旋钮 "code" > 最新 json (新鲜度闸内 overrides[kind]) > {} 代码表。
    json 内该 kind 无条目 / 条目非 dict / 空 dict → {} (不动该头)。
    """
    from config.settings import LEGACY_PARAM_MAX_STALE_DAYS, LEGACY_PARAM_SOURCE

    chosen = LEGACY_PARAM_SOURCE.get(board, "auto") if knob is None else knob
    if chosen != "auto":
        return {}
    try:
        rec = load_latest_param_source(board, directory)
        if rec is None:
            return {}
        trained_through = rec.get("trained_through")
        if not trained_through:
            print(
                f"[param_source] {board} json 缺 trained_through → 代码表",
                flush=True,
            )
            return {}
        stale = (
            LEGACY_PARAM_MAX_STALE_DAYS if max_stale_days is None else int(max_stale_days)
        )
        as_of_ts = (
            pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
        )
        if abs((as_of_ts - pd.Timestamp(trained_through)).days) > stale:
            print(
                f"[param_source] {board} 超参 json 过旧 (trained_through="
                f"{trained_through}, 距今 >{stale} 日) → 代码表",
                flush=True,
            )
            return {}
        ov = rec.get("overrides")
        if not isinstance(ov, dict):
            print(f"[param_source] {board} json overrides 非法 → 代码表", flush=True)
            return {}
        got = ov.get(kind)
        return dict(got) if isinstance(got, dict) and got else {}
    except Exception as exc:
        print(f"[param_source] {board} 解析异常 ({exc}) → 代码表", flush=True)
        return {}
