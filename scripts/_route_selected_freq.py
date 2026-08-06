# -*- coding: utf-8 -*-
"""_route_selected_freq.py — 把生产精选特征路由到 {月/周/日} 三张表.

读 factor_registry 里最新 selected_{board}_*.json (真实训练产物), 按 freq_of
(已测 FREQ_ASSIGNMENT → 同族类比 FAMILY_ANALOG → 事件 → 未分类) 路由, 产出:
  - factor_registry/selected_{board}_{freq}_{ts}.json 三张频率表 (WORM)
  - factor_registry/selected_{board}_freq_{ts}.json   覆盖率报告 (含未分类明细)
  - data/_route_selected_freq_{board}_{ts}.log         摘要 (WORM)
核心铁律: 月频特征不进日频表. 未分类特征显式暴露, 不静默默认.

用法: python scripts/_route_selected_freq.py [board] [version_id]
  board: main | dual | all (默认 all)
  version_id: 指定 selected_{board}_{version_id}.json (默认最新)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from app.pipeline1.feature_selector import FREQ_ORDER, freq_of

from config.settings import data_others_path

REGISTRY = str(data_others_path("data/factor_registry"))


def _latest(board: str) -> str:
    files = sorted(
        f
        for f in os.listdir(REGISTRY)
        if f.startswith(f"selected_{board}_")
        and f.endswith(".json")
        and "_current" not in f
        and "_freq_" not in f
        and not any(f"_{frq}_" in f for frq in FREQ_ORDER)
    )
    if not files:
        raise FileNotFoundError(f"no selected_{board}_*.json in {REGISTRY}")
    return files[-1]


def route_board(board: str, version_id: str | None) -> list[str]:
    fname = (
        _latest(board) if version_id is None else f"selected_{board}_{version_id}.json"
    )
    with open(os.path.join(REGISTRY, fname), encoding="utf-8") as fh:
        obj = json.load(fh)
    feats = obj.get("features", [])
    buckets = {freq: [] for freq in FREQ_ORDER}
    buckets["事件"] = []
    buckets["未分类"] = []
    for f in feats:
        buckets[freq_of(f)].append(f)

    ts = obj.get("created", "").replace(":", "").replace("T", "_") or __import__(
        "datetime"
    ).datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [f"=== 三频路由 [{board}] source={fname} n={len(feats)} ts={ts} ==="]
    lines.append("覆盖率: " + "  ".join(f"{k}={len(v):,}" for k, v in buckets.items()))
    lines.append("")
    for freq in FREQ_ORDER:
        lines.append(f"--- {freq}频表 ({len(buckets[freq]):,}) ---")
        lines.append("  " + ", ".join(buckets[freq][:20]))
        lines.append("")
    lines.append(f"--- 事件桶 ({len(buckets['事件']):,}) ---")
    lines.append("  " + ", ".join(buckets["事件"][:12]))
    lines.append("")
    lines.append(f"--- 未分类 ({len(buckets['未分类']):,}) — 需扩映射/判定 ---")
    for f in buckets["未分类"]:
        lines.append("  ? " + f)

    for freq in FREQ_ORDER:
        with open(
            os.path.join(REGISTRY, f"selected_{board}_{freq}_{ts}.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(
                {
                    "board": board,
                    "freq": freq,
                    "source": fname,
                    "selected_count": len(buckets[freq]),
                    "features": buckets[freq],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
    with open(
        os.path.join(REGISTRY, f"selected_{board}_freq_{ts}.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "board": board,
                "ts": ts,
                "source": fname,
                "coverage": {k: len(v) for k, v in buckets.items()},
                "unknown": buckets["未分类"],
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    text = "\n".join(lines)
    print(text, flush=True)
    log = os.path.join("data", f"_route_selected_freq_{board}_{ts}.log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {log}", flush=True)
    return lines


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    board = sys.argv[1] if len(sys.argv) > 1 else "all"
    version_id = sys.argv[2] if len(sys.argv) > 2 else None
    boards = ["main", "dual"] if board == "all" else [board]
    for b in boards:
        try:
            route_board(b, version_id)
        except FileNotFoundError as exc:
            print(f"[{b}] {exc}", flush=True)


if __name__ == "__main__":
    main()
