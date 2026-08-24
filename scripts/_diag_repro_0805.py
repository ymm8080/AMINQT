"""诊断: 复现 08-05 交付名单生成, 定位 301326 被剔除环节 (2026-08-07).

对指定 FULL RUN 目录运行 _shortlist_t5_t10 的主流程 (不重算特征, 只读 full run 产物),
打印: 制度门保留组合 / 301326 在 select_confident 前后的完整预测 / 被剔除原因.

用法: python scripts/_diag_repro_0805.py <fullrun_dir_name>
      例: python scripts/_diag_repro_0805.py 20260805_005343
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from config.settings import DATA_OTHERS_DIR

TARGET = sys.argv[1] if len(sys.argv) > 1 else "20260805_005343"
FULLRUN = DATA_OTHERS_DIR / "BACKTESTING RESULT" / TARGET
WATCH = "301326"

spec = importlib.util.spec_from_file_location(
    "shortlist_mod", Path(__file__).parent / "_shortlist_t5_t10.py"
)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

S.FULLRUN_DIR = FULLRUN
print(f"[fullrun] {FULLRUN}", flush=True)


def main() -> None:
    records = S.load_oos_records()
    gate = S.regime_gate(records)
    # 2026-08-23 制度门改板块级: gate key=(board,), 显示板块名
    active = [S.BOARD_LABEL.get(b, b) for (b,), g in gate.items() if g["active"]]
    print(f"[regime] 保留板块: {', '.join(active) if active else '无'}", flush=True)

    frames = []
    for board in ("main", "dual"):
        fp = FULLRUN / f"shortlist_{board}.csv"
        if not fp.exists():
            continue
        frames.append(pd.read_csv(fp, dtype={"symbol": str}))
    full = pd.concat(frames, ignore_index=True)

    raw_res = S.add_oos_pred(full, records)
    w = raw_res[raw_res["symbol"] == WATCH]
    if len(w):
        cols = ["board", "cut", "rk", "systems", "co_occur", "score"] + [
            f"{k}_{h}" for h in S.HORIZONS for k in ("pred_mag", "pred_prob")
        ]
        print(f"\n[select 前] {WATCH} 行 (raw 预测):")
        print(w[cols].to_string(index=False))
    else:
        print(f"\n[select 前] {WATCH} 不在 shortlist 中!")

    res = S.select_confident(raw_res)
    w2 = res[res["symbol"] == WATCH]
    print(f"\n[select 后] {WATCH} 保留: {len(w2) > 0}")
    if len(w2):
        res2 = S.add_score(res)
        res2["regime_active"] = True
        m = S.build_merged(res2)
        print(
            m[m["symbol"] == WATCH][
                [
                    "rank",
                    "board",
                    "systems",
                    "co_occur",
                    "score",
                    "score_w",
                    "regime_active",
                    "pred_mag_2d",
                    "pred_prob_2d",
                    "pred_mag_3d",
                    "pred_prob_3d",
                ]
            ].to_string(index=False)
        )
    else:
        print("  -> 被 select_confident 剔除 (pred_mag_3d <= 0 或 prob <= 0)")

    print("\n[merged 全量, dual]")
    res2 = S.add_score(res)
    res2["regime_active"] = True
    m = S.build_merged(res2)
    md = m[m["board"] == "dual"]
    print(
        md[
            [
                "rank",
                "symbol",
                "systems",
                "co_occur",
                "score",
                "score_w",
                "regime_active",
                "pred_mag_2d",
                "pred_mag_3d",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
