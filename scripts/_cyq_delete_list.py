# -*- coding: utf-8 -*-
"""从缓存 OOS 帧 + C bundle 模型 + DedupL2 输出 CYQ 删列/保留清单.

复用 _compare_base15_vs_ext 已缓存的 _oos_{board}.parquet, 避免重算特征引擎.
决策规则:
  KEEP  = DedupL2 幸存 且 (main 或 dual 边际 drop > 0) 且 非高冗余
  DELETE = 其余 (drop<=0 无正贡献 / 高冗余 / 被去相关剔除)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import _verify_cyq_drop as V  # noqa: E402
import _compare_base15_vs_ext as C  # noqa: E402

REDUN_REFUSE = 0.70  # 冗余度超过此值判高冗余


def main() -> None:
    oos = {b: C.load_oos(b) for b in ("main", "dual")}
    bundle_c = {
        b: V.DualTrackTrainer.load(os.path.join(V.MODEL_DIR, f"{b}_ab_c_bundle.pkl"))
        for b in ("main", "dual")
    }
    model_cols = {
        b: [c for c in bundle_c[b]["feature_cols"] if c in oos[b].columns]
        for b in ("main", "dual")
    }
    gains = {b: V.extract_importances(bundle_c[b]) for b in ("main", "dual")}

    panel_c = pd.read_parquet(os.path.join(V.MODEL_DIR, "_panel_c_full.parquet"))
    kept_ext = V.run_dedup(panel_c)

    rows = []
    for col in sorted(set(V.EXTRA_FEATURES) | V.BASE_15):
        r = {"col": col, "dedup": col in kept_ext}
        for b in ("main", "dual"):
            df = oos[b]
            if col not in df.columns:
                r[b] = None
                continue
            _, drop = V.drop_col_ic(df, bundle_c[b], model_cols[b], col)
            r[b] = {
                "stand": V.col_ics(df, col)["3d"],
                "drop": drop,
                "gain": gains[b].get(col, 0.0),
                "redun": V.redundancy_corr(df, col),
            }
        rows.append(r)

    # 决策
    for r in rows:
        m, d = r["main"], r["dual"]
        drop_max = max((m or {}).get("drop", 0.0), (d or {}).get("drop", 0.0))
        redun = (m or d).get("redun", 0.0) if (m or d) else 0.0
        r["drop_max"] = drop_max
        r["redun"] = redun
        r["verdict"] = (
            "KEEP"
            if (r["dedup"] and drop_max > 0 and redun < REDUN_REFUSE)
            else "DELETE"
        )

    # 扩展列先, 基础列后
    ext = [r for r in rows if r["col"] in V.EXTRA_FEATURES]
    base = [
        r for r in rows if r["col"] in V.BASE_15 and r["col"] not in V.EXTRA_FEATURES
    ]
    ext.sort(key=lambda r: (r["verdict"], -r["drop_max"]))
    base.sort(key=lambda r: (r["verdict"], -r["drop_max"]))

    print("=" * 110)
    print(
        "CYQ 删列/保留清单 — C bundle, July OOS (drop=全模型IC-打乱后IC; redun=vs base-15 最大|spearman|)"
    )
    print(
        f"{'col':<22}{'verdict':>8}{'dedup':>6}{'drop_m':>8}{'drop_d':>8}{'stand_m':>9}{'stand_d':>9}{'gain_m':>7}{'gain_d':>7}{'redun':>7}"
    )
    print("-" * 110)
    for r in ext + base:
        m, d = r["main"], r["dual"]
        print(
            f"{r['col']:<22}{r['verdict']:>8}{'Y' if r['dedup'] else '.':>6}"
            f"{(m['drop'] if m else 0.0):+8.4f}{(d['drop'] if d else 0.0):+8.4f}"
            f"{(m['stand'] if m else 0.0):+9.4f}{(d['stand'] if d else 0.0):+9.4f}"
            f"{(m['gain'] if m else 0):>7.0f}{(d['gain'] if d else 0):>7.0f}"
            f"{r['redun']:>7.3f}"
        )
    print("=" * 110)

    keep = [r["col"] for r in rows if r["verdict"] == "KEEP"]
    dele = [r["col"] for r in rows if r["verdict"] == "DELETE"]
    keep_ext = [c for c in keep if c in V.EXTRA_FEATURES]
    print(f"\nKEEP   ({len(keep)}): {', '.join(keep)}")
    print(f"DELETE ({len(dele)}): {', '.join(dele)}")
    print(
        f"\n[扩展列] KEEP {len(keep_ext)}/{len(V.EXTRA_FEATURES)}: {', '.join(keep_ext)}"
    )


if __name__ == "__main__":
    main()
