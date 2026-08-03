# -*- coding: utf-8 -*-
"""B 基础15列 vs 扩展候选列: 同一 C bundle, 同一 July OOS 帧.

对每列 (BASE_15 + 5个扩展候选) 输出:
  standalone IC (独立 rank IC), drop (打乱边际), gain (真实 importance),
  redun (vs base-15 最大|spearman|). 便于对比扩展列是否在 B 之上提供新信息.

OOS 帧缓存到 data/_ab_cyq_models/_oos_{board}.parquet.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import _verify_cyq_drop as V  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_base15_vs_ext")

CANDIDATES = [
    "chip_entropy",
    "chip_skew_dist",
    "peak_roc_5d",
    "peak_roc_20d",
    "peak_mass",
]

# A/B/C 整包 July OOS 3d rank IC (来自主运行日志) — 参考基线行
VARIANT_IC = {
    "A_variant(整包)": {"main": -0.2039, "dual": -0.1185},
    "B_variant(整包)": {"main": -0.2033, "dual": -0.1240},
    "C_variant(整包)": {"main": -0.2265, "dual": -0.1236},
}


def load_oos(board: str) -> pd.DataFrame:
    path = os.path.join(V.MODEL_DIR, f"_oos_{board}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    logger.info("构建 OOS 帧 [%s] ...", board)
    df = V.build_oos_frame(
        pd.read_parquet(os.path.join(V.MODEL_DIR, "_panel_c_full.parquet")), board
    )
    df.to_parquet(path, index=False)
    return df


def main() -> None:
    oos_df = {b: load_oos(b) for b in ("main", "dual")}
    bundle_c = {
        b: V.DualTrackTrainer.load(os.path.join(V.MODEL_DIR, f"{b}_ab_c_bundle.pkl"))
        for b in ("main", "dual")
    }
    model_cols = {
        b: [c for c in bundle_c[b]["feature_cols"] if c in oos_df[b].columns]
        for b in ("main", "dual")
    }
    gains = {b: V.extract_importances(bundle_c[b]) for b in ("main", "dual")}

    cols = [c for c in list(V.BASE_15) + CANDIDATES]
    rows = []
    for col in cols:
        r = {"col": col}
        for b in ("main", "dual"):
            df = oos_df[b]
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

    hdr = f"{'col':<20}{'stand_m':>9}{'drop_m':>8}{'gain_m':>7}{'stand_d':>9}{'drop_d':>8}{'gain_d':>7}{'redun':>7}"
    print("=" * len(hdr))
    print("B 基础15列 vs 扩展候选 — C bundle, July OOS (drop=全模型IC-打乱后IC)")
    print(hdr)
    print("-" * len(hdr))
    for name, ics in VARIANT_IC.items():
        print(
            name.ljust(20)
            + f"{ics['main']:+.4f}".rjust(9)
            + "        "
            + "       "
            + f"{ics['dual']:+.4f}".rjust(9)
            + "        "
            + "       "
            + "       "
        )
    print("-" * len(hdr))
    for r in rows:
        cells = [r["col"].ljust(20)]
        for b in ("main", "dual"):
            v = r[b]
            if v is None:
                cells += ["     -", "     -", "     -"]
            else:
                cells += [
                    f"{v['stand']:+.4f}".rjust(9),
                    f"{v['drop']:+.4f}".rjust(8),
                    f"{v['gain']:.0f}".rjust(7),
                ]
        redun = r["main"] or r["dual"]
        redun_v = redun["redun"] if redun else None
        cells.append(f"{redun_v:.3f}".rjust(7) if redun_v is not None else "     -")
        print("".join(cells))
    print("=" * len(hdr))
    print(
        "注: stand=独立rank IC; drop>0模型使用该列; gain=真实importance(位置对齐); redun=vs base-15最大|spearman|"
    )


if __name__ == "__main__":
    main()
