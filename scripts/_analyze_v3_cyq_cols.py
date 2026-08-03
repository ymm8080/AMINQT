# -*- coding: utf-8 -*-
"""V3 实际 CYQ 列分析 — 针对 A 面板 (_x/_y 双源原始列) + dim21 派生列.

用户纠正: 我之前的基础15列 keep/drop 分析针对 Calculator base-15 (干净名),
而 V3 生产面板实际喂给模型的是 _x/_y 双源原始列 (28列) + dim21 派生列
(conc_90 / cost_bias / conc_trend_20d / cost50_rank 等).

本脚本:
  1. 用 A 面板 (panel_full_enriched_v3.parquet) 重建 OOS 帧 (与 A bundle 对齐)
  2. 报告 14 对 _x/_y 孪生列的相关性 (同一字段双源 → 纯冗余证据)
  3. 对 V3 实际 CYQ 簇逐列: standalone IC / 边际 drop / gain / 簇内冗余
  4. dedup_l2 在 V3 实际 CYQ 簇上的幸存列
  5. 给出 keep/drop 建议 (针对 V3 真实列)

OOS 帧缓存到 data/_ab_cyq_models/_oos_a_{board}.parquet
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import _verify_cyq_drop as V  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("analyze_v3_cyq_cols")

MODEL_DIR = V.MODEL_DIR
PANEL_PATH = V.PANEL_PATH

# V3 面板实际的 CYQ 原始列 (_x/_y 双源)
RAW_FIELDS = [
    "benefit_part", "avg_cost",
    "pct_70_low", "pct_70_high", "pct_70_con",
    "pct_90_low", "pct_90_high", "pct_90_con",
    "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
    "weight_avg",
]
X_COLS = [f"{f}_x" for f in RAW_FIELDS]
Y_COLS = [f"{f}_y" for f in RAW_FIELDS]

# dim21 派生列 (A bundle feature_cols 中实际存在者)
DIM21_OUT = ["conc_90", "cost_bias", "conc_trend_20d", "cost50_rank"]
DIM21_HELP = ["conc_90_industry_rank", "is_missing_conc_90", "is_missing_winner_ratio"]


def load_a_oos(board: str) -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, f"_oos_a_{board}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    logger.info("构建 A 面板 OOS 帧 [%s] ...", board)
    panel = pd.read_parquet(PANEL_PATH)
    panel = V.select_universe(panel, V.N_DUAL)
    panel = panel[panel["date"] >= V.LOOKBACK].copy()
    df = V.build_oos_frame(panel, board)
    df.to_parquet(path, index=False)
    return df


def cluster_cols(df: pd.DataFrame) -> list[str]:
    pool = X_COLS + Y_COLS + DIM21_OUT + DIM21_HELP
    return [c for c in pool if c in df.columns]


def twin_corr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in RAW_FIELDS:
        xc, yc = f"{f}_x", f"{f}_y"
        if xc in df.columns and yc in df.columns:
            rho = df[[xc, yc]].corr(method="spearman").iloc[0, 1]
            rows.append((f, float(rho)))
    return pd.DataFrame(rows, columns=["field", "rho_x_y"])


def redundancy_in_cluster(df: pd.DataFrame, col: str, cluster: list[str]) -> float:
    others = [c for c in cluster if c != col and c in df.columns]
    if col not in df.columns or not others:
        return 0.0
    return float(df[[col] + others].corr(method="spearman").abs().loc[col, others].max())


def main() -> None:
    oos = {b: load_a_oos(b) for b in ("main", "dual")}
    bundles = {
        b: V.DualTrackTrainer.load(os.path.join(MODEL_DIR, f"{b}_ab_a.pkl"))
        for b in ("main", "dual")
    }
    model_cols = {
        b: [c for c in bundles[b]["feature_cols"] if c in oos[b].columns]
        for b in ("main", "dual")
    }
    gains = {b: V.extract_importances(bundles[b]) for b in ("main", "dual")}

    # ── 1. 孪生相关性 ──
    print("\n======== 1. _x/_y 孪生列 spearman 相关性 (V3 面板双源) ========")
    tw = twin_corr(oos["main"])
    tw["n"] = int(len(oos["main"]))
    for _, r in tw.iterrows():
        flag = "  <-- 高度重复" if abs(r["rho_x_y"]) >= 0.95 else ""
        print(f"  {r['field']:<14} ρ(x,y)={r['rho_x_y']:+.4f}{flag}")

    # ── 2. 逐列统计 ──
    cluster = cluster_cols(oos["main"])
    cluster_d = cluster_cols(oos["dual"])
    print(f"\n======== 2. V3 实际 CYQ 簇 (main={len(cluster)}, dual={len(cluster_d)}) ========")
    print(f"  簇: {', '.join(cluster)}")

    hdr = f"{'col':<26}{'stand_m':>9}{'drop_m':>9}{'gain_m':>8}{'stand_d':>9}{'drop_d':>9}{'gain_d':>8}{'redun':>7}"
    print(hdr)
    print("-" * len(hdr))
    for col in sorted(set(cluster) | set(cluster_d)):
        cells = [col.ljust(26)]
        redun_v = None
        for b in ("main", "dual"):
            df = oos[b]
            if col not in df.columns:
                cells += ["     -", "     -", "      -"]
                continue
            _, ic_drop = V.drop_col_ic(df, bundles[b], model_cols[b], col)
            cells += [
                f"{V.col_ics(df, col)['3d']:+.5f}".rjust(9),
                f"{ic_drop:+.5f}".rjust(9),
                f"{gains[b].get(col, 0.0):.0f}".rjust(8),
            ]
            if redun_v is None:
                redun_v = redundancy_in_cluster(df, col, cluster if b == "main" else cluster_d)
        cells.append(f"{redun_v:.3f}".rjust(7) if redun_v is not None else "     -")
        print("".join(cells))
    print("-" * len(hdr))
    print("stand=独立rankIC; drop=全模型IC-打乱该列IC; gain=真实importance; redun=簇内最大|spearman|")

    # ── 3. dedup_l2 在 V3 簇上 ──
    print("\n======== 3. dedup_l2 (0.7) 在 V3 实际 CYQ 簇 ========")
    for b in ("main", "dual"):
        df = oos[b]
        cl = cluster if b == "main" else cluster_d
        try:
            from app.pipeline1.feature_selector import dedup_l2
            kept = dedup_l2(cl, df, threshold=0.7)
        except Exception as e:  # noqa: BLE001
            logger.warning("dedup_l2 失败 [%s]: %s", b, e)
            kept = []
        drop = [c for c in cl if c not in kept]
        print(f"  [{b}] 簇 {len(cl)} -> 幸存 {len(kept)} | DROP {len(drop)}")
        print(f"      KEEP: {', '.join(kept)}")
        print(f"      DROP: {', '.join(drop)}")

    print("\n======== DONE ========")


if __name__ == "__main__":
    main()
