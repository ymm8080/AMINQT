# -*- coding: utf-8 -*-
"""恢复被 render 崩溃中断的 per-column 输出.

主脚本 _verify_cyq_drop.py 在 extract_importances 处崩溃 (feature_name→feature_name_),
已修复; 但 per-column 表格未打印. 本脚本复用已缓存的 panel_c_full + 已保存的
C bundles, 仅重跑 DedupL2 幸存列 + 逐列统计 (col_ics / drop_col_ic / gain / redun),
跳过 40min 的 A/B/C 重训与 OOS 重算.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import _verify_cyq_drop as V  # noqa: E402  (导入即应用其 monkeypatch, 无害)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("recover_cyq_percol")

MODEL_DIR = V.MODEL_DIR
EXTRA_FEATURES = V.EXTRA_FEATURES


def main() -> None:
    pcf_path = os.path.join(MODEL_DIR, "_panel_c_full.parquet")
    logger.info("复用缓存面板 C: %s", pcf_path)
    panel_c_full = pd.read_parquet(pcf_path)

    # DedupL2 幸存扩展列 (与主脚本同口径)
    kept_ext = V.run_dedup(panel_c_full)
    logger.info("DedupL2 幸存扩展列: %s", ", ".join(kept_ext) if kept_ext else "(空)")

    # 构建 OOS 帧 + 加载 C bundles
    oos_df: dict[str, pd.DataFrame] = {}
    bundle_c: dict[str, dict] = {}
    model_cols: dict[str, list[str]] = {}
    for b in ("main", "dual"):
        logger.info("构建 OOS 帧 [%s] ...", b)
        oos_df[b] = V.build_oos_frame(panel_c_full, b)
        bundle_c[b] = V.DualTrackTrainer.load(
            os.path.join(MODEL_DIR, f"{b}_ab_c_bundle.pkl")
        )
        model_cols[b] = [
            c for c in bundle_c[b]["feature_cols"] if c in oos_df[b].columns
        ]
        logger.info("[%s] OOS 帧 %d rows / %d cols", b, len(oos_df[b]), len(model_cols[b]))

    # 逐列统计 (与主脚本 per-col 循环同逻辑)
    per_col: dict[str, dict] = {}
    for col in EXTRA_FEATURES:
        info: dict = {"dedup": col in kept_ext}
        for b in ("main", "dual"):
            df = oos_df[b]
            if col not in df.columns:
                continue
            _, ic_drop = V.drop_col_ic(df, bundle_c[b], model_cols[b], col)
            info[b] = {
                "standalone": V.col_ics(df, col)["3d"],
                "drop": ic_drop,
                "gain": V.extract_importances(bundle_c[b]).get(col, 0.0),
                "redun": V.redundancy_corr(df, col),
            }
        per_col[col] = info

    # ── 输出 ──
    print("\n======== DedupL2 幸存扩展列 (%d/%d) ========" % (len(kept_ext), len(EXTRA_FEATURES)))
    print(", ".join(kept_ext) if kept_ext else "(全部被去重)")

    print("\n======== PER-COLUMN (C 模型单次训练; drop=全模型IC-打乱该列后IC) ========")
    print(f"{'col':<22}{'keep':>5}{'stand_m':>8}{'drop_m':>8}{'gain_m':>7}{'stand_d':>8}{'drop_d':>8}{'gain_d':>7}{'redun':>7}")
    for col, info in sorted(
        per_col.items(),
        key=lambda kv: -max(
            kv[1].get("main", {}).get("drop", 0.0),
            kv[1].get("dual", {}).get("drop", 0.0),
        ),
    ):
        m, d = info.get("main"), info.get("dual")
        keep = "Y" if info["dedup"] else "."
        print(
            f"{col:<22}{keep:>5}"
            f"{V._cell(m['standalone'] if m else None, '{:8.4f}')}"
            f"{V._cell(m['drop'] if m else None, '{:+8.4f}')}"
            f"{V._cell(m['gain'] if m else None, '{:7.0f}')}"
            f"{V._cell(d['standalone'] if d else None, '{:8.4f}')}"
            f"{V._cell(d['drop'] if d else None, '{:+8.4f}')}"
            f"{V._cell(d['gain'] if d else None, '{:7.0f}')}"
            f"{V._cell((m or d)['redun'] if (m or d) else None, '{:7.3f}')}"
        )

    print("\n======== DONE ========")


if __name__ == "__main__":
    main()
