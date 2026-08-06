# -*- coding: utf-8 -*-
"""_diag_analog_stage.py — 钉死 features.build 是否污染 close/close_hfq/price_1455 与标签 (2026-08-04 v2).

v1 教训: C 段用 label_pm_20d_net 但 LABEL_HORIZONS=(1,2,3,5) 不产 20d → 必 KeyError,
         且整段只在末尾 print, 中途崩则白扔 ~50min features.build.

v2 修正:
  1) C 段改用存在标签 label_pm_3d_net;
  2) B3 证据扩展到 price_1455 (晚盘执行价, build_labels 的执行口径来源);
  3) 每节 try/except 独立落盘部分日志, 单节失败不丢已有结果;
  4) features.build(main) 的 3y 生产切片落 parquet 检查点
     data/_diag_stage_main_3y.parquet, 供 _reclassify_all_features.py 复用 (省重建).

方法 (仅主板, 省时):
  行集 = run_train(main) → features.build(registry=None) → build_path_labels
       → build_labels (B9 晚盘执行口径) → mask_suspension → mask_recent_days
       → 3y 窗口切片 (与 prepare_board_frame 完全一致).
  验证: 清洗后 vs features.build 输出的 close/close_hfq/price_1455 逐行最大差;
        label_pm_3d_net 相关性 + 符号反转比例; 特征输出上重算 6格 是否漂移.
输出: data/_diag_analog_stage_<ts>.log (WORM) + data/_diag_stage_main_3y.parquet (检查点).
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import LABELS, MASK_RECENT_DAYS
from scripts._classify_freq_full import (
    MIN_CROSS,
    MIN_OBS,
    WINDOWS,
    _f,
    _wtsic,
    group_spearman,
)

CHECKPOINT = os.path.join("data", "_diag_stage_main_3y.parquet")


def classify(work, col):
    g_grp = work.groupby("symbol")
    wins = {}
    for w in WINDOWS.values():
        wins[f"{col}_p{w}"] = (work[col] / g_grp[col].shift(w) - 1.0).astype("float64")
    g_sym, g_date = work["symbol"], work["date"]
    lab_sym = {label: work.groupby("symbol")[label].rank() for label in LABELS}
    lab_date = {label: work.groupby("date")[label].rank() for label in LABELS}
    tsic, xic = {}, {}
    for f, wc in wins.items():
        wr_sym = wc.groupby(g_sym.values).rank()
        wr_date = wc.groupby(g_date.values).rank()
        tsic[f] = {
            label: group_spearman(wr_sym, lab_sym[label], g_sym, MIN_OBS)
            for label in LABELS
        }
        xic[f] = {
            label: group_spearman(wr_date, lab_date[label], g_date, MIN_CROSS)
            for label in LABELS
        }
    ts = {w: _wtsic(tsic[f"{col}_p{w}"]) for w in (1, 5, 20)}
    xs = {w: _wtsic(xic[f"{col}_p{w}"]) for w in (1, 5, 20)}
    cells = {
        "TS日": ts[1],
        "TS周": ts[5],
        "TS月": ts[20],
        "XS日": xs[1],
        "XS周": xs[5],
        "XS月": xs[20],
    }
    return cells


def _fmt_cells(cells):
    best = max(cells, key=lambda k: abs(cells[k]))
    return (
        f"{_f(cells['TS日']):>8}{_f(cells['TS周']):>8}{_f(cells['TS月']):>8}"
        f"{_f(cells['XS日']):>8}{_f(cells['XS周']):>8}{_f(cells['XS月']):>8}"
        f"  ← {best} ({abs(cells[best]):.4f})"
    )


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    lines = []
    lines.append("=" * 78)
    lines.append(
        "  features.build 污染钉死 v2 — 主板 (检查点复用协议: " + CHECKPOINT + ")"
    )
    lines.append("=" * 78)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("data", f"_diag_analog_stage_{ts}.log")

    def flush():
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)

    # ── A. 面板直读口径 (基准) ──
    try:
        read_cols = ["date", "symbol", "is_suspended", "close_hfq", "close", "amount"]
        base_df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
        base_df = LabelEngine.build_labels(base_df, session="PM")
        base_df = LabelEngine.mask_suspension(base_df)
        base_df = LabelEngine.mask_recent_days(base_df, days=MASK_RECENT_DAYS)
        cutoff = base_df["date"].max() - pd.DateOffset(years=3)
        base_work = base_df[base_df["date"] >= cutoff].reset_index(drop=True)
        lines.append(
            f"[A 面板直读] rows={len(base_work):,} stocks={base_work['symbol'].nunique():,}"
        )
        for c in ["close", "close_hfq", "amount"]:
            lines.append(f"  {c:<12}{_fmt_cells(classify(base_work, c))}  [基准]")
        del base_df, base_work
        gc.collect()
    except Exception as e:
        lines.append(f"[A 异常] {type(e).__name__}: {e}")
    flush()

    # ── B. run_train(main) → features.build → 3y 切片 + 检查点 ──
    try:
        panel = pd.read_parquet(PANEL_V3_PATH)
        main_df, dual_df = CleaningPipeline().run_train(panel)
        del panel
        gc.collect()
        lines.append(
            f"[B1 run_train main] rows={len(main_df):,} (dual 另行处理, 此处仅主板)"
        )
        clean_snap = main_df[
            ["symbol", "date", "is_suspended", "close", "close_hfq", "amount"]
        ].copy()
        if "price_1455" in main_df.columns:
            clean_snap["price_1455"] = main_df["price_1455"]
        fe = FeatureEngineV35()
        d = fe.build(main_df, None, cross_sectional_rank=False, registry=None)
        del main_df
        gc.collect()
        lines.append(
            f"[B2 features.build] rows={len(d):,} cols={d.shape[1]:,} "
            f"(行数差 = {len(d) - len(clean_snap):,})"
        )
        lines.append(
            f"[B2b 清洗行集含 price_1455] {'是' if 'price_1455' in clean_snap.columns else '否'} | "
            f"features 输出含 price_1455: {'是' if 'price_1455' in d.columns else '否'}"
        )

        # B3: 逐行对比关键价列
        keys = [c for c in ["close", "close_hfq", "price_1455"] if c in d.columns]
        m = clean_snap.merge(
            d[["symbol", "date"] + keys], on=["symbol", "date"], suffixes=("_c", "_b")
        )
        diffs = []
        for c in keys:
            diffs.append(f"{c} max|Δ|={abs(m[f'{c}_c'] - m[f'{c}_b']).max():.6f}")
        lines.append(f"[B3 重叠(symbol,date)={len(m):,}] " + " | ".join(diffs))
        del m
        gc.collect()

        # 生产序列: build_path_labels + build_labels + 掩码 + 3y 切片 → 检查点
        d = LabelEngine.build_path_labels(d)
        d = LabelEngine.build_labels(d, session="PM")
        d = LabelEngine.mask_suspension(d)
        d = LabelEngine.mask_recent_days(d, days=MASK_RECENT_DAYS)
        latest = d["date"].max()
        cutoff = latest - pd.DateOffset(years=3)
        d3 = d[d["date"] >= cutoff].reset_index(drop=True)
        lines.append(
            f"[B4 生产切片] 全历史 rows={len(d):,} → 3y rows={len(d3):,} "
            f"stocks={d3['symbol'].nunique():,} | latest={latest:%Y-%m-%d}"
        )
        d3.to_parquet(CHECKPOINT, index=False)
        lines.append(
            f"[B4 检查点] 已落盘 {CHECKPOINT} ({os.path.getsize(CHECKPOINT) / 1e9:.2f} GB)"
        )
        del d
        gc.collect()

        # ── C. 标签完整性: 同 3y 窗口+掩码, 清洗行 vs features 输出的 label 是否一致 ──
        try:
            # 顺序须与生产一致: build_labels → mask_suspension → mask_recent_days
            # (mask_* 引用 label_*d, 必须先有标签列, 否则 KeyError)
            clean3 = (
                clean_snap[clean_snap["date"] >= cutoff]
                .sort_values(["symbol", "date"])
                .reset_index(drop=True)
            )
            clean3 = LabelEngine.build_labels(clean3, session="PM")
            clean3 = LabelEngine.mask_suspension(clean3)
            clean3 = LabelEngine.mask_recent_days(clean3, days=MASK_RECENT_DAYS)
            lab_clean = clean3[["symbol", "date", "label_pm_3d_net"]]
            lab_built = LabelEngine.build_labels(
                d3.sort_values(["symbol", "date"]).reset_index(drop=True).copy(),
                session="PM",
            )[["symbol", "date", "label_pm_3d_net"]]
            j = lab_clean.merge(lab_built, on=["symbol", "date"], suffixes=("_c", "_b"))
            j = j.dropna(subset=["label_pm_3d_net_c", "label_pm_3d_net_b"])
            jc = j["label_pm_3d_net_c"].corr(j["label_pm_3d_net_b"])
            md = (j["label_pm_3d_net_c"] - j["label_pm_3d_net_b"]).abs().max()
            flip = (
                np.sign(j["label_pm_3d_net_c"]) != np.sign(j["label_pm_3d_net_b"])
            ).mean()
            lines.append(
                f"[C 标签完整性] 重叠={len(j):,} | label_pm_3d_net 相关系数={jc:.4f} | "
                f"最大差={md:.6f} | 符号反转比例={flip:.2%}"
            )
        except Exception as e:
            lines.append(f"[C 异常] {type(e).__name__}: {e}")
        del clean_snap, clean3, lab_clean, lab_built, j
        gc.collect()

        # ── D. 在 features 输出上重算 6格 (预期与 B 行集一致) ──
        for c in ["close", "close_hfq", "amount"]:
            if c not in d3.columns:
                lines.append(f"  {c:<12} (列缺失)")
                continue
            lines.append(
                f"  {c:<12}{_fmt_cells(classify(d3, c))}  [features.build 生产切片]"
            )
        del d3
        gc.collect()
    except Exception as e:
        lines.append(f"[B 异常] {type(e).__name__}: {e}")
    flush()

    print(f"\n落盘: {log_path}", flush=True)


if __name__ == "__main__":
    main()
