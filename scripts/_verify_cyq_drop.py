# -*- coding: utf-8 -*-
"""A/B 验证: 当前面板(_x Tushare + _y Calculator) vs Calculator-only.

问题: 去掉 Tushare CYQ、只用 Calculator, 是否会提升(或至少不损害)预测质量?
C 变体: Calculator 基础15列 + 扩展25列, 单次 bundled 训练 (不复训),
      再用 DedupL2 对 CYQ 簇去相关, 逐列报告 {独立IC, 边际drop, gain, 冗余度}.

设计:
  - 同一训练管线 (run_training), 唯一差异是 CYQ 列表示:
      A_current   = panel_full_enriched_v3 原样 (cost_50pct_x + cost_50pct_y 等)
                    → dim21 无裸列, 实际为 OHLCV 代理 CYQ (非真实 Tushare)
      B_calc_only = 丢弃 *_x/*_y, merge cyq_panel.parquet 裸列 (Calculator 15列 + winner_ratio)
      C_calc_bundle = B + 25 扩展列 (单次 bundled 训练; DedupL2 报告去相关幸存列)
  - 样本宇宙: 主板=沪深300成分, 双创=成交额活跃 top-300
  - 训练窗口: 2025-11-01 -> 2026-06-30 (>=150 交易日, trainer 硬下限)
  - OOS 窗口: 2026-07-01 -> 2026-07-28 (t+3 rank IC)
  - 为隔离 CYQ 源效应: use_ic_screen=False, use_registry=False (全量特征, 无 brute-force)
  - 可恢复: panel/cyq_ext 缓存到 MODEL_DIR, 中断不重算 10min CYQ
"""

from __future__ import annotations

import logging
import os
import sys

# 允许 `python scripts/_verify_cyq_drop.py` 直接运行 (sys.path[0] 指向 scripts/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("verify_cyq_drop")

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.cyq_ext import compute_cyq_panel
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.train_runner import prepare_board_frame, run_training

# ── 样本测试: 仅训练 + 评估 t+3 视界 ──
# 跳过 1d/2d/5d 与 E1/E2/LambdaRank extras; 校准器 no-op (fit_calibrator 硬依赖 1d_cls)
import app.pipeline1.dual_track_trainer as _dtt

_dtt.MODEL_KINDS = ("3d_reg",)
_dtt.DualTrackTrainer._train_extras = lambda self, out, checkpoint=None: None


def _noop_calibrator(trained):
    trained["calibrator"] = None
    return None


_dtt.DualTrackTrainer.fit_calibrator = staticmethod(_noop_calibrator)

PANEL_PATH = "data/panel_full_enriched_v3.parquet"
CYQ_CACHE = "data/cyq_panel.parquet"
CSI300_CACHE = "data/_csi300_members.parquet"
MODEL_DIR = "data/_ab_cyq_models"

# ── 最近样本窗口 (t+3 only) ──
# 训练窗口必须 >= 150 交易日 (trainer 硬下限 ES20+CALIB20+TEST60+MIN50);
# 最近半年仅 ~116 交易日, 按用户裁决 "IF 150 DAYS IS MINIMUM, THEN MAKE IT",
# 训练窗口放宽到 >= 150 交易日 (2025-11 -> 2026-06 ≈ 159d). OOS 用最近一个月.
TRAIN_LO = pd.Timestamp("2025-11-01")
TRAIN_END = pd.Timestamp("2026-06-30")
OOS_LO = pd.Timestamp("2026-07-01")
OOS_HI = pd.Timestamp("2026-07-28")  # 3d 标签需前瞻, 面板截至 2026-07-31
LOOKBACK = pd.Timestamp("2025-07-01")  # OOS 特征回看起点 (bias_250 ≈ 1年)

SEED = 42
N_MAIN = 300   # 沪深300 成分 (主板仅此)
N_DUAL = 300   # 双创成交额活跃 top-300

LABEL = "label_pm_3d_net"

CYQ_COLS = [
    "benefit_part",
    "avg_cost",
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
    "pct_90_low",
    "pct_90_high",
    "pct_90_con",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
]

# 扩展候选列: Calculator 150桶分布导出的新列 + Tier-0 组合 + 众数迁移
EXTRA_FEATURES = [
    "cost_10pct", "cost_20pct", "cost_30pct", "cost_40pct",
    "cost_60pct", "cost_70pct", "cost_80pct",
    "peak_price", "peak_mass", "chip_entropy", "chip_gini", "chip_skew_dist",
    "mass_above_close", "mass_above_1_1x", "mass_above_1_2x", "mass_below_0_9x",
    "resistance_dist", "support_dist", "pct_60_con", "pct_80_con",
    "benefit_x_con", "price_position", "chip_range_width",
    "peak_roc_5d", "peak_roc_20d",
]
# cyq_panel 缓存已有的基础列 (含 Calculator 的 winner_ratio 别名)
BASE_15 = set(CYQ_COLS) | {"winner_ratio"}


def load_csi300() -> set[str]:
    """沪深300 成分代码 (akshare, 缓存到 parquet 避免重复抓取)."""
    import akshare as ak

    if os.path.exists(CSI300_CACHE):
        return set(pd.read_parquet(CSI300_CACHE)["code"].astype(str).str.zfill(6))
    # akshare 返回的列名为 GBK 乱码 (Ʒ�ִ��� = 品种代码), 按位置取首列
    df = ak.index_stock_cons(symbol="000300")
    codes = df.iloc[:, 0].astype(str).str.zfill(6).tolist()
    pd.DataFrame({"code": codes}).to_parquet(CSI300_CACHE, index=False)
    return set(codes)


def select_universe(
    panel: pd.DataFrame, n_dual: int = N_DUAL
) -> pd.DataFrame:
    """主板=沪深300成分; 双创=近90日成交额活跃 top-n_dual.

    主板与双创按证券代码前缀划分 (board_of), 沪深300 含少数双创成分,
    仅取 main 前缀进入主板样本, 避免混入双创轨.
    """
    from app.pipeline1.cleaning_pipeline import board_of

    sym = panel["symbol"].astype(str).str.zfill(6)
    main_uni = {s for s in sym.unique() if s in load_csi300() and board_of(s) == "main"}

    recent = panel[panel["date"] >= panel["date"].max() - pd.Timedelta(days=90)].copy()
    recent["_board"] = recent["symbol"].astype(str).str.zfill(6).map(board_of)
    dual_pool = recent[recent["_board"].isin(("GEM", "STAR"))]
    top_dual = (
        dual_pool.groupby("symbol")["amount"]
        .median()
        .sort_values(ascending=False)
        .head(n_dual)
        .index.astype(str)
        .tolist()
    )
    keep = main_uni | set(top_dual)
    return panel[sym.isin(keep)].copy()


def build_variant_b(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculator-only: 丢 Tushare _x 与 Calculator _y, merge 裸 Calculator 列."""
    cyq = pd.read_parquet(CYQ_CACHE)[["symbol", "date"] + CYQ_COLS]
    cyq["winner_ratio"] = cyq["benefit_part"]  # 获利盘比例 (Calculator 15列之一)
    drop = [c for c in panel.columns if c.endswith("_x") or c.endswith("_y")]
    out = panel.drop(columns=drop).merge(cyq, on=["symbol", "date"], how="left")
    return out


def build_variant_c(panel: pd.DataFrame, cyq_ext: pd.DataFrame) -> pd.DataFrame:
    """Calculator-only 基础 15 列 + Calculator 新导出列 + Tier-0 组合 + 众数迁移."""
    out = build_variant_b(panel)
    extra = [
        c
        for c in cyq_ext.columns
        if c not in ("symbol", "date", "winner_ratio") and c not in BASE_15
    ]
    if extra:
        out = out.merge(
            cyq_ext[["symbol", "date"] + extra], on=["symbol", "date"], how="left"
        )
    # Tier-0 组合 (来自基础列)
    out["benefit_x_con"] = out["benefit_part"] * out["pct_90_con"]
    d5 = (out["cost_95pct"] - out["cost_5pct"]).replace(0, np.nan)
    out["price_position"] = (out["close"] - out["cost_5pct"]) / d5
    out["chip_range_width"] = d5 / out["close"].replace(0, np.nan)
    # 众数迁移 (时间导数)
    out = out.sort_values(["symbol", "date"])
    g = out.groupby("symbol")["peak_price"]
    out["peak_roc_5d"] = g.pct_change(5, fill_method=None)
    out["peak_roc_20d"] = g.pct_change(20, fill_method=None)
    return out


def build_oos_frame(board_panel: pd.DataFrame, board: str) -> pd.DataFrame:
    """构建板块 OOS 标签帧 (外部评估, 独立于模型包自带 oos)."""
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    df = cleaner.run_train(board_panel)[0 if board == "main" else 1]
    df = prepare_board_frame(df, features, cross_sectional_rank=(board != "main"))
    t = df[(df["date"] >= OOS_LO) & (df["date"] <= OOS_HI)]
    return t.dropna(subset=[LABEL])


def oos_eval(board_panel: pd.DataFrame, board: str, bundle_path: str) -> dict:
    """外部 OOS 3d rank IC (用独立 OOS 窗口)."""
    bundle = DualTrackTrainer.load(bundle_path)
    t = build_oos_frame(board_panel, board)
    if len(t) < 30:
        return {"ics": {"3d_reg": 0.0}}
    cols = [c for c in bundle["feature_cols"] if c in t.columns]
    X = np.nan_to_num(t[cols].values, nan=0.0)
    pred = bundle["models"]["3d_reg"][0].predict(X)
    ic3 = ICScreener.rank_ic(t.assign(score=pred), "score", LABEL)
    return {"ics": {"3d_reg": ic3}}


def col_ics(df: pd.DataFrame, col: str) -> dict:
    """单列独立 OOS rank IC."""
    t = df.dropna(subset=[col, LABEL])
    if len(t) < 30:
        return {"3d": 0.0, "max_abs_ic": 0.0}
    ic = ICScreener.rank_ic(t[["date", col, LABEL]], col, LABEL)
    return {"3d": ic, "max_abs_ic": abs(ic)}


def extract_importances(bundle: dict) -> dict[str, float]:
    """模型 feature_importance (累加各 _reg 模型).

    trainer 以 numpy .values 拟合 (无列名), feature_name_ 为 Column_N 占位;
    必须按 bundle['feature_cols'] 位置对齐 feature_importances_.
    """
    imp: dict[str, float] = {}
    fc = bundle["feature_cols"]
    for kind, pair in bundle["models"].items():
        if not kind.endswith("_reg"):
            continue
        model = pair[0]
        for name, val in zip(fc, model.feature_importances_):
            imp[name] = imp.get(name, 0.0) + float(val)
    return imp


def redundancy_corr(df: pd.DataFrame, col: str) -> float:
    """该列 vs 基础15列的最大 |spearman| 相关性 (冗余度)."""
    base = [c for c in BASE_15 if c in df.columns and c != col]
    if not base or col not in df.columns:
        return 0.0
    return float(df[[col] + base].corr(method="spearman").abs().loc[col, base].max())


def drop_col_ic(
    df: pd.DataFrame,
    bundle: dict,
    cols: list[str],
    col: str,
    seed: int = SEED,
) -> tuple[float, float]:
    """单列边际贡献: 全模型 OOS rank IC - 打乱该列后 OOS rank IC.

    drop > 0 → 该列信息被模型使用, 对预测有正贡献 (增量近似);
    drop ≈ 0 → 冗余 (已有特征覆盖); drop < 0 → 打乱后反而更高 (噪声/过拟合).
    """
    rng = np.random.default_rng(seed)
    model = bundle["models"]["3d_reg"][0]
    t = df.dropna(subset=[LABEL]).copy()
    if col not in cols or len(t) < 30:
        return 0.0, 0.0
    X = np.nan_to_num(t[cols].values, nan=0.0)

    def _ic(Xm: np.ndarray) -> float:
        return ICScreener.rank_ic(t.assign(score=model.predict(Xm)), "score", LABEL)

    ic_full = _ic(X)
    i = cols.index(col)
    Xs = X.copy()
    Xs[:, i] = rng.permutation(Xs[:, i])
    return ic_full, ic_full - _ic(Xs)


def run_dedup(panel_c_full: pd.DataFrame) -> list[str]:
    """DedupL2: 对 CYQ 簇 (基础15 + 扩展25) 去相关, 返回幸存扩展列."""
    from app.pipeline1.feature_selector import dedup_l2

    train = panel_c_full[
        (panel_c_full["date"] >= TRAIN_LO) & (panel_c_full["date"] <= TRAIN_END)
    ]
    pool = [c for c in list(BASE_15) + EXTRA_FEATURES if c in train.columns]
    kept = dedup_l2(pool, train, threshold=0.7)
    kept_ext = [c for c in EXTRA_FEATURES if c in kept]
    logger.info(
        "DedupL2: CYQ簇 %d -> %d 列; 扩展列幸存 %d/%d",
        len(pool),
        len(kept),
        len(kept_ext),
        len(EXTRA_FEATURES),
    )
    return kept_ext


def _cell(x: float | None, fmt: str, dash: str = "     -") -> str:
    return fmt.format(x) if x is not None else dash


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    pb_path = os.path.join(MODEL_DIR, "_panel_b.parquet")
    pcf_path = os.path.join(MODEL_DIR, "_panel_c_full.parquet")
    ext_path = os.path.join(MODEL_DIR, "_cyq_ext.parquet")

    # ── 样本宇宙 (A 面板) ──
    panel = pd.read_parquet(PANEL_PATH)
    panel = select_universe(panel, N_DUAL)
    panel = panel[panel["date"] >= LOOKBACK].copy()
    logger.info("样本宇宙: %d syms / %d rows", panel["symbol"].nunique(), len(panel))

    # ── B/C 面板 (可恢复: 缓存避免 10min CYQ 重算) ──
    if not (os.path.exists(pb_path) and os.path.exists(pcf_path)):
        if os.path.exists(ext_path):
            logger.info("复用缓存 cyq_ext")
            cyq_ext = pd.read_parquet(ext_path)
        else:
            logger.info("=== 计算扩展 CYQ (%d syms) ===", panel["symbol"].nunique())
            cyq_ext = compute_cyq_panel(panel)
            cyq_ext.to_parquet(ext_path, index=False)
        panel_b = build_variant_b(panel)
        panel_c_full = build_variant_c(panel, cyq_ext)
        panel_b.to_parquet(pb_path, index=False)
        panel_c_full.to_parquet(pcf_path, index=False)
    else:
        logger.info("复用缓存面板 B/C")
        panel_b = pd.read_parquet(pb_path)
        panel_c_full = pd.read_parquet(pcf_path)

    # ── A/B/C 训练或复用 ──
    variants = {
        "A_current": (panel, "ab_a"),
        "B_calc_only": (panel_b, "ab_b_calc_only"),
        "C_calc_bundle": (panel_c_full, "ab_c_bundle"),
    }
    results: dict[tuple[str, str], float] = {}
    for name, (vpanel, tag) in variants.items():
        train = vpanel[
            (vpanel["date"] >= TRAIN_LO) & (vpanel["date"] <= TRAIN_END)
        ]
        n_days = int(train["date"].nunique())
        assert n_days >= 150, f"训练窗口过短: {n_days} < 150"
        paths = {
            b: os.path.join(MODEL_DIR, f"{b}_{tag}.pkl") for b in ("main", "dual")
        }
        if not all(os.path.exists(p) for p in paths.values()):
            logger.info(
                "=== TRAIN %s: %d rows / %d syms / %d days ===",
                name,
                len(train),
                train["symbol"].nunique(),
                n_days,
            )
            run_training(
                train,
                tag,
                model_dir=MODEL_DIR,
                use_ic_screen=False,
                use_registry=False,
                enable_adoption=False,
            )
        for b, p in paths.items():
            ic3 = oos_eval(vpanel, b, p)["ics"]["3d_reg"]
            results[(name, b)] = ic3
            logger.info("[%s/%s] OOS 3d IC=%.4f", name, b, ic3)

    # ── DedupL2: CYQ 簇去相关 ──
    kept_ext = run_dedup(panel_c_full)
    b_base = {b: results[("B_calc_only", b)] for b in ("main", "dual")}
    c_ic = {b: results[("C_calc_bundle", b)] for b in ("main", "dual")}

    # ── 逐列统计 (单次 bundled-C, OOS) ──
    oos_df: dict[str, pd.DataFrame] = {}
    bundle_c: dict[str, dict] = {}
    model_cols: dict[str, list[str]] = {}
    for b in ("main", "dual"):
        oos_df[b] = build_oos_frame(panel_c_full, b)
        bundle_c[b] = DualTrackTrainer.load(
            os.path.join(MODEL_DIR, f"{b}_ab_c_bundle.pkl")
        )
        model_cols[b] = [
            c for c in bundle_c[b]["feature_cols"] if c in oos_df[b].columns
        ]

    per_col: dict[str, dict] = {}
    for col in EXTRA_FEATURES:
        info: dict = {"dedup": col in kept_ext}
        for b in ("main", "dual"):
            df = oos_df[b]
            if col not in df.columns:
                continue
            _, ic_drop = drop_col_ic(df, bundle_c[b], model_cols[b], col)
            info[b] = {
                "standalone": col_ics(df, col)["3d"],
                "drop": ic_drop,
                "gain": extract_importances(bundle_c[b]).get(col, 0.0),
                "redun": redundancy_corr(df, col),
            }
        per_col[col] = info

    # ── 汇总输出 ──
    print("\n======== A/B/C 汇总 (外部 OOS 2026-07, t+3 rank IC) ========")
    print("注: A=原面板(_x/_y→dim21 OHLCV代理CYQ); B=真实Calculator基础15; C=B+25扩展列(单次bundled).")
    for b in ("main", "dual"):
        a = results[("A_current", b)]
        print(
            f"[{b}] A(proxy)={a:.4f}  B(calc)={b_base[b]:.4f}  Δ(B-A)={b_base[b]-a:+.4f}  "
            f"C(bundle)={c_ic[b]:.4f}  Δ(C-B)={c_ic[b]-b_base[b]:+.4f}"
        )

    print(f"\n======== DedupL2 幸存扩展列 ({len(kept_ext)}/{len(EXTRA_FEATURES)}) ========")
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
            f"{_cell(m['standalone'] if m else None, '{:8.4f}')}"
            f"{_cell(m['drop'] if m else None, '{:+8.4f}')}"
            f"{_cell(m['gain'] if m else None, '{:7.0f}')}"
            f"{_cell(d['standalone'] if d else None, '{:8.4f}')}"
            f"{_cell(d['drop'] if d else None, '{:+8.4f}')}"
            f"{_cell(d['gain'] if d else None, '{:7.0f}')}"
            f"{_cell((m or d)['redun'] if (m or d) else None, '{:7.3f}')}"
        )


if __name__ == "__main__":
    main()
