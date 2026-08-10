"""模块绩效追踪 (只读) — 按模块 ID 追踪每日清单的已实现收益.

数据流:
  STOCK_LIST_DIR 预测文件 (日期+模块 双标识) → 统一清单 (模块标签保留)
  → 与 V3 面板 close_hfq 对齐 → 已实现 T+2/3/5/10 close-to-close 收益
  → 按 模块 / Top-N 短名单 / 全市场底稿 分档聚合.

口径 (对齐 2026-08-07 d10 c2c 定案): 幅度按 close-to-close 校准 (非 MFE).
  - parallel 短名单/底稿的 date 列 = 选股日 (优先); legacy 无 date 列 → 文件名日期.
  - 已实现收益仅在面板含 T+h 收盘价时计算; 数据未成熟 → NaN, 不纳入统计.
  - 全部纯函数, 页面只做渲染, 不重算训练.

聚合维:
  module_id = family·module (如 parallel·M20260806__D20260806r_q2345 / legacy·20260806x_r_q2345).
  scope    = 交付短名单 (legacy/parallel/slow_bull) vs 全市场底稿 (*_raw).
  top_k    = 每 (module_id, date) 取前 N (无 rk 时按 score 降序; 都没有则保留全部).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import STOCK_LIST_DIR

from . import data_service as ds

logger = logging.getLogger(__name__)

HORIZONS = ("3d", "5d", "10d")
_SCOPE_DELIVERED = ("legacy", "parallel", "slow_bull")
_SCOPE_RAW = ("legacy_raw", "parallel_raw")

_UNIFIED_COLS = list(ds._UNIFIED_COLS)


# ───────────────────────── 加载: 清单 → 统一行 ─────────────────────────
def _to_iso_date(v) -> str:
    """归一化日期到 YYYY-MM-DD (兼容 20260806 / 2026-08-06 / Timestamp)."""
    s = str(v).strip()[:10].replace("-", "")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return str(v).strip()[:10]


def _pick_date(df: pd.DataFrame, date_compact: str) -> str:
    """选股日: 优先文件内 date 列 (选股日), 缺失回退文件名日期."""
    if "date" in df.columns and df["date"].notna().any():
        return _to_iso_date(df["date"].iloc[0])
    return _to_iso_date(date_compact)


def load_module_picks(list_dir: str = STOCK_LIST_DIR) -> pd.DataFrame:
    """扫描预测文件 → 统一清单 (含模块标签 + 选股日 + 已实现收益空列).

    Returns:
        DataFrame (date, symbol, family, module, board, system, rk, score,
                   gain_3d/5d/10d, prob_*, module_id) — 无文件 → 空.
    """
    frames: list[pd.DataFrame] = []
    for info in ds.list_prediction_files(list_dir):
        try:
            df = pd.read_csv(info["path"], dtype={"symbol": str})
        except Exception:
            logger.warning("预测文件读取失败 (跳过): %s", info["path"], exc_info=True)
            continue
        if df is None or df.empty or "symbol" not in df.columns:
            continue
        norm = ds._normalize_pred_rows(info["family"], info["date"], info["module"], df)
        if norm.empty:
            continue
        # 选股日优先内部 date 列; 无 rk 列的清单按文件行序赋隐式排名
        norm["date"] = _pick_date(df, info["date"])
        if "rk" not in df.columns or df["rk"].isna().all():
            norm["rk"] = np.arange(1, len(norm) + 1)
        # 统一数值列 float64, 避免 concat 全 NA 列触发 FutureWarning
        for c in (
            ("score", "rk")
            + tuple(f"gain_{h}" for h in HORIZONS)
            + tuple(f"prob_{h}" for h in HORIZONS)
        ):
            norm[c] = pd.to_numeric(norm[c], errors="coerce")
        frames.append(norm)
    if not frames:
        return pd.DataFrame(columns=_UNIFIED_COLS + ["module_id"])
    out = pd.concat(frames, ignore_index=True)
    out["module_id"] = out["family"] + "·" + out["module"].astype(str)
    for c in (
        ("rk", "score")
        + tuple(f"gain_{h}" for h in HORIZONS)
        + tuple(f"prob_{h}" for h in HORIZONS)
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[_UNIFIED_COLS + ["module_id"]].copy()


# ───────────────────────── 已实现收益 (对齐面板) ─────────────────────────
def compute_realized_returns(picks: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """给清单补已实现 close-to-close 收益 real_3d/5d/10d.

    panel 需含 date(datetime)/symbol/close_hfq. 逐股向量化 (unstack + shift),
    无未来价 (数据未成熟/停牌缺行) → NaN 剔除.
    """
    if picks is None or picks.empty or panel is None or panel.empty:
        return picks
    need = {"date", "symbol", "close_hfq"}
    if not need.issubset(panel.columns):
        logger.warning("面板缺列 %s, 无法算已实现收益", need - set(panel.columns))
        return picks
    piv = (
        panel.groupby(["date", "symbol"])["close_hfq"]
        .last()
        .unstack(fill_value=np.nan)
        .sort_index()
    )
    pv = piv.values
    date_pos = {pd.Timestamp(d).date().isoformat(): i for i, d in enumerate(piv.index)}
    col_map = {s: i for i, s in enumerate(piv.columns)}

    p = picks.copy()
    pos = p["date"].map(date_pos)
    col = p["symbol"].map(col_map)
    valid = pos.notna().values & col.notna().values
    pos_v = pos.fillna(0).astype(int).values
    col_v = col.fillna(0).astype(int).values

    c0 = pv[pos_v, col_v]
    denom_ok = ~np.isnan(c0) & (c0 != 0)
    for h in HORIZONS:
        n = int(h[:-1])  # "10d" → 10 (勿用 h[0]: 会把 10d 当成 1d)
        fwd = np.full_like(pv, np.nan)
        if n < len(pv):
            fwd[:-n] = pv[n:]
        ch = fwd[pos_v, col_v]
        ret = np.full(len(p), np.nan)
        ok = denom_ok & ~np.isnan(ch)
        ret[ok] = ch[ok] / c0[ok] - 1.0
        ret[~valid] = np.nan
        p[f"real_{h}"] = ret
    return p


# ───────────────────────── 分档聚合 ─────────────────────────
def recent_module_ids(picks: pd.DataFrame, n: int = 5) -> list[str]:
    """最近活跃的 n 个模块 (按最后交付日降序; 同日在 module_id 倒序).

    用于看板模型下拉: 取最近交付过的模型版本, 供用户选择回看其绩效.
    """
    if picks is None or picks.empty or "module_id" not in picks.columns:
        return []
    rec = (
        picks.groupby("module_id")["date"]
        .max()
        .reset_index()
        .sort_values(["date", "module_id"], ascending=[False, False])
    )
    return rec["module_id"].head(n).tolist()


def recent_module_ids_per_model(picks: pd.DataFrame, n: int = 5) -> list[str]:
    """每个模型族 (family) 各取最近 n 个版本, 拼成下拉选项.

    module_id = family·module (family=legacy/parallel/slow_bull, module=版本);
    以 family 为模型, module 为版本, 每族内按最后交付日取最新 n 个.
    """
    if picks is None or picks.empty or "module_id" not in picks.columns:
        return []
    rec = (
        picks.groupby(["module_id", "family"])["date"]
        .max()
        .reset_index()
        .sort_values(["family", "date", "module_id"], ascending=[True, False, False])
    )
    out: list[str] = []
    for _fam, grp in rec.groupby("family"):
        out.extend(grp["module_id"].head(n).tolist())
    return out


def filter_scope(picks: pd.DataFrame, scope: str) -> pd.DataFrame:
    """按数据源范围过滤: 交付短名单 / 全市场底稿 / 全部."""
    if picks is None or picks.empty or scope in ("全部", ""):
        return picks
    fams = _SCOPE_DELIVERED if scope == "交付短名单" else _SCOPE_RAW
    return picks[picks["family"].isin(fams)].copy()


def top_k(picks: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """每 (module_id, date) 取前 N: 有 rk 用 rk, 否则按 score 降序, 都没有 → 保留全部."""
    if picks is None or picks.empty or not n:
        return picks
    n = int(n)
    if "rk" in picks.columns and picks["rk"].notna().any():
        ordered = picks.sort_values("rk", na_position="last")
    elif "score" in picks.columns and picks["score"].notna().any():
        ordered = picks.sort_values("score", ascending=False)
    else:
        ordered = picks
    return ordered.groupby(["module_id", "date"]).head(n)


def perf_summary(perf: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """模块绩效汇总 (命中率/平均/中位/分位)."""
    col = f"real_{horizon}"
    if perf is None or perf.empty or col not in perf.columns:
        return pd.DataFrame(
            columns=["module_id", "n", "hit_rate", "mean", "median", "p10", "p90"]
        )
    r = perf[["module_id", col]].dropna()
    if r.empty:
        return pd.DataFrame(
            columns=["module_id", "n", "hit_rate", "mean", "median", "p10", "p90"]
        )
    g = r.groupby("module_id")[col]
    rows = {
        "module_id": g.size().index.tolist(),
        "n": g.size().values,
        "hit_rate": g.apply(lambda x: float((x > 0).mean())).values,
        "mean": g.mean().values,
        "median": g.median().values,
        "p10": g.quantile(0.10).values,
        "p90": g.quantile(0.90).values,
    }
    return (
        pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    )


def daily_mean_return(perf: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """每 (module_id, date) 当日平均已实现收益 → 时间序列 (绘图/累积用)."""
    col = f"real_{horizon}"
    if perf is None or perf.empty or col not in perf.columns:
        return pd.DataFrame(columns=["module_id", "date", "mean_ret", "n"])
    d = (
        perf.dropna(subset=[col])
        .groupby(["module_id", "date"])[col]
        .agg(["mean", "count"])
        .reset_index()
    )
    d.columns = ["module_id", "date", "mean_ret", "n"]
    return d.sort_values(["module_id", "date"]).reset_index(drop=True)


# ───────────────────────── 预测质量检查 (预测 vs 实际) ─────────────────────────
_QUALITY_COLS = ["n", "dates", "pred_mean", "real_mean", "bias", "hit_rate", "rank_ic"]


def _daily_rank_ic(g: pd.DataFrame, gcol: str, rcol: str, min_n: int = 5) -> float:
    """逐日横截面 Spearman(预测涨幅, 实际收益) 的平均; 单日样本 < min_n → 不计."""
    ics = []
    for _d, gg in g.groupby("date"):
        pair = gg[[gcol, rcol]].dropna()
        if len(pair) >= min_n:
            v = pair.corr(method="spearman").iloc[0, 1]
            if not np.isnan(v):
                ics.append(v)
    return float(np.nanmean(ics)) if ics else np.nan


def quality_by_family(perf: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """系统 (family) 级预测质量汇总: 预测 vs 实际.

    perf 由 compute_realized_returns 产出 (含 gain_{h}/real_{h}/date/family).
    bias = 实际均值 − 预测均值 (>0 低估 / <0 高估); rank_ic = 逐日横截面 Spearman 平均.
    无预测涨幅列的系统 (slow_bull 风控观察池) 仅实际表现可用 (n/hit_rate/real_mean),
    预测相关列 (pred_mean/bias/rank_ic) 置 NaN.
    """
    gcol, rcol = f"gain_{horizon}", f"real_{horizon}"
    cols = ["family"] + _QUALITY_COLS
    if perf is None or perf.empty or rcol not in perf.columns:
        return pd.DataFrame(columns=cols)
    r = perf.dropna(subset=[rcol]).copy()
    if r.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for fam, g in r.groupby("family"):
        has_g = gcol in g.columns and g[gcol].notna().any()
        real_mean = g[rcol].mean()
        pred_mean = g[gcol].mean() if has_g else np.nan
        rows.append(
            {
                "family": fam,
                "n": len(g),
                "dates": g["date"].nunique(),
                "pred_mean": pred_mean,
                "real_mean": real_mean,
                "bias": real_mean - pred_mean if has_g else np.nan,
                "hit_rate": float((g[rcol] > 0).mean()),
                "rank_ic": _daily_rank_ic(g, gcol, rcol) if has_g else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("real_mean", ascending=False).reset_index(drop=True)


def quality_by_module(perf: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """模块级预测质量汇总 (module_id = family·version), 含 family 标签."""
    gcol, rcol = f"gain_{horizon}", f"real_{horizon}"
    cols = ["module_id", "family"] + _QUALITY_COLS
    if perf is None or perf.empty or rcol not in perf.columns:
        return pd.DataFrame(columns=cols)
    r = perf.dropna(subset=[rcol]).copy()
    if r.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for mid, g in r.groupby("module_id"):
        has_g = gcol in g.columns and g[gcol].notna().any()
        real_mean = g[rcol].mean()
        pred_mean = g[gcol].mean() if has_g else np.nan
        rows.append(
            {
                "module_id": mid,
                "family": g["family"].iloc[0] if "family" in g.columns else "",
                "n": len(g),
                "dates": g["date"].nunique(),
                "pred_mean": pred_mean,
                "real_mean": real_mean,
                "bias": real_mean - pred_mean if has_g else np.nan,
                "hit_rate": float((g[rcol] > 0).mean()),
                "rank_ic": _daily_rank_ic(g, gcol, rcol) if has_g else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("real_mean", ascending=False).reset_index(drop=True)


def module_daily_quality(perf: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """单模块逐日质量 (仅已成熟日期): n / 预测均值 / 实际均值 / 命中率.

    perf 应为已按 module_id 过滤的行; 返回每个选股日一组, 供预测-vs-实际图.
    无预测涨幅列 (slow_bull) → pred_mean 为 NaN.
    """
    gcol, rcol = f"gain_{horizon}", f"real_{horizon}"
    cols = ["date", "n", "pred_mean", "real_mean", "hit_rate"]
    if perf is None or perf.empty or rcol not in perf.columns:
        return pd.DataFrame(columns=cols)
    r = perf.dropna(subset=[rcol]).copy()
    if r.empty:
        return pd.DataFrame(columns=cols)
    agg = {"n": (rcol, "size"), "real_mean": (rcol, "mean")}
    if gcol in r.columns and r[gcol].notna().any():
        agg["pred_mean"] = (gcol, "mean")
    daily = r.groupby("date").agg(**agg).reset_index()
    if "pred_mean" not in daily.columns:
        daily["pred_mean"] = np.nan
    up = (
        r.assign(_up=(r[rcol] > 0).astype(float))
        .groupby("date")["_up"]
        .mean()
        .reset_index()
        .rename(columns={"_up": "hit_rate"})
    )
    daily = daily.merge(up, on="date", how="left")
    return daily[cols].sort_values("date").reset_index(drop=True)


def calibration_table(
    perf: pd.DataFrame, horizon: str, bins: list[float] | None = None
) -> pd.DataFrame:
    """概率校准: 预测概率分桶 → 实际命中率/实际均值.

    理想校准下桶内命中率 ≈ 桶概率; 显著偏低 → 概率高估.
    """
    pcol, rcol = f"prob_{horizon}", f"real_{horizon}"
    cols = ["prob_bin", "n", "hit_rate", "real_mean"]
    if perf is None or perf.empty or not {pcol, rcol}.issubset(perf.columns):
        return pd.DataFrame(columns=cols)
    r = perf.dropna(subset=[pcol, rcol]).copy()
    if r.empty:
        return pd.DataFrame(columns=cols)
    edges = bins or [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    r["_bin"] = pd.cut(r[pcol], edges, right=False)
    g = r.groupby("_bin", observed=False)[rcol]
    out = pd.DataFrame(
        {
            "prob_bin": [str(b) for b in g.groups],
            "n": g.size().values,
            "hit_rate": g.apply(lambda x: float((x > 0).mean())).values,
            "real_mean": g.mean().values,
        }
    )
    return out[out["n"] > 0].reset_index(drop=True)
