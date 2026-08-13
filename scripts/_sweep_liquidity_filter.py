"""过滤参数扫描 — 5000万/后20% 是否最优? 预测质量对比 (2026-08-10).

问题: CleaningConfig.min_amount(默认 5000万) × bottom_amount_pct(默认 20%)
是否有更优数值, 能提升 OOS 预测质量 (加权 Rank IC + Top-10 命中率/幅度)?

设计:
  1. 特征在最大宇宙 (min_amount=0, bottom_pct=0, 非停牌) 构建一次并缓存.
     → 所有过滤组合在完全相同的特征值上比较 (过滤只改训练行集, 不改特征值).
  2. 测试段固定 = 最后 60 交易日; 测试行 = 生产过滤行 (amount>=5e7 + E6 rank>0.2)
     → 每组组合在完全相同的测试行上评估, 公平对比.
  3. 每组组合在训练段按该组合过滤 (amount>=min_amount → 子集内按 date 的
     amount rank>bottom_pct, 与 run_train step2→step5 顺序一致), 用生产
     feature_cols + 生产 LGB 超参训练 3d/5d/10d 回归模型 (含半衰期权重+ES早停).
  4. 指标: 加权 Rank IC (LABEL_WEIGHTS) + Top-10 命中率/幅度 (label_pm_10d_net,
     诚实门口径) + 子窗口稳定性 (60d 拆 2×30d).
  5. WORM 落盘: BACKTESTING RESULT/<ts>/filter_sweep.json + 控制台.

用法:
  python scripts/_sweep_liquidity_filter.py            # build(如无缓存)+sweep
  python scripts/_sweep_liquidity_filter.py --build-only
  python scripts/_sweep_liquidity_filter.py --sweep-only
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.dual_track_trainer import (
    LGB_PARAMS_REG,
    NUM_LEAVES_OVERRIDE,
    DualTrackTrainer,
)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LABEL_WEIGHTS
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

OUT_ROOT = os.path.join(str(data_others_path("")), "BACKTESTING RESULT")
CACHE_DIR = os.path.join("data", "_sweep_uni")
CACHE = {
    "main": os.path.join(CACHE_DIR, "main.parquet"),
    "dual": os.path.join(CACHE_DIR, "dual.parquet"),
}
# 断点续跑: 每个组合完成即落盘, 中途崩溃重跑时跳过已完成组合 (内存脆弱, 防丢进度).
PARTIAL = os.path.join(CACHE_DIR, "sweep_partial.json")

# 扫描组合 (min_amount, bottom_amount_pct); (5e7, 0.2) = 生产基线.
COMBOS = [
    (0.0, 0.0),  # 无过滤 (最大宇宙)
    (3e7, 0.2),  # 绝对线更松
    (5e7, 0.2),  # 生产基线
    (1e8, 0.2),  # 绝对线更紧
    (5e7, 0.0),  # 无 E6
    (5e7, 0.1),  # E6 更松
    (5e7, 0.3),  # E6 更紧
    (0.0, 0.2),  # 只 E6
    (1e8, 0.3),  # 双紧
]
TEST_DAYS = 60  # 固定 OOS 测试段 (交易日)
ES_DAYS = 20  # 早停验证段 (仅切分用; 2026-08-11 起扫描训练关早停)
PROD_MIN_AMOUNT = 5e7
# 扫描关早停, 固定树数: es 段 huber loss 平贴常数基线时早停不稳定 (5d 曾坍缩到
# 3 棵树 → 预测每日常数 → RankIC=0), 扫描只需组合间公平, 不需要贴近生产早停.
SWEEP_N_TREES = 200
PROD_BOTTOM_PCT = 0.2

# 市场状态自适应阈值 (状态 → (min_amount, bottom_pct)): 冰点流动性自然低, 固定
# 5000万 砍太深 → 放松多留; 高潮流动性充沛、池子虚胖 → 收紧择优. range = 生产基线.
# 与早前"环境分类器作选股日闸"被否决不同, 这里只调过滤参数, 不切入选清单.
REGIME_MAPS = {
    "regime_ice_loose": {  # 仅冰点放松 (冷市假设单边验证)
        "ice": (3e7, 0.1),
        "range": (5e7, 0.2),
        "hot": (5e7, 0.2),
    },
    "regime_full": {  # 双向自适应: 冰点放松 + 高潮收紧
        "ice": (3e7, 0.1),
        "range": (5e7, 0.2),
        "hot": (1e8, 0.3),
    },
}

REGRESSIONS = ("3d_reg", "5d_reg", "10d_reg")


def _combos_to_str(ma: float, bp: float) -> str:
    return f"{ma:.0e}_{bp:.1f}"


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _install_feature_sort_patch() -> None:
    """扫描专用: 跳过 feature_engine_v35 两处冗余 sort_values(["symbol","date"]).

    build() 开头 (line 234) 已按 (symbol,date) 排序; _apply_per_stock 用
    groupby("symbol") 保序迭代 + per_stock 保行 → 输出本就 (symbol,date) 序;
    _add_time_series_changes 的 sort 只改物理行序, 后续 groupby/transform/merge
    均按 index 对齐, 特征值不变. 这两处 sort 在宽表上深拷贝触发 block
    consolidation — 1.72M 行 × 222 列 float64 需 2.84GiB 连续块, 本机 15.8GB
    物理内存 OOM. 跳过排序不改变任何逐行 (symbol,date) 特征值, 扫描各环节
    (apply_filter/train/evaluate 的 groupby/rank/nlargest) 都按 index 对齐且
    对物理行序不敏感. 生产代码不动, 仅本脚本进程内 patch.
    """
    import app.pipeline1.feature_engine_v35 as fe

    # 构建全程 float32 存储: 1.72M 行 × 576 列 float64 构造块需 8GiB 连续内存,
    # 本机 15.8GB 物理 RAM 会把工作集全部换出 (WS~180MB/commit 15.9GB, 99% 分页,
    # CPU 掉到 <0.7 核 = 死等 page fault). float32 存储把构造块减半到 4GiB,
    # 不再触发 thrash. 最终缓存本就下转 float32, 中间 float32 计算精度损失
    # (~1e-7 相对) 对组合间相对比较无影响.
    orig_alloc = fe._alloc_buffer

    def lean_alloc(series, nrows):
        if series.dtype.kind == "f":
            return np.empty(nrows, dtype="float32")
        return orig_alloc(series, nrows)

    fe._alloc_buffer = lean_alloc

    def lean_apply_per_stock(df, fn):
        # 与生产 _apply_per_stock 逐字节等价, 仅去掉末尾深拷贝排序.
        cols = None
        ref_dtypes = None
        bufs = None
        pos = 0
        nrows = len(df)
        for _, g in df.groupby("symbol"):
            part = fn(g.copy())
            n = len(part)
            if bufs is None:
                cols = list(part.columns)
                ref_dtypes = part.dtypes.to_dict()
                bufs = {c: fe._alloc_buffer(part[c], nrows) for c in cols}
            if pos + n > nrows:
                raise ValueError(
                    f"_apply_per_stock: per_stock 输出 {pos + n} 行 > 输入 "
                    f"{nrows} 行 — 特征函数必须保行"
                )
            for c in cols:
                bufs[c][pos : pos + n] = part[c].to_numpy()
            pos += n
        if bufs is None:
            return df.copy()
        result = pd.DataFrame({c: bufs[c][:pos] for c in cols})
        for col, dtype in ref_dtypes.items():
            cur = result[col].dtype
            if cur == dtype:
                continue
            if np.issubdtype(dtype, np.floating) and cur == np.dtype("float32"):
                continue  # float32 存储即最终精度 (缓存写盘本就下转 float32)
            if np.issubdtype(dtype, np.integer):
                try:
                    finite = bool(np.isfinite(result[col].astype(float)).all())
                except (TypeError, ValueError):
                    finite = False
                if not finite:
                    result[col] = result[col].astype(float)
                    continue
            result[col] = result[col].astype(dtype)
        return result

    def lean_add_time_series_changes(cls, df, inference_cols=None):
        # 与生产 _add_time_series_changes 等价, 仅去掉开头冗余 sort_values 深拷贝.
        whitelist_in_panel = cls._TS_WHITELIST & set(df.columns)
        src_cols = [
            c
            for c in whitelist_in_panel
            if df[c].dtype in ("float64", "float32", "int64", "int32")
            and df[c].isna().mean() < 0.7
        ]
        if inference_cols is not None:
            WINDOWS = (1, 3, 5, 10, 20)
            needed_bases = set()
            for ic in inference_cols:
                for w in WINDOWS:
                    if ic == f"_chg{w}" or ic.endswith(f"_chg{w}"):
                        needed_bases.add(ic[: -len(f"_chg{w}")])
                    if ic == f"_pct_chg{w}" or ic.endswith(f"_pct_chg{w}"):
                        needed_bases.add(ic[: -len(f"_pct_chg{w}")])
            src_cols = [c for c in src_cols if c in needed_bases]
        if not src_cols:
            return df
        WINDOWS = (1, 3, 5, 10, 20)
        for col in src_cols:
            grp = df.groupby("symbol")[col]
            for w in WINDOWS:
                abs_chg = grp.diff(w)
                if abs_chg.notna().sum() > 100:
                    df[f"{col}_chg{w}"] = abs_chg
                pct_chg = grp.pct_change(w, fill_method=None)
                if pct_chg.notna().sum() > 100:
                    df[f"{col}_pct_chg{w}"] = pct_chg
        return df

    fe._apply_per_stock = lean_apply_per_stock
    fe.FeatureEngineV35._add_time_series_changes = classmethod(
        lean_add_time_series_changes
    )
    print("[build] 已 patch 特征引擎冗余排序 (跳过深拷贝)", flush=True)


def build_max_universe_cache(force: bool = False) -> None:
    """在最大宇宙 (min_amount=0, bottom_pct=0) 上构建特征, 逐板块缓存."""
    if not force and all(os.path.exists(CACHE[b]) for b in ("main", "dual")):
        print("[build] 缓存已存在, 跳过 (--force 重建)", flush=True)
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    _install_feature_sort_patch()
    print("读取 V3 面板 (非停牌, 无 amount 过滤) ...", flush=True)
    panel = pq.read_table(
        str(PANEL_V3_PATH), filters=[("is_suspended", "=", False)]
    ).to_pandas()
    cleaner = CleaningPipeline(CleaningConfig(min_amount=0.0, bottom_amount_pct=0.0))
    fe = FeatureEngineV35()
    board_dfs = dict(zip(("main", "dual"), cleaner.run_train(panel)))
    del panel
    gc.collect()
    print(
        f"[build] run_train 最大宇宙: main={len(board_dfs['main']):,} "
        f"/ dual={len(board_dfs['dual']):,}",
        flush=True,
    )
    # 先 dual (962k 行, 峰值低) 后 main (1.72M 行, 峰值高): main 构建时 dual 的
    # float64 切片已释放, 省 ~1GB 常驻, 降低 main 特征构建峰值.
    for board in ("dual", "main"):
        if not force and os.path.exists(CACHE[board]):
            print(f"[build][{board}] 缓存已存在, 跳过 (--force 重建)", flush=True)
            bdf = board_dfs.pop(board)
            del bdf
            gc.collect()
            continue
        bdf = board_dfs.pop(board)
        if bdf is None or len(bdf) == 0:
            print(f"[build][{board}] 空, 跳过", flush=True)
            continue
        use_xrank = board != "main"  # 仅双创加截面排名 (与生产一致)
        # 构建前先 float32 下转: 最大宇宙 1.72M 行 × 222 列 float64 的
        # sort_values 深拷贝需要 2.84GiB 连续块, 本机 15.8GB 物理会 OOM.
        # 扫描内所有组合共用同一套 float32 特征, 对比公平; 写盘时本就 float32.
        for c in bdf.columns:
            if c not in ("symbol", "date", "board") and pd.api.types.is_numeric_dtype(
                bdf[c]
            ):
                bdf[c] = bdf[c].astype("float32")
        df = prepare_board_frame(
            bdf, fe, None, cross_sectional_rank=use_xrank, registry=None
        )
        del bdf
        gc.collect()
        # float32 下转省内存/磁盘 (symbol/date/board 保持原 dtype)
        for c in df.columns:
            if c not in ("symbol", "date", "board") and pd.api.types.is_numeric_dtype(
                df[c]
            ):
                df[c] = df[c].astype("float32")
        df.to_parquet(CACHE[board], index=False)
        print(
            f"[build][{board}] 缓存已写 {CACHE[board]} rows={len(df):,} "
            f"cols={df.shape[1]:,} latest={df['date'].max():%Y-%m-%d}",
            flush=True,
        )
        del df
        gc.collect()
    del board_dfs, fe, cleaner
    gc.collect()


def build_regime_cache() -> dict:
    """逐日市场情绪状态 (ice/range/hot) — 无前视.

    用 V3 面板 date/board/pre_close/close/amount 按日聚合涨跌停家数, 对每日用
    近 HIST_WINDOW 日基线复刻 market_environment.classify_market_state (冰点优先
    判). 状态只依赖当日收盘+过去数据, 过滤训练行按该行所在日期状态判定, 标签为
    未来收益 → 无 look-ahead. 落盘 CACHE_DIR/regime.parquet, 返回 {date: state}.
    """
    import pyarrow.parquet as pq

    from app.pipeline1.market_environment import (
        _HOT_RATIO,
        _HOT_UP_PCT,
        _ICE_RATIO,
        _ICE_UP_PCT,
        HIST_WINDOW,
        _limit_mask,
    )

    out = os.path.join(CACHE_DIR, "regime.parquet")
    if os.path.exists(out):
        rdf = pd.read_parquet(out)
        return dict(zip(pd.to_datetime(rdf["date"]), rdf["state"]))
    os.makedirs(CACHE_DIR, exist_ok=True)
    tab = pq.read_table(
        str(PANEL_V3_PATH),
        columns=["date", "board", "pre_close", "close", "amount"],
        filters=[("is_suspended", "=", False)],
    )
    panel = tab.to_pandas()
    up, dn = _limit_mask(panel)
    daily = (
        panel[["date"]]
        .assign(count_limit_up=up.astype(int), count_limit_down=dn.astype(int))
        .groupby("date", as_index=False)
        .agg(
            count_limit_up=("count_limit_up", "sum"),
            count_limit_down=("count_limit_down", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    up_base = daily["count_limit_up"].rolling(HIST_WINDOW, min_periods=20).mean()
    has_hist = up_base.notna()
    up_pct = daily["count_limit_up"] / up_base.where(has_hist)
    ratio = (daily["count_limit_up"] + 1) / (daily["count_limit_down"] + 1)
    state = pd.Series("range", index=daily.index, dtype=object)
    with_hist_ice = has_hist & ((up_pct < _ICE_UP_PCT) | (ratio < _ICE_RATIO))
    state[with_hist_ice] = "ice"
    state[has_hist & ~with_hist_ice & (up_pct > _HOT_UP_PCT) & (ratio > _HOT_RATIO)] = (
        "hot"
    )
    no_hist_ice = ~has_hist & (ratio < _ICE_RATIO)
    state[no_hist_ice] = "ice"
    state[~has_hist & ~no_hist_ice & (ratio > _HOT_RATIO)] = "hot"
    daily["state"] = state
    rdf = daily[["date", "state"]]
    rdf.to_parquet(out, index=False)
    last60 = rdf.tail(TEST_DAYS)
    print(
        "[regime] 全期分布:",
        {k: int(v) for k, v in rdf["state"].value_counts().items()},
        "| 末60日分布:",
        {k: int(v) for k, v in last60["state"].value_counts().items()},
        flush=True,
    )
    del panel, daily
    gc.collect()
    return dict(zip(pd.to_datetime(rdf["date"]), rdf["state"]))


def apply_filter(
    df: pd.DataFrame,
    min_amount: float,
    bottom_pct: float,
    regime_map: dict | None = None,
    regime: dict | None = None,
) -> pd.DataFrame:
    """复刻 run_train step2(amount>=min_amount) → step5(E6 子集内 rank>bottom_pct).

    布尔索引已产出新帧 (自带拷贝), 不再 .copy() 二次深拷贝 — 1.25M 行 × 320 列
    的重复拷贝正是组合 5 时 2.54GiB 分配失败的元凶之一.

    regime_map 非空 → 逐日按市场状态套用不同阈值 (regime = {date: state}),
    训练/测试端到端自适应 (部署口径).
    """
    if regime_map is None:
        out = df[df["amount"] >= min_amount]
        if bottom_pct > 0:
            rank = out.groupby("date")["amount"].rank(pct=True)
            out = out[rank > bottom_pct]
        return out
    states = df["date"].map(regime).fillna("range")
    keep = pd.Series(False, index=df.index)
    for state, (ma, bp) in regime_map.items():
        m = states == state
        amt = df.loc[m, "amount"]
        if bp > 0:
            r = amt.groupby(df.loc[m, "date"]).rank(pct=True)
            keep[m] = (amt >= ma) & (r > bp)
        else:
            keep[m] = amt >= ma
    return df[keep]


def _heapmin() -> None:
    """强制把 Python/CRT 堆的空闲块交还 OS, 防止跨组合 commit 爬升 (Windows)."""
    try:
        import ctypes

        ctypes.CDLL("msvcrt.dll")._heapmin()
    except Exception:
        pass


def train_reg(board: str, kind: str, train_df, es_df, cols) -> object:
    import lightgbm as lgb

    h = kind.split("d")[0]
    label = f"label_pm_{h}d_net"
    cols_p = [c for c in cols if c in train_df.columns]
    # 单布尔掩码行选 (label 非空 & 非停牌) → 紧凑 float32 X, 不产全宽中间帧:
    # dropna/risk_filter 会各造 ~1.65GiB 拷贝, 跨组合把本机 8GiB 空闲顶穿.
    m = train_df[label].notna().values
    if "is_suspended" in train_df.columns:
        m &= ~train_df["is_suspended"].astype(bool).values
    X = np.ascontiguousarray(train_df[cols_p].values[m], dtype=np.float32)
    # 逐列填 NaN/±inf: 掩码每列仅 ~1.2MB, 避开一次性 (1.25M×316 bool≈376MB) 分配.
    for i in range(X.shape[1]):
        np.nan_to_num(X[:, i], copy=False)
    y = train_df[label].values[m].astype(np.float64, copy=False)
    w = DualTrackTrainer.time_weights(train_df)[m].astype(np.float64, copy=False)
    del train_df, es_df
    gc.collect()
    params = dict(LGB_PARAMS_REG)
    nl = NUM_LEAVES_OVERRIDE.get((board, kind))
    if nl is not None:
        params["num_leaves"] = nl
    params["n_estimators"] = SWEEP_N_TREES
    params["force_row_wise"] = True  # 316 特征宽表行式直方图省内存 (列式开销大)
    params["num_threads"] = 4  # 减直方图线程缓冲, 防提交压力
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, sample_weight=w)
    return model


def _rank_ic(sub: pd.DataFrame, pred_col: str, label: str) -> float:
    from app.pipeline1.ic_screener import ICScreener

    if len(sub) < 30:
        return 0.0
    return ICScreener.rank_ic(sub.rename(columns={pred_col: "score"}), "score", label)


def evaluate_combo(test_rows: pd.DataFrame, models: dict, cols: list[str]) -> dict:
    """在固定测试行上评估: 加权 Rank IC + Top-10 (10d 诚实门口径) + 子窗口."""

    test = test_rows.copy()
    for c in cols:
        if c in test.columns and pd.api.types.is_numeric_dtype(test[c]):
            test[c] = test[c].astype("float32")
    ics = {}
    for kind in REGRESSIONS:
        h = kind.split("d")[0]
        label = f"label_pm_{h}d_net"
        sub = test.dropna(subset=[label])
        if len(sub) < 30:
            ics[kind] = 0.0
            continue
        sub = sub.copy()
        sub["_pred"] = models[kind].predict(np.nan_to_num(sub[cols].values, nan=0.0))
        ics[kind] = _rank_ic(sub, "_pred", label)
    total_w = sum(LABEL_WEIGHTS.values())
    weighted_ic = (
        sum(LABEL_WEIGHTS[int(k.split("d")[0])] * ics[k] for k in REGRESSIONS) / total_w
    )
    # Top-10 (10d 诚实门口径): 每日期按 10d 预测取前 10, 量 label_pm_10d_net.
    sub10 = test.dropna(subset=["label_pm_10d_net"])
    top_rows = []
    if len(sub10) >= 10:
        sub10 = sub10.copy()
        sub10["_pred10"] = models["10d_reg"].predict(
            np.nan_to_num(sub10[cols].values, nan=0.0)
        )
        for _, g in sub10.groupby("date"):
            top_rows.append(g.nlargest(10, "_pred10"))
    top = pd.concat(top_rows) if top_rows else pd.DataFrame()
    top10 = {"n_days": len(top_rows), "n": len(top)}
    if len(top):
        top10["winrate"] = float((top["label_pm_10d_net"] > 0).mean())
        top10["mag"] = float(top["label_pm_10d_net"].mean())
    else:
        top10["winrate"], top10["mag"] = 0.0, 0.0
    # 子窗口稳定性: 测试段 60d 拆 2×30d, 各算加权 IC.
    dates = sorted(test["date"].unique())
    half = len(dates) // 2
    subwin = {}
    for name, wd in (("w1", dates[:half]), ("w2", dates[half:])):
        seg = test[test["date"].isin(wd)]
        seg_ics = {}
        for kind in REGRESSIONS:
            h = kind.split("d")[0]
            label = f"label_pm_{h}d_net"
            s = seg.dropna(subset=[label]).copy()
            if len(s) < 30:
                seg_ics[kind] = 0.0
                continue
            s["_pred"] = models[kind].predict(np.nan_to_num(s[cols].values, nan=0.0))
            seg_ics[kind] = _rank_ic(s, "_pred", label)
        subwin[name] = (
            sum(LABEL_WEIGHTS[int(k.split("d")[0])] * seg_ics[k] for k in REGRESSIONS)
            / total_w
        )
    return {
        "ics": {k: round(v, 4) for k, v in ics.items()},
        "weighted_ic": round(weighted_ic, 4),
        "top10": top10,
        "subwin": {k: round(v, 4) for k, v in subwin.items()},
    }


def _load_partial() -> dict:
    """读断点文件; 结构 = {board: {combos: {key: {...}}}} (无 "boards" 包装)."""
    import json as _json

    if not os.path.exists(PARTIAL):
        return {}
    try:
        with open(PARTIAL, encoding="utf-8") as fh:
            return _json.load(fh)
    except Exception:
        return {}


def _read_pq(board: str, need: list[str], dates, amount_min=None) -> pd.DataFrame:
    """pyarrow 按 日期(+金额下限) 过滤读切片 — 每组合独立读, 不常驻全 board.

    本机物理 15.8GB / 空闲 ~8GB 且 pagefile 未启用, 单进程内跨组合 LGBM 的 C++
    堆不释放、碎片累积, 第 5 个组合就 OOM. 改为每组合子进程 + 各自 pyarrow 切片,
    峰值 ≈ 切片帧 + X 视图 + LGB 转置 (~5GB), 进程退出即整体归还.
    """
    flt = [("date", "in", [pd.Timestamp(d).to_pydatetime() for d in dates])]
    if amount_min is not None:
        flt.append(("amount", ">=", amount_min))
    return pq.read_table(CACHE[board], columns=need, filters=flt).to_pandas()


def _combo_worker(board: str, key: str, ma, bp, rmap: dict | None) -> int:
    """单个组合端到端 (子进程内): 读切片 → 训练 3 视界 → 评估 → 写 partial.json."""
    import json as _json

    from app.pipeline1.dual_track_trainer import DualTrackTrainer

    t = DualTrackTrainer.load(f"models/pipeline1/{board}_20260810.pkl")
    feat_cols = t["feature_cols"]
    del t
    gc.collect()

    schema = pq.ParquetFile(CACHE[board]).schema.names
    cols = [c for c in feat_cols if c in schema]
    labels = [f"label_pm_{h}d_net" for h in (3, 5, 10)]
    need = [c for c in ["date", "amount"] + cols + labels if c in schema]

    dcol = pq.read_table(CACHE[board], columns=["date"]).to_pandas()["date"]
    dates = sorted(pd.unique(dcol))
    del dcol
    gc.collect()
    test_dates = dates[-TEST_DAYS:]
    es_dates = dates[-TEST_DAYS - ES_DAYS : -TEST_DAYS]
    train_dates = dates[: -TEST_DAYS - ES_DAYS]

    t0 = time.time()
    if rmap is None:
        # 固定组合: 训练行按 (date, amount>=ma) 过滤读入, 测试行 = 生产过滤行.
        tr = apply_filter(
            _read_pq(board, need, test_dates), PROD_MIN_AMOUNT, PROD_BOTTOM_PCT
        )
        train = _read_pq(board, need, train_dates, ma)
        if bp > 0:
            r = train.groupby("date")["amount"].rank(pct=True)
            train = train[r > bp]
        es = _read_pq(board, need, es_dates, ma)
        if bp > 0:
            re = es.groupby("date")["amount"].rank(pct=True)
            es = es[re > bp]
    else:
        # regime 组合: 读全部训练/验证/测试日行, 端到端按市场状态套用阈值.
        regime = build_regime_cache()
        all_train = _read_pq(board, need, train_dates)
        all_es = _read_pq(board, need, es_dates)
        all_test = _read_pq(board, need, test_dates)
        train = apply_filter(all_train, None, None, rmap, regime)
        es = apply_filter(all_es, None, None, rmap, regime)
        tr = apply_filter(all_test, None, None, rmap, regime)
        del all_train, all_es, all_test
        gc.collect()

    models = {}
    for kind in REGRESSIONS:
        models[kind] = train_reg(board, kind, train, es, cols)
    ev = evaluate_combo(tr, models, cols)
    entry: dict = {
        "n_train": int(len(train)),
        "n_test": int(len(tr)),
        "secs": round(time.time() - t0, 1),
        **ev,
    }
    if rmap is None:
        entry["min_amount"] = ma
        entry["bottom_pct"] = bp
    else:
        entry["regime_map"] = {k: list(v) for k, v in rmap.items()}
    partial = _load_partial()
    partial.setdefault(board, {}).setdefault("combos", {})[key] = entry
    with open(PARTIAL, "w", encoding="utf-8") as fh:
        _json.dump(partial, fh, indent=2, ensure_ascii=False)
    print(
        f"[sweep][{board}] {key} n_train={len(train):,} "
        f"wIC={ev['weighted_ic']:.4f} top10wr={ev['top10']['winrate']:.1%} "
        f"mag={ev['top10']['mag']:+.2%} [{time.time() - t0:.0f}s]",
        flush=True,
    )
    return 0


def sweep() -> dict:
    """编排: 每个待跑组合在独立子进程里跑 (干净内存, 免跨组合碎片 OOM)."""
    import subprocess

    results = _load_partial()
    done_all = [k for b in results for k in results[b].get("combos", {})]
    print(f"[sweep] 载入部分结果, 跳过已完成组合: {done_all}", flush=True)
    script = os.path.abspath(__file__)
    for board in ("main", "dual"):
        combo_res = results.get(board, {}).get("combos", {})
        specs = [(_combos_to_str(ma, bp), ma, bp, None) for ma, bp in COMBOS]
        specs += [(name, None, None, rmap) for name, rmap in REGIME_MAPS.items()]
        for key, ma, bp, rmap in specs:
            if key in combo_res:
                continue
            print(f"[sweep][{board}] {key} 开始 (子进程, 干净内存)", flush=True)
            rmap_json = json.dumps(rmap) if rmap else ""
            r = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--combo",
                    board,
                    key,
                    "None" if ma is None else str(ma),
                    "None" if bp is None else str(bp),
                    rmap_json,
                ],
                capture_output=True,
                text=True,
            )
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            if r.returncode != 0:
                print(
                    f"[sweep][{board}] {key} FAILED rc={r.returncode} — 中断, "
                    f"已完成组合保留在断点文件",
                    flush=True,
                )
                return results
            # 子进程已写盘, 重读让跳过判断与结果视图同步
            results = _load_partial()
            combo_res = results.get(board, {}).get("combos", {})
    return results


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--combo",
        nargs=5,
        metavar=("BOARD", "KEY", "MA", "BP", "RMAP_JSON"),
        help="单组合子进程入口 (由 sweep 编排调用)",
    )
    args = ap.parse_args()

    if args.combo:
        board, key, ma_s, bp_s, rmap_s = args.combo
        ma = float(ma_s) if ma_s != "None" else None
        bp = float(bp_s) if bp_s != "None" else None
        rmap = json.loads(rmap_s) if rmap_s else None
        return _combo_worker(board, key, ma, bp, rmap)

    if not args.sweep_only:
        build_max_universe_cache(force=args.force)
    if args.build_only:
        print("[done] build-only")
        return 0

    results = sweep()

    ts = _ts()
    out_dir = os.path.join(OUT_ROOT, ts)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "meta": {
            "experiment": "liquidity_filter_sweep",
            "built": ts,
            "test_days": TEST_DAYS,
            "es_days": ES_DAYS,
            "prod_min_amount": PROD_MIN_AMOUNT,
            "prod_bottom_pct": PROD_BOTTOM_PCT,
            "combos": [_combos_to_str(ma, bp) for ma, bp in COMBOS],
            "regime_maps": REGIME_MAPS,
        },
        "boards": results,
    }
    out_json = os.path.join(out_dir, "filter_sweep.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    if os.path.exists(PARTIAL):
        os.remove(PARTIAL)  # 完整落盘后清除断点
    print(f"[done] 结果已落盘 {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
