# -*- coding: utf-8 -*-
"""训练/推理面板装配 (PIPELINE1 生产数据入口)
=====================================================
把 ``DataSupplyChain.backfill_ohlcv`` 的原始 OHLCV 面板补齐为
cleaning/feature/label 全链路可用的标准面板:

- ``board``        : 缺失时按代码前缀推导 (复用 cleaning_pipeline.board_of)
- ``is_suspended`` : 默认 False (停牌日天然无 bar, 不影响训练)
- ``industry``     : industry_map 提供, 缺失默认 "UNKNOWN"
- ``free_float_turnover_rate`` : 缺失时回退 turnover_rate

默认数据深度: 最近 3 年 akshare 日线 (用户 2026-07-26 裁决);
[B11] 深度不足 1250 交易日时训练窗口自动降为 540 日过渡 (见 dual_track_trainer).
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from config.settings import data_others_path

from app.core.config_loader import load_config

from .cleaning_pipeline import board_of
from .data_supply import DataSupplyChain, _ak_call

logger = logging.getLogger(__name__)

_PB_CFG = load_config("data_pipeline_config").get("panel_builder", {})

DEFAULT_YEARS = int(_PB_CFG.get("default_years", 3))
PANEL_CACHE_DIR = _PB_CFG.get("panel_cache_dir", os.path.join("data", "processed"))
_ENRICH_WORKERS = int(
    os.environ.get("ENRICH_WORKERS", _PB_CFG.get("enrich_workers", 4))
)
_META_FETCH_SLEEP = float(_PB_CFG.get("meta_fetch_sleep", 0.3))
_META_MAX_CONSECUTIVE_FAIL = int(_PB_CFG.get("meta_max_consecutive_fail", 3))
_PROGRESS_EVERY_N = int(_PB_CFG.get("progress_every_n", 10))
_PROGRESS_FILE = _PB_CFG.get("progress_file", "data/enrich_progress.txt")
_DEFAULT_ALT_SOURCES = list(
    _PB_CFG.get(
        "default_alt_sources",
        [
            "lhb",
            "holdertrade",
            "sector_index",
            "margin",
            "fina_indicator",
            "holdernumber",
            "daily_basic",
            "stk_limit",
            "cyq_tushare",
        ],
    )
)


def load_or_fetch_meta(
    cache_dir: str = PANEL_CACHE_DIR, refresh: bool = False
) -> tuple[dict[str, str], dict[str, str]]:
    """股票元数据 (industry_map, name_map): 东财行业板块成分 + 现货名称.

    industry_map 用于行业中性化/行业集中度上限 (缺失时全 UNKNOWN → 清单 ≤4 只);
    name_map 用于 ST 标记. 缓存 stock_meta_<YYYYMMDD>.json (WORM, 当日有效).
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = str(
        data_others_path(
            os.path.join(cache_dir, f"stock_meta_{datetime.now():%Y%m%d}.json")
        )
    )
    if not refresh and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
            return meta["industry_map"], meta["name_map"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("stock_meta cache corrupted (%s): refetching", exc)

    import akshare as ak

    spot = _ak_call(ak.stock_zh_a_spot_em)
    name_map = dict(zip(spot["代码"].astype(str).str[-6:], spot["名称"].astype(str)))

    boards = _ak_call(ak.stock_board_industry_name_em)
    industry_map: dict[str, str] = {}
    consecutive_fail = 0
    for i, board in enumerate(boards["板块名称"].astype(str)):
        try:
            cons = _ak_call(ak.stock_board_industry_cons_em, symbol=board)
            for code in cons["代码"].astype(str).str[-6:]:
                industry_map[code] = board
            consecutive_fail = 0
        except Exception as exc:  # 单板块失败跳过; 连续失败=IP 被封, 放弃本次
            consecutive_fail += 1
            logger.warning("行业板块 %s 成分拉取失败: %s", board, exc)
            if consecutive_fail >= _META_MAX_CONSECUTIVE_FAIL:
                raise RuntimeError(
                    f"东财连续 {consecutive_fail} 板块拉取失败 (疑似封 IP), 放弃元数据"
                ) from exc
        time.sleep(_META_FETCH_SLEEP)
        if (i + 1) % 30 == 0:
            logger.info("行业元数据进度: %d/%d 板块", i + 1, len(boards))

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {"industry_map": industry_map, "name_map": name_map}, fh, ensure_ascii=False
        )
    logger.info(
        "元数据: %d 行业映射, %d 名称 → %s", len(industry_map), len(name_map), path
    )
    return industry_map, name_map


def enrich_panel(
    df: pd.DataFrame,
    industry_map: dict[str, str] | None = None,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """补齐面板元数据列 (详见模块文档). 输入需含 symbol/date/turnover_rate."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
    if "board" not in df.columns:
        df["board"] = df["symbol"].map(board_of)
    if "is_suspended" not in df.columns:
        df["is_suspended"] = False
    if "industry" not in df.columns:
        if industry_map:
            df["industry"] = df["symbol"].map(industry_map).fillna("UNKNOWN")
        else:
            df["industry"] = "UNKNOWN"
    if "free_float_turnover_rate" not in df.columns and "turnover_rate" in df.columns:
        df["free_float_turnover_rate"] = df["turnover_rate"]
    return df


def _keep_cyq_base_cols(cyq: pd.DataFrame) -> pd.DataFrame:
    """CYQ 只保留 KEEP 基础列 (2026-08-02 删列决策).

    cyq_cache 中间件保持全列, 只有 merge 进面板时过滤; 不透传已删列
    (pct_70_*/pct_90_low/cost_5pct/15pct/85pct) 与原始 Tushare 字段
    (winner_rate/his_low/his_high).
    """
    from config.settings import CYQ_BASE_KEEP

    return cyq[["symbol", "date"] + [c for c in CYQ_BASE_KEEP if c in cyq.columns]]


def enrich_cyq(
    panel: pd.DataFrame,
    cyq_cache: str = "data/cyq_panel.parquet",
    refresh: bool = False,
) -> pd.DataFrame:
    """将 CYQ 筹码分布列 merge 进面板 (按 symbol+date left join).

    CYQ 计算通过 cyq_calculator.compute_cyq_panel, 结果缓存到 cyq_cache.
    首次运行计算全量 (1042 股约 30-60 分钟), 后续命中缓存秒级加载.

    Args:
        panel: enrich 后的 OHLCV 面板 (需含 open/close/high/low/turnover_rate)
        cyq_cache: CYQ 面板缓存路径
        refresh: True 强制重新计算

    Returns:
        merge 后的面板 (保留 CYQ_BASE_KEEP 列)
    """
    import os

    from .cyq_calculator import compute_cyq_panel

    if not refresh and os.path.exists(cyq_cache):
        cyq = pd.read_parquet(cyq_cache)
        # 检查是否覆盖当前 panel 的 symbol+date 范围
        cyq_symbols = set(cyq["symbol"].unique())
        panel_symbols = set(panel["symbol"].unique())
        if cyq_symbols >= panel_symbols:
            missing = panel_symbols - cyq_symbols
            if not missing:
                return panel.merge(
                    _keep_cyq_base_cols(cyq), on=["symbol", "date"], how="left"
                )
            # 部分缺失: 只计算缺失的股票
            need = [s for s in panel["symbol"].unique() if s in missing]
            new_cyq = compute_cyq_panel(panel[panel["symbol"].isin(need)])
            cyq = pd.concat([cyq, new_cyq], ignore_index=True)
            cyq.to_parquet(cyq_cache, index=False)
            return panel.merge(
                _keep_cyq_base_cols(cyq), on=["symbol", "date"], how="left"
            )

    # 全量计算 + 缓存
    cyq = compute_cyq_panel(panel)
    os.makedirs(os.path.dirname(cyq_cache) or ".", exist_ok=True)
    cyq.to_parquet(cyq_cache, index=False)
    return panel.merge(_keep_cyq_base_cols(cyq), on=["symbol", "date"], how="left")


_ENRICH_WORKERS = int(os.environ.get("ENRICH_WORKERS", "4"))


def _parallel_fetch(
    fn,
    items: list,
    *,
    workers: int = _ENRICH_WORKERS,
    desc: str = "",
    progress_file: str | None = str(data_others_path(_PROGRESS_FILE)),
    unpack: bool = True,
) -> list:
    """Run fn(item) or fn(*item) over items with ThreadPoolExecutor.

    Each call is independent (cache-first, per-stock/per-date). Errors are caught
    and logged; failed items return None and are filtered out. A shared progress
    counter writes to ``progress_file`` every 10 completions so external monitors
    can track ETA.

    Rate limiting is implicit: Tushare API latency (~1-2s/call) with 4 workers
    stays under the 200 calls/min free-token ceiling.
    """
    results: list = []
    total = len(items)
    t0 = time.time()
    completed = 0
    lock = threading.Lock()

    def _worker(item):
        nonlocal completed
        try:
            if unpack:
                return fn(*item) if isinstance(item, tuple) else fn(item)
            else:
                return fn(item)
        except Exception:
            return None
        finally:
            with lock:
                completed += 1
                if progress_file and completed % _PROGRESS_EVERY_N == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    try:
                        with open(progress_file, "w", encoding="utf-8") as pf:
                            pf.write(
                                f"{desc}: {completed}/{total} ({completed / total:.1%}), "
                                f"{elapsed:.0f}s elap, ETA {eta:.0f}s\n"
                            )
                    except OSError:
                        pass

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, item) for item in items]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None and (
                    not hasattr(result, "__len__") or len(result)
                ):
                    results.append(result)
            except Exception:
                pass

    return results


def enrich_alt_data(
    panel: pd.DataFrame,
    supply: DataSupplyChain,
    sources: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """[Phase 0] 将替代数据源 merge 进面板 (按 symbol+date left join).

    可选数据源 (默认全部):
      - northbound      : 北向资金 (日频, 仅沪深港通标的)
      - margin          : 融资融券 (日频, 仅两融标的)
      - fina_indicator  : 基本面PIT (季频, 全A)
      - lhb             : 龙虎榜 (日频, 仅上榜标的)
      - holdernumber    : 股东户数 (季频, 全A)
      - holdertrade     : 股东增减持 (不定期, 部分标的)

    每个数据源独立拉取+缓存 → left join 到面板, 缺失填 NaN.
    不阻断训练: 单个数据源失败告警继续, 不影响其他源.

    Args:
        panel:      enrich_panel 后的 OHLCV 面板
        supply:     DataSupplyChain 实例
        sources:    要拉取的数据源列表 (None=全部 6 个)
        start_date: 起始日期 'YYYYMMDD' (None=面板最早日期)
        end_date:   截止日期 'YYYYMMDD' (None=面板最晚日期)
        refresh:    True 强制重新拉取

    Returns:
        merge 后的面板 (新增各数据源的特征列)
    """
    if sources is None:
        sources = list(_DEFAULT_ALT_SOURCES)

    if start_date is None:
        start_date = panel["date"].min().strftime("%Y%m%d")
    if end_date is None:
        end_date = panel["date"].max().strftime("%Y%m%d")

    logger.info(
        "替代数据源 enrich: %s, 日期范围: %s - %s",
        sources,
        start_date,
        end_date,
    )

    for src in sources:
        try:
            df = None
            # northbound 已移除
            if src == "margin":
                df = supply.fetch_margin(
                    start_date=start_date,
                    end_date=end_date,
                    refresh=refresh,
                )
                if len(df):
                    merge_cols = ["symbol", "date"]
                    avail = [
                        c
                        for c in df.columns
                        if c not in merge_cols and not c.startswith("_")
                    ]
                    panel = panel.merge(
                        df[merge_cols + avail],
                        on=merge_cols,
                        how="left",
                    )

            elif src == "fina_indicator":
                # 逐股拉取 (免费token不支持全市场查询), 多线程并发
                symbols = panel["symbol"].unique().tolist()

                def _fetch_one_fina(sym):
                    ts_code = (
                        f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
                    )
                    df_one = supply.fetch_fina_indicator(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        refresh=refresh,
                    )
                    return df_one if len(df_one) else None

                frames = _parallel_fetch(
                    _fetch_one_fina,
                    symbols,
                    desc="fina",
                    progress_file=None,
                )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    logger.info("fina_indicator 逐股拉取完成: %d stocks", len(frames))
                else:
                    df = pd.DataFrame()
                if len(df) and "announce_date" in df.columns:
                    # 基本面PIT: 按 announce_date 做 merge_asof (严禁直接用 report_period)
                    fin_cols = [
                        c
                        for c in df.columns
                        if c
                        not in ("symbol", "announce_date", "report_period", "_ts_code")
                    ]
                    f = df[["symbol", "announce_date"] + fin_cols].copy()
                    f = f.sort_values("announce_date")
                    panel = panel.sort_values("date")
                    panel = pd.merge_asof(
                        panel,
                        f,
                        left_on="date",
                        right_on="announce_date",
                        by="symbol",
                        direction="backward",
                    )

            elif src == "lhb":
                df = supply.fetch_lhb(
                    start_date=start_date,
                    end_date=end_date,
                    refresh=refresh,
                )
                if len(df) and "symbol" in df.columns and "date" in df.columns:
                    merge_cols = ["symbol", "date"]
                    avail = [
                        c
                        for c in df.columns
                        if c not in merge_cols and not c.startswith("_")
                    ]
                    panel = panel.merge(
                        df[merge_cols + avail],
                        on=merge_cols,
                        how="left",
                    )

            elif src == "holdernumber":
                symbols = panel["symbol"].unique().tolist()

                def _fetch_one_holder(sym):
                    ts_code = (
                        f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
                    )
                    df_one = supply.fetch_holdernumber(
                        ts_code=ts_code,
                        start_date=start_date,
                        end_date=end_date,
                        refresh=refresh,
                    )
                    return df_one if len(df_one) else None

                frames = _parallel_fetch(
                    _fetch_one_holder,
                    symbols,
                    desc="holdernumber",
                    progress_file=None,
                )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    # PIT: 按 announce_date 做 merge_asof 到日频面板
                    if "announce_date" in df.columns:
                        hn_cols = [
                            c
                            for c in df.columns
                            if c not in ("symbol", "date", "_ts_code")
                        ]
                        f = df[["symbol", "announce_date"] + hn_cols].copy()
                        f = f.sort_values("announce_date")
                        panel = panel.sort_values("date")
                        panel = pd.merge_asof(
                            panel,
                            f,
                            left_on="date",
                            right_on="announce_date",
                            by="symbol",
                            direction="backward",
                        )

            elif src == "holdertrade":
                # 全量拉取 (dim29 依赖此数据源) — bulk 模式带自动分页
                df = supply.fetch_holdertrade(
                    start_date=start_date,
                    end_date=end_date,
                    refresh=refresh,
                )
                if (
                    len(df)
                    and "announce_date" in df.columns
                    and "sh_net_sign" in df.columns
                ):
                    # 按公告日期聚合: 日频面板上当日净增减持
                    agg_map = {
                        "sh_net_change_sign": ("sh_net_sign", "sum"),
                        "sh_change_amt_total": ("sh_change_amt", "sum"),
                    }
                    if "evt_start_date" in df.columns:
                        agg_map["sh_evt_start_date"] = ("evt_start_date", "min")
                    if "evt_end_date" in df.columns:
                        agg_map["sh_evt_end_date"] = ("evt_end_date", "max")
                    daily_net = (
                        df.groupby(["symbol", "announce_date"]).agg(**agg_map).reset_index()
                    )
                    daily_net = daily_net.rename(columns={"announce_date": "date"})
                    daily_net["date"] = pd.to_datetime(daily_net["date"])
                    # merge 到面板
                    panel = panel.merge(
                        daily_net,
                        on=["symbol", "date"],
                        how="left",
                    )
                    logger.info(
                        "holdertrade: %d records, %d unique symbols, %d unique dates",
                        len(df),
                        daily_net["symbol"].nunique(),
                        daily_net["date"].nunique(),
                    )

            elif src == "sector_index":
                df = supply.fetch_sector_index(
                    start_date=start_date,
                    end_date=end_date,
                    refresh=refresh,
                )
                if len(df) and "industry" in panel.columns:
                    # 行业指数按 date+industry 广播到个股:
                    # panel.industry 是东财行业名, sector index 是申万行业名
                    # 需要 industry 映射表; 无映射时用模糊匹配降级
                    name_to_code = {
                        name: code
                        for code, name in df[["index_code", "index_name"]]
                        .drop_duplicates()
                        .itertuples(index=False)
                    }
                    # 如果 industry_map 包含 SW→DFCF 映射, 直接用; 否则尝试模糊匹配
                    # 简单策略: 取包含关系 (e.g. "电子" in "电子设备" or vice versa)
                    ind_map: dict[str, str] = {}
                    for ind_name in panel["industry"].dropna().unique():
                        if ind_name in name_to_code:
                            ind_map[ind_name] = ind_name
                        else:
                            for sw_name in name_to_code:
                                if ind_name in sw_name or sw_name in ind_name:
                                    ind_map[ind_name] = sw_name
                                    break
                    if ind_map:
                        panel["_sw_name"] = panel["industry"].map(ind_map)
                        sw_data = df.rename(
                            columns={
                                "ret_pct": "sw_ret_1d",
                                "close": "sw_index_close",
                                "volume": "sw_index_vol",
                            }
                        )
                        avail = [
                            c
                            for c in sw_data.columns
                            if c not in ("index_code", "date")
                        ]
                        panel = panel.merge(
                            sw_data[["index_name", "date"] + avail],
                            left_on=["_sw_name", "date"],
                            right_on=["index_name", "date"],
                            how="left",
                        )
                        panel = panel.drop(
                            columns=["_sw_name", "index_name"], errors="ignore"
                        )

            elif src == "daily_basic":
                # 逐日拉取全市场 daily_basic, 多线程并发
                dates = panel["date"].drop_duplicates().sort_values()

                def _fetch_one_date(d):
                    ds = d.strftime("%Y%m%d")
                    df_one = supply.fetch_daily_basic(
                        trade_date=ds,
                        refresh=refresh,
                    )
                    return df_one if len(df_one) else None

                frames = _parallel_fetch(
                    _fetch_one_date,
                    dates.tolist(),
                    desc="daily_basic",
                    progress_file=None,
                )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    merge_cols = ["symbol", "date"]
                    avail = [
                        c
                        for c in df.columns
                        if c not in merge_cols and not c.startswith("_")
                    ]
                    panel = panel.merge(
                        df[merge_cols + avail],
                        on=merge_cols,
                        how="left",
                    )

            elif src == "stk_limit":
                dates = panel["date"].drop_duplicates().sort_values()

                def _fetch_one_limit_date(d):
                    ds = d.strftime("%Y%m%d")
                    df_one = supply.fetch_stk_limit(
                        trade_date=ds,
                        refresh=refresh,
                    )
                    return df_one if len(df_one) else None

                frames = _parallel_fetch(
                    _fetch_one_limit_date,
                    dates.tolist(),
                    desc="stk_limit",
                    progress_file=None,
                )
                if frames:
                    df = pd.concat(frames, ignore_index=True)
                    merge_cols = ["symbol", "date"]
                    avail = [
                        c
                        for c in df.columns
                        if c not in merge_cols and not c.startswith("_")
                    ]
                    panel = panel.merge(
                        df[merge_cols + avail],
                        on=merge_cols,
                        how="left",
                    )

            elif src == "cyq_tushare":
                # Tushare cyq_perf 真实筹码分布 (his_low/his_high/winner_rate
                # + cost_5pct..95pct/weight_avg) — 比 OHLCV 推导精确
                symbols = panel["symbol"].unique().tolist()
                df = supply.fetch_chip_distribution_batch(
                    symbols,
                    start_date=start_date,
                    end_date=end_date,
                    refresh=refresh,
                )
                if len(df):
                    # V3 删列 (2026-08-02): 只合并 KEEP 基础列, 不透传已删列
                    # (pct_70_*/pct_90_low/cost_5pct/15pct/85pct) 与原始 Tushare 字段
                    # (winner_rate/his_low/his_high). 派生 KEEP 列 (winner_ratio/
                    # pct_90_high/pct_90_con/avg_cost) 由 dim21 或面板既有值负责.
                    # cyq_perf 原始列中仅 cost_50pct/cost_95pct/weight_avg 属于 KEEP.
                    df = _keep_cyq_base_cols(df)
                    merge_cols = ["symbol", "date"]
                    avail = [
                        c
                        for c in df.columns
                        if c not in merge_cols and not c.startswith("_")
                    ]
                    panel = panel.merge(
                        df[merge_cols + avail],
                        on=merge_cols,
                        how="left",
                    )

            n_new_cols = len(df.columns) - 2 if df is not None and len(df) else 0
            n_rows = len(df) if df is not None else 0
            logger.info(
                "  %s: %d 行, %d 新列 -> panel",
                src,
                n_rows,
                n_new_cols,
            )
        except Exception as exc:
            logger.warning("替代数据源 %s 跳过: %s", src, exc)

    return panel


def assemble_panel(
    supply: DataSupplyChain,
    symbols: list[str],
    end: str | None = None,
    years: int = DEFAULT_YEARS,
    refresh: bool = False,
    cache_dir: str = PANEL_CACHE_DIR,
    industry_map: dict[str, str] | None = None,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """批量拉取个股历史 (akshare, 默认 3 年) → enrich → 缓存 parquet (WORM, 按日期后缀).

    Args:
        supply: DataSupplyChain (生产 akshare / 测试注入 fetcher_hist)
        symbols: 训练/推理 universe
        end: 截止日期 'YYYY-MM-DD' (None=今天); 缓存文件名按此日期后缀, 不覆盖旧文件
        years: 回看年数 (默认 3 年)
        refresh: True 强制重新拉取个股历史

    Returns:
        enrich 后的全 symbol 面板 (symbol/date 升序)
    """
    end_str = end or datetime.now().strftime("%Y-%m-%d")
    # 缓存键含 universe 哈希: 同一截止日期不同股票池不得共享面板缓存
    import hashlib

    universe_hash = hashlib.md5("|".join(sorted(symbols)).encode()).hexdigest()[:8]
    cache_path = os.path.join(
        cache_dir, f"panel_{end_str.replace('-', '')}_{years}y_{universe_hash}.parquet"
    )
    if not refresh and os.path.exists(cache_path):
        logger.info("命中面板缓存: %s", cache_path)
        return pd.read_parquet(cache_path)
    panel = supply.backfill_ohlcv(symbols, years=years, end=end_str, refresh=refresh)
    panel = enrich_panel(panel, industry_map=industry_map, name_map=name_map)
    # 替代数据 enrich (不阻断: 任意数据源失败告警继续, 不影响训练/推理)
    try:
        _start = panel["date"].min().strftime("%Y%m%d")
        panel = enrich_alt_data(
            panel,
            supply,
            sources=list(_DEFAULT_ALT_SOURCES),
            start_date=_start,
            end_date=end_str,
            refresh=refresh,
        )
        logger.info("替代数据 enrich 完成, 面板列数: %d", len(panel.columns))
    except Exception as e:
        logger.warning("替代数据 enrich 失败 (不阻断): %s", e)
    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    logger.info(
        "面板装配完成: %d 股 %d 行 → %s",
        panel["symbol"].nunique(),
        len(panel),
        cache_path,
    )
    return panel
