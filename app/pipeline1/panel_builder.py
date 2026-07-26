# -*- coding: utf-8 -*-
"""训练/推理面板装配 (PIPELINE1 生产数据入口)
=====================================================
把 ``DataSupplyChain.backfill_ohlcv`` 的原始 OHLCV 面板补齐为
cleaning/feature/label 全链路可用的标准面板:

- ``board``        : 缺失时按代码前缀推导 (复用 cleaning_pipeline.board_of)
- ``is_st``        : 由 name_map 判断 (universe_manager.name_is_st), 无名称表默认 False
- ``is_suspended`` : 默认 False (停牌日天然无 bar, 不影响训练)
- ``list_days``    : 面板内每 symbol 的累计交易日数 (近似上市天数)
- ``industry``     : industry_map 提供, 缺失默认 "UNKNOWN"
- ``free_float_turnover_rate`` : 缺失时回退 turnover_rate

默认数据深度: 最近 3 年 akshare 日线 (用户 2026-07-26 裁决);
[B11] 深度不足 1250 交易日时训练窗口自动降为 540 日过渡 (见 dual_track_trainer).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd

from app.core.universe_manager import name_is_st

from .cleaning_pipeline import board_of
from .data_supply import DataSupplyChain

logger = logging.getLogger(__name__)

DEFAULT_YEARS = 3  # 用户裁决 (2026-07-26): 训练数据取最近 3 年
PANEL_CACHE_DIR = os.path.join("data", "processed")


def enrich_panel(
    df: pd.DataFrame,
    industry_map: dict[str, str] | None = None,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """补齐面板元数据列 (详见模块文档). 输入需含 symbol/date/turnover_rate."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
    if "board" not in df.columns:
        df["board"] = df["symbol"].map(board_of)
    if "is_st" not in df.columns:
        if name_map:
            df["is_st"] = df["symbol"].map(lambda s: name_is_st(name_map.get(s, "")))
        else:
            df["is_st"] = False
    if "is_suspended" not in df.columns:
        df["is_suspended"] = False
    if "list_days" not in df.columns:
        df["list_days"] = df.groupby("symbol").cumcount() + 1
    if "industry" not in df.columns:
        if industry_map:
            df["industry"] = df["symbol"].map(industry_map).fillna("UNKNOWN")
        else:
            df["industry"] = "UNKNOWN"
    if "free_float_turnover_rate" not in df.columns and "turnover_rate" in df.columns:
        df["free_float_turnover_rate"] = df["turnover_rate"]
    return df


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
    cache_path = os.path.join(
        cache_dir, f"panel_{end_str.replace('-', '')}_{years}y.parquet"
    )
    if not refresh and os.path.exists(cache_path):
        logger.info("命中面板缓存: %s", cache_path)
        return pd.read_parquet(cache_path)
    panel = supply.backfill_ohlcv(symbols, years=years, end=end_str, refresh=refresh)
    panel = enrich_panel(panel, industry_map=industry_map, name_map=name_map)
    os.makedirs(cache_dir, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    logger.info(
        "面板装配完成: %d 股 %d 行 → %s",
        panel["symbol"].nunique(),
        len(panel),
        cache_path,
    )
    return panel
