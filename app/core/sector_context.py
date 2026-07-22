# -*- coding: utf-8 -*-
"""板块平均效应因子 (P5, ARCH §5.8 — 4 维).

板块内个股联动: 板块平均涨跌幅/板块相对强度/板块内排名/板块资金流向。

因子清单 (4 列):
    - sector_avg_pct_change:     板块成员最新日涨跌幅均值
    - sector_relative_strength:  个股涨跌幅 - 板块均值 (正=跑赢板块)
    - sector_rank_pct:           个股涨跌幅在板块内的分位 (0~1, 越大越强)
    - sector_net_flow:           板块资金净流入 Σ(close-open)*volume (最新日)

板块归属: 优先使用构造时传入的 sector_map ({symbol: 板块名});
未命中时退化为交易所/板块前缀启发式 (60→SSE_MAIN, 00→SZSE_MAIN,
30→CHINEXT, 68→STAR, 其他→OTHER)。

未来函数禁止: 仅使用截面 (最新一行) 数据, 无任何时序回看未来。
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SECTOR_FACTOR_COLUMNS = [
    "sector_avg_pct_change",     # 板块平均涨跌幅
    "sector_relative_strength",  # 个股相对板块强度
    "sector_rank_pct",           # 个股在板块内排名分位
    "sector_net_flow",           # 板块资金净流入
]

# ── 默认配置 (可由 config dict 覆盖) ─────────────────────────────────
DEFAULT_CONFIG = {
    "min_rows": 2,          # 计算涨跌幅所需的最少行数
}


class SectorContext:
    """板块上下文因子计算器.

    Args:
        sector_map: 可选, {symbol: 板块名} 映射; None 时全部走前缀启发式。
        config: 可选配置 dict, 覆盖 DEFAULT_CONFIG。
    """

    def __init__(self, sector_map: Optional[Dict[str, str]] = None,
                 config: Optional[dict] = None) -> None:
        self._sector_map: Dict[str, str] = dict(sector_map or {})
        self._config = {**DEFAULT_CONFIG, **(config or {})}

    # ───────────────────────────────────────────────────────────────
    #  板块归属
    # ───────────────────────────────────────────────────────────────

    def sector_for(self, symbol: str) -> str:
        """返回股票所属板块名 (sector_map 优先, 否则前缀启发式)."""
        code = str(symbol).strip()
        if code in self._sector_map:
            return self._sector_map[code]
        return self._prefix_sector(code)

    @staticmethod
    def _prefix_sector(code: str) -> str:
        """按代码前缀推断粗粒度板块 (交易所/上市板)."""
        if code.startswith("60"):
            return "SSE_MAIN"
        if code.startswith("00"):
            return "SZSE_MAIN"
        if code.startswith("30"):
            return "CHINEXT"
        if code.startswith("68"):
            return "STAR"
        return "OTHER"

    # ───────────────────────────────────────────────────────────────
    #  因子计算
    # ───────────────────────────────────────────────────────────────

    def _latest_pct_and_flow(self, df: pd.DataFrame) -> Optional[tuple]:
        """取个股最新日 (涨跌幅, 资金净流入估算); 数据不足返回 None."""
        if df is None or df.empty or "close" not in df.columns:
            return None
        min_rows = int(self._config.get("min_rows", DEFAULT_CONFIG["min_rows"]))
        if len(df) < min_rows:
            return None
        close = df["close"].astype(float)
        pct = float(close.pct_change().iloc[-1])
        flow = 0.0
        if {"open", "close", "volume"}.issubset(df.columns):
            last = df.iloc[-1]
            flow = float((float(last["close"]) - float(last["open"]))
                         * float(last["volume"]))
        return pct, flow

    def compute(self, symbol: str, all_stocks: Dict[str, pd.DataFrame]) -> dict:
        """计算个股所属板块的 4 维上下文因子.

        Args:
            symbol: 股票代码。
            all_stocks: 全市场日线数据 {symbol: DataFrame}
                (至少含 close; 含 open/volume 时计算资金流向)。

        Returns:
            {factor_name: value} 4 维字典; 数据不足时全 0。
        """
        result = {c: 0.0 for c in SECTOR_FACTOR_COLUMNS}
        code = str(symbol).strip()

        if not all_stocks or code not in all_stocks:
            logger.warning("symbol %s 不在 all_stocks 中, 返回全 0 因子", code)
            return result

        sector = self.sector_for(code)
        members = [s for s in all_stocks if self.sector_for(s) == sector]

        pcts: List[float] = []
        flows: List[float] = []
        own_pct: Optional[float] = None
        for s in members:
            pf = self._latest_pct_and_flow(all_stocks[s])
            if pf is None:
                continue
            pcts.append(pf[0])
            flows.append(pf[1])
            if s == code:
                own_pct = pf[0]

        if not pcts or own_pct is None:
            logger.warning("板块 %s 有效成员不足或 %s 数据不足, 返回全 0 因子",
                           sector, code)
            return result

        pcts_arr = np.nan_to_num(np.asarray(pcts, dtype=float),
                                 nan=0.0, posinf=0.0, neginf=0.0)
        flows_arr = np.nan_to_num(np.asarray(flows, dtype=float),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        own_pct = float(np.nan_to_num(own_pct, nan=0.0,
                                      posinf=0.0, neginf=0.0))

        sector_avg = float(pcts_arr.mean())
        result["sector_avg_pct_change"] = sector_avg
        result["sector_relative_strength"] = own_pct - sector_avg
        # 分位: 板块内 涨跌幅 <= 自身 的成员占比 (0,1]; 最强者 = 1.0
        result["sector_rank_pct"] = float(np.mean(pcts_arr <= own_pct))
        result["sector_net_flow"] = float(flows_arr.sum())

        logger.debug("板块因子 %s [%s]: avg=%.4f rel=%.4f rank=%.2f flow=%.0f",
                     code, sector, result["sector_avg_pct_change"],
                     result["sector_relative_strength"],
                     result["sector_rank_pct"], result["sector_net_flow"])
        return result

    # ───────────────────────────────────────────────────────────────
    #  辅助
    # ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_factor_columns() -> List[str]:
        """返回板块因子列名列表."""
        return SECTOR_FACTOR_COLUMNS.copy()
