# -*- coding: utf-8 -*-
"""Universe 股票池分池管理 (P1, ARCH §4).

主板 (±10%) 与创业板/科创板 (±20%) 分池, 各自路由对应 ONNX 模型。

代码前缀规则 (ARCH §4):
    - 60xxxx / 00xxxx (000/001/002...) → MAIN_BOARD (沪深主板, ±10%)
    - 30xxxx (300/301) / 688xxx        → GROWTH_BOARDS (创业板+科创板, ±20%)
    - 其他前缀 (北交所等)               → 默认 MAIN_BOARD 并告警 (10% 更保守)
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 代码前缀 → Universe 映射 (ARCH §4) ──────────────────────────────
MAIN_BOARD_PREFIXES = ("60", "00")  # 沪主板 60xxxx + 深主板 00xxxx
GROWTH_BOARD_PREFIXES = ("30", "68")  # 创业板 30xxxx + 科创板 688xxx

# ── ST / 退市 名称前缀 ───────────────────────────────────────────────
ST_NAME_PREFIXES = ("ST", "*ST")
DELIST_NAME_PREFIX = "退"


class Universe(Enum):
    """股票池分类."""

    ALL = "all"
    MAIN_BOARD = "main_board"  # 沪深主板, 涨跌停 ±10%
    GROWTH_BOARDS = "growth_boards"  # 创业板+科创板, 涨跌停 ±20%


def classify_symbol(symbol: str) -> Universe:
    """按代码前缀判断股票所属 Universe.

    Args:
        symbol: 6 位股票代码 (允许带空格, 将 strip)。

    Returns:
        Universe 枚举值; 未识别前缀默认 MAIN_BOARD 并告警。
    """
    code = str(symbol).strip()
    if code.startswith(MAIN_BOARD_PREFIXES):
        return Universe.MAIN_BOARD
    if code.startswith(GROWTH_BOARD_PREFIXES):
        return Universe.GROWTH_BOARDS
    logger.warning("未识别代码前缀 %s, 默认归为 MAIN_BOARD (±10%% 更保守)", code)
    return Universe.MAIN_BOARD


def name_is_st(name: str) -> bool:
    """按名称判断是否 ST / *ST / 退市整理股.

    Args:
        name: 股票名称。

    Returns:
        True 表示 ST/*ST/退市股。
    """
    n = str(name).strip()
    if not n:
        return False
    upper = n.upper()
    if upper.startswith(ST_NAME_PREFIXES):
        return True
    if n.startswith(DELIST_NAME_PREFIX):
        return True
    return False


class UniverseManager:
    """Universe 分池管理器.

    职责: 股票代码 → Universe 路由; 分池列表维护; ST/退市标记。

    Args:
        stocks: 可选, 初始股票代码列表 (用于 get_universe_stocks)。
        name_map: 可选, {symbol: 股票名称} 映射 (用于 is_st 判断)。
    """

    def __init__(
        self,
        stocks: Optional[List[str]] = None,
        name_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._stocks: List[str] = [str(s).strip() for s in (stocks or [])]
        self._name_map: Dict[str, str] = dict(name_map or {})

    def classify(self, symbol: str) -> Universe:
        """判断股票所属 Universe (60/00→主板, 30/68→创业科创).

        Args:
            symbol: 6 位股票代码。

        Returns:
            Universe 枚举值。
        """
        return classify_symbol(symbol)

    def get_universe_stocks(self, universe: Universe) -> List[str]:
        """返回指定 Universe 的全部股票代码.

        Args:
            universe: Universe.ALL 返回全部, 否则按 classify 过滤。

        Returns:
            股票代码列表 (保持注册顺序)。
        """
        if universe == Universe.ALL:
            return list(self._stocks)
        return [s for s in self._stocks if self.classify(s) == universe]

    def is_st(self, symbol: str) -> bool:
        """是否 ST/退市整理股.

        依赖构造时传入的 name_map; 未知名称返回 False (宁放过不误杀)。
        """
        return name_is_st(self._name_map.get(str(symbol).strip(), ""))

    # ── 维护接口 ────────────────────────────────────────────────────

    def set_stocks(self, stocks: List[str]) -> None:
        """更新股票代码列表."""
        self._stocks = [str(s).strip() for s in stocks]

    def set_names(self, name_map: Dict[str, str]) -> None:
        """更新 股票代码→名称 映射."""
        self._name_map = dict(name_map)
