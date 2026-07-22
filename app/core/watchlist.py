# -*- coding: utf-8 -*-
"""重点关注股标记管理 (Watchlist).

用户可以在股票池中标记某些股票为"重点关注"，系统会：
  - 持久化到 JSON 文件
  - 在看板中高亮显示
  - 在交易日 Pipeline 中优先检查
"""

import json
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认存储路径
DEFAULT_WATCHLIST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "watchlist.json",
)


class WatchlistManager:
    """管理重点关注股的增删查改.

    JSON 结构：
    {
        "stocks": {
            "600519": {
                "added_date": "2026-07-20",
                "note": "茅台，等回调买",
                "tags": ["白酒", "龙头"]
            }
        },
        "last_updated": "2026-07-20T01:00:00"
    }
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """初始化.

        Args:
            path: JSON 文件路径，默认 data/watchlist.json.
        """
        self.path = path or DEFAULT_WATCHLIST_PATH
        self._data: Dict = self._load()

    def _load(self) -> Dict:
        """从磁盘加载 watchlist."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Watchlist 加载失败: %s", exc)
        return {"stocks": {}, "last_updated": None}

    def _save(self) -> None:
        """持久化到磁盘."""
        self._data["last_updated"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info("Watchlist 已保存: %d 只股票", len(self._data["stocks"]))

    def add(
        self, symbol: str, note: str = "", tags: Optional[List[str]] = None
    ) -> None:
        """添加股票到重点关注.

        Args:
            symbol: 股票代码.
            note: 用户备注.
            tags: 标签列表.
        """
        self._data["stocks"][symbol] = {
            "added_date": date.today().isoformat(),
            "note": note,
            "tags": tags or [],
        }
        self._save()
        logger.info("已添加关注: %s (%s)", symbol, note)

    def remove(self, symbol: str) -> None:
        """移除重点关注.

        Args:
            symbol: 股票代码.
        """
        if symbol in self._data["stocks"]:
            del self._data["stocks"][symbol]
            self._save()
            logger.info("已移除关注: %s", symbol)

    def is_watched(self, symbol: str) -> bool:
        """判断是否在关注列表中."""
        return symbol in self._data["stocks"]

    def get_all(self) -> Dict:
        """返回全部关注股票."""
        return self._data["stocks"]

    def get_symbols(self) -> List[str]:
        """返回全部关注的股票代码列表."""
        return list(self._data["stocks"].keys())

    def get_info(self, symbol: str) -> Optional[Dict]:
        """获取某只股票的关注信息."""
        return self._data["stocks"].get(symbol)

    def update_note(self, symbol: str, note: str) -> None:
        """更新备注."""
        if symbol in self._data["stocks"]:
            self._data["stocks"][symbol]["note"] = note
            self._save()
