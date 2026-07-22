# -*- coding: utf-8 -*-
"""推荐股票池管理系统 (P10.11, ARCH §5.16, DESIGN_V1 §3).

5 种 TICK 标记 (全部可手工更改) + 按标记筛选 + 手工增删 +
资金分配 (PERCENTAGE ALLOCATION) + 固定保留 (is_fixed)。
持久化: data/stock_pool.json。
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TICK_FIELDS = [
    "is_watchlist",  # 自选
    "is_daily_buy",  # 日线买点
    "is_intraday_buy",  # 日内买点
    "is_daily_sell",  # 日线卖点
    "is_intraday_sell",  # 日内卖点
]

# 打标后自动固定 (is_fixed=True) 的 TICK (ARCH §5.16.2)
_FIXING_TICKS = ("is_daily_buy", "is_daily_sell")

DEFAULT_POOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "stock_pool.json",
)


def _default_ticks() -> Dict[str, bool]:
    """全部 TICK 初始化为 False."""
    return {tick: False for tick in TICK_FIELDS}


class StockPoolManager:
    """推荐股票池管理器.

    规则 (DESIGN_V1 §3.2, ARCH §5.16.3):
        - 每日 16:00 后 Pipeline 1 刷新股票池
        - 带日线买入/卖出标记的股票 is_fixed=True, 刷新时不被覆盖
        - 所有 TICK 可手工打/撤; 可手工输入代码增删
        - percentage_allocation 仅对 is_daily_buy 股票生效, 合计 ≤100%
    """

    def __init__(self, pool_path: str = DEFAULT_POOL_PATH) -> None:
        """初始化并加载持久化股票池.

        Args:
            pool_path: JSON 文件路径, 默认 data/stock_pool.json。
        """
        self.pool_path = pool_path
        self._data: Dict = self._load()

    # ── 持久化 ──────────────────────────────────────────────

    def _load(self) -> Dict:
        """从磁盘加载股票池."""
        if os.path.exists(self.pool_path):
            try:
                with open(self.pool_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("pool", [])
                return data
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("股票池加载失败 (%s), 使用空池", exc)
        return {"last_updated": None, "pool": []}

    def _save(self) -> None:
        """持久化到磁盘."""
        self._data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.pool_path) or ".", exist_ok=True)
        with open(self.pool_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.info("股票池已保存: %d 只股票", len(self._data["pool"]))

    # ── 内部查询 ────────────────────────────────────────────

    def _find(self, symbol: str) -> Optional[dict]:
        """按代码查找池内记录, 不存在返回 None."""
        for rec in self._data["pool"]:
            if rec.get("symbol") == symbol:
                return rec
        return None

    @staticmethod
    def _new_record(
        symbol: str,
        added_by: str,
        name: str = "",
        score: float = 0.0,
        prob_up: float = 0.0,
        pct_up: float = 0.0,
    ) -> dict:
        """构造新股票记录 (ticks 全 False)."""
        return {
            "symbol": symbol,
            "name": name,
            "score": float(score),
            "prob_up": float(prob_up),
            "pct_up": float(pct_up),
            "ticks": _default_ticks(),
            "is_fixed": False,
            "percentage_allocation": 0.0,
            "added_at": datetime.now().strftime("%Y-%m-%d"),
            "added_by": added_by,
            "ticks_history": [],
        }

    # ── 公开接口 ────────────────────────────────────────────

    def get_pool(self, filter_by_tick: Optional[str] = None) -> List[dict]:
        """获取股票池 (可按 TICK 筛选).

        Args:
            filter_by_tick: TICK 字段名 (TICK_FIELDS 之一); None 返回全部。

        Returns:
            股票记录列表 (按 score 降序)。

        Raises:
            ValueError: filter_by_tick 不在 TICK_FIELDS 中。
        """
        if filter_by_tick is not None and filter_by_tick not in TICK_FIELDS:
            raise ValueError(f"未知 TICK 标记: {filter_by_tick}, 可选: {TICK_FIELDS}")
        pool = self._data["pool"]
        if filter_by_tick is not None:
            pool = [r for r in pool if r.get("ticks", {}).get(filter_by_tick, False)]
        return sorted(pool, key=lambda r: r.get("score", 0.0), reverse=True)

    def add_stock(self, symbol: str, added_by: str = "manual", name: str = "") -> None:
        """手工添加股票.

        Args:
            symbol: 股票代码。
            added_by: 来源 (manual/pipeline1/pipeline2)。
            name: 股票名称 (可选)。
        """
        if self._find(symbol) is not None:
            logger.info("股票已在池中, 跳过: %s", symbol)
            return
        self._data["pool"].append(self._new_record(symbol, added_by, name=name))
        self._save()
        logger.info("已添加股票: %s (%s)", symbol, added_by)

    def remove_stock(self, symbol: str) -> None:
        """删除股票.

        Args:
            symbol: 股票代码。
        """
        before = len(self._data["pool"])
        self._data["pool"] = [
            r for r in self._data["pool"] if r.get("symbol") != symbol
        ]
        if len(self._data["pool"]) < before:
            self._save()
            logger.info("已删除股票: %s", symbol)
        else:
            logger.warning("股票不在池中, 无法删除: %s", symbol)

    def set_tick(
        self, symbol: str, tick: str, value: bool, source: str = "manual"
    ) -> None:
        """打/撤 TICK 标记 (source: manual/pipeline).

        设置 is_daily_buy / is_daily_sell 为 True 时自动 is_fixed=True;
        取消时自动 is_fixed=False (ARCH §5.16.4)。

        Args:
            symbol: 股票代码。
            tick: TICK 字段名 (TICK_FIELDS 之一)。
            value: True=打标, False=撤标。
            source: 标记来源 (manual/pipeline1/pipeline2)。

        Raises:
            ValueError: tick 不在 TICK_FIELDS 中, 或股票不在池中。
        """
        if tick not in TICK_FIELDS:
            raise ValueError(f"未知 TICK 标记: {tick}, 可选: {TICK_FIELDS}")
        rec = self._find(symbol)
        if rec is None:
            raise ValueError(f"股票不在池中: {symbol}")
        rec.setdefault("ticks", _default_ticks())[tick] = bool(value)
        if tick in _FIXING_TICKS:
            rec["is_fixed"] = bool(value)
        rec.setdefault("ticks_history", []).append(
            {
                "tick": tick,
                "value": bool(value),
                "source": source,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._save()
        logger.info("TICK 变更: %s %s=%s (%s)", symbol, tick, value, source)

    def set_allocation(self, symbol: str, percentage: float) -> None:
        """设置资金分配比例 (仅 is_daily_buy 股票, 合计 ≤100%).

        Args:
            symbol: 股票代码。
            percentage: 分配百分比 (0~100)。

        Raises:
            ValueError: 股票不在池中 / 未标记 is_daily_buy /
                        比例越界 / 合计超过 100%。
        """
        rec = self._find(symbol)
        if rec is None:
            raise ValueError(f"股票不在池中: {symbol}")
        if not rec.get("ticks", {}).get("is_daily_buy", False):
            raise ValueError(f"仅日线买入股票可设置资金分配: {symbol}")
        pct = float(percentage)
        if pct < 0.0 or pct > 100.0:
            raise ValueError(f"分配比例须在 0~100 之间: {pct}")
        others_total = sum(
            float(r.get("percentage_allocation", 0.0))
            for r in self._data["pool"]
            if r.get("symbol") != symbol
            and r.get("ticks", {}).get("is_daily_buy", False)
        )
        if others_total + pct > 100.0:
            raise ValueError(
                f"分配合计将超过 100% (其他 {others_total:.1f}% + 本次 {pct:.1f}%)"
            )
        rec["percentage_allocation"] = pct
        self._save()
        logger.info(
            "资金分配: %s -> %.1f%% (合计 %.1f%%)", symbol, pct, others_total + pct
        )

    def get_buy_allocation_total(self) -> float:
        """当前日线买入股票分配比例合计."""
        return float(
            sum(
                float(r.get("percentage_allocation", 0.0))
                for r in self._data["pool"]
                if r.get("ticks", {}).get("is_daily_buy", False)
            )
        )

    def get_fixed_stocks(self) -> List[str]:
        """返回固定保留股票 (is_fixed=True)."""
        return [r["symbol"] for r in self._data["pool"] if r.get("is_fixed", False)]

    def update_from_pipeline1(self, new_pool: List[dict]) -> None:
        """Pipeline 1 每日刷新: 更新股票池但保留 is_fixed 股票.

        规则 (ARCH §5.16.3):
            - is_fixed 股票整体保留 (score/ticks/allocation 均不覆盖)
            - 非固定且在新池中的股票: 更新 score/prob_up/pct_up, 保留手工 TICK
            - 非固定且不在新池中的股票: 移除 (is_watchlist 自选股保留)
            - 新池中新增股票: ticks 全部初始化为 False

        Args:
            new_pool: Pipeline 1 产出的新股票池
                      [{symbol, name?, score, prob_up, pct_up, ...}]。
        """
        new_by_symbol = {p["symbol"]: p for p in new_pool}
        old_by_symbol = {r["symbol"]: r for r in self._data["pool"]}

        merged: List[dict] = []
        # 1. 保留固定股票 + 不在新池中的自选股
        for rec in self._data["pool"]:
            if rec.get("is_fixed", False):
                merged.append(rec)
            elif (
                rec.get("ticks", {}).get("is_watchlist", False)
                and rec["symbol"] not in new_by_symbol
            ):
                merged.append(rec)

        # 2. 新池股票: 更新或新增
        for symbol, p in new_by_symbol.items():
            old = old_by_symbol.get(symbol)
            if old is not None and old.get("is_fixed", False):
                continue  # 固定股票已在上方保留, 不覆盖
            if old is not None:
                old["score"] = float(p.get("score", old.get("score", 0.0)))
                old["prob_up"] = float(p.get("prob_up", old.get("prob_up", 0.0)))
                old["pct_up"] = float(p.get("pct_up", old.get("pct_up", 0.0)))
                if p.get("name"):
                    old["name"] = p["name"]
                merged.append(old)
            else:
                merged.append(
                    self._new_record(
                        symbol,
                        added_by="pipeline1",
                        name=p.get("name", ""),
                        score=p.get("score", 0.0),
                        prob_up=p.get("prob_up", 0.0),
                        pct_up=p.get("pct_up", 0.0),
                    )
                )

        self._data["pool"] = merged
        self._save()
        logger.info(
            "Pipeline1 刷新完成: 池内 %d 只 (固定 %d 只)",
            len(merged),
            len(self.get_fixed_stocks()),
        )
