# -*- coding: utf-8 -*-
"""
清单生成器 (DESIGN §14.4, PIPELINE1_V3.5 §四)
=================================================
每日清单 = 全量推荐池 Top 15 (固定), schema version="1.0".
排序分: compound_ret × prob/base_rate + Holding Bonus.
动量: 盈亏防火墙 + 日均衰减比率 (V3.5 补丁, 修复 C 场景悖论).
空仓触发 (D18, 安全网 #12) + 清单推送失败三档降级 + 失效条件传递.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
TOP_N = 15
MAX_PER_INDUSTRY = 4
MIN_LIST_SIZE = 10          # 顺延后不足则接受不足
COMPOUND_W = (0.5, 0.35, 0.15)   # 1d/3d/5d
HOLDING_BONUS = 0.2
# 动量阈值
FW_HARD = -0.03             # 预测跌幅 > 3% → 强制 low
FW_EPS = 0.001              # 预测值太小无法算比率
RATIO_UP = 1.0
RATIO_DOWN = 0.8
# D18 空仓触发
HS300_DROP_EMPTY = 0.03
MARKET_LIMIT_DOWN_EMPTY = 50
HS300_CONSEC_DOWN_CAP = 3
CAP_POSITION_REDUCED = 0.3

SCHEMA_FIELDS = ["symbol", "board", "pred_ret_1d", "pred_ret_3d", "pred_ret_5d",
                 "prob_up", "momentum", "consensus_score", "signal_conflict",
                 "is_limit_up_close", "is_one_word_limit", "market_state",
                 "score", "schema_version"]


@dataclass
class MarketEnv:
    """大盘环境 (D18 空仓触发输入)."""
    hs300_drop_today: float = 0.0          # 沪深300 当日跌幅 (正数=跌)
    count_limit_down_market: int = 0       # 全市场跌停家数
    hs300_consecutive_down: int = 0        # 沪深300 连跌天数


class ListGenerator:
    """每日 Top 15 清单生成."""

    # ---------------- 排序分 ----------------
    @staticmethod
    def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
        """compound_ret = 0.5*pred_1d + 0.35*pred_3d + 0.15*pred_5d
        score = compound_ret * (prob_up / base_rate);  base_rate = 当期全池基准胜率
        adjusted = score + 0.2 * is_in_yesterday_list  (Holding Bonus, 目标日均换手 40-60%)"""
        w1, w3, w5 = COMPOUND_W
        df = df.copy()
        df["compound_ret"] = (w1 * df["pred_ret_1d"] + w3 * df["pred_ret_3d"]
                              + w5 * df["pred_ret_5d"])
        base_rate = df["prob_up"].mean()
        base_rate = base_rate if base_rate > 1e-6 else 1.0
        df["base_rate"] = base_rate
        df["score"] = df["compound_ret"] * (df["prob_up"] / base_rate)
        df["score"] = df["score"] + HOLDING_BONUS * df.get("is_in_yesterday_list", 0)
        return df

    # ---------------- 动量持续性 ----------------
    @staticmethod
    def compute_momentum(pred_1d: float, pred_3d: float, pred_5d: float) -> str:
        """盈亏防火墙优先, 否则日均衰减比率 (量纲对齐, 不用绝对值比较).

        pred_1d < -3% → 强制 low;  < 0 → 最高 medium;  |pred_1d| < 0.1% → medium.
        ratio_kd = (pred_kd/k)/pred_1d:  3d>1 且 5d>1 → high;  3d<0.8 → low;  余 medium.
        """
        if pred_1d < FW_HARD:
            return "low"
        if pred_1d < 0:
            return "medium"
        if abs(pred_1d) < FW_EPS:
            return "medium"
        ratio_3d = (pred_3d / 3) / pred_1d
        ratio_5d = (pred_5d / 5) / pred_1d
        if ratio_3d > RATIO_UP and ratio_5d > RATIO_UP:
            return "high"
        if ratio_3d < RATIO_DOWN:
            return "low"
        return "medium"

    # ---------------- 信号一致性 / 冲突 ----------------
    @staticmethod
    def consensus_and_conflict(df: pd.DataFrame) -> pd.DataFrame:
        """consensus_score = 三模型排名均值 (越小越一致);  signal_conflict: 点估计与概率方向冲突."""
        df = df.copy()
        for k in ("1d", "3d", "5d"):
            df[f"_rank_{k}"] = df[f"pred_ret_{k}"].rank(ascending=False)
        df["consensus_score"] = (df["_rank_1d"] + df["_rank_3d"] + df["_rank_5d"]) / 3
        df["signal_conflict"] = (
            ((df["pred_ret_1d"] > 0) & (df["prob_up"] < df["base_rate"]))
            | ((df["pred_ret_1d"] < 0) & (df["prob_up"] > df["base_rate"]))).astype(int)
        return df.drop(columns=["_rank_1d", "_rank_3d", "_rank_5d"])

    # ---------------- 市场状态 ----------------
    @staticmethod
    def market_state(close: float, ma250: float, slope_20d: float) -> str:
        """沪深300 收盘价 vs 250 日均线 + 20 日斜率 (双条件)."""
        if close > ma250 and slope_20d > 0:
            return "bull"
        if close < ma250 and slope_20d < 0:
            return "bear"
        return "range"

    # ---------------- 行业集中度 ----------------
    @staticmethod
    def apply_industry_limit(ranked: pd.DataFrame,
                             max_per_industry: int = MAX_PER_INDUSTRY) -> pd.DataFrame:
        """同一申万一级行业 <= 4 只, 超出顺延; 顺延后 < 10 只则接受不足 (不强凑数)."""
        counts: dict[str, int] = {}
        keep = []
        for _, row in ranked.iterrows():
            ind = row.get("industry", "UNKNOWN")
            if counts.get(ind, 0) < max_per_industry:
                counts[ind] = counts.get(ind, 0) + 1
                keep.append(True)
            else:
                keep.append(False)
        return ranked[keep]

    # ---------------- D18 空仓触发 (安全网 #12) ----------------
    @staticmethod
    def check_empty_triggers(env: MarketEnv) -> tuple[bool, float]:
        """返回 (是否强制空清单, 仓位上限).

        沪深300 跌>3% 或 全市场跌停>50 → 空清单;  连跌3日 → 仓位上限 30% (仅 Top 5).
        """
        if env.hs300_drop_today > HS300_DROP_EMPTY:
            logger.error("D18 空仓触发: 沪深300 当日跌幅 %.1f%% > 3%%", env.hs300_drop_today * 100)
            return True, 0.0
        if env.count_limit_down_market > MARKET_LIMIT_DOWN_EMPTY:
            logger.error("D18 空仓触发: 全市场跌停 %d 只 > 50", env.count_limit_down_market)
            return True, 0.0
        if env.hs300_consecutive_down >= HS300_CONSEC_DOWN_CAP:
            logger.warning("D18 降仓: 沪深300 连跌 %d 日, 仓位上限 30%%", env.hs300_consecutive_down)
            return False, CAP_POSITION_REDUCED
        return False, 1.0

    # ---------------- 总装 ----------------
    def emit(self, candidates: pd.DataFrame, env: MarketEnv | None = None,
             market_state: str = "range") -> dict:
        """生成清单 schema V1.0.

        candidates: 需含 symbol/board/industry/pred_ret_1d/3d/5d/prob_up(校准后)
                    [/is_limit_up_close/is_one_word_limit/is_in_yesterday_list]
        Returns:
            {'list': DataFrame(≤15 行, SCHEMA_FIELDS), 'cap_position': float,
             'empty': bool, 'schema_version': '1.0'}
        """
        env = env or MarketEnv()
        empty, cap = self.check_empty_triggers(env)
        if empty or len(candidates) == 0:
            return {"list": pd.DataFrame(columns=SCHEMA_FIELDS), "cap_position": 0.0,
                    "empty": True, "schema_version": SCHEMA_VERSION}

        df = self.compute_scores(candidates)
        df = self.consensus_and_conflict(df)
        df["momentum"] = [self.compute_momentum(a, b, c) for a, b, c in
                          zip(df["pred_ret_1d"], df["pred_ret_3d"], df["pred_ret_5d"])]
        df["market_state"] = market_state
        df["prob_up"] = df["prob_up"].round(3)
        df = df.sort_values("score", ascending=False)
        df = self.apply_industry_limit(df)
        top = TOP_N if cap >= 1.0 else 5          # D18 降仓 → 仅 Top 5
        df = df.head(top)
        df["schema_version"] = SCHEMA_VERSION
        for col in ("is_limit_up_close", "is_one_word_limit"):
            if col not in df.columns:
                df[col] = 0
        return {"list": df[SCHEMA_FIELDS].reset_index(drop=True),
                "cap_position": cap, "empty": False, "schema_version": SCHEMA_VERSION}


# ============================================================
# 清单推送失败三档降级 (安全网, §14.4)
# ============================================================
@dataclass
class ListDeliveryGuard:
    """1 日失败: 沿用昨日清单(1日)+告警; 连续 2 日: 只卖不买; 连续 3 日: 人工介入."""
    consecutive_failures: int = 0
    last_list: pd.DataFrame | None = field(default=None)

    def on_success(self, lst: pd.DataFrame) -> dict:
        self.consecutive_failures = 0
        self.last_list = lst
        return {"mode": "normal", "list": lst}

    def on_failure(self) -> dict:
        self.consecutive_failures += 1
        n = self.consecutive_failures
        if n == 1:
            logger.error("清单推送失败 (1日): 沿用昨日清单 + 告警")
            return {"mode": "reuse_yesterday", "list": self.last_list}
        if n == 2:
            logger.error("清单推送失败 (连续2日): 只卖不买")
            return {"mode": "sell_only", "list": None}
        logger.critical("清单推送失败 (连续%d日): 人工介入 (检查数据源/模型/服务器)", n)
        return {"mode": "manual_intervention", "list": None}


# ============================================================
# 清单失效条件 (T+1 盘中, 5 分钟模型执行)
# ============================================================
def check_invalidation(open_gap_pct: float, limit_down_within_30min: bool,
                       sector_drop_pct: float, surge_5min_pct: float) -> str | None:
    """任一触发 → 从清单移除, 5 分钟模型不得买入.

    1. 开盘跳空 > ±5% (相对 T 日收盘价)
    2. 开盘后 30 分钟内触发跌停
    3. 板块指数跌幅 > 3% (系统性风险)
    4. 个股开盘后 5 分钟内涨幅 > 7% (防追高)
    """
    if abs(open_gap_pct) > 5.0:
        return f"开盘跳空{open_gap_pct:+.1f}%>±5%"
    if limit_down_within_30min:
        return "开盘30分钟内跌停"
    if sector_drop_pct < -3.0:
        return f"板块指数跌{sector_drop_pct:.1f}%>3%"
    if surge_5min_pct > 7.0:
        return f"开盘5分钟涨{surge_5min_pct:.1f}%>7%"
    return None
