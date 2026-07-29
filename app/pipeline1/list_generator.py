"""
Pipeline-1 每日选股清单生成器 (DESIGN §14.5, §14.6, [E7] 动态准入)
====================================================================
[E7] 动态准入: 1~15 只, 日度 14:50 前输出, 隔夜挂单.
[E11] Bear 模式: 连跌 3 日触发, 收紧准入 (prob_up≥0.65, ret≥3×COST), 半仓.
[E10] 破净资产: SSE 破净比 > 12% → 全仓.
[P25.1] list_mode = 'normal' / 'bear' / 'value' (E10, 破净价值)
[E2] 痛苦预警: pain_prob > 0.5 → 剔除 (在 score 中自然惩罚, 不清除条目)
[E1] 分位数: pred_q10..q90 + uncertainty_width 列传递至清单以供仓位决策
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.2"
TOP_N = 15
MAX_PER_INDUSTRY = 4
COMPOUND_W = (0.45, 0.35, 0.30)  # 1d/3d/5d
HOLDING_BONUS = 0.2
# B3: Holding Bonus 按持仓天数衰减 day1=1.0/day2=0.5/day3=0.0
HOLDING_DAY_WEIGHTS = {0: 0.0, 1: 1.0, 2: 0.5, 3: 0.0}
BASE_RATE_WINDOW = 20  # B4: base_rate 滚动窗口
# [E10] 破净资产阈值
SSE_BREAK_PCT_THRESHOLD = 0.12


SCHEMA_FIELDS = [
    "symbol",
    "board",
    "pred_ret_1d",
    "pred_ret_3d",
    "pred_ret_5d",
    "prob_up",
    "momentum",
    "consensus_score",
    "signal_conflict",
    "is_limit_up_close",
    "is_one_word_limit",
    "market_state",
    "score",
    # V1.2 新增 (E1/E2/公告/分布权重)
    "pred_q10",
    "pred_q50",
    "pred_q90",
    "uncertainty_width",
    "pain_prob",
    "announce_score",
    "weight",
    "schema_version",
]


@dataclass
class MarketEnv:
    """大盘环境 (D18 空仓触发输入, E11 Bear 模式)."""

    hs300_drop_today: float = 0.0
    count_limit_down_market: int = 0
    hs300_consecutive_down: int = 0
    bear_mode: bool = False
    sse_break_pct: float = 0.0


def _is_bear(env) -> bool:
    """E11 Bear 模式判定: 大盘连跌 3 日."""
    if not isinstance(env, MarketEnv):
        return False
    return env.bear_mode


class ListGenerator:
    """每日动态准入 0~15 只清单生成 [E7].

    Args:
        entry_prob: 正常状态 prob_up 准入门槛 [E7] (bear 自动收紧至 0.65 [E11])
        entry_ret_mult: 正常状态 pred_ret_1d 准入门槛 = mult×COST (bear 3×)
    """

    def __init__(self, entry_prob: float = 0.60, entry_ret_mult: float = 2.0):
        self._base_rate_history: list[float] = []
        self.entry_prob = entry_prob
        self.entry_ret_mult = entry_ret_mult
        self.entry_prob_bear = 0.65  # [E11] 准入线 bear 收紧
        self.entry_ret_mult_bear = 3.0
        self._prob_pctile = 0.80  # 若绝对阈值不可达, 回退取 prob_up 前 20%

    # ---------------- 分布权重 ----------------
    @staticmethod
    def _compute_weights(df: pd.DataFrame) -> pd.Series:
        """[E1] 基于 uncertainty_width 的分布权重, 单票上限 0.10.

        uncertainty_width 越大 → 预测越不可靠 → 权重越低.
        若无 uncertainty_width 列则等权分配.
        """
        n = len(df)
        if n == 0:
            return pd.Series(dtype=float)
        if "uncertainty_width" in df.columns:
            # 逆权重: uncertainty_width 越大 → 权重越低
            inv_uw = 1.0 / df["uncertainty_width"].clip(1e-6)
            raw = inv_uw / inv_uw.sum()
        else:
            raw = pd.Series(1.0 / n, index=df.index)
        # 单票上限 0.10, 不重新归一化 (余下留现金)
        return raw.clip(upper=0.10)

    # ---------------- 排序分 ----------------
    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """compound_ret = 0.5*pred_1d + 0.35*pred_3d + 0.15*pred_5d
        [V3.7] rank_score 存在时: score = rank_score × (1 + 0.3·tanh(compound×100))
               × (prob_up / base_rate); 否则回退 V3.5 公式 compound_ret × prob_adjust.
        [E2] 痛苦惩罚: score × (1 - 0.5×pain_prob)  (pain_prob=0.3 → ×0.85)
        [公告] score × (1 + 0.3×announce_score)  (安全网 #17)
        adjusted = score + 0.2 * holding_day_weight * is_in_yesterday_list  (B3 衰减)"""
        w1, w3, w5 = COMPOUND_W
        df = df.copy()
        df["compound_ret"] = (
            w1 * df["pred_ret_1d"] + w3 * df["pred_ret_3d"] + w5 * df["pred_ret_5d"]
        )
        # B4: base_rate = 20 日滚动均值 (原单日均值日间波动大致 score 尺度不稳)
        daily_mean = float(df["prob_up"].mean())
        self._base_rate_history.append(daily_mean)
        recent = self._base_rate_history[-BASE_RATE_WINDOW:]
        base_rate = float(pd.Series(recent).mean())
        base_rate = base_rate if base_rate > 1e-6 else 1.0
        df["base_rate"] = base_rate
        prob_adjust = df["prob_up"] / base_rate
        use_rank = False
        if "rank_score" in df.columns:
            rank_std = float(df["rank_score"].std())
            if rank_std > 1e-6:
                # [V3.7 排序分公式] LambdaRank 排序分 × compound 修正 × prob_adjust
                df["score"] = (
                    df["rank_score"]
                    * (1 + 0.3 * np.tanh(df["compound_ret"] * 100))
                    * prob_adjust
                )
                use_rank = True
        if not use_rank:
            # V3.5 回退: compound_ret × prob_adjust (简单有效)
            df["score"] = df["compound_ret"] * prob_adjust
        # [E2] 痛苦惩罚
        if "pain_prob" in df.columns:
            pain_penalty = 1 - 0.5 * df["pain_prob"].fillna(0).clip(0, 1)
            df["score"] = df["score"] * pain_penalty
        # [E1] 分布不确定性惩罚: uncertainty_width 过大 → 预测不可靠
        if "uncertainty_width" in df.columns:
            uw = df["uncertainty_width"].fillna(0)
            uw_penalty = 1 - 0.2 * (uw / uw.quantile(0.95).clip(1e-6))
            df["score"] = df["score"] * uw_penalty.clip(0.5, 1.0)
        # 公告因子
        if "announce_score" in df.columns:
            df["score"] = df["score"] * (1 + 0.3 * df["announce_score"].fillna(0))
        # B3: Holding Bonus 衰减
        if "is_in_yesterday_list" in df.columns and "holding_day" in df.columns:
            hw = df["holding_day"].map(HOLDING_DAY_WEIGHTS).fillna(0)
            df["score"] = df["score"] + HOLDING_BONUS * hw * df.get(
                "is_in_yesterday_list", 0
            )
        # [E7] 空仓触发时 score 列保持不变, 由 emit 决定是否输出空清单
        # 行业排名归因 (看板列, 不影响排序)
        if "industry" in df.columns:
            df["industry_rank"] = df.groupby("industry")["score"].rank(
                ascending=False, pct=True
            )
        return df

    # ---------------- 准入 ---------------
    @staticmethod
    def _estimate_cost(
        df: pd.DataFrame, cost_col: str = "COST", fallback: float = 0.002
    ) -> float:
        """COST = 滑点+佣金+印花税, 取 20 日滚动均值 (B4)."""
        if cost_col in df.columns:
            cost = float(df[cost_col].replace(0, np.nan).mean())
            if cost > 0:
                return cost
        return fallback

    # ---------------- E7 准入门槛 (动态) ----------
    def entry_filter(
        self,
        df: pd.DataFrame,
        market_state: str = "range",
        cost: float | None = None,
    ) -> pd.DataFrame:
        """[E7] 动态准入过滤.

        Args:
            df: compute_scores 输出 (含 score, prob_up, pred_ret_1d)
            market_state: 'range' / 'bear'
            cost: 交易成本; None → 自动估计

        Returns:
            过滤后的 DataFrame (entry_prob 松绑, 关注 score 排名)
        """
        if len(df) == 0:
            return df
        cost = cost if cost is not None else self._estimate_cost(df)
        # 准入门槛 (E7 §1): prob_up > entry_prob 且 pred_ret_1d > mult × cost
        # [E11] Bear 模式收紧
        is_bear = market_state == "bear"
        prob_thresh = self.entry_prob_bear if is_bear else self.entry_prob
        ret_mult = self.entry_ret_mult_bear if is_bear else self.entry_ret_mult
        ret_thresh = ret_mult * cost
        passed = (df["prob_up"] >= prob_thresh) & (df["pred_ret_1d"] >= ret_thresh)
        # 如果绝对阈值过滤后不足 3 只, 回退取 prob_up + pred_ret_1d 前 80% 分位
        if passed.sum() < 3:
            pctile_prob = float(df["prob_up"].quantile(self._prob_pctile))
            pctile_ret = float(df["pred_ret_1d"].quantile(self._prob_pctile))
            passed = (df["prob_up"] >= pctile_prob) & (df["pred_ret_1d"] >= pctile_ret)
            logger.info(
                "E7 准入回退: prob_up 前 %.0f%% (%.4f) + pred_ret_1d 前 %.0f%% (%.4f), %d 只通过",
                self._prob_pctile * 100,
                pctile_prob,
                self._prob_pctile * 100,
                pctile_ret,
                int(passed.sum()),
            )
        # [E2] 痛苦预警: pain_prob > 0.5 直接剔除 (安全网 #16)
        if "pain_prob" in df.columns:
            passed &= df["pain_prob"].fillna(0) <= 0.5
        return df[passed].copy()

    # ---------------- 最终清单 ---------------
    def emit(
        self,
        candidates: pd.DataFrame,
        env=None,
        market_state: str = "range",
    ) -> dict:
        """输出最终清单.

        Args:
            candidates: predict() 输出 (含 symbol, pred_ret_*, prob_up, score)
            env: MarketEnv 大盘环境 (含 bear_mode, sse_break_pct)
            market_state: 'range' / 'bear'

        Returns:
            {'mode': 'normal'|'bear'|'value'|'empty',
             'list': DataFrame (按 score 降序, 前 TOP_N 只),
             'cap_position': float 仓位比例,
             'empty': bool}
        """
        if len(candidates) == 0:
            return {
                "mode": "empty",
                "list": pd.DataFrame(),
                "cap_position": 0.0,
                "empty": True,
            }
        # 计算排序分
        scored = self.compute_scores(candidates)
        # 准入过滤
        passed = self.entry_filter(scored, market_state=market_state)
        if len(passed) == 0:
            logger.warning("E7 准入过滤后无候选, 输出空清单")
            return {
                "mode": "empty",
                "list": pd.DataFrame(),
                "cap_position": 0.0,
                "empty": True,
            }
        # 按 score 降序取 TOP_N (行业分散在 list_generator 层面处理)
        passed = passed.sort_values("score", ascending=False)
        final = passed.head(TOP_N).reset_index(drop=True)
        # 决定 mode
        mode = market_state if market_state in ("bear", "value") else "normal"
        # 仓位: [E11] bear 半仓; [E10] 破净价值全仓
        cap_position = 0.5 if mode == "bear" else 1.0
        logger.info(
            "清单: mode=%s, %d 只, cap=%.2f, top_score=%.4f",
            mode,
            len(final),
            cap_position,
            float(final["score"].max()) if len(final) else 0,
        )
        # [E1] 分布权重: 基于 uncertainty_width 的逆权重, 单票上限 0.10
        final["weight"] = self._compute_weights(final)
        # 返回标准 schema (V1.2)
        return {
            "mode": mode,
            "list": final,
            "cap_position": cap_position,
            "empty": False,
            "schema_version": SCHEMA_VERSION,
        }


# ============================================================
# 清单溯源追踪 (源自 a-share-selection-strategy, provenance)
# ============================================================
@dataclass
class ProvenanceTracker:
    """记录每只候选股的数据来源/模型版本/计算时间戳."""

    _records: dict[str, dict] = field(default_factory=dict)

    def record(
        self,
        symbol: str,
        data_source: str = "",
        model_tag: str = "",
        feature_version: str = "",
    ) -> None:
        self._records[symbol] = {
            "data_source": data_source,
            "model_tag": model_tag,
            "feature_version": feature_version,
        }

    def get(self, symbol: str) -> dict:
        return self._records.get(symbol, {})

    def to_frame(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        rows = [{"symbol": k, **v} for k, v in self._records.items()]
        return pd.DataFrame(rows)


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
def check_invalidation(
    open_gap_pct: float,
    limit_down_within_30min: bool,
    sector_drop_pct: float,
    surge_5min_pct: float,
) -> str | None:
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
