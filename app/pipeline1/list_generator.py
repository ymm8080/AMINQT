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

import datetime
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .label_engine import LABEL_WEIGHTS

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.4"
TOP_N = 15
MAX_PER_INDUSTRY = 4
COMPOUND_W = tuple(LABEL_WEIGHTS[k] for k in (1, 2, 3, 5, 10))  # 1d/2d/3d/5d/10d
HOLDING_BONUS = 0.2
# B3: Holding Bonus 按持仓天数衰减 day1=1.0/day2=0.5/day3=0.0
HOLDING_DAY_WEIGHTS = {0: 0.0, 1: 1.0, 2: 0.5, 3: 0.0}
BASE_RATE_WINDOW = 20  # B4: base_rate 滚动窗口
# [E10] 破净资产阈值
SSE_BREAK_PCT_THRESHOLD = 0.12
# 动量阈值 (V3.4 陷阱修复)
FW_HARD = -0.03  # 预测跌幅 > 3% → 强制 low
FW_EPS = 0.001  # 预测值太小无法算比率
RATIO_UP = 1.0
RATIO_DOWN = 0.8
# D18 空仓触发
HS300_DROP_EMPTY = 0.03
MARKET_LIMIT_DOWN_EMPTY = 50
HS300_CONSEC_DOWN_CAP = 3
CAP_POSITION_REDUCED = 0.3


SCHEMA_FIELDS = [
    "symbol",
    "board",
    "day_change",
    "pred_ret_1d",
    "pred_ret_2d",
    "pred_ret_3d",
    "pred_ret_5d",
    "pred_ret_10d",
    "prob_up",
    "prob_up_2d",
    "prob_up_3d",
    "prob_up_5d",
    "prob_up_10d",
    "momentum",
    "consensus_score",
    "signal_conflict",
    "is_limit_up_close",
    "is_one_word_limit",
    "market_state",
    "score",
    "compound_ret",
    "compound_prob",
    # V1.2 新增 (E1/E2/公告/分布权重)
    "pred_q10",
    "pred_q50",
    "pred_q90",
    # E7 闸3 用 2d/3d/5d 中位数 (2026-08-05 用户定案, 替代 1d pred_q50)
    "pred_q50_2d",
    "pred_q50_3d",
    "pred_q50_5d",
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
        """compound_ret = LABEL_WEIGHTS 加权 (1/2/3/5/10d; 弃 2d, 10d 权重最高).
        [V3.7] rank_score 存在时: score = rank_score × (1 + 0.3·tanh(compound×100))
               × (prob_up / base_rate); 否则回退 V3.5 公式 compound_ret × prob_adjust.
        [E2] 痛苦惩罚: score × (1 - 0.5×pain_prob)  (pain_prob=0.3 → ×0.85)
        [公告] score × (1 + 0.3×announce_score)  (安全网 #17)
        adjusted = score + 0.2 * holding_day_weight * is_in_yesterday_list  (B3 衰减)"""
        w1, w2, w3, w5, w10 = COMPOUND_W
        df = df.copy()
        # 只对"存在且有有效值"的视界加权并按权重和归一 (旧 bundle 缺 10d 时精确回退,
        # 与 predictor.composite_score 同款 present-weights 模式, 避免 NaN 污染 compound_ret).
        ret_cols = {
            1: "pred_ret_1d",
            2: "pred_ret_2d",
            3: "pred_ret_3d",
            5: "pred_ret_5d",
            10: "pred_ret_10d",
        }
        w_map = {1: w1, 2: w2, 3: w3, 5: w5, 10: w10}
        present = {
            k: c for k, c in ret_cols.items() if c in df.columns and df[c].notna().any()
        }
        tw = sum(w_map[k] for k in present)
        if tw > 1e-12:
            df["compound_ret"] = sum(w_map[k] * df[c] for k, c in present.items()) / tw
        else:
            df["compound_ret"] = 0.0
        # [多视界] 加权概率 (t+1 权重=0; 旧 bundle 缺 2/3/5/10d 概率列/有效值 → 精确回退 prob_up)
        prob_cols = [f"prob_up_{k}d" for k in (2, 3, 5, 10)]
        p_present = [c for c in prob_cols if c in df.columns and df[c].notna().any()]
        if len(p_present) == len(prob_cols):
            df["compound_prob"] = (
                w1 * df["prob_up"]
                + w2 * df["prob_up_2d"]
                + w3 * df["prob_up_3d"]
                + w5 * df["prob_up_5d"]
                + w10 * df["prob_up_10d"]
            )
        else:
            df["compound_prob"] = df["prob_up"]
        # B4: base_rate = 20 日滚动均值 (compound_prob 加权概率)
        daily_mean = float(df["compound_prob"].mean())
        self._base_rate_history.append(daily_mean)
        recent = self._base_rate_history[-BASE_RATE_WINDOW:]
        base_rate = float(pd.Series(recent).mean())
        base_rate = base_rate if base_rate > 1e-6 else 1.0
        df["base_rate"] = base_rate
        prob_adjust = df["compound_prob"] / base_rate
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
            # V3.5 回退: compound_ret 横截面 rank(pct=True) × prob_adjust (t+1 权重已 0)
            df["score"] = df["compound_ret"].rank(pct=True) * prob_adjust
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
        if "is_in_yesterday_list" in df.columns:
            if "holding_day" in df.columns:
                hw = df["holding_day"].map(HOLDING_DAY_WEIGHTS).fillna(0)
            else:
                # 向后兼容: 无 holding_day 列时, is_in_yesterday_list=1 视为 day1 (weight=1.0)
                hw = df.get("is_in_yesterday_list", 0)
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

    # ---------------- 行业集中度 ----------------
    @staticmethod
    def apply_industry_limit(
        ranked: pd.DataFrame, max_per_industry: int = MAX_PER_INDUSTRY
    ) -> pd.DataFrame:
        """同一申万一级行业 <= 4 只, 超出顺延; 顺延后 < 10 只则接受不足 (不强凑数)."""
        if "industry" not in ranked.columns:
            return ranked
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
    def check_empty_triggers(env) -> tuple[bool, float]:
        """返回 (是否强制空清单, 仓位上限).

        沪深300 跌>3% 或 全市场跌停>50 → 空清单;  连跌3日 → 仓位上限 30% (仅 Top 5).
        """
        if not isinstance(env, MarketEnv):
            return False, 1.0
        if env.hs300_drop_today > HS300_DROP_EMPTY:
            logger.error(
                "D18 空仓触发: 沪深300 当日跌幅 %.1f%% > 3%%",
                env.hs300_drop_today * 100,
            )
            return True, 0.0
        if env.count_limit_down_market > MARKET_LIMIT_DOWN_EMPTY:
            logger.error(
                "D18 空仓触发: 全市场跌停 %d 只 > 50", env.count_limit_down_market
            )
            return True, 0.0
        if env.hs300_consecutive_down >= HS300_CONSEC_DOWN_CAP:
            logger.warning(
                "D18 降仓: 沪深300 连跌 %d 日, 仓位上限 30%%",
                env.hs300_consecutive_down,
            )
            return False, CAP_POSITION_REDUCED
        return False, 1.0

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
        """[E7] 计算型准入过滤 (2026-07-26 用户裁决: 门槛由数据算出, 不设绝对常数).

        老常数闸 (prob>=entry_prob 且 pred_ret_1d>=mult*COST) 在净收益标签口径下
        82/82 天全不可达 (成本双算 + Huber 收缩), 评估见 scripts/eval_gate_options.py.
        新闸 (全部基于当日预测数据计算):
          1. prob_up > base_rate (B4 20日滚动基准率; bear 按 entry_prob_bear/
             entry_prob 参数比率收紧, 不引入新常数)
          2. compound_ret (1/2/3/5/10d 净预测按 COMPOUND_W 加权, 弃 2d/10d 最高) > 0 —
             净预期为正, 成本已在训练标签口径内扣除
          3. pred_q50_2d/3d/5d > 0 (E1 2d/3d/5d 可执行视界中位数均为正;
             2026-08-05 用户定案: 用 2d/3d/5d 中位数替代 1d, 1d 不可执行易误杀;
             旧 bundle 无 2d/3d/5d 列时回退 1d pred_q50)
          4. bear 额外要求 pred_ret_1d > 0 (最近端净预期为正) [E11]
        符合票可能为 0 — 这是特性不是故障.
        escape hatch (测试/研究): entry_prob<=0 跳过 prob 闸, entry_ret_mult<=0 跳过边际闸.

        Args:
            df: compute_scores 输出 (含 score, prob_up, pred_ret_1d, base_rate, compound_ret)
            market_state: 'range' / 'bear'
            cost: 交易成本 (保留签名兼容; 计算闸不直接使用绝对成本阈值)
        """
        if len(df) == 0:
            return df
        # cost 仅用于兼容旧调用方; 计算闸不再设绝对成本倍数门槛
        if cost is None:
            self._estimate_cost(df)
        is_bear = market_state == "bear"
        ok = pd.Series(True, index=df.index)
        # 1. 加权概率 > base_rate (bear 按声明参数比率收紧, 无新常数)
        if self.entry_prob > 0:
            base = (
                df["base_rate"] if "base_rate" in df.columns else df["prob_up"].mean()
            )
            ratio = self.entry_prob_bear / self.entry_prob if is_bear else 1.0
            cp = df["compound_prob"] if "compound_prob" in df.columns else df["prob_up"]
            ok &= cp > base * ratio
        # 2/3/4. 净预期为正 (compound_ret > 0; pred_q50 > 0; bear 额外 pred_ret_1d > 0)
        if self.entry_ret_mult > 0:
            if "compound_ret" in df.columns:
                compound = df["compound_ret"]
            else:
                # present-weights 回退 (与 compute_scores 同款): 旧 bundle 缺 10d
                # 或其他视界列时, 只对存在且有有效值的视界加权并按权重和归一, 防 KeyError.
                w1, w2, w3, w5, w10 = COMPOUND_W
                ret_cols = {
                    1: "pred_ret_1d",
                    2: "pred_ret_2d",
                    3: "pred_ret_3d",
                    5: "pred_ret_5d",
                    10: "pred_ret_10d",
                }
                w_map = {1: w1, 2: w2, 3: w3, 5: w5, 10: w10}
                present = {
                    k: c
                    for k, c in ret_cols.items()
                    if c in df.columns and df[c].notna().any()
                }
                tw = sum(w_map[k] for k in present)
                if tw > 1e-12:
                    compound = sum(w_map[k] * df[c] for k, c in present.items()) / tw
                else:
                    compound = pd.Series(0.0, index=df.index)
            ok &= compound > 0
            # 闸3 (2026-08-05): 2d/3d/5d 可执行视界中位数均须为正; 回退 1d pred_q50 (旧 bundle)
            if all(
                c in df.columns for c in ("pred_q50_2d", "pred_q50_3d", "pred_q50_5d")
            ):
                ok &= (
                    (df["pred_q50_2d"].fillna(compound) > 0)
                    & (df["pred_q50_3d"].fillna(compound) > 0)
                    & (df["pred_q50_5d"].fillna(compound) > 0)
                )
            elif "pred_q50" in df.columns and df["pred_q50"].notna().any():
                ok &= df["pred_q50"].fillna(compound) > 0
            if is_bear:
                ok &= df["pred_ret_1d"] > 0
        # [E2] 痛苦预警: pain_prob > 0.5 直接剔除 (安全网 #16)
        if "pain_prob" in df.columns:
            ok &= df["pain_prob"].fillna(0) <= 0.5
        passed = df[ok]
        if len(passed) == 0:
            logger.warning(
                "E7 计算型准入: 0 只过闸 (prob>基准率%s, 净预期>0), 今日空清单",
                "×bear收紧" if is_bear else "",
            )
        return passed.copy()

    # ---------------- 最终清单 ---------------
    @staticmethod
    def _scan_ref_date(candidates: pd.DataFrame) -> pd.Timestamp:
        """FINAL STOCK SCAN 锚点日期: candidates 的 date 列最大值, 缺失则用今日."""
        if "date" in candidates.columns and candidates["date"].notna().any():
            return pd.Timestamp(candidates["date"].max())
        return pd.Timestamp(datetime.date.today())

    @staticmethod
    def _rank_by_magnitude(df: pd.DataFrame) -> pd.DataFrame:
        """E7 准入后按预测幅度排序: pred_ret_10d 降序 (2026-08-07 定案).

        回测依据: 并行 250d OOS 纯 10d 幅度排名在 close-to-close 实得口径下赢纯 3d
        (main 5d实得 +1.02% vs +0.06%) 与 T3+T5 组合 (+0.16%); 10d 挑的是全程强势票,
        前 3-5 天实得同样更高, 非"后程发力" (diag_10d_point_ret_20260807_100807).
        旧混合排名 (0.5×norm(pred_ret_3d)+0.5×norm(prob_up_3d)) 降级为影子对照组
        (prediction_shadow), 真实数据 1~2 月后裁决, 打脸则 revert.
        缺 pred_ret_10d (旧 bundle) → 级联回退 pred_ret_5d → pred_ret_3d → score 降序.
        """
        for col in ("pred_ret_10d", "pred_ret_5d", "pred_ret_3d"):
            if col in df.columns and df[col].notna().any():
                return df.sort_values(col, ascending=False)
        return df.sort_values("score", ascending=False)

    def emit(
        self,
        candidates: pd.DataFrame,
        env=None,
        market_state: str = "range",
        ref_date: str | None = None,
    ) -> dict:
        """输出最终清单.

        Args:
            candidates: predict() 输出 (含 symbol, pred_ret_*, prob_up, score)
            env: MarketEnv 大盘环境 (含 bear_mode, sse_break_pct)
            market_state: 'range' / 'bear'
            ref_date: 名单生成日 (T 日), 用于 FINAL STOCK SCAN 窗口锚定;
                缺省从 candidates['date'] 推导, 再无则用今日.

        Returns:
            {'mode': 'normal'|'bear'|'value'|'empty',
             'list': DataFrame (按 score 降序, 前 TOP_N 只),
             'cap_position': float 仓位比例,
             'empty': bool}
        """
        # [D18] 空仓触发: HS300 单日跌幅 > 3% / 全市场跌停 > 50 / 连跌 3 日降仓
        empty, cap = self.check_empty_triggers(env)
        if empty or len(candidates) == 0:
            return {
                "mode": "empty",
                "list": pd.DataFrame(columns=SCHEMA_FIELDS),
                "cap_position": 0.0 if empty else cap,
                "empty": True,
                "schema_version": SCHEMA_VERSION,
            }
        if len(candidates) == 0:
            return {
                "mode": "empty",
                "list": pd.DataFrame(columns=SCHEMA_FIELDS),
                "cap_position": 0.0,
                "empty": True,
                "schema_version": SCHEMA_VERSION,
            }
        # 计算排序分
        scored = self.compute_scores(candidates)
        # 准入过滤
        passed = self.entry_filter(scored, market_state=market_state)
        if len(passed) == 0:
            logger.warning("E7 准入过滤后无候选, 输出空清单")
            return {
                "mode": "empty",
                "list": pd.DataFrame(columns=SCHEMA_FIELDS),
                "cap_position": 0.0,
                "empty": True,
                "schema_version": SCHEMA_VERSION,
            }
        # 按预测幅度排序取 TOP_N (2026-08-07: 纯 pred_ret_3d 幅度, 回测赢 d3 混合; 行业分散在清单层面)
        passed = self._rank_by_magnitude(passed)
        # 行业集中度限制: 同一行业 <= MAX_PER_INDUSTRY 只
        final = self.apply_industry_limit(passed).reset_index(drop=True)
        # D18 降仓 → 仅 Top 5; 正常 → Top 15
        top_n = TOP_N if cap >= 1.0 else 5
        final = final.head(top_n)
        # [FINAL STOCK SCAN] 风控过滤: 近一周有大宗交易的股票不买 (用户定案 2026-08-03)
        try:
            from .risk_overlays import (
                block_trade_recent_scan,
                share_float_upcoming_scan,
            )

            scan_ref = (
                pd.Timestamp(ref_date) if ref_date else self._scan_ref_date(candidates)
            )
            excluded = block_trade_recent_scan(final["symbol"].tolist(), scan_ref)
            if excluded:
                final = final[~final["symbol"].isin(excluded)].reset_index(drop=True)
                logger.warning(
                    "FINAL STOCK SCAN: 名单剔除 %d 只 (剩 %d 只)",
                    len(excluded),
                    len(final),
                )
            # 二次扫描: 近月有大规模解禁(解锁)的股票不买
            excluded = share_float_upcoming_scan(final["symbol"].tolist(), scan_ref)
            if excluded:
                final = final[~final["symbol"].isin(excluded)].reset_index(drop=True)
                logger.warning(
                    "FINAL STOCK SCAN: 解禁剔除 %d 只 (剩 %d 只)",
                    len(excluded),
                    len(final),
                )
        except Exception as exc:
            logger.warning("FINAL STOCK SCAN: 失败, 放行名单: %s", exc)
        # 动量持续性 (盈亏防火墙)
        if len(final) and {"pred_ret_1d", "pred_ret_3d", "pred_ret_5d"} <= set(
            final.columns
        ):
            final["momentum"] = [
                self.compute_momentum(a, b, c)
                for a, b, c in zip(
                    final["pred_ret_1d"],
                    final["pred_ret_3d"],
                    final["pred_ret_5d"],
                    strict=False,
                )
            ]
        # 决定 mode
        mode = market_state if market_state in ("bear", "value") else "normal"
        # 仓位: [E11] bear 半仓; [E10] 破净价值全仓; [D18] 连跌降仓 30%
        if cap < 1.0:
            cap_position = cap  # D18 降仓
        else:
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
        if "prob_up" in final.columns:
            final["prob_up"] = final["prob_up"].round(3)
        for col in ("prob_up_2d", "prob_up_3d", "prob_up_5d", "prob_up_10d"):
            if col in final.columns:
                final[col] = final[col].round(3)
        for col in ("is_limit_up_close", "is_one_word_limit"):
            if col not in final.columns:
                final[col] = 0
        for col in (
            "day_change",
            "pred_ret_10d",
            "prob_up_2d",
            "prob_up_3d",
            "prob_up_5d",
            "prob_up_10d",
            "pred_q10",
            "pred_q50",
            "pred_q90",
            "pred_q50_2d",
            "pred_q50_3d",
            "pred_q50_5d",
            "uncertainty_width",
            "pain_prob",
            "announce_score",
            "momentum",
            "consensus_score",
            "signal_conflict",
        ):
            if col not in final.columns:
                final[col] = np.nan
        final["market_state"] = market_state
        final["schema_version"] = SCHEMA_VERSION
        return {
            "mode": mode,
            "list": final[SCHEMA_FIELDS].reset_index(drop=True),
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
