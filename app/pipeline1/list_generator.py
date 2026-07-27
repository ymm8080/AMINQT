"""
清单生成器 (DESIGN §14.4, PIPELINE1_V3.8 §四/§四 ter)
=====================================================
每日清单 = 动态准入 0~15 只 [E7] (不再固定 15 只), schema version="1.2".
排序分: compound_ret × prob/base_rate × (1-0.5×pain_prob)[E2] × 公告调整 + Holding Bonus.
[E1] 分布预测输出 pred_q10/q50/q90 + uncertainty_width; 分布版仓位权重:
     raw_w = pred_q50×prob_up / (ATR_pct × (1+uncertainty_width)), 不确定性越大权重越低.
[E6] liquidity_cap (ADV20×1%, bear 0.5%); [E8] 相关性簇阻断 (簇≤15%, bear 12%).
[E11] bear 收紧: prob 门槛 0.60→0.65, pred_ret 2×成本→3×成本, 单票 10%→7%.
动量: 盈亏防火墙 + 日均衰减比率 (V3.5 补丁, 修复 C 场景悖论).
空仓触发 (D18, 安全网 #12) + 清单推送失败三档降级 + 失效条件传递.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .risk_overlays import (
    CLUSTER_CAP,
    CLUSTER_CAP_BEAR,
    apply_cluster_caps,
    cluster_block,
    liquidity_cap,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.2"
TOP_N = 15
MAX_PER_INDUSTRY = 4
COMPOUND_W = (0.5, 0.35, 0.15)  # 1d/3d/5d
HOLDING_BONUS = 0.2
# B3: Holding Bonus 按持仓天数衰减 day1=1.0/day2=0.5/day3=0.0
HOLDING_DAY_WEIGHTS = {0: 0.0, 1: 1.0, 2: 0.5, 3: 0.0}
# B4: base_rate 20 日滚动窗口
BASE_RATE_WINDOW = 20
# 动量阈值
FW_HARD = -0.03  # 预测跌幅 > 3% → 强制 low
FW_EPS = 0.001  # 预测值太小无法算比率
RATIO_UP = 1.0
RATIO_DOWN = 0.8
# D18 空仓触发
HS300_DROP_EMPTY = 0.03
MARKET_LIMIT_DOWN_EMPTY = 50
HS300_CONSEC_DOWN_CAP = 3
CAP_POSITION_REDUCED = 0.3
# V3.8 成本与准入
COST = 0.0013  # round-trip 费用 (E5 口径)
# E2 痛苦惩罚
PAIN_PENALTY = 0.5  # score × (1 - 0.5×pain_prob)
# 公告情感调整
ANNOUNCE_ADJ = 0.3  # score × (1 + 0.3×announce_score)
# 分布版仓位权重
SINGLE_CAP = 0.10  # 单票上限 (bear 7%)
SINGLE_CAP_BEAR = 0.07

SCHEMA_FIELDS = [
    "symbol",
    "board",
    "day_change",
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
    """大盘环境 (D18 空仓触发输入)."""

    hs300_drop_today: float = 0.0  # 沪深300 当日跌幅 (正数=跌)
    count_limit_down_market: int = 0  # 全市场跌停家数
    hs300_consecutive_down: int = 0  # 沪深300 连跌天数


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
            else:
                logger.warning(
                    "LambdaRank rank_score 退化 (std=%.6f), 回退 pred_ret_1d 横截面排名",
                    rank_std,
                )
        if not use_rank:
            # 回退: pred_ret_1d 组内百分位排名 (按 board 独立排名, 尺度对齐)
            df["_fallback_rank"] = df.groupby("board")["pred_ret_1d"].rank(pct=True)
            df["score"] = df["_fallback_rank"] * prob_adjust
            df = df.drop(columns=["_fallback_rank"])
        # [E2] 痛苦惩罚: pain_prob 高 → 排序分降权
        if "pain_prob" in df.columns:
            df["score"] = df["score"] * (1 - PAIN_PENALTY * df["pain_prob"].fillna(0.0))
        # [公告因子] 利好加分/利空减分 (安全网 #17)
        if "announce_score" in df.columns:
            df["score"] = df["score"] * (
                1 + ANNOUNCE_ADJ * df["announce_score"].fillna(0.0)
            )
        # B3: Holding Bonus 按持仓天数衰减 (day1=1.0/day2=0.5/day3=0.0)
        if "holding_day" in df.columns:
            df["_hd_weight"] = df["holding_day"].map(HOLDING_DAY_WEIGHTS).fillna(0.0)
        else:
            # 向后兼容: 无 holding_day 列时, is_in_yesterday_list=1 视为 day1 (weight=1.0)
            df["_hd_weight"] = df.get("is_in_yesterday_list", 0)
        df["score"] = df["score"] + HOLDING_BONUS * df["_hd_weight"] * df.get(
            "is_in_yesterday_list", 0
        )
        df = df.drop(columns=["_hd_weight"])
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
            | ((df["pred_ret_1d"] < 0) & (df["prob_up"] > df["base_rate"]))
        ).astype(int)
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
    def apply_industry_limit(
        ranked: pd.DataFrame, max_per_industry: int = MAX_PER_INDUSTRY
    ) -> pd.DataFrame:
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

    # ---------------- E7 动态准入 ----------------
    def dynamic_entry(self, df: pd.DataFrame, bear: bool = False) -> pd.DataFrame:
        """[E7] 质量阈值准入: prob_up > 门槛 且 pred_ret_1d > mult×COST → 0~15 只.

        [E11] bear 状态收紧: prob 0.60→0.65, ret 2×成本→3×成本.
        若绝对阈值不可达, 自动回退至 prob_up 分位数门槛 (取前 20%).
        符合票可能为 0 — 这是特性不是故障.
        """
        prob_th = self.entry_prob_bear if bear else self.entry_prob
        ret_th = (self.entry_ret_mult_bear if bear else self.entry_ret_mult) * COST
        ok = df[(df["prob_up"] > prob_th) & (df["pred_ret_1d"] > ret_th)]
        n = len(ok)
        # 绝对阈值未命中 → 分位数回退 (模型概率天然保守, 取相对排名)
        if n == 0 and len(df) > 0:
            pctile_th = float(df["prob_up"].quantile(self._prob_pctile))
            ret_pctile_th = float(df["pred_ret_1d"].quantile(self._prob_pctile))
            ok = df[(df["prob_up"] >= pctile_th) & (df["pred_ret_1d"] >= ret_pctile_th)]
            n_pct = len(ok)
            logger.warning(
                "E7 动态准入: 绝对阈值 0 只 (prob>%.2f, ret>%.2f%%), "
                "分位数回退取前 %.0f%% → %d 只 (prob≥%.4f, ret≥%.4f)",
                prob_th,
                ret_th * 100,
                (1 - self._prob_pctile) * 100,
                n_pct,
                pctile_th,
                ret_pctile_th,
            )
        elif n == 0:
            logger.warning(
                "E7 动态准入: 0 只过闸 (prob>%.2f, ret>%.2f%%), 今日空清单",
                prob_th,
                ret_th * 100,
            )
        return ok

    # ---------------- E1 分布版仓位权重 ----------------
    @staticmethod
    def distribution_weights(
        df: pd.DataFrame,
        cap: float,
        bear: bool = False,
        capital: float | None = None,
    ) -> pd.Series:
        """[E1] raw_w = pred_q50×prob_up / (ATR_pct × (1+uncertainty_width));
        w = min(raw/Σraw, 单票上限) × position_multiplier × liquidity_cap [E6].

        不确定性越大权重越低; ATR/分布列缺失 → 回落等权.
        liquidity_cap 需要 capital (推算 order_value) 与 adv20 列, 缺一跳过.
        """
        single_cap = SINGLE_CAP_BEAR if bear else SINGLE_CAP
        dist_cols = {"pred_q50", "uncertainty_width", "ATR_pct"}
        if dist_cols <= set(df.columns) and len(df):
            raw = (
                df["pred_q50"]
                * df["prob_up"]
                / (
                    df["ATR_pct"].clip(lower=0.005)
                    * (1 + df["uncertainty_width"].clip(lower=0.001))
                )
            ).clip(lower=0)
        else:
            raw = pd.Series(1.0, index=df.index)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=df.index)
        # weight_i = min(raw/Σraw, 单票上限) × multiplier × liquidity_cap (V3.8 §四)
        # 注意: clip 后不再归一 — 被削部分留现金 (归一会把单票上限顶破)
        w = (raw / raw.sum()).clip(upper=single_cap)
        w = w * cap  # position_multiplier (D18/D4 出口)
        # [E6] liquidity_cap: 单票买入 ≤ ADV20×1% (bear 0.5%)
        if capital and "adv20" in df.columns:
            caps = [
                liquidity_cap(w.loc[i] * capital, a, bear)
                for i, a in df["adv20"].items()
            ]
            w = w * pd.Series(caps, index=df.index)
        return w.round(4)

    # ---------------- D18 空仓触发 (安全网 #12) ----------------
    @staticmethod
    def check_empty_triggers(env: MarketEnv) -> tuple[bool, float]:
        """返回 (是否强制空清单, 仓位上限).

        沪深300 跌>3% 或 全市场跌停>50 → 空清单;  连跌3日 → 仓位上限 30% (仅 Top 5).
        """
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

    # ---------------- 总装 ----------------
    def emit(
        self,
        candidates: pd.DataFrame,
        env: MarketEnv | None = None,
        market_state: str = "range",
        capital: float | None = None,
        ret_window_20d: pd.DataFrame | None = None,
    ) -> dict:
        """生成清单 schema V1.2 (动态准入 0~15 只 [E7]).

        candidates: 需含 symbol/board/industry/pred_ret_1d/3d/5d/prob_up(校准后)
                    [/is_limit_up_close/is_one_word_limit/is_in_yesterday_list]
                    [E1: pred_q50/uncertainty_width/ATR_pct] [E2: pain_prob]
                    [公告: announce_score] [E6: adv20]
        capital: 总资金 (E6 liquidity_cap 推算 order_value; None 跳过)
        ret_window_20d: 20 日收益矩阵 (E8 簇阻断; None 跳过)
        Returns:
            {'list': DataFrame(0~15 行, SCHEMA_FIELDS), 'cap_position': float,
             'empty': bool, 'schema_version': '1.2'}
        """
        env = env or MarketEnv()
        bear = market_state == "bear"  # [E11] bear 收紧联动
        empty, cap = self.check_empty_triggers(env)
        if empty or len(candidates) == 0:
            return {
                "list": pd.DataFrame(columns=SCHEMA_FIELDS),
                "cap_position": 0.0,
                "empty": True,
                "schema_version": SCHEMA_VERSION,
            }

        df = self.compute_scores(candidates)
        df = self.consensus_and_conflict(df)
        # §14.2.3 跨组归一化: 主板/双创 score 尺度不同, 组内 rank_pct 后合并排序
        df["score_rank_pct"] = df.groupby("board")["score"].rank(pct=True)
        # [E7] 动态准入 (bear 收紧 [E11])
        df = self.dynamic_entry(df, bear=bear)
        if len(df) == 0:
            return {
                "list": pd.DataFrame(columns=SCHEMA_FIELDS),
                "cap_position": cap,
                "empty": True,
                "schema_version": SCHEMA_VERSION,
            }
        df["momentum"] = [
            self.compute_momentum(a, b, c)
            for a, b, c in zip(df["pred_ret_1d"], df["pred_ret_3d"], df["pred_ret_5d"])
        ]
        df["market_state"] = market_state
        df["prob_up"] = df["prob_up"].round(3)
        df = df.sort_values("score_rank_pct", ascending=False)
        df = self.apply_industry_limit(df)
        top = TOP_N if cap >= 1.0 else 5  # D18 降仓 → 仅 Top 5
        df = df.head(top)
        # [E1] 分布版仓位权重 (含 E6 liquidity_cap)
        df["weight"] = self.distribution_weights(df, cap, bear=bear, capital=capital)
        # [E8] 相关性簇阻断 (簇总权重 ≤ 15%, bear 12%)
        if ret_window_20d is not None and len(df) > 1:
            clusters = cluster_block(list(df["symbol"]), ret_window_20d)
            df["weight"] = apply_cluster_caps(
                df.set_index("symbol")["weight"],
                clusters,
                cap=CLUSTER_CAP_BEAR if bear else CLUSTER_CAP,
            ).values
        df["schema_version"] = SCHEMA_VERSION
        for col in ("is_limit_up_close", "is_one_word_limit"):
            if col not in df.columns:
                df[col] = 0
        # day_change 由 predictor 透传 (close/pre_close-1); 候选缺失时回填 NaN
        if "day_change" not in df.columns:
            df["day_change"] = np.nan
        for col in (
            "pred_q10",
            "pred_q50",
            "pred_q90",
            "uncertainty_width",
            "pain_prob",
            "announce_score",
        ):
            if col not in df.columns:
                df[col] = np.nan
        return {
            "list": df[SCHEMA_FIELDS].reset_index(drop=True),
            "cap_position": cap,
            "empty": False,
            "schema_version": SCHEMA_VERSION,
        }


# ============================================================
# 清单溯源追踪 (源自 a-share-selection-strategy, provenance)
# ============================================================
@dataclass
class ProvenanceTracker:
    """记录每只候选股的数据来源/模型版本/计算时间戳.

    用法:
        tracker = ProvenanceTracker()
        tracker.record(symbol, data_source="baostock", model_tag="main_20260724")
        meta = tracker.get(symbol)
    """

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
    return None
