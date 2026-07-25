"""
附录E 动态参数引擎 — 参数即输出 (PIPELINE1_V3.8 附录E, 检查清单 E-1~E-5)
================================================================================
设计原则: 人工只设定结果层面的四个旋钮; 门槛、仓位、止损、盈亏比、空仓比例
全部由市场数据每日反推. 附录E 与附录D 冲突处以 E 为准
(D.3 纪律、D.5 样本纪律、D.10 裁决协议不变).

E.2 每日每票计算链 (全部为输出):
  p      = prob_up_calibrated          # 滚动 Isotonic 校准曲线
  target = max(pred_q50, 1.5×ATR_pct)  # 目标收益: 预测与波动下限取大
  stop   = max(1.2×ATR_pct, target/2)  # 止损: 噪音带与盈亏比取宽者
  RR     = target / stop               # 盈亏比: 每票每日不同
  kelly  = p - (1-p)/RR                # 凯利值: 仅作否决项 (不作仓位公式)
  position = min(max_loss_per_trade / stop, 1.00)  # 亏损预算驱动 — 满仓是自然解
E.3 排序分波动率阻尼: score_adj = score / (1 + uncertainty_width)
  (要的是确定的暴利, 不是波动大的暴利)
E.4 分波动桶 IC 诊断: 按 ATR 五桶独立 Rank IC; 高波动桶 IC<0.02 → 阻尼权重
  上调 + 高波动桶仓位上限 ×0.7, 并书面记录裁决.

阶段一至三纪律 (F.5): 动态引擎仅允许以"影子计算"方式运行
(每日输出但不驱动交易), 积累参数分布数据, 为阶段四启用做准备.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# E.4 诊断门槛
BUCKET_IC_MIN = 0.02  # 高波动桶 Rank IC ≥ 0.02 → 满仓高波动结构成立
HIGH_VOL_DAMP = 0.7  # 不达标 → 高波动桶仓位上限额外收紧 ×0.7


@dataclass(frozen=True)
class DynamicKnobs:
    """E.1 用户旋钮 (唯一人工输入, 季度审视 + 书面记录, 安全网 #15)."""

    min_win_prob: float = 0.65  # 胜率闸门
    min_rr: float = 1.8  # 盈亏比闸门 (算出来, 不是定死)
    min_kelly: float = 0.25  # 凯利否决: 期望值过低不玩
    max_loss_per_trade: float = 0.04  # 单笔亏损预算 (总资金 4%, 狙击档)


# 固定护栏 (E.5, 不可数据化)
PAIN_PROB_MAX = 0.15  # 痛苦概率上限
QUANTITY_QUOTA = 2  # 数量闸门: rank_score 前 2
ATR_NOISE_MULT = 1.2  # 止损噪音带: ≥1.2×ATR (防正常波动扫损)
ATR_TARGET_MULT = 1.5  # 目标收益下限: ≥1.5×ATR


class DynamicEngine:
    """附录E 动态参数引擎 (影子模式: 每日输出, 不驱动交易, F.5 纪律).

    用法:
        eng = DynamicEngine()
        out = eng.per_stock_calc(p=0.70, pred_q50=0.06, atr_pct=0.03, pain_prob=0.10)
        # → {'target', 'stop', 'rr', 'kelly', 'entry', 'position', 'daily_fuse'}
    """

    def __init__(self, knobs: DynamicKnobs | None = None):
        self.knobs = knobs or DynamicKnobs()

    # ---------------- E.2 每票计算链 ----------------
    def per_stock_calc(
        self,
        p: float,
        pred_q50: float,
        atr_pct: float,
        pain_prob: float = 0.0,
    ) -> dict:
        """每日每票反推: 目标/止损/盈亏比/凯利/入场/仓位 (全字段留痕, E-2).

        Args:
            p: prob_up_calibrated (滚动 Isotonic 校准)
            pred_q50: 分位数中位预测 (E1)
            atr_pct: ATR/收盘价 (日波动)
            pain_prob: 痛苦概率 (E2)
        Returns:
            entry=True 时 position>0 为建议仓位 (满仓是自然解而非设定).
        """
        k = self.knobs
        atr = max(float(atr_pct), 1e-4)
        target = max(float(pred_q50), ATR_TARGET_MULT * atr)
        stop = max(ATR_NOISE_MULT * atr, target / 2.0)
        rr = target / stop
        kelly = p - (1 - p) / rr if rr > 0 else -1.0
        entry = bool(
            p >= k.min_win_prob
            and rr >= k.min_rr
            and kelly >= k.min_kelly
            and pain_prob < PAIN_PROB_MAX
        )
        position = min(k.max_loss_per_trade / stop, 1.00) if entry else 0.0
        return {
            "p": round(float(p), 4),
            "target": round(target, 4),
            "stop": round(stop, 4),
            "rr": round(rr, 3),
            "kelly": round(kelly, 4),
            "entry": entry,
            "position": round(position, 4),
            # 派生输出 (不再人工设定): 日保险丝 = 一笔止损收工
            "daily_fuse": round(position * stop, 4),
        }

    # ---------------- E.3 排序分波动率阻尼 ----------------
    @staticmethod
    def damped_score(score: pd.Series, uncertainty_width: pd.Series) -> pd.Series:
        """score_adj = score / (1 + uncertainty_width).

        预测涨8%但区间±12%的票, 排在预测涨5%但区间±4%的票之后.
        A 级信号最终选取用阻尼版 (E-3 双输出).
        """
        return score / (1 + uncertainty_width.clip(lower=0.0))

    # ---------------- E.4 分波动桶 IC 诊断 ----------------
    @staticmethod
    def bucket_ic(
        df: pd.DataFrame,
        score_col: str,
        label_col: str,
        atr_col: str = "ATR_pct",
        n_buckets: int = 5,
    ) -> dict:
        """按 ATR 分桶独立 Rank IC (上线前必跑 + 季度复核).

        Returns:
            {'buckets': {Q1..Q5: ic}, 'high_vol_ic': float,
             'high_vol_ok': bool (≥0.02 → 满仓高波动结构成立),
             'action': 'ok' / 'dampen'}  不达标 → E.3 阻尼上调 + 高波动桶上限×0.7
        """
        from scipy.stats import spearmanr

        sub = df[[atr_col, score_col, label_col]].dropna()
        if len(sub) < n_buckets * 10:
            return {"buckets": {}, "high_vol_ic": 0.0,
                    "high_vol_ok": False, "action": "insufficient_data"}
        sub = sub.copy()
        sub["bucket"] = pd.qcut(
            sub[atr_col], n_buckets, labels=[f"Q{i + 1}" for i in range(n_buckets)]
        )
        ics = {}
        for b, g in sub.groupby("bucket", observed=True):
            if g[score_col].nunique() > 2 and g[label_col].nunique() > 1:
                ics[str(b)] = round(
                    float(spearmanr(g[score_col], g[label_col]).statistic), 4)
            else:
                ics[str(b)] = 0.0
        high_ic = ics.get(f"Q{n_buckets}", 0.0)
        ok = high_ic >= BUCKET_IC_MIN
        if not ok:
            logger.error(
                "E.4 高波动桶 IC=%.4f < %.2f: 模型在最敢下注的地方最不准 → "
                "阻尼排序权重上调 + 高波动桶仓位上限 ×%.1f, 书面记录裁决",
                high_ic, BUCKET_IC_MIN, HIGH_VOL_DAMP)
        return {
            "buckets": ics,
            "high_vol_ic": high_ic,
            "high_vol_ok": ok,
            "action": "ok" if ok else "dampen",
        }
