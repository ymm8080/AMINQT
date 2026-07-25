"""
E10 模拟盘验收 + D.8 影子清单 (PIPELINE1_V3.8 §四 bis/附录D.8, 检查清单 #79-#80)
================================================================================
[E10] 模拟盘 3 个月验收门禁: 成交率 ≥ 80% (回测拟买入票的实际成交比例;
      <80% 判定冲击成本模型错误, 一票否决上线) + 回测-模拟盘收益偏差 < 30%.
[E10] 坏单复盘周机制: 预测大涨实际大跌 → 人工归因 → 固化清洗特征 (#80).
[D.8] 影子清单: 与执行清单同一份模型输出、另一档参数, 不投入资金,
      每日记录"假如执行"的完整损益 (含全部失效条件/分层滑点/公告惩罚/换仓阈值);
      两档净值同图对比, 月度 GT-Score 双档裁决 (D.6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

FILL_RATE_GATE = 0.80  # E10: 成交率门禁 ≥80% (一票否决)
DEVIATION_GATE = 0.30  # E10: 回测-模拟盘收益偏差 < 30%
SHADOW_PROFILES = ("stable", "aggressive")  # D.8 双清单档位


# ============================================================
# E10 成交率统计 (门禁 ≥80%)
# ============================================================
@dataclass
class FillRateTracker:
    """每日成交率: 回测拟买入票中实际成交的比例.

    用法:
        fr = FillRateTracker()
        fr.record("2026-07-25", planned=["600519", "600000"], filled=["600519"])
        fr.rolling_rate(10)  # 连续10日 ≥80% 才过门禁 (P19.1 通过标准)
    """

    _records: list[dict] = field(default_factory=list)

    def record(self, date: str, planned: list[str], filled: list[str]) -> float:
        """记录一日. 返回当日成交率 (无拟买入 → 1.0, 不计失败)."""
        planned_set, filled_set = set(planned), set(filled)
        hit = len(planned_set & filled_set)
        rate = hit / len(planned_set) if planned_set else 1.0
        self._records.append(
            {"date": str(date), "planned": len(planned_set), "filled": hit,
             "rate": rate}
        )
        if planned_set and rate < FILL_RATE_GATE:
            logger.error("E10 成交率告警: %s 成交率 %.0f%% < 80%% (%d/%d)",
                         date, rate * 100, hit, len(planned_set))
        return rate

    def rolling_rate(self, days: int = 10) -> float:
        """近 N 日总成交率 (按票数加权, 非日率均值)."""
        recent = self._records[-days:]
        planned = sum(r["planned"] for r in recent)
        filled = sum(r["filled"] for r in recent)
        return filled / planned if planned else 1.0

    def gate_pass(self, days: int = 10) -> dict:
        """E10 门禁: 连续 N 日成交率 ≥80%; 不达标 → 冲击成本模型错误, 一票否决."""
        rate = self.rolling_rate(days)
        ok = rate >= FILL_RATE_GATE
        if not ok:
            logger.critical(
                "E10 门禁否决: 近%d日成交率 %.1f%% < 80%%, 冲击成本模型错误, 禁止上线",
                days, rate * 100)
        return {"pass": ok, "fill_rate": round(rate, 4), "days": days}


# ============================================================
# D.8 影子清单追踪 (假如执行的完整损益)
# ============================================================
@dataclass
class ShadowListTracker:
    """影子清单: 记录另一档参数的"假如执行"净值 (不投入资金).

    规则 (D.8): 影子清单必须走完与执行清单完全相同的失效条件与成本口径,
    否则对比失真; 两份清单均入 WORM 日志 (schema 含 profile 字段);
    任何时刻只允许一份清单进入真实执行, 严禁"两份都买".

    用法:
        st = ShadowListTracker(profile="stable")
        st.record_list("2026-07-24", list_df)          # T 日清单
        st.mark_to_market("2026-07-25", close_prices)  # T+1 收盘估值
    """

    profile: str = "stable"
    initial_capital: float = 1_000_000
    _lists: list[dict] = field(default_factory=list)  # WORM: 只追加
    _nav: list[dict] = field(default_factory=list)
    _cash: float = field(init=False)
    _positions: dict = field(init=False, default_factory=dict)
    _consumed_upto: str = field(init=False, default="")

    def __post_init__(self):
        assert self.profile in SHADOW_PROFILES, f"profile 须为 {SHADOW_PROFILES}"
        self._cash = self.initial_capital

    def record_list(self, date: str, list_df: pd.DataFrame) -> None:
        """记录 T 日影子清单 (WORM, 只追加不覆盖)."""
        self._lists.append(
            {"date": str(date), "profile": self.profile,
             "symbols": list(list_df["symbol"]),
             "weights": list(list_df.get("weight", [1 / max(len(list_df), 1)] * len(list_df)))}
        )

    def mark_to_market(self, date: str, close_prices: pd.Series) -> float:
        """T 日收盘估值: 按昨日清单权重建仓 (T+1), 持仓满 3 日卖出.

        简化假设 (与回测日K近似口径一致): T+1 收盘价买入, 持有 ≤3 日,
        成本口径由调用方保证与执行清单一致 (D.8).
        """
        date = str(date)
        prices = close_prices
        # 建仓: 最近一份未消费的昨日清单 (WORM 不删除, 游标推进)
        pending = [
            lst for lst in self._lists if self._consumed_upto < lst["date"] < date
        ]
        if pending:
            last = pending[-1]
            self._consumed_upto = last["date"]
            for sym, w in zip(last["symbols"], last["weights"]):
                if sym in prices.index and sym not in self._positions:
                    amount = self._cash * float(w)
                    if amount > 0:
                        self._positions[sym] = {
                            "cost": float(prices[sym]), "amount": amount,
                            "hold_days": 0}
                        self._cash -= amount
        # 估值 + 到期卖出 (卖出所得计入现金, 当日 NAV 仍含该价值)
        nav = self._cash
        for sym in list(self._positions):
            pos = self._positions[sym]
            pos["hold_days"] += 1
            px = float(prices.get(sym, pos["cost"]))
            value = pos["amount"] * px / pos["cost"]
            nav += value
            if pos["hold_days"] >= 3:  # 持仓上限 3 日 (V3.8 §四 bis)
                self._cash += value
                del self._positions[sym]
        self._nav.append({"date": date, "nav": nav, "profile": self.profile})
        return nav

    def nav_curve(self) -> pd.DataFrame:
        """影子净值曲线 (与执行清单净值同图输出, D.6 月度 GT-Score 双档对比)."""
        return pd.DataFrame(self._nav, columns=["date", "nav", "profile"])


# ============================================================
# E10 回测 vs 模拟盘偏差 (门禁 <30%)
# ============================================================
def backtest_vs_paper_deviation(
    backtest_nav: pd.Series, paper_nav: pd.Series
) -> dict:
    """回测-模拟盘收益偏差: |paper_ret - bt_ret| / max(|bt_ret|, 1e-9).

    偏差 ≥30% → 查标签口径 (B9) 与滑点分层 (E5), 一票否决上线.
    """
    bt_ret = float(backtest_nav.iloc[-1] / backtest_nav.iloc[0] - 1)
    pp_ret = float(paper_nav.iloc[-1] / paper_nav.iloc[0] - 1)
    dev = abs(pp_ret - bt_ret) / max(abs(bt_ret), 1e-9)
    ok = dev < DEVIATION_GATE
    if not ok:
        logger.critical(
            "E10 偏差否决: 回测 %.2f%% vs 模拟盘 %.2f%%, 偏差 %.1f%% ≥ 30%%",
            bt_ret * 100, pp_ret * 100, dev * 100)
    return {"pass": ok, "deviation": round(dev, 4),
            "backtest_return": round(bt_ret, 4), "paper_return": round(pp_ret, 4)}


# ============================================================
# E10 坏单复盘 (#80)
# ============================================================
def bad_trade_review(
    predictions: pd.DataFrame, actual_returns: pd.Series, threshold: float = -0.05
) -> pd.DataFrame:
    """坏单复盘: 预测大涨 (pred_ret_1d > 2×COST) 实际大跌 (< -5%) 的票.

    输出人工归因清单 → 归因结论固化为清洗规则/特征 (#80 周机制).
    """
    df = predictions.copy()
    df["actual_ret"] = df["symbol"].map(actual_returns)
    bad = df[(df["pred_ret_1d"] > 2 * 0.0013) & (df["actual_ret"] < threshold)]
    if len(bad):
        logger.error("E10 坏单复盘: %d 只预测大涨实际大跌, 待人工归因", len(bad))
    return bad[["symbol", "pred_ret_1d", "actual_ret"]].reset_index(drop=True)
