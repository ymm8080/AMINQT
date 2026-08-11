"""环境市场分类器 + 否决层 (PIPELINE DESIGN 用法1 否决层 + 用法3 环境分类器).

情绪三态 (ice 冰点 / range 平稳 / hot 高潮), 每日收盘后判定, 无训练、无前视.
输入 = 当日全市场 涨停家数/跌停家数/成交额 + 近 60 日历史基线 (数据驱动, 非硬编码常数).

融合:
  - legacy: 分类 → market_state ('ice'→'bear' 收紧准入+降仓 0.5, hot/range→'range') 喂给
    ListGenerator.emit / entry_filter (复用已有 bear 收紧机制, 不加新旋钮).
  - parallel: 冰点 → 狙击/融合短名单不开仓 (对齐 slow_bull down_mode=no_open 惯例).

否决层:
  - 极端冰点 (涨停<10 且 跌停≥涨停) → veto, 强制空清单 (不开仓).
  - 逐股硬规则否决 (连板≥4/当日炸板/封单<5%或>80%/近5日炸板率>40%) 见
    limit_board.py 的 veto_limit_rows, 面向涨停板接力模型, 不作用于 legacy 日频清单.

涨停近似口径: 主板 ±9.8%, 双创 ±19.8% (close vs pre_close). 与 data_supply.fetch_market_sentiment
全局 9.8% 口径不同, 本模块按 board 分档更精确; 无 board 列时回退全局 9.8%.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 涨跌停家数近似阈值 (相对 pre_close)
_UP_MAIN, _DN_MAIN = 0.098, 0.098
_UP_DUAL, _DN_DUAL = 0.198, 0.198

# 分类参数 (基线为近 60 日均值, 此处是相对倍数, 非绝对常数)
_ICE_UP_PCT = 0.5  # 涨停家数 < 常态 50% → 冰点
_ICE_RATIO = 0.5  # 涨跌停比 < 0.5 → 冰点
_HOT_UP_PCT = 1.5  # 涨停家数 > 常态 150%
_HOT_RATIO = 3.0  # 涨跌停比 > 3.0 → 高潮
_VETO_UP = 10  # 极端冰点: 涨停 < 10 且 跌停 ≥ 涨停
HIST_WINDOW = 60


def _limit_mask(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """按板块分档的涨停/跌停掩码 (close vs pre_close)."""
    if "board" in panel.columns:
        is_dual = panel["board"].astype(str).eq("dual")
    else:
        is_dual = pd.Series(False, index=panel.index)
    up_th = np.where(is_dual, _UP_DUAL, _UP_MAIN)
    dn_th = np.where(is_dual, _DN_DUAL, _DN_MAIN)
    pre = panel["pre_close"].replace(0, np.nan)
    up = panel["close"] >= pre * (1 + up_th) - 1e-9
    dn = panel["close"] <= pre * (1 - dn_th) + 1e-9
    return up.fillna(False), dn.fillna(False)


def sentiment_from_panel(panel: pd.DataFrame, trade_date: str | None = None) -> dict:
    """当日全市场情绪: 涨停/跌停家数 + 两市总成交额.

    trade_date 缺省 → 取面板最新交易日的全部行 (当日). 无当日行 → 空 dict.
    """
    if len(panel) == 0:
        return {}
    if trade_date is not None:
        d = pd.to_datetime(trade_date)
        today = panel[panel["date"] == d]
        if len(today) == 0:
            logger.warning("sentiment_from_panel: %s 无当日行, 用最新日", trade_date)
            today = panel[panel["date"] == panel["date"].max()]
    else:
        today = panel[panel["date"] == panel["date"].max()]
    if len(today) == 0:
        return {}
    up, dn = _limit_mask(today)
    return {
        "date": str(today["date"].iloc[0].date()),
        "count_limit_up": int(up.sum()),
        "count_limit_down": int(dn.sum()),
        "market_turnover": float(today["amount"].sum())
        if "amount" in today.columns
        else 0.0,
    }


def sentiment_history_from_panel(
    panel: pd.DataFrame, n: int = HIST_WINDOW
) -> pd.DataFrame:
    """近 n 个交易日逐日情绪序列 (涨停/跌停家数, 含当日)."""
    up, dn = _limit_mask(panel)
    if "amount" in panel.columns:
        turn = panel["amount"].fillna(0.0)
    else:
        turn = pd.Series(0.0, index=panel.index)
    hist = (
        panel[["date"]]
        .assign(
            count_limit_up=up.astype(int),
            count_limit_down=dn.astype(int),
            turnover=turn,
        )
        .groupby("date", as_index=False)
        .agg(
            count_limit_up=("count_limit_up", "sum"),
            count_limit_down=("count_limit_down", "sum"),
            market_turnover=("turnover", "sum"),
        )
        .sort_values("date")
        .tail(n)
    )
    return hist.reset_index(drop=True)


def classify_market_state(sentiment: dict, history: pd.DataFrame | None = None) -> str:
    """三态情绪分类. 历史基线缺失 → 纯涨跌停比回退 (永不抛异常)."""
    up = float(sentiment.get("count_limit_up", 0))
    dn = float(sentiment.get("count_limit_down", 0))
    ratio = (up + 1.0) / (dn + 1.0)  # 涨跌停比, +1 平滑防除零
    if history is not None and len(history) >= 20:
        up_base = float(history["count_limit_up"].mean())
        up_pct = up / up_base if up_base > 1e-9 else 1.0
        if up_pct < _ICE_UP_PCT or ratio < _ICE_RATIO:
            return "ice"
        if up_pct > _HOT_UP_PCT and ratio > _HOT_RATIO:
            return "hot"
        return "range"
    if ratio < _ICE_RATIO:
        return "ice"
    if ratio > _HOT_RATIO:
        return "hot"
    return "range"


def state_policy(state: str) -> dict:
    """state → 融合策略 (复用清单层已有旋钮)."""
    if state == "ice":
        # 冰点 → 走已有 bear 收紧: entry_prob_bear 门槛 + pred_ret_3d>0 + cap 0.5
        return {"market_state": "bear", "cap_position": 0.5, "veto": False}
    return {"market_state": "range", "cap_position": 1.0, "veto": False}


def is_veto(state: str, sentiment: dict) -> bool:
    """极端冰点否决: 涨停家数极少 且 跌停 ≥ 涨停 → 强制空清单."""
    if state != "ice":
        return False
    up = float(sentiment.get("count_limit_up", 0))
    dn = float(sentiment.get("count_limit_down", 0))
    return up < _VETO_UP and dn >= up


def build_env_and_state(
    panel: pd.DataFrame, trade_date: str | None = None
) -> tuple[dict, dict]:
    """一次性封装: 面板 → (sentiment, {market_state, cap_position, veto}).

    供 legacy/parallel 调用方复用. 数据缺失 → 安全回退 range (行为不变).
    """
    sent = sentiment_from_panel(panel, trade_date)
    if not sent:
        return sent, {"market_state": "range", "cap_position": 1.0, "veto": False}
    hist = sentiment_history_from_panel(panel, HIST_WINDOW)
    state = classify_market_state(sent, hist)
    policy = state_policy(state)
    policy["veto"] = is_veto(state, sent)
    policy["state"] = state
    logger.info(
        "环境分类: state=%s 涨停=%d 跌停=%d 涨跌停比=%.2f | market_state=%s cap=%.2f veto=%s",
        state,
        sent["count_limit_up"],
        sent["count_limit_down"],
        (sent["count_limit_up"] + 1) / (sent["count_limit_down"] + 1),
        policy["market_state"],
        policy["cap_position"],
        policy["veto"],
    )
    return sent, policy
