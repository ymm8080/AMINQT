"""滞涨标记共享逻辑 (2026-08-19 用户定案, legacy+parallel 双交付).

入选 = 今日交付清单股 (legacy list_*.parquet / parallel shortlist).
滞涨 = 近 10 日涨幅 < STALL_MARKER.ret_10d (面板 close_hfq shift(10), T 日收盘可得 PIT).
高频 = 近 STALL_MARKER.window_days 个交付交易日入选 ≥ min_sel 次 (历史交付 CSV 统计).
市场 = 当日 base_rate < STALL_MARKER.base_rate_max (低基线日) — 决定性条件 (见下).
命中 → stall_flag = "洗盘待爆发". 不改选股不改排序, 纯运营辅助标记.

250d 检验 (_diag_stall_regime, 2026-08-19): 入选+滞涨+近20日入选≥3 全窗命中 63.2%/
实得 +5.88%, 但决定性变量是市场状态 — 强市日 80.5%/+12.35% vs 弱市日 23.5%/-8.94%,
低基线日 82.9%/+13.28% vs 高基线日 -4.40%; 2025 vs 2026 差异 = 市场状态分布差异
(2025 弱市日 64% vs 2026 强市日 64%), 非组合本身. → 交付层打标仅限低基线日,
勿做进模型.
"""

import os
import re

import numpy as np
import pandas as pd

from app.pipeline1.prob_head import _add_mfe_3d
from config.settings import (
    LEGACY_PROB_GATE,
    PANEL_V3_PATH,
    STALL_MARKER,
    STOCK_LIST_DIR,
)


def _history_counts(hist_dir: str, trade_date: str, prefix: str, window: int) -> dict:
    """近 window 个交付交易日的历史清单 symbol → 入选次数.

    trade_date = YYYYMMDD (8 位); 只统计早于当日的历史文件, 按日期取最近 window 个.
    文件缺失 (某日未跑) 不补偿 — 统计的是"实际交付的最近 N 个交易日".
    """
    files = []
    for name in os.listdir(hist_dir):
        m = re.match(rf"{re.escape(prefix)}(\d{{8}})(?:__.*)?\.csv$", name)
        if m and m.group(1) < trade_date:
            files.append((m.group(1), name))
    counts: dict[str, int] = {}
    for _d, name in sorted(files)[-window:]:
        try:
            d = pd.read_csv(
                os.path.join(hist_dir, name), usecols=["symbol"], dtype={"symbol": str}
            )
        except Exception:
            continue
        for sym in d["symbol"].dropna().unique():
            counts[sym] = counts.get(sym, 0) + 1
    return counts


def _day_base_rate(panel: pd.DataFrame) -> float | None:
    """当日 dual 池 base_rate (prob_head._base_rate 同口径, 纯面板无 bundle 依赖).

    最近 base_rate_days 个可观测日 (T+2..T+4 未来价可得) 逐日 mfe_3d≥abs_target
    达标率均值; 不足 → None. dual = 面板 board GEM/STAR (与 replay 诊断 base_prod 对齐).
    """
    n = LEGACY_PROB_GATE["base_rate_days"] + 14
    dual = panel[panel["board"].isin(("GEM", "STAR"))].copy()
    dates = np.unique(pd.to_datetime(dual["date"]).to_numpy())
    if len(dates) < n:
        return None
    dual = dual[pd.to_datetime(dual["date"]) >= dates[-n]].copy()
    dual = dual.sort_values(["symbol", "date"])
    dual["adv20"] = (
        dual.groupby("symbol")["amount"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dual = _add_mfe_3d(dual)
    dual = dual[dual["mfe_3d"].notna()]
    hit = (
        (dual["mfe_3d"] >= LEGACY_PROB_GATE["abs_target"])
        .groupby(pd.to_datetime(dual["date"]))
        .mean()
    )
    if len(hit) < LEGACY_PROB_GATE["base_rate_days"]:
        return None
    return float(hit.tail(LEGACY_PROB_GATE["base_rate_days"]).mean())


def stall_marker(
    df: pd.DataFrame,
    trade_date: str,
    hist_prefix: str,
    hist_dir: str | None = None,
    panel_path=None,
) -> pd.DataFrame:
    """返回加 stall_flag/limit_flag/ret_10d/sel_20d/market_base_rate/advice 列的副本.

    滞涨标记 = 入选清单股 & 近10日滞涨<ret_10d & 近 window_days 入选≥min_sel &
    当日低基线日 (base_rate < base_rate_max). 任一条件不满足/数据缺失 → stall_flag 空串.
    涨停标记 = 昨日 (T-1) 涨幅 ≥ 板块涨停阈值 → limit_flag "涨停次日不追".
    advice = 当日参与度建议 (高基线日降参与). hist_prefix: 历史交付文件前缀
    ("legacy_stocklist_" / "parallel_shortlist_").
    """
    cfg = STALL_MARKER
    out = df.copy()
    out["stall_flag"] = ""
    panel_path = PANEL_V3_PATH if panel_path is None else panel_path
    if panel_path is not None and os.path.exists(str(panel_path)):
        p = pd.read_parquet(
            str(panel_path),
            columns=["symbol", "date", "close_hfq", "high_hfq", "amount", "board"],
        )
        base_rate = _day_base_rate(p)  # 需完整面板窗口, 过滤前算
        p["symbol"] = p["symbol"].astype(str)
        p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
        g = p.groupby("symbol")
        p["ret_10d"] = p["close_hfq"] / g["close_hfq"].shift(10) - 1.0
        p["ret_1d"] = (
            p["close_hfq"] / g["close_hfq"].shift(1) - 1.0
        )  # 昨日涨幅 (T-1, PIT)
        p = p[p["date"] == pd.Timestamp(trade_date).strftime("%Y-%m-%d")]
        out = out.merge(
            p[["symbol", "ret_10d", "ret_1d", "board"]],
            on="symbol",
            how="left",
            suffixes=("", "_panel"),  # 清单无 board 列时用面板板块判涨停阈值
        )
    else:
        out["ret_10d"] = float("nan")
        out["ret_1d"] = float("nan")
        base_rate = None
    out["market_base_rate"] = base_rate
    # 参与度提示 (2026-08-19 第五轮定案): 高基线日 (base_rate≥base_rate_max) 模型
    # 整体负期望 (全窗 -4.40%) → 建议降参与; 低基线日正常参与. 不改选股不改模型.
    if base_rate is None:
        out["advice"] = ""
    elif base_rate < cfg["base_rate_max"]:
        out["advice"] = (
            f"市场条件偏强 (base_rate={base_rate:.3f}): 模型近期胜率高, 正常参与"
        )
    else:
        out["advice"] = (
            f"市场条件偏弱 (base_rate={base_rate:.3f}): 模型近期整体负期望, 建议降低参与度/轻仓"
        )
    hist_dir = str(STOCK_LIST_DIR) if hist_dir is None else str(hist_dir)
    counts = _history_counts(hist_dir, trade_date, hist_prefix, cfg["window_days"])
    out["sel_20d"] = out["symbol"].astype(str).map(counts).fillna(0)
    cold = base_rate is not None and base_rate < cfg["base_rate_max"]
    sig = (out["ret_10d"] < cfg["ret_10d"]) & (out["sel_20d"] >= cfg["min_sel"]) & cold
    out.loc[sig, "stall_flag"] = "洗盘待爆发"
    # 涨停次日不追纪律 (2026-08-19 第六轮): T 日涨停 T+1 买 T+11 卖 890d 全池
    # 均值 -0.82% (中位 -4.82%, 命中 37%) → 清单中昨日涨停股打标. 不改选股.
    out["limit_flag"] = ""
    if "board" in out.columns and "ret_1d" in out.columns:
        # 清单 board 值小写 (main/gem/star); 阈值表键大写 → 统一转大写再 map,
        # 否则全 miss 落入 fillna 9.5% (dual 涨停 19.5% 被误当主板阈值)
        lim = (
            out["board"]
            .astype(str)
            .str.upper()
            .map(cfg["limit_ret_by_board"])
            .fillna(0.095)
        )
        out.loc[out["ret_1d"] >= lim, "limit_flag"] = "涨停次日不追"
    return out
