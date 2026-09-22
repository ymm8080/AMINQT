"""横盘提示共享逻辑 (legacy+parallel+genious 三交付).

2026-09-22 用户令: 列名与值全部中文化可读 + 砍掉频次条件。
入选 = 今日交付清单股 (legacy list_*.parquet / parallel shortlist / genious 冠军四段).
横盘 = 近 10 日涨幅 < STALL_MARKER.ret_10d (面板 close_hfq shift(10), T 日收盘可得 PIT).
市场 = 当日 base_rate < STALL_MARKER.base_rate_max (冷静市) — 决定性条件.
命中 → 横盘提示 = "近10日未涨·冷静市". 不改选股不改排序, 纯运营辅助标注.

0922 三档消融判词 (tmp_t/_0922_stall_ablation_replay.py, 250d replay):
- 日闸 (冷静市) = 唯一稳定真边际: 日均差 f10 +4.11pp, 2025/2026 段全正;
- 频次条件 (近20日入选≥3) = 死重: 配对差 f10 仅 +0.28%/rnet −0.01%, 2025 段负 → 已砍;
- 横盘条件 = 条件性红利: 配对差 +1.70%/日 (f20 +3.07%), 2026 段 +2.92% vs 2025 段
  −0.19% — 红利期属性, 只配当标注勿升格。
"近20日入选次数" 列保留为参考 (原频次条件), 不再参与判定。
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
    """返回加 横盘提示/涨停提示/近10日涨幅/昨日涨幅/市场温度/参与建议/近20日入选次数 列的副本.

    横盘提示 = 入选清单股 & 近10日涨幅<ret_10d & 当日冷静市 (base_rate < base_rate_max).
    任一条件不满足/数据缺失 → 空串. (频次条件已于 0922 消融判死砍掉, 列保留为参考.)
    涨停提示 = 昨日 (T-1) 涨幅 ≥ 板块涨停阈值 → "涨停次日不追".
    参与建议 = 当日参与度建议 (高基线日建议轻仓). hist_prefix: 历史交付文件前缀
    ("legacy_stocklist_" / "parallel_shortlist_" / "genious_stocklist_").
    """
    cfg = STALL_MARKER
    out = df.copy()
    out["横盘提示"] = ""
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
    out = out.rename(columns={"ret_10d": "近10日涨幅", "ret_1d": "昨日涨幅"})
    out["市场温度"] = base_rate
    # 参与度提示 (2026-08-19 第五轮定案): 高基线日 (base_rate≥base_rate_max) 模型
    # 整体负期望 (全窗 -4.40%) → 建议降参与; 低基线日正常参与. 不改选股不改模型.
    if base_rate is None:
        out["参与建议"] = ""
    elif base_rate < cfg["base_rate_max"]:
        out["参与建议"] = f"市场温度 {base_rate:.0%}（偏低·对模型有利）: 正常参与"
    else:
        out["参与建议"] = f"市场温度 {base_rate:.0%}（偏高·追高拥挤）: 建议轻仓/降参与"
    hist_dir = str(STOCK_LIST_DIR) if hist_dir is None else str(hist_dir)
    counts = _history_counts(hist_dir, trade_date, hist_prefix, cfg["window_days"])
    out["近20日入选次数"] = out["symbol"].astype(str).map(counts).fillna(0)
    cold = base_rate is not None and base_rate < cfg["base_rate_max"]
    sig = (out["近10日涨幅"] < cfg["ret_10d"]) & cold
    out.loc[sig, "横盘提示"] = "近10日未涨·冷静市"
    # 涨停次日不追纪律 (2026-08-19 第六轮): T 日涨停 T+1 买 T+11 卖 890d 全池
    # 均值 -0.82% (中位 -4.82%, 命中 37%) → 清单中昨日涨停股打标. 不改选股.
    out["涨停提示"] = ""
    if "board" in out.columns and "昨日涨幅" in out.columns:
        # 清单 board 值小写 (main/gem/star); 阈值表键大写 → 统一转大写再 map,
        # 否则全 miss 落入 fillna 9.5% (dual 涨停 19.5% 被误当主板阈值)
        lim = (
            out["board"]
            .astype(str)
            .str.upper()
            .map(cfg["limit_ret_by_board"])
            .fillna(0.095)
        )
        out.loc[out["昨日涨幅"] >= lim, "涨停提示"] = "涨停次日不追"
    return out
