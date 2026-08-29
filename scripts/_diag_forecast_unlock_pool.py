"""_diag_forecast_unlock_pool.py — 业绩预告 + 解禁 事件池时间对齐研究 (2026-08-26).

方法论 (feature-evaluation-methodology, 用户 2026-08-04): 事件类特征不用个股 TS /
不用日截面 — 把同类事件聚合到相对日 [-20,+20], 池内找 feature 作用. 先例:
_diag_event_pool.py (BT/HOLDER). 本研究是 V3 特征入闸前的裁决步, 不改生产.

数据源 (均已拉齐, 见 memory forecast-unlock-evt-data-state):
  forecast    = data/supply_cache/alt_data/forecast/all_*.parquet ×4
                (ann_date 20220827..20260825, 21,576 行, 修正公告 update_flag 保留)
  share_float = data/supply_cache/alt_data/share_float/share_float_full.parquet
                (437 万行逐持有人粒度 → 先聚合 (symbol, float_date): ratio 求和,
                 holders 计数; float_ratio 单位=% 占总股本, risk_overlays 已核)

事件锚: 业绩预告 = ann_date 行 (公告日, 与 HOLDER 先例同口径, 非交易日公告无面板行
→ 跳过 = panel_builder 左 merge 历史一致行为); 解禁 = float_date 行 (解禁生效日,
解禁日历提前公告, 但轨迹研究锚在生效日看实际抛压).

定位边界 (勿重复): risk_overlays.share_float_upcoming_scan 是交付级二值剔除
(未来 30 天累计>5% 剔除) — 本研究是 alpha/特征潜力研究 (过去余震 + 压力分级),
两者互补不冲突.

每事件组输出:
1. 窗口平均相对收益轨迹 [-20,+20] 段
2. 分层 T+2/3/5/10/20 平均收益: 预告按类型 (预增/略增/扭亏/预盈 vs 预减/略减/
   首亏/续亏 vs 其他) + p_change_mid 规模分位; 解禁按 ratio 分位 + 板块
3. 池内 rank IC: 预告 p_change_mid / 解禁 float_ratio_sum 对事件后收益

口径注意: rel 为毛收益 (相对事件日收盘, 与先例一致无成本) — 分层结论看相对差,
绝对值入生产前须扣 COST+滑点重验.
"""

from __future__ import annotations

import gc
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, PANEL_V3_PATH

W = 20
HORIZONS = (2, 3, 5, 10, 20)

FORECAST_DIR = DATA_DIR / "supply_cache" / "alt_data" / "forecast"
FLOAT_PATH = DATA_DIR / "supply_cache" / "alt_data" / "share_float" / "share_float_full.parquet"

# 预告类型 → 方向组 (Tushare 11 类; 未列出的进 "其他")
FC_POSITIVE = {"预增", "略增", "扭亏", "续盈", "减亏"}
FC_NEGATIVE = {"预减", "略减", "首亏", "续亏", "预亏"}


def _f(v) -> str:
    return f"{v:+.4f}" if v == v else "   nan"


def _pct(v) -> str:
    return f"{v:+.2%}" if v == v else "    —"


def board_of(symbol: str) -> str:
    if symbol.startswith(("60", "00")):
        return "main"
    if symbol.startswith(("30", "68")):
        return "dual"
    return "other"


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(
        str(PANEL_V3_PATH), columns=["date", "symbol", "is_suspended", "close_hfq"]
    )
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    cutoff = df["date"].max() - pd.DateOffset(years=3)
    df = df[df["date"] >= cutoff]
    df = df[~df["is_suspended"].astype(bool)].reset_index(drop=True)
    df["ridx"] = df.groupby("symbol").cumcount()
    df["ann_key"] = df["date"].dt.strftime("%Y%m%d")  # 事件锚匹配键
    return df


def load_forecast() -> pd.DataFrame:
    fps = sorted(glob.glob(str(FORECAST_DIR / "all_*.parquet")))
    parts = [pd.read_parquet(fp) for fp in fps]
    fc = pd.concat(parts, ignore_index=True)
    # 同 (symbol, ann_date, end_date) 多行 = 修正公告, keep-last (文件名序=时间序)
    fc = fc.drop_duplicates(subset=["symbol", "ann_date", "end_date"], keep="last")
    fc["p_change_mid"] = (fc["p_change_min"] + fc["p_change_max"]) / 2
    fc["dir_group"] = np.where(
        fc["type"].isin(FC_POSITIVE), "positive",
        np.where(fc["type"].isin(FC_NEGATIVE), "negative", "other"),
    )
    return fc


def load_unlock() -> pd.DataFrame:
    sf = pd.read_parquet(
        str(FLOAT_PATH),
        columns=["symbol", "ann_date", "float_date", "float_share", "float_ratio"],
    )
    sf = sf.dropna(subset=["float_date"])
    agg = (
        sf.groupby(["symbol", "float_date"])
        .agg(
            ratio_sum=("float_ratio", "sum"),
            share_sum=("float_share", "sum"),
            n_rows=("float_ratio", "size"),
        )
        .reset_index()
    )
    return agg


def anchor_events(panel: pd.DataFrame, ev: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """事件表 → 面板行锚 (date==事件日; 非交易日事件无面板行 → 丢弃, 先例口径)."""
    ev = ev.rename(columns={key_col: "ann_key"})
    m = ev.merge(
        panel[["symbol", "ann_key", "ridx", "date"]],
        on=["symbol", "ann_key"],
        how="inner",
    )
    m["evt_id"] = np.arange(len(m))
    m["board"] = m["symbol"].astype(str).str.zfill(6).map(board_of)
    return m


def build_piv(panel: pd.DataFrame, events: pd.DataFrame, w: int = W) -> pd.DataFrame:
    """事件 × 偏移 → 相对事件日收盘收益宽表 (index=evt_id, columns=off)."""
    off = pd.DataFrame({"off": np.arange(-w, w + 1)})
    ev_long = events[["evt_id", "symbol", "ridx"]].merge(off, how="cross")
    ev_long["trg"] = ev_long["ridx"] + ev_long["off"]
    win = ev_long.merge(
        panel[["symbol", "ridx", "close_hfq"]],
        left_on=["symbol", "trg"],
        right_on=["symbol", "ridx"],
        how="left",
        suffixes=("", "_x"),
    )
    base = (
        win.loc[win["off"] == 0, ["evt_id", "close_hfq"]]
        .rename(columns={"close_hfq": "base"})
        .dropna(subset=["base"])
    )
    win = win.merge(base, on="evt_id", how="inner")  # 事件日无收盘 (异常) 整事件弃
    win["rel"] = win["close_hfq"] / win["base"] - 1.0
    piv = win.pivot_table(index="evt_id", columns="off", values="rel")
    del win, ev_long
    gc.collect()
    return piv


def print_trajectory(out: list, piv: pd.DataFrame) -> None:
    out.append("\n  1) 窗口平均相对收益 (相对事件日收盘):")
    out.append(f"  {'段':<12}{'平均rel':>10}{'胜率>0':>8}{'n':>9}")
    for a, b in [(-20, -16), (-15, -11), (-10, -6), (-5, -1), (0, 0),
                 (1, 5), (6, 10), (11, 15), (16, 20)]:
        s = piv.loc[:, a : b if a != b else a].mean(axis=1).dropna() if a != b else piv[a].dropna()
        if not len(s):
            continue
        wr = (s > 0).mean()
        out.append(f"  [{a:+d},{b:+d}]{'':<6}{_pct(s.mean()):>10}{_pct(wr):>8}{len(s):>9,}")


def print_strata(out: list, piv: pd.DataFrame, events: pd.DataFrame,
                 strata: list[tuple[str, pd.Series]]) -> None:
    """strata = [(组名, evt_id 布尔/索引 Series 对齐 events), ...]"""
    out.append("\n  2) 分层事件后平均收益 (T+2/3/5/10/20):")
    out.append(f"  {'分组':<24}{'n':>7}{'T+2':>9}{'T+3':>9}{'T+5':>9}{'T+10':>9}{'T+20':>9}")
    for tag, sel in strata:
        idx = events.index[sel] if isinstance(sel, pd.Series) else sel
        eids = events.loc[idx, "evt_id"]
        row = [f"  {tag:<22}{len(eids):>7,}"]
        for h in HORIZONS:
            s = piv.loc[piv.index.isin(eids), h].dropna() if h in piv else pd.Series(dtype=float)
            row.append(f"{_pct(s.mean()) if len(s) else '—':>9}")
        out.append("".join(row))


def print_ic(out: list, piv: pd.DataFrame, events: pd.DataFrame,
             feat_name: str, feat_vals: pd.Series) -> None:
    out.append(f"\n  3) 池内 rank IC: {feat_name} → 事件后收益")
    out.append(f"  {'':<24}{'T+2':>9}{'T+3':>9}{'T+5':>9}{'T+10':>9}{'T+20':>9}")
    x = feat_vals.astype(float).reset_index(drop=True)
    row = [f"  {feat_name:<22}"]
    for h in HORIZONS:
        if h not in piv:
            row.append(f"{'—':>9}")
            continue
        y = piv[h].reindex(events["evt_id"].to_numpy()).reset_index(drop=True)
        m = x.notna() & y.notna()
        if m.sum() < 30:
            row.append(f"{'—':>9}")
            continue
        r = x[m].rank().corr(y[m].rank())
        row.append(f"{_f(r):>9}")
    out.append("".join(row))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out: list[str] = []
    panel = load_panel()
    out.append(
        f"--- 面板 3y: rows={len(panel):,} stocks={panel['symbol'].nunique()} "
        f"({panel['date'].min().date()}..{panel['date'].max().date()}) ---"
    )

    # ══ 业绩预告 ══
    fc = load_forecast()
    out.append(f"\n{'=' * 78}\n  业绩预告 事件池 (raw {len(fc):,} 条)\n{'=' * 78}")
    out.append("类型分布: " + ", ".join(
        f"{t}={n:,}" for t, n in fc["type"].value_counts().items()
    ))
    ev_fc = anchor_events(panel, fc, "ann_date")
    out.append(
        f"锚到面板交易日行: {len(ev_fc):,} 事件 / {ev_fc['symbol'].nunique()} 股 "
        f"(非交易日公告丢弃 = 先例口径)"
    )
    if len(ev_fc) >= 200:
        piv = build_piv(panel, ev_fc)
        print_trajectory(out, piv)
        print_strata(
            out, piv, ev_fc,
            [
                ("全部", pd.Series(True, index=ev_fc.index)),
                ("正向(预增/略增/扭亏..)", ev_fc["dir_group"] == "positive"),
                ("负向(预减/略减/首亏..)", ev_fc["dir_group"] == "negative"),
                ("其他(不确定等)", ev_fc["dir_group"] == "other"),
                ("正向|main", (ev_fc["dir_group"] == "positive") & (ev_fc["board"] == "main")),
                ("正向|dual", (ev_fc["dir_group"] == "positive") & (ev_fc["board"] == "dual")),
                ("负向|main", (ev_fc["dir_group"] == "negative") & (ev_fc["board"] == "main")),
                ("负向|dual", (ev_fc["dir_group"] == "negative") & (ev_fc["board"] == "dual")),
            ],
        )
        # p_change_mid 规模分位 (仅预告了幅度的子集)
        pcm = ev_fc["p_change_mid"]
        nzc = pcm.dropna()
        if len(nzc) > 300:
            q1, q2, q3 = nzc.quantile([0.25, 0.5, 0.75])
            print_strata(
                out, piv, ev_fc,
                [
                    ("幅度Q1(最弱)", pcm <= q1),
                    ("幅度Q2", (pcm > q1) & (pcm <= q2)),
                    ("幅度Q3", (pcm > q2) & (pcm <= q3)),
                    ("幅度Q4(最强)", pcm > q3),
                ],
            )
        print_ic(out, piv, ev_fc, "p_change_mid", pcm)
        del piv
        gc.collect()
    else:
        out.append("样本 <200, 跳过")

    # ══ 解禁 ══
    ul = load_unlock()
    out.append(f"\n{'=' * 78}\n  解禁 事件池 (聚合后 {len(ul):,} 个 (symbol,float_date))\n{'=' * 78}")
    out.append(f"ratio_sum 分布: {ul['ratio_sum'].describe().to_dict()}")
    ev_ul = anchor_events(panel, ul, "float_date")
    out.append(
        f"锚到面板交易日行: {len(ev_ul):,} 事件 / {ev_ul['symbol'].nunique()} 股"
    )
    if len(ev_ul) >= 200:
        piv = build_piv(panel, ev_ul)
        print_trajectory(out, piv)
        rs = ev_ul["ratio_sum"]
        q1, q2, q3 = rs.quantile([0.25, 0.5, 0.75])
        print_strata(
            out, piv, ev_ul,
            [
                ("全部", pd.Series(True, index=ev_ul.index)),
                ("ratio Q1(最小)", rs <= q1),
                ("ratio Q2", (rs > q1) & (rs <= q2)),
                ("ratio Q3", (rs > q2) & (rs <= q3)),
                ("ratio Q4(最大)", rs > q3),
                ("≥5% (风控同阈值)", rs >= 5.0),
                ("≥5%|main", (rs >= 5.0) & (ev_ul["board"] == "main")),
                ("≥5%|dual", (rs >= 5.0) & (ev_ul["board"] == "dual")),
                ("<1% (微小解禁)", rs < 1.0),
            ],
        )
        print_ic(out, piv, ev_ul, "float_ratio_sum", rs)
        # 余震方向: 大解禁事件后买入窗口 (T+2..T+20) 是否有修复/继续杀
        del piv
        gc.collect()
    else:
        out.append("样本 <200, 跳过")

    text = "\n".join(out)
    print(text)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_diag_forecast_unlock_pool_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
