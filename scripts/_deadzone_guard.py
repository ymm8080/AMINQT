"""死区停推闸 (2026-09-05 用户拍板 "那就一起停吧"): 已完结票滚动赢率过低 → 当夜
不往同花顺加新股; 清单 CSV 照出照存 (txt 不写 — _ths_flush_guard 按
ths_watchlist_*__*.txt glob, 不写即不会误动未入自选的票)。

三线独立探测器 (legacy TOP10 / parallel TOP10 / 密度影子单), 参数同 V4
(tmp_t/_deadzone_hyst_tune_0905.py 变体扫描定稿; 2026-09-05 晚用户拍板
legacy/parallel 各用各自纯样本赢率互不混合 — V4 参数原在混合池上标定,
拆分后纯样本沿用同参, 待积累后再校):
  出票日 t 夜可知的结局 = T+1 买 → T+4 卖 (net4 = C[t+4]/C[t+1] − 1 − 0.2%),
  赢 = net4 ≥ 5%; 信号 = 出票日在 [t−10, t−4] 的已完结票赢率
  < 25% → 报警; 连续 2 个采样日 ≥ max(25%, 1.2×市场同口径赢率) → 解除
  (滞回; 09-08 自适应化: 弱市固定 40% 够不着 → 误报警拖长; 强市 1.2×市场
  ≥ 40% 抗抖语义不变; 市场基线缺失/故障 → 回退固定 40%)。
  无样本日重置解除计数 (停夜不残留旧进度)。
实现 = 每夜从交付史整段重算 (无跨夜状态文件): 幂等、停夜自愈、可单测。
样本 < 5 / 今日不在格点 / 任何异常 → fail-open 照常推 (返回不报警)。
回放证据 (125d): 生产 TOP10 池 V4 停推票合计 −0.18pp/票 (7月 −1.57pp/大跌17%
全段躲掉, 6月/8月误伤 +0.92/+5.94pp 为用户接受); 密度池 7月 −5.78pp/大跌22%
躲掉, 8月 +1.77pp 误伤 = "一起停" 拍板接受的尾部保险代价。
冷启动: top10 史回溯 2026-08-05、parallel 史回溯 2026-08-04, 拆分当晚即武装;
密度单史自 09-03 起需 ~2-3 周攒样本, 期间 fail-open 照推。
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import STOCK_LIST_DIR

# 参数 (V4 定稿; 改动须同步 tests/test_deadzone_guard.py 锁值断言)
DZ_ENABLED = True
DZ_ENTER = 0.25  # 滚动赢率 < 25% → 报警
DZ_EXIT = 0.40  # 连续 DZ_EXIT_DAYS 个采样日 ≥ 40% → 解除 (强市锚/回退线)
DZ_EXIT_DAYS = 2
DZ_WINDOW = 10  # 出票日回看窗口 (交易日序)
DZ_MIN_SAMPLES = 5  # 窗口内完结票少于此 → 不报警 (fail-open)
DZ_SETTLE = 4  # T+4 结算: 出票 di+4 ≤ 今日才可知结局
DZ_COST = 0.0020  # T+1 买 → T+4 卖 往返成本
DZ_WIN = 0.05  # 赢 = net4 ≥ 5%

# [09-08] 解除线自适应 (硬指标自适应化): 弱市里市场同口径赢率本身 4~20%, 线赢率
# 够不着固定 0.40 → 报警拖长误伤 (125d 回放 dual 停掉票均值 +1.90pp, main +0.20pp;
# 混合臂 max(地板, 1.2×市场) 双板占优: main 停掉票 −0.09pp 更准, dual 少停 20 日)。
# 强市 mwr≥33% 时 1.2×mwr ≥ 0.40, 维持 V4 抗抖语义不变。enter 比例自适应判死
# (市场赢率全窗上限 ~30%, 0.5~0.6× 永够不到线赢率 = 等于无守卫)。
DZ_EXIT_FLOOR = 0.25  # 弱市解除地板 (= enter 线; 抗抖靠连续 2 采样日确认)
DZ_EXIT_MKT_K = 1.2  # exit_t = max(DZ_EXIT_FLOOR, DZ_EXIT_MKT_K × mwr_t)
DZ_MKT_MIN_SYMBOLS = 200  # 市场基线单日有效股票数下限 (不足 → 当日无基线)


def rolling_win_rates(di: np.ndarray, win: np.ndarray, n_days: int) -> dict[int, float]:
    """第 t 日可知的滚动赢率: 出票日序在 [t−W, t−settle] 的已完结票。"""
    out: dict[int, float] = {}
    for t in range(DZ_SETTLE, n_days):
        m = (di >= t - DZ_WINDOW) & (di <= t - DZ_SETTLE)
        n = int(m.sum())
        if n >= DZ_MIN_SAMPLES:
            out[t] = float(win[m].mean())
    return out


def alarm_indices(
    di: np.ndarray,
    win: np.ndarray,
    n_days: int,
    exit_thr: dict[int, float] | None = None,
) -> set[int]:
    """V4 状态机: <进阈值报警; 报警中连续 exit_days 个采样日 ≥ 出阈值才解除。

    纯函数 (di/win 为出票日序与赢标记)。无样本日不改报警态、重置解除计数。
    exit_thr: {t: 当日解除阈值} (自适应解除线); None/缺日 → DZ_EXIT 固定 0.40。
    """
    wr = rolling_win_rates(di, win, n_days)
    on, good, out = False, 0, set()
    for t in range(DZ_SETTLE, n_days):
        if t not in wr:
            good = 0
            continue
        v = wr[t]
        if not on:
            if v < DZ_ENTER:
                on, good = True, 0
                out.add(t)
        elif v >= (exit_thr.get(t, DZ_EXIT) if exit_thr else DZ_EXIT):
            good += 1
            if good >= DZ_EXIT_DAYS:
                on, good = False, 0
        else:
            good = 0
            out.add(t)
    return out


_MKT_WR_CACHE: dict = {}


def _mkt_wr_by_date(pstart: pd.Timestamp, pend: pd.Timestamp) -> dict:
    """{date: 市场同口径滚动赢率} — 解除线自适应的输入 (09-08 接线).

    口径与线赢率同构: 全市场票 net4 (T+1→T+4 −成本, 赢≥DZ_WIN) 的当日占比,
    t 日可知窗 [t−DZ_WINDOW, t−DZ_SETTLE] (同滞后, 无 look-ahead)。单条目缓存
    (同夜三线同窗只算一次)。读/算失败 → {} (调用方回退固定 DZ_EXIT)。
    """
    out: dict = {}
    try:
        from config.settings import PANEL_V3_PATH

        key = (
            str(PANEL_V3_PATH),
            str(pd.Timestamp(pstart).date()),
            str(pd.Timestamp(pend).date()),
        )
        if _MKT_WR_CACHE.get("key") == key:
            return _MKT_WR_CACHE["val"]
        px = pd.read_parquet(
            PANEL_V3_PATH,
            columns=["symbol", "date", "close_hfq"],
            filters=[
                ("date", ">=", pd.Timestamp(pstart) - pd.Timedelta(days=40)),
                ("date", "<=", pend),
            ],
        )
        px["symbol"] = (
            px["symbol"].astype(str).str.zfill(6).str.replace(r"\..*", "", regex=True)
        )
        px = px[~px["symbol"].str.startswith(("43", "83", "87", "92"))]
        if len(px):
            c = px.pivot(
                index="date", columns="symbol", values="close_hfq"
            ).sort_index()
            cv = c.values
            n = len(c)
            if n > DZ_SETTLE + 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    net = np.full_like(cv, np.nan)
                    net[:-DZ_SETTLE] = (
                        cv[DZ_SETTLE:] / cv[1 : 1 - DZ_SETTLE] - 1 - DZ_COST
                    )
                mkt_win_day = np.full(n, np.nan)
                for i in range(n - DZ_SETTLE - 1):
                    v = net[i]
                    v = v[np.isfinite(v)]
                    if len(v) >= DZ_MKT_MIN_SYMBOLS:
                        mkt_win_day[i] = float((v >= DZ_WIN).mean())
                for t in range(DZ_SETTLE, n):
                    seg = mkt_win_day[t - DZ_WINDOW : t - DZ_SETTLE + 1]
                    seg = seg[np.isfinite(seg)]
                    if len(seg) >= DZ_WINDOW - DZ_SETTLE - 1:
                        out[c.index[t]] = float(seg.mean())
    except Exception:  # noqa: BLE001 — 市场基线任何故障 → 回退固定解除线
        out = {}
    _MKT_WR_CACHE.clear()
    _MKT_WR_CACHE["key"] = key
    _MKT_WR_CACHE["val"] = out
    return out


def _settled_outcomes(
    picks: pd.DataFrame,
    today: pd.Timestamp,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """(date, symbol) 史 → (date, symbol, di, win) 已完结票 + 完整出票日格点。

    di = 出票日在全史格点 (含未完结的近 4 日) 中的序 — 与 125d 回放同构。
    结局口径 = T+1 收盘买 → T+4 收盘卖, 净扣 DZ_COST; win = net4 ≥ DZ_WIN。
    缺价 (停牌等) 的票不进结局表。
    """
    picks = picks.copy()
    picks["date"] = pd.to_datetime(picks["date"])
    picks["symbol"] = picks["symbol"].astype(str).str.zfill(6)
    picks = picks[~picks["symbol"].str.endswith(".BJ")]
    picks = picks.drop_duplicates(["date", "symbol"]).reset_index(drop=True)
    grid = sorted(picks["date"].unique())
    if picks.empty:
        return picks.assign(di=pd.Series(dtype=int), win=pd.Series(dtype=bool)), grid
    from config.settings import PANEL_V3_PATH

    px = pd.read_parquet(
        PANEL_V3_PATH,
        columns=["symbol", "date", "close_hfq"],
        filters=[("date", ">=", grid[0]), ("date", "<=", today)],
    )
    px["symbol"] = (
        px["symbol"].astype(str).str.zfill(6).str.replace(r"\..*", "", regex=True)
    )
    c = px.pivot(index="date", columns="symbol", values="close_hfq").sort_index()
    idx = {d: k for k, d in enumerate(c.index)}
    col = {s: j for j, s in enumerate(c.columns)}
    p = picks[picks["date"].isin(idx) & picks["symbol"].isin(col)].copy()
    if p.empty:
        return p.assign(di=pd.Series(dtype=int), win=pd.Series(dtype=bool)), grid
    p["i"] = p["date"].map(idx).astype(int)
    p["j"] = p["symbol"].map(col).astype(int)
    cv = c.values
    p = p[(p["i"] >= 1) & (p["i"] + DZ_SETTLE <= len(c) - 1)].copy()
    if p.empty:
        return p.assign(di=pd.Series(dtype=int), win=pd.Series(dtype=bool)), grid
    ii, jj = p["i"].values, p["j"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        p["net4"] = cv[ii + DZ_SETTLE, jj] / cv[ii + 1, jj] - 1 - DZ_COST
    p = p[p["net4"].notna()].copy()
    didx = {d: k for k, d in enumerate(grid)}
    p["di"] = p["date"].map(didx).astype(int)
    p["win"] = p["net4"] >= DZ_WIN
    return p[["date", "symbol", "di", "win"]], grid


def _collect_history(source: str, list_dir=STOCK_LIST_DIR) -> pd.DataFrame:
    """collect_lists 口径下指定源 ("legacy"/"parallel") 单独成史的出票样本。

    2026-09-05 晚用户拍板: legacy/parallel 各用各自纯样本赢率, 不再混合。
    函数内导入 collect_lists 防循环 (_ths_watchlist_push 依赖本模块)。
    """
    from scripts._ths_watchlist_push import collect_lists

    if source == "legacy":
        fps = glob.glob(str(list_dir / "legacy_stocklist_????????__*.csv"))
    else:
        fps = glob.glob(str(list_dir / "parallel_shortlist_????????__*.csv"))
    dates = sorted(
        {
            m.group(1)
            for f in fps
            if (
                m := re.search(
                    r"(?:legacy_stocklist|parallel_shortlist)_(\d{8})__",
                    os.path.basename(f),
                )
            )
        }
    )
    rows: list[tuple[str, str]] = []
    for d in dates:
        try:
            lists = collect_lists(d, list_dir=list_dir)
        except SystemExit:
            continue
        for tag, codes in lists:
            if tag.startswith(source):
                rows.extend((d, c) for c in codes)
    return pd.DataFrame(rows, columns=["date", "symbol"])


def load_legacy_history(list_dir=STOCK_LIST_DIR) -> pd.DataFrame:
    """legacy TOP10 出票史 = legacy 清单序前10 (collect_lists 口径纯 legacy 单),
    回溯 legacy_stocklist_{date}__*.csv 全集。"""
    return _collect_history("legacy", list_dir)


def load_parallel_history(list_dir=STOCK_LIST_DIR) -> pd.DataFrame:
    """parallel TOP10 出票史 = parallel_shortlist rank 前10 (collect_lists 口径
    纯 parallel 单), 回溯 parallel_shortlist_{date}__*.csv 全集。"""
    return _collect_history("parallel", list_dir)


def load_density_history(list_dir=STOCK_LIST_DIR) -> pd.DataFrame:
    """密度影子单出票史 = prob10dens_{date}__*.csv 交付全集 (09-03 首夜起)。"""
    rows: list[tuple[str, str]] = []
    for fp in glob.glob(str(list_dir / "prob10dens_????????__*.csv")):
        m = re.search(r"prob10dens_(\d{8})__", os.path.basename(fp))
        if not m:
            continue
        df = pd.read_csv(fp, dtype={"symbol": str})
        for s in df["symbol"].dropna().astype(str):
            s = s.strip()
            if re.fullmatch(r"\d{6}", s):
                rows.append((m.group(1), s))
    return pd.DataFrame(rows, columns=["date", "symbol"])


_LOADERS = {
    "top10": load_legacy_history,
    "parallel": load_parallel_history,
    "prob10dens": load_density_history,
}


def _line_state(line: str, date: str) -> tuple[dict[int, float], set[int], int]:
    """{line} 线在 {date} 夜的 (滚动赢率表, 报警日集合, 当日格点序)。

    完结样本不足 / 当日不在格点 → ValueError (调用方按 fail-open 处置);
    未知线名/其他异常 → 原样上抛 (is_alarm 捕获, win_rate 返 None)。
    """
    today = pd.Timestamp(date)
    picks = _LOADERS[line]()
    out, grid = _settled_outcomes(picks, today)
    if len(out) < DZ_MIN_SAMPLES:
        raise ValueError(f"完结样本不足 ({len(out)} < {DZ_MIN_SAMPLES})")
    if today not in grid:
        raise ValueError("今日不在出票格点")
    wr = rolling_win_rates(out["di"].to_numpy(), out["win"].to_numpy(), len(grid))
    # 解除线自适应 (09-08): 市场赢率可知 → max(地板, 1.2×市场); 否则固定 0.40
    mwr = _mkt_wr_by_date(grid[0], today)
    exit_thr = {
        k: max(DZ_EXIT_FLOOR, DZ_EXIT_MKT_K * float(v))
        for k, d in enumerate(grid)
        if np.isfinite(v := mwr.get(d, np.nan))
    }
    alarmed = alarm_indices(
        out["di"].to_numpy(), out["win"].to_numpy(), len(grid), exit_thr or None
    )
    return wr, alarmed, grid.index(today)


def win_rate(line: str, date: str) -> float | None:
    """当夜 {line} 线滚动赢率数值 (终版清单 win_rate 列单源)。

    样本不足 / 当日不在格点 / 任何异常 → None (fail-open 语义同 is_alarm)。
    """
    if not DZ_ENABLED:
        return None
    try:
        wr, _alarmed, t = _line_state(line, date)
    except Exception:  # noqa: BLE001 — 闸的任何故障都不拦展示
        return None
    return wr.get(t)


def is_alarm(line: str, date: str) -> tuple[bool, str]:
    """当夜 {line} 线是否死区报警 (真 → 停推)。fail-open 永远返回不报警。"""
    if not DZ_ENABLED:
        return False, "死区停推闸关闭 (DZ_ENABLED=False)"
    try:
        wr, alarmed, t = _line_state(line, date)
    except ValueError as exc:
        return False, f"{exc}, fail-open 照常推"
    except Exception as exc:  # noqa: BLE001 — 闸的任何故障都不拦推送
        return False, f"死区闸计算失败, fail-open 照常推: {exc}"
    cur = wr.get(t)
    cur_s = f"{cur:.1%}" if cur is not None else "样本不足"
    if t not in alarmed:
        return False, f"滚动赢率 {cur_s} (未达报警线)"
    return True, (
        f"滚动{DZ_WINDOW}日完结票赢率 {cur_s} < 报警线 "
        f"{DZ_ENTER:.0%} (解除: 连续{DZ_EXIT_DAYS}日 ≥ max("
        f"{DZ_EXIT_FLOOR:.0%}, {DZ_EXIT_MKT_K:g}×市场赢率), "
        f"市场基线缺失时回退 {DZ_EXIT:.0%})"
    )


def annotate_stop(line: str, date: str, why: str, list_dir=STOCK_LIST_DIR) -> Path:
    """停推夜标注 (2026-09-05 用户: 不标注分不清 "闸停推" 和 "没推成功")。

    ① STOPPED_DEADZONE_{date}__{line}.txt 醒目标记 (write-if-absent, 含原因);
    ② 同日 legacy 清单 *.md 尾部横幅 (top10 线; 幂等, 重跑不重复追加)。
    推送结果单 status="deadzone" 由调用方 write_push_result 落 (看板卡同通道)。
    """
    d = Path(list_dir)
    marker = d / f"STOPPED_DEADZONE_{date}__{line}.txt"
    if not marker.exists():
        marker.write_text(
            f"死区停推 {date} ({line}): {why}\n"
            "清单照出, 今晚未入同花顺自选 (只加不删不变; 非推送故障)\n",
            encoding="utf-8",
        )
    if line == "top10":
        banner = (
            f"\n---\n> ⛔ 死区停推 {date}: {why} — "
            "清单照出但今晚不入同花顺自选 (只加不删不变)\n"
        )
        for fp in glob.glob(str(d / f"legacy_stocklist_{date}__*.md")):
            if "死区停推" in Path(fp).read_text(encoding="utf-8"):
                continue
            with open(fp, "a", encoding="utf-8") as fh:
                fh.write(banner)
    return marker
