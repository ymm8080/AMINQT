# -*- coding: utf-8 -*-
"""W17m 首板分钟封板质量特征提取器 (smoke 验证版, 全量 baostock 拉完后同入口重跑).

用途: 对主板首板事件 (symbol, d0) 从 baostock 5min 不复权数据提取 D0 封板结构 + D-1..D-5 前史特征,
供 W4 生产模型 (scripts/_firstboard_pages.py, 勿改) A/B 增量评估。
增量规则: Δk2@3>=+0.5pp 且 k2@1 跌幅<1pp=候选; |Δk2@3|<=1pp=平。

无前视: 特征只用 D0 及更早分钟数据; D+1..D+5 窗数据不进特征 (仅供将来标签分析)。
纯确定性计算 (无随机成分), 幂等可重跑 (同输入 → 同输出, 直接覆盖)。

分钟 bar 约定 (已在 smoke 实测): 每日恰好 48 根, bar 按**终点** HHMM 标注,
09:35..11:30 (mpos 5..120) + 13:05..15:00 (mpos 125..240); 行只含真实成交会话。

涨停价口径 (与生产 _firstboard_pages.py 同式): 主板 limit = round(面板 pre_close(D0) × 1.10, 2)。
口径对拍 (打印, 不静默): 面板 pre_close vs baostock D-1 收盘 / 面板 D0 close vs baostock D0 收盘
偏差分布 + 「面板收盘≈涨停」事件的分钟触板命中率; 诊断列随输出落盘。

用法:
  python tmp_t/_0923_w17m_feat_build.py
  python tmp_t/_0923_w17m_feat_build.py --minute-parquet tmp_t/_0923_bs5min_events.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

# ---- 常量 (口径集中, 勿写死进逻辑) ----
TOL = 0.005            # 触板/封板判定容差 (元), 与生产 zha 检测一致: >= limit - 0.005
GAP_PCT_WARN = 0.005   # 面板 vs baostock 收盘偏差超 0.5% 视为口径不齐 (多为除权除息)
PRE5_N = 5             # 前史窗口 D-1..D-5
PANEL_COLS = ["symbol", "date", "close", "pre_close", "pctChg", "amount"]

# 连续交易分钟轴: 早盘 bar 终点 09:35..11:30 → 5..120; 午盘 13:05..15:00 → 125..240
_M_OPEN_AM, _M_PM_START = 570, 780  # 09:30 / 13:00 钟面分钟

FEAT_COLS = [
    "fb_min", "fb_hhmm", "n_breaks", "seal_minutes", "late_seal_flag",
    "dd_preboard", "am_mom", "pm_thrust", "vol_pct_first30", "vol_pct_last30",
    "close_vwap_gap", "vol_slope_pre", "pre5_amp_med", "pre5_eodret_mean",
]
DIAG_COLS = ["pre5_days_avail", "n_bars", "touch_any", "close_at_limit",
             "preclose_gap_pct", "d0close_gap_pct", "limit_source_ok", "limit_touch_match"]


def _mpos(hhmm: pd.Series) -> np.ndarray:
    """bar 终点 HHMM → 连续交易分钟位置 (09:35→5, 11:30→120, 13:05→125, 15:00→240)。"""
    m = (hhmm // 100) * 60 + (hhmm % 100)
    return np.where(m >= _M_PM_START, 120 + (m - _M_PM_START), m - _M_OPEN_AM)


def _day_struct(g: pd.DataFrame, limit: float | None) -> dict:
    """单日 (symbol,date) 分钟结构标量。limit=None → 只算通用日级量 (供 D-1..D-5 前史/对拍)。

    每日 <=48 根 bar, 日内全 np 向量化。
    """
    g = g.sort_values("time", kind="mergesort")
    mpos = g["_mpos"].to_numpy()
    high, low, close = g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
    vol = g["volume"].to_numpy(dtype=float)
    amt = g["amount"].to_numpy(dtype=float)
    d_open = float(g["open"].iloc[0])
    d_close = float(close[-1])
    vol_sum = float(vol.sum())
    out = dict(
        d_open=d_open, d_close=d_close, d_high=float(high.max()), d_low=float(low.min()),
        vwap=(float(amt.sum() / vol_sum) if vol_sum > 0 else np.nan), vol_total=vol_sum,
        # 上午动量基准: 11:30 bar 收盘 (早盘最后一根)
        morn_close=(float(close[mpos <= 120][-1]) if (mpos <= 120).any() else np.nan),
        # 首/尾 30 分钟量占比 (bar 终点 <=10:00 即 09:30-10:00 窗; >=14:35 即 14:30-15:00 窗)
        vol_pct_first30=(float(vol[mpos <= 60].sum() / vol_sum) if vol_sum > 0 else np.nan),
        vol_pct_last30=(float(vol[mpos >= 235].sum() / vol_sum) if vol_sum > 0 else np.nan),
        # 尾盘 30 分钟收益 (15:00 收盘 / 14:30 bar 收盘 − 1); 源数据个别 bar close=0 ⇒ 分母护栏
        eod30_ret=((d_close / float(close[mpos <= 230][-1]) - 1.0)
                   if ((mpos <= 230).any() and float(close[mpos <= 230][-1]) > 0) else np.nan),
        n_bars=int(len(g)),
    )
    if limit is None or not np.isfinite(limit):
        return out

    # ---- 封板结构 (仅事件 D0, 传入 limit) ----
    touch = high >= limit - TOL   # 触板: bar 内最高价触及涨停
    seal = close >= limit - TOL   # 封住: bar 收盘停在涨停
    if not touch.any():
        out.update(fb_min=np.nan, fb_hhmm=np.nan, n_breaks=np.nan, seal_minutes=np.nan,
                   late_seal_flag=np.nan, dd_preboard=np.nan, pm_thrust=np.nan,
                   vol_slope_pre=np.nan, touch_any=0,
                   close_at_limit=int(d_close >= limit - TOL))
        return out

    i0 = int(np.argmax(touch))   # 首次触板 bar 下标
    out["fb_min"] = float(mpos[i0])
    out["fb_hhmm"] = float(g["hhmm"].iloc[i0])
    out["touch_any"] = 1
    out["close_at_limit"] = int(d_close >= limit - TOL)

    s_tail = seal[i0:]
    if s_tail[-1]:               # 收盘封死: 终封段 = 延续到收盘的最后一段连续 seal
        flip = s_tail[::-1]      # 从末尾数连续 True 段长 (argmin=首个 False; 全 True=整段)
        n_trail = int(np.argmin(flip)) if bool((~flip).any()) else len(flip)
        run_from = len(s_tail) - n_trail
        seg = s_tail[:run_from]
        out["seal_minutes"] = float(mpos[-1] - mpos[i0 + run_from] + 5)
        out["late_seal_flag"] = float(g["hhmm"].iloc[i0 + run_from] > 1430)  # 回封尾盘旗
    else:                        # 炸板日: 全部离板段计入, 封死时长=0, 尾旗=NaN
        seg = s_tail
        out["seal_minutes"] = 0.0
        out["late_seal_flag"] = np.nan
    edges = np.diff(np.concatenate([[False], seg, [False]]).astype(int))
    out["n_breaks"] = float((edges == 1).sum())   # 封死前开板次数

    out["dd_preboard"] = (float(low[: i0 + 1].min() / d_open - 1.0)
                          if d_open > 0 else np.nan)  # 板前最大回撤 (开盘→首触)
    # 午后推升 (13:00→首触): 午后首触 = 触板 bar high / 上午收盘 − 1; 早板 = NaN
    _mc = out["morn_close"]
    out["pm_thrust"] = (float(high[i0]) / _mc - 1.0) if (mpos[i0] > 120 and _mc > 0) else np.nan
    # 触板前量能斜率: 首触前 (不含触板 bar) 每根量对 bar 序的 OLS 斜率 / 窗口均量 (无量纲)
    if i0 >= 2:
        v_pre = vol[:i0]
        slope = np.polyfit(np.arange(i0, dtype=float), v_pre, 1)[0]
        out["vol_slope_pre"] = float(slope / v_pre.mean()) if v_pre.mean() > 0 else np.nan
    else:
        out["vol_slope_pre"] = np.nan
    return out


def build_minute_daily(minute: pd.DataFrame, d0_limit: pd.DataFrame) -> pd.DataFrame:
    """分钟表 → 逐日结构表 (每 symbol,date 一行; 事件 D0 行额外带封板结构)。

    d0_limit: DataFrame[symbol, date(=D0), _limit] — 命中的日子才算封板结构。
    """
    # 原地改造 (不 copy, 全量 ~11M 行省一次整帧复制); 调用方此后不再使用原 minute
    minute["date"] = pd.to_datetime(minute["date"])
    minute["hhmm"] = (minute["time"] // 100000) % 10000
    df = minute.drop_duplicates(["symbol", "date", "time"]).sort_values(
        ["symbol", "date", "time"], kind="mergesort")
    df["_mpos"] = _mpos(df["hhmm"])
    lim = d0_limit.set_index(["symbol", "date"])["_limit"]
    rows, keys = [], []
    for (sym, dt), g in df.groupby(["symbol", "date"], sort=False):   # 分钟表逐日 apply (允许)
        rows.append(_day_struct(g, lim.get((sym, dt))))
        keys.append((sym, dt))
    daily = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=["symbol", "date"]))
    return daily.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description="W17m 首板分钟封板质量特征 (smoke/全量同入口)")
    ap.add_argument("--minute-parquet", default=str(ROOT / "tmp_t/_0923_bs5min_events.parquet"))
    ap.add_argument("--events", default=str(ROOT / "tmp_t/_0923_w14_seat_feats.parquet"))
    ap.add_argument("--panel", default="D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet")
    ap.add_argument("--out", default=str(ROOT / "tmp_t/_0923_w17m_feats.parquet"))
    a = ap.parse_args()

    minute = pd.read_parquet(a.minute_parquet)
    minute["symbol"] = minute["symbol"].astype(str)
    events = pd.read_parquet(a.events, columns=["symbol", "d0"])
    events["symbol"] = events["symbol"].astype(str)
    events = events.drop_duplicates(["symbol", "d0"])
    print(f"[load] 分钟 {minute.shape} ({minute['symbol'].nunique()} 股, "
          f"{minute['date'].nunique()} 日), 事件 {len(events)}")

    # 面板: 事件 D0 行 (主板 00/60), 取 pre_close 算涨停价 + close 供口径对拍
    pan = pq.read_table(a.panel, columns=PANEL_COLS).to_pandas()
    pan["symbol"] = pan["symbol"].astype(str)
    pan = pan[pan["symbol"].str[:2].isin(["00", "60"])]
    pan["date"] = pd.to_datetime(pan["date"])
    ev = events.merge(pan.rename(columns={"date": "d0"})[
        ["symbol", "d0", "close", "pre_close", "pctChg", "amount"]],
        on=["symbol", "d0"], how="left")
    n_nopanel = int(ev["pre_close"].isna().sum())
    print(f"[panel] 事件缺面板行 (将被丢弃): {n_nopanel}/{len(ev)}")
    ev = ev.dropna(subset=["pre_close"]).copy()
    ev["_limit"] = (ev["pre_close"] * 1.10).round(2)   # 主板涨停价, 与生产同式

    # 分钟逐日结构 (一次扫描: 事件 D0 带封板结构, 其余日通用量)
    d0_limit = ev[["symbol", "d0", "_limit"]].rename(columns={"d0": "date"})
    daily = build_minute_daily(minute, d0_limit)

    feat = ev.merge(daily.rename(columns={"date": "d0"}), on=["symbol", "d0"], how="left")
    n_no_d0 = int(feat["n_bars"].isna().sum())
    print(f"[d0] 事件缺 D0 分钟行 (将被丢弃, 旧口径窗口 bug / 拉数未覆盖): {n_no_d0}/{len(feat)}")

    # ---- D-1..D-5 前史 (纯分钟侧日历) + 面板/baostock 收盘对拍基准 ----
    daily_all = daily.sort_values(["symbol", "date"]).reset_index(drop=True)
    daily_all["amp"] = (daily_all["d_high"] - daily_all["d_low"]) / daily_all["d_open"]
    dc = daily_all.set_index(["symbol", "date"])["d_close"]
    days_by_sym = daily_all.groupby("symbol")["date"].apply(list).to_dict()
    # 位置查表 (避免事件级全帧布尔扫描; 全量 ~2 万事件 × ~20 万日会平方级)
    dmap = {(s, d): i for i, (s, d) in enumerate(
        zip(daily_all["symbol"], daily_all["date"]))}
    amp_arr = daily_all["amp"].to_numpy()
    eod_arr = daily_all["eod30_ret"].to_numpy()

    feat = feat.sort_values(["symbol", "d0"]).reset_index(drop=True)
    pre5_amp, pre5_eod, pre5_n, prev_c, d0_c = [], [], [], [], []
    for sym, d0 in zip(feat["symbol"], feat["d0"]):
        prior = [d for d in days_by_sym.get(sym, []) if d < d0][-PRE5_N:]  # 最近 ≤5 个分钟交易日
        idxs = [dmap[(sym, d)] for d in prior]
        pre5_amp.append(float(np.nanmedian(amp_arr[idxs])) if idxs else np.nan)
        pre5_eod.append(float(np.nanmean(eod_arr[idxs])) if idxs else np.nan)
        pre5_n.append(int(len(idxs)))
        prev_c.append(dc.get((sym, prior[-1]), np.nan) if prior else np.nan)  # baostock D-1 收盘
        d0_c.append(dc.get((sym, d0), np.nan))                                # baostock D0 收盘
    feat["pre5_amp_med"] = pre5_amp        # 前5日分钟日内波幅中位数
    feat["pre5_eodret_mean"] = pre5_eod    # 前5日尾盘30分钟收益均值 (预洗盘痕迹)
    feat["pre5_days_avail"] = pre5_n

    # ---- 口径对拍: 面板 vs baostock (打印, 不静默) ----
    feat["preclose_gap_pct"] = feat["pre_close"].to_numpy() / np.asarray(prev_c, dtype=float) - 1.0
    feat["d0close_gap_pct"] = feat["close"].to_numpy() / np.asarray(d0_c, dtype=float) - 1.0
    for c in ("preclose_gap_pct", "d0close_gap_pct"):
        s = feat[c].dropna()
        if len(s):
            print(f"[口径] {c}: n={len(s)} 中位={s.median():+.6f} "
                  f"P95(|·|)={s.abs().quantile(0.95):.6f} "
                  f"超{GAP_PCT_WARN:.1%}不齐={(s.abs() > GAP_PCT_WARN).sum()}")
        else:
            print(f"[口径] {c}: 无可对拍样本")
    s = feat["preclose_gap_pct"].abs()
    feat["limit_source_ok"] = (s <= GAP_PCT_WARN).astype(float).where(s.notna())

    # 涨停价对拍: 面板口径「D0 收盘 ≈ 涨停」的事件, 分钟侧最高价也应触及
    closed_at_limit = feat["close"] >= feat["_limit"] - TOL
    sub = feat[closed_at_limit & feat["touch_any"].notna()]
    if len(sub):
        print(f"[涨停对拍] 面板收盘≈涨停事件 n={len(sub)}: 分钟 high 触板率="
              f"{(sub['touch_any'] == 1).mean():.1%}, 分钟 15:00 收在涨停率="
              f"{(sub['close_at_limit'] == 1).mean():.1%}")
    else:
        print("[涨停对拍] 无「面板收盘≈涨停」事件可对拍")
    feat["limit_touch_match"] = (sub["touch_any"] == 1).astype(float).reindex(feat.index)
    print(f"[涨停对拍] 全部有分钟 D0 事件: 触板率={feat['touch_any'].mean():.1%} "
          f"(含炸板后回封/尾封)")

    # ---- 由日级中间量合成模型特征 ----
    feat["am_mom"] = feat["morn_close"] / feat["d_open"] - 1.0     # 上午动量 (open→11:30)
    feat["close_vwap_gap"] = feat["d_close"] / feat["vwap"] - 1.0  # 收盘与 VWAP 距离

    # ---- 源数据 0 价/0 量: 向量除法不报错但会产出 inf ⇒ 统一降为 NaN 并报数 ----
    _num = feat.select_dtypes(include=[np.number])
    _n_inf = int(np.isinf(_num.to_numpy()).sum())
    feat[_num.columns] = _num.replace([np.inf, -np.inf], np.nan)
    print(f"[clean] inf→NaN: {_n_inf} 个 cell (源数据 0 价/0 量所致)")

    # ---- 落盘 (丢缺 D0 分钟行, 数量已打印) ----
    out = feat[["symbol", "d0"] + FEAT_COLS + DIAG_COLS].copy()
    out = out[out["n_bars"].notna()].reset_index(drop=True)
    out.to_parquet(a.out, index=False)
    print(f"[out] {a.out}: {out.shape} (键 symbol+d0, 事件行 join 友好)")

    # ---- smoke 摘要: 每特征非空率 + 分布 ----
    print("[摘要] 模型特征非空率 / 分位数:")
    for c in FEAT_COLS:
        s = pd.to_numeric(out[c], errors="coerce")
        nn = s.notna()
        if nn.any():
            print(f"  {c:<18} 非空率={nn.mean():6.1%}  中位={s[nn].median():>12.6g}  "
                  f"P05={s[nn].quantile(0.05):>12.6g}  P95={s[nn].quantile(0.95):>12.6g}")
        else:
            print(f"  {c:<18} 非空率=  0.0%")
    print(f"[摘要] pre5_days_avail: {out['pre5_days_avail'].value_counts().sort_index().to_dict()}")
    print(f"[摘要] n_bars: min={out['n_bars'].min()} 中位={out['n_bars'].median()} "
          f"max={out['n_bars'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
