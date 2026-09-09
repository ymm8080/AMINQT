# -*- coding: utf-8 -*-
"""常态波动下爱冲高回落个股的特征画像.

样本池 (常态冲高):
  - 个股日内最大涨幅 g = high/pre_close - 1 >= 3%
  - 剔除触涨停日 (high >= round(pre_close*涨停比例,2)-0.001) — 制度性炸板不算常态
  - 剔除指数极端日 (上证 |ret|>2%) — 剔除系统性崩盘/暴涨
度量:
  - giveback = (high-close)/(high-pre_close)  吐回比例 (>1 表示冲高后翻绿)
  - fade     = giveback >= 0.7                冲高回落日
特征 (除 gap/vratio 外均为 t-1 已知, 无 look-ahead):
  r5/r20/r60 前期涨幅, vol20 波动, turn20 换手, mcap 流通市值proxy, pos250 距一年高点,
  vratio 当日量比(盘中可知), gap 当日开盘跳空
"""
import numpy as np
import pandas as pd

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
IDX = r"data/processed/sh_index_daily_20260909.parquet"
MIN_CS = 30          # 截面最少股票数
POOL_G = 0.03        # 冲高门槛
FADE_GIVEBACK = 0.7  # 吐回七成算回落


def load_wide() -> dict:
    df = pd.read_parquet(PANEL, columns=[
        "symbol", "date", "open", "high", "low", "close", "pre_close",
        "volume", "amount", "turnover_rate"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    bad = (df["high"] < df["low"]) | (df["high"] < df[["open", "close"]].max(axis=1)) \
        | (df["low"] > df[["open", "close"]].min(axis=1)) | (df["volume"] < 0)
    if bad.any():
        print(f"[QC] OHLCV 异常 {bad.sum()} 行剔除")
        df = df[~bad]
    w = {}
    for c in ["open", "high", "low", "close", "pre_close", "amount", "turnover_rate"]:
        w[c] = df.pivot(index="date", columns="symbol", values=c).sort_index()
    return w


def build(w: dict) -> tuple[dict, pd.DataFrame]:
    idx = pd.read_parquet(IDX).sort_values("trade_date")
    idx["date"] = pd.to_datetime(idx["trade_date"])
    idx = idx.set_index("date")
    idx_ret = idx["close"] / idx["pre_close"] - 1
    idx_ma20 = idx["close"].rolling(20).mean()
    idx_above = (idx["close"] / idx_ma20 - 1)

    o, h, l, c, pc = w["open"], w["high"], w["low"], w["close"], w["pre_close"]
    amt, turn = w["amount"], w["turnover_rate"]
    syms = c.columns
    ratio = pd.Series([0.2 if s[:3] in ("300", "301", "688", "689") else 0.1 for s in syms], index=syms)

    g = h / pc - 1
    # 面板价格为复权价, 涨停价无法硬算 -> 行为学判定: 日内最大涨幅达到板幅-0.4% 视为触板
    touch_limit = g.sub(ratio, axis=1) >= -0.004
    ret = c / pc - 1
    gap = o / pc - 1
    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng
    giveback = (h - c) / (h - pc)

    ok_day = pd.Series((idx_ret.abs() <= 0.02).values, index=idx_ret.index)
    dates = c.index
    ok_row = ok_day.reindex(dates).fillna(False)
    regime = pd.cut(idx_above, [-np.inf, -0.02, 0, 0.02, np.inf],
                    labels=["深跌破MA20", "贴线下", "贴线上", "线上强"]).reindex(dates)

    pool = (g >= POOL_G) & (~touch_limit) & pd.DataFrame(
        np.broadcast_to(ok_row.values[:, None], c.shape), index=dates, columns=syms)

    # t-1 特征
    r5 = c.pct_change(5, fill_method=None)
    r20 = c.pct_change(20, fill_method=None)
    r60 = c.pct_change(60, fill_method=None)
    vol20 = ret.rolling(20).std()
    turn20 = turn.rolling(20).mean()
    amt20 = amt.rolling(20).mean()
    pos250 = c / h.rolling(250, min_periods=60).max() - 1
    mcap = np.log((amt20 / turn20 * 100).clip(lower=1e7))  # 流通市值(元)取对数
    vratio = (amt / amt20.shift(1)).clip(upper=30)           # 当日量比(盘中可知), 截尾防爆炸

    feats = {"r5": r5.shift(1), "r20": r20.shift(1), "r60": r60.shift(1),
             "vol20": vol20.shift(1), "turn20": turn20.shift(1), "mcap": mcap.shift(1),
             "pos250": pos250.shift(1), "vratio": vratio, "gap": gap}
    mats = {"g": g, "ret": ret, "giveback": giveback, "close_pos": close_pos,
            "ret1": c.shift(-1) / c - 1}
    return {"feats": feats, "mats": mats, "pool": pool, "regime": regime,
            "dates": dates, "syms": syms}


def rowwise_corr(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    xm = x.sub(x.mean(axis=1), axis=0)
    ym = y.sub(y.mean(axis=1), axis=0)
    denom = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    return (xm * ym).sum(axis=1) / denom.replace(0, np.nan)


def main():
    w = load_wide()
    B = build(w)
    feats, mats, pool = B["feats"], B["mats"], B["pool"]
    dates, syms = B["dates"], B["syms"]
    n_pool = pool.sum().sum()
    print("=" * 76)
    print(f"[池] 常态冲高样本 (g>={POOL_G:.0%}, 无触板, 指数平稳日): {n_pool} 股票日, "
          f"{int(pool.sum(axis=1).mean()):.0f} 只/日均")
    gb_pool = mats["giveback"].where(pool)
    fade = (mats["giveback"] >= FADE_GIVEBACK).where(pool)
    fade_rate = fade.sum().sum() / n_pool if n_pool else float("nan")
    print(f"  池内 giveback 中位 {gb_pool.stack().median():.2f} | 回落率(吐回>=70%) {fade_rate:.1%}")
    print(f"  池内当日收益中位 {mats['ret'].where(pool).stack().median():.2%} | 次日收益中位 {mats['ret1'].where(pool).stack().median():.2%}")

    # ── A. 日截面秩 IC: t-1 特征 vs 当日 giveback ──
    print("=" * 76)
    print("[A] 日截面 Spearman IC (特征 t-1 / 盘中可知 vs 当日吐回比例), 2023-01~2026-09")
    give_rank = gb_pool.rank(axis=1)
    rows = []
    for name, f in feats.items():
        fr = f.where(pool).rank(axis=1)
        cnt = f.where(pool).notna().sum(axis=1)
        ic = rowwise_corr(fr, give_rank)
        ic = ic[cnt >= MIN_CS].dropna()
        if len(ic) < 100:
            continue
        tstat = ic.mean() / ic.std() * np.sqrt(len(ic))
        rows.append({"特征": name, "IC均值": ic.mean(), "IC标准差": ic.std(),
                     "t": tstat, "IC>0占比": (ic > 0).mean(), "天数": len(ic)})
    ic_tab = pd.DataFrame(rows).sort_values("IC均值")
    print(ic_tab.round(4).to_string(index=False))

    # ── B. 分位表: t-1 特征五分位 → 当日吐回/回落率/次日收益 ──
    print("=" * 76)
    print("[B] 特征五分位 (Q1 低 ~ Q5 高) → 池内表现")
    long = {}
    for name, f in feats.items():
        long[name] = f.where(pool).stack()
    L = pd.DataFrame(long)
    L["giveback"] = gb_pool.stack()
    L["fade"] = fade.stack()
    L["ret1"] = mats["ret1"].where(pool).stack()
    L["ret"] = mats["ret"].where(pool).stack()
    print(f"  长表 {len(L)} 行")
    for name in feats:
        if name not in L or L[name].isna().all():
            continue
        sub = L.dropna(subset=[name])
        sub = sub.assign(q=pd.qcut(sub[name], 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
        t2 = sub.groupby("q", observed=True).agg(
            giveback中位=("giveback", "median"), 回落率=("fade", "mean"),
            次日收益=("ret1", "mean"), n=(name, "size"))
        spread = t2["回落率"].iloc[-1] - t2["回落率"].iloc[0]
        print(f"  -- {name}  (Q5-Q1 回落率差 {spread:+.1%})")
        print(t2.round(3).to_string())

    # ── C. 惯犯持续性: 前后半样本股票级回落率相关 ──
    print("=" * 76)
    print("[C] 冲高回落倾向是否股票稳定特质 (前后半样本)")
    fade_long = fade.stack().rename("fade")
    gb_long = gb_pool.stack().rename("gb")
    half = dates[dates <= dates[len(dates) // 2]][-1]
    D = pd.DataFrame({"fade": fade_long, "gb": gb_long})
    D["half"] = np.where(D.index.get_level_values(0) <= half, "前", "后")
    piv = D.groupby([D.index.get_level_values(1), "half"]).agg(
        fade=("fade", "mean"), n=("fade", "size")).unstack("half")
    piv = piv[(piv[("n", "前")] >= 8) & (piv[("n", "后")] >= 8)]
    corr = piv[("fade", "前")].corr(piv[("fade", "后")], method="spearman")
    top = piv[piv[("fade", "前")] >= piv[("fade", "前")].quantile(0.8)]
    bot = piv[piv[("fade", "前")] <= piv[("fade", "前")].quantile(0.2)]
    print(f"  样本劈分点 {half:%Y-%m-%d}, 合格股票 {len(piv)}")
    print(f"  前半回落率 vs 后半回落率 Spearman: {corr:.3f}")
    print(f"  前半Top20%惯犯 -> 后半回落率 {top[('fade', '后')].mean():.1%} | "
          f"前半Bottom20% -> 后半 {bot[('fade', '后')].mean():.1%} | 全体后半 {piv[('fade', '后')].mean():.1%}")

    # ── D. 惯犯特征画像 ──
    print("=" * 76)
    print("[D] 惯犯画像: 股票级回落率 Top30% vs Bottom30% 的 t-1 特征均值")
    sf = fade_long.groupby(level=1).agg(["mean", "size"])
    sf = sf[sf["size"] >= 30]
    hi = sf[sf["mean"] >= sf["mean"].quantile(0.7)].index
    lo = sf[sf["mean"] <= sf["mean"].quantile(0.3)].index
    prof = {}
    for name in feats:
        s = L[name].groupby(level=1).mean()
        prof[name] = [s.reindex(hi).mean(), s.reindex(lo).mean()]
    pf = pd.DataFrame(prof, index=["惯犯Top30%", "温和Bottom30%"]).T
    pf["差(Top-Bot)"] = pf.iloc[:, 0] - pf.iloc[:, 1]
    print(pf.round(3).to_string())
    print(f"  (惯犯 {len(hi)} 只, 温和 {len(lo)} 只; 特征=池内日均值)")
    print("  提示: mcap 为 ln(流通市值元), pos250 为距一年高点距离(负=接近高点)")

    # ── E. 行情交互: 指数状态 → 个股池内回落率 ──
    print("=" * 76)
    print("[E] 行情交互: 上证位置 → 常态冲高样本的回落率")
    reg = B["regime"]
    reg_row = pd.Series(reg.values, index=dates)
    fade_by_day = fade.mean(axis=1)
    gb_by_day = gb_pool.mean(axis=1)
    cnt_by_day = pool.sum(axis=1)
    t3 = pd.DataFrame({"fade": fade_by_day, "gb": gb_by_day, "n": cnt_by_day,
                       "reg": reg_row}).dropna(subset=["reg"])
    t3 = t3[t3["n"] >= MIN_CS]
    print(t3.groupby("reg", observed=True).agg(
        日均回落率=("fade", "mean"), 吐回中位=("gb", "median"), 天数=("n", "size")).round(3).to_string())

    # ── F. 池内直接对比: 当日回落 vs 当日守住 -> 次日 ──
    print("=" * 76)
    print("[F] 同为常态冲高, 当日回落(吐回>=70%) vs 守住(<40%) 的次日/次日波动")
    grp = pd.Series("中间", index=L.index)
    grp[L["giveback"] >= FADE_GIVEBACK] = "回落"
    grp[L["giveback"] < 0.40] = "守住"
    L2 = L.assign(grp=grp)
    f6 = L2[L2["grp"] != "中间"].groupby("grp").agg(
        n=("ret1", "size"), 次日均值=("ret1", "mean"), 次日中位=("ret1", "median"),
        次日上涨率=("ret1", lambda x: (x > 0).mean()))
    print((f6[["次日均值", "次日中位", "次日上涨率"]] * [100, 100, 100]).round(2).to_string())
    # 分行情状态看回落日的次日
    reg_map = pd.Series(reg.values, index=dates)
    L2["reg"] = L2.index.get_level_values(0).map(reg_map)
    f6b = L2[L2["grp"] == "回落"].groupby("reg", observed=True)["ret1"].agg(["mean", "median", "size"])
    print("  回落日次日按行情状态:")
    print((f6b[["mean", "median"]] * 100).round(2).to_string())


if __name__ == "__main__":
    main()
