"""隔板口袋影子单: 首板后 d3~7 缩量守板回踩 top15 → 同花顺自选股 (2026-09-04 用户拍板).

口径 (250d 全市场回放, tmp_t/_gap_d37_final_0904.py + _risk_audit_0904.py):
  池 = 首板(前一日非板)后第 3~7 夜, 期间无再板(事件结束), 且
       守板价 = 当日收 >= 首板日收, 缩量 = 近3日均量/首板日量 < 0.8 (09-05 拍板 0.7→0.8)。
  安全闸 (09-04 用户拍板全单统一): 剔 10日涨幅>30% 或 换手>15% (nan→0 视为过闸)。
  键 = bias60 最热 top15 (top10(diff) 对照: 7.7只/日 +0.64pp 1.61赢家/日;
       top15(bias60) 9.5只/日 +0.66pp 1.98赢家/日 — 用户拍板 TOP15)。
  09-05 池扫描 (tmp_t/_gap_pool_sweep_0905.py): dry0.8 边际桶自身 5.0只/日 +0.52pp
       ≈基线质量; 合并后 11.4只/日 20.7% +0.52pp 2.36赢家/日 大亏4.4%
       (vs dry0.7: 8.7只/日 +0.55pp 1.73赢家 大亏3.7%) — 赢家密度+36%, 均值-0.03pp。
       其余放宽(窗/守板/dry0.9+)与精品(dry<=0.5)全判死, 池条件线关闭。
  交易口径: 次日接力 T+1收买 T+4收卖。
  北交所剔除; 窗口 d3~7 已定版勿再扫。

生成 (STOCK_LIST_DIR, WORM):
  gappocket_{date}__gappocket.csv         交付文档 (rank/symbol/d/dry/pull/r10/tov/pctChg)
  ths_watchlist_{date}__gappocket.txt     ths_push 同款导入格式; _ths_flush_guard
                                          成员并集 glob (ths_watchlist_*__*.txt) 自动覆盖

用法: python scripts/_gap_pocket_shadow.py [YYYYMMDD] [--gen-only] [--dry-run]
  缺省 date = 面板最新交易日; 面板无该日数据则跳过 (fail-safe, 链上非关键步骤)。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PANEL_V3_PATH, STOCK_LIST_DIR

MODULE = "gappocket"
GAP_TOP_N = 15
D_LO, D_HI = 3, 7  # 首板后夜数窗口 (09-04 拍板, 勿再扫)
DRY_MAX = 0.8  # 近3日均量/首板日量 上限 (09-05 拍板 0.7→0.8, 赢家密度+36%)
R10_MAX = 0.30  # 安全闸: 10日涨幅上限
TOV_MAX = 15.0  # 安全闸: 换手率上限
BIAS_WIN = 60  # bias 排名窗
LOOKBACK_DAYS = 180  # 面板读取日历窗 (bias60 + 闸 needs ~70 交易日)


def gap_pocket_picks(
    close: pd.DataFrame,
    pctchg: pd.DataFrame,
    amount: pd.DataFrame,
    turnover: pd.DataFrame,
    day_ts: pd.Timestamp,
    top_n: int = GAP_TOP_N,
) -> pd.DataFrame:
    """四个透视表 (date × symbol) → 当日隔板口袋 top15 (纯函数, 可单测).

    返回列: rank/symbol/d/dry/pull/r10/tov, bias60 降序 (最热优先)。
    bias 不足 60 交易日为 NaN 剔除; 北交所在候选阶段剔除。
    """
    idx = close.index[close.index <= day_ts]
    if len(idx) < BIAS_WIN + 1:
        return pd.DataFrame(
            columns=["rank", "symbol", "d", "dry", "pull", "r10", "tov"]
        )
    c = close.loc[idx]
    p = pctchg.loc[idx]
    a = amount.loc[idx]
    tv = turnover.loc[idx]
    syms = [s for s in c.columns if not s.endswith(".BJ")]
    c, p, a, tv = c[syms], p[syms], a[syms], tv[syms]

    big = np.asarray([s.startswith(("300", "688")) for s in syms])
    lim = (p >= np.where(big[None, :], 19.5, 9.8)).fillna(False)
    first = lim & ~lim.shift(1, fill_value=False)
    lv = lim.values

    t = len(idx) - 1
    bias = c.iloc[t] / c.rolling(BIAS_WIN).mean().iloc[t] - 1
    bias = bias.dropna()
    if bias.empty:
        return pd.DataFrame(
            columns=["rank", "symbol", "d", "dry", "pull", "r10", "tov"]
        )

    cv, av, tvv = c.values, a.values, tv.values
    rows = []
    for i0 in range(max(0, t - D_HI), t):
        d = t - i0
        if not (D_LO <= d <= D_HI):
            continue
        for j in np.where(first.values[i0])[0]:
            c0, a0 = cv[i0, j], av[i0, j]
            if np.isnan(c0) or np.isnan(a0) or a0 <= 0 or np.isnan(cv[t, j]):
                continue
            if lv[i0 + 1 : t + 1, j].any():  # 事件期内再板 → 事件结束
                continue
            pull = cv[t, j] / c0 - 1
            if pull < 0:  # 守板价
                continue
            dry = np.nanmean(av[max(i0, t - 2) : t + 1, j]) / a0
            if not (dry < DRY_MAX):  # 缩量
                continue
            r10 = cv[t, j] / cv[t - 10, j] - 1 if not np.isnan(cv[t - 10, j]) else 0.0
            tov = tvv[t, j]
            if not (
                np.nan_to_num(r10, nan=0.0) <= R10_MAX
                and np.nan_to_num(tov, nan=0.0) <= TOV_MAX
            ):
                continue  # 安全闸
            rows.append(
                (
                    syms[j],
                    d,
                    dry,
                    pull,
                    r10,
                    0.0 if np.isnan(tov) else tov,
                    bias.get(syms[j], np.nan),
                )
            )
    if not rows:
        return pd.DataFrame(
            columns=["rank", "symbol", "d", "dry", "pull", "r10", "tov"]
        )
    out = pd.DataFrame(
        rows, columns=["symbol", "d", "dry", "pull", "r10", "tov", "bias"]
    )
    out = out.dropna(subset=["bias"])
    out = out.sort_values(["bias", "symbol"], ascending=[False, True]).head(top_n)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen_only = "--gen-only" in sys.argv
    dry_run = "--dry-run" in sys.argv

    if args:
        day_ts = pd.Timestamp(args[0])
    else:
        dates = pd.read_parquet(PANEL_V3_PATH, columns=["date"])
        day_ts = pd.to_datetime(dates["date"]).max()
    date = day_ts.strftime("%Y%m%d")

    day = pd.read_parquet(
        PANEL_V3_PATH,
        columns=["symbol", "pctChg"],
        filters=[("date", "=", day_ts)],
    )
    if day.empty:
        print(f"[gappocket] 面板无 {date} 数据, 跳过 (fail-safe)")
        return 0

    panel = pd.read_parquet(
        PANEL_V3_PATH,
        columns=["symbol", "date", "close_hfq", "pctChg", "amount", "turnover_rate"],
        filters=[
            ("date", ">=", day_ts - pd.Timedelta(days=LOOKBACK_DAYS)),
            ("date", "<=", day_ts),
        ],
    )
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    close = panel.pivot(index="date", columns="symbol", values="close_hfq").sort_index()
    pct = panel.pivot(index="date", columns="symbol", values="pctChg").sort_index()
    amt = panel.pivot(index="date", columns="symbol", values="amount").sort_index()
    tov = panel.pivot(
        index="date", columns="symbol", values="turnover_rate"
    ).sort_index()

    picks = gap_pocket_picks(close, pct, amt, tov, day_ts)
    if picks.empty:
        print(f"[gappocket] {date} 无隔板口袋候选, 跳过")
        return 0
    day["symbol"] = day["symbol"].astype(str).str.zfill(6)
    picks = picks.merge(day[["symbol", "pctChg"]], on="symbol", how="left")

    csv_path = STOCK_LIST_DIR / f"{MODULE}_{date}__{MODULE}.csv"
    picks.to_csv(csv_path, index=False, encoding="utf-8-sig")
    txt_path = STOCK_LIST_DIR / f"ths_watchlist_{date}__{MODULE}.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(picks["symbol"]) + "\n")
    print(f"[gappocket] {csv_path}")
    print(f"[gappocket] {txt_path} ({len(picks)} 只)")
    print(picks.to_string(index=False))

    if not gen_only:
        from scripts._ths_ui import THS_HEXIN_PATH
        from scripts._ths_watchlist_push import push_via_ths

        if not THS_HEXIN_PATH.exists():
            print(f"[warn] 同花顺客户端不存在: {THS_HEXIN_PATH}")
            return 0
        if not push_via_ths(txt_path, dry_run):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
