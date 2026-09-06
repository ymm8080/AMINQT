"""差值加速度影子单: (近5日涨幅 - 前一个5日涨幅) top10 → 同花顺自选股 (2026-09-04 用户拍板).

口径 (250d 全市场回放, 09-04 对比扫描): 差值 = r5 - r5_prev (百分比差, 非比值 —
  严格比值 top10 ≈ 市场基线无信号, 09-04 判死)。
  池 = 全市场差值 top10, 无桶过滤 — 三桶 (rebound/costbias/deeppull) 会杀差值臂
  79% 赢家 (2.78→0.58/日), 09-04 判死勿再加。
回放: 10.0 只/日, 赢家 2.78/日 (prod main 1.21 的 2.3x), 赢率 27.8%
  (h1 26.7/h2 28.9 双稳), 深跌 36.6%, 均值 -0.49pp (h1 -0.95/h2 -0.02) —
  左尾肥与旧累计池同签名, 买的是赢家密度。与旧三桶池重叠仅 3%。
历史沿革: a1union 纯并集 (4.69 赢家/日) → a1tri 三桶 (1.33 赢家/日) →
  a1diff 差值加速度 (2.78 赢家/日, 本版)。累计动量臂 (A1/A1raw/bias60) 已换掉。

生成 (STOCK_LIST_DIR, WORM):
  a1diff_{date}__a1diff.csv              交付文档 (rank/symbol/pctChg/diff/r5/r5p)
  ths_watchlist_{date}__a1diff.txt       ths_push 同款导入格式; _ths_flush_guard
                                         成员并集 glob (ths_watchlist_*__*.txt)
                                         自动覆盖本清单, 盘中放量下跌守卫同样生效

用法: python scripts/_a1_momentum_shadow.py [YYYYMMDD] [--gen-only] [--dry-run]
  缺省 date = 面板最新交易日; 面板无该日数据则跳过 (fail-safe, 链上非关键步骤)。
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PANEL_V3_PATH, STOCK_LIST_DIR

ARM_TOP_N = 10
MODULE = "a1diff"
R_WIN = 5   # 差值两腿各 5 个交易日


def diff_picks(close: pd.DataFrame, day_ts: pd.Timestamp, arm_top_n: int = ARM_TOP_N) -> pd.DataFrame:
    """close = 透视表 (date × symbol, close_hfq) → 当日差值 top10 (纯函数, 可单测).

    diff = r5 - r5_prev; 历史不足 R_WIN*2+1 交易日为 NaN 剔除。
    返回列: rank/symbol/diff/r5/r5p, diff 降序。
    """
    c = close[close.index <= day_ts].sort_index()
    need = R_WIN * 2 + 1
    if len(c) < need:
        return pd.DataFrame(columns=["rank", "symbol", "diff", "r5", "r5p"])
    last = c.iloc[-1]
    r5 = last / c.iloc[-1 - R_WIN] - 1
    r5p = c.iloc[-1 - R_WIN] / c.iloc[-1 - 2 * R_WIN] - 1
    d = (r5 - r5p).dropna()
    d = d[~d.index.str.endswith(".BJ")]  # 北交所剔除 (2026-09-04 用户指示, 候选阶段剔, top10 补满)
    top = d.nlargest(arm_top_n)
    out = pd.DataFrame({
        "symbol": top.index,
        "diff": top.values,
        "r5": r5.reindex(top.index).values,
        "r5p": r5p.reindex(top.index).values,
    }).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


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
        print(f"[a1diff] 面板无 {date} 数据, 跳过 (fail-safe)")
        return 0

    close = pd.read_parquet(
        PANEL_V3_PATH, columns=["symbol", "date", "close_hfq"],
        filters=[("date", ">=", day_ts - pd.Timedelta(days=60)),
                 ("date", "<=", day_ts)],
    )
    close["symbol"] = close["symbol"].astype(str).str.zfill(6)
    piv = close.pivot(index="date", columns="symbol", values="close_hfq")
    picks = diff_picks(piv, day_ts)
    if picks.empty:
        print(f"[a1diff] {date} 有效候选不足, 跳过")
        return 0
    day["symbol"] = day["symbol"].astype(str).str.zfill(6)
    picks = picks.merge(day[["symbol", "pctChg"]], on="symbol", how="left")

    csv_path = STOCK_LIST_DIR / f"a1diff_{date}__{MODULE}.csv"
    picks.to_csv(csv_path, index=False, encoding="utf-8-sig")
    txt_path = STOCK_LIST_DIR / f"ths_watchlist_{date}__{MODULE}.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(picks["symbol"]) + "\n")
    print(f"[a1diff] {csv_path}")
    print(f"[a1diff] {txt_path} ({len(picks)} 只)")
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
