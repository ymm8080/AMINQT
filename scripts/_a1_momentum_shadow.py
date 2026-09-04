"""A1 动量影子单: 当日 A1 top10 → 同花顺自选股 (2026-09-04 用户批准并推).

口径 (125d 回放 tmp_t/_mom_flush_sub_0903.py, 判决 2):
  A1 = bias_20/60/120/250 当日截面百分位秩 nanmean (scripts/_diag_prerise_detector.py 同式)
  剔当日强涨 (pctChg>=9.5%, 双创 19%) — prerise 交付清单同款交付语义
  取 A1 降序 top10。
回放: 赢家密度 2.3x prod TOP10 (28% vs 12-15% 每票 ≥5% 命中, 双板双半窗稳),
仅影子观察 — 与 prod TOP10 并推自选股, 不改生产排名键。

生成 (STOCK_LIST_DIR, WORM):
  a1mom_top10_{date}__a1mom.csv          交付文档 (rank/symbol/pctChg/a1)
  ths_watchlist_{date}__a1mom.txt        ths_push 同款导入格式; _ths_flush_guard
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

BIAS_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
TOP_N = 10
MODULE = "a1mom"


def a1_top10(day: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """单日截面 → A1 top10 (纯函数, 可单测).

    day 须含 symbol/bias_20/bias_60/bias_120/bias_250/pctChg 列, 单个交易日切片。
    返回列: rank/symbol/pctChg/a1 (a1=截面百分位秩均值 0~1)。
    """
    df = day.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    a1 = pd.concat(
        [df[c].rank(pct=True) for c in BIAS_COLS], axis=1
    ).mean(axis=1, skipna=True)
    out = pd.DataFrame({"symbol": df["symbol"], "pctChg": df["pctChg"], "a1": a1})
    hot = np.where(
        out["symbol"].str.startswith(("30", "68")),
        out["pctChg"] >= 19.0,
        out["pctChg"] >= 9.5,
    )
    out = out[~hot].dropna(subset=["a1"])
    out = out.sort_values("a1", ascending=False).head(top_n).reset_index(drop=True)
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
        columns=["symbol", "pctChg", *BIAS_COLS],
        filters=[("date", "=", day_ts)],
    )
    if day.empty:
        print(f"[a1] 面板无 {date} 数据, 跳过 (fail-safe)")
        return 0

    picks = a1_top10(day)
    if picks.empty:
        print(f"[a1] {date} 无有效 A1 候选, 跳过")
        return 0

    csv_path = STOCK_LIST_DIR / f"a1mom_top10_{date}__{MODULE}.csv"
    picks.to_csv(csv_path, index=False, encoding="utf-8-sig")
    txt_path = STOCK_LIST_DIR / f"ths_watchlist_{date}__{MODULE}.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(picks["symbol"]) + "\n")
    print(f"[a1] {csv_path}")
    print(f"[a1] {txt_path} ({len(picks)} 只)")
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
