# -*- coding: utf-8 -*-
"""次日大跌 — 规则筛选 (不用模型, 纯阈值规则)。

标签: 次日 close_hfq[T+1]/close_hfq[T]-1 <= -DROP_TH (默认 -5%)
口径: 所有条件只用 T 日收盘及更早的信息; T 日停牌剔除; 次日跨停牌>15自然日的剔除。
输出: 单条件命中率/提升倍数 + 两段样本稳定性 + 双/三条件组合搜索。
用法: python scripts/bigdrop_rule.py                 # 全样本规则评估
      python scripts/bigdrop_rule.py 000978 002815   # 看这几只今天触发了哪些规则
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from config.settings import PANEL_V3_PATH  # noqa: E402

DROP_TH = 0.05  # 主口径: 次日跌幅 >= 5%
GAP_MAX = 15  # 次日跨停牌自然日上限
SPLIT = "20260101"  # 稳定性切分: 之前=样本内, 之后=OOS
MIN_COV = 0.002  # 组合搜索的覆盖率下限 (股票-日占比)

COLS = [
    "symbol",
    "date",
    "board",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "close_hfq",
    "pctChg",
    "up_limit_raw",
    "down_limit_raw",
    "is_suspended",
    "amount",
    "turnover_rate",
    "free_float_turnover_rate",
    "volume_ratio",
    "bias_5",
    "bias_20",
    "bias_60",
    "winner_ratio",
    "chip_gini",
    "resistance_dist",
    "support_dist",
    "amplitude_5d",
    "intraday_range",
    "ovd_15p",
    "cost_bias",
]


# ---------------------------------------------------------------- 数据
def load() -> pd.DataFrame:
    d = pq.read_table(str(PANEL_V3_PATH), columns=COLS).to_pandas()
    d["symbol"] = (
        d["symbol"]
        .astype(str)
        .str.replace(r"\.(SH|SZ|BJ)$", "", regex=True)
        .str.zfill(6)
    )
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y%m%d")
    d = d.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("symbol", sort=False)

    d["nxt"] = g["close_hfq"].shift(-1)
    d["nxt_date"] = g["date"].shift(-1)
    gap = (
        pd.to_datetime(d["nxt_date"], format="%Y%m%d")
        - pd.to_datetime(d["date"], format="%Y%m%d")
    ).dt.days
    ok = (
        d["nxt"].notna()
        & (gap <= GAP_MAX)
        & d["nxt"].gt(0)
        & d["is_suspended"].fillna(1).eq(0)
    )
    d["fwd"] = (d["nxt"] / d["close_hfq"] - 1.0).where(ok)
    d["y"] = np.where(d["fwd"].notna(), (d["fwd"] <= -DROP_TH).astype(float), np.nan)

    # 位置类
    d["low60_gain"] = (
        d["close_hfq"]
        / g["close_hfq"].transform(lambda s: s.rolling(60, min_periods=60).min())
        - 1.0
    )
    d["hh250"] = g["close_hfq"].transform(
        lambda s: s.rolling(250, min_periods=200).max()
    )
    # K 线形态
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["upper_shadow"] = ((d["high"] - d["close"]) / rng).fillna(0.0)
    up, dn = d["up_limit_raw"], d["down_limit_raw"]
    d["is_limit_up"] = (d["close"] >= up * (1 - 1e-4)).astype(int)
    d["is_limit_dn"] = (d["close"] <= dn * (1 + 1e-4)).astype(int)
    d["zt_break"] = (
        (d["high"] >= up * (1 - 1e-4)) & (d["close"] < up * (1 - 1e-4))
    ).astype(int)
    z = d["is_limit_up"]
    blk = z.ne(z.shift()).groupby(d["symbol"], observed=True).cumsum()
    d["conseq_zt"] = z.groupby([d["symbol"], blk]).cumsum()
    d["gap_open"] = d["open"] / d["pre_close"] - 1.0
    d["open_high_fade"] = ((d["gap_open"] > 0.03) & (d["pctChg"] < 0)).astype(int)
    # 市场态 (T 日收盘可得)
    gd = d.groupby("date")
    mkt = pd.DataFrame({"mkt_ret": gd["pctChg"].median()})
    lu = d.groupby("date")["is_limit_up"].sum()
    ld = d.groupby("date")["is_limit_dn"].sum()
    mkt["mkt_zt_ratio"] = lu / (lu + ld).clip(lower=1)
    d = d.merge(mkt.reset_index(), on="date", how="left")
    return d


# ---------------------------------------------------------------- 条件库
def conditions(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "高位 bias20>15%": d["bias_20"] > 0.15,
        "高位 bias20>25%": d["bias_20"] > 0.25,
        "高位 bias60>30%": d["bias_60"] > 0.30,
        "高位 bias60>50%": d["bias_60"] > 0.50,
        "60日涨>50%": d["low60_gain"] > 0.50,
        "60日涨>80%": d["low60_gain"] > 0.80,
        "近250日新高": d["close_hfq"] >= d["hh250"] * 0.98,
        "获利盘>90%": d["winner_ratio"] > 0.90,
        "获利盘>95%": d["winner_ratio"] > 0.95,
        "筹码集中 gini>0.70": d["chip_gini"] > 0.70,
        "逼近压力位 <2%": d["resistance_dist"] < 0.02,
        "放量 量比>2": d["volume_ratio"] > 2,
        "放量 量比>3": d["volume_ratio"] > 3,
        "换手>10%": d["turnover_rate"] > 10,
        "换手>15%": d["turnover_rate"] > 15,
        "换手>20%": d["turnover_rate"] > 20,
        "长上影>50%": d["upper_shadow"] > 0.50,
        "长上影>70%": d["upper_shadow"] > 0.70,
        "炸板(触涨停未封)": d["zt_break"] == 1,
        "收盘涨停": d["is_limit_up"] == 1,
        "连板>=3": d["conseq_zt"] >= 3,
        "高开低走": d["open_high_fade"] == 1,
        "今日跌>5%": d["pctChg"] <= -5,
        "今日跌停": d["is_limit_dn"] == 1,
        "今日涨>9%": d["pctChg"] >= 9,
        "振幅5日>15%": d["amplitude_5d"] > 15,
        "大盘中位跌>1%": d["mkt_ret"] < -1,
        "跌停多于涨停": d["mkt_zt_ratio"] < 0.3,
    }


def stats(y: np.ndarray, m: np.ndarray, fwd: np.ndarray, base: float):
    n = int(m.sum())
    if n == 0:
        return None
    p = float(y[m].mean())
    return dict(
        n=n, cov=n / len(y), prec=p, lift=p / base, ret=float(np.nanmean(fwd[m]))
    )


def main() -> None:
    d = load()
    val = d["y"].notna().to_numpy()
    y = d.loc[val, "y"].to_numpy(float)
    fwd = d.loc[val, "fwd"].to_numpy(float)
    dv = d.loc[val, "date"].to_numpy()
    base = y.mean()
    print(
        f"样本 {len(y):,} 股票-日 ({d['symbol'].nunique():,} 只, {dv.min()}~{dv.max()})"
    )
    print(
        f"基准: 次日跌幅>={DROP_TH:.0%} 占 {base:.3%}  "
        f"(次日收益 中位 {np.median(fwd):.3%} 均值 {np.nanmean(fwd):.3%})"
    )

    C = {k: v.to_numpy(bool)[val] for k, v in conditions(d).items()}
    tr = dv < SPLIT
    te = ~tr

    print(f"\n{'=' * 116}\n(a) 单条件 — 命中次日大跌的概率 (按 lift 排序)")
    print(
        f"{'条件':<22}{'股票-日':>10}{'占比':>8}{'次日大跌率':>11}{'lift':>7}"
        f"{'次日均收益':>11}|{'样本内lift':>11}{'OOS lift':>10}"
    )
    rows = []
    for k, m in C.items():
        s = stats(y, m, fwd, base)
        if s is None:
            continue
        si = stats(y[tr], m[tr], fwd[tr], base)
        so = stats(y[te], m[te], fwd[te], base)
        rows.append((k, s, si, so))
    rows.sort(key=lambda r: -r[1]["lift"])
    for k, s, si, so in rows:
        li = f"{si['lift']:.2f}" if si else "-"
        lo = f"{so['lift']:.2f}" if so else "-"
        print(
            f"{k:<22}{s['n']:>10,}{s['cov']:>8.2%}{s['prec']:>11.2%}"
            f"{s['lift']:>7.2f}{s['ret']:>11.3%}|{li:>11}{lo:>10}"
        )

    # ---------------- 组合搜索 ----------------
    keys = [r[0] for r in rows]
    print(f"\n{'=' * 116}\n(b) 双条件 AND 组合 (覆盖率>={MIN_COV:.1%}, lift 前 15)")
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a.split()[0] == b.split()[0]:
                continue
            m = C[a] & C[b]
            if m.mean() < MIN_COV:
                continue
            s = stats(y, m, fwd, base)
            si = stats(y[tr], m[tr], fwd[tr], base)
            so = stats(y[te], m[te], fwd[te], base)
            if s and si and so:
                pairs.append((f"{a} + {b}", s, si, so))
    pairs.sort(key=lambda r: -r[1]["lift"])
    print(
        f"{'组合':<46}{'股票-日':>10}{'占比':>8}{'次日大跌率':>11}{'lift':>7}"
        f"|{'样本内lift':>11}{'OOS lift':>10}"
    )
    for k, s, si, so in pairs[:15]:
        print(
            f"{k:<46}{s['n']:>10,}{s['cov']:>8.2%}{s['prec']:>11.2%}"
            f"{s['lift']:>7.2f}|{si['lift']:>11.2f}{so['lift']:>10.2f}"
        )

    print(
        f"\n{'=' * 116}\n(c) 三条件 AND (从前 40 个双条件再叠一条, 同段稳定者, lift 前 10)"
    )
    tri = []
    for k, _, _, _ in pairs[:40]:
        a, b = [x.strip() for x in k.split(" + ")]
        for c in keys:
            if c.split()[0] in (a.split()[0], b.split()[0]):
                continue
            m = C[a] & C[b] & C[c]
            if m.mean() < MIN_COV:
                continue
            s = stats(y, m, fwd, base)
            si = stats(y[tr], m[tr], fwd[tr], base)
            so = stats(y[te], m[te], fwd[te], base)
            if s and si and so and si["lift"] > 1 and so["lift"] > 1:
                tri.append((f"{a} + {b} + {c}", s, si, so))
    tri.sort(key=lambda r: -r[1]["lift"])
    print(
        f"{'组合':<58}{'股票-日':>10}{'占比':>8}{'大跌率':>9}{'lift':>7}"
        f"|{'样本内':>9}{'OOS':>8}"
    )
    for k, s, si, so in tri[:10]:
        print(
            f"{k:<58}{s['n']:>10,}{s['cov']:>8.2%}{s['prec']:>9.2%}"
            f"{s['lift']:>7.2f}|{si['lift']:>9.2f}{so['lift']:>8.2f}"
        )

    # ---------------- 个股查询 ----------------
    if len(sys.argv) > 1:
        want = [a.zfill(6) for a in sys.argv[1:]]
        C_all = conditions(d)
        last = d.groupby("symbol")["date"].transform("max") == d["date"]
        cur = d[last & d["symbol"].isin(want)].copy()
        print(f"\n{'=' * 116}\n(d) 个股查询 — 最新交易日 {d['date'].max()}")
        for sym in want:
            r = cur[cur["symbol"] == sym]
            if not len(r):
                print(f"\n{sym}  不在面板最新交易日")
                continue
            i = r.index[0]
            hit = [k for k, v in C_all.items() if bool(v.loc[i])]
            print(
                f"\n{sym}  收盘 {r['close'].iat[0]:.2f}  "
                f"当日 {float(r['pctChg'].iat[0]):+.2f}%  "
                f"bias20 {float(r['bias_20'].iat[0]):+.1%}  "
                f"获利盘 {float(r['winner_ratio'].iat[0]):.2f}  "
                f"换手 {float(r['turnover_rate'].iat[0]):.2f}%"
            )
            print("   触发: " + ("; ".join(hit) if hit else "无"))


if __name__ == "__main__":
    main()
