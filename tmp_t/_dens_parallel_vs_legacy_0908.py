# -*- coding: utf-8 -*-
"""密度线 prob 源对比: legacy 概率头 vs parallel (2026-09-08 用户拍板要跑).

问题: 密度线 (prob 前20带 + occ5>=3 + 回撤闸) 的选股源定在 legacy 概率头,
换成 parallel 的 pred_prob_10d 是否同样成立?

口径照抄生产 _prob10_density_shadow.py:
  带 = 每板 (main; dual=GEM+STAR) prob 降序前20
  occ5 = rolling(5) 含当日 >=3  (今日在带 + 近4个上榜日在带数)
  回撤闸 = 收盘距10日高点 >= -10%
  派发闸两边同撤 (研究对称: 关键变量 = prob 源; 派发闸与源无关, 同撤不偏袒)

臂:
  L0  legacy 带 occ5>=1        (无密度闸, 看 occ5 边际贡献)
  L1  legacy 带 occ5>=3+回撤闸  (生产口径重算参照)
  PB  parallel prob 带 occ5>=3+回撤闸  (真口径, 但 preds_raw 只有 24 天)
  PA0 parallel 1y top10 选票全体      (run_dir 当前配置重放, score 键非 prob)
  PA1 parallel 1y top10 选票 occ5>=3  (概念版: 持续上榜密度在 parallel 排名上)

标签统一 harness: panel close c2c fwd10 (t+10 收盘/ t 收盘 -1);
  win = fwd10>0; excess_pp = win率 - 当日全股win率; 另报 mean/p5.
  (生产引用数 54.7%/+7.82pp 的 excess 定义与此一致量级; 本脚本两侧同 harness
   内部自洽, 绝对值与引用数可有小差。)
"""

import glob
import os

import numpy as np
import pandas as pd

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
HIST = r"D:\AMINQT\AMINQT CODES\data\prob10_density_history.parquet"
PREDS = r"D:\AMINQT\Daily Operation\STOCK LIST\parallel_preds_raw_*.csv"
RUNDIR = r"D:\AMINQT\DATA OTHERS\BACKTESTING RESULT\20260908_154759"
OUT = os.path.splitext(__file__)[0] + "_report.txt"

TOP_N, OCC_MIN, PULL_FLOOR, OCC_WIN = 20, 3, -0.10, 5
EVAL_DAYS = 125  # 主评估窗 (交易日)


def load_panel():
    import pyarrow.parquet as pq

    tb = pq.read_table(PANEL, columns=["date", "symbol", "close"])
    df = tb.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)["close"]
    df["fwd10"] = g.shift(-10) / df["close"] - 1.0
    df["hi10"] = g.transform(lambda s: s.rolling(10, min_periods=1).max())
    df["pull"] = df["close"] / df["hi10"] - 1.0
    return df


def baseline(panel):
    d = panel.dropna(subset=["fwd10"])
    return d.groupby("date")["fwd10"].agg(["mean", lambda s: (s > 0).mean()])


def eval_picks(picks, panel, base, tag, min_hist_dates=None):
    """picks: DataFrame[date, symbol] → 指标行. 只评有 t+10 标签的."""
    m = picks.merge(
        panel[["date", "symbol", "fwd10", "pull"]], on=["date", "symbol"], how="inner"
    )
    m = m.dropna(subset=["fwd10"])
    if m.empty:
        return dict(arm=tag, days=0, picks_day=0.0, win=np.nan, base=np.nan,
                    exc=np.nan, mean=np.nan, p5=np.nan)
    m = m.merge(base, left_on="date", right_index=True, how="left")
    days = m["date"].nunique()
    n = len(m)
    return dict(
        arm=tag,
        days=days,
        picks_day=round(n / max(days, 1), 1),
        win=round(100 * (m["fwd10"] > 0).mean(), 1),
        base=round(100 * m["win_1"].mean(), 1),
        exc=round(
            100 * ((m["fwd10"] > 0).groupby(m["date"]).mean()
                   - m.groupby("date")["win_1"].first()).mean(), 1
        ),
        mean=round(100 * m["fwd10"].mean(), 2),
        p5=round(100 * m["fwd10"].quantile(0.05), 2),
    )


def occ5_of(memb):
    """memb: DataFrame[date, board, symbol] → 加 occ5 列 (rolling5 含当日)."""
    out = []
    for (b,), g in memb.groupby(["board"], sort=False):
        g = g.sort_values("date").copy()
        sets = {}
        for d, sub in g.groupby("date"):
            sets[d] = set(sub["symbol"])
        dates = sorted(sets)
        rows = []
        for i, d in enumerate(dates):
            prior = dates[max(0, i - (OCC_WIN - 1)):i]
            occ = {s: 1 + sum(s in sets[p] for p in prior) for s in sets[d]}
            rows.append(pd.DataFrame(
                {"date": d, "board": b, "symbol": list(sets[d]),
                 "occ5": [occ[s] for s in sets[d]]}))
        out.append(pd.concat(rows, ignore_index=True))
    return pd.concat(out, ignore_index=True)


# ── L 臂: legacy 带史 (as-was 145 交易日) ─────────────────────────────────────
def arm_legacy(panel, base, dates_all):
    h = pd.read_parquet(HIST)
    h["date"] = pd.to_datetime(h["date"])
    h["symbol"] = h["symbol"].astype(str)
    m = occ5_of(h[["date", "board", "symbol"]])
    m = m.merge(
        panel[["date", "symbol", "pull"]], on=["date", "symbol"], how="inner"
    )
    # 评估窗: 最近 125 个有历史交易日 (标签缺失自动被 eval 剔除)
    hist_dates = sorted(m["date"].unique())
    win_dates = set(hist_dates[-EVAL_DAYS:])
    m = m[m["date"].isin(win_dates)]
    l0 = m[["date", "symbol"]]
    l1 = m[(m["occ5"] >= OCC_MIN) & (m["pull"] >= PULL_FLOOR)][["date", "symbol"]]
    l1_noDD = m[m["occ5"] >= OCC_MIN][["date", "symbol"]]
    return [
        eval_picks(l0, panel, base, "L0 legacy band occ5>=1"),
        eval_picks(l1_noDD, panel, base, "L1 legacy occ5>=3 noDD"),
        eval_picks(l1, panel, base, "L1 legacy occ5>=3 +DD"),
    ]


# ── PB 臂: parallel 全池 prob 真口径 (preds_raw 24 天) ───────────────────────
def arm_parB(panel, base):
    frames = []
    for fp in sorted(glob.glob(PREDS)):
        d = os.path.basename(fp)
        date = pd.to_datetime(d.split("__")[0].replace("parallel_preds_raw_", ""))
        df = pd.read_csv(fp, dtype={"symbol": str})
        # 只认全池导出日: 早期文件只有部分行 (4KB/18KB), band 成员史被污染
        if len(df) < 1000:
            continue
        df = df.dropna(subset=["pred_prob_10d"])
        df["board"] = df["board"].map(
            {"main": "main", "GEM": "dual", "STAR": "dual"}
        )
        # 全池去重 (多系统行同股): 取该股最大 prob
        df = df.sort_values("pred_prob_10d").groupby(
            ["board", "symbol"], as_index=False
        ).last()
        df["date"] = date
        frames.append(df[["date", "board", "symbol", "pred_prob_10d"]])
    cand = pd.concat(frames, ignore_index=True)
    # 每板每天 prob 降序前 TOP_N (并列 symbol 升序, 同生产)
    cand = cand.sort_values(
        ["date", "board", "pred_prob_10d", "symbol"],
        ascending=[True, True, False, True],
    )
    memb = cand.groupby(["date", "board"], as_index=False).head(TOP_N)
    m = occ5_of(memb[["date", "board", "symbol"]]).merge(
        panel[["date", "symbol", "pull"]], on=["date", "symbol"], how="inner"
    )
    pb0 = m[["date", "symbol"]]
    pb1 = m[(m["occ5"] >= OCC_MIN) & (m["pull"] >= PULL_FLOOR)][["date", "symbol"]]
    pb1n = m[m["occ5"] >= OCC_MIN][["date", "symbol"]]
    return [
        eval_picks(pb0, panel, base, "PB par band occ5>=1 (24d)"),
        eval_picks(pb1n, panel, base, "PB par occ5>=3 noDD (24d)"),
        eval_picks(pb1, panel, base, "PB par occ5>=3 +DD (24d)"),
    ]


# ── PA 臂: parallel 1y top10 选票密度 (score 键, 概念版) ─────────────────────
def arm_parA(panel, base):
    frames = []
    for b in ("main", "dual"):
        fp = os.path.join(RUNDIR, f"stocks_{b}_fusion_full.csv")
        df = pd.read_csv(fp, dtype={"symbol": str})
        df["board"] = b
        frames.append(df[["date", "board", "symbol"]])
    picks = pd.concat(frames, ignore_index=True)
    picks["date"] = pd.to_datetime(picks["date"])
    m = occ5_of(picks).merge(
        panel[["date", "symbol", "pull"]], on=["date", "symbol"], how="inner"
    )
    win_dates = set(sorted(m["date"].unique())[-EVAL_DAYS:])
    m = m[m["date"].isin(win_dates)]
    pa0 = m[["date", "symbol"]]
    pa1 = m[(m["occ5"] >= OCC_MIN) & (m["pull"] >= PULL_FLOOR)][["date", "symbol"]]
    pa1n = m[m["occ5"] >= OCC_MIN][["date", "symbol"]]
    return [
        eval_picks(pa0, panel, base, "PA par top10 picks ALL (125d)"),
        eval_picks(pa1n, panel, base, "PA par picks occ5>=3 noDD"),
        eval_picks(pa1, panel, base, "PA par picks occ5>=3 +DD"),
    ]


def main():
    panel = load_panel()
    base = baseline(panel)
    base.columns = ["mean_1", "win_1"]
    dates_all = None
    rows = []
    rows += arm_legacy(panel, base, dates_all)
    rows += arm_parB(panel, base)
    rows += arm_parA(panel, base)
    rep = pd.DataFrame(rows)
    lines = [
        "密度线 prob 源对比 legacy vs parallel (2026-09-08)",
        "口径: 带每板 prob 前20 / occ5=rolling5含当日>=3 / 回撤闸=距10日高点>=-10% / 派发闸两侧同撤",
        "标签: c2c fwd10; win=fwd10>0; exc_pp=win率-当日全股win率; p5=5%分位(大亏尾)",
        "PB 窗仅 24 交易日 (preds_raw 起存 0806) 且需5日热身 — 样本极小只看方向",
        "PA 是 parallel 1y 重放的 top10 选票 (score 键非 prob) — 概念参考非真口径",
        rep.to_string(index=False),
    ]
    txt = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(rep.to_string(index=False))
    print("[out]", OUT)


if __name__ == "__main__":
    main()
