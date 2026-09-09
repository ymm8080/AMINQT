# -*- coding: utf-8 -*-
"""密度线 parallel 源 历史全量回放 (2026-09-08 用户: "PARALLEL可以跑回测..拿到
足够数据再定要不加进密度模型").

原理 (读码定案):
  parallel serving 的 pred_prob_10d = 横截面 Platt(score→P(mfe≥目标)) — 单调,
  带内 (每板 prob 前20) 排名 ≡ score 排名; score = max(狙击池分, 融合池分),
  池分 = 特征列逐日截面分位等权合成 (pool_score, 纯特征无拟合无前视, 逐日独立)
  → 历史带成员可从 data/_diag_stage_{board}_3y.parquet 精确重建, 不用等
  preds_raw 日积月累。
  (_shortlist_t5_t10._anchor_frame 同公式同数据, 注释明言 "就是模型当日入选集"。)

臂 (与 0908 对比脚本同一 harness):
  L1  legacy occ5>=3 + 回撤闸   (密度带史重算, 同窗参照)
  PBH parallel 历史重建带 occ5>=3 + 回撤闸  (主答案, 125d 真样本)
  另报 band occ5>=1 两源 + 重建 vs preds_raw 真带的重叠率 (校验重建保真度)。

标签: panel close c2c fwd10; excess = 赢率 − 当日全股赢率 (pp); 派发闸两侧同撤。
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)

from app.pipeline_parallel.config import FUSION, SNIPER  # noqa: E402
from app.pipeline_parallel.scoring import pool_score  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
HIST = os.path.join(ROOT, "data", "prob10_density_history.parquet")
STAGE = os.path.join(ROOT, "data", "_diag_stage_{board}_3y.parquet")
PREDS = r"D:\AMINQT\Daily Operation\STOCK LIST\parallel_preds_raw_*.csv"
OUT = os.path.splitext(__file__)[0] + "_report.txt"

TOP_N, OCC_MIN, PULL_FLOOR, OCC_WIN = 20, 3, -0.10, 5
EVAL_DAYS = 125
CAL_DAYS = 340  # _diag_stage 日期下推 (日历日), 覆盖 125 评估日+occ 热身+缓冲


def load_panel():
    tb = pq.read_table(PANEL, columns=["date", "symbol", "close"])
    df = tb.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)["close"]
    df["fwd10"] = g.shift(-10) / df["close"] - 1.0
    df["hi10"] = g.transform(lambda s: s.rolling(10, min_periods=1).max())
    df["pull"] = df["close"] / df["hi10"] - 1.0
    return df


def baseline(panel):
    d = panel.dropna(subset=["fwd10"])
    b = d.groupby("date")["fwd10"].agg(["mean", lambda s: (s > 0).mean()])
    b.columns = ["mean_1", "win_1"]
    return b


def eval_picks(picks, panel, base, tag):
    m = picks.merge(
        panel[["date", "symbol", "fwd10"]], on=["date", "symbol"], how="inner"
    ).dropna(subset=["fwd10"])
    if m.empty:
        return dict(arm=tag, days=0, picks_day=0.0, win=np.nan, base=np.nan,
                    exc=np.nan, mean=np.nan, p5=np.nan)
    m = m.merge(base, left_on="date", right_index=True, how="left")
    days = m["date"].nunique()
    return dict(
        arm=tag,
        days=days,
        picks_day=round(len(m) / max(days, 1), 1),
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
    out = []
    for (b,), g in memb.groupby(["board"], sort=False):
        sets = {d: set(sub["symbol"]) for d, sub in g.groupby("date")}
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


def rebuild_parallel_band():
    """_diag_stage → score=max(sniper,fusion) → 每板每日前20 (date/board/symbol)."""
    pool_cols = [c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"]
    frames = []
    for board in ("main", "dual"):
        fp = STAGE.format(board=board)
        t = pq.read_table(
            fp, columns=["symbol", "date"] + pool_cols,
            filters=[("date", ">=", pd.Timestamp.today() - pd.Timedelta(days=CAL_DAYS))],
        ).to_pandas()
        t["symbol"] = t["symbol"].astype(str).str.zfill(6)
        t["date"] = pd.to_datetime(t["date"])
        sn = pool_score(t, SNIPER.pool)
        fu = pool_score(t, FUSION.pool)
        t["score"] = np.maximum(sn.values, fu.values)
        t = t.dropna(subset=["score"])
        t = t.sort_values(["date", "score", "symbol"],
                          ascending=[True, False, True])
        top = t.groupby("date", sort=False).head(TOP_N)
        top = top.assign(board=board)
        frames.append(top[["date", "board", "symbol", "score"]])
        print(f"[stage] {board}: {t['date'].nunique()} 交易日 带重建完成")
    return pd.concat(frames, ignore_index=True)


def validate_against_preds_raw(band):
    """重建带 vs preds_raw 真带 (max prob per symbol, 每板前20) 重叠率."""
    overlaps = []
    for fp in sorted(glob.glob(PREDS)):
        d = os.path.basename(fp)
        date = pd.to_datetime(d.split("__")[0].replace("parallel_preds_raw_", ""))
        df = pd.read_csv(fp, dtype={"symbol": str})
        if len(df) < 1000:
            continue
        df = df.dropna(subset=["pred_prob_10d"])
        df["symbol"] = df["symbol"].str.zfill(6)
        df["board"] = df["board"].map({"main": "main", "GEM": "dual", "STAR": "dual"})
        df = df.sort_values("pred_prob_10d").groupby(
            ["board", "symbol"], as_index=False).last()
        df = df.sort_values(["board", "pred_prob_10d", "symbol"],
                            ascending=[True, False, True])
        real = set(map(tuple, df.groupby(["board", "date"] if "date" in df else ["board"])
                       .head(TOP_N)[["board", "symbol"]].values))
        rec = band[band["date"] == date]
        if rec.empty:
            continue
        rec_set = set(map(tuple, rec[["board", "symbol"]].values))
        inter = len(real & rec_set)
        overlaps.append((date.strftime("%m-%d"), inter / max(len(real), 1)))
    return overlaps


def main():
    print("[1/4] panel 标签...")
    panel = load_panel()
    base = baseline(panel)

    print("[2/4] parallel 历史带重建 (截面分位打分, PIT)...")
    band = rebuild_parallel_band()

    print("[3/4] 重建保真度校验 vs preds_raw 真带...")
    ov = validate_against_preds_raw(band)
    if ov:
        mean_ov = np.mean([o for _, o in ov])
        ov_line = "; ".join(f"{d}:{o:.0%}" for d, o in ov[-8:])
        print(f"[valid] {len(ov)} 天可校验, 平均带重叠率 {mean_ov:.0%} | 近8天 {ov_line}")
    else:
        mean_ov, ov_line = np.nan, "无可校验日"

    # 同窗对齐: 取两源共同日期的末 EVAL_DAYS 个
    print("[4/4] 同窗对比评估...")
    h = pd.read_parquet(HIST)
    h["date"] = pd.to_datetime(h["date"])
    h["symbol"] = h["symbol"].astype(str).str.zfill(6)
    common = sorted(set(h["date"].unique()) & set(band["date"].unique()))
    win_dates = set(common[-EVAL_DAYS:])

    # occ 热身需带史更早日期 → 用全史算 occ, 再切评估窗
    mh = occ5_of(h[["date", "board", "symbol"]])
    mh = mh[mh["date"].isin(win_dates)]
    mh = mh.merge(panel[["date", "symbol", "pull"]], on=["date", "symbol"], how="inner")
    l1 = mh[(mh["occ5"] >= OCC_MIN) & (mh["pull"] >= PULL_FLOOR)][["date", "symbol"]]
    l0 = mh[mh["date"].isin(win_dates)][["date", "symbol"]]

    mp_full = occ5_of(band[["date", "board", "symbol"]])
    mp = mp_full[mp_full["date"].isin(win_dates)]
    mp = mp.merge(panel[["date", "symbol", "pull"]], on=["date", "symbol"], how="inner")
    pbh1 = mp[(mp["occ5"] >= OCC_MIN) & (mp["pull"] >= PULL_FLOOR)][["date", "symbol"]]
    pbh0 = mp[["date", "symbol"]]

    rows = [
        eval_picks(l0, panel, base, "L0 legacy band occ5>=1"),
        eval_picks(l1, panel, base, "L1 legacy occ5>=3 +DD"),
        eval_picks(pbh0, panel, base, "PBH par重建 band occ5>=1"),
        eval_picks(pbh1, panel, base, "PBH par重建 occ5>=3 +DD"),
    ]
    rep = pd.DataFrame(rows)
    lines = [
        "密度线 parallel 源 历史全量回放 (2026-09-08)",
        "原理: parallel prob=Platt(score) 单调 → 带排名≡score排名; score=逐日截面分位",
        "      (纯特征 PIT) → 从 _diag_stage 历史重建带, 不等 preds_raw 积累",
        f"校验: 重建带 vs preds_raw 真带 平均重叠率 {mean_ov:.0%} ({len(ov)} 天) | {ov_line}"
        if ov else "校验: 无可校验日",
        "口径: 每板前20 / occ5=rolling5含当日>=3 / 回撤闸>=-10% / 派发闸两侧同撤",
        "标签: c2c fwd10; exc_pp=赢率-当日全股赢率; 同窗=两源共同日期末125交易日",
        rep.to_string(index=False),
    ]
    txt = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(rep.to_string(index=False))
    print("[out]", OUT)


if __name__ == "__main__":
    main()
