"""_diag_fc_sign_family_ab.py — 预告符号特征族 A/B (125d 全池 walk-forward, 2026-09-04).

动机 (fc_sign_ic_gate_v2_20260903 PASS, 公告族唯一重开入口): 利空预告后 3-10d
利空出尽 (+9.45%/89% 预告季窗; v4 全历史 +4.56%/63.5%), days_since_fc_neg 达
strong 级 IC (-0.031). 本 A/B 检验: 给概率头加「预告新鲜度」特征族能否提升
TOP10 质量. 影子协议 — 生产面板/模型零改动, 过闸才进生产 PR.

特征族 (2 列, PIT asof join 自 forecast 缓存, 行日期-事件日 ≥0 无前瞻;
利好端 days_since_fc_pos 闸判 dead, 用户定案利好特征勿做):
  days_since_fc      距最近预告 (不分符号) 日历天数 — 闸 weak (-0.024) ✓
  days_since_fc_neg  距最近利空预告日历天数 — 闸 strong (-0.031) ✓

协议 (对齐 _diag_vp_family_ab.py, 双臂共享除 prob 特征外全部环节):
  score = max(狙击, 融合) 截面分位; mag = calibrate_mag10d walk-forward (双臂同);
  prob  = LGBM(label_mfe_10d_net>=0.06) 每 21 交易日扩窗重拟合;
  rank_blend = mag × prob; TOP10 per board/day.
  ARM BASE = 现有特征空间 (fc 列剔除); ARM FAM = 现有 + 2 列预告族.

评估指标:
  net3/net10 TOP10 均值 (T+1 买入 c2c, 扣 0.20% 成本), hit3, 真赢家重叠,
  半窗拆分 — 闸同 vp A/B: Δnet3 双半窗同向为正 (至少一板, 另一板不负).
  利空出尽捕获 (诊断, 非闸): 评估窗 TOP10 中处于利空后 3-10 日历日窗口的
  均只数/日 — fam 机制所在, 应不降.

WORM: DATA OTHERS/diag/fc_sign_family_ab_<ts>.parquet + .json
用法:
  python scripts/_diag_fc_sign_family_ab.py                    # slice 420, eval 125
  python scripts/_diag_fc_sign_family_ab.py --slice 260 --eval 10   # 冒烟
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from app.pipeline_parallel.backtest import add_mfe_labels, tradability_gate
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, PANEL, SNIPER
from app.pipeline_parallel.prob_head import LGB_PARAMS, feature_cols
from app.pipeline_parallel.scoring import pool_score
from config.settings import PANEL_V3_PATH, data_others_path
from scripts._reclassify_all_features import _finalize_slice

COST = 0.0020
PROB_TARGET = 0.06
PROB_REFIT_EVERY = 21
PROB_REFIT_FROM = 40
PROB_MIN_ROWS = 500
TOP_N = 10
WINNER_T = 0.05
MAX_LAG = 11  # net10 卖出日 = T+11, 评估日须距日历尾 ≥11
FC_COLS = ["days_since_fc", "days_since_fc_neg"]
FC_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "supply_cache",
    "alt_data",
    "forecast",
    "*.parquet",
)
POS = {"预增", "略增", "扭亏", "续盈"}
NEG = {"预减", "略减", "首亏", "续亏", "预亏", "增亏"}
NEG_WIN = (3, 10)  # 利空出尽窗 (日历日, 与闸 sanity 桶同口径)


def load_fc_events(fc_glob: str = FC_GLOB):
    """forecast 缓存 → (全部事件, 利空事件) 两张 (symbol, evt_date) 去重表."""
    fc = pd.concat(
        [pd.read_parquet(f) for f in sorted(glob.glob(fc_glob))], ignore_index=True
    )
    acol = "ann_date" if "ann_date" in fc.columns else "announce_date"
    tcol = "type" if "type" in fc.columns else "forecast_type"
    fc["evt_date"] = pd.to_datetime(fc[acol], errors="coerce")
    fc = fc[fc["evt_date"].notna() & (fc["evt_date"] >= "2022-01-01")]
    grp = np.where(fc[tcol].isin(POS), "POS", np.where(fc[tcol].isin(NEG), "NEG", "O"))
    ev_all = fc[["symbol", "evt_date"]].copy()
    ev_neg = fc.loc[pd.Series(grp, index=fc.index) == "NEG", ["symbol", "evt_date"]]
    return ev_all, ev_neg


def asof_days_since(panel: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """每行距最近已知事件日天数 (日历日); 无记录 → NaN. 行序/index 不变.

    PIT: merge_asof backward 保证每股只用 announce_date ≤ 行日期 的事件.
    """
    dts = panel["date"].to_numpy()
    perm = np.argsort(dts, kind="stable")
    left = pd.DataFrame(
        {
            "symbol": panel["symbol"].astype(str).to_numpy()[perm],
            "date": dts[perm],
        }
    )
    ev = events.copy()
    ev["symbol"] = ev["symbol"].astype(str)
    ev = ev.drop_duplicates(["symbol", "evt_date"]).sort_values(
        "evt_date", kind="stable"
    )
    m = pd.merge_asof(
        left,
        ev,
        by="symbol",
        left_on="date",
        right_on="evt_date",
        direction="backward",
    )
    days = (m["date"] - m["evt_date"]).dt.days.astype("float64")
    out = np.full(len(dts), np.nan)
    out[perm] = days.to_numpy()
    return pd.Series(out, index=panel.index)


def add_fc_family(df: pd.DataFrame, ev_all, ev_neg) -> pd.DataFrame:
    df["days_since_fc"] = asof_days_since(df, ev_all).to_numpy()
    df["days_since_fc_neg"] = asof_days_since(df, ev_neg).to_numpy()
    return df


def _lgbm_walkforward(
    X: np.ndarray, y: np.ndarray, fit_ok: np.ndarray, dv: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """扩窗 LGBM: refit 日用 ≤当日已实现行拟合, 应用到 [refit, 下个 refit)."""
    out = np.full(len(dv), np.nan, dtype=float)
    model = None
    for i, d in enumerate(dates):
        if i >= PROB_REFIT_FROM and i % PROB_REFIT_EVERY == 0:
            m = (dv <= d) & fit_ok
            if int(m.sum()) >= PROB_MIN_ROWS:
                model = LGBMClassifier(**LGB_PARAMS)
                model.fit(X[m], y[m])
        if model is None:
            continue
        rows = np.nonzero(dv == d)[0]
        if len(rows):
            out[rows] = model.predict_proba(X[rows])[:, 1]
    return out


def _pivots(panel: pd.DataFrame):
    cal = np.sort(
        np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy())
    )
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    px = (
        panel.assign(dt=dt)
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.to_datetime(cal))
        .ffill(axis=1)
    )
    px.index = px.index.astype(str).str.zfill(6)
    return px, cal


def _panel_pivots(cutoff):
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq"],
        filters=[("date", ">=", cutoff)],
    )
    return _pivots(panel)


def _realized_net(
    pm: np.ndarray, sym_rows: np.ndarray, j_cols: np.ndarray, horizon: int
) -> np.ndarray:
    """向量化 T+1 买入、T+horizon 卖出的 c2c 净值 (含成本)."""
    buy_j = j_cols + 1
    sell_j = j_cols + horizon + 1
    out = np.full(len(j_cols), np.nan, dtype=float)
    ok = (sym_rows >= 0) & (sell_j < pm.shape[1])
    if ok.any():
        pb = pm[sym_rows[ok], buy_j[ok]]
        ps = pm[sym_rows[ok], sell_j[ok]]
        net = ps / pb - 1.0 - COST
        net[~(np.isfinite(net) & (pb > 0))] = np.nan
        out[ok] = net
    return out


def _top10_indices(blend: np.ndarray, dv: np.ndarray, eval_days: np.ndarray) -> list:
    """每评估日取 blend 前 TOP_N 行号 (并列按行序稳定)."""
    picks = []
    for d in eval_days:
        rows = np.nonzero(dv == d)[0]
        b = blend[rows]
        b = np.where(np.isfinite(b), b, -np.inf)
        order = np.argsort(-b, kind="stable")[:TOP_N]
        picks.append(rows[order])
    return picks


def _daily_metrics(
    picks: list, eval_days, sym, net3, net10, fc_neg, winners
) -> pd.DataFrame:
    rows = []
    for d, idx in zip(eval_days, picks):
        n3 = net3[idx]
        n3 = n3[np.isfinite(n3)]
        in_win = int(sum(1 for s in sym[idx] if s in winners))
        fcn = fc_neg[idx]
        rows.append(
            {
                "date": pd.Timestamp(d),
                "n": int(len(idx)),
                "net3": float(n3.mean()) if len(n3) else np.nan,
                "net10": float(net10[idx][np.isfinite(net10[idx])].mean()),
                "hit3": float((n3 > 0).mean()) if len(n3) else np.nan,
                "winner_overlap": in_win,
                "fc_neg_capture": int(
                    ((fcn >= NEG_WIN[0]) & (fcn <= NEG_WIN[1])).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def process_board(
    board: str, cutoff: pd.Timestamp, eval_n: int, px, cal, ev_all, ev_neg, t0: float
) -> pd.DataFrame:
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    print(f"[{board}] read {ckpt}", flush=True)
    df = pd.read_parquet(ckpt, filters=[("date", ">=", cutoff)])
    df = _finalize_slice(df)
    df = add_mfe_labels(df, horizons=(10,), already_sorted=True)
    df, gate = tradability_gate(df)
    print(
        f"[{board}] rows {len(df):,} gate -{gate['removed_rows']:,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    df["board"] = board
    df = add_fc_family(df, ev_all, ev_neg)
    df["_rid"] = np.arange(len(df))
    base_cols = [c for c in feature_cols(df) if c not in set(FC_COLS) and c != "_rid"]
    print(f"[{board}] feat base={len(base_cols)} fam=+{len(FC_COLS)}", flush=True)

    score = np.maximum(
        pool_score(df, SNIPER.pool).to_numpy(dtype=float),
        pool_score(df, FUSION.pool).to_numpy(dtype=float),
    )
    scored = df[
        ["_rid", "symbol", "date", "board", "label_pm_10d_net", "label_mfe_10d_net"]
    ].copy()
    scored["score"] = score
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    scored["symbol"] = scored["symbol"].astype(str).str.zfill(6)
    scored = scored.dropna(subset=["score"]).reset_index(drop=True)
    mag = calibrate_mag10d(scored, score_col="score", target_col="label_pm_10d_net")
    scored = scored.merge(
        mag[["symbol", "date", "mag"]], on=["symbol", "date"], how="inner"
    ).reset_index(drop=True)
    del mag, score
    gc.collect()
    print(f"[{board}] scored {len(scored):,}r ({time.time() - t0:.0f}s)", flush=True)

    sub = df.set_index("_rid").loc[scored["_rid"].to_numpy()].reset_index(drop=True)
    del df
    gc.collect()
    if len(sub) != len(scored):
        raise RuntimeError(f"[{board}] _rid 对齐失败 {len(sub)} vs {len(scored)}")
    X_base = sub[base_cols].to_numpy(dtype="float32")
    X_fc = sub[FC_COLS].to_numpy(dtype="float32")
    X_fam = np.hstack([X_base, X_fc])
    fc_neg_days = X_fc[:, 1].astype(float)
    syms = scored["symbol"].to_numpy()
    dv_days = scored["date"].to_numpy()

    pm = px.to_numpy(dtype="float64")
    sym_rows = px.index.get_indexer(pd.Index(syms))
    j_cols = np.searchsorted(cal, dv_days)
    net3 = _realized_net(pm, sym_rows, j_cols, 3)
    net10 = _realized_net(pm, sym_rows, j_cols, 10)
    del sub, pm
    gc.collect()

    y = (scored["label_mfe_10d_net"].to_numpy() >= PROB_TARGET).astype(int)
    fit_ok = np.isfinite(scored["label_mfe_10d_net"].to_numpy())
    dates = np.sort(pd.unique(scored["date"].to_numpy()))
    dv = scored["date"].to_numpy()
    print(
        f"[{board}] X_base {X_base.shape} fit_rows={int(fit_ok.sum()):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    prob_base = _lgbm_walkforward(X_base, y, fit_ok, dv, dates)
    print(f"[{board}] prob_base done ({time.time() - t0:.0f}s)", flush=True)
    prob_fam = _lgbm_walkforward(X_fam, y, fit_ok, dv, dates)
    print(f"[{board}] prob_fam done ({time.time() - t0:.0f}s)", flush=True)
    del X_base, X_fam
    gc.collect()

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    day_dates = sorted(pd.unique(scored["date"]))
    eval_days = [
        d for d in day_dates if d in i_of and i_of[d] + MAX_LAG < len(all_cal)
    ][-eval_n:]
    print(
        f"[{board}] eval={len(eval_days)} "
        f"({pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()})",
        flush=True,
    )

    mag_v = scored["mag"].to_numpy(dtype=float)
    blend = {"base": mag_v * prob_base, "fam": mag_v * prob_fam}
    del prob_base, prob_fam
    gc.collect()

    eval_d64 = pd.to_datetime(pd.Series(eval_days)).to_numpy()
    eval_mask = np.isin(dv, eval_d64)
    winners = set(syms[eval_mask & np.isfinite(net3) & (net3 >= WINNER_T)])
    daily = []
    for arm in ("base", "fam"):
        picks = _top10_indices(blend[arm], dv, eval_d64)
        m = _daily_metrics(picks, eval_days, syms, net3, net10, fc_neg_days, winners)
        m["arm"] = arm
        m["board"] = board
        daily.append(m)
        del blend[arm]
        gc.collect()
    return pd.concat(daily, ignore_index=True)


def _half_deltas(daily: pd.DataFrame) -> dict:
    """配对日 Δ(fam−base), 全窗 + 双半窗 (按评估日中位切分)."""
    out = {}
    for board, g in daily.groupby("board"):
        p = g.pivot(index="date", columns="arm", values="net3").dropna()
        d = p["fam"] - p["base"]
        h = len(d) // 2
        out[board] = {
            "days_paired": int(len(d)),
            "d_net3": float(d.mean()),
            "d_net3_h1": float(d.iloc[:h].mean()),
            "d_net3_h2": float(d.iloc[h:].mean()),
            "d_net10": float(
                (
                    g.pivot(index="date", columns="arm", values="net10")
                    .dropna()
                    .pipe(lambda x: x["fam"] - x["base"])
                ).mean()
            ),
            "hit3_base": float(g[g["arm"] == "base"]["hit3"].mean()),
            "hit3_fam": float(g[g["arm"] == "fam"]["hit3"].mean()),
            "win_base": float(g[g["arm"] == "base"]["winner_overlap"].mean()),
            "win_fam": float(g[g["arm"] == "fam"]["winner_overlap"].mean()),
            "cap_base": float(g[g["arm"] == "base"]["fc_neg_capture"].mean()),
            "cap_fam": float(g[g["arm"] == "fam"]["fc_neg_capture"].mean()),
        }
    return out


def _verdict(deltas: dict) -> dict:
    """闸: Δnet3 双半窗同向为正 (至少一板, 另一板不负) + fc 捕获不降."""
    board_pass = {}
    for b, d in deltas.items():
        board_pass[b] = d["d_net3_h1"] > 0 and d["d_net3_h2"] > 0
    pos_boards = [b for b, ok in board_pass.items() if ok]
    neg_boards = [
        b for b, d in deltas.items() if d["d_net3_h1"] < 0 and d["d_net3_h2"] < 0
    ]
    cap_ok = all(deltas[b]["cap_fam"] >= deltas[b]["cap_base"] - 1e-9 for b in deltas)
    gate = bool(pos_boards) and not neg_boards and cap_ok
    return {
        "gate": gate,
        "board_pass": board_pass,
        "neg_boards": neg_boards,
        "capture_non_degradation": cap_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420)
    ap.add_argument("--eval", type=int, default=125)
    args = ap.parse_args()
    t0 = time.time()

    dts = pd.read_parquet(PANEL.main_checkpoint, columns=["date"])["date"].unique()
    cutoff = np.sort(pd.to_datetime(pd.Series(dts)))[-args.slice]
    print(f"[slice] {args.slice}d cutoff={pd.Timestamp(cutoff).date()}", flush=True)

    px, cal = _panel_pivots(cutoff)
    print(
        f"[pivot] {len(px)} syms x {len(cal)} days ({time.time() - t0:.0f}s)",
        flush=True,
    )
    ev_all, ev_neg = load_fc_events()
    print(
        f"[fc] events all={len(ev_all):,} neg={len(ev_neg):,} "
        f"({pd.Timestamp(ev_all['evt_date'].min()).date()}.."
        f"{pd.Timestamp(ev_all['evt_date'].max()).date()})",
        flush=True,
    )

    daily = []
    for board in ("main", "dual"):
        d = process_board(board, cutoff, args.eval, px, cal, ev_all, ev_neg, t0)
        daily.append(d)
    daily = pd.concat(daily, ignore_index=True)

    deltas = _half_deltas(daily)
    verdict = _verdict(deltas)
    print("\n== Δ(fam−base) TOP10 ==")
    for b, d in deltas.items():
        print(
            f"[{b}] days={d['days_paired']} d_net3={d['d_net3'] * 100:+.3f}pp "
            f"(h1 {d['d_net3_h1'] * 100:+.3f} / h2 {d['d_net3_h2'] * 100:+.3f}) "
            f"d_net10={d['d_net10'] * 100:+.3f}pp "
            f"hit3 {d['hit3_base']:.3f}->{d['hit3_fam']:.3f} "
            f"win {d['win_base']:.2f}->{d['win_fam']:.2f} "
            f"cap {d['cap_base']:.2f}->{d['cap_fam']:.2f}"
        )
    print(
        f"\n[gate] {'PASS' if verdict['gate'] else 'FAIL'} "
        f"board_pass={verdict['board_pass']} neg={verdict['neg_boards']} "
        f"cap_ok={verdict['capture_non_degradation']}"
    )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    daily.to_parquet(out_dir / f"fc_sign_family_ab_{ts}.parquet", index=False)
    report = {
        "ts": ts,
        "slice": args.slice,
        "eval": args.eval,
        "fc_cols": FC_COLS,
        "deltas": deltas,
        "verdict": verdict,
    }
    (out_dir / f"fc_sign_family_ab_{ts}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        f"[saved] {out_dir}\\fc_sign_family_ab_{ts}.json + .parquet "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
