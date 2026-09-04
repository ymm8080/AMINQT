"""_diag_fakeleg_event.py — 真假腿判别力事件研究 (全特征族×事件AUC, 2026-09-03).

动机 (用户 2026-09-02): 002881 探针发现模型看得见上涨腿但分不清真假腿
(假腿=缩量上涨, 真腿=持续放量穿越基准). 问题: 其它特征族能否在腿出现时
(及前 1-2 日) 判别这条腿是真 (延续) 还是假 (回落)?

事件: 上涨腿检测日 T = close_hfq 3 日涨幅 >= +8% (main) / +15% (dual).
标签: 腿后 3 日净回报 fwd3 = px[T+3]/px[T+1] - 1 - COST
  REAL  fwd3 >= +2%   (延续)
  FAKE  fwd3 <= -2%   (回落)
  中间地带剔除.

特征: 事件日 T / T-1 / T-2 的特征族复合分 (= 族内各列截面 pct-rank 均值):
  chip(筹码 cyq_) / holder(股东 sh_) / block(bt_ 大宗) / margin(两融) /
  fina(财务) / sector(行业 sw_) / tech(技术 bias/amplitude/volume_ratio/turnover) /
  vp(量价配合族, 判死族在此作为参照)
指标: 每族每偏移 AUC (real=1), 前后半窗拆分 (稳定性), precision@0.8
  (复合分 >= 0.8 的事件中 real 占比). AUC > 0.5 = 该族看多真腿.

双板同跑 (main/dual 事件各一份; 阈值相同, 板内截面).
WORM: DATA OTHERS/diag/fakeleg_event_<ts>.parquet + .json
用法:
  python scripts/_diag_fakeleg_event.py                # 近 550 交易日
  python scripts/_diag_fakeleg_event.py --days 120     # 冒烟
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

COST = 0.0020
LEG_THR = {"main": 0.08, "dual": 0.15}
REAL_THR = 0.02
FAKE_THR = -0.02
OFFSETS = (0, 1, 2)  # T, T-1, T-2
MIN_VALID = 3

# 面板基本面列是裸名 (无 fina_ 前缀); margin_ 前缀不存在, 两融列仅两条
FUND_COLS = (
    "pe_ttm",
    "dv_ratio",
    "dv_ttm",
    "roe",
    "roe_deducted",
    "rev_yoy",
    "asset_turnover",
    "ocf_to_or",
    "eps_yoy",
    "profit_yoy",
    "ocfps",
    "revenue_ps",
    "eps",
    "dt_eps",
    "roe_yoy",
    "q_roe",
    "q_ocf_to_sales",
)
FAMILY_PREFIXES = {
    # 筹码列在面板是裸名 (无 cyq_ 前缀)
    "chip": (
        "pct_90_high",
        "pct_90_con",
        "winner_ratio",
        "cost_50pct",
        "cost_95pct",
        "peak_price",
        "chip_entropy",
        "chip_skew_dist",
        "peak_roc_5d",
        "peak_roc_20d",
        "cost_bias",
        "conc_trend_20d",
        "conc_90_industry_rank",
        "chip_gini",
        "resistance_dist",
    ),
    "holder": ("sh_",),
    "block": ("bt_",),
    "margin": ("margin_balance", "margin_buy_amt"),
    "fund": FUND_COLS,
    "sector": ("sw_",),
    "tech": ("bias_", "amplitude", "volume_ratio", "turnover"),
    "pv": ("pct_chg", "volume", "amount", "open", "high", "low", "close"),
}


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


def _panel_cols(prefixes: tuple[str, ...]) -> list[str]:
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(str(PANEL_V3_PATH)).schema_arrow
    out = []
    for f in schema:
        if f.name in ("symbol", "date", "announce_date") or f.name.startswith("label_"):
            continue
        if any(f.name == p or f.name.startswith(p) for p in prefixes):
            if pa_numeric(f.type):
                out.append(f.name)
    return out


def pa_numeric(t) -> bool:
    import pyarrow as pa

    return (
        pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_boolean(t)
        or pa.types.is_decimal(t)
    )


def _build_leg_events(px: pd.DataFrame, w0: int, last_i: int, thr: float):
    """上涨腿事件 (sym_str, T_idx, label real=1/fake=0)."""
    lo, hi = w0 + 3, last_i - 4
    ret3 = px / px.shift(3) - 1.0
    fwd3 = px.shift(-3) / px.shift(-1) - 1.0 - COST
    syms, Ts, labels = [], [], []
    for j in range(lo, hi + 1):
        leg = ret3.iloc[:, j]
        f = fwd3.iloc[:, j]
        leg_ok = leg[leg >= thr].index
        for s in leg_ok:
            fv = f.get(s, np.nan)
            if not np.isfinite(fv):
                continue
            if fv >= REAL_THR:
                syms.append(s)
                Ts.append(j)
                labels.append(1)
            elif fv <= FAKE_THR:
                syms.append(s)
                Ts.append(j)
                labels.append(0)
    return (
        np.asarray(syms, dtype=object),
        np.asarray(Ts, dtype=int),
        np.asarray(labels, dtype=int),
    )


def _rank_matrix(panel: pd.DataFrame, col: str):
    df = panel[["symbol", "date", col]].copy()
    dt = pd.to_datetime(df["date"]).dt.normalize()
    df["rk"] = df.groupby(dt)[col].rank(pct=True)
    mat = df.assign(dt=dt).pivot(index="symbol", columns="dt", values="rk").sort_index()
    sym_order = mat.index.astype(str).str.zfill(6).to_numpy()
    return mat, sym_order


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    ok = np.isfinite(scores)
    s, y = scores[ok], labels[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 并列给均秩
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[y == 1].sum()
    return float((rp - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=550)
    args = ap.parse_args()
    t0 = time.time()

    dts = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].unique()
    cal_all = np.sort(pd.to_datetime(pd.Series(dts)).unique())
    cutoff = cal_all[-args.days]
    print(f"[cutoff] {pd.Timestamp(cutoff).date()} days={args.days}", flush=True)

    fam_cols = {fam: _panel_cols(pre) for fam, pre in FAMILY_PREFIXES.items()}
    all_cols = sorted({c for cs in fam_cols.values() for c in cs})
    print(
        f"[cols] {sum(len(v) for v in fam_cols.values())} in {len(fam_cols)} families",
        flush=True,
    )

    read_cols = ["symbol", "date"]
    for c in ["close_hfq", "volume", *all_cols]:
        if c not in read_cols:
            read_cols.append(c)
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=read_cols,
        filters=[("date", ">=", cutoff)],
    )
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    px, cal = _pivots(panel[["symbol", "date", "close_hfq"]])
    print(
        f"[pivot] symbols={len(px)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # vp 族 (判死族参照) 在面板上重算
    from scripts._diag_vp_family_ab import VP_COLS, add_vp_family

    panel = add_vp_family(panel)
    fam_cols["vp"] = list(VP_COLS)
    all_cols = sorted(set(all_cols) | set(VP_COLS))

    # ---- OHLC 微观结构衍生族 (09-03 补齐: 原始 OHLCV 已在 pv/tech, 此处为形态衍生, 与⑤同口径) ----
    dt_all = pd.to_datetime(panel["date"]).dt.normalize()

    def _raw_pivot(col: str, idx) -> np.ndarray:
        w = (
            panel.assign(d=dt_all)  # noqa: F821 (归档研究: OHLC 扩展块, panel 在调用点才定义)
            .pivot_table(index="symbol", columns="d", values=col, aggfunc="last")
            .sort_index()
            .reindex(index=idx, columns=pd.DatetimeIndex(cal))
        )
        return w.to_numpy(dtype="float64")

    _close_w = (
        panel.assign(d=dt_all)
        .pivot_table(index="symbol", columns="d", values="close", aggfunc="last")
        .sort_index()
    )
    _sym_idx = _close_w.index.astype(str).str.zfill(6)
    O = _raw_pivot("open", _sym_idx)  # noqa: E741 (OHLC 矩阵惯例单字母)
    H = _raw_pivot("high", _sym_idx)
    L = _raw_pivot("low", _sym_idx)
    C = _raw_pivot("close", _sym_idx)
    V = _raw_pivot("volume", _sym_idx)
    T_ = _raw_pivot("turnover_rate", _sym_idx)
    del _close_w
    prevC = np.roll(C, 1, axis=1)
    prevC[:, 0] = np.nan
    rng = H - L
    with np.errstate(invalid="ignore", divide="ignore"):
        ohlc_raw = {
            "close_pos": np.where(rng > 0, (C - L) / rng, np.nan),
            "upper_wick": np.where(rng > 0, (H - np.maximum(O, C)) / rng, np.nan),
            "lower_wick": np.where(rng > 0, (np.minimum(O, C) - L) / rng, np.nan),
            "gap": O / prevC - 1.0,
            "intraday": C / O - 1.0,
            "spike_rev": C / H - 1.0,  # 收盘距最高 (冲高回落)
            "amp_pct": (H - L) / prevC,
        }
    del O, H, L
    gc.collect()
    for _name, _base, _roll in (("vol_x5", V, 5), ("turn_x5", T_, 5)):
        _b = pd.DataFrame(_base)
        ohlc_raw[_name] = (_b / _b.T.rolling(_roll, min_periods=3).mean().T).to_numpy()
        del _b
    del V, T_, C
    gc.collect()
    sym_order_ohlc = _sym_idx.to_numpy()
    ohlc_mats: dict[str, tuple] = {}
    for _name, _arr in ohlc_raw.items():
        _rk = pd.DataFrame(_arr).rank(axis=0, pct=True).to_numpy("float32")
        ohlc_mats[f"ohlc_{_name}"] = (_rk, sym_order_ohlc)
        del _arr
    del ohlc_raw
    gc.collect()
    fam_cols["ohlc"] = list(ohlc_mats.keys())
    all_cols = sorted(set(all_cols) | set(ohlc_mats.keys()))
    print(f"[ohlc] 衍生族 {len(ohlc_mats)} 列已注册", flush=True)

    boards_events = {}
    for board in ("main", "dual"):
        boards_events[board] = _build_leg_events(
            px, w0=3, last_i=len(cal) - 5, thr=LEG_THR[board]
        )
        print(
            f"[events] {board} thr={LEG_THR[board]}: {len(boards_events[board][0]):,}",
            flush=True,
        )

    date_to_idx = {d: i for i, d in enumerate(cal)}
    col_vals: dict[str, dict[int, np.ndarray]] = {}
    for k, col in enumerate(all_cols):
        if col in ohlc_mats:
            vals, sym_order = ohlc_mats[col]
        else:
            mat, sym_order = _rank_matrix(panel, col)
            vals = mat.to_numpy(dtype="float32")
            del mat
        sym_to_idx = {s: i for i, s in enumerate(sym_order)}
        gc.collect()
        col_vals[col] = {}
        for off in OFFSETS:
            per_board = {}
            for board, (ev_sy, ev_T, _) in boards_events.items():
                sidx = np.array([sym_to_idx.get(s, -1) for s in ev_sy])
                didx = np.array(
                    [date_to_idx.get(pd.Timestamp(cal[t - off]), -1) for t in ev_T]
                )
                ok = (sidx >= 0) & (didx >= 0)
                arr = np.full(len(ev_sy), np.nan, dtype=float)
                arr[ok] = vals[sidx[ok], didx[ok]]
                per_board[board] = arr
            col_vals[col][off] = per_board
        if (k + 1) % 20 == 0:
            print(
                f"  [rank] {k + 1}/{len(all_cols)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    del panel
    gc.collect()

    rows = []
    for board in ("main", "dual"):
        _, ev_T, labels = boards_events[board]
        med = int(np.median(ev_T))
        half_lab = (ev_T <= med).astype(int)  # 1=后半窗
        for fam, cols in fam_cols.items():
            if not cols:
                print(f"  [warn] family {fam} 空, 跳过", flush=True)
                continue
            for off in OFFSETS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    comp = np.nanmean(
                        np.vstack([col_vals[c][off][board] for c in cols]), axis=0
                    )
                ok = np.isfinite(comp)
                auc_all = _auc(comp, labels)
                aucs = []
                for h in (0, 1):
                    mh = ok & (half_lab == h)
                    aucs.append(_auc(comp[mh], labels[mh]))
                p80 = (
                    float(np.mean(labels[ok & (comp >= 0.8)]))
                    if (ok & (comp >= 0.8)).sum() >= 10
                    else np.nan
                )
                rows.append(
                    {
                        "board": board,
                        "family": fam,
                        "lag": off,
                        "n": int(ok.sum()),
                        "auc": round(auc_all, 4),
                        "auc_h1": round(aucs[0], 4),
                        "auc_h2": round(aucs[1], 4),
                        "base_real_rate": round(float(np.mean(labels[ok])), 4),
                        "precision80": round(p80, 4) if np.isfinite(p80) else None,
                    }
                )
    res = pd.DataFrame(rows)
    res = res.sort_values(["board", "auc"], ascending=[True, False])

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"fakeleg_event_{ts}.parquet"
    res.to_parquet(pq_path, index=False)
    top = res[res["lag"] == 0].head(12)
    meta = {
        "ts": ts,
        "days": args.days,
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "leg_thr": LEG_THR,
        "real_fake_thr": [REAL_THR, FAKE_THR],
        "offsets": list(OFFSETS),
        "families": {k: len(v) for k, v in fam_cols.items()},
        "events": {b: int(len(v[0])) for b, v in boards_events.items()},
        "top_lag0": top.to_dict("records"),
        "protocol_deviations": [
            "腿阈 main 8% / dual 15% (3日); 标签=腿后3日净回报 ±2% 分界, 中间带剔除",
            "vp 族为判死族参照 (仅事件研究 AUC, 非特征族复活)",
            "09-03 补 ohlc 衍生族 (close_pos/影线/gap/intraday/spike_rev/amp_pct/vol_x5/turn_x5, 与⑤同口径)",
        ],
    }
    (out_dir / f"fakeleg_event_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} rows={len(res)} ({time.time() - t0:.0f}s)", flush=True)
    print(top.to_string(index=False))
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
