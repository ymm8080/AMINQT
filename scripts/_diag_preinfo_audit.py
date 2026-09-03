"""_diag_preinfo_audit.py — 涨前信息覆盖审计 (全列×赢家回走, 2026-09-03).

动机 (用户 2026-09-02): 「涨前有信息的..为什么没有看见」— 质疑模型漏用涨前
信号. 本探针只读不改模型: 对原始面板每一列, 测量它在「赢家事件前 1-5 日」
的截面分位是否系统性偏离其 20 日前的安慰剂窗, 产出「存在但看不见/看反」
清单, 并交叉参照两个模块的特征空间 (parallel checkpoint 列 / legacy registry
active 特征) + 每列新鲜度.

事件类 (L=事件日):
  win_net3  当日全池 forward net3 (px[L+4]/px[L+1]-1-成本) 截面前 1%  — 大赢家
  lu_main   close_hfq 日涨幅 >= 9.5%  — 主板涨停
  lu_dual   close_hfq 日涨幅 >= 19%   — 双创涨停

每列指标 (per 事件类, 截面 pct-rank 口径, 窗口=月度 per 用户 09-03 修订):
  pre_lift    涨前一个月 (L-20..L-1) 分位均值 − 0.5   (>0 = 涨前月内已被看多)
  placebo     安慰剂窗 (L-40..L-21) 分位均值 − 0.5    (基线漂移对照)
  net_signal  pre − placebo                            (真·涨前月度信息量)
  net_w1..w4  四个 5 日子窗 (L-5..L-1 / L-10..L-6 / L-15..L-11 / L-20..L-16)
              各自 net 信号 — 剖面看信息在涨前第几周出现
  enrich80    涨前月分位 >= 0.8 的事件占比             (高分位富集度)
  n_valid     有效事件数 (双窗各 >=8 个有效日)

交叉参照:
  in_parallel_main/dual  列名在 stage checkpoint schema
  legacy_seen            精确名 = registry active / source_cols 包含 / 前缀族
                         映射到 active dim_group
  freshness              每日非空覆盖率 >= 50% 的最后日期 + 近 20 日均覆盖

WORM: DATA OTHERS/diag/preinfo_audit_<ts>.parquet + .json
用法:
  python scripts/_diag_preinfo_audit.py                # 近 550 交易日
  python scripts/_diag_preinfo_audit.py --days 120     # 快速冒烟
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
TOP_FRAC = 0.01
LU_THR = {"lu_main": 0.095, "lu_dual": 0.19}
PRE_OFFS = tuple(range(1, 21))  # 涨前一个月 L-20..L-1
PLC_OFFS = tuple(range(21, 41))  # 安慰剂 = 再前一个月 L-40..L-21
SUB_WINDOWS = {"w1": (1, 5), "w2": (6, 10), "w3": (11, 15), "w4": (16, 20)}
MIN_VALID = 8
EXCLUDE_COLS = {"symbol", "date", "announce_date"}
FRESH_COV = 0.50

# 原始面板列前缀 → legacy registry dim_group (active 才算看见)
LEGACY_PREFIX_MAP = {
    "cyq": ("dim21_chip_tushare", "chip_morphology"),
    "sh_": ("dim23_shareholder_structure", "dim29_holdertrade"),
    "bt_": ("dim33_block_trade",),
    "margin_": ("dim24_margin_trading",),
    "fina": ("dim22_fundamental_pit", "dim03_fundamentals"),
    "sw_": ("dim28_sector_index",),
    "lhb": ("dim18_lhb", "dim26_lhb_enhanced", "dim32_lhb_glm"),
}


def _legacy_registry_seen() -> tuple[set[str], dict[str, set[str]], set[str]]:
    """(active 特征名集合, dim_group → active 特征名)."""
    p = data_others_path("factor_registry") / "feature_registry.json"
    reg = json.loads(p.read_text(encoding="utf-8"))
    active = {k for k, v in reg["features"].items() if v.get("active")}
    by_dim: dict[str, set[str]] = {}
    for k, v in reg["features"].items():
        if v.get("active"):
            by_dim.setdefault(v.get("dim_group", ""), set()).add(k)
    src_cols: set[str] = set()
    for v in reg["features"].values():
        if v.get("active"):
            src_cols.update(v.get("source_cols") or [])
    return active, by_dim, src_cols


def _checkpoint_cols() -> dict[str, set[str]]:
    import pyarrow.parquet as pq

    from app.pipeline_parallel.config import PANEL

    out = {}
    for board in ("main", "dual"):
        ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
        out[board] = set(pq.ParquetFile(str(ckpt)).schema_arrow.names)
    return out


def _panel_numeric_cols() -> list[str]:
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(str(PANEL_V3_PATH)).schema_arrow
    keep = []
    for f in schema:
        if f.name in EXCLUDE_COLS or f.name.startswith("label_"):
            continue
        if pa_numeric(f.type):
            keep.append(f.name)
    return keep


def pa_numeric(t) -> bool:
    import pyarrow as pa

    return (
        pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_boolean(t)
        or pa.types.is_decimal(t)
    )


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


def _build_events(px: pd.DataFrame, w0: int, last_i: int):
    """三类事件 → {class: (sym_str, L_idx)}; 双月窗都落在面板范围内."""
    lo, hi = w0 + max(PLC_OFFS), last_i - 11
    events: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    net3 = px.shift(-4) / px.shift(-1) - 1.0 - COST
    win_sy, win_L = [], []
    for j in range(lo, hi + 1):
        col = net3.iloc[:, j].dropna()
        if not len(col):
            continue
        k = max(1, int(round(len(col) * TOP_FRAC)))
        top = col.nlargest(k)
        win_sy.extend(top.index.astype(str))
        win_L.extend([j] * k)
    events["win_net3"] = (np.asarray(win_sy), np.asarray(win_L, dtype=int))

    pct = px.pct_change(axis=1, fill_method=None)
    for cls, thr in LU_THR.items():
        mask = pct >= thr
        ss, LL = [], []
        for j in range(lo, hi + 1):
            idx = np.nonzero(mask.iloc[:, j].to_numpy())[0]
            ss.extend(mask.index[idx].astype(str))
            LL.extend([j] * len(idx))
        events[cls] = (np.asarray(ss), np.asarray(LL, dtype=int))
    return events


def _audit_column(
    panel: pd.DataFrame, col: str, events: dict, date_vals: np.ndarray, t0: float
) -> tuple[dict, dict]:
    df = panel[["symbol", "date", col]].copy()
    dt = pd.to_datetime(df["date"]).dt.normalize()
    df["rk"] = df.groupby(dt)[col].rank(pct=True)
    mat = df.assign(dt=dt).pivot(index="symbol", columns="dt", values="rk").sort_index()
    sym_order = mat.index.astype(str).str.zfill(6).to_numpy()

    # 新鲜度: 每日非空覆盖率
    cov = 1.0 - mat.isna().to_numpy().mean(axis=0)
    ok_dates = mat.columns[cov >= FRESH_COV]
    fresh = {
        "last_fresh_date": str(ok_dates[-1].date()) if len(ok_dates) else None,
        "cov20": round(float(cov[-20:].mean()), 3) if len(cov) >= 20 else None,
    }

    sym_to_idx = {s: i for i, s in enumerate(sym_order)}
    date_to_idx = {d: i for i, d in enumerate(mat.columns)}
    vals = mat.to_numpy(dtype="float32")
    del df, mat
    gc.collect()

    out: dict = {}
    for cls, (ev_sy, ev_L) in events.items():
        sidx = np.array([sym_to_idx.get(s, -1) for s in ev_sy])
        keep = sidx >= 0
        sidx, Lidx = sidx[keep], ev_L[keep]
        didx = np.array([date_to_idx.get(pd.Timestamp(d), -1) for d in date_vals[Lidx]])
        keep2 = didx >= 0
        sidx, didx = sidx[keep2], didx[keep2]
        if len(sidx) == 0:
            out[cls] = {"n_valid": 0}
            continue
        pre = vals[sidx[:, None], (didx[:, None] - np.array(PRE_OFFS)[None, :])]
        plc = vals[sidx[:, None], (didx[:, None] - np.array(PLC_OFFS)[None, :])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre_r = np.nanmean(pre, axis=1)
            plc_r = np.nanmean(plc, axis=1)
        ok = (
            (~np.isnan(pre_r))
            & (~np.isnan(plc_r))
            & (np.sum(~np.isnan(pre), axis=1) >= MIN_VALID)
            & (np.sum(~np.isnan(plc), axis=1) >= MIN_VALID)
        )
        pre_f, plc_f = pre_r[ok], plc_r[ok]
        rec = {
            "n_valid": int(ok.sum()),
            "pre_lift": round(float(np.mean(pre_f) - 0.5), 4),
            "placebo": round(float(np.mean(plc_f) - 0.5), 4),
            "net_signal": round(float(np.mean(pre_f - plc_f)), 4),
            "enrich80": round(float(np.mean(pre_f >= 0.8)), 4),
        }
        for wname, (a, b) in SUB_WINDOWS.items():
            offs = np.arange(a, b + 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wr = np.nanmean(
                    vals[sidx[:, None], (didx[:, None] - offs[None, :])], axis=1
                )
            m = ok & np.isfinite(wr)
            rec[f"net_{wname}"] = (
                round(float(np.mean(wr[m] - plc_r[m])), 4) if m.any() else None
            )
        out[cls] = rec
    print(
        f"  [col] {col} {out.get('win_net3', {}).get('net_signal')} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return out, fresh


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

    cols_all = _panel_numeric_cols()
    print(f"[cols] {len(cols_all)} numeric", flush=True)

    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date"] + cols_all,  # close_hfq ∈ cols_all, 勿重复列
        filters=[("date", ">=", cutoff)],
    )
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    px, cal = _pivots(panel[["symbol", "date", "close_hfq"]])
    print(
        f"[pivot] symbols={len(px)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    date_vals = pd.to_datetime(panel["date"]).dt.normalize().to_numpy()
    events = _build_events(px, w0=40, last_i=len(cal) - 12)
    for cls, (s, _L) in events.items():
        print(f"[events] {cls}: {len(s):,}", flush=True)

    active, by_dim, src_cols = _legacy_registry_seen()
    ckpt_cols = _checkpoint_cols()
    print(
        f"[xref] registry active={len(active)} ckpt main={len(ckpt_cols['main'])} dual={len(ckpt_cols['dual'])}",
        flush=True,
    )

    rows = []
    for col in cols_all:
        metrics, fresh = _audit_column(panel, col, events, date_vals, t0)
        legacy_seen, legacy_how = False, "none"
        if col in active:
            legacy_seen, legacy_how = True, "exact"
        elif col in src_cols:
            legacy_seen, legacy_how = True, "source_cols"
        else:
            for pref, dims in LEGACY_PREFIX_MAP.items():
                if col.startswith(pref) and any(by_dim.get(d) for d in dims):
                    legacy_seen, legacy_how = True, f"prefix:{pref}"
                    break
        row = {
            "col": col,
            **fresh,
            "legacy_seen": legacy_seen,
            "legacy_how": legacy_how,
            "in_parallel_main": col in ckpt_cols["main"],
            "in_parallel_dual": col in ckpt_cols["dual"],
        }
        for cls in events:
            row.update({f"{cls}_{k}": v for k, v in metrics.get(cls, {}).items()})
        rows.append(row)

    res = pd.DataFrame(rows)
    sig = "win_net3_net_signal"
    res["abs_net"] = res[sig].abs()
    res = res.sort_values("abs_net", ascending=False)

    blind = res[(res["abs_net"] >= res["abs_net"].quantile(0.75))]
    blind_list = {
        "not_in_parallel_main": blind[~blind["in_parallel_main"]]["col"].tolist(),
        "not_in_parallel_dual": blind[~blind["in_parallel_dual"]]["col"].tolist(),
        "not_legacy_seen": blind[~blind["legacy_seen"]]["col"].tolist(),
        "top10_abs_net": res.head(10)[
            ["col", sig, "win_net3_pre_lift", "win_net3_enrich80"]
        ].to_dict("records"),
    }

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"preinfo_audit_{ts}.parquet"
    res.drop(columns=["abs_net"]).to_parquet(pq_path, index=False)
    meta = {
        "ts": ts,
        "days": args.days,
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "top_frac": TOP_FRAC,
        "cost": COST,
        "windows": {
            "pre": list(PRE_OFFS),
            "placebo": list(PLC_OFFS),
            "min_valid": MIN_VALID,
        },
        "n_cols": len(cols_all),
        "events": {cls: int(len(s)) for cls, (s, _) in events.items()},
        "xref": {
            "registry_active": len(active),
            "ckpt_main_cols": len(ckpt_cols["main"]),
            "ckpt_dual_cols": len(ckpt_cols["dual"]),
            "legacy_seen": int(res["legacy_seen"].sum()),
            "in_parallel_main": int(res["in_parallel_main"].sum()),
        },
        "blind_lists": blind_list,
        "protocol_deviations": [
            "截面=面板在场股票 (停牌日缺席不入截面)",
            "close_hfq 日涨幅近似真实涨幅 (除权日含分红效应)",
            "winner=全池 net3 前 1% (模块无关口径)",
        ],
    }
    (out_dir / f"preinfo_audit_{ts}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} rows={len(res)} ({time.time() - t0:.0f}s)", flush=True)
    print(json.dumps(blind_list, ensure_ascii=False, indent=1))
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
