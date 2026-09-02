# -*- coding: utf-8 -*-
"""_q90_slot_eval.py — q90 彩票槽位规则网格终审 (重建版, 原 08-30 稿被清理).

输入: scripts/_diag_q90_slot_replay.py 产出的全池回放 parquet
      (symbol/pred_ret_10d/prob_up/base_rate/pain_prob/pred_q90_3d/net_3d/net_10d/amount/date/board).

口径 (与 08-30 原稿一致, 按旧 JSON 对拍校验):
- 闸重建 = 当期 LEGACY_ENTRY_GATE 非 bear: prob_up(=compound_prob) > base_rate +
  prob_margin; pred_ret_10d(=compound_ret) > 0; pain_prob <= pain_max.
  q50 符号闸 2026-09-02 已撤 (q50_sign_gate=False), 不重建.
- base 排名键 = pred_ret_10d 降序 TOP10 (legacy 定案, 平局按 symbol 稳定).
- 槽位: s{k} = base TOP10 末 k 席换成槽位候选 (候选=过闸池中不在 base TOP10):
    raw   = pred_q90_3d 最高
    band  = q90 当日过闸池分位在 [band_lo, band_hi] 内者中 pred_q90_3d 最高
            (无合格候选 → 保留原席, 该日照计)
    blend = 0.5*(ret10 分位 + q90 分位) 最高
- 真赢家 = 当日全池 (board+date) amount>=floor 且 net_3d 非 NaN 中 net_3d TOP10.
- cov = 臂 TOP10 ∩ 真赢家 计数 / 日均; Δ = 配对日 (臂−base) net3 均值, 全窗+两半窗.
- 新增: 真赢家在当日过闸池 q90/ret10 分位带位 (旧判词: 中位 81 分位, 带外极端头=疯票).

WORM: data/others/_q90_slot_eval_<ts>.json

用法:
  python tmp_t/_q90_slot_eval.py --replay "D:/AMINQT/DATA OTHERS/diag/q90_slot_replay_<ts>.parquet"
  python tmp_t/_q90_slot_eval.py --replay <旧parquet> --expect data/others/_q90_slot_eval_20260830_185420.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import LEGACY_ENTRY_GATE, data_others_path

TOPN = 10
WINNER_TOPN = 10


def gate_mask(df: pd.DataFrame, board: str, q50_gate: bool) -> pd.Series:
    margin = LEGACY_ENTRY_GATE["prob_margin"][board]
    pain_max = LEGACY_ENTRY_GATE["pain_max"][board]
    prob_ok = df["prob_up"] > (df["base_rate"] + margin)
    ret_ok = df["pred_ret_10d"] > 0
    pain_ok = df["pain_prob"].fillna(0) <= pain_max
    ok = prob_ok & ret_ok & pain_ok
    if q50_gate:  # 08-30 旧环境复现: q50 符号闸 (09-02 已撤)
        if {"pred_q50_3d", "pred_q50_5d"}.issubset(df.columns):
            ok &= (df["pred_q50_3d"].fillna(df["pred_ret_10d"]) > 0) & (
                df["pred_q50_5d"].fillna(df["pred_ret_10d"]) > 0
            )
        elif "pred_q50" in df.columns:
            ok &= df["pred_q50"].fillna(df["pred_ret_10d"]) > 0
    return ok


def arm_stats(picks: list[set[str]], day_net: dict[str, tuple[dict, dict]],
              winners: dict[str, set[str]]) -> dict:
    """picks: 日→symbol 集; day_net: 日→(sym→net3, sym→net10); winners: 日→真赢家集."""
    net3s, net10s, hits, covs = [], [], [], []
    for d, picks_set in picks.items():
        n3, n10 = day_net[d]
        vals3 = [n3[s] for s in picks_set if s in n3 and pd.notna(n3[s])]
        vals10 = [n10[s] for s in picks_set if s in n10 and pd.notna(n10[s])]
        if vals3:
            net3s.append(np.mean(vals3))
            hits.append(np.mean([v > 0 for v in vals3]))
        if vals10:
            net10s.append(np.mean(vals10))
        covs.append(len(picks_set & winners.get(d, set())))
    return {
        "days": len(picks),
        "net3": float(np.mean(net3s)) if net3s else None,
        "net10": float(np.mean(net10s)) if net10s else None,
        "hit3": float(np.mean(hits)) if hits else None,
        "cov": float(np.mean(covs)) if covs else None,
    }


def paired_delta(arm: dict[str, float], base: dict[str, float]) -> dict:
    common = sorted(set(arm) & set(base))
    if not common:
        return {"d3_full": None, "d3_h1": None, "d3_h2": None, "days": 0}
    dif = np.array([arm[d] - base[d] for d in common])
    h = len(common) // 2
    return {
        "d3_full": float(dif.mean()),
        "d3_h1": float(dif[:h].mean()) if h else None,
        "d3_h2": float(dif[h:].mean()) if len(common) - h else None,
        "days": len(common),
    }


def process_board(df: pd.DataFrame, board: str, amount_floor: float,
                  band_lo: float, band_hi: float, q50_gate: bool) -> dict:
    b = df[df["board"] == board].copy()
    b["symbol"] = b["symbol"].astype(str).str.zfill(6)
    b["date"] = b["date"].astype(str)

    ok = gate_mask(b, board, q50_gate)
    gated = b[ok].copy()
    # 分位带基座 = 当日全板池分位 (与 08-29 诚实窗"q90 排 81 分位"同语)
    b["q90_pct"] = b.groupby("date")["pred_q90_3d"].rank(pct=True)
    b["ret_pct"] = b.groupby("date")["pred_ret_10d"].rank(pct=True)
    gated = gated.merge(
        b[["q90_pct", "ret_pct"]], left_index=True, right_index=True, how="left"
    )
    gated["blend"] = 0.5 * (gated["ret_pct"].fillna(0) + gated["q90_pct"].fillna(0))

    # 真赢家: 当日双板合并池 amount 地板 + net3 top (08-30 对拍: union 复现 11/103)
    full_ok = (df["amount"] >= amount_floor) & df["net_3d"].notna()
    u = df[full_ok].copy()
    u["symbol"] = u["symbol"].astype(str).str.zfill(6)
    u["date"] = u["date"].astype(str)
    u = u.drop_duplicates(["date", "symbol"])
    winners: dict[str, set[str]] = {}
    for d, g in u.groupby("date"):
        top = g.nlargest(WINNER_TOPN, "net_3d")
        winners[d] = set(top["symbol"])

    day_net: dict[str, tuple[dict, dict]] = {}
    for d, g in b.groupby("date"):
        day_net[d] = (
            dict(zip(g["symbol"], g["net_3d"])),
            dict(zip(g["symbol"], g["net_10d"])),
        )

    out: dict = {"gate": {
        "prob_margin": LEGACY_ENTRY_GATE["prob_margin"][board],
        "pain_max": LEGACY_ENTRY_GATE["pain_max"][board],
        "q50_sign_gate": LEGACY_ENTRY_GATE.get("q50_sign_gate", False),
        "pool_days": int(gated.groupby("date").ngroups),
        "pool_mean_per_day": float(len(gated) / max(gated.groupby("date").ngroups, 1)),
    }}

    daily_base: dict[str, list[str]] = {}
    for d, g in gated.groupby("date"):
        top = g.sort_values(["pred_ret_10d", "symbol"], ascending=[False, True]).head(TOPN)
        daily_base[d] = list(top["symbol"])

    def build_arm(kind: str, k: int) -> dict[str, list[str]]:
        daily: dict[str, list[str]] = {}
        for d, base in daily_base.items():
            g = gated[gated["date"] == d]
            pool = g[~g["symbol"].isin(base)]
            if kind == "raw":
                cand = pool.nlargest(k, "pred_q90_3d")["symbol"].tolist()
            elif kind == "band":
                sub = pool[(pool["q90_pct"] >= band_lo) & (pool["q90_pct"] <= band_hi)]
                cand = sub.nlargest(k, "pred_q90_3d")["symbol"].tolist()
            else:  # blend
                cand = pool.nlargest(k, "blend")["symbol"].tolist()
            picks = base[: TOPN - k] + cand + base[TOPN - len(cand):] if cand else base
            daily[d] = list(dict.fromkeys(picks))[:TOPN]
        return daily

    arms: dict[str, dict[str, list[str]]] = {"base": daily_base}
    for k in (1, 2):
        for kind in ("raw", "band", "blend"):
            arms[f"s{k}_{kind}"] = build_arm(kind, k)

    res: dict = {}
    base_daily_net3 = {}
    for name, daily in arms.items():
        picks_sets = {d: set(v) for d, v in daily.items()}
        st = arm_stats(picks_sets, day_net, winners)
        if name == "base":
            base_daily_net3 = {
                d: np.mean([day_net[d][0][s] for s in v
                            if s in day_net[d][0] and pd.notna(day_net[d][0][s])])
                for d, v in daily.items()
            }
        else:
            arm_daily_net3 = {
                d: np.mean([day_net[d][0][s] for s in v
                            if s in day_net[d][0] and pd.notna(day_net[d][0][s])])
                for d, v in daily.items()
            }
            st.update(paired_delta(arm_daily_net3, base_daily_net3))
        res[name] = st
    out["arms"] = res

    # 席位衰减 (base)
    seat: dict[str, dict] = {}
    for d, picks in daily_base.items():
        for i, s in enumerate(picks, 1):
            n3, n10 = day_net[d]
            r = seat.setdefault(str(i), {"net3": [], "net10": []})
            if s in n3 and pd.notna(n3[s]):
                r["net3"].append(n3[s])
            if s in n10 and pd.notna(n10[s]):
                r["net10"].append(n10[s])
    out["seat_decay"] = {
        s: {"net3": float(np.mean(v["net3"])) if v["net3"] else None,
            "net10": float(np.mean(v["net10"])) if v["net10"] else None}
        for s, v in sorted(seat.items(), key=lambda kv: int(kv[0]))
    }

    # 真赢家带位: 赢家在当日过闸池的 q90 / ret10 分位
    wrows = []
    for d, wset in winners.items():
        g = gated[gated["date"] == d]
        if g.empty:
            continue
        g = g.set_index("symbol")
        for s in wset:
            if s in g.index:
                wrows.append({
                    "q90_pct": g.at[s, "q90_pct"] if pd.notna(g.at[s, "q90_pct"]) else np.nan,
                    "ret_pct": g.at[s, "ret_pct"] if pd.notna(g.at[s, "ret_pct"]) else np.nan,
                })
    w = pd.DataFrame(wrows)
    out["winner_band"] = {
        "n": int(len(w)),
        "q90_pct_median": float(w["q90_pct"].median()) if len(w) else None,
        "q90_in_band": float(((w["q90_pct"] >= band_lo) & (w["q90_pct"] <= band_hi)).mean())
        if len(w) else None,
        "q90_top20": float((w["q90_pct"] >= 0.80).mean()) if len(w) else None,
        "q90_top10": float((w["q90_pct"] >= 0.90).mean()) if len(w) else None,
        "ret_pct_median": float(w["ret_pct"].median()) if len(w) else None,
        "ret_top20": float((w["ret_pct"] >= 0.80).mean()) if len(w) else None,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--amount-floor", type=float, default=3e7)
    ap.add_argument("--band-lo", type=float, default=0.80)
    ap.add_argument("--band-hi", type=float, default=0.95)
    ap.add_argument("--expect", default=None, help="旧评估 JSON 路径, 对拍校验")
    ap.add_argument("--q50-gate", action="store_true",
                    help="复现 08-30 旧环境 (q50 符号闸当时未撤)")
    args = ap.parse_args()

    df = pd.read_parquet(args.replay)
    df["net_3d"] = pd.to_numeric(df["net_3d"], errors="coerce")
    df["net_10d"] = pd.to_numeric(df["net_10d"], errors="coerce")
    result = {
        "ts": pd.Timestamp.now().isoformat(),
        "replay": os.path.abspath(args.replay),
        "days": int(df["date"].nunique()),
        "rows": int(len(df)),
        "band": [args.band_lo, args.band_hi],
        "amount_floor": args.amount_floor,
    }
    for board in ("main", "dual"):
        result[board] = process_board(df, board, args.amount_floor,
                                      args.band_lo, args.band_hi, args.q50_gate)

    out_path = data_others_path(f"_q90_slot_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path}")

    for board in ("main", "dual"):
        print(f"\n== {board} ==  池/日 {result[board]['gate']['pool_mean_per_day']:.0f}")
        for name, st in result[board]["arms"].items():
            fmt = lambda v: f"{v:+.5f}" if v is not None else "  n/a  "
            print(f"  {name:10s} days={st['days']:3d} net3={st['net3']:+.4f} "
                  f"Δ={fmt(st.get('d3_full'))} (h1 {fmt(st.get('d3_h1'))} / h2 {fmt(st.get('d3_h2'))}) "
                  f"hit3={st['hit3']:.3f} cov={st['cov']:.3f}")
        wb = result[board]["winner_band"]
        print(f"  赢家带位 n={wb['n']} q90中位={wb['q90_pct_median']:.2f} "
              f"带内={wb['q90_in_band']:.2f} top20={wb['q90_top20']:.2f} "
              f"| ret中位={wb['ret_pct_median']:.2f}")

    if args.expect:
        old = json.loads(open(args.expect, encoding="utf-8").read())
        worst = 0.0
        for board in ("main", "dual"):
            for name, st in result[board]["arms"].items():
                o = old["arms"].get(f"{board}:{name}")
                if not o:
                    continue
                for key in ("net3", "net10", "hit3", "cov", "d3_full", "d3_h1", "d3_h2"):
                    if st.get(key) is not None and o.get(key) is not None:
                        worst = max(worst, abs(st[key] - o[key]))
        print(f"\n[expect] 对拍 {os.path.basename(args.expect)} 最大绝对差 = {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
