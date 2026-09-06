"""_diag_winner_leak.py — TOP10 真赢家泄漏复盘 (2026-09-03).

背景: 用户目标 = TOP10 抓真赢家. 既有诊断 (08-29/09-03) 显示赢家在 TOP10 名次上
近乎均匀分布 = 头部无区分信号, 但缺一张总账: 赢家到底死在哪个环节. 本脚本在
_rankkey_multiseed_sweep 的 scored 检查点 (e125, 逐日全池打分 + T+10 实得净) 上
做纯后处理, 三泄漏点分账:

  (a) 闸杀    = universe 赢家 − E7 池赢家   (entry gate 把赢家挡在池外)
  (b) 排名漏  = E7 池赢家 − TOP10 赢家     (进池但排名键没送进前十)
  (c) 精度    = TOP10 赢家 / TOP10 席位    (进了前十但没涨 = 假阳性)

赢家口径: realized_net (T+10 c2c 净, 扣 0.2% 往返) ≥ WIN_T (主 5%, 辅 10%, 0% 参考).
池口径: 镜像 _load_pool_from_ckpt (prob>base_rate 且 ret>0 且 [q50闸按当前配置],
E7 = 再剔 pain>0.5), TOP10 = E7 按 pred_ret_10d (生产排名键) 降序前 10.
范围: E7 非空的评估日; universe = 当日全部 scored 行 (已含清洗/板块宇宙).

可分离性诊断: E7 池内 赢家 vs 非赢家的排名键 AUC (逐日 Mann-Whitney) —
  AUC≈0.5 = 现有信号头部无赢家信息 (换训练目标也无米下锅);
  AUC 高 = 有信息但没被利用 (赢家加权/TopK 目标头有空间).

自检: 逐日 top10 均值 net vs rankkey_multiseed summary (key:mag, depth10, E7池)
基线 (main 0.125995 / dual 0.149523) 须逐位一致, 不一致即口径漂移, 大字报错.

WORM: DATA OTHERS/diag/winner_leak_{ts}.json + *_daily_{ts}.csv + *_winners_{ts}.csv
      (全池赢家行含池内排名, 供后续影子臂复用).

用法: python scripts/_diag_winner_leak.py [--eval 125] [--win-t 0.05]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, LEGACY_ENTRY_GATE, data_others_path

EVAL_N_DEFAULT = 125
WIN_T = 0.05
WIN_T_ALT = 0.10
DEPTH = 10
DEPTHS_INFO = (5, 10, 15)
COST = 0.0020  # 与 sweep realized_net 同口径 (仅文档提示, 净值已在检查点内)
ANCHOR = {"main": 0.125995, "dual": 0.149523}  # rankkey_multiseed_20260902_042943
ANCHOR_TOL = 5e-4


def gate_non_pain(fr: pd.DataFrame, q50_gate: bool) -> pd.Series:
    """非 pain 条件 (镜像 _load_pool_from_ckpt: prob>base_rate 且 ret>0 且 [q50])."""
    ok = (fr["prob"] > fr["base_rate"]) & (fr["pred_ret_10d"] > 0)
    if q50_gate and {"pred_q50_3d", "pred_q50_5d"}.issubset(fr.columns):
        ok &= (fr["pred_q50_3d"].fillna(fr["pred_ret_10d"]) > 0) & (
            fr["pred_q50_5d"].fillna(fr["pred_ret_10d"]) > 0
        )
    return ok


def topn_per_day(fr: pd.DataFrame, key: str, depth: int) -> pd.DataFrame:
    """逐日 key 降序前 depth (NaN 排最后, 与 _daily_topn 同口径)."""
    return (
        fr.sort_values(["date", key], ascending=[True, False])
        .groupby("date", sort=False)
        .head(depth)
    )


def day_auc(win_vals: np.ndarray, rest_vals: np.ndarray) -> float:
    """排名键 AUC (Mann-Whitney, 平均秩处理并列); 样本不足返回 NaN."""
    n_w, n_r = len(win_vals), len(rest_vals)
    if n_w < 1 or n_r < 1:
        return float("nan")
    allv = np.concatenate([win_vals, rest_vals])
    ranks = pd.Series(allv).rank().to_numpy()
    r_w = ranks[:n_w].sum()
    return float((r_w - n_w * (n_w + 1) / 2) / (n_w * n_r))


def analyze_board(
    fr: pd.DataFrame, board: str, win_t: float
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """单板三泄漏分账. 返回 (汇总, 逐日表, 赢家明细表)."""
    q50_gate = bool(LEGACY_ENTRY_GATE.get("q50_sign_gate", False))
    fr = fr.copy()
    fr["date"] = pd.to_datetime(fr["date"])
    fr["symbol"] = fr["symbol"].astype(str)
    fr = fr[np.isfinite(fr["realized_net"])].reset_index(drop=True)

    ok_gate = gate_non_pain(fr, q50_gate)
    fr["in_gate"] = ok_gate.to_numpy()
    fr["pain_excluded"] = (fr["pain_prob"].fillna(0) > 0.5).to_numpy()
    fr["in_e7"] = fr["in_gate"] & ~fr["pain_excluded"]
    fr["is_win"] = fr["realized_net"] >= win_t
    fr["is_win_alt"] = fr["realized_net"] >= WIN_T_ALT
    fr["is_gain"] = fr["realized_net"] > 0

    e7_days = sorted(fr.loc[fr["in_e7"], "date"].unique())
    fr = fr[fr["date"].isin(e7_days)].reset_index(drop=True)

    top = topn_per_day(fr[fr["in_e7"]], "pred_ret_10d", DEPTH)
    top_ids = set(map(tuple, top[["date", "symbol"]].to_numpy()))
    fr["in_top10"] = [(d, s) in top_ids for d, s in zip(fr["date"], fr["symbol"])]
    # 池内排名 (key 降序, 1 起)
    e7 = fr[fr["in_e7"]].copy()
    e7["pool_rank"] = (
        e7.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
        .groupby("date", sort=False)
        .cumcount()
        + 1
    )
    fr.loc[e7.index, "pool_rank"] = e7["pool_rank"]

    daily_rows: list[dict] = []
    auc_days: list[float] = []
    win_ranks: list[float] = []
    winner_records: list[dict] = []
    for d, g in fr.groupby("date", sort=True):
        u, p = g, g[g["in_e7"]]
        t = p[p["in_top10"]]
        n_u_win, n_p_win, n_t_win = (
            int(u["is_win"].sum()),
            int(p["is_win"].sum()),
            int(t["is_win"].sum()),
        )
        if n_p_win and len(p) > n_p_win:
            a = day_auc(
                p.loc[p["is_win"], "pred_ret_10d"].to_numpy(),
                p.loc[~p["is_win"], "pred_ret_10d"].to_numpy(),
            )
            if np.isfinite(a):
                auc_days.append(a)
        else:
            a = float("nan")
        win_ranks.extend(p.loc[p["is_win"], "pool_rank"].tolist())
        daily_rows.append(
            {
                "date": pd.Timestamp(d),
                "board": board,
                "n_univ": len(u),
                "n_e7": len(p),
                "n_top10": len(t),
                "win_univ": n_u_win,
                "win_e7": n_p_win,
                "win_top10": n_t_win,
                "gate_killed_win": n_u_win - n_p_win,
                "rank_missed_win": n_p_win - n_t_win,
                "top10_mean_net": float(t["realized_net"].mean()) if len(t) else np.nan,
                "top10_median_net": float(t["realized_net"].median())
                if len(t)
                else np.nan,
                "auc_key": a,
            }
        )
        for _, r in p[p["is_win"]].iterrows():
            winner_records.append(
                {
                    "date": pd.Timestamp(d),
                    "board": board,
                    "symbol": r["symbol"],
                    "pool_rank": r["pool_rank"],
                    "in_top10": bool(r["in_top10"]),
                    "pred_ret_10d": float(r["pred_ret_10d"]),
                    "prob": float(r["prob"]),
                    "realized_net": float(r["realized_net"]),
                }
            )

    daily = pd.DataFrame(daily_rows)
    d10 = daily[daily["n_top10"] > 0]

    tot_u, tot_p, tot_t = (
        int(daily["win_univ"].sum()),
        int(daily["win_e7"].sum()),
        int(daily["win_top10"].sum()),
    )
    slots = int(d10["n_top10"].sum())
    pool_rows = int(daily["n_e7"].sum())
    rank_arr = np.array(win_ranks, dtype=float)
    top10_nets = fr.loc[fr["in_top10"], "realized_net"]
    counts_hist = d10["win_top10"].value_counts().sort_index()

    summary = {
        "board": board,
        "win_t": win_t,
        "days": int(len(daily)),
        "univ_rows": int(len(fr)),
        "pool_rows": pool_rows,
        "slots": slots,
        "win_univ": tot_u,
        "win_e7": tot_p,
        "win_top10": tot_t,
        "capture_of_univ": tot_t / tot_u if tot_u else np.nan,
        "gate_survival": tot_p / tot_u if tot_u else np.nan,
        "precision_per_slot": tot_t / slots if slots else np.nan,
        "pool_base_rate": tot_p / pool_rows if pool_rows else np.nan,
        "lift_vs_pool": (tot_t / slots) / (tot_p / pool_rows)
        if slots and pool_rows
        else np.nan,
        "days_top10_zero_win": int((d10["win_top10"] == 0).sum()),
        "days_top10_zero_win_share": float((d10["win_top10"] == 0).mean())
        if len(d10)
        else np.nan,
        "winners_per_top10_hist": {str(k): int(v) for k, v in counts_hist.items()},
        "winner_pool_rank_med": float(np.median(rank_arr)) if len(rank_arr) else np.nan,
        "winner_pool_rank_p25": float(np.percentile(rank_arr, 25))
        if len(rank_arr)
        else np.nan,
        "winner_pool_rank_p75": float(np.percentile(rank_arr, 75))
        if len(rank_arr)
        else np.nan,
        "winner_share_beyond_top15": float((rank_arr > 15).mean())
        if len(rank_arr)
        else np.nan,
        "auc_key_med": float(np.median(auc_days)) if auc_days else np.nan,
        "auc_key_gt55_share": float(np.mean(np.array(auc_days) > 0.55))
        if auc_days
        else np.nan,
        "auc_key_gt60_share": float(np.mean(np.array(auc_days) > 0.60))
        if auc_days
        else np.nan,
        "top10_net_mean": float(top10_nets.mean()) if len(top10_nets) else np.nan,
        "top10_net_median": float(top10_nets.median()) if len(top10_nets) else np.nan,
        "top10_net_p90": float(top10_nets.quantile(0.9)) if len(top10_nets) else np.nan,
        "top10_member_share_ge_win_t": float((top10_nets >= win_t).mean())
        if len(top10_nets)
        else np.nan,
        "top10_member_share_neg": float((top10_nets < 0).mean())
        if len(top10_nets)
        else np.nan,
        "alt_win10_capture": (
            int(fr.loc[fr["in_top10"], "is_win_alt"].sum())
            / max(1, int(fr["is_win_alt"].sum()))
        ),
        "alt_win10_precision_per_slot": (
            int(fr.loc[fr["in_top10"], "is_win_alt"].sum()) / slots if slots else np.nan
        ),
        "anchor_check": {
            "anchor": ANCHOR[board],
            "computed_daily_mean": float(d10["top10_mean_net"].mean())
            if len(d10)
            else np.nan,
        },
    }
    # 深度扰动 (5/15) 捕获率
    for dep in DEPTHS_INFO:
        tp = topn_per_day(fr[fr["in_e7"]], "pred_ret_10d", dep)
        ids = set(map(tuple, tp[["date", "symbol"]].to_numpy()))
        m = [(d, s) in ids for d, s in zip(fr["date"], fr["symbol"])]
        w = int(fr.loc[m, "is_win"].sum())
        summary[f"capture_depth{dep}"] = w / tot_u if tot_u else np.nan
        summary[f"precision_depth{dep}"] = w / max(1, int(np.sum(m)))
    return summary, daily, pd.DataFrame(winner_records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=EVAL_N_DEFAULT)
    ap.add_argument("--win-t", type=float, default=WIN_T)
    args = ap.parse_args()
    t0 = time.time()

    summaries, dailies, winners = [], [], []
    for board in ("main", "dual"):
        ck = DATA_DIR / f"_diag_rankkey_scored_{board}_e{args.eval}.parquet"
        fr = pd.read_parquet(str(ck))
        s, d, w = analyze_board(fr, board, args.win_t)
        chk = s["anchor_check"]
        diff = abs(chk["computed_daily_mean"] - chk["anchor"])
        status = "OK" if diff <= ANCHOR_TOL else "MISMATCH"
        print(
            f"[{board}] 口径自检 vs sweep 基线: computed={chk['computed_daily_mean']:.6f} "
            f"anchor={chk['anchor']:.6f} diff={diff:.2e} → {status}"
        )
        if status == "MISMATCH":
            print(
                "!!! 口径漂移: 池掩码与 sweep 不一致, 结果不可信, 先排查 q50_sign_gate 配置"
            )
        summaries.append(s)
        dailies.append(d)
        winners.append(w)

    print(
        f"\n[load] rows={sum(len(x) for x in dailies):,} 评估日 "
        f"({pd.to_datetime(dailies[0]['date']).min().date()}.."
        f"{pd.to_datetime(dailies[0]['date']).max().date()}) ({time.time() - t0:.0f}s)"
    )
    for s in summaries:
        b = s["board"]
        print(f"\n===== {b} | 赢家 = T+10 净 ≥ {s['win_t']:.0%} =====")
        print(
            f"  90/125日总账: universe 赢家 {s['win_univ']} → E7池 {s['win_e7']} "
            f"({s['gate_survival']:.0%}) → TOP10 {s['win_top10']} "
            f"({s['capture_of_univ']:.0%})"
        )
        print(
            f"  闸杀赢家 {s['win_univ'] - s['win_e7']} 只 | 排名漏掉赢家 "
            f"{s['win_e7'] - s['win_top10']} 只 | TOP10 捕获 {s['win_top10']} 只"
        )
        print(
            f"  精度: TOP10 席位赢家率 {s['precision_per_slot']:.0%} "
            f"(≈{s['precision_per_slot'] * 10:.1f}/10 每日; "
            f"池基线 {s['pool_base_rate']:.1%} → 提升 {s['lift_vs_pool']:.2f}x)"
        )
        print(
            f"  空枪日: {s['days_top10_zero_win']}/{s['days']} 日 TOP10 无赢家 "
            f"({s['days_top10_zero_win_share']:.0%})"
        )
        print(
            f"  池内赢家排名: 中位 {s['winner_pool_rank_med']:.0f} "
            f"[p25 {s['winner_pool_rank_p25']:.0f}, p75 {s['winner_pool_rank_p75']:.0f}], "
            f"15名外占比 {s['winner_share_beyond_top15']:.0%}"
        )
        print(
            f"  可分离性 AUC(排名键, 赢家vs非): 中位 {s['auc_key_med']:.3f}, "
            f">0.55 占 {s['auc_key_gt55_share']:.0%}, >0.60 占 {s['auc_key_gt60_share']:.0%}"
        )
        print(
            f"  TOP10 内部: 均值 {s['top10_net_mean']:+.1%} vs 中位 {s['top10_net_median']:+.1%} "
            f"(p90 {s['top10_net_p90']:+.1%}); 成员 ≥{s['win_t']:.0%} 占 "
            f"{s['top10_member_share_ge_win_t']:.0%}, 亏损占 {s['top10_member_share_neg']:.0%}"
        )
        print(
            f"  大赢家(≥10%): 捕获 {s['alt_win10_capture']:.0%}, "
            f"席位率 {s['alt_win10_precision_per_slot']:.0%}"
        )
        print(
            f"  深度扰动捕获: d5 {s['capture_depth5']:.0%} / d10 {s['capture_depth10']:.0%} "
            f"/ d15 {s['capture_depth15']:.0%}"
        )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pd.concat(dailies, ignore_index=True).to_csv(
        out_dir / f"winner_leak_daily_{ts}.csv", index=False
    )
    pd.concat(winners, ignore_index=True).to_csv(
        out_dir / f"winner_leak_winners_{ts}.csv", index=False
    )
    (out_dir / f"winner_leak_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval": args.eval,
                "win_t": args.win_t,
                "win_t_alt": WIN_T_ALT,
                "q50_sign_gate": bool(LEGACY_ENTRY_GATE.get("q50_sign_gate", False)),
                "depth": DEPTH,
                "summaries": summaries,
                "note": "scored ckpt 数据截止评估窗末 (realized 完整); 池口径镜像 _load_pool_from_ckpt",
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"\n[saved] {out_dir}\\winner_leak_{ts}.json + _daily_ + _winners_ ({time.time() - t0:.0f}s)"
    )
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
