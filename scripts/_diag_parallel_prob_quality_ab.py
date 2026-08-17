"""_diag_parallel_prob_quality_ab.py — 并行 vs legacy 概率头质量对比 + 跨系统排名 A/B (2026-08-16).

用户问题: (1) 并行概率头比 legacy 概率头质量高吗? (2) 若高, legacy 直接采用并行概率
(两系统产出同股票池前提) → mag_10d × 并行概率 是否提升 legacy 预测质量?

方法:
- 标签统一: mfe_3d >= 0.03 (两 prob_head._add_mfe_3d 公式逐字一致, 同 hfq 价同成本).
- 并行质量: 用 08-16 prod_replay 检查点 data/_diag_replay_wf_pred_<board>.parquet 的
  walk-forward 概率 (pred_wf = prob_ok 行, 与 legacy 同 ok 语义; pred = 全截面,
  生产 gate 语义), 末 250 已实现交易日, 两口径 AUC:
    A. 并行全板 AUC (自己的池)
    B. 并行概率在 legacy E7 行上的 AUC (采用场景 — 与 legacy 新头同池逐行对比)
- legacy 质量: 同窗同标签 AUC (WORM legacy_prob_head_replay_20260816_220029, 审计修复后).
- 排名 A/B: E7 池 (pain_excluded=False) 上键 =
    mag=pred_ret_10d (生产现状) / blend_pp=mag×并行prob / blend_pp_ex=mag×(并行prob−并行base) /
    prob_pp=纯并行prob / blend_new=mag×legacy新头 (参照), TOP-5/10/15 × full/126d/63d × 4 子窗.
    并行 prob NaN → fail-open (按 mag). 另跑 P2=生产闸池 (E7+legacy 新头闸).
- 重叠: 逐日 legacy mag TOP-10 vs 并行 (t3 门+pred_mag_10d) TOP-10 Jaccard;
  E7 行上并行 prob 与 legacy 新头逐日 Spearman (独立信息检验) + 覆盖率.

WORM: DATA_OTHERS/diag/parallel_prob_quality_ab_<ts>.csv/.json
用法: python scripts/_diag_parallel_prob_quality_ab.py
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from app.pipeline_parallel import prob_head
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR, data_others_path

EVAL_DAYS = 250
ABS_TARGET = 0.03
T3_LANDED = {"main": 0.0, "dual": 0.005}
BASE_TAIL = 35  # 生产 base_rate 尾部窗口
BASE_RATE_DAYS = 20
DEPTHS = (5, 10, 15)
WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
N_SUB = 4
KEYS = (
    ("mag", "pred_ret_10d"),
    ("blend_pp", "blend_pp"),
    ("blend_pp_ex", "blend_pp_ex"),
    ("prob_pp", "prob_pp"),
    ("blend_new", "blend_new"),
)
# 并行池打分所需列 (与全量加载时 pool_score 的 avail 一致 → 行集对齐检查点)
POOL_COLS = (
    "amihud_illiq",
    "small_mv_premium",
    "amihud_illiquidity",
    "VAR51",
    "ret_reversal_5d",
    "pv_corr_5",
    "limit_dist_pct",
)


def _load_board_min(board: str) -> pd.DataFrame | None:
    """最小列复现 _diag_parallel_gbm_prod_replay._load_board 的行集 (排序+score过滤+_add_mfe_3d),
    使检查点 pred/pred_wf 按行位置对齐; 只读池打分与 mfe/base 所需列."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    if not fp.exists():
        return None
    schema = pq.read_schema(str(fp)).names
    pool_avail = [c for c in POOL_COLS if c in schema]
    need = [
        "symbol",
        "date",
        "label_pain",
        "close_hfq",
        "high_hfq",
        "adv20",
        "label_pm_3d_net",
        "label_pm_10d_net",
    ] + pool_avail
    t = pq.read_table(str(fp), columns=list(dict.fromkeys(need))).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    t = prob_head._add_mfe_3d(t)
    return t


def _prod_base_series(t: pd.DataFrame, dates: np.ndarray, want: set) -> pd.Series:
    """生产口径 base_rate (无前瞻, 同 prod_replay): 每日 D 尾 35 日切片剔 NaN 后近 20 可观测日均值."""
    base_map: dict[pd.Timestamp, float] = {}
    pos_all = np.searchsorted(dates, t["date"].values)
    for k, d in enumerate(dates):
        if k < BASE_RATE_DAYS + 14:
            continue
        if pd.Timestamp(d) not in want:
            continue
        rows = np.where((pos_all >= k - BASE_TAIL + 1) & (pos_all <= k))[0]
        tail = t.iloc[rows][["symbol", "date", "close_hfq", "high_hfq", "adv20"]].copy()
        b = prob_head._base_rate(tail)
        if b is not None:
            base_map[pd.Timestamp(d)] = b
    return pd.Series(base_map, name="pbase")


def _stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "n_days": 0,
            "picks": 0,
            "avg_picks": 0.0,
            "hit": float("nan"),
            "mean": float("nan"),
            "med": float("nan"),
            "ge5": float("nan"),
            "ge10": float("nan"),
        }
    r = sub["realized_net"].dropna()
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": float(len(sub) / max(1, sub["date"].nunique())),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
    }


def _sub_windows(top: pd.DataFrame, n_sub: int) -> list[dict]:
    dates = np.sort(top["date"].unique())
    step = max(1, len(dates) // n_sub)
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(dates) if i == n_sub - 1 else (i + 1) * step
        seg = top[top["date"].isin(dates[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "rows": int(len(seg)),
                "hit": float((seg["realized_net"] > 0).mean())
                if len(seg)
                else float("nan"),
                "mean": float(seg["realized_net"].mean()) if len(seg) else float("nan"),
            }
        )
    return subs


def _diag(v: pd.Series) -> dict:
    if v.empty:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "n_unique": int(v.nunique()),
        "mode_share": float(v.value_counts().iloc[0] / len(v)),
        "q25": float(v.quantile(0.25)),
        "q50": float(v.quantile(0.50)),
        "q75": float(v.quantile(0.75)),
        "iqr": float(v.quantile(0.75) - v.quantile(0.25)),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def main() -> int:
    replays = sorted(
        glob.glob(str(data_others_path("diag") / "legacy_prob_head_replay_*.csv"))
    )
    if not replays:
        print("[error] 无 legacy replay CSV", flush=True)
        return 1
    leg = pd.read_csv(replays[-1], dtype={"symbol": str})
    leg["date"] = pd.to_datetime(leg["date"])
    print(
        f"[legacy] {os.path.basename(replays[-1])}: {len(leg):,} 票 / "
        f"{leg['date'].nunique()} 日 {leg['date'].min().date()} -> {leg['date'].max().date()}",
        flush=True,
    )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    rows: list[dict] = []
    quality: dict = {}
    overlap_rows: list[dict] = []

    for board in ("main", "dual"):
        t = _load_board_min(board)
        if t is None:
            print(f"[{board}] 无并行面板 -> skip", flush=True)
            continue
        dates = np.unique(t["date"].values)
        cal_test = dates[-EVAL_DAYS:]
        print(
            f"\n===== {board}: 并行面板 {len(t):,} 行 / {len(dates)} 日 "
            f"(末日 {pd.Timestamp(dates[-1]).date()}), 评估窗 {pd.Timestamp(cal_test[0]).date()} -> "
            f"{pd.Timestamp(cal_test[-1]).date()} =====",
            flush=True,
        )

        ckpt = DATA_DIR / f"_diag_replay_wf_pred_{board}.parquet"
        cp = pq.read_table(str(ckpt)).to_pandas()
        pred = cp["pred"].to_numpy()
        pred_wf = cp["pred_wf"].to_numpy()
        print(
            f"[ckpt] pred notna {np.isfinite(pred).sum():,} / pred_wf notna {np.isfinite(pred_wf).sum():,}",
            flush=True,
        )
        del cp
        gc.collect()

        y = (t["mfe_3d"] >= ABS_TARGET).astype(float)
        ok = y.notna() & t["label_pain"].notna()
        in_cal = np.isin(t["date"].values, cal_test)

        # ---- A. 并行全板 walk-forward AUC ----
        m = in_cal & ok & np.isfinite(pred_wf)
        auc_full = (
            float(roc_auc_score(y[m], pred_wf[m])) if m.sum() > 0 else float("nan")
        )

        # ---- 并行 base_rate (评估窗每日, 供 blend_pp_ex) ----
        want = {pd.Timestamp(d) for d in cal_test}
        pbase = _prod_base_series(t, dates, want)
        print(
            f"[base] 并行 base_rate 覆盖 {len(pbase)}/{len(cal_test)} 日 (均值 {pbase.mean():.4f})",
            flush=True,
        )

        # ---- 并行 top10 (生产口径: t3 门 + pred_mag_10d) 供重叠 ----
        work = t[
            ["symbol", "date", "board", "score", "label_pm_3d_net", "label_pm_10d_net"]
        ].copy()
        p3 = calibrate_mag10d(work, target_col="label_pm_3d_net", label_horizon=3)
        p10 = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        mm = work.merge(
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        ).merge(
            p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        par_top = {}
        for d, g in mm[mm["date"].isin(pd.to_datetime(cal_test))].groupby("date"):
            g2 = g[g["pred_ret_3d"] > T3_LANDED[board]]
            if g2.empty:
                continue
            par_top[d] = set(
                g2.sort_values("pred_mag_10d", ascending=False).head(10)["symbol"]
            )
        del work, p3, p10, mm
        gc.collect()

        # ---- 并行概率合并到 legacy 行 ----
        pp = pd.DataFrame(
            {
                "symbol": t["symbol"],
                "date": t["date"],
                "pprob": pred_wf,  # prob_ok 语义 (AUC 用)
                "pprob_all": pred,  # 全截面 (A/B 排名用, 生产 gate 语义)
                "mfe_3d": t["mfe_3d"].to_numpy(),  # 同公式同价源 → legacy 行标签
                "label_pain": t["label_pain"].to_numpy(),
            }
        )
        leg_b = leg[leg["board"] == board].copy()
        leg_b = leg_b.merge(pp, on=["symbol", "date"], how="left")
        leg_b = leg_b.merge(pbase, left_on="date", right_index=True, how="left")
        cov = leg_b["pprob_all"].notna().mean()
        print(
            f"[merge] legacy {board} 行 {len(leg_b):,} | 并行 prob 覆盖率 {cov:.1%}",
            flush=True,
        )

        # ---- B. 采用场景: 并行概率在 legacy E7 行上的 AUC (与 legacy 新头同池同标签) ----
        e7 = leg_b[~leg_b["pain_excluded"].fillna(False)].copy()
        j = e7.dropna(subset=["pprob"]).copy()
        m2 = j["mfe_3d"].notna() & j["label_pain"].notna() & j["pred_prob_new"].notna()
        auc_par_on_legacy = (
            float(
                roc_auc_score(
                    (j.loc[m2, "mfe_3d"] >= ABS_TARGET).astype(int), j.loc[m2, "pprob"]
                )
            )
            if m2.sum() > 0
            else float("nan")
        )
        auc_leg_on_legacy = (
            float(
                roc_auc_score(
                    (j.loc[m2, "mfe_3d"] >= ABS_TARGET).astype(int),
                    j.loc[m2, "pred_prob_new"],
                )
            )
            if m2.sum() > 0
            else float("nan")
        )
        base_rate = (
            float((j.loc[m2, "mfe_3d"] >= ABS_TARGET).mean())
            if m2.sum() > 0
            else float("nan")
        )

        # 逐日 Spearman: 并行 prob vs legacy 新头 (独立信息检验)
        spears = []
        for _, day in j.groupby("date"):
            sub = day[["pred_prob_new", "pprob"]].dropna()
            if (
                len(sub) >= 5
                and sub["pred_prob_new"].nunique() > 1
                and sub["pprob"].nunique() > 1
            ):
                spears.append(sub.corr(method="spearman").iloc[0, 1])
        daily_spear = float(np.mean(spears)) if spears else float("nan")

        quality[board] = {
            "n_parallel_rows": int(len(t)),
            "eval_days": int(len(cal_test)),
            "auc_parallel_fullboard": auc_full,
            "auc_parallel_on_legacy_e7": auc_par_on_legacy,
            "auc_legacy_head_on_legacy_e7": auc_leg_on_legacy,
            "base_rate_e7": base_rate,
            "n_e7_joined": int(len(j)),
            "coverage_pprob": float(cov),
            "daily_spearman_pprob_vs_legacy_new": daily_spear,
            "diag_parallel": _diag(e7["pprob"].dropna()),
            "diag_legacy_new": _diag(e7["pred_prob_new"].dropna()),
        }
        print(
            f"[质量] 并行全板 AUC {auc_full:.3f} | legacy E7 行上: 并行 {auc_par_on_legacy:.3f} vs "
            f"legacy 新头 {auc_leg_on_legacy:.3f} (base {base_rate:.3f}, n {int(m2.sum()):,}) | "
            f"逐日 Spearman {daily_spear:.3f} | IQR 并行 {quality[board]['diag_parallel']['iqr']:.3f} "
            f"vs legacy {quality[board]['diag_legacy_new']['iqr']:.3f}",
            flush=True,
        )

        # ---- 排名 A/B ----
        e7["blend_pp"] = (e7["pred_ret_10d"] * e7["pprob_all"]).where(
            e7["pprob_all"].notna(), e7["pred_ret_10d"]
        )
        e7["blend_pp_ex"] = (
            e7["pred_ret_10d"] * (e7["pprob_all"] - e7["pbase"])
        ).where(e7["pprob_all"].notna() & e7["pbase"].notna(), e7["pred_ret_10d"])
        e7["prob_pp"] = e7["pprob_all"]
        e7["blend_new"] = e7["pred_ret_10d"] * e7["pred_prob_new"]
        keep_new = (
            (e7["pred_prob_new"] > e7["base_prod"] + 0.08)
            | e7["pred_prob_new"].isna()
            | e7["base_prod"].isna()
        )
        p2 = e7[keep_new].copy()
        print(
            f"\n[{board}] E7 池 {len(e7):,} 票/{e7['date'].nunique()} 日 | "
            f"生产闸池 {len(p2):,} 票/{p2['date'].nunique()} 日",
            flush=True,
        )

        for pool_name, pool in (("E7池", e7), ("生产闸池", p2)):
            dates_all = np.sort(pool["date"].unique())
            for rkname, rkcol in KEYS:
                for depth in DEPTHS:
                    for wname, wdays in WINDOWS.items():
                        cutoff = (
                            dates_all[0]
                            if wdays >= len(dates_all)
                            else dates_all[-wdays]
                        )
                        w = pool[pool["date"].values >= cutoff]
                        top = (
                            w.sort_values(["date", rkcol], ascending=[True, False])
                            .groupby("date", sort=False)
                            .head(depth)
                        )
                        s = _stats(top)
                        rows.append(
                            {
                                "board": board,
                                "pool": pool_name,
                                "rank": rkname,
                                "depth": depth,
                                "window": wname,
                                **s,
                            }
                        )
                        sub_s = "  ".join(
                            f"{x['win']}:{x['hit']:.0%}/{x['mean']:+.2%}"
                            for x in _sub_windows(top, N_SUB)
                        )
                        print(
                            f"[{rkname:>9}/top{depth:>2}/{wname:>4}] "
                            f"日{s['n_days']:>3} 票/日{s['avg_picks']:>5.2f} "
                            f"命中{s['hit']:>6.1%} 实得{s['mean']:>+8.2%} "
                            f"中位{s['med']:>+8.2%} ≥5%{s['ge5']:>6.1%} ≥10%{s['ge10']:>6.1%}",
                            flush=True,
                        )
                        print(f"    sub: {sub_s}", flush=True)

        # ---- 重叠: legacy mag TOP-10 vs 并行生产 TOP-10 ----
        jac = []
        for d, g in e7.groupby("date"):
            if d not in par_top:
                continue
            lg = set(g.sort_values("pred_ret_10d", ascending=False).head(10)["symbol"])
            pt = par_top[d]
            if not lg or not pt:
                continue
            jac.append(len(lg & pt) / len(lg | pt))
        jac_mean = float(np.mean(jac)) if jac else float("nan")
        overlap_rows.append(
            {
                "board": board,
                "dates_with_both": int(len(jac)),
                "jaccard_top10_mean": jac_mean,
            }
        )
        print(
            f"[重叠] 并行 TOP-10 vs legacy mag TOP-10 逐日 Jaccard 均值 {jac_mean:.3f} ({len(jac)} 日)",
            flush=True,
        )

        del t, pp, leg_b, e7, p2, j
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"parallel_prob_quality_ab_{ts}.csv", index=False)
    (out_dir / f"parallel_prob_quality_ab_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "source_legacy_replay": os.path.basename(replays[-1]),
                "label": f"mfe_3d >= {ABS_TARGET} (两 prob_head._add_mfe_3d 逐字一致)",
                "rank_keys": (
                    "mag=pred_ret_10d(生产现状); blend_pp=mag×并行prob(fail-open NaN→mag); "
                    "blend_pp_ex=mag×(并行prob−并行base); prob_pp=纯并行prob; "
                    "blend_new=mag×legacy新头(参照)"
                ),
                "quality": quality,
                "overlap_top10": overlap_rows,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/parallel_prob_quality_ab_{ts}.csv/.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
