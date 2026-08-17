"""_diag_legacy_mag10d_cal.py — legacy 幅度 10d 校准 A/B (2026-08-17).

用户指令: IMPROVE 幅度模型本身 LEGACY 10D 校准 (并行 calibrate_mag10d 同款移植).

机制 (并行定案配方, calibration.py):
  每日 D: 尾 21 已实现交易日 (只用 ≤D-11 行, realized_drop=11) 横截面 OLS
  realized_net ~ pred_ret_10d → cs=abs(cs), ci; mag = |cs|·pred + ci.
  关键推论: mag 是 pred 的逐日单调仿射变换 → **对排名 no-op** (同日 top-N 不变);
  校准真正能改的是**正收益闸**: mag>0 ⟺ pred > τ(D) (τ=-ci/|cs| 自适应阈值),
  替代现生产闸 pred>0 (实测 100% 行全过 = 空转, 且 pred 均值 +4.0~6.6% vs 实现
  +1.1~1.3% 系统高估, 偏差随时间扩大).

A/B:
  P_e7 = E7 池 (基线, 现生产闸=无操作)
  P_cal = E7 ∧ mag>0 (校准闸, 自适应阈值)
  两池同键排名 pred_ret_10d; 另加 mag 键作 no-op 自检 (应与 pred 键逐日同序).
  TOP-5/10/15 × full/126d/63d × 4 子窗; 报告闸活跃日/行数削减/τ 分布.
  拟合窗行数 < 50 → 当日不出票 (两池同规则, 同日比较).

WORM: DATA_OTHERS/diag/legacy_mag10d_cal_<ts>.csv/.json
用法: python scripts/_diag_legacy_mag10d_cal.py [replay_csv_path]
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import data_others_path

CAL_N = 21  # 并行定案窗
REALIZED_DROP = 11  # buy_lag=1 + label_horizon=10
CROSS_MIN_N = 50  # 并行定案 cross_min_n
DEPTHS = (5, 10, 15)
WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
N_SUB = 4


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


def _eval(pool: pd.DataFrame, dates_all: np.ndarray) -> list[dict]:
    rows = []
    for rkname, rkcol in (("mag_raw", "pred_ret_10d"), ("mag_cal", "mag")):
        for depth in DEPTHS:
            for wname, wdays in WINDOWS.items():
                cutoff = dates_all[0] if wdays >= len(dates_all) else dates_all[-wdays]
                w = pool[pool["date"].values >= cutoff]
                top = (
                    w.sort_values(["date", rkcol], ascending=[True, False])
                    .groupby("date", sort=False)
                    .head(depth)
                )
                s = _stats(top)
                rows.append(
                    {"pool": None, "rank": rkname, "depth": depth, "window": wname, **s}
                )
                sub_s = "  ".join(
                    f"{x['win']}:{x['hit']:.0%}/{x['mean']:+.2%}"
                    for x in _sub_windows(top, N_SUB)
                )
                print(
                    f"[{rkname:>7}/top{depth:>2}/{wname:>4}] "
                    f"日{s['n_days']:>3} 票/日{s['avg_picks']:>5.2f} "
                    f"命中{s['hit']:>6.1%} 实得{s['mean']:>+8.2%} "
                    f"中位{s['med']:>+8.2%} ≥5%{s['ge5']:>6.1%} ≥10%{s['ge10']:>6.1%}",
                    flush=True,
                )
                print(f"    sub: {sub_s}", flush=True)
    return rows


def main() -> int:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(
            glob.glob(str(data_others_path("diag") / "legacy_prob_head_replay_*.csv"))
        )[-1]
    )
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    print(
        f"[load] {os.path.basename(path)}: {len(df):,} 票 / {df['date'].nunique()} 日",
        flush=True,
    )
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    rows: list[dict] = []
    gate_diag: dict = {}

    for board in ("main", "dual"):
        b = df[df["board"] == board].copy()
        b = b.dropna(subset=["pred_ret_10d", "realized_net"])
        dates = np.sort(b["date"].unique())
        b = b.sort_values(["date", "symbol"]).reset_index(drop=True)
        date_idx = {d: i for i, d in enumerate(dates)}
        idx = np.array([date_idx[d] for d in b["date"]])
        pred = b["pred_ret_10d"].to_numpy()
        real = b["realized_net"].to_numpy()
        e7 = ~b["pain_excluded"].fillna(False).to_numpy()

        # 逐日横截面 OLS (尾 21 已实现交易日, 只含 ≤D-11 行)
        mag = np.full(len(b), np.nan)
        tau_map: dict[pd.Timestamp, float] = {}
        cs_map: dict[pd.Timestamp, float] = {}
        n_fit: dict[pd.Timestamp, int] = {}
        for i, d in enumerate(dates):
            lo, hi = max(0, i - CAL_N), i - REALIZED_DROP
            if hi < lo:
                continue
            m = (idx >= lo) & (idx <= hi) & e7 & np.isfinite(pred) & np.isfinite(real)
            n = int(m.sum())
            if n < CROSS_MIN_N:
                continue
            x, y = pred[m], real[m]
            xm, ym = x.mean(), y.mean()
            var = float(((x - xm) ** 2).sum())
            if var <= 1e-12:
                cs, ci = 0.0, ym
            else:
                cs = float(((x - xm) * (y - ym)).sum() / var)
                cs = abs(cs)
                ci = ym - cs * xm
            today = idx == i
            mag[today] = cs * pred[today] + ci
            tau_map[d] = -ci / cs if cs > 1e-12 else np.inf
            cs_map[d] = cs
            n_fit[d] = n

        b["mag"] = mag
        # no-op 自检: mag 键 top-10 与 pred 键同日同集 (单调仿射 → 应完全一致)
        e7df = b[e7].copy()
        same = True
        for d, g in e7df.groupby("date"):
            if len(g) < 2 or g["mag"].isna().any():
                continue
            a = set(g.nlargest(10, "pred_ret_10d")["symbol"])
            c = set(g.nlargest(10, "mag")["symbol"])
            if a != c:
                same = False
                break
        print(
            f"\n===== {board}: {len(dates)} 日, 拟合 {sum(1 for v in n_fit.values() if v)} 日 "
            f"(no-op 自检: mag 排名==pred 排名 {same}) =====",
            flush=True,
        )

        # 闸分布
        tau_arr = np.array([v for v in tau_map.values() if np.isfinite(v)])
        active = e7 & np.isfinite(mag)
        cuts = e7 & np.isfinite(mag) & (mag <= 0)
        gate_diag[board] = {
            "days_fitted": len(tau_map),
            "days_gate_active": int(len(set(b.loc[cuts, "date"].unique()))),
            "rows_e7": int(e7.sum()),
            "rows_with_mag": int(active.sum()),
            "rows_cut_by_gate": int(cuts.sum()),
            "cut_share": float(cuts.sum() / max(1, active.sum())),
            "tau_mean": float(tau_arr.mean()) if len(tau_arr) else float("nan"),
            "tau_q25": float(np.quantile(tau_arr, 0.25))
            if len(tau_arr)
            else float("nan"),
            "tau_q50": float(np.quantile(tau_arr, 0.50))
            if len(tau_arr)
            else float("nan"),
            "tau_q75": float(np.quantile(tau_arr, 0.75))
            if len(tau_arr)
            else float("nan"),
            "cs_mean": float(np.mean(list(cs_map.values())))
            if cs_map
            else float("nan"),
        }
        print(
            f"[闸] 活跃 {gate_diag[board]['days_gate_active']}/{gate_diag[board]['days_fitted']} 日 | "
            f"削减 {gate_diag[board]['cut_share']:.1%} 行 | "
            f"τ q25/q50/q75 = {gate_diag[board]['tau_q25']:+.2%}/{gate_diag[board]['tau_q50']:+.2%}/"
            f"{gate_diag[board]['tau_q75']:+.2%} | |cs| 均值 {gate_diag[board]['cs_mean']:.3f}",
            flush=True,
        )

        for pool_name, pool in (
            ("E7池(现闸=无操作)", e7df[e7df["pred_ret_10d"] > 0]),
            ("校准闸池(mag>0)", e7df[e7df["mag"] > 0]),
        ):
            print(
                f"\n[{board} / {pool_name}] {len(pool):,} 票/{pool['date'].nunique()} 日",
                flush=True,
            )
            for r in _eval(pool, dates):
                r["board"] = board
                r["pool"] = pool_name
                rows.append(r)

    pd.DataFrame(rows).to_csv(out_dir / f"legacy_mag10d_cal_{ts}.csv", index=False)
    (out_dir / f"legacy_mag10d_cal_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "source": os.path.basename(path),
                "recipe": (
                    "每日尾21已实现交易日(≤D-11)横截面OLS realized~pred_ret_10d, "
                    "cs=abs(cs), mag=|cs|·pred+ci; 校准闸=mag>0 ⟺ pred>τ(D); "
                    "现生产闸=pred_ret_10d>0 (实测100%行全过)"
                ),
                "gate_diag": gate_diag,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/legacy_mag10d_cal_{ts}.csv/.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
