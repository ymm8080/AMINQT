"""参数扫描结果验证 — 子窗口稳定性 + 参数敏感度 (2026-08-08, 依据《AI扫参可靠性》).

用户分享 Kimi 文档核心: 全局扫参不可靠, 选参不能只看全期平均最高 (挑最高 = 对验证集
多重比较过拟合). 必须补两个检验:
  1) 参数稳定性检验 — 赢家须在 OOS 子窗口(前/后半 + 四分位)都不翻负、且多数窗口赢 ref;
     只靠全期平均赢 = 可能靠个别极端日/股撑起来的虚高.
  2) 参数敏感度检验 — 扰动某个参数后 Top10 清单不应剧烈变化 (用扫描 daily.csv 的
     overlap 列, = 该 config 当日与 ref 清单的重合度). overlap 低(清单大变)却无增益
     = 校准/排名对参数过度敏感 = 实盘不可信.
  3) (可选 --lgbm) LGBM 结果对照 Kimi 表2 建议区间: num_leaves 31-63 /
     min_child_samples 50-100 / reg_lambda 1.0-10.0 — 落在区间外且被挑中的组合,
     标记"需警惕过拟合信号".

用法:
  python scripts/_diag_param_sweep_verify.py [sweep_dir]
      # sweep_dir 含 daily.csv + summary.json; 默认取最新 diag_parallel_allparam_*
  python scripts/_diag_param_sweep_verify.py --lgbm data/_diag_lgbm_*.json
      # LGBM 只做表2 prior 对照 (JSON 无逐窗口数据, 子窗口检验需重跑扫描才可得)
WORM: BACKTEST_RESULT_DIR/diag_param_sweep_verify_<ts>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import BACKTEST_RESULT_DIR

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 敏感度门槛: 平均 overlap 低于此 = 清单变化过大
OVERLAP_SENSITIVE = 0.70
# 子窗口: (名字, 份数)
WINDOW_SPLITS = (("half", 2), ("quarter", 4))
# Kimi 表2 建议区间 (LGBM 超参 prior)
LGBM_PRIOR = {
    "num_leaves": (31, 63),
    "max_depth": (5, 7),
    "learning_rate": (0.05, 0.05),
    "feature_fraction": (0.6, 0.8),
    "colsample_bytree": (0.6, 0.8),
    "subsample": (0.7, 1.0),
    "bagging_fraction": (0.7, 0.7),
    "min_child_samples": (50, 100),
    "reg_lambda": (1.0, 10.0),
    "reg_alpha": (0.1, 1.0),
}


def _nmean(a: pd.Series) -> float:
    v = a.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float("nan")


def load_daily(sweep_dir: str) -> tuple[pd.DataFrame, dict]:
    daily = pd.read_csv(os.path.join(sweep_dir, "daily.csv"))
    summary = {}
    sp = os.path.join(sweep_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            summary = json.load(fh)
    return daily, summary


def verify_ranking(sweep_dir: str, out_dir) -> dict:
    daily, summary = load_daily(sweep_dir)
    need = ["date", "board", "config", "pr5d", "pr10d", "win5d", "win10d", "overlap"]
    for c in need:
        if c not in daily.columns:
            raise RuntimeError(f"daily.csv 缺列 {c} (columns={list(daily.columns)})")
    daily["date"] = pd.to_datetime(daily["date"])
    all_dates = sorted(daily["date"].unique())
    print(f"[verify] {os.path.basename(sweep_dir)}: {len(all_dates)} 交易日 / "
          f"{daily['board'].nunique()} 板块 / configs="
          f"{sorted(daily['config'].unique())}", flush=True)

    ref_name = "ref"
    results = {"sweep_dir": sweep_dir, "n_days": len(all_dates), "boards": {}}
    for board in sorted(daily["board"].unique()):
        sub = daily[daily["board"] == board]
        ref = sub[sub["config"] == ref_name]
        if ref.empty:
            print(f"  [{board}] 无 ref, 跳过", flush=True)
            continue
        b_res = {"ref_pr10d": _nmean(ref["pr10d"]), "ref_pr5d": _nmean(ref["pr5d"]),
                 "configs": []}
        # 子窗口稳定性: 每个 config 只建一条记录, 各窗口往里填
        for cname in sorted(sub["config"].unique()):
            c = sub[sub["config"] == cname]
            rec = {"config": cname, "overall_pr10d": _nmean(c["pr10d"]),
                   "overall_pr5d": _nmean(c["pr5d"]),
                   "win10d_rate": _nmean(c["win10d"]), "overlap": _nmean(c["overlap"]),
                   "windows": {}}
            for wname, k in WINDOW_SPLITS:
                bounds = [all_dates[i * len(all_dates) // k] for i in range(k)] + [all_dates[-1] + pd.Timedelta(days=1)]
                for i in range(k):
                    m = (c["date"] >= bounds[i]) & (c["date"] < bounds[i + 1])
                    mr = (ref["date"] >= bounds[i]) & (ref["date"] < bounds[i + 1])
                    p10c, p10r = _nmean(c.loc[m, "pr10d"]), _nmean(ref.loc[mr, "pr10d"])
                    p5c, p5r = _nmean(c.loc[m, "pr5d"]), _nmean(ref.loc[mr, "pr5d"])
                    rec["windows"][f"{wname}{i}"] = {
                        "pr10d": p10c, "pr10d_ref": p10r,
                        "pr5d": p5c, "pr5d_ref": p5r,
                        "win10": p10c > p10r, "win5": p5c > p5r,
                        "neg10": p10c < 0, "neg5": p5c < 0,
                    }
            b_res["configs"].append(rec)
        # 汇总 verdict (只在 quarter 窗上判, 更严)
        qwins = [f"quarter{i}" for i in range(4)]
        for rec in b_res["configs"]:
            w10 = [rec["windows"][w]["win10"] for w in qwins if w in rec["windows"]]
            neg10 = [rec["windows"][w]["neg10"] for w in qwins if w in rec["windows"]]
            ov = rec["overlap"]
            rec["win10_quarters"] = sum(w10)
            rec["n_quarters"] = len(w10)
            rec["neg10_any_quarter"] = any(neg10)
            better = rec["overall_pr10d"] > b_res["ref_pr10d"]
            if len(w10) == 4 and all(w10) and not any(neg10) and better:
                rec["verdict"] = "STABLE_WIN"
            elif len(w10) >= 2 and sum(w10) >= len(w10) // 2 and not any(neg10) and better:
                rec["verdict"] = "MARGINAL"
            else:
                rec["verdict"] = "FRAGILE"
            # 敏感度
            if ov < OVERLAP_SENSITIVE and not (better and sum(w10) >= 3):
                rec["sensitivity"] = "SENSITIVE(list大变却无增益)"
            elif ov < OVERLAP_SENSITIVE:
                rec["sensitivity"] = "list大变但确有增益"
            else:
                rec["sensitivity"] = "list稳定"
        results["boards"][board] = b_res
    return results


def print_ranking(results: dict) -> None:
    for board, b in results["boards"].items():
        print(f"\n=== [{board}]  ref pr10d={b['ref_pr10d']:+.4f} pr5d={b['ref_pr5d']:+.4f} ===", flush=True)
        print(f"  {'config':<22}{'pr10d':>9}{'pr5d':>9}{'win10d':>8}{'overlap':>9}"
              f"{'qwin':>6}{'negq':>6}  {'verdict':<12}{'sensitivity'}", flush=True)
        for rec in sorted(b["configs"], key=lambda r: r["overall_pr10d"], reverse=True):
            q = f"{rec['win10_quarters']}/{rec['n_quarters']}"
            neg = "Y" if rec["neg10_any_quarter"] else "."
            print(f"  {rec['config']:<22}{rec['overall_pr10d']:>+9.4f}{rec['overall_pr5d']:>+9.4f}"
                  f"{rec['win10d_rate']:>8.1%}{rec['overlap']:>9.2f}{q:>6}{neg:>6}  "
                  f"{rec['verdict']:<12}{rec['sensitivity']}", flush=True)


def verify_lgbm(json_path: str, out_dir) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    meta = d.get("meta", {})
    results = d.get("results", {})
    rows = []
    for key, rec in results.items():
        params = rec.get("params", {})
        flags = []
        for p, (lo, hi) in LGBM_PRIOR.items():
            if p in params:
                v = params[p]
                if v < lo or v > hi:
                    flags.append(f"{p}={v}(建议{lo}-{hi})")
        rows.append({"combo": key, "weighted_ic": rec.get("weighted_ic"),
                     "flags": "; ".join(flags) or "OK", "params": params})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[lgbm] 无结果", flush=True)
        return {"json": json_path, "note": "empty"}
    df = df.sort_values("weighted_ic", ascending=False, na_position="last")
    print(f"\n=== LGBM leaderboard {os.path.basename(json_path)} "
          f"(board={meta.get('board')}, grid={meta.get('grid')}) ===", flush=True)
    for _, r in df.iterrows():
        flag = "  << OUT-OF-PRIOR" if r["flags"] != "OK" else ""
        print(f"  {r['weighted_ic']:.5f}  [{r['combo']}]{flag}", flush=True)
        if r["flags"] != "OK":
            print(f"      prior 对照: {r['flags']}", flush=True)
    df.to_csv(os.path.join(out_dir, "lgbm_prior.csv"), index=False)
    return {"json": json_path, "n_combos": len(df)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", nargs="?", default=None,
                    help="含 daily.csv 的扫描目录 (默认最新 diag_parallel_allparam_*)")
    ap.add_argument("--lgbm", default="", help="LGBM JSON 结果文件, 做表2 prior 对照")
    ap.add_argument("--json", action="store_true", help="只出 JSON 汇总不打印明细")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_param_sweep_verify_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"ts": ts, "lgbm": None, "ranking": None}

    if args.lgbm:
        res["lgbm"] = verify_lgbm(args.lgbm, out_dir)

    sweep_dir = args.sweep_dir
    if sweep_dir is None and not args.lgbm:
        cands = sorted(BACKTEST_RESULT_DIR.glob("diag_parallel_allparam_*"), reverse=True)
        sweep_dir = str(cands[0]) if cands else None
    if sweep_dir:
        res["ranking"] = verify_ranking(sweep_dir, out_dir)
        if not args.json:
            print_ranking(res["ranking"])

    with open(out_dir / "verify.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
