"""_diag_lgbm_overlap.py — 参数扰动测试: 两候选参数的每日 TOP-N 名单重合度 (AISWEEP2 §8).

配方 (Kimi): 选定参数后微调 num_leaves ±20%, 若每日 TOP-N 推荐名单变化 >50%,
说明模型对参数过度敏感 → 该组参数弃用. 只有扰动后名单稳定的参数才可实盘.

与 _diag_lgbm_param_sweep.py 复用 build_slices/_fit_* (同协议: OOS 250d, es 早停).

用法:
  python scripts/_diag_lgbm_overlap.py --board main --kind 10d_cls \
      --a num_leaves=15 --b num_leaves=31 --n 15
输出: stdout + data/_diag_lgbm_overlap_{board}_{kind}_{ts}.json (WORM)
判据: mean_jaccard < 0.5 (= 名单变化>50%) → 参数敏感, 弃用较激进者.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(ROOT, "_diag_lgbm_param_sweep.py")
sys.path.insert(0, os.path.join(ROOT, ".."))


def _load_sweep():
    spec = importlib.util.spec_from_file_location("sweep_mod", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_cand(spec: str) -> dict:
    c = {}
    for kv in spec.replace(" ", "").split(","):
        if not kv:
            continue
        k, v = kv.split("=")
        c[k] = float(v) if "." in v else int(v)
    return c


def daily_topn_sets(df: pd.DataFrame, pred_col: str, n: int) -> dict:
    out = {}
    for d, sub in df.groupby("date"):
        top = sub.sort_values(pred_col, ascending=False).head(n)["symbol"].tolist()
        out[d] = set(top)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--a", required=True, help="候选 A, 如 num_leaves=15")
    ap.add_argument("--b", required=True, help="候选 B, 如 num_leaves=31")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--oos", type=int, default=250)
    ap.add_argument("--es", type=int, default=20)
    args = ap.parse_args()

    sw = _load_sweep()
    ckpt_path, bundle_path = sw.CKPT[args.board]
    bundle = pd.read_pickle(bundle_path)
    feature_cols = [c for c in bundle["feature_cols"]]
    panel = pd.read_parquet(ckpt_path)
    miss = [c for c in feature_cols if c not in panel.columns]
    if miss:
        raise RuntimeError(f"检查点缺特征 {len(miss)}: {miss[:10]}")
    # 与 sweep main 一致: 派生缺失的 cls_net 标签 (label_engine 口径 cls_net = net > 0)
    for h in (1, 2, 3, 5, 10):
        col = f"label_pm_{h}d_cls_net"
        if col in sw.KIND_LABEL.values() and col not in panel.columns:
            base = f"label_pm_{h}d_net"
            if base in panel.columns:
                panel[col] = (panel[base] > 0).astype("float")
                panel.loc[panel[base].isna(), col] = np.nan
                print(f"[derive] {col} <- {base}>0", flush=True)

    s = sw.build_slices(panel, feature_cols, args.kind, args.oos, args.es)
    params_a = {**sw.BASE_PARAMS, **parse_cand(args.a)}
    params_b = {**sw.BASE_PARAMS, **parse_cand(args.b)}

    use_pain = args.kind == "pain"
    use_cls = args.kind.endswith("_cls")
    t0 = time.time()
    if use_pain:
        pa, _ = sw._fit_pain(s, params_a)
        pb, _ = sw._fit_pain(s, params_b)
    elif use_cls:
        pa = sw._fit_classifier(s, params_a).predict_proba(s["oos_X"])[:, 1]
        pb = sw._fit_classifier(s, params_b).predict_proba(s["oos_X"])[:, 1]
    elif args.kind.endswith("_q"):
        pa, _ = sw._fit_quantile(s, params_a)
        pb, _ = sw._fit_quantile(s, params_b)
    elif args.kind.endswith("_rank"):
        pa, _ = sw._fit_ranker(s, params_a)
        pb, _ = sw._fit_ranker(s, params_b)
    else:
        pa = sw._fit_regressor(s, params_a).predict(s["oos_X"])
        pb = sw._fit_regressor(s, params_b).predict(s["oos_X"])

    oos = s["oos_frame"].copy()
    oos["_pa"] = pa
    oos["_pb"] = pb
    set_a = daily_topn_sets(oos, "_pa", args.n)
    set_b = daily_topn_sets(oos, "_pb", args.n)
    common = sorted(set(set_a) & set(set_b))
    jacs = []
    for d in common:
        u = set_a[d] | set_b[d]
        jacs.append(len(set_a[d] & set_b[d]) / len(u) if u else 1.0)
    jacs = np.asarray(jacs, dtype=float)
    res = {
        "board": args.board, "kind": args.kind, "n": args.n,
        "a": parse_cand(args.a), "b": parse_cand(args.b),
        "n_days": int(len(common)),
        "mean_jaccard": float(jacs.mean()) if len(jacs) else np.nan,
        "median_jaccard": float(np.median(jacs)) if len(jacs) else np.nan,
        "change_pct": float((1 - jacs.mean()) * 100) if len(jacs) else np.nan,
        "pct_days_change_gt50": float((jacs < 0.5).mean() * 100) if len(jacs) else np.nan,
        "els": float(time.time() - t0),
    }
    print(
        f"[overlap] {args.board}/{args.kind} N={args.n} "
        f"{args.a} vs {args.b} -> mean_jaccard={res['mean_jaccard']:.3f} "
        f"change={res['change_pct']:.1f}% days_gt50={res['pct_days_change_gt50']:.0f}% "
        f"({len(common)}d)", flush=True
    )
    out_path = (
        f"data/_diag_lgbm_overlap_{args.board}_{args.kind}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"[saved] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
