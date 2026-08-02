# -*- coding: utf-8 -*-
"""SW feature eval: T+1, T+3, T+5."""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import data_others_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
H = [1, 3, 5]

SC = []
DROPS = {"sw_l2_pe", "sw_l3_pe", "sw_l3_ret_1d"}
for L in ["l1", "l2", "l3"]:
    for s in [
        "close",
        "vol",
        "amount",
        "pe",
        "pb",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "momentum_accel",
        "turnover_anomaly",
        "rotation_position",
        "relative_strength",
    ]:
        c = f"sw_{L}_{s}"
        if c not in DROPS:
            SC.append(c)


def load_df(n=200, d0="2022-01-01"):
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35 as FE

    cl = [
        "date",
        "symbol",
        "close",
        "high",
        "low",
        "open",
        "volume",
        "amount",
        "pre_close",
        "industry",
        "sw_l1_name",
        "sw_l2_name",
        "sw_l3_name",
    ]
    logger.info("Loading panel...")
    df = pd.read_parquet(PANEL, columns=cl)
    df = df[df["date"] >= pd.Timestamp(d0)]
    if n > 0:
        st = sorted(df["symbol"].unique())
        if len(st) > n:
            df = df[
                df["symbol"].isin(np.random.RandomState(42).choice(st, n, False))
            ].copy()
    logger.info("Sample: %d rows, %d stocks", len(df), df["symbol"].nunique())
    logger.info("dim28...")
    df = FE.dim28_sector_index(df)
    df = df.sort_values(["symbol", "date"])
    for h in H:
        df[f"_r{h}"] = df.groupby("symbol")["close"].shift(-h) / df["close"] - 1
    return df


def ic_eval(df, cols, lc):
    out = []
    for c in cols:
        if c not in df.columns:
            out.append(
                {
                    "f": c,
                    "ic": np.nan,
                    "icir": np.nan,
                    "t": np.nan,
                    "n": 0,
                    "cov": np.nan,
                }
            )
            continue
        cv = df[c].notna().mean()
        ics = []
        for _, g in df.groupby("date"):
            v = g[[c, lc]].dropna()
            if len(v) < 10:
                continue
            try:
                r, _ = spearmanr(v[c], v[lc])
                if not np.isnan(r):
                    ics.append(r)
            except Exception:
                pass
        nn = len(ics)
        if nn < 20:
            out.append({"f": c, "ic": 0.0, "icir": 0.0, "t": 0.0, "n": nn, "cov": cv})
            continue
        a = np.array(ics)
        m = a.mean()
        s = a.std(ddof=1)
        out.append(
            {
                "f": c,
                "ic": round(m, 5),
                "icir": round(m / s if s > 0 else 0, 4),
                "t": round(m / (s / np.sqrt(nn)) if s > 0 else 0, 2),
                "n": nn,
                "cov": round(cv, 4),
            }
        )
    return pd.DataFrame(out).rename(columns={"f": "feature"})


def lgb_eval(df, cols, lc):
    av = [c for c in cols if c in df.columns and df[c].notna().mean() > 0.5]
    if not av:
        return pd.DataFrame(), 0.0
    sub = df[av + [lc, "date"]].dropna(subset=[lc])
    dt = sorted(sub["date"].unique())
    sp = int(len(dt) * 0.8)
    tr = sub[sub["date"].isin(dt[:sp])].copy()
    te = sub[sub["date"].isin(dt[sp:])].copy()
    if len(tr) < 100 or len(te) < 50:
        return pd.DataFrame(), 0.0
    for c in av:
        tr[c] = tr[c].replace([np.inf, -np.inf], np.nan)
        te[c] = te[c].replace([np.inf, -np.inf], np.nan)
    tr = tr.dropna(subset=av)
    te = te.dropna(subset=av)
    m = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(tr[av], tr[lc])
    imp = pd.DataFrame(
        {"feature": av, "g": m.booster_.feature_importance(importance_type="gain")}
    ).sort_values("g", ascending=False)
    imp["gain_pct"] = (imp["g"] / imp["g"].sum() * 100).round(2)
    imp = imp.drop(columns=["g"])
    te["p"] = m.predict(te[av])
    ics = []
    for _, g in te.groupby("date"):
        if len(g) >= 10:
            r, _ = spearmanr(g["p"], g[lc])
            if not np.isnan(r):
                ics.append(r)
    a = np.array(ics)
    pir = float(a.mean() / a.std()) if a.std() > 0 else 0.0
    logger.info("LGB[%s]: ICIR=%.4f %d/%d", lc, pir, len(tr), len(te))
    return imp, pir


def cclust(df, cols, th=0.85):
    av = [c for c in cols if c in df.columns and df[c].notna().mean() > 0.5]
    if not av:
        return {}
    s = df[av].sample(min(10000, len(df)), random_state=42)
    cr = s.corr(method="spearman").abs()
    asg = {}
    cid = 0
    for c in av:
        if c in asg:
            continue
        asg[c] = cid
        for o in av:
            if o not in asg and cr.loc[c, o] > th:
                asg[o] = cid
        cid += 1
    return asg


def main():
    df = load_df()
    logger.info("SW cols: %d/%d", sum(c in df.columns for c in SC), len(SC))
    cl = cclust(df, SC)
    R = {}
    for h in H:
        lc = f"_r{h}"
        logger.info("=== T+%d ===", h)
        ic = ic_eval(df, SC, lc)
        imp, pir = lgb_eval(df, SC, lc)
        r = ic.copy()
        if len(imp) > 0:
            r = r.merge(imp, on="feature", how="left")
        r["cl"] = r["feature"].map(cl).fillna(-1).astype(int)
        r["a"] = r["icir"].abs()
        r = r.sort_values("a", ascending=False).drop(columns=["a"])
        R[h] = (r, pir)

    print()
    print("=" * 120)
    print("  SW FEATURE EVAL -- MULTI-HORIZON (T+1 / T+3 / T+5)")
    print("=" * 120)
    print(f"  Sample: {df['symbol'].nunique()} stocks, {df['date'].nunique()} days")
    print()
    print(f"  {'Hor':>5s} {'Portfolio ICIR':>15s}")
    print(f"  {'-' * 5} {'-' * 15}")
    for h in H:
        print(f"  T+{h:<3d} {R[h][1]:>+15.4f}")
    print()

    b = R[1][0][["feature", "cov", "cl"]].copy()
    for h in H:
        r = R[h][0].set_index("feature")
        b[f"ic{h}"] = r.reindex(b["feature"])["ic"].values
        b[f"ir{h}"] = r.reindex(b["feature"])["icir"].values
        b[f"t{h}"] = r.reindex(b["feature"])["t"].values
        b[f"g{h}"] = r.reindex(b["feature"])["gain_pct"].values
    b["best"] = b[[f"ir{h}" for h in H]].abs().max(axis=1)
    b = b.sort_values("best", ascending=False).drop(columns=["best"])

    print(
        f"  {'Feature':<32s} {'Cov':>5s} | {'IC':>8s} {'ICIR':>8s} {'t':>6s} {'Gain':>5s} | {'IC':>8s} {'ICIR':>8s} {'t':>6s} {'Gain':>5s} | {'IC':>8s} {'ICIR':>8s} {'t':>6s} {'Gain':>5s} | Cl"
    )
    print(f"  {'-' * 32} {'-' * 5} | {'-' * 33} | {'-' * 33} | {'-' * 33} | --")
    print(f"  {'':32s} {'':5s} | {'T+1':^33s} | {'T+3':^33s} | {'T+5':^33s} |")
    print(f"  {'-' * 32} {'-' * 5} | {'-' * 33} | {'-' * 33} | {'-' * 33} | --")

    for _, row in b.iterrows():

        def f(v, fmt="+.4f"):
            return f"{v:{fmt}}" if pd.notna(v) else "  N/A"

        def g(v):
            return f"{v:4.1f}%" if pd.notna(v) else " N/A"

        print(
            f"  {row['feature']:<32s} {row['cov']:.3f} | "
            f"{f(row['ic1'])} {f(row['ir1'])} {f(row['t1'], '+.1f')} {g(row['g1'])} | "
            f"{f(row['ic3'])} {f(row['ir3'])} {f(row['t3'], '+.1f')} {g(row['g3'])} | "
            f"{f(row['ic5'])} {f(row['ir5'])} {f(row['t5'], '+.1f')} {g(row['g5'])} | "
            f"{row['cl']:2d}"
        )

    print()
    print("=" * 90)
    print("  SUMMARY PER HORIZON:")
    for h in H:
        r = R[h][0]
        s = r[r["icir"].abs() >= 0.30]
        m = r[(r["icir"].abs() >= 0.15) & (r["icir"].abs() < 0.30)]
        w = r[(r["icir"].abs() >= 0.05) & (r["icir"].abs() < 0.15)]
        ns = r[r["icir"].abs() < 0.05]
        print(
            f"    T+{h}: STRONG={len(s):2d}  MOD={len(m):2d}  WEAK={len(w):2d}  NOISE={len(ns):2d}  PF_ICIR={R[h][1]:+.4f}"
        )

    print()
    print("  TOP 5 PER HORIZON:")
    for h in H:
        print(f"    T+{h}:")
        for _, row in R[h][0].head(5).iterrows():
            gp = (
                f"{row.get('gain_pct', 0):.1f}%"
                if pd.notna(row.get("gain_pct", np.nan))
                else "N/A"
            )
            print(f"      {row['feature']:<32s} ICIR={row['icir']:+.4f}  gain={gp}")

    na = b.copy()
    for h in H:
        na = na[na[f"ir{h}"].abs() < 0.05]
    print()
    print(f"  DROP (noise ALL horizons): {len(na)}")
    for _, r in na.iterrows():
        print(
            f"    {r['feature']:<32s} T1={r['ir1']:+.4f}  T3={r['ir3']:+.4f}  T5={r['ir5']:+.4f}"
        )
    print()

    out_dir = data_others_path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir / "sw_feature_eval_multihorizon.csv")
    b.to_csv(out, index=False)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
