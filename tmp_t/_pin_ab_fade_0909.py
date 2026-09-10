# -*- coding: utf-8 -*-
"""
pin 特征集受控 A/B: fade_score 入 pin 裁决 (2026-09-09 行情切片复测)
协议依据: feature_selector.py pin 冻结注释 — "更新 pin = 显式动作 (新特征家族落地),
人工核对 + 250d replay 通过才写新 pin". 人工核对已完成 (模型特征空间78列对照残差IC -0.025 t=-16 全行情带双半稳, 对拍单测齐).
本脚本 = 250d replay: A=pin 原样 vs B=pin+fade_score, main/dual 两板,
OOS=末250交易日 (train=面板其余), LGBM 参数对齐 _abc_test_harness 先例.
预注册判据 (跑前冻结, 防事后拟合):
  通过 := 两板 × 双label 共4组中 >=3 组满足
          [B-A 的 ICIR 改善>0 或 Top10Sharpe 改善>0, 且另一项恶化<=10%]
  通过 → 写新 pin (backup 旧文件); 不通过 → pin 不动.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tmp_t" else HERE
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts._run_guard import find_conflicts

conflicts = find_conflicts()
if conflicts:
    print(f"[run-guard] 重活进程冲突, 拒启: {conflicts}", flush=True)
    sys.exit(2)

import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

np.random.seed(42)

from scipy.stats import spearmanr

import lightgbm as lgb

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
CSI300 = "data/_csi300_members.parquet"
REGDIR = r"D:\AMINQT\DATA OTHERS\factor_registry"
OUT = os.path.join(HERE, "_pin_ab_fade_0909_result.json")
NEWCOLS = ["fade_score"]
OOS_DAYS = 250
LABELS = ["label_pm_10d_net", "label_pm_1d_net"]

t0 = time.time()
dates_all = sorted(
    pq.read_table(PANEL, columns=["date"]).to_pandas()["date"].unique()
)
cut = dates_all[-(OOS_DAYS + 40)] - pd.Timedelta(days=365 * 2)  # train≈2y + OOS 250d + 预热
panel = pq.read_table(PANEL, filters=[("date", ">=", cut)]).to_pandas()
print(f"panel rows={len(panel):,} {panel['date'].min().date()}..{panel['date'].max().date()}",
      flush=True)

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame
import tempfile
import shutil


def build_board(board_df, use_xrank):
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(board_df)
    df = dual_df if len(dual_df) > len(main_df) else main_df
    reg_dir = tempfile.mkdtemp()
    reg = FeatureRegistry(path=os.path.join(reg_dir, "feature_registry.json"))
    sample = df.groupby("symbol", group_keys=False).apply(
        lambda g: g.head(min(30, len(g)))).reset_index(drop=True)
    reg._seed(sample)
    fe = FeatureEngineV35()
    out = prepare_board_frame(df, fe, cross_sectional_rank=use_xrank, registry=reg)
    shutil.rmtree(reg_dir)
    return out


def run_arm(df, feats, label):
    dates = sorted(df["date"].unique())
    oos_start = dates[-OOS_DAYS]
    tr = df[df["date"] < oos_start].dropna(subset=[label])
    te = df[df["date"] >= oos_start].dropna(subset=[label])
    X_tr, y_tr = tr[feats].fillna(0), tr[label]
    X_te = te[feats].fillna(0)
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
                          learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y_tr)
    te = te.copy()
    te["pred"] = m.predict(X_te)
    ics = [spearmanr(g["pred"], g[label])[0] for _, g in te.groupby("date") if len(g) >= 10]
    a = np.array([x for x in ics if not np.isnan(x)])
    tops = pd.concat([g.nlargest(10, "pred") for _, g in te.groupby("date")])
    r = tops[label].dropna()
    return {
        "n_train": len(tr), "n_oos": len(te), "oos_days": len(a),
        "ic": float(a.mean()), "icir": float(a.mean() / a.std()) if a.std() > 0 else 0.0,
        "top10_mean": float(r.mean()), "top10_win": float((r > 0).mean()),
        "top10_sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
    }


# ---------- main ----------
csi = set(pd.read_parquet(CSI300)["code"].astype(str).str.zfill(6))
main_panel = panel[panel["symbol"].isin(csi & set(panel["symbol"].unique()))].copy()
print(f"\n[main] CSI300 rows={len(main_panel):,} stocks={main_panel['symbol'].nunique()}",
      flush=True)
t1 = time.time()
df_main = build_board(main_panel, use_xrank=False)
print(f"[main] build {time.time()-t1:.0f}s cols={len(df_main.columns)} "
      f"(newcols: {[c for c in NEWCOLS if c in df_main.columns]})", flush=True)

dual_panel = panel[panel["board"].isin(["GEM", "STAR"])].copy()
del panel
gc.collect()
if dual_panel["symbol"].nunique() > 300:
    st = np.random.choice(dual_panel["symbol"].unique(), 300, replace=False)
    dual_panel = dual_panel[dual_panel["symbol"].isin(st)]
print(f"[dual] rows={len(dual_panel):,} stocks={dual_panel['symbol'].nunique()}", flush=True)
t1 = time.time()
df_dual = build_board(dual_panel, use_xrank=True)
print(f"[dual] build {time.time()-t1:.0f}s cols={len(df_dual.columns)}", flush=True)

results = {}
for board, dfb, pinf in [("main", df_main, "selected_main_pinned.json"),
                         ("dual", df_dual, "selected_dual_pinned.json")]:
    pin = json.load(open(os.path.join(REGDIR, pinf), encoding="utf-8"))
    pin_feats = pin["features"] if isinstance(pin, dict) else pin
    avail = set(dfb.columns)
    featsA = [c for c in pin_feats if c in avail]
    featsB = featsA + [c for c in NEWCOLS if c in avail and c not in featsA]
    print(f"\n[{board}] pin={len(pin_feats)} availA={len(featsA)} B=+{len(featsB)-len(featsA)}",
          flush=True)
    for lab in LABELS:
        if lab not in dfb.columns:
            print(f"[{board}] {lab} 不存在, 跳过", flush=True)
            continue
        rA = run_arm(dfb, featsA, lab)
        rB = run_arm(dfb, featsB, lab)
        results[f"{board}__{lab}"] = {"A": rA, "B": rB}
        print(f"[{board}][{lab}] A: IC={rA['ic']:+.4f} ICIR={rA['icir']:+.3f} "
              f"T10μ={rA['top10_mean']:+.4f} Sharpe={rA['top10_sharpe']:+.2f} win={rA['top10_win']:.0%}",
              flush=True)
        print(f"[{board}][{lab}] B: IC={rB['ic']:+.4f} ICIR={rB['icir']:+.3f} "
              f"T10μ={rB['top10_mean']:+.4f} Sharpe={rB['top10_sharpe']:+.2f} win={rB['top10_win']:.0%}",
              flush=True)
        print(f"[{board}][{lab}] Δ: ICIR={rB['icir']-rA['icir']:+.4f} "
              f"Sharpe={rB['top10_sharpe']-rA['top10_sharpe']:+.3f}", flush=True)

# ---------- 预注册终判 ----------
print("\n== 预注册判据: 4组中>=3组 [ICIR或Sharpe改善>0 且另一项恶化<=10%] ==", flush=True)
passes = []
for k, v in results.items():
    A, B = v["A"], v["B"]
    icir_ok = (B["icir"] - A["icir"]) > 0
    sh_ok = (B["top10_sharpe"] - A["top10_sharpe"]) > 0
    icir_not_bad = (B["icir"] - A["icir"]) >= -abs(A["icir"]) * 0.10
    sh_not_bad = (B["top10_sharpe"] - A["top10_sharpe"]) >= -abs(A["top10_sharpe"]) * 0.10
    ok = (icir_ok and sh_not_bad) or (sh_ok and icir_not_bad)
    passes.append(ok)
    print(f"  {k}: {'PASS' if ok else 'FAIL'} (icir_ok={icir_ok} sh_ok={sh_ok} "
          f"icir_not_bad={icir_not_bad} sh_not_bad={sh_not_bad})", flush=True)
n_pass = sum(passes)
verdict = n_pass >= 3
print(f"\nVERDICT: {'WRITE-PIN' if verdict else 'KEEP-PIN'} ({n_pass}/4 通过)", flush=True)
json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"saved {OUT}  elapsed={time.time()-t0:.0f}s", flush=True)
