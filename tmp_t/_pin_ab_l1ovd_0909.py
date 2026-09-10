# -*- coding: utf-8 -*-
"""
pin 特征集受控 A/B (三臂): dim20 三特征 + ovd 两特征 入 pin 裁决 (2026-09-09).
协议依据: 用户指令 — 特征达线 (|t|>3 双半窗同号 残差显著) 立即受控 A/B, 不等季度重选.
模板: tmp_t/_pin_ab_cl_0909.py (close_vs_low_ma5/ma20 A/B 先例, 0/4 FAIL=KEEP-PIN).

三臂:
  A  = pin 原样 (selected_{main,dual}_pinned.json features ∩ 可用列)
  B1 = A + dim20 三特征: days_since_board_break / vol_decay_ratio / quiet_drift
       (引擎 feature_engine_v35.py dim20_short_horizon 已注入, build 后自动在列)
  B2 = B1 + ovd 两特征: ovd_wdist / ovd_15p
       (cyq_ext.py eae1f343 已注入但生产面板无这两列 → 本脚本从 PANEL 现算;
        向量化实现复用 tmp_chip/_full_true150.py 已验证 12/12 与生产一致;
        生产口径: ovd_wdist=Σ(x·d)/Σ(x) for d>0 (上方筹码距离加权均值, 无上方=0.0),
        ovd_15p=Σ(x for d>=0.15)/Σ(x_total); turnover_rate NaN 按 0 处理不带污染)

板 × label: main=CSI300 use_xrank=False / dual=GEM/STAR 抽样300 use_xrank=True,
双 label (label_pm_10d_net, label_pm_1d_net) = 4 组 × 3 臂 = 12 次训练.
OOS=末250交易日, train=面板其余 (cut 公式同先例), LGBM 参数逐字对齐先例.

预注册判据 (跑前冻结, 防事后拟合; 公式与 _pin_ab_cl_0909.py 完全一致):
  单组通过 := [X-A 的 ICIR 改善>0 或 Top10Sharpe 改善>0, 且另一项恶化<=10% (相对|A|)]
  主判据 (B2 写 pin 候选): 4 组中 >=3 组通过 (B2 vs A)
  归因判据: B1 vs A 与 B2 vs B1 各自数 PASS —
    B2 主判据 FAIL 且 B1-A >=3/4 → 结论"拆包复议 dim20 三";
    B2 PASS 但 B2-B1 全 0 PASS → 注明增量全来自 dim20.
结果只报数不写 pin (写 pin 需回主会话人工核对后另做).
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
OUT = os.path.join(HERE, "_pin_ab_l1ovd_0909_result.json")
NEWCOLS_B1 = ["days_since_board_break", "vol_decay_ratio", "quiet_drift"]
NEWCOLS_B2 = ["ovd_wdist", "ovd_15p"]
OOS_DAYS = 250
LABELS = ["label_pm_10d_net", "label_pm_1d_net"]

# ovd 计算常量 (cyq_calculator 生产同源)
FACTOR = 150
RANGE_DAYS = 120
AR = np.arange(FACTOR)

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
    del m
    gc.collect()
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


def compute_ovd_board(symbols, cut_read):
    """150档 CYQ 分布 → ovd_wdist/ovd_15p (tmp_chip/_full_true150.py 已验证向量化副本,
    生产 cyq_ext eae1f343 同口径: 逐档精确距离; 无上方筹码 ovd_wdist=0.0)."""
    tbl = pq.read_table(
        PANEL,
        columns=["symbol", "date", "open", "high", "low", "close", "turnover_rate"],
        filters=[("symbol", "in", list(symbols)), ("date", ">=", cut_read)],
    )
    dfl = tbl.to_pandas()
    del tbl
    dfl["date"] = pd.to_datetime(dfl["date"])
    rows = []
    tc = time.time()
    n_done = 0
    for sym, g in dfl.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        g = g.dropna(subset=["open", "high", "low", "close"])
        n = len(g)
        if n < RANGE_DAYS:
            continue
        o, hi, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        # turnover_rate NaN 按 0 处理 (生产存量列 min(1.0,nan) 污染不带进实验侧)
        hsl = np.clip(np.nan_to_num(g["turnover_rate"].to_numpy(float)) / 100.0, 0.0, 1.0)
        wdist = np.full(n, np.nan)
        p15 = np.full(n, np.nan)
        for T in range(RANGE_DAYS - 1, n):
            s0 = T - RANGE_DAYS + 1
            hh, ll, oo, cc = hi[s0:T + 1], lo[s0:T + 1], o[s0:T + 1], c[s0:T + 1]
            hw = hsl[s0:T + 1]
            cT = c[T]
            if not np.isfinite(cT) or cT <= 0:
                continue
            maxp, minp = float(hh.max()), float(ll.min())
            acc = max(0.01, (maxp - minp) / (FACTOR - 1))
            avg = (oo + hh + ll + cc) / 4.0
            grid = minp + acc * AR
            gr = grid[None, :]

            # 三角形权重 (生产算法向量化副本)
            flat = hh <= ll
            rngv = np.maximum(hh - ll, 1e-12)
            left = gr <= avg[:, None]
            num = np.where(left, gr - ll[:, None], hh[:, None] - gr)
            den = np.where(left, (avg - ll)[:, None], (hh - avg)[:, None])
            den = np.where(np.abs(den) < 1e-12, 1.0, den)
            w = num / den
            mask = (gr >= ll[:, None]) & (gr <= hh[:, None])
            density = np.where(flat, float(FACTOR - 1), 2.0 / rngv)
            w = np.where(mask, w, 0.0) * (density * hw)[:, None]
            if flat.any():
                w[flat] = 0.0
                ab = np.floor((avg - minp) / acc).astype(int)
                fi = np.nonzero(flat & (ab >= 0) & (ab < FACTOR))[0]
                w[fi, ab[fi]] += (FACTOR - 1) * hw[fi] / 2.0

            # 窗内倒序累积衰减 (与生产顺序衰减逐日等价, 且无全局下溢)
            D = np.empty(len(hw))
            D[:-1] = np.cumprod((1.0 - hw)[::-1])[-2::-1]
            D[-1] = 1.0
            xdata = np.maximum((w * D[:, None]).sum(axis=0), 0.0)
            tot = xdata.sum()
            if tot <= 1e-12:
                continue
            d = (grid - cT) / cT
            abv = d > 0.0
            mab = float(xdata[abv].sum())
            wdist[T] = float((xdata[abv] * d[abv]).sum() / mab) if mab > 1e-12 else 0.0
            p15[T] = float(xdata[d >= 0.15 - 1e-9].sum() / tot)
        rows.append(pd.DataFrame({"symbol": sym, "date": g["date"],
                                  "ovd_wdist": wdist, "ovd_15p": p15}))
        n_done += 1
        if n_done % 50 == 0:
            el = time.time() - tc
            print(f"    [ovd] {n_done}/{dfl['symbol'].nunique()} {el:.0f}s "
                  f"({el / n_done:.2f}s/stock)", flush=True)
    out = pd.concat(rows, ignore_index=True)
    print(f"  [ovd] stocks={out['symbol'].nunique()} rows={len(out):,} "
          f"({time.time() - tc:.0f}s)", flush=True)
    return out


def merge_ovd(dfb, symbols):
    cut_read = cut - pd.Timedelta(days=300)  # 120td 滚动窗预热 (≈170交易日)
    ovd = compute_ovd_board(symbols, cut_read)
    # 防御: build 输出若已带 ovd 列 (registry _seed 捡列), 先删再 merge 保持列名干净
    dup = [c for c in NEWCOLS_B2 if c in dfb.columns]
    if dup:
        print(f"  [ovd] drop pre-existing cols from build: {dup}", flush=True)
        dfb = dfb.drop(columns=dup)
    before = len(dfb)
    merged = dfb.merge(ovd, on=["symbol", "date"], how="left")
    assert len(merged) == before, "merge 行数变化!"
    cov = {
        "ovd_wdist": float(merged["ovd_wdist"].notna().mean()),
        "ovd_15p": float(merged["ovd_15p"].notna().mean()),
    }
    print(f"  [ovd] merge coverage: wdist={cov['ovd_wdist']:.1%} "
          f"15p={cov['ovd_15p']:.1%}", flush=True)
    return merged, cov


def pass_pair(A, B):
    """预注册单组判据 (与 _pin_ab_cl_0909.py 完全一致)."""
    icir_ok = (B["icir"] - A["icir"]) > 0
    sh_ok = (B["top10_sharpe"] - A["top10_sharpe"]) > 0
    icir_not_bad = (B["icir"] - A["icir"]) >= -abs(A["icir"]) * 0.10
    sh_not_bad = (B["top10_sharpe"] - A["top10_sharpe"]) >= -abs(A["top10_sharpe"]) * 0.10
    ok = (icir_ok and sh_not_bad) or (sh_ok and icir_not_bad)
    detail = {"icir_ok": bool(icir_ok), "sh_ok": bool(sh_ok),
              "icir_not_bad": bool(icir_not_bad), "sh_not_bad": bool(sh_not_bad)}
    return bool(ok), detail


# ---------- main ----------
results = {}
coverage = {}
pin_counts = {}
csi = set(pd.read_parquet(CSI300)["code"].astype(str).str.zfill(6))
main_panel = panel[panel["symbol"].isin(csi & set(panel["symbol"].unique()))].copy()
print(f"\n[main] CSI300 rows={len(main_panel):,} stocks={main_panel['symbol'].nunique()}",
      flush=True)
t1 = time.time()
df_main = build_board(main_panel, use_xrank=False)
print(f"[main] build {time.time()-t1:.0f}s cols={len(df_main.columns)} "
      f"(dim20new: {[c for c in NEWCOLS_B1 if c in df_main.columns]})", flush=True)
missing = [c for c in NEWCOLS_B1 if c not in df_main.columns]
if missing:
    print(f"[FATAL] dim20 三特征不在引擎输出列: {missing} = 接线问题, 停", flush=True)
    sys.exit(3)

dual_panel = panel[panel["board"].isin(["GEM", "STAR"])].copy()
del panel
gc.collect()
if dual_panel["symbol"].nunique() > 300:
    st = np.random.choice(dual_panel["symbol"].unique(), 300, replace=False)
    dual_panel = dual_panel[dual_panel["symbol"].isin(st)]
print(f"[dual] rows={len(dual_panel):,} stocks={dual_panel['symbol'].nunique()}", flush=True)

# ---- main 板: ovd 现算 + 三臂 (build 后即处理, 不与 dual 同时持有) ----
t1 = time.time()
df_main, cov_m = merge_ovd(df_main, df_main["symbol"].unique())
coverage["main"] = cov_m
print(f"[main] ovd total {time.time()-t1:.0f}s", flush=True)

for board, dfb, pinf in [("main", df_main, "selected_main_pinned.json")]:
    pin = json.load(open(os.path.join(REGDIR, pinf), encoding="utf-8"))
    pin_feats = pin["features"] if isinstance(pin, dict) else pin
    pin_counts[board] = len(pin_feats)
    avail = set(dfb.columns)
    featsA = [c for c in pin_feats if c in avail]
    featsB1 = featsA + [c for c in NEWCOLS_B1 if c in avail and c not in featsA]
    featsB2 = featsB1 + [c for c in NEWCOLS_B2 if c in avail and c not in featsB1]
    print(f"\n[{board}] pin={len(pin_feats)} availA={len(featsA)} "
          f"B1=+{len(featsB1)-len(featsA)} B2=+{len(featsB2)-len(featsB1)}", flush=True)
    for lab in LABELS:
        if lab not in dfb.columns:
            print(f"[{board}] {lab} 不存在, 跳过", flush=True)
            continue
        rA = run_arm(dfb, featsA, lab)
        rB1 = run_arm(dfb, featsB1, lab)
        rB2 = run_arm(dfb, featsB2, lab)
        results[f"{board}__{lab}"] = {"A": rA, "B1": rB1, "B2": rB2}
        for arm, r in [("A", rA), ("B1", rB1), ("B2", rB2)]:
            print(f"[{board}][{lab}] {arm}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} "
                  f"T10μ={r['top10_mean']:+.4f} Sharpe={r['top10_sharpe']:+.2f} "
                  f"win={r['top10_win']:.0%}", flush=True)
        print(f"[{board}][{lab}] ΔB1-A: ICIR={rB1['icir']-rA['icir']:+.4f} "
              f"Sharpe={rB1['top10_sharpe']-rA['top10_sharpe']:+.3f} | "
              f"ΔB2-A: ICIR={rB2['icir']-rA['icir']:+.4f} "
              f"Sharpe={rB2['top10_sharpe']-rA['top10_sharpe']:+.3f} | "
              f"ΔB2-B1: ICIR={rB2['icir']-rB1['icir']:+.4f} "
              f"Sharpe={rB2['top10_sharpe']-rB1['top10_sharpe']:+.3f}", flush=True)
        gc.collect()

del df_main, main_panel
gc.collect()

# ---- dual 板: build → ovd → 三臂 ----
t1 = time.time()
df_dual = build_board(dual_panel, use_xrank=True)
print(f"[dual] build {time.time()-t1:.0f}s cols={len(df_dual.columns)}", flush=True)
missing = [c for c in NEWCOLS_B1 if c not in df_dual.columns]
if missing:
    print(f"[FATAL] dim20 三特征不在 dual 引擎输出列: {missing} = 接线问题, 停", flush=True)
    sys.exit(3)

t1 = time.time()
df_dual, cov_d = merge_ovd(df_dual, df_dual["symbol"].unique())
coverage["dual"] = cov_d
print(f"[dual] ovd total {time.time()-t1:.0f}s", flush=True)

board, dfb, pinf = "dual", df_dual, "selected_dual_pinned.json"
pin = json.load(open(os.path.join(REGDIR, pinf), encoding="utf-8"))
pin_feats = pin["features"] if isinstance(pin, dict) else pin
pin_counts[board] = len(pin_feats)
avail = set(dfb.columns)
featsA = [c for c in pin_feats if c in avail]
featsB1 = featsA + [c for c in NEWCOLS_B1 if c in avail and c not in featsA]
featsB2 = featsB1 + [c for c in NEWCOLS_B2 if c in avail and c not in featsB1]
print(f"\n[{board}] pin={len(pin_feats)} availA={len(featsA)} "
      f"B1=+{len(featsB1)-len(featsA)} B2=+{len(featsB2)-len(featsB1)}", flush=True)
for lab in LABELS:
    if lab not in dfb.columns:
        print(f"[{board}] {lab} 不存在, 跳过", flush=True)
        continue
    rA = run_arm(dfb, featsA, lab)
    rB1 = run_arm(dfb, featsB1, lab)
    rB2 = run_arm(dfb, featsB2, lab)
    results[f"{board}__{lab}"] = {"A": rA, "B1": rB1, "B2": rB2}
    for arm, r in [("A", rA), ("B1", rB1), ("B2", rB2)]:
        print(f"[{board}][{lab}] {arm}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} "
              f"T10μ={r['top10_mean']:+.4f} Sharpe={r['top10_sharpe']:+.2f} "
              f"win={r['top10_win']:.0%}", flush=True)
    print(f"[{board}][{lab}] ΔB1-A: ICIR={rB1['icir']-rA['icir']:+.4f} "
          f"Sharpe={rB1['top10_sharpe']-rA['top10_sharpe']:+.3f} | "
          f"ΔB2-A: ICIR={rB2['icir']-rA['icir']:+.4f} "
          f"Sharpe={rB2['top10_sharpe']-rA['top10_sharpe']:+.3f} | "
          f"ΔB2-B1: ICIR={rB2['icir']-rB1['icir']:+.4f} "
          f"Sharpe={rB2['top10_sharpe']-rB1['top10_sharpe']:+.3f}", flush=True)
    gc.collect()

del df_dual, dual_panel
gc.collect()

# ---------- 预注册终判 ----------
print("\n== 预注册判据 (单组=ICIR或Sharpe改善>0 且另一项恶化<=10%) ==", flush=True)
detail_all = {}
passes = {"B2_vs_A": {}, "B1_vs_A": {}, "B2_vs_B1": {}}
for k, v in results.items():
    A, B1, B2 = v["A"], v["B1"], v["B2"]
    ok2, d2 = pass_pair(A, B2)
    ok1, d1 = pass_pair(A, B1)
    ok3, d3 = pass_pair(B1, B2)
    passes["B2_vs_A"][k] = ok2
    passes["B1_vs_A"][k] = ok1
    passes["B2_vs_B1"][k] = ok3
    detail_all[k] = {"B2_vs_A": d2, "B1_vs_A": d1, "B2_vs_B1": d3}
    print(f"  {k}: B2-A={'PASS' if ok2 else 'FAIL'} B1-A={'PASS' if ok1 else 'FAIL'} "
          f"B2-B1={'PASS' if ok3 else 'FAIL'}", flush=True)

n_main = sum(passes["B2_vs_A"].values())
n_b1 = sum(passes["B1_vs_A"].values())
n_b2b1 = sum(passes["B2_vs_B1"].values())
verdict_main = n_main >= 3
print(f"\n主判据 (B2 vs A): {'WRITE-PIN 候选' if verdict_main else 'KEEP-PIN'} "
      f"({n_main}/4)", flush=True)
print(f"归因: B1-A {n_b1}/4 | B2-B1 {n_b2b1}/4", flush=True)
if not verdict_main and n_b1 >= 3:
    print("结论备注: B2 FAIL 但 B1-A 强 → 拆包复议 dim20 三", flush=True)
if verdict_main and n_b2b1 == 0:
    print("结论备注: B2 PASS 但增量全来自 B1, ovd 无边际贡献", flush=True)
print(f"本脚本只报数不写 pin (写 pin 需主会话人工核对后另做)", flush=True)

json.dump({"results": results, "passes": passes, "pass_detail": detail_all,
           "coverage": coverage,
           "n_pass": {"B2_vs_A": n_main, "B1_vs_A": n_b1, "B2_vs_B1": n_b2b1},
           "verdict_main": verdict_main, "feat_counts": pin_counts},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"saved {OUT}  elapsed={time.time()-t0:.0f}s", flush=True)
