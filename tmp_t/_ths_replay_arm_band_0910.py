# -*- coding: utf-8 -*-
"""0910 离线回放 (初步, 样本<40日 → 只出方向性读数, 不达接线门槛):
① 5d期限倒挂臂: mag5顶带(≥q95) & prob5≥0.5 & mag10≤0 → FWD5/FWD10 vs 池均值
② top-30带质量: 交付键排名 1-15 vs 16-30 vs 31-50 带的 FWD 衰减 (两线)
闸准入协议: ≥40日+双半窗稳+误杀富集≤1.5x — 本样本~22日, 双半窗仅作粗检
FWD 用面板 pctChg 复利 (qfq污染安全). WORM: 结果带日期后缀.
"""
import os, glob
import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")

LST = r"D:/AMINQT/Daily Operation/STOCK LIST"
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_replay_arm_band_20260910.txt"
pd.set_option("display.width", 250)
rng_log = []

def pick(files):
    """每 trade date 取一个文件: 优先带 __M 的生产模块 tag, 同日多份取 M 最新."""
    best = {}
    for f in files:
        b = os.path.basename(f)
        d = b.split("__")[0].replace("legacy_preds_raw_", "").replace("parallel_preds_raw_", "")
        has_m = "__M" in b
        key = (has_m, b)
        if d not in best or key > best[d][0]:
            best[d] = (key, f)
    return {d: f for d, (k, f) in best.items()}

leg_f = pick(glob.glob(f"{LST}/legacy_preds_raw_2026*.csv"))
par_f = pick(glob.glob(f"{LST}/parallel_preds_raw_2026*.csv"))
# 只留两线都存在且 >=20260806 的常规日 (0720 是回放系, 模型未来信息污染, 剔)
dates = sorted(set(leg_f) & set(par_f))
dates = [d for d in dates if d >= "20260806"]
print(f"replay dates n={len(dates)}: {dates[0]}..{dates[-1]}")

# ---- 面板前向收益 (数据校验: |pctChg|>22 = 脏行, 整窗判NaN, 计数上报) ----
p = pd.read_parquet(PNL, columns=["symbol", "date", "pctChg"])
p["date"] = pd.to_datetime(p["date"])
p = p.sort_values(["symbol", "date"])
bad = p["pctChg"].abs() > 22  # 涨跌停物理界 (双创20%+余量); 超界=数据错误
n_bad = int(bad.sum())
g = p.groupby("symbol")["pctChg"]
pct_ok = p["pctChg"].where(~bad, np.nan)
def win_ret(k):
    comps = [g.shift(-i) for i in range(1, k + 1)]
    mat = pd.concat(comps, axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    return ((1 + mat / 100).prod(axis=1) - 1).where(valid)
p["fwd5"] = win_ret(5)
p["fwd10"] = win_ret(10)
p["dstr"] = p["date"].dt.strftime("%Y%m%d")
fwd = p[["symbol", "dstr", "fwd5", "fwd10"]]
print(f"[data-quality] pctChg |x|>22 脏行: {n_bad} / {len(p)} ({n_bad/len(p):.3%}) — 已整窗剔除")

# ---- 逐日读 raw preds ----
rows_arm, rows_band = [], []
for dstr in dates:
    try:
        par = pd.read_csv(par_f[dstr], dtype={"symbol": str})
        leg = pd.read_csv(leg_f[dstr], dtype={"symbol": str})
    except Exception as e:
        rng_log.append(f"{dstr} read fail {e}"); continue
    for name, df, keycol, mag5c, prob5c, mag10c in (
        ("PAR-dual", par[par["board"] != "main"].copy(), "score", "pred_mag_5d", "pred_prob_5d", "pred_mag_10d"),
        ("PAR-main", par[par["board"] == "main"].copy(), "score", "pred_mag_5d", "pred_prob_5d", "pred_mag_10d"),
        ("LEG-all", leg.copy(), "pred_ret_10d", "pred_ret_5d", "prob_up_5d", "pred_ret_10d"),
    ):
        need = [keycol, mag5c, prob5c, mag10c]
        if len(df) < 300 or any(c not in df.columns for c in need):
            rng_log.append(f"{dstr} {name} skip: cols={ [c for c in need if c not in df.columns]}")
            continue
        df["rk_key"] = df.groupby("board")[keycol].rank(ascending=False, method="first") if "board" in df else np.arange(1, len(df) + 1)
        df["rp5"] = df[mag5c].rank(pct=True)
        df["rk10"] = df[mag10c].rank(ascending=False, method="first")
        df["dstr"] = dstr
        # ① 倒挂臂: mag5 顶带 + prob5 看多 + mag10 平/负 (键位深)
        arm = df[(df["rp5"] >= 0.95) & (df[prob5c] >= 0.5) & (df[mag10c] <= 0)].copy()
        arm["lane"] = name
        arm["missed_by_top30"] = arm["rk_key"] > 30
        rows_arm.append(arm[["dstr", "lane", "symbol", "rp5", "rk_key", "missed_by_top30", mag5c, mag10c, prob5c]])
        # ② 带质量
        for lo, hi in ((1, 15), (16, 30), (31, 50)):
            band = df[(df["rk_key"] >= lo) & (df["rk_key"] <= hi)].copy()
            band["lane"], band["band"] = name, f"{lo}-{hi}"
            rows_band.append(band[["dstr", "lane", "band", "symbol"]])

arm = pd.concat(rows_arm, ignore_index=True) if rows_arm else pd.DataFrame()
band = pd.concat(rows_band, ignore_index=True) if rows_band else pd.DataFrame()
arm = arm.merge(fwd, on=["symbol", "dstr"], how="left")
band = band.merge(fwd, on=["symbol", "dstr"], how="left")

out_lines = []
def P(s=""):
    print(s); out_lines.append(str(s))

P("=" * 70)
P("① 5d期限倒挂臂 (mag5≥q95 & prob5≥0.5 & mag10≤0) — 初步读数")
P("=" * 70)
P(f"窗口 {dates[0]}..{dates[-1]}  n日={len(dates)} (<40 → 不达接线门槛)")
for lane, sub in arm.groupby("lane"):
    pool_fwd5 = band[band["lane"] == lane]["fwd5"]
    P(f"\n--- {lane}: 臂样本 n={sub['fwd5'].notna().sum()} (跨日去重股次)")
    P(f"    臂 FWD5 mean={sub['fwd5'].mean():+.4f} med={sub['fwd5'].median():+.4f} hit5={ (sub['fwd5']>=0.05).mean():.1%} hit10={(sub['fwd5']>=0.10).mean():.1%}")
    P(f"    臂 FWD10 mean={sub['fwd10'].mean():+.4f} hit10_10d={(sub['fwd10']>=0.10).mean():.1%}")
    if len(pool_fwd5):
        P(f"    池基线 FWD5 mean={pool_fwd5.mean():+.4f} → 臂超额 {sub['fwd5'].mean()-pool_fwd5.mean():+.4f}")
    # 双半窗粗检
    ds = sorted(sub["dstr"].unique()); h = len(ds) // 2
    for tag, dd in (("H1", ds[:h]), ("H2", ds[h:])):
        s2 = sub[sub["dstr"].isin(dd)]
        if s2["fwd5"].notna().sum() >= 5:
            P(f"    {tag}({len(dd)}日) FWD5 mean={s2['fwd5'].mean():+.4f} n={s2['fwd5'].notna().sum()}")
    m = sub[sub["missed_by_top30"]]
    if len(m):
        P(f"    被top30漏掉部分: n={m['fwd5'].notna().sum()} FWD5 mean={m['fwd5'].mean():+.4f} hit10={ (m['fwd5']>=0.10).mean():.1%}")
P("\n臂内股次明细 (FWD5≥10% 的, 全窗):")
w = arm[arm["fwd5"] >= 0.10].sort_values("fwd5", ascending=False)
P(w[["dstr", "lane", "symbol", "rp5", "rk_key", "missed_by_top30", "fwd5"]].round(3).to_string(index=False) if len(w) else "  (无)")

P("\n" + "=" * 70)
P("② top-30 带质量 — 交付键排名带 FWD 衰减 (两线)")
P("=" * 70)
gb = band.groupby(["lane", "band"]).agg(
    n=("fwd5", "size"), fwd5=("fwd5", "mean"), fwd10=("fwd10", "mean"),
    hit10_5d=("fwd5", lambda x: (x >= 0.10).mean()))
P(gb.round(4).to_string())

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines))
print(f"\nsaved -> {OUT}")
