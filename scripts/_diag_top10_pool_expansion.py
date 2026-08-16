"""扩建宇宙对 TOP10 影响的实证: 旧池 vs 扩建池, 代理信号+oracle 双口径 (2026-08-15)."""
import glob

import numpy as np
import pandas as pd

ROOT = r"D:\AMINQT\AMINQT CODES\data"
DAILY_DIR = rf"{ROOT}\supply_cache\alt_data\daily"
BACKUP_PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet.backup_20260815_200014"
NEW_UNIVERSE = rf"{ROOT}\new_universe\new_symbols_20260815_020959.parquet"
WINDOW = 250
MIN_AMT = 5e7  # 读取预过滤 5000万

def board_of(s):
    s = str(s)
    if s.startswith(("60", "00", "01", "02")):
        return "main"
    if s.startswith(("30", "68")):
        return "dual"
    return "bse"

new = pd.read_parquet(NEW_UNIVERSE)
new_syms = set(new["symbol"])
old_syms = set(pd.read_parquet(BACKUP_PANEL, columns=["symbol"])["symbol"].unique())
# 次新 gate: 剔 2026 上市 <150 交易日 (实测 45 只)
new = new[pd.to_datetime(new["list_date"], format="%Y%m%d") < "2025-08-01"]
new_syms = set(new["symbol"])  # 有效新增
print(f"旧宇宙 {len(old_syms)} | 有效新增 {len(new_syms)} (剔45只2026次新)")

files = sorted(glob.glob(rf"{DAILY_DIR}\daily_*.parquet"))[-WINDOW:]
frames = [pd.read_parquet(f, columns=["symbol", "trade_date", "close", "pre_close", "amount"])
          for f in files]
df = pd.concat(frames, ignore_index=True)
df["amount"] = df["amount"] * 1000.0
df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df["grp"] = np.where(df["symbol"].isin(new_syms), "new", "old")
df["board"] = df["symbol"].map(board_of)

# 逐股: 动量代理分 (10d) 与 真实 T+10
df = df.sort_values(["symbol", "date"])
df["mom10"] = df.groupby("symbol")["close"].transform(lambda s: s / s.shift(10) - 1)
df["t10"] = df.groupby("symbol")["close"].transform(lambda s: s.shift(-10) / s - 1)

def run(board, pool_sel, label, top_n=10, filter_amt=True):
    """pool_sel: df 子集; 逐日 top-N by 代理分 → 真实 t10; 另算 oracle top-N."""
    sub = df[df["board"] == board]
    if filter_amt:
        sub = sub[sub["amount"] >= MIN_AMT]  # 读取预过滤, 新旧同口径
    if pool_sel == "old":
        sub = sub[sub["grp"] == "old"]
    # 逐日截面
    days = sub["date"].unique()
    rows = []
    for d in days:
        day = sub[sub["date"] == d].dropna(subset=["mom10", "t10"])
        if len(day) < top_n:
            continue
        prox = day.nlargest(top_n, "mom10")
        oracle = day.nlargest(top_n, "t10")
        rows.append({
            "date": d, "n": len(day),
            "proxy_t10": prox["t10"].mean(), "proxy_pos": (prox["t10"] > 0).mean(),
            "new_in_proxy": (prox["grp"] == "new").sum(),
            "oracle_t10": oracle["t10"].mean(),
        })
    r = pd.DataFrame(rows)
    return pd.Series({
        "label": label, "days": len(r),
        "avg_pool_n": r["n"].mean(),
        "proxy_t10_mean": r["proxy_t10"].mean(),
        "proxy_pos": r["proxy_pos"].mean(),
        "proxy_win_vs_baseline": (r["proxy_t10"] > r["oracle_t10"].mean() * 0 + 0).mean(),
        "new_share_in_proxy": r["new_in_proxy"].mean(),
        "oracle_t10_mean": r["oracle_t10"].mean(),
    })

results = []
for b in ["main", "dual"]:
    results.append(run(b, "old", f"{b} 旧池"))
    results.append(run(b, "exp", f"{b} 扩建池(旧+新,≥5000万)"))
res = pd.DataFrame(results)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n=== 代理信号 (10d动量) TOP10 → 真实T+10: 旧池 vs 扩建池 ===")
print(res.to_string(index=False))
res.to_csv(rf"{ROOT}\_diag_top10_pool_expansion.csv", index=False)

# 附加: 不加 5000万 过滤时 (全池扩建) 对比 — 看过滤的作用
print("\n=== 灵敏度: 扩建池 不加5000万过滤 ===")
for b in ["main", "dual"]:
    s = run(b, "exp", f"{b} 扩建池(不过滤)", filter_amt=False)
    print(s.to_string())
