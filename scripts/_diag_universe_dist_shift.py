"""旧宇宙 vs 新增1780只 分布对比 — 评估参数是否受宇宙扩建影响 (2026-08-15)."""

import glob

import numpy as np
import pandas as pd

ROOT = r"D:\AMINQT\AMINQT CODES\data"
DAILY_DIR = rf"{ROOT}\supply_cache\alt_data\daily"
BACKUP_PANEL = (
    r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet.backup_20260815_200014"
)
NEW_UNIVERSE = rf"{ROOT}\new_universe\new_symbols_20260815_020959.parquet"
WINDOW = 250  # 近250交易日


def board_of(s):
    s = str(s)
    if s.startswith(("60", "00", "01", "02")):
        return "main"
    if s.startswith(("30", "68")):
        return "dual"
    return "bse"


# ── 1. 宇宙清单 ──
new = pd.read_parquet(NEW_UNIVERSE)
new["board"] = new["symbol"].map(board_of)
new_syms = set(new["symbol"])
old = pd.read_parquet(BACKUP_PANEL, columns=["symbol"])
old_syms = set(old["symbol"].unique())
print(f"旧宇宙 {len(old_syms)} 只 | 新增 {len(new_syms)} 只")

# ── 2. 读近250交易日 daily (全市场) ──
files = sorted(glob.glob(rf"{DAILY_DIR}\daily_*.parquet"))[-WINDOW:]
frames = []
for f in files:
    d = pd.read_parquet(
        f,
        columns=["symbol", "trade_date", "close", "high", "low", "pre_close", "amount"],
    )
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["amount"] = df["amount"] * 1000.0  # 千元→元
df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
print(
    f"daily 载入 {df.shape[0]} 行, 日期 {df['date'].min().date()} ~ {df['date'].max().date()}"
)
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
df["grp"] = np.where(df["symbol"].isin(new_syms), "new", "old")
df["board"] = df["symbol"].map(board_of)

# ── 3. 逐股统计 (全向量化) ──
sym = df["symbol"]
pct = df["close"] / df["pre_close"] - 1
t10 = df.groupby("symbol")["close"].shift(-10) / df["close"] - 1
df["_pct"] = pct
df["_t10"] = t10

g = df.groupby("symbol")
stats = pd.DataFrame(
    {
        "med_amount": g["amount"].median(),
        "pct_amt_lt_50m": g["_pct"].apply(lambda x: 0),  # placeholder replaced below
        "med_close": g["close"].median(),
        "mean_abs_ret": g["_pct"].apply(lambda s: s.abs().mean()),
        "atr_pct": g.apply(
            lambda s: ((s["high"] - s["low"]) / s["pre_close"]).mean(),
            include_groups=False,
        ),
        "limit_up_main": g["_pct"].apply(lambda s: (s >= 0.098).mean()),
        "limit_up_dual": g["_pct"].apply(lambda s: (s >= 0.198).mean()),
        "t10_mean": g["_t10"].mean(),
        "t10_median": g["_t10"].median(),
        "t10_pos": g["_t10"].apply(lambda s: (s > 0).mean()),
        "t10_q10": g["_t10"].quantile(0.10),
        "t10_q25": g["_t10"].quantile(0.25),
        "t10_q75": g["_t10"].quantile(0.75),
        "t10_q90": g["_t10"].quantile(0.90),
        "susp_rate": 1 - g.size() / WINDOW,
        "n_rows": g.size(),
    }
)
stats["pct_amt_lt_50m"] = df.groupby("symbol")["amount"].apply(
    lambda s: (s < 5e7).mean()
)
stats["grp"] = np.where(stats.index.isin(new_syms), "new", "old")
stats["board"] = stats.index.map(board_of)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
perf = (
    stats.groupby(["grp", "board"])
    .agg(
        n=("med_amount", "size"),
        med_amount=("med_amount", "median"),
        lt50m=("pct_amt_lt_50m", "mean"),
        med_close=("med_close", "median"),
        abs_ret=("mean_abs_ret", "mean"),
        atr=("atr_pct", "mean"),
        lmt_up=("limit_up_main", "mean"),
        t10m=("t10_mean", "mean"),
        t10med=("t10_median", "median"),
        t10pos=("t10_pos", "mean"),
        t10q10=("t10_q10", "median"),
        t10q25=("t10_q25", "median"),
        t10q75=("t10_q75", "median"),
        t10q90=("t10_q90", "median"),
        susp=("susp_rate", "mean"),
    )
    .reset_index()
)
print("\n=== 逐股统计 → 分组汇总 (旧 vs 新 × main/dual) ===")
print(perf.to_string(index=False))
perf.to_csv(rf"{ROOT}\_diag_universe_dist_shift_summary.csv", index=False)

# ── 4. 关键参数冲击 ──
print("\n=== 4.1 读取预过滤: 个股中位额 < 5000万 占比 ===")
for b in ["main", "dual"]:
    for g in ["old", "new"]:
        sub = stats[(stats["board"] == b) & (stats["grp"] == g)]
        print(
            f"  {b:>4} {g:>3}: {len(sub):>4}只 中 {(sub['med_amount'] < 5e7).mean() * 100:.1f}% 低于5000万"
        )

print("\n=== 4.2 dual serving 池 top-200 (合并后才是实际池) ===")
all_dual = stats[stats["board"] == "dual"].sort_values("med_amount", ascending=False)
old_dual = all_dual[all_dual["grp"] == "old"]
new_dual = all_dual[all_dual["grp"] == "new"]
print(
    f"  old dual: {len(old_dual)}只 top200门槛={old_dual['med_amount'].iloc[199] / 1e8:.2f}亿"
)
print(
    f"  new dual: {len(new_dual)}只 全进池需要的门槛={new_dual['med_amount'].iloc[199] / 1e8:.2f}亿"
)
print(
    f"  合并池: {len(all_dual)}只 top200门槛={all_dual['med_amount'].iloc[199] / 1e8:.2f}亿"
)
top200 = all_dual.head(200)
print(f"  合并 top200 中 新股占比: {(top200['grp'] == 'new').mean() * 100:.1f}%")
print(f"  合并 top200 中 新股排名最低位: {top200['grp'].eq('new').idxmax()}", end=" ")
print(
    f"(新股在合并dual池的最低名次={list(all_dual.index).index(top200[top200['grp'] == 'new'].index.min()) + 1 if len(top200[top200['grp'] == 'new']) else '无'})"
)

print("\n=== 4.3 dual 20% 流动性过滤 (分位闸) ===")
cut = all_dual["med_amount"].quantile(0.20)
new_frac_in_bottom20 = (new_dual["med_amount"] <= cut).mean()
print(
    f"  合并池 20% 分位={cut / 1e8:.2f}亿 | 新dualk 落入底部20%占比: {new_frac_in_bottom20 * 100:.1f}%"
)

print("\n=== 4.4 T+10 池级收益分布 (选股/验收闸基准) ===")
t10_all = (
    df[df["board"] != "bse"]
    .groupby(["grp", "board"])["_t10"]
    .apply(lambda s: s.dropna())
)
for (g, b), s in t10_all.groupby(level=[0, 1]):
    print(
        f"  {b:>4} {g:>3}: n={len(s):>7} mean={s.mean():+.4f} median={s.median():+.4f} P10={s.quantile(0.1):+.4f} P25={s.quantile(0.25):+.4f} P75={s.quantile(0.75):+.4f} P90={s.quantile(0.9):+.4f} pos={(s > 0).mean() * 100:.1f}%"
    )

# ── 5. 次新 gate (全量876天日历) ──
all_files = sorted(glob.glob(rf"{DAILY_DIR}\daily_*.parquet"))
cal_all = pd.DatetimeIndex(
    sorted(
        pd.to_datetime(
            pd.read_parquet(all_files[0], columns=["trade_date"])["trade_date"],
            format="%Y%m%d",
        ).unique()
    )
)
for f in all_files[1:]:
    dts = pd.read_parquet(f, columns=["trade_date"])["trade_date"].astype(str)
    cal_all = cal_all.union(pd.DatetimeIndex(pd.to_datetime(dts, format="%Y%m%d")))
cal = cal_all
new["list_dt"] = pd.to_datetime(new["list_date"], format="%Y%m%d", errors="coerce")
new["traded_since"] = new["list_dt"].map(
    lambda d: len(cal) - cal.searchsorted(d) if pd.notna(d) else np.nan
)
new["below_150d"] = new["traded_since"] < 150
print("\n=== 5. 次新 gate: 新增中 上市<150交易日 ===")
print(new.groupby(["board", "below_150d"]).size().to_string())
new2 = new[not new["below_150d"]]
print("\n=== 5.1 剔除<150d后, 新增按上市年份 ===")
print(new2.groupby([new2["list_dt"].dt.year, "board"]).size().to_string())
