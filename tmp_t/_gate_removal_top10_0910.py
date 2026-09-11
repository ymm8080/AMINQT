# -*- coding: utf-8 -*-
"""0910 闸删撤对TOP10的影响量化 (用户问: 5000万/8000万删了对TOP10有好处吗; 一字板剔除也不应该要?)
A. 金额闸(现窗粗检): parallel raw preds 23日, TOP10含<8000万带 vs 剔除<8000万带, 配对日delta
B. 一字板剔除(250td面板): 一字涨停票次日可成交率 + 可成交时T+1开盘进场的FWD5/10 vs 池基线
   一字定义: high==low & close==high & pctChg>=9.5 (含20cm); 池语境=当日额>=5e7
   脏pctChg(|x|>22)整窗剔除. WORM: 输出带日期后缀.
"""
import os, glob
import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")

LST = r"D:/AMINQT/Daily Operation/STOCK LIST"
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_gate_removal_top10_20260910.txt"
pd.set_option("display.width", 250)
L = []
def P(s=""):
    print(s); L.append(str(s))

# ================= A. 金额闸现窗粗检 =================
P("=" * 72)
P("A. 金额闸 — parallel TOP10: 含<8000万带 vs 剔除<8000万带 (23日, 初步)")
P("=" * 72)
pnl_amt = pd.read_parquet(PNL, columns=["symbol", "date", "pctChg", "amount"])
pnl_amt["date"] = pd.to_datetime(pnl_amt["date"])
pnl_amt = pnl_amt.sort_values(["symbol", "date"])
g = pnl_amt.groupby("symbol")["pctChg"]
def win_ret(k):
    mat = pd.concat([g.shift(-i) for i in range(1, k + 1)], axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    return ((1 + mat / 100).prod(axis=1) - 1).where(valid)
pnl_amt["fwd5"] = win_ret(5)
pnl_amt["fwd10"] = win_ret(10)
pnl_amt["dstr"] = pnl_amt["date"].dt.strftime("%Y%m%d")
FWD = pnl_amt[["symbol", "dstr", "fwd5", "fwd10"]]

def pick(files):
    best = {}
    for f in files:
        b = os.path.basename(f)
        d = b.split("__")[0].replace("parallel_preds_raw_", "")
        if d not in best or b > best[d][0]:
            best[d] = (b, f)
    return {d: f for d, (k, f) in best.items()}

abs_rows = []
for dstr, f in sorted(pick(glob.glob(f"{LST}/parallel_preds_raw_2026*.csv")).items()):
    if dstr < "20260806":
        continue
    par = pd.read_csv(f, dtype={"symbol": str})
    if "score" not in par.columns:
        continue
    par["dstr"] = dstr
    par = par.merge(pnl_amt[["symbol", "dstr", "amount", "fwd5", "fwd10"]], on=["symbol", "dstr"], how="left")
    par = par.dropna(subset=["amount"]).sort_values("score", ascending=False).head(10)
    abs_rows.append({"dstr": dstr, "n_low": int((par["amount"] < 8e7).sum()),
                     "all_f5": par["fwd5"].mean(), "all_f10": par["fwd10"].mean(),
                     "hi_f5": par[par["amount"] >= 8e7]["fwd5"].mean(),
                     "hi_f10": par[par["amount"] >= 8e7]["fwd10"].mean(),
                     "low_f5": par[par["amount"] < 8e7]["fwd5"].mean()})
AB = pd.DataFrame(abs_rows)
P(f"日数={len(AB)}, <8000万带进TOP10: {(AB['n_low']>0).sum()}日, 均{AB['n_low'].mean():.1f}只/日")
P(f"TOP10(含低带) FWD5日均={AB['all_f5'].mean():+.4f} | 剔低带TOP10 FWD5日均={AB['hi_f5'].mean():+.4f} | 低带成员自身FWD5={AB['low_f5'].mean():+.4f}")
P(f"配对日delta中位={AB['all_f5'].sub(AB['hi_f5']).median():+.4f}, 正日占比={(AB['all_f5']>AB['hi_f5']).mean():.0%} (正=低带有贡献)")
P(f"FWD10 (仅≤0826信号有效): 含={AB['all_f10'].mean():+.4f} 剔={AB['hi_f10'].mean():+.4f}")

# ================= B. 一字板剔除 =================
P("\n" + "=" * 72)
P("B. 一字板剔除 — 一字涨停票的次日成交现实与后续收益 (近250td, 池语境额>=5e7)")
P("=" * 72)
cols = ["symbol", "date", "open", "high", "low", "close", "pctChg", "amount"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
g2 = df.groupby("symbol")
df["pre_close"] = g2["close"].shift(1)
for k in (1, 2, 5, 10):
    mat = pd.concat([g2["pctChg"].shift(-i) for i in range(1, k + 1)], axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    df[f"f{k}"] = ((1 + mat / 100).prod(axis=1) - 1).where(valid)
df["o1"] = g2["open"].shift(-1)          # T+1 开盘
df["h1"] = g2["high"].shift(-1)
df["l1"] = g2["low"].shift(-1)
df["c1"] = g2["close"].shift(-1)
df["d1"] = g2["date"].shift(-1)
# 次日仍一字/开盘即封 → 不可成交
df["t1_sealed"] = (df["h1"] == df["l1"]) & (df["c1"] == df["h1"])
# 次日涨停价(按板别): 300/301/688=20%, 其余10%
dual20 = df["symbol"].str.startswith(("300", "301", "688"))
df["lim1"] = (df["close"] * np.where(dual20, 1.2, 1.1)).round(2)
df["t1_fillable"] = (df["o1"] < df["lim1"]) & ~df["t1_sealed"]
# 一字日(池语境)
ow = df[(df["high"] == df["low"]) & (df["close"] == df["high"]) &
        (df["pctChg"] >= 9.5) & (df["amount"] >= 5e7) & df["pre_close"].notna()].copy()
cutoff = df["date"].max() - pd.Timedelta(days=365)
ow = ow[ow["date"] >= cutoff]
# 首板一字 vs 连板一字: 上一交易日是否也涨停
df["is_lim"] = (df["pctChg"] >= 9.5).astype(int)
df["prev_is_lim"] = df.groupby("symbol")["is_lim"].shift(1)
ow = df[(df["high"] == df["low"]) & (df["close"] == df["high"]) &
        (df["pctChg"] >= 9.5) & (df["amount"] >= 5e7) & (df["date"] >= cutoff)].copy()
ow["cont"] = ow["prev_is_lim"] == 1
n = len(ow)
P(f"一字涨停(额>=5e7) 事件: {n} 起/近1年; 连板一字占比 {ow['cont'].mean():.0%}")
P(f"T+1 仍一字(买不进): {ow['t1_sealed'].mean():.0%}")
P(f"T+1 开盘可成交(开<涨停价且非一字): {ow['t1_fillable'].mean():.0%}")
# 基线: 全池语境日 FWD5/FWD10
base = df[(df["amount"] >= 5e7) & (df["date"] >= cutoff)]
P(f"\n池基线(额>=5e7全样本): FWD5={base['f5'].mean():+.4f} FWD10={base['f10'].mean():+.4f} hit5(FWD5>=10%)={(base['f5']>=0.10).mean():.1%}")
for tag, sub in (("全部一字", ow), ("首板一字", ow[~ow["cont"]]), ("连板一字", ow[ow["cont"]])):
    fill = sub[sub["t1_fillable"]]
    # T+1开盘进场收益: 从 o1 到 T+1..T+k 收盘复利 ≈ (1+f5)*(1+o1/c -1)... 精确: o1进→T+5收
    # 用 close_T 为基: o1/c - 1 为首日开盘滑点, 之后 f5 为 T+1收→T+5收
    entry5 = (1 + fill["f5"]) * fill["close"] / fill["o1"] - 1 if len(fill) else pd.Series(dtype=float)
    entry10 = (1 + fill["f10"]) * fill["close"] / fill["o1"] - 1 if len(fill) else pd.Series(dtype=float)
    P(f"\n--- {tag}: n={len(sub)}, 可成交n={len(fill)}")
    if len(fill):
        P(f"    可成交者 T+1开盘进场 FWD5={entry5.mean():+.4f} (中位{entry5.median():+.4f}) hit10%={(entry5>=0.10).mean():.1%}")
        P(f"    可成交者 T+1开盘进场 FWD10={entry10.mean():+.4f} hit10%={(entry10>=0.10).mean():.1%}")
        P(f"    理论(T+1收盘能买) FWD5={fill['f5'].mean():+.4f} FWD10={fill['f10'].mean():+.4f}")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
