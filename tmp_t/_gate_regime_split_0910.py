# -*- coding: utf-8 -*-
"""0910 闸×行情分桶回测 (用户: 是不是应该看市场情况, 而不是统一含糊处理)
两轴行情 (预登记, 不挑):
  R1 大盘潮汐 = 池内等权 mean(pctChg) 20日和 >0 / <=0
  R2 小票潮汐 = 低额半场mean(pctChg) - 高额半场mean(pctChg) 20日和 >0 / <=0
问题A (金额闸): e125宽池(>=5e7) vs 窄池(>=8e7) TOP15 净收益日差, 按行情桶分列
问题B (一字板): 一字票T+1开盘可买者FWD5, 按行情桶分列
判据: 存在桶内稳定正 (n>=30日且均值>0且正日占比>=55%) → 值得谈行情条件闸; 否则维持统一闸
"""
import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")

E125 = r"D:/AMINQT/AMINQT CODES/data/_diag_rankkey_scored_wide_e125.parquet"
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_gate_regime_split_20260910.txt"
pd.set_option("display.width", 250)
L = []
def P(s=""):
    print(s); L.append(str(s))

# ---- 行情轴 (面板全池等权) ----
p = pd.read_parquet(PNL, columns=["symbol", "date", "pctChg", "amount"])
p["date"] = pd.to_datetime(p["date"])
pool = p[p["amount"] >= 5e7]
daily = pool.groupby("date").apply(lambda d: pd.Series({
    "ew": d["pctChg"].mean(),
    "low": d[d["amount"] <= d["amount"].median()]["pctChg"].mean(),
    "high": d[d["amount"] > d["amount"].median()]["pctChg"].mean(),
}), include_groups=False).reset_index()
daily = daily.sort_values("date")
daily["R1"] = daily["ew"].rolling(20).sum()
daily["R2"] = (daily["low"] - daily["high"]).rolling(20).sum()
daily["dstr"] = daily["date"].dt.strftime("%Y%m%d")
REG = daily[["date", "R1", "R2"]].dropna()
P(f"行情轴: {REG['date'].min():%m-%d}..{REG['date'].max():%m-%d}, n={len(REG)}日")
P(f"R1>0占比={(REG['R1']>0).mean():.0%}  R2>0(小票强)占比={(REG['R2']>0).mean():.0%}")

def bucket_label(r1, r2):
    return ("牛" if r1 > 0 else "熊") + ("×小票强" if r2 > 0 else "×小票弱")

# ================= A. 金额闸 e125 分桶 =================
P("\n" + "=" * 76)
P("A. 金额闸: 宽池(>=5e7) vs 窄池(>=8e7) TOP15(按pred_ret_10d) 日净差 × 行情桶")
P("=" * 76)
e = pd.read_parquet(E125)
e = e.merge(REG, on="date", how="inner")
rows = []
for (d, b), sub in e.groupby(["date", "board"]):
    sub = sub.dropna(subset=["pred_ret_10d", "realized_net"]).sort_values("pred_ret_10d", ascending=False)
    wide = sub.head(15)
    narrow = sub[sub["amount"] >= 8e7].head(15)
    if len(narrow) < 10:
        continue
    dw = wide["realized_net"].mean() - narrow["realized_net"].mean()
    setn, setw = set(narrow["symbol"]), set(wide["symbol"])
    blood = setw - setn  # 换血: 宽池新入
    b_out = narrow[~narrow["symbol"].isin(setw)]
    b_in = wide[wide["symbol"].isin(blood)]
    rows.append({
        "date": d, "board": b, "delta": dw,
        "n_blood": len(blood),
        "blood_in": b_in["realized_net"].mean() if len(b_in) else np.nan,
        "blood_out": b_out["realized_net"].mean() if len(b_out) else np.nan,
        "R1": sub["R1"].iloc[0], "R2": sub["R2"].iloc[0],
    })
A = pd.DataFrame(rows)
A["bucket"] = [bucket_label(a, b) for a, b in zip(A["R1"], A["R2"])]
for board, sb in A.groupby("board"):
    P(f"\n--- {board} (n={len(sb)}晚) 全窗delta均值={sb['delta'].mean():+.4f} 中位={sb['delta'].median():+.4f} 正日占比={(sb['delta']>0).mean():.0%}")
    g = sb.groupby("bucket").agg(
        n晚=("delta", "size"), delta均值=("delta", "mean"), delta中位=("delta", "median"),
        宽池更优占比=("delta", lambda x: (x > 0).mean()),
        换血均只数=("n_blood", "mean"), 新入净=("blood_in", "mean"), 被挤净=("blood_out", "mean"))
    g["换血每只差"] = g["新入净"] - g["被挤净"]
    P(g.round(4).to_string())

# ================= B. 一字板 分桶 =================
P("\n" + "=" * 76)
P("B. 一字板: T+1开盘可买者 开盘进场FWD5 × 行情桶 (近1年, 额>=5e7)")
P("=" * 76)
cols = ["symbol", "date", "open", "high", "low", "close", "pctChg", "amount"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
g2 = df.groupby("symbol")
for k in (5,):
    mat = pd.concat([g2["pctChg"].shift(-i) for i in range(1, k + 1)], axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    df[f"f{k}"] = ((1 + mat / 100).prod(axis=1) - 1).where(valid)
df["o1"] = g2["open"].shift(-1)
df["h1"] = g2["high"].shift(-1)
df["l1"] = g2["low"].shift(-1)
df["c1"] = g2["close"].shift(-1)
df["t1_sealed"] = (df["h1"] == df["l1"]) & (df["c1"] == df["h1"])
dual20 = df["symbol"].str.startswith(("300", "301", "688"))
df["lim1"] = (df["close"] * np.where(dual20, 1.2, 1.1)).round(2)
df["t1_fillable"] = (df["o1"] < df["lim1"]) & ~df["t1_sealed"]
df["is_lim"] = (df["pctChg"] >= 9.5).astype(int)
df["prev_is_lim"] = df.groupby("symbol")["is_lim"].shift(1)
df = df.merge(REG[["date", "R1", "R2"]], on="date", how="inner")
ow = df[(df["high"] == df["low"]) & (df["close"] == df["high"]) &
        (df["pctChg"] >= 9.5) & (df["amount"] >= 5e7)].copy()
ow["cont"] = ow["prev_is_lim"] == 1
ow["entry5"] = (1 + ow["f5"]) * ow["close"] / ow["o1"] - 1
ow["bucket"] = [bucket_label(a, b) for a, b in zip(ow["R1"], ow["R2"])]
fill = ow[ow["t1_fillable"]]
P(f"一字板 n={len(ow)}, 可成交 n={len(fill)}")
rows = []
for bkt, sub in fill.groupby("bucket"):
    rows.append({"bucket": bkt, "n": len(sub),
                 "entry5均": sub["entry5"].mean(), "entry5中位": sub["entry5"].median(),
                 "hit10%": (sub["entry5"] >= 0.10).mean(),
                 "首板n": len(sub[~sub["cont"]]), "首板entry5": sub[~sub["cont"]]["entry5"].mean(),
                 "连板n": len(sub[sub["cont"]]), "连板entry5": sub[sub["cont"]]["entry5"].mean()})
B = pd.DataFrame(rows).sort_values("bucket")
P(B.round(4).to_string(index=False))
# 池基线同桶对照
base = df[(df["amount"] >= 5e7)].dropna(subset=["f5"]).copy()
base["bucket"] = [bucket_label(a, b) for a, b in zip(base["R1"], base["R2"])]
rows = []
for bkt, sub in base.groupby("bucket"):
    rows.append({"bucket": bkt, "n股日": len(sub), "基线FWD5": sub["f5"].mean(),
                 "基线hit10%": (sub["f5"] >= 0.10).mean()})
P("\n池基线同桶对照:")
P(pd.DataFrame(rows).sort_values("bucket").round(4).to_string(index=False))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
