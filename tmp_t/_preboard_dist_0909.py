# -*- coding: utf-8 -*-
"""板前派发研究 (09-09 用户问: 板之前一般都要做派发? 分首板/二板看).

口径:
  板 = 收盘涨停 (close >= round(pre_close*(1+ratio),2)-0.005; ratio 按板 10%/20%)
  连板位 = 连续涨停run内第几板 (1=首板, 2=二板, 3+=高度板)
  板前派发 = 板前 t-k 日的 wr5 (获利盘 - 5日前获利盘) < 0
  基线 = 全部非板前日的 wr5<0 占比 (同一股票池同窗口, 控噪声底)
数据: panel v3 (close/pre_close 识别板) + cyq_panel (winner_ratio)
全向量化 (groupby+shift, 无逐股循环)。
"""
import numpy as np
import pandas as pd

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
CYQ = r"D:\AMINQT\AMINQT CODES\data\cyq_panel.parquet"

pan = pd.read_parquet(PANEL, columns=["symbol", "date", "close", "pre_close"])
pan["symbol"] = pan["symbol"].astype(str).str.zfill(6)
pan = pan.sort_values(["symbol", "date"]).reset_index(drop=True)

pref = pan["symbol"].str[:2]
ratio = np.select(
    [pref.isin(["60", "00"]), pref.isin(["30", "68"])], [0.10, 0.20], default=0.30
)
limit = (pan["pre_close"] * (1 + ratio)).round(2)
pan["is_board"] = pan["close"] >= (limit - 0.005)

# 连板位 (09-09修: 原 run=cumsum(非板日) 分组把前置非板日计入 pos → 首板全被
# 记成 2, bpos∈[1,3] 只剩"历史首行即板"的假首板 51 只; 改为板 streak 内计数)
prev = pan.groupby("symbol")["is_board"].shift(1)
new_streak = pan["is_board"] & ~prev.fillna(False).astype(bool)
pan["_streak"] = new_streak.groupby(pan["symbol"]).cumsum()
pos = pan.groupby([pan["symbol"], pan["_streak"]]).cumcount() + 1
pan["bpos"] = np.where(pan["is_board"], pos, 0)

bc = pan.loc[pan["is_board"], "bpos"].value_counts().sort_index()
print(
    "连板位分布 (panel 全量): "
    + ", ".join(f"{b}板:{int(c):,}" for b, c in bc.head(6).items())
)

# 特征在 cyq 侧先算 (shift 语义=cyq交易日行, 不受 inner merge 丢行影响)
cyq = pd.read_parquet(CYQ, columns=["symbol", "date", "winner_ratio"])
cyq["symbol"] = cyq["symbol"].astype(str).str.zfill(6)
cyq = cyq.drop_duplicates(["symbol", "date"], keep="last").sort_values(
    ["symbol", "date"]
)
g = cyq.groupby("symbol")["winner_ratio"]
cyq["wr5_self"] = (cyq["winner_ratio"] - g.shift(5)) * 100  # 当日 wr5, pp
cyq["wr1_self"] = (cyq["winner_ratio"] - g.shift(1)) * 100  # 当日获利盘单日变化
cyq["wr_lvl"] = cyq["winner_ratio"] * 100
gs = cyq.groupby("symbol")
for k in (1, 2, 3):
    cyq[f"wr5_m{k}"] = gs["wr5_self"].shift(k)  # 板前第k日的 wr5
cyq["wr1_m1"] = gs["wr1_self"].shift(1)  # 板前一日获利盘单日变化
cyq["wr_lvl_m1"] = gs["wr_lvl"].shift(1)  # 板前一日获利盘水位
cyq = cyq[cyq["symbol"].isin(pan["symbol"].unique())]

df = pan.merge(
    cyq.drop(columns=["winner_ratio"]), on=["symbol", "date"], how="inner"
).sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"合并行 {len(df):,} | 股票 {df.symbol.nunique()} | {df.date.min().date()}..{df.date.max().date()}")
bc2 = df.loc[df["is_board"], "bpos"].value_counts().sort_index()
print(
    "连板位分布 (合并后): "
    + ", ".join(f"{b}板:{int(c):,}" for b, c in bc2.head(6).items())
)

base_mask = df["is_board"].eq(False)
print("\n== 基线 (全部非板日 wr5<0 占比, 噪声底) ==")
for k in (1, 2, 3):
    col = f"wr5_m{k}"
    b = df.loc[base_mask, col].dropna()
    print(f"t-{k}: P(wr5<0)={ (b<0).mean()*100:.1f}%  中位 {b.median():+.1f}pp  n={len(b):,}")

board = df[df["is_board"] & df["bpos"].between(1, 3)].copy()
board["pos_label"] = board["bpos"].map(
    {1: "首板", 2: "二板", 3: "三板+ (bpos=3)"}
)
# 三板及以上合并: 重算 (bpos>=3)
board["pos_label"] = np.where(board["bpos"] >= 3, "三板+", board["bpos"].map({1: "首板", 2: "二板"}))

print("\n== 板前派发率 (P(板前第k日 wr5<0)), 按连板位 ==")
rows = []
for pl, sub in board.groupby("pos_label"):
    row = {"板位": pl, "n板": len(sub)}
    for k in (1, 2, 3):
        v = sub[f"wr5_m{k}"].dropna()
        row[f"t-{k} P(wr5<0)"] = f"{(v<0).mean()*100:.1f}%"
        row[f"t-{k} 中位wr5"] = f"{v.median():+.1f}pp"
    rows.append(row)
print(pd.DataFrame(rows).set_index("板位").to_string())

print("\n== 板前一日获利盘水位 & 单日变化 ==")
for pl, sub in board.groupby("pos_label"):
    lvl = sub["wr_lvl_m1"].dropna()
    d1 = sub["wr1_m1"].dropna()
    print(
        f"{pl}: n={len(sub):,} | 水位中位 {lvl.median():.0f}% 分位[25/75]={lvl.quantile(.25):.0f}/{lvl.quantile(.75):.0f}%"
        f" | 单日Δ中位 {d1.median():+.1f}pp P(Δ<0)={(d1<0).mean()*100:.1f}%"
    )

print("\n== 对照: 全体非板日的下一日也不板 (纯随机底) wr5<0 ==")
nb = df[df["is_board"].eq(False)]
v = nb["wr5_m1"].dropna()  # wr5_m1 对非板行=次日板的板前特征, 需另算
# 正确底: 非板行自身的 wr5 (当日口径)
df["wr5_self"] = (df["winner_ratio"] - g.shift(5)) * 100
v = df.loc[base_mask, "wr5_self"].dropna()
print(f"任意非板日 P(wr5<0)={(v<0).mean()*100:.1f}% 中位 {v.median():+.1f}pp n={len(v):,}")

# 板前一日 vs 次日板/不板 的判别力 (信息量视角)
df["next_board"] = df.groupby("symbol")["is_board"].shift(-1)
both = df[df["wr5_self"].notna() & df["next_board"].notna() & df["is_board"].eq(False)]
for lab, m in [("次日板", both["next_board"]), ("次日不板", ~both["next_board"])]:
    v2 = both.loc[m, "wr5_self"]
    print(f"{lab}: P(wr5<0)={(v2<0).mean()*100:.1f}% 中位 {v2.median():+.1f}pp n={len(v2):,}")
