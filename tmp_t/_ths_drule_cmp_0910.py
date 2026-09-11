# -*- coding: utf-8 -*-
"""D-rule vs THS看涨/看跌 对照 — 本地面板算D-rule信号"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd

BULL = pd.read_parquet(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_signal_bull_0910.parquet")
BEAR = pd.read_parquet(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_signal_bear_0910.parquet")
bull_codes = BULL["股票代码"].astype(str).tolist()
bear_codes = BEAR["股票代码"].astype(str).tolist()
print(f"THS看涨 {len(bull_codes)}只: {bull_codes}")
print(f"THS看跌 {len(bear_codes)}只: {bear_codes}")

pf = pd.read_parquet(
    "D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
    columns=["symbol", "date", "open", "high", "low", "close", "amount"])
pf["date"] = pd.to_datetime(pf["date"])
pf = pf.sort_values(["symbol", "date"])
print(f"面板: {pf['symbol'].nunique()}只, {pf['date'].max().date()} 止")


def kdj(df, n=9):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d


def drule(df):
    """D-rule: close>MA10 + r10>0 + 阳线 + KDJ金叉≤3d"""
    ma10 = df["close"].rolling(10).mean()
    r10 = df["close"].pct_change(10)
    k, d = kdj(df)
    cross = (k > d) & (k.shift(1) <= d.shift(1))
    since = cross[::-1].groupby(~cross[::-1].cumsum()).cumsum()[::-1]  # 距最近金叉天数
    conds = (df["close"] > ma10) & (r10 > 0) & (df["close"] > df["open"]) & (since <= 3)
    return conds, since


out_rows = []
for label, codes in [("看涨", bull_codes), ("看跌", bear_codes)]:
    for code in codes:
        sym = code if "." in code else code
        sub = pf[pf["symbol"].str.startswith(sym)].tail(30)
        if len(sub) < 15:
            out_rows.append({"THS": label, "code": code, "D-rule": "无数据"})
            continue
        conds, since = drule(sub)
        last = conds.iloc[-1]
        last_bar = sub.iloc[-1]
        out_rows.append({
            "THS": label, "code": code,
            "简称": (BULL.set_index("股票代码")["股票简称"].get(code, "")
                     if label == "看涨" else BEAR.set_index("股票代码")["股票简称"].get(code, "")),
            "D-rule": "PASS" if last else "fail",
            "KDJ距金叉": int(since.iloc[-1]) if np.isfinite(since.iloc[-1]) else -1,
            "当日涨跌幅%": round((last_bar["close"] / sub["close"].iloc[-2] - 1) * 100, 2),
            "amount_亿": round(last_bar["amount"] / 1e8, 2),
            "date": str(last_bar["date"].date()),
        })

res = pd.DataFrame(out_rows)
print("\n===== 对照表 =====")
print(res.to_string(index=False))
res.to_parquet(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_drule_cmp_0910.parquet")

for label in ["看涨", "看跌"]:
    seg = res[res["THS"] == label]
    ok = (seg["D-rule"] == "PASS").sum()
    print(f"\n{label}: D-rule PASS {ok}/{len(seg)}")
