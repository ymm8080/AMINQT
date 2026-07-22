# -*- coding: utf-8 -*-
"""
发现牛股 指标 Python 复刻
==========================
六条EMA均线族 + 买入信号SS：
    SS = EMA3上穿EMA20  AND  收阳(C>O)  AND  上涨(C>昨收)  AND  涨幅≥1.8%

作者原意：量价配合同一天发信号=主力强；大盘上涨周期中选出的才是强中强。
→ 对应引擎：L3 日线买点的增强条件；与"市场广度≥60%"（路径B）天然互补。

注：SS 与 SSS 在原公式中完全相同（SSS 是冗余行），只保留一个。
"""

import pandas as pd


def EMA(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def CROSS(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def faxian_niugu(df: pd.DataFrame) -> pd.DataFrame:
    c, o = df["close"], df["open"]
    df["A1"] = EMA(c, 3)
    df["A2"] = EMA(c, 5)
    df["A3"] = EMA(c, 7)
    df["A4"] = EMA(c, 12)
    df["A5"] = EMA(c, 20)
    df["A6"] = EMA(c, 50)

    df["SS"] = (CROSS(df["A1"], df["A5"])
                & (c > o)
                & (c > c.shift(1))
                & (c / c.shift(1) >= 1.018))
    # 多头排列辅助标记：六线完全多头（A1>A2>A3>A4>A5>A6）
    df["多头排列"] = ((df["A1"] > df["A2"]) & (df["A2"] > df["A3"])
                    & (df["A3"] > df["A4"]) & (df["A4"] > df["A5"])
                    & (df["A5"] > df["A6"]))
    return df


if __name__ == "__main__":
    df = pd.read_csv("/mnt/agents/output/data_600519_2y.csv")
    df = faxian_niugu(df)

    sig = df[df["SS"]]
    print(f"两年 SS 买入信号 {len(sig)} 次：")
    for _, r in sig.iterrows():
        # 信号后5日/10日收益，粗看信号质量
        i = df.index.get_loc(_)
        r5 = df["close"].iloc[i + 5] / r["close"] - 1 if i + 5 < len(df) else None
        r10 = df["close"].iloc[i + 10] / r["close"] - 1 if i + 10 < len(df) else None
        print(f"  {r['time']}  收盘{r['close']:.2f}  "
              f"5日后{r5:+.1%}  10日后{r10:+.1%}" if r5 is not None and r10 is not None
              else f"  {r['time']}  收盘{r['close']:.2f}  (样本尾端)")
    print(f"\n当前多头排列: {bool(df['多头排列'].iloc[-1])}")
    print(f"最新三线: A1={df['A1'].iloc[-1]:.2f} A5={df['A5'].iloc[-1]:.2f} A6={df['A6'].iloc[-1]:.2f}")
