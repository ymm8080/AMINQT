# -*- coding: utf-8 -*-
"""
益盟趋势顶底（新公式）Python 复刻
==================================
原公式逐行翻译，纯 OHLC 计算，可用 iFinD 日线数据回测。

原始公式：
    B := 100*(CLOSE-HHV(HIGH,14))/(HHV(HIGH,14)-LLV(LOW,14));
    d := EMA(100*(CLOSE-HHV(HIGH,34))/(HHV(HIGH,34)-LLV(LOW,34)),4);
    A := MA (100*(CLOSE-HHV(HIGH,34))/(HHV(HIGH,34)-LLV(LOW,34)),19);
    短期线: B+100;  中期线: d+100;  长期线: A+100;
    + 见顶 / 顶部区域 / 底部区域 / 低位金叉 四类信号

与规则引擎的对接（IndicatorFeed 接口）：
    red_above_blue_since_peak()  ← 长期线持续>中期线（自某起点）
    red_blue_distance_min()      ← |长期线-中期线| 创N日最小
    见顶/顶部                    → 可接入 L2 日线卖出信号
    底部区域/低位金叉            → 可作为"吸筹峰"的代理（待确认）
"""

import numpy as np
import pandas as pd


# ---- 公式语言基础函数 ----
def HHV(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()


def LLV(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()


def MA(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def EMA(s: pd.Series, n: int) -> pd.Series:
    # 国产公式软件的 EMA：alpha = 2/(n+1)，非调整递推，与 ewm(span=n, adjust=False) 一致
    return s.ewm(span=n, adjust=False).mean()


def REF(s: pd.Series, n: int) -> pd.Series:
    return s.shift(n)


def CROSS(a: pd.Series, b: pd.Series) -> pd.Series:
    """a 上穿 b：今日 a>b 且昨日 a<=b"""
    return (a > b) & (REF(a, 1) <= REF(b, 1))


def FILTER(x: pd.Series, n: int) -> pd.Series:
    """x 为真后，其后 n-1 个周期内的信号置 0（N日内只保留第一个）"""
    out = np.zeros(len(x), dtype=bool)
    last = -n
    xv = x.fillna(False).values
    for i in range(len(x)):
        if xv[i] and i - last >= n:
            out[i] = True
            last = i
    return pd.Series(out, index=x.index)


def _rsv(close, high, low, n):
    """100*(C-HHV(H,n))/(HHV(H,n)-LLV(L,n))，分母为0时取前值（横盘保护）"""
    hhv, llv = HHV(high, n), LLV(low, n)
    denom = (hhv - llv).replace(0, np.nan)
    rsv = 100 * (close - hhv) / denom
    return rsv.ffill()


def yimeng_dingdi(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: DataFrame 含 open/high/low/close 列（按日期升序）
    输出: 原表 + 短期线/中期线/长期线/见顶/顶部/底部区域/低位金叉
    """
    c, h, l = df["close"], df["high"], df["low"]

    rsv14 = _rsv(c, h, l, 14)
    rsv34 = _rsv(c, h, l, 34)

    df["短期线"] = rsv14 + 100
    df["中期线"] = EMA(rsv34, 4) + 100
    df["长期线"] = MA(rsv34, 19) + 100

    S, M, L = df["短期线"], df["中期线"], df["长期线"]

    df["见顶"] = (REF(M, 1) > 85) & (REF(S, 1) > 85) & (REF(L, 1) > 65) & CROSS(L, S)

    top_zone = (
        (M < REF(M, 1))
        & (REF(M, 1) > 80)
        & ((REF(S, 1) > 95) | (REF(S, 2) > 95))
        & (L > 60)
        & (S < 83.5)
        & (S < M)
        & (S < L + 4)
    )
    df["顶部"] = FILTER(top_zone, 4)

    df["底部区域"] = (
        (
            (L < 12)
            & (M < 8)
            & ((S < 7.2) | (REF(S, 1) < 5))
            & ((M > REF(M, 1)) | (S > REF(S, 1)))
        )
        | ((L < 8) & (M < 7) & (S < 15) & (S > REF(S, 1)))
        | ((L < 10) & (M < 7) & (S < 1))
    )

    df["低位金叉"] = (
        (L < 15)
        & (REF(L, 1) < 15)
        & (M < 18)
        & (S > REF(S, 1))
        & CROSS(S, L)
        & (S > M)
        & ((REF(S, 1) < 5) | (REF(S, 2) < 5))
        & ((M >= L) | (REF(S, 1) < 1))
    )

    # ---- 引擎接口用的派生量 ----
    df["红蓝距离"] = (df["长期线"] - df["中期线"]).abs()  # 蓝线=中期线（待确认）
    df["红在蓝上"] = df["长期线"] > df["中期线"]
    return df


# ---- IndicatorFeed 接口实现（红蓝线部分）----
class YimengFeed:
    """把益盟趋势顶底的计算结果包装成引擎接口"""

    def __init__(self, hist: dict[str, pd.DataFrame], distance_window: int = 20):
        # hist: {code: yimeng_dingdi()后的DataFrame}
        self.hist = hist
        self.w = distance_window

    def red_above_blue_since_peak(self, code: str) -> bool:
        """从最近一个'底部区域'信号开始，红线是否一直在蓝线之上"""
        df = self.hist[code].dropna(subset=["长期线"])
        bottoms = df.index[df["底部区域"]]
        if len(bottoms) == 0:
            return False
        seg = df.loc[bottoms[-1] :]
        return bool(seg["红在蓝上"].all())

    def red_blue_distance_min(self, code: str) -> bool:
        """红蓝距离创 N 日最小"""
        df = self.hist[code].dropna(subset=["红蓝距离"])
        if len(df) < self.w:
            return False
        return bool(df["红蓝距离"].iloc[-1] <= df["红蓝距离"].iloc[-self.w :].min())


if __name__ == "__main__":
    df = pd.read_csv("/mnt/agents/output/data_600519_2y.csv")
    df = yimeng_dingdi(df)
    cols = ["time", "close", "短期线", "中期线", "长期线"]
    print("最近5日三线值：")
    print(df[cols].tail(5).round(2).to_string(index=False))
    print("\n近两年信号统计：")
    for sig in ["见顶", "顶部", "底部区域", "低位金叉"]:
        dates = df.loc[df[sig], "time"].tolist()
        print(f"  {sig}: {len(dates)}次 {dates}")
    feed = YimengFeed({"600519.SH": df})
    print(f"\n红在蓝上(自最近底部): {feed.red_above_blue_since_peak('600519.SH')}")
    print(f"红蓝距离20日最小: {feed.red_blue_distance_min('600519.SH')}")
