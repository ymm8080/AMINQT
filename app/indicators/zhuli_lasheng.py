# -*- coding: utf-8 -*-
"""
主力监控（主力拉升/吸筹）指标 Python 复刻
==========================================
白线=主力轨迹，黄线=主力平均线(MAZL)
红柱=吸筹(主力进场)，红黄=洗盘，黄柱=拉高，白/灰柱=出货

原公式要点：
    主力轨迹 = 100*EMA(EMA(MTM,9),9) / EMA(EMA(ABS(MTM),9),9)
    吸筹: VAR5 上升段（VAR5>0）
    拉高: VAR51 下降段    出货: VAR51 上升段

注：VAR21 分母 SMA(MIN(HIGH-VAR1,0),10,1) 多为负值或零，
    VAR51 因此为负值域信号——方向变化才是信息，与原版软件一致。
    分母为0时前值填充（横盘保护），不改变信号方向逻辑。
"""

import numpy as np
import pandas as pd


def EMA(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def MA(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def LLV(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()


def HHV(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()


def REF(s: pd.Series, n: int) -> pd.Series:
    return s.shift(n)


def SMA(s: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA: Y = (X*m + Y_prev*(n-m)) / n，递归"""
    out = np.full(len(s), np.nan)
    x = s.values
    prev = np.nan
    for i in range(len(x)):
        xi = x[i]
        if np.isnan(xi):
            out[i] = prev
            continue
        prev = xi if np.isnan(prev) else (xi * m + prev * (n - m)) / n
        out[i] = prev
    return pd.Series(out, index=s.index)


def zhuli_lasheng(df: pd.DataFrame) -> pd.DataFrame:
    """输入含 open/high/low/close，输出主力轨迹/MAZL/吸筹/洗盘/拉高/出货/吸筹峰"""
    N1, N2 = 9, 5
    c, h, low, o = df["close"], df["high"], df["low"], df["open"]

    MTM = c - REF(c, 1)
    denom = EMA(EMA(MTM.abs(), N1), N1).replace(0, np.nan)
    df["主力轨迹"] = (100 * EMA(EMA(MTM, N1), N1) / denom).ffill()
    df["MAZL"] = MA(df["主力轨迹"], N2)

    # ---- 吸筹/洗盘（低位侧）----
    VAR1 = REF((low + o + c + h) / 4, 1)
    d2 = SMA((low - VAR1).clip(lower=0), 10, 1).replace(0, np.nan)
    VAR2 = (SMA((low - VAR1).abs(), 13, 1) / d2).ffill().fillna(0)
    VAR3 = EMA(VAR2, 10)
    VAR4 = LLV(low, 33)
    VAR5 = EMA(pd.Series(np.where(low <= VAR4, VAR3, 0.0), index=df.index), 3)
    df["VAR5"] = VAR5
    df["吸筹"] = (VAR5 > REF(VAR5, 1)) & (VAR5 > 0)  # 红柱
    df["洗盘"] = (VAR5 < REF(VAR5, 1)) & (VAR5 > 0)  # 红黄柱

    # ---- 拉高/出货（高位侧）----
    d21 = SMA((h - VAR1).clip(upper=0), 10, 1).replace(0, np.nan)  # MIN(H-VAR1,0)
    VAR21 = (SMA((h - VAR1).abs(), 13, 1) / d21).ffill().fillna(0)
    VAR31 = EMA(VAR21, 10)
    VAR41 = HHV(h, 33)
    VAR51 = EMA(pd.Series(np.where(h >= VAR41, VAR31, 0.0), index=df.index), 3)
    df["VAR51"] = VAR51
    df["拉高"] = (VAR51 < REF(VAR51, 1)) & (VAR51 != 0)  # 黄柱
    df["出货"] = (VAR51 > REF(VAR51, 1)) & (REF(VAR51, 1) != 0)  # 白/灰柱

    # ---- 吸筹峰：VAR5 局部最高点（右侧1日确认，盘后使用无未来函数）----
    df["吸筹峰"] = (
        (VAR5 > 0)
        & (VAR5 >= REF(VAR5, 1))
        & (VAR5 >= REF(VAR5, -1))
        & (REF(VAR5, -1) < REF(VAR5, -2))
    )

    # ---- 口诀信号：0轴上方死叉 + 出货柱 → 卖 ----
    df["轨迹死叉"] = (df["主力轨迹"] < df["MAZL"]) & (
        REF(df["主力轨迹"], 1) >= REF(df["MAZL"], 1)
    )
    df["上方死叉出货"] = df["轨迹死叉"] & (REF(df["主力轨迹"], 1) > 0)
    return df


def had_accumulation_peak(df: pd.DataFrame, lookback: int = 20) -> bool:
    """引擎接口：最近 lookback 个交易日内是否出现过吸筹峰"""
    d = df.dropna(subset=["VAR5"])
    if len(d) == 0:
        return False
    return bool(d["吸筹峰"].iloc[-lookback:].any())


if __name__ == "__main__":
    df = pd.read_csv("/mnt/agents/output/data_600519_2y.csv")
    df = zhuli_lasheng(df)

    print("最近3日指标值：")
    print(
        df[["time", "close", "主力轨迹", "MAZL", "VAR5", "VAR51"]]
        .tail(3)
        .round(3)
        .to_string(index=False)
    )

    print("\n近两年信号分布：")
    for sig in ["吸筹", "洗盘", "拉高", "出货", "吸筹峰", "上方死叉出货"]:
        dates = df.loc[df[sig], "time"].astype(str).tolist()
        # 按月归并，便于肉眼校验
        months = sorted({d[:6] for d in dates})
        print(f"  {sig}: {len(dates)}次  月份{months}")

    print(f"\n近20日有吸筹峰: {had_accumulation_peak(df, 20)}")
    # 锚点校验：2024年9月下旬大反弹前应有吸筹；9月底-10月初拉升段应有拉高
    for anchor in ["202409", "202410"]:
        seg = df[df["time"].astype(str).str.startswith(anchor)]
        print(
            f"{anchor}: 吸筹{int(seg['吸筹'].sum())}天 拉高{int(seg['拉高'].sum())}天 "
            f"出货{int(seg['出货'].sum())}天"
        )
