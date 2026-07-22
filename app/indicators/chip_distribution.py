# -*- coding: utf-8 -*-
"""
筹码分布引擎 + 主力筹码控盘程度N 复刻
======================================
WINNER(x) = 成本低于 x 的筹码占比，通达信/益盟的经典算法近似：
    每日筹码迁移：存量 ×(1-换手率)，当日换手部分按 H~L 三角分布加入
    峰值设在当日典型价 A01=(C+O+L+H)/4

控盘程度N 公式：
    A02 = WINNER(A01*1.04)*100   上方4%处获利盘
    A03 = WINNER(A01*0.96)*100   下方4%处获利盘 → A04红柱
    A08 = A02-A03                ±4%区间筹码集中度
    A0A = (HHV(A04,15)-A03)/A03 > 30% 且 HHV(A04,15)>50  → 控盘增强信号

已知近似误差：无逐笔成交明细，筹码迁移用换手率均匀衰减假设；
历史越长初始误差越小（换手累计>>100%后初始分布被冲刷掉）。
"""

import numpy as np
import pandas as pd


class ChipDistribution:
    def __init__(self, n_bins: int = 400):
        self.n_bins = n_bins
        self.grid: np.ndarray | None = None
        self.dist: np.ndarray | None = None

    def _triangle(self, low: float, high: float, peak: float) -> np.ndarray:
        """当日新增筹码的三角分布（峰值在典型价）"""
        g = self.grid
        w = np.zeros(self.n_bins)
        in_range = (g >= low) & (g <= high)
        if not in_range.any():
            idx = np.argmin(np.abs(g - peak))
            w[idx] = 1.0
            return w
        gl, gh = low, high
        w[in_range] = np.where(
            g[in_range] <= peak,
            (g[in_range] - gl) / max(peak - gl, 1e-9),
            (gh - g[in_range]) / max(gh - peak, 1e-9))
        w = np.clip(w, 0, None)
        return w / w.sum()

    def build(self, df: pd.DataFrame, float_shares: float) -> pd.DataFrame:
        """
        df: 需含 open/high/low/close/volume，按日期升序
        float_shares: 流通股本（股），换手率 = volume/float_shares
        返回: 原表 + A01/A02/A03/A04/A08/A0A/获利盘
        """
        lo, hi = df["low"].min() * 0.9, df["high"].max() * 1.1
        self.grid = np.linspace(lo, hi, self.n_bins)
        self.dist = np.zeros(self.n_bins)

        out = {k: [] for k in ("A01", "A02", "A03", "获利盘")}
        for _, r in df.iterrows():
            a01 = (r["close"] + r["open"] + r["low"] + r["high"]) / 4
            t = min(r["volume"] / float_shares, 1.0)     # 当日换手率(小数)
            if self.dist.sum() == 0:                      # 首日全部筹码入分布
                self.dist = self._triangle(r["low"], r["high"], a01)
            else:
                self.dist *= (1 - t)
                self.dist += t * self._triangle(r["low"], r["high"], a01)

            out["A01"].append(a01)
            out["A02"].append(self.winner(a01 * 1.04) * 100)
            out["A03"].append(self.winner(a01 * 0.96) * 100)
            out["获利盘"].append(self.winner(r["close"]) * 100)

        for k, v in out.items():
            df[k] = v
        df["A04"] = df["A03"]
        df["A08"] = df["A02"] - df["A03"]
        hhv15 = df["A04"].rolling(15, min_periods=15).max()
        df["A0A"] = (((hhv15 - df["A03"]) / df["A03"].replace(0, np.nan) * 100 > 30)
                     & (hhv15 > 50)).fillna(False)
        return df

    def winner(self, price: float) -> float:
        return float(self.dist[self.grid < price].sum() / max(self.dist.sum(), 1e-12))


# ---- 引擎接口实现（控盘红柱/获利盘）----
class ChipFeed:
    def __init__(self, hist: dict[str, pd.DataFrame]):
        self.hist = hist                       # {code: build()后的df}

    def red_bar_rising_and_majority(self, code: str) -> bool:
        d = self.hist[code]
        return bool(d["A04"].iloc[-1] > d["A04"].iloc[-2]
                    and d["A04"].iloc[-1] > 50)

    def profit_chip_ratio(self, code: str) -> float:
        return float(self.hist[code]["获利盘"].iloc[-1])

    def control_signal_A0A(self, code: str) -> bool:
        return bool(self.hist[code]["A0A"].iloc[-1])


if __name__ == "__main__":
    df = pd.read_csv("/mnt/agents/output/data_600519_2y.csv")
    FLOAT_SHARES_600519 = 1.256e9          # 茅台流通股本约12.56亿股

    chip = ChipDistribution()
    df = chip.build(df, FLOAT_SHARES_600519)

    print("最近5日：")
    print(df[["time", "close", "A04", "A08", "获利盘"]]
          .tail(5).round(2).to_string(index=False))

    print(f"\n值域检查: A04∈[{df['A04'].min():.1f},{df['A04'].max():.1f}]  "
          f"获利盘∈[{df['获利盘'].min():.1f},{df['获利盘'].max():.1f}]")
    a0a = df[df["A0A"]]
    print(f"A0A控盘增强信号 {len(a0a)} 次: "
          f"{a0a['time'].astype(str).str[:6].unique().tolist()}")

    feed = ChipFeed({"600519.SH": df})
    print(f"\n红柱升高且>50: {feed.red_bar_rising_and_majority('600519.SH')}")
    print(f"当前获利盘: {feed.profit_chip_ratio('600519.SH'):.1f}%")
    # 对照锚点：2024年9月大涨后获利盘应接近100%，2026年5-6月低位应显著低
    for anchor in ["20241008", "20260525"]:
        row = df[df["time"].astype(str) == anchor]
        if len(row):
            print(f"锚点 {anchor}: 获利盘={row['获利盘'].iloc[0]:.1f}%  "
                  f"A04={row['A04'].iloc[0]:.1f}")
