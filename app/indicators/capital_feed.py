# -*- coding: utf-8 -*-
"""
资金流数据层：B路径(akshare东财) + C路径(同花顺手工导入)
==========================================================
优先级：C(同花顺真实口径) > B(东财代理)。
同一只股票，当天有手工导入值就用手工值；没有就走东财代理。

C路径工作流（每天盘后1分钟）：
    同花顺自选股 → 问财/数据导出 → 存成 csv/xlsx → 丢进 data/ths_manual/
    列名支持：代码/股票代码, 日期, 主力控盘比例, 大单净量（后两列可选其一）

B路径字段映射（东财 stock_individual_fund_flow）：
    主力净流入-净额    → big_order_net（大单净量代理）
    超大单+大单净额    → big_order_net 的替代口径
    主力净流入-净占比  → 当日主力强度
    控盘代理 control_proxy = 近10日主力净流入合计 / 当日成交额 × 100
    （阈值30%需按分布重校准，见 calibrate_threshold()）
"""

from __future__ import annotations
import os
import glob
import time
import pandas as pd


# ============================================================
# B路径：akshare 东财资金流
# ============================================================
class AkshareCapitalFeed:
    def __init__(self, cache_dir: str = "data/capital_cache", fetcher=None):
        """
        fetcher: 可注入的数据函数(code6, market)->DataFrame，
                 默认用 akshare；测试时传 mock 函数。
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._fetcher = fetcher or self._akshare_fetch

    @staticmethod
    def _akshare_fetch(code6: str, market: str) -> pd.DataFrame:
        import akshare as ak
        return ak.stock_individual_fund_flow(stock=code6, market=market)

    @staticmethod
    def _market_of(code: str) -> str:
        return "sh" if code.startswith(("6", "9")) else "sz"

    def _load(self, code: str, refresh: bool = False) -> pd.DataFrame:
        """按日缓存：当天拉过就不重复拉"""
        code6 = code.split(".")[0]
        cache = os.path.join(self.cache_dir, f"{code6}.csv")
        today = time.strftime("%Y%m%d")
        if not refresh and os.path.exists(cache):
            df = pd.read_csv(cache, dtype={"日期": str})
            if str(df["日期"].iloc[-1]).replace("-", "") >= today:
                return df
        df = self._fetcher(code6, self._market_of(code6))
        df.to_csv(cache, index=False)
        return df

    def _row(self, code: str, date: str | None = None) -> pd.Series | None:
        df = self._load(code)
        if date is None:
            return df.iloc[-1]
        row = df[df["日期"].astype(str).str.replace("-", "") == date.replace("-", "")]
        return row.iloc[-1] if len(row) else None

    def main_net_inflow(self, code: str, date: str | None = None) -> float:
        r = self._row(code, date)
        return float(r["主力净流入-净额"]) if r is not None else 0.0

    def big_order_net(self, code: str, date: str | None = None) -> float:
        """大单净量代理：超大单+大单净额（无则退主力净流入）"""
        r = self._row(code, date)
        if r is None:
            return 0.0
        try:
            return float(r["超大单净流入-净额"]) + float(r["大单净流入-净额"])
        except KeyError:
            return float(r.get("主力净流入-净额", 0.0))

    def control_proxy(self, code: str, date: str | None = None,
                      window: int = 10) -> float:
        """控盘代理：近N日主力净流入合计 / 近N日成交额合计 × 100"""
        df = self._load(code).tail(window)
        if len(df) == 0:
            return 0.0
        inflow = df["主力净流入-净额"].astype(float).sum()
        # 东财该接口无成交额列时，用主力净占比反推或退化为净占比均值
        if "主力净流入-净占比" in df.columns:
            return float(df["主力净流入-净占比"].astype(float).mean())
        return float(inflow) / 1e8  # 亿元口径，仅作排序参考

    @staticmethod
    def calibrate_threshold(proxies: list[float], top_pct: float = 0.20) -> float:
        """阈值重校准：取股票池控盘代理的前 top_pct 分位作为'高控盘'线"""
        s = pd.Series(proxies)
        return float(s.quantile(1 - top_pct))


# ============================================================
# C路径：同花顺手工导出导入
# ============================================================
class THSManualFeed:
    COLMAP = {
        "代码": "code", "股票代码": "code", "证券代码": "code",
        "日期": "date", "交易日": "date",
        "主力控盘比例": "control_ratio", "控盘比例": "control_ratio",
        "大单净量": "big_order_net",
    }

    def __init__(self, folder: str = "data/ths_manual"):
        self.folder = folder
        self.df = self._load_all()

    def _load_all(self) -> pd.DataFrame:
        frames = []
        for f in sorted(glob.glob(os.path.join(self.folder, "*.csv"))
                        + glob.glob(os.path.join(self.folder, "*.xlsx"))):
            d = pd.read_excel(f) if f.endswith("xlsx") else pd.read_csv(f)
            d = d.rename(columns={k: v for k, v in self.COLMAP.items() if k in d.columns})
            if "code" not in d.columns:
                continue
            if "date" not in d.columns:   # 无日期列 → 用文件名中的日期
                d["date"] = os.path.basename(f).split(".")[0]
            frames.append(d[["code", "date",
                             *[c for c in ("control_ratio", "big_order_net")
                               if c in d.columns]]])
        if not frames:
            return pd.DataFrame(columns=["code", "date"])
        out = pd.concat(frames, ignore_index=True)
        out["code"] = out["code"].astype(str).str.zfill(6)
        out["date"] = out["date"].astype(str).str.replace("-", "")
        return out.drop_duplicates(["code", "date"], keep="last")

    def _latest(self, code: str, col: str) -> tuple[float | None, str | None]:
        d = self.df[self.df["code"] == code.split(".")[0]].sort_values("date")
        if len(d) == 0 or col not in d.columns:
            return None, None
        r = d.dropna(subset=[col]).iloc[-1]
        return float(r[col]), str(r["date"])

    def control_ratio(self, code: str) -> tuple[float | None, str | None]:
        """返回 (值, 数据日期)；None 表示无手工数据"""
        return self._latest(code, "control_ratio")

    def big_order_net(self, code: str) -> tuple[float | None, str | None]:
        return self._latest(code, "big_order_net")


# ============================================================
# 合成入口：C 优先，B 兜底，附数据新鲜度告警
# ============================================================
class CapitalFeed:
    def __init__(self, ak_feed: AkshareCapitalFeed, ths_feed: THSManualFeed,
                 stale_days: int = 1):
        self.ak = ak_feed
        self.ths = ths_feed
        self.stale_days = stale_days

    def _is_stale(self, data_date: str, today: str) -> bool:
        # 简化：自然日差 > stale_days+2 视为过期（跨周末宽松处理）
        return (pd.Timestamp(today) - pd.Timestamp(data_date)).days > self.stale_days + 2

    def control_ratio(self, code: str, today: str | None = None) -> dict:
        today = today or time.strftime("%Y%m%d")
        v, d = self.ths.control_ratio(code)
        if v is not None:
            return {"value": v, "source": "THS手工", "date": d,
                    "stale": self._is_stale(d, today)}
        return {"value": self.ak.control_proxy(code), "source": "东财代理",
                "date": today, "stale": False,
                "note": "代理口径，阈值需用calibrate_threshold重校准"}

    def big_order_net(self, code: str, today: str | None = None) -> dict:
        today = today or time.strftime("%Y%m%d")
        v, d = self.ths.big_order_net(code)
        if v is not None:
            return {"value": v, "source": "THS手工", "date": d,
                    "stale": self._is_stale(d, today)}
        return {"value": self.ak.big_order_net(code), "source": "东财",
                "date": today, "stale": False}


# ============================================================
# 测试：合成数据验证 B/C 优先级与降级逻辑
# ============================================================
if __name__ == "__main__":
    import tempfile

    # ---- 构造 akshare 返回样式的合成数据 ----
    def mock_fetch(code6, market):
        return pd.DataFrame({
            "日期": ["2026-07-20", "2026-07-21"],
            "主力净流入-净额": [1.2e8, 0.8e8],
            "超大单净流入-净额": [0.7e8, 0.5e8],
            "大单净流入-净额": [0.5e8, 0.3e8],
            "主力净流入-净占比": [6.5, 4.2],
        })

    tmp = tempfile.mkdtemp()
    ak_feed = AkshareCapitalFeed(cache_dir=tmp, fetcher=mock_fetch)

    # ---- 构造 C 路径手工文件：只给 600519 导入了控盘比例 ----
    os.makedirs(tmp + "/ths", exist_ok=True)
    pd.DataFrame({"股票代码": ["600519"], "日期": ["20260721"],
                  "主力控盘比例": [38.5]}).to_csv(tmp + "/ths/20260721.csv", index=False)
    ths_feed = THSManualFeed(folder=tmp + "/ths")

    cap = CapitalFeed(ak_feed, ths_feed)

    print("600519（有手工数据）:", cap.control_ratio("600519", "20260721"))
    print("600519（次日，手工数据过期检查）:", cap.control_ratio("600519", "20260730"))
    print("300750（无手工数据→东财代理）:", cap.control_ratio("300750", "20260721"))
    print("600519 大单净量（东财=超大单+大单）:", cap.big_order_net("600519"))
    print("阈值校准示例 top20%:", AkshareCapitalFeed.calibrate_threshold(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
