# -*- coding: utf-8 -*-
"""0923Q hazard 收益门 — 先问「这个事件有没有钱可赚」, 再决定要不要跑模型 A/B。

问题背景: 0923P 做的是「首板 D0 后剩余窗内再封板」的二分类模型增量 A/B。模型 AUC 再高,
若"再封板"这件事本身不带来更高的前向收益, 就没有可捕获的经济梯度 ⇒ A/B 不值得跑。

门的口径 (可证伪, 与 0923P 同宇宙同标签定义):
  宇宙 uni = 单板非一字 (D+1 未再封板, pct_1 < 9.5) + 标签窗完整 ⇒ D+1 收盘**可成交**
  入场 = D+1 收盘价
  出场 = D+n 收盘价 (n = 3/5/10, 根口径三档全给)
  组 A = 剩余窗内再封板 (h=1);  组 B = 不再封板 (h=0)

判据:
  Δ(A-B) <= 0  或  全池收益 >= A 收益              ⇒ 判死 (事件无经济梯度, 模型无用武之地)
  Δ(A-B) > 0 但 A 扣成本后 <= 全池扣成本后          ⇒ 判死 (有梯度但不够覆盖成本)
  否则                                            ⇒ 有戏, 才谈跑 A/B

方向自查: pctChg_j = close_j/pre_close_j - 1 (pre_close_j = close_{j-1}), 故
  从 D+1 收盘到 D+n 收盘的收益 = prod_{j=2..n}(1 + pctChg_j/100) - 1, 链式精确。
  只用 D+1 及以后数据 + D0 当日标签条件, 无前视 ✓

用法: python tmp_t/_0923Q_hazard_benefit.py
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.settings import PANEL_V3_PATH, data_others_path

TAG = "hazard_benefit_0923Q"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "_0923Q_hazard_benefit.log")
REPORT_DIR = data_others_path("diag")
EV_PATH = os.path.join(HERE, "_0922_chain2_events.parquet")

HC, HCD = "001216", pd.Timestamp("2026-09-15")
TE_END = pd.Timestamp("2026-09-21")
BRD = 9.5
# 单边成本假设 (佣金+滑点) + 卖出印花税; 只作门的量级参考, 非结算金额
COST_ROUND_TRIP = 0.0025

log = logging.getLogger(TAG)


def _setup_logging():
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def load_events() -> pd.DataFrame:
    """抄 _0923P_hazard_head.load_events 的筛选口径 (不 import 它: 该模块 import lightgbm, 重)。"""
    ev = pd.read_parquet(EV_PATH, columns=["symbol", "d0", "forced"])
    ev["d0"] = pd.to_datetime(ev["d0"])
    ev["symbol"] = ev["symbol"].astype(str)
    ev = ev[~((ev["symbol"] == HC) & (ev["d0"] == HCD) & ev["forced"])]
    ev = ev.drop_duplicates(["symbol", "d0"], keep="first").reset_index(drop=True)
    ev = ev[(ev["d0"] >= "2023-01-01") & (ev["d0"] <= TE_END)]
    ev = ev[ev["symbol"].str[:2].isin(["00", "60"])].reset_index(drop=True)
    log.info("[load] events n=%d  d0 %s..%s", len(ev), ev["d0"].min().date(), ev["d0"].max().date())
    return ev


def build(M_ev: pd.DataFrame) -> pd.DataFrame:
    """只读 4 列面板 → 事件级 D+1..D+10 的 pctChg/close 位移, 复刻 0923P 的 uni 与标签。"""
    cols = ["symbol", "date", "pctChg", "close"]
    P = pq.read_table(PANEL_V3_PATH, columns=cols).to_pandas()
    P["symbol"] = P["symbol"].astype(str)
    P = P[P["symbol"].str[:2].isin(["00", "60"]) & P["symbol"].isin(set(M_ev["symbol"]))]
    P["date"] = pd.to_datetime(P["date"])
    P = P[P["date"] >= "2023-01-01"]
    P = P.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    g = P.groupby("symbol", sort=False)
    for i in range(1, 11):
        P[f"pct_{i}"] = g["pctChg"].shift(-i)
        P[f"close_{i}"] = g["close"].shift(-i)
    log.info("[load] 面板(事件symbol,主板,2023+) 行=%d", len(P))

    M = M_ev.merge(P[["symbol", "date", "close"] + [f"pct_{i}" for i in range(1, 11)]
                     + [f"close_{i}" for i in range(1, 11)]],
                   left_on=["symbol", "d0"], right_on=["symbol", "date"], how="left")
    M = M.drop(columns=["date"])
    # uni / 标签 与 0923P build_features 逐字对齐
    p1 = M["pct_1"].to_numpy()
    has1 = ~np.isnan(M["close_1"].to_numpy()) & ~np.isnan(p1)
    winok = (~np.isnan(M[["pct_2", "pct_3", "pct_4", "pct_5"]].to_numpy()).any(1)
             & ~np.isnan(M["close_5"].to_numpy()))
    M["uni"] = has1 & (p1 < BRD) & winok
    win10 = winok & ~np.isnan(M[[f"pct_{j}" for j in range(6, 11)]].to_numpy()).all(1)
    pJ = {j: M[f"pct_{j}"].to_numpy() for j in range(2, 11)}

    def _hit(js):
        return np.logical_or.reduce([pJ[j] >= BRD for j in js])

    M["h3"], M["h5"], M["h10"] = _hit((2, 3)), _hit((2, 3, 4, 5)), _hit(range(2, 11))
    M["ok3"] = M["ok5"] = winok
    M["ok10"] = win10
    return M


def fwd_ret(M: pd.DataFrame, n: int) -> np.ndarray:
    """D+1 收盘 → D+n 收盘 的链式收益 (扣 0 成本, 成本单独扣)。"""
    r = np.ones(len(M))
    for j in range(2, n + 1):
        r = r * (1.0 + M[f"pct_{j}"].to_numpy() / 100.0)
    return r - 1.0


def stats(M: pd.DataFrame, mask: np.ndarray, ret: np.ndarray) -> dict:
    v = ret[mask]
    v = v[~np.isnan(v)]
    if not len(v):
        return dict(n=0)
    d = M.loc[mask, "d0"]
    d = d[~np.isnan(ret[mask])]
    daily = pd.Series(v).groupby(d.to_numpy()).mean()      # 日均: 等权到天, 防日期聚簇
    return dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)),
                win_rate=float((v > 0).mean()), day_mean=float(daily.mean()), n_days=int(len(daily)))


def main() -> int:
    _setup_logging()
    ev = load_events()
    M = build(ev)
    uni = M["uni"].to_numpy()
    log.info("[uni] 单板非一字且窗完整 n=%d / 事件 %d = %.1f%%",
             int(uni.sum()), len(M), 100 * uni.mean())

    payload = dict(tag=TAG, cost_round_trip=COST_ROUND_TRIP, horizons={})
    for hname, n in (("T+3", 3), ("T+5", 5), ("T+10", 10)):
        hcol, okcol = {"T+3": ("h3", "ok3"), "T+5": ("h5", "ok5"), "T+10": ("h10", "ok10")}[hname]
        m = uni & M[okcol].to_numpy()
        lab = M[hcol].to_numpy().astype(bool)
        ret = fwd_ret(M, n)
        A, B, all_ = stats(M, m & lab, ret), stats(M, m & ~lab, ret), stats(M, m, ret)
        dA = A["mean"] - B["mean"]
        log.info("=" * 92)
        log.info("[%s] 入场=D+1收盘 出场=D+%d收盘 | A(再封板) n=%d 占%.1f%% 均值%+.2f%% 胜率%.1f%% | "
                 "B(不再封板) n=%d 均值%+.2f%% 胜率%.1f%% | 全池 n=%d 均值%+.2f%% 胜率%.1f%%",
                 hname, n, A["n"], 100 * A["n"] / all_["n"], 100 * A["mean"], 100 * A["win_rate"],
                 B["n"], 100 * B["mean"], 100 * B["win_rate"], all_["n"], 100 * all_["mean"],
                 100 * all_["win_rate"])
        log.info("[%s] Δ(A-B)=%+.2fpp | 日均: A %+.2f%% vs B %+.2f%% vs 全池 %+.2f%% | "
                 "扣成本(%.2f%%)后 A %+.2f%% vs 全池 %+.2f%%",
                 hname, 100 * dA, 100 * A["day_mean"], 100 * B["day_mean"], 100 * all_["day_mean"],
                 100 * COST_ROUND_TRIP, 100 * (A["mean"] - COST_ROUND_TRIP),
                 100 * (all_["mean"] - COST_ROUND_TRIP))
        if dA <= 0 or all_["mean"] >= A["mean"]:
            verdict = "判死(事件无经济梯度)"
        elif (A["mean"] - COST_ROUND_TRIP) <= (all_["mean"] - COST_ROUND_TRIP):
            verdict = "判死(有梯度但不够覆盖成本)"
        else:
            verdict = "有戏(可考虑跑 A/B)"
        log.info("[%s][verdict] Δ=%+.2fpp ⇒ %s", hname, 100 * dA, verdict)
        payload["horizons"][hname] = dict(A=A, B=B, universe=all_, delta_pp=100 * dA, verdict=verdict)

    payload["rows"] = dict(events=int(len(M)), uni=int(uni.sum()))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORT_DIR / f"{TAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", p)
    log.info("[done] %s", time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
