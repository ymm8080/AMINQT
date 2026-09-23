# -*- coding: utf-8 -*-
"""0923U hazard 阶段性收益检验 — 「收益是不是只在某些阶段存在? 选有利阶段能不能救」。

背景: 0923Q 收益门三档全过(事件 A-B 梯度 +11.6~+12.4pp), 0923P 模型三档全否,
0923S 定案亏损来自失败尾巴(D+1 振幅高 42%)。用户提出: 若收益是**阶段性**的,
可在模型里只选有益阶段。

两条必须守住的纪律 (否则结论是假的):
  1. **阶段必须事前可判**: 用交易时点已知的市场状态划分, 不用"回头挑好的那段"。
     定义 mom20 = 全市场日收益中位数 的 20 日滚动和 (等权大盘动量), 取 D0 当日值
     (D0 收盘即已知, 早于 D+1 入场) ⇒ 无前视。
  2. **阈值只能定在 TR 上**: 三分位切点由 TR(2023-01-03..2025-12-31) 冻结, 再套 TE。
     在 TE 上调阈值 = 又一种过拟合。

同时给两种"阶段"读法, 免得口径之争: 市场状态段(事前) + 日历年段(诊断用, 事后, 仅作稳定性)。

用法: python tmp_t/_0923U_hazard_phase.py
"""

import gc
import json
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import _0923P_hazard_head as H
from app.pipeline1.ram_guard import check_startup_gate
from config.settings import PANEL_V3_PATH, RETRAIN_RAM_GUARD_MIN_FREE_GB, data_others_path
from scripts._run_guard import find_conflicts

TAG = "hazard_phase_0923U"
DEC = 0.9
MOM_WIN = 20          # 市场动量回看交易日
N_TIER = 3            # 三分位

log = logging.getLogger(TAG)


def _setup_logging():
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(os.path.join(HERE, "_0923U_hazard_phase.log"),
                                  mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def market_momentum() -> pd.Series:
    """全市场等权日收益中位数 → MOM_WIN 日滚动和。只用当日及更早 ⇒ 无前视。"""
    P = pq.read_table(PANEL_V3_PATH, columns=["date", "pctChg"]).to_pandas()
    P["date"] = pd.to_datetime(P["date"])
    mkt = P.groupby("date")["pctChg"].median().sort_index()
    del P
    gc.collect()
    mom = mkt.rolling(MOM_WIN, min_periods=MOM_WIN).sum()
    log.info("[mkt] 日收益中位数 n=%d  %s..%s | mom20 有效 %d 日",
             len(mkt), mkt.index.min().date(), mkt.index.max().date(), int(mom.notna().sum()))
    return mom


def split_stats(te_ret: np.ndarray, y: np.ndarray, dec: np.ndarray, cost: float) -> dict:
    """一段内: n / 板率 / Δ(A-B) / 头部十分位收益(净)。"""
    ok = np.isfinite(te_ret)
    if not ok.any():
        return dict(n=0)
    r, yy, dd = te_ret[ok], y[ok], dec[ok]
    a, b = yy == 1, yy == 0
    head = dd
    hr = r[head]
    return dict(n=int(ok.sum()), n_days=0, a_rate=float(yy.mean()),
                delta_pp=100 * (r[a].mean() - r[b].mean()) if a.any() and b.any() else float("nan"),
                univ_ret=float(r.mean()),
                head_n=int(head.sum()),
                head_ret=float(hr.mean()) if head.any() else float("nan"),
                head_ret_net=float(hr.mean()) - cost if head.any() else float("nan"),
                head_a_rate=float(yy[head].mean()) if head.any() else float("nan"))


def main() -> int:
    _setup_logging()
    conflicts = find_conflicts()
    if conflicts:
        log.error("[guard] 重活冲突: %s (PID %d) — 退出", conflicts[0]["sentinel"], conflicts[0]["pid"])
        return 3
    check_startup_gate(int(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024 ** 3))

    mom = market_momentum()

    ev = H.load_events()
    P = H.load_panel(set(ev["symbol"]))
    M = H.build_frame(ev, P)
    del P
    gc.collect()

    X_d0, X_all, lab, uni, fill_ratio = H.build_features(M)
    d0 = M["d0"]
    # 阶段变量: D0 收盘已知的市场动量 (早于 D+1 入场 ⇒ 事前)
    M["mom20"] = pd.to_datetime(d0).map(mom).to_numpy()
    log.info("[uni] n=%d  fill=%.1f%%  mom20 非空=%.1f%%",
             int(uni.sum()), 100 * fill_ratio, 100 * M.loc[uni, "mom20"].notna().mean())

    payload = dict(tag=TAG, decile=DEC, cost=H.COST_ROUND_TRIP, mom_win=MOM_WIN,
                   n_tier=N_TIER, horizons={})
    for hname, (hlab, hok) in lab.items():
        n_out = H.N_OF[hname]
        m = uni & hok
        tr_m = (d0 <= H.TR_END).to_numpy() & m
        te_m = ((d0 >= H.TE_START) & (d0 <= H.TE_END)).to_numpy() & m
        splitdf = pd.DataFrame({"date": d0.to_numpy()})[tr_m]
        segs = H.DualTrackTrainer.split_window(splitdf, H.WINDOW)
        idx_tr, idx_va, idx_te = segs["train"].index, segs["es"].index, np.where(te_m)[0]
        y = hlab.astype(int)
        y_te = np.asarray(y)[idx_te]
        te_ret = H.fwd_ret(M, n_out)[idx_te]
        te_mom = M["mom20"].to_numpy()[idx_te]

        # 三分位切点只由 TR 冻结
        mom_tr = M["mom20"].to_numpy()[idx_tr]
        mom_tr = mom_tr[np.isfinite(mom_tr)]
        q = np.quantile(mom_tr, [1 / N_TIER, 2 / N_TIER])
        log.info("=" * 100)
        log.info("[%s] TR mom20 三分位切点 = %+.2f / %+.2f (只由 TR 定, 套到 TE)", hname, q[0], q[1])

        tier_te = np.digitize(te_mom, q)          # 0=低 1=中 2=高
        tier_te = np.where(np.isfinite(te_mom), tier_te, -1)

        preds = {}
        for arm, X in (("D0-only", X_d0), ("D0+D1", X_all)):
            _, pred_te, _, _ = H.train_arm(X, y, idx_tr, idx_va, idx_te)
            preds[arm] = pd.Series(pred_te).rank(pct=True).to_numpy() > DEC
            gc.collect()

        for arm in ("D0-only", "D0+D1"):
            dec = preds[arm]
            log.info("-" * 100)
            log.info("[%s][%s] 全 TE: n=%d 板率=%.1f%% Δ(A-B)=%+.2fpp | 头部 n=%d 收益=%+.2f%% (净%+.2f%%)",
                     hname, arm, len(y_te), 100 * y_te.mean(),
                     *[100 * x for x in (split_stats(te_ret, y_te, dec, 0)["delta_pp"],)],
                     int(dec.sum()), 100 * te_ret[dec].mean(), 100 * (te_ret[dec].mean() - H.COST_ROUND_TRIP))
            rows = {}
            for t, nm in ((0, "低动量"), (1, "中动量"), (2, "高动量")):
                sel = tier_te == t
                if not sel.any():
                    continue
                s = split_stats(te_ret[sel], y_te[sel], dec[sel], H.COST_ROUND_TRIP)
                s["n_days"] = int(pd.Series(d0.to_numpy()[idx_te][sel]).nunique())
                rows[nm] = s
                log.info("[%s][%s][%s] n=%d(%d日) 板率=%.1f%% Δ(A-B)=%+.2fpp | 头部 n=%d 板率=%.1f%% "
                         "收益=%+.2f%% (净%+.2f%%) vs 段内全池=%+.2f%%",
                         hname, arm, nm, s["n"], s["n_days"], 100 * s["a_rate"], s["delta_pp"],
                         s["head_n"], 100 * s["head_a_rate"], 100 * s["head_ret"],
                         100 * s["head_ret_net"], 100 * s["univ_ret"])
            # 日历年段 (事后诊断, 只作稳定性参考, 不作可交易判据)
            for yr in sorted(pd.DatetimeIndex(d0.to_numpy()[idx_te]).year.unique()):
                sel = pd.DatetimeIndex(d0.to_numpy()[idx_te]).year == yr
                s = split_stats(te_ret[sel], y_te[sel], dec[sel], H.COST_ROUND_TRIP)
                log.info("[%s][%s][%d年] n=%d 板率=%.1f%% Δ(A-B)=%+.2fpp | 头部 n=%d 收益=%+.2f%% (净%+.2f%%)",
                         hname, arm, yr, s["n"], 100 * s["a_rate"], s["delta_pp"],
                         s["head_n"], 100 * s["head_ret"], 100 * s["head_ret_net"])
                rows.setdefault("_by_year", {})[str(yr)] = s
            payload["horizons"].setdefault(hname, {})[arm] = dict(
                cut=list(map(float, q)), tiers=rows)

    payload["rows"] = dict(events=int(len(M)), uni=int(uni.sum()))
    out = data_others_path("diag")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{TAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", p)
    log.info("[done] %s", time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
