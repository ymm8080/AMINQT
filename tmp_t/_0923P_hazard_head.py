# -*- coding: utf-8 -*-
"""0923P 首板后路径 hazard 研究 — 第三臂 = 模型头 (前两臂规则分桶格均判死).

问题: 用模型能否比规则格更好地预测「首板(D0)后剩余窗内再封板」?
判据: 加入 D+1 路径特征是否比只用 D0 特征有**增量**。

两臂 (同 params, 同切分):
  D0-only  = 只用 D0 特征 (面板 D0 行 + 事件表 D0 侧栏)
  D0+D1    = D0 特征 + 17 族 D+1 路径特征
TR 段内切分: DualTrackTrainer.split_window(df, 770) 语义 (train / es=验证 / test),
  训练用 train, 早停用 es (验证集在训练集之后, 不重叠); TE 段评估。

判词口径 (0912 范式, 根口径 0923: T+3/T+5/T+10 逐档各判一次):
  D0+D1 相对 D0-only ⇒ TE AUC Δ >= +0.005 且 Top-decile Δ >= +1.0pp ⇒ 有增量(值得继续)
  任一指标下降 (AUC Δ <= -0.005 或 Top-decile Δ <= -1.0pp)         ⇒ 负增量, 判死
  其余                                                            ⇒ 零增量(噪音带)

铁规: 原生 lgb.train (LightGBM 4.7.0 sklearn API 早停失真); 勿前视; 种子固定42; WORM 落盘。

────────────────────────────────────────────────────────────────────────────
方向自查 (shift(-j) = 下一行 = D+j; 个股自身 K 线行序顺延, 停牌自动跳过):
  close_1/open_1/pct_1/pre_1     = shift(-1)   = D+1              ✓
  high_1/low_1/vol_1/wr_p1/p90_p1= shift(-1)   = D+1              ✓
  close_2..5 / pct_2..5          = shift(-2..-5)= D+2..D+5         ✓
  vol_m1..m5                     = shift(+1..+5)= D0 前1..5日(不含D0) ✓
  vol5pre = mean(vol_m1..m5)     = D0 前5日均量 (不含 D0 当日)      ✓
  D+1 路径特征 只吃 D0/D+1 数据; 标签 h5 吃 pct_2..pct_5 (D+2..D+5) ⇒ 无前视 ✓
  宇宙条件 uni 含 pct_1<9.5 (D+1 未再封板) ⇒ 标签窗从 D+2 起才是干净 hazard ✓
────────────────────────────────────────────────────────────────────────────
用法: python tmp_t/_0923P_hazard_head.py   (单次跑完两臂; 哨兵已入 HEAVY_SENTINELS)
"""

import gc
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

np.random.seed(42)  # 铁规: 固定种子

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from config.settings import (
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

TAG = "hazard_head_0923P"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_0923P_hazard_head.log")
REPORT_DIR = data_others_path("diag")
EV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_0922_chain2_events.parquet")

HC, HCD = "001216", pd.Timestamp("2026-09-15")
TR_END = pd.Timestamp("2025-12-31")
TE_START = pd.Timestamp("2026-01-01")
TE_END = pd.Timestamp("2026-09-21")
BRD = 9.5                       # 硬编码涨停阈, 不用面板 up_limit_raw (两臂口径一致)
WINDOW = 770                    # DualTrackTrainer.split_window 窗口

# 面板只读需要的列 (内存友好)
PANEL_COLS = ["date", "symbol", "open", "high", "low", "close", "pre_close", "pctChg",
              "volume", "winner_ratio", "pct_90_con", "turnover_rate", "volume_ratio",
              "sw_l2_name"]

D0_PANEL_FEATS = ["pctChg", "volume_ratio", "winner_ratio", "pct_90_con", "turnover_rate",
                  "close", "high", "low", "open", "pre_close"]   # 面板 D0 行 (均存在)
D0_EV_FEATS = ["wr1", "p90", "stair", "pret", "t5"]             # 事件表 D0 侧栏 (=板前/当日信息)
D1_FEATS = ["volr", "volr5", "dd_high", "ret_c0", "cpos", "gap1", "amp1", "pct1", "zha",
            "wr1c", "dwr", "p90c", "dp90", "wr1pre", "gap0", "promo", "indb"]  # 17 族

PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05,
              min_child_samples=50, seed=42, metric="binary_logloss", verbosity=-1)
NUM_BOOST_ROUND = 400
EARLY_STOP = 50

N_OF = {"T+3": 3, "T+5": 5, "T+10": 10}          # 档名 → 出场日
COST_ROUND_TRIP = 0.0025                          # 与 0923Q 收益门同口径 (佣金+滑点+印花税量级)

log = logging.getLogger(TAG)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def _worm(name: str, payload: dict) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", path)
    return str(path)


def load_events() -> pd.DataFrame:
    """首板事件宇宙 (抄 _0923_post1_cells.py:53-61; d0 已由上游 _0922_chain2_main 定义, 不重算)."""
    ev = pd.read_parquet(EV_PATH, columns=["symbol", "d0", "forced"] + D0_EV_FEATS)
    ev["d0"] = pd.to_datetime(ev["d0"])
    ev["symbol"] = ev["symbol"].astype(str)
    ev = ev[~((ev["symbol"] == HC) & (ev["d0"] == HCD) & ev["forced"])]
    ev = ev.drop_duplicates(["symbol", "d0"], keep="first").reset_index(drop=True)
    ev = ev[(ev["d0"] >= "2023-01-01") & (ev["d0"] <= TE_END)]
    ev = ev[ev["symbol"].str[:2].isin(["00", "60"])].reset_index(drop=True)
    log.info("[load] events n=%d  d0 %s..%s", len(ev), ev["d0"].min().date(), ev["d0"].max().date())
    return ev


def load_panel(symbols: set) -> pd.DataFrame:
    """面板 (仅事件 symbol; 主板 00/60; 2023起). 只读 PANEL_COLS. OHLCV 校验铁规."""
    P = pq.read_table(PANEL_V3_PATH, columns=PANEL_COLS).to_pandas()
    P["symbol"] = P["symbol"].astype(str)
    P = P[P["symbol"].str[:2].isin(["00", "60"]) & P["symbol"].isin(symbols)]
    P["date"] = pd.to_datetime(P["date"])
    P = P[P["date"] >= "2023-01-01"]
    P = P.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    viol = int(((P["high"] < P["low"]) |
                (P["high"] < P[["open", "close"]].max(axis=1)) |
                (P["low"] > P[["open", "close"]].min(axis=1)) |
                (P["volume"] < 0)).sum())   # 铁规: 校验不静默丢弃
    log.info("[load] 面板(事件symbol,主板,2023+) 行=%d  OHLCV违例=%d", len(P), viol)
    return P


def build_frame(ev: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """事件级合并: D0 行 + D+1 路径列 + 标签列 + promo/indb. 抄 _0923_post1_cells.py:78-125."""
    g = P.groupby("symbol", sort=False)
    for i in range(1, 6):                       # 后向 D+i (shift(-i) = 下一行)
        P[f"close_{i}"] = g["close"].shift(-i)
        P[f"open_{i}"] = g["open"].shift(-i)
        P[f"pct_{i}"] = g["pctChg"].shift(-i)
        P[f"pre_{i}"] = g["pre_close"].shift(-i)
    for c, name in (("high", "high_1"), ("low", "low_1"), ("volume", "vol_1"),
                    ("winner_ratio", "wr_p1"), ("pct_90_con", "p90_p1")):
        P[name] = g[c].shift(-1)                # = D+1
    for i in range(6, 11):                      # 根口径 T+3/+5/+10: 标签窗要吃到 pct_10
        P[f"pct_{i}"] = g["pctChg"].shift(-i)
    for i in range(1, 6):                       # 前向 D0-i → vol5pre (含不含D0: 用 m1..m5 = D0前1..5)
        P[f"vol_m{i}"] = g["volume"].shift(i)
    P["vol5pre"] = P[[f"vol_m{i}" for i in range(1, 6)]].mean(axis=1)

    need = (["symbol", "date"] + D0_PANEL_FEATS + ["sw_l2_name", "volume",
            "vol_1", "vol5pre", "high_1", "low_1", "wr_p1", "p90_p1"]
            + [f"{c}_{i}" for i in range(1, 6) for c in ("close", "open", "pct", "pre")]
            + [f"pct_{i}" for i in range(6, 11)])
    M = ev.merge(P[need], left_on=["symbol", "d0"], right_on=["symbol", "date"], how="left")
    M = M.drop(columns=["date"])

    # promo_yd(D0日) 修正版 (抄 post1_cells:78-91) + ind_boards(D0板块涨停数, 抄:92-93)
    board = P["pctChg"] >= BRD
    dates_all = sorted(P["date"].unique())
    next_map = {dates_all[i - 1]: dates_all[i] for i in range(1, len(dates_all))}
    bd = P.loc[board, ["date", "symbol"]].assign(b=1)
    bd_next = bd.copy()
    bd_next["date"] = bd_next["date"].map(next_map)
    bd_next = bd_next.rename(columns={"b": "boarded_yd"})
    promo = bd.merge(bd_next, on=["date", "symbol"], how="left")
    nprev = bd_next.groupby("date").size()
    both = promo.groupby("date")["boarded_yd"].sum()
    eco = pd.DataFrame({"n_boards": bd.groupby("date").size()})
    eco["promo_yd"] = ((both / nprev).reindex(eco.index)).fillna(0)
    eco = eco.reset_index()[["date", "promo_yd"]]
    M = M.merge(eco, left_on="d0", right_on="date", how="left").drop(columns=["date"])
    M["promo_yd"] = M["promo_yd"].fillna(0)

    ind = (P.loc[board].assign(sw=lambda x: x["sw_l2_name"].fillna("NA"))
           .groupby(["date", "sw"]).size().rename("ind_boards").reset_index())
    M["sw"] = M["sw_l2_name"].fillna("NA")
    M = M.merge(ind, left_on=["d0", "sw"], right_on=["date", "sw"], how="left").drop(columns=["date"])
    M["ind_boards"] = M["ind_boards"].fillna(0)
    log.info("[load] D0 面板未匹配=%d  promo缺=%d",
             int(M["close"].isna().sum()), int(M["promo_yd"].isna().sum()))
    return M


def build_features(M: pd.DataFrame):
    """返回 (X_d0, X_all, lab, uni, fill_ratio); lab={horizon: (标签, 窗完整掩码)}.
    D+1 路径特征 抄 _0923_post1_cells.py:156-178."""
    c0, h0, o0, pre0, v0 = (M[k].to_numpy() for k in ("close", "high", "open", "pre_close", "volume"))
    c1, h1, l1, o1, pre1, v1 = (M[k].to_numpy() for k in
                                ("close_1", "high_1", "low_1", "open_1", "pre_1", "vol_1"))
    lim1 = np.round(pre1 * 1.10, 2)
    rng1 = h1 - l1
    pct1 = M["pct_1"].to_numpy()

    F = pd.DataFrame(index=M.index)
    F["volr"] = v1 / v0
    F["volr5"] = v1 / M["vol5pre"].to_numpy()
    F["dd_high"] = (c1 - h0) / h0
    F["ret_c0"] = c1 / c0 - 1
    F["cpos"] = np.where(rng1 > 0, (c1 - l1) / np.where(rng1 > 0, rng1, np.nan), np.nan)
    F["gap1"] = o1 / c0 - 1
    F["amp1"] = rng1 / pre1
    F["pct1"] = pct1
    F["zha"] = ((h1 >= lim1 - 0.005) & (c1 < lim1 - 0.005)).astype(float)
    F["wr1c"] = M["wr_p1"].to_numpy()
    F["dwr"] = F["wr1c"] - M["winner_ratio"].to_numpy()
    F["p90c"] = M["p90_p1"].to_numpy()
    F["dp90"] = F["p90c"] - M["pct_90_con"].to_numpy()
    F["wr1pre"] = pd.to_numeric(M["wr1"], errors="coerce").to_numpy()
    F["gap0"] = o0 / pre0 - 1
    F["promo"] = M["promo_yd"].to_numpy()
    F["indb"] = M["ind_boards"].to_numpy()

    X_d0 = M[D0_PANEL_FEATS + D0_EV_FEATS].copy()
    for c in X_d0.columns:
        X_d0[c] = pd.to_numeric(X_d0[c], errors="coerce")
    X_all = pd.concat([X_d0, F], axis=1)

    has1 = ~np.isnan(M["close_1"].to_numpy()) & ~np.isnan(pct1)
    winok = (~np.isnan(M[["pct_2", "pct_3", "pct_4", "pct_5"]].to_numpy()).any(1)
             & ~np.isnan(M["close_5"].to_numpy()))
    uni = has1 & (pct1 < BRD) & winok
    pJ = {j: M[f"pct_{j}"].to_numpy() for j in range(2, 11)}

    def _hit(js):
        return np.logical_or.reduce([pJ[j] >= BRD for j in js])

    # 根口径 (用户令 0923): 三档全给. 各档只在其窗口**完整存在**的行上评估 —
    # 缺行 (停牌/未上市满) 若当 False 会静默压低基础率, 故逐档给 ok 掩码并报 n.
    h3, h5, h10 = _hit((2, 3)), _hit((2, 3, 4, 5)), _hit(range(2, 11))
    win10 = winok & ~np.isnan(M[[f"pct_{j}" for j in range(6, 11)]].to_numpy()).all(1)
    lab = {"T+3": (h3, winok), "T+5": (h5, winok), "T+10": (h10, win10)}

    # 诊断: D+1 收盘未封死 = 可成交 (uni 已含 pct_1<9.5, 但 ST 5% 等边角可能漏)
    fill = has1 & ~(c1 >= lim1 - 0.005)
    fill_ratio = float(fill[uni].mean()) if uni.sum() else float("nan")
    return X_d0, X_all, lab, uni, fill_ratio


def train_arm(X: pd.DataFrame, y: np.ndarray, idx_tr, idx_va, idx_te):
    """原生 lgb.train (抄 scripts/_firstboard_pages.py:360-397). 返回 (best_iter, pred_te, pred_tr, pred_va)."""
    dtr = lgb.Dataset(X.loc[idx_tr], y[idx_tr], free_raw_data=False)
    dva = lgb.Dataset(X.loc[idx_va], y[idx_va], reference=dtr, free_raw_data=False)
    bst = lgb.train(PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    best_iter = int(bst.best_iteration or NUM_BOOST_ROUND)   # 原生属性, 无下划线
    pred_te = bst.predict(X.loc[idx_te])                     # 裸 predict, 不传 num_iteration
    pred_tr = bst.predict(X.loc[idx_tr])
    pred_va = bst.predict(X.loc[idx_va])
    return best_iter, pred_te, pred_tr, pred_va


def fwd_ret(M: pd.DataFrame, n: int) -> np.ndarray:
    """D+1 收盘 → D+n 收盘 链式收益 (与 _0923Q_hazard_benefit 同口径).
    入场 D+1 收盘可成交 (uni 已含 pct_1<9.5); 只用 D+1 及以后数据, 无前视."""
    r = np.ones(len(M))
    for j in range(2, n + 1):
        r = r * (1.0 + M[f"pct_{j}"].to_numpy() / 100.0)
    return r - 1.0


def evaluate(y: np.ndarray, pred: np.ndarray, dates: pd.Series) -> dict:
    """TE 指标: AUC / 基础率 / Top-decile / Top-5% / 日截面 rank-IC."""
    q = pd.Series(pred).rank(pct=True).to_numpy()
    ics = []
    g = pd.DataFrame({"d": dates.to_numpy(), "p": pred, "y": y}).groupby("d")
    for _, sub in g:
        if len(sub) >= 5 and sub["y"].nunique() > 1:
            ics.append(sub["p"].rank().corr(sub["y"].rank()))
    return dict(
        n=int(len(y)), n_days=int(dates.nunique()), base_rate=float(y.mean()),
        auc=float(roc_auc_score(y, pred)),
        top_decile=float(y[q > 0.9].mean()) if (q > 0.9).any() else float("nan"),
        top5=float(y[q > 0.95].mean()) if (q > 0.95).any() else float("nan"),
        rank_ic=float(np.nanmean(ics)) if ics else float("nan"), rank_ic_days=len(ics),
    )


def main() -> int:
    _setup_logging()
    from scripts._run_guard import find_conflicts

    conflicts = find_conflicts()
    if conflicts:
        c = conflicts[0]
        log.error("[guard] 重活冲突: %s (PID %d) 在跑 — 退出", c["sentinel"], c["pid"])
        return 3
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB)  # 不足时 SystemExit(2)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB, RETRAIN_RAM_GUARD_POLL_S)

    ev = load_events()
    P = load_panel(set(ev["symbol"]))
    M = build_frame(ev, P)
    del P
    gc.collect()

    X_d0, X_all, lab, uni, fill_ratio = build_features(M)
    d0 = M["d0"]
    log.info("[uni] 事件 n=%d  uni(单板非一字)总数=%d  fill(uni内可成交)占比=%.1f%%",
             len(M), int(uni.sum()), 100 * fill_ratio)

    # 根口径 (用户令 0923): 结果必须同时给 T+3/T+5/T+10. 三档正样本定义不同 (窗越长
    # 基础率越高), 合用一个模型等于混标签 ⇒ 逐档独立切窗独立训练独立判词.
    reports = {}
    for hname, (hlab, hok) in lab.items():
        m = uni & hok
        tr_m = (d0 <= TR_END).to_numpy() & m
        te_m = ((d0 >= TE_START) & (d0 <= TE_END)).to_numpy() & m
        if not tr_m.sum() or not te_m.sum():
            log.error("[%s] 样本不足 TR=%d TE=%d — 跳过", hname, int(tr_m.sum()), int(te_m.sum()))
            continue
        log.info("[%s][seg] TR n=%d  TE n=%d  TE基础率=%.1f%%",
                 hname, int(tr_m.sum()), int(te_m.sum()), 100 * hlab[te_m].mean())

        # TR 段内按时间切 (split_window 语义): 验证集=es 段, 在 train 之后
        splitdf = pd.DataFrame({"date": d0.to_numpy()})[tr_m]
        segs = DualTrackTrainer.split_window(splitdf, WINDOW)
        idx_tr = segs["train"].index
        idx_va = segs["es"].index
        idx_te = np.where(te_m)[0]
        assert len(idx_tr) and len(idx_va), f"{hname} 切分不足: train/es 段为空"
        assert d0.loc[idx_tr].max() < d0.loc[idx_va].min(), f"{hname} 前视: 验证集不在训练集之后"
        log.info("[%s][split] train=%d (..%s)  va/es=%d (%s..%s)  test=%d",
                 hname, len(idx_tr), d0.loc[idx_tr].max().date(),
                 len(idx_va), d0.loc[idx_va].min().date(), d0.loc[idx_va].max().date(), len(idx_te))

        y = hlab.astype(int)
        y_te = np.asarray(y)[idx_te]
        te_dates = d0.iloc[idx_te].reset_index(drop=True)
        # 经济读数 (收益门 0923Q 口径): 先知上界=组A收益, 可交易=模型头部十分位收益
        te_ret = fwd_ret(M, N_OF[hname])[idx_te]
        cov = np.isfinite(te_ret)
        arms = {}
        for arm, X in (("D0-only", X_d0), ("D0+D1", X_all)):
            t0 = time.time()
            best_iter, pred_te, pred_tr, pred_va = train_arm(X, y, idx_tr, idx_va, idx_te)
            rep = evaluate(y[idx_te], pred_te, te_dates)
            rep["best_iter"] = best_iter
            rep["n_features"] = int(X.shape[1])
            rep["train_auc"] = float(roc_auc_score(y[idx_tr], pred_tr))
            rep["va_auc"] = float(roc_auc_score(y[idx_va], pred_va))
            q = pd.Series(pred_te).rank(pct=True).to_numpy()
            dec = (q > 0.9) & cov
            rep["decile_n"] = int(dec.sum())
            rep["decile_ret"] = float(te_ret[dec].mean()) if dec.any() else float("nan")
            rep["decile_ret_net"] = rep["decile_ret"] - COST_ROUND_TRIP
            arms[arm] = rep
            log.info("[%s][arm] %-8s best_iter=%3d feats=%2d | TE AUC=%.4f 基础率=%.1f%% "
                     "TopDec=%.1f%% Top5=%.1f%% rankIC=%.4f | %.1fs",
                     hname, arm, best_iter, X.shape[1], rep["auc"], 100 * rep["base_rate"],
                     100 * rep["top_decile"], 100 * rep["top5"], rep["rank_ic"], time.time() - t0)
            log.info("[%s][econ] %-8s 头部十分位 n=%d 均值收益=%+.2f%% (扣成本%+.2f%%) | "
                     "先知上界(组A)=%+.2f%% | 全池=%+.2f%%",
                     hname, arm, rep["decile_n"], 100 * rep["decile_ret"], 100 * rep["decile_ret_net"],
                     100 * np.nanmean(te_ret[y_te == 1]), 100 * np.nanmean(te_ret))
            gc.collect()

        d_auc = arms["D0+D1"]["auc"] - arms["D0-only"]["auc"]
        d_top = 100 * (arms["D0+D1"]["top_decile"] - arms["D0-only"]["top_decile"])
        d_ic = arms["D0+D1"]["rank_ic"] - arms["D0-only"]["rank_ic"]
        if d_auc >= 0.005 and d_top >= 1.0:
            verdict = "有增量(值得继续)"
        elif d_auc <= -0.005 or d_top <= -1.0:
            verdict = "负增量, 判死"
        else:
            verdict = "零增量(噪音带)"
        log.info("=" * 88)
        log.info("[%s][verdict] D0+D1 - D0-only: ΔAUC=%+.4f (线+0.005)  ΔTopDec=%+.2fpp (线+1.0pp)  ΔrankIC=%+.4f",
                 hname, d_auc, d_top, d_ic)
        log.info("[%s][verdict] ⇒ %s", hname, verdict)

        d_dec = 100 * (arms["D0+D1"]["decile_ret"] - arms["D0-only"]["decile_ret"])
        log.info("[%s][econ verdict] D0+D1 头部十分位收益=%+.2f%% (扣成本%+.2f%%) vs 全池=%+.2f%%; "
                 "相对 D0-only Δ=%+.2fpp ⇒ %s",
                 hname, 100 * arms["D0+D1"]["decile_ret"], 100 * arms["D0+D1"]["decile_ret_net"],
                 100 * float(np.nanmean(te_ret)), d_dec,
                 "经济上站得住" if arms["D0+D1"]["decile_ret_net"] > float(np.nanmean(te_ret))
                 else "经济上站不住(扣成本后不及全池)")

        reports[hname] = dict(
            verdict=verdict,
            delta=dict(auc=d_auc, top_decile_pp=d_top, rank_ic=d_ic,
                       threshold=dict(auc=0.005, top_decile_pp=1.0)),
            arms=arms,
            econ=dict(oracle_ret=float(np.nanmean(te_ret[y_te == 1])),
                      universe_ret=float(np.nanmean(te_ret)),
                      cost=COST_ROUND_TRIP, delta_decile_pp=d_dec),
            rows=dict(tr=int(tr_m.sum()), te=int(te_m.sum()),
                      train=int(len(idx_tr)), va=int(len(idx_va)), test=int(len(idx_te))),
            split=dict(train_end=str(d0.loc[idx_tr].max().date()),
                       va_start=str(d0.loc[idx_va].min().date()),
                       va_end=str(d0.loc[idx_va].max().date())),
        )

    log.info("=" * 88)
    for hname, rep in reports.items():
        log.info("[summary] %-5s ΔAUC=%+.4f  ΔTopDec=%+.2fpp  ⇒ %s",
                 hname, rep["delta"]["auc"], rep["delta"]["top_decile_pp"], rep["verdict"])

    payload = dict(
        tag=TAG, reports=reports, horizons=sorted(reports),
        fill_ratio_uni=fill_ratio,
        rows=dict(events=int(len(M)), uni=int(uni.sum())),
        window_bounds=dict(tr_end=str(TR_END.date()), te_start=str(TE_START.date()),
                           te_end=str(TE_END.date())),
        features=dict(D0_panel=D0_PANEL_FEATS, D0_events=D0_EV_FEATS, D1_path=D1_FEATS),
        params=PARAMS, seed=42, brd=BRD, window=WINDOW,
        note=("根口径: T+3/T+5/T+10 逐档独立切窗独立训练独立判词; 各档只在其标签窗**完整存在**"
              "的行上评估 (T+3/T+5 用 winok, T+10 含 pct_6..pct_10 全非缺), 故各档 n 不等; "
              "D0 特征=面板D0行 + 事件侧栏; D+1 路径=17族; promo/indb 由面板 pctChg/sw_l2_name 复算"
              "(面板无 promo_yd 列); fill=uni 内 D+1 收盘未封死(可成交)占比."),
    )
    _worm(TAG, payload)
    log.info("[done] %s", time.strftime("%H:%M:%S"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
