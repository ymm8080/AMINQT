# -*- coding: utf-8 -*-
"""首板点名页 + 板前哨页 (GENIOUS 两张新页的数据层, 0922 用户令).

首板点名页 v2 (0923 用户令): 全量当日事件按 T+3板概率(p_k3)降序 (不再 Top-N 截断,
  超 FB_MAX_ROWS 截断但冠军格★行豁免), 列含 T+5板概率(p_k2)/校准档位(模型分→TE实测板率);
  三概率头 p_k3(3日内续板)/p_k2(5日内续板)/p_next(次日封板),
  模型训练截止 2025-12-31 落盘 models/firstboard/ (booster + meta.json),
  **观察页勿当买入清单** — 次日开盘买入口径 TE 笔均 −2.40%/笔 (W11),
  冠军格可成交子集 −4.17%/笔 (W20, 反向选择: 买得到的是弱冠军).
板前哨页 (v4 0923 用户令): 滚动20日监视名单 — 历史命中日志 (近20日每个深睡签名日一行, 每股可多行),
  仅 今日点火旗≠'' 的股才进 Excel (其余只落后台表 preboard_watch_hits_*.csv, 全量);
  次数10日/次数20日 = 过去10/20个交易日命中总数, 仅上下文列勿筛选 (回测: 点火前命中数不预测);
  反信号组只作可见性, 点火旗/通道优先=今日口径, 勿作买入触发 (回测: 点火后各天数次日进 TE 全负 −0.7~−1.6%).
点名页 LHB 席位标注列 (0923): W14 A/B 终判席位特征无模型增量 (pre20=平 / D0=负) → 按规则降级为
  纯标注列 (LHB上榜日数/净买亿/游资席位/机构在场 20日窗 + D0 当日两列); 绝不进 FEATS/训练/闸,
  无数据留空绝不删行; 数据 = PARQUET 根稳定副本 lhb_seat_detail_labeled.parquet (只读).

用法:
    python scripts/_firstboard_pages.py --train   # 训练+落盘+TE复现校验(建包时跑一次)
    python scripts/_firstboard_pages.py --serve   # 手动出两页 DataFrame (调试)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PANEL_V3_PATH, PROJECT_ROOT, STOCK_LIST_DIR  # noqa: E402

log = logging.getLogger(__name__)

MODELS_DIR = Path(PROJECT_ROOT) / "models" / "firstboard"
TR_END = "2024-12-31"
VA_END = "2025-12-31"

PANEL_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pctChg",
    "turnover_rate",
    "volume_ratio",
    "amount",
    "circ_mv",
    "free_float_turnover_rate",
    "winner_ratio",
    "sw_l2_name",
    "lhb_net_buy",
    "lhb_inst_buy",
]

F18 = [
    "wr1",
    "gap",
    "ret60",
    "cmv",
    "ftr",
    "amount",
    "turn_d0",
    "vr_d0",
    "vr_prev",
    "pct_prev",
    "intraday",
    "amplitude",
    "seal_hard",
    "close_ret",
    "amt_rank_pct",
    "ind_boards",
    "streak_prev",
]
ECO4 = ["mkt_boards_n", "mkt_chain_n", "mkt_max_streak", "zha_rate"]
# 0922 W22: promo 用修正版(next_map 真昨日晋级率); 原漏版被否决出生产.
FEATS = F18 + ECO4 + ["promo_yd"]

PARAMS = dict(
    objective="binary",
    n_estimators=400,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=50,
    random_state=42,
    verbose=-1,
)

# TE 复现验收线 (W22 修正版实测; 训练后 reload 断言, 漂移即 fail, 禁静默换模型)
# k3 头验收线 = 0923 rank_calib 实测 (tmp_t/_0923_rank_calib.out), 容差按用户令 ±0.5 / ±0.005
# 0923 刷新 k2 两线: 46.1/42.0 建于 0922 V3 面板重建(BJ剔除)前 = 旧基线; 当前帧生产 booster
# 实读 k2@1=45.78 / k2@3=41.77 → 刷新为 45.8/41.8 (auc_next/k3 线原样不动)
_EXPECT_TE = {
    "k2@1": 45.8,
    "k2@3": 41.8,
    "auc_next": 0.5911,
    "k3@1": 39.8,
    "auc_k3": 0.5716,
    "tol_hits": 0.6,
    "tol_auc": 0.004,
    "tol_k3_hits": 0.5,
    "tol_k3_auc": 0.005,
}

# 点名页 v2: 全量清单截断上限 (冠军格★行截断豁免, W19 并集规则保留)
FB_MAX_ROWS = 80
# 次日一字风险旗: 一字 ∩ p_k3≥此值 → 次日大概率仍一字买不到 (0923: ≥0.4 档实测 T+3 板率 ~54%)
K3_RISK_TH = 0.40
# 校准档位文案 (0923 rank_calib TE n=5608 分桶实测率; 模型分为下界读数, 顶部实测率更高)
_CALIB_TIERS = [
    (0.40, "≥0.4→实测~54%"),
    (0.30, "0.3-0.4→实测~28%"),
    (0.20, "0.2-0.3→实测~21%"),
    (0.00, "<0.2→实测~18%"),
]

# 板前哨 次数窗口: 近 N 个交易日 sig=True 天数 (v4 起只作上下文列, 不再作显示门槛)
COUNT_WIN = 10

# 0923 LHB 席位级标注列 (W14 A/B 终判: pre20 聚合增量=平, D0 结构=负 → 按规则降级为纯标注列;
# 绝不进 FEATS/训练/闸)。tmp_t/_0923_w13_top_inst_labeled.parquet 的稳定只读副本:
LHB_SEAT_PARQUET = Path("D:/AMINQT/PARQUET/lhb_seat_detail_labeled.parquet")
LHB_PRE_WIN = (
    21  # pre20 窗 = D-20..D0 (rolling 21 交易日含 D0; LHB 收盘后发布, T+1 视角无前视)
)
LHB_ANNOT_COLS = [
    "LHB上榜日数(20日)",
    "LHB净买亿(20日)",
    "游资席位数(20日)",
    "机构在场(20日)",
    "D0游资席位",
    "D0净买亿",
]


def load_mainboard(
    panel_path=PANEL_V3_PATH, tail_dates: int | None = 450
) -> pd.DataFrame:
    """读主板日线 (列子集). tail_dates=None → 全史 (训练用); 否则只留最后 N 个交易日 (服务用)."""
    df = pq.read_table(str(panel_path), columns=PANEL_COLS).to_pandas()
    df["symbol"] = df["symbol"].astype(str)
    df = df[df["symbol"].str[:2].isin(["00", "60"])]
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.drop_duplicates(["symbol", "date"])
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    if tail_dates is not None:
        keep = sorted(df["date"].unique())[-tail_dates:]
        df = df[df["date"].isin(keep)].reset_index(drop=True)
    return df


def _ecology(df: pd.DataFrame) -> pd.DataFrame:
    """B5 市场情绪生态 (全部 ≤D 信息; promo=昨日晋级率, next_map 修正版)."""
    board = df["pctChg"] >= 9.5
    df = df.assign(_board=board)
    brk = (board != board.shift(1)) | (df["symbol"] != df["symbol"].shift(1))
    bstream_ = board.groupby(brk.cumsum()).cumsum()
    df["_bstreak"] = bstream_
    df.loc[~board, "_bstreak"] = 0
    limit = (df["pre_close"] * 1.10).round(2)
    zha = (df["high"] >= limit - 0.005) & ~board

    per_d = df.groupby("date")
    eco = pd.DataFrame(
        {"mkt_boards_n": per_d["pctChg"].apply(lambda s: int((s >= 9.5).sum()))}
    )
    eco["mkt_chain_n"] = (
        df[board & (df["_bstreak"] >= 2)]
        .groupby("date")
        .size()
        .reindex(eco.index)
        .fillna(0)
    )
    eco["mkt_max_streak"] = (
        df[board].groupby("date")["_bstreak"].max().reindex(eco.index).fillna(0)
    )
    zha_n = df[zha].groupby("date").size().reindex(eco.index).fillna(0)
    eco["zha_rate"] = zha_n / (zha_n + eco["mkt_boards_n"]).clip(lower=1)

    bd = df[board][["date", "symbol"]].assign(b=1)
    dates = sorted(df["date"].unique())
    next_map = {dates[i - 1]: dates[i] for i in range(1, len(dates))}
    bd_next = bd.copy()
    bd_next["date"] = bd_next["date"].map(next_map)
    bd_next = bd_next.rename(columns={"b": "boarded_yd"})
    promo = bd.merge(bd_next, on=["date", "symbol"], how="left")
    nprev = bd_next.groupby("date").size()
    both = promo.groupby("date")["boarded_yd"].sum()
    eco["promo_yd"] = ((both / nprev).reindex(eco.index)).fillna(0)
    return eco.drop(columns=["_bstreak"], errors="ignore").reset_index()[
        [
            "date",
            "mkt_boards_n",
            "mkt_chain_n",
            "mkt_max_streak",
            "zha_rate",
            "promo_yd",
        ]
    ]


def build_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """全部首板事件的特征+标签 (训练/服务共用同一实现, 防两源分裂).
    标签 k2/next_board 在历史尾部不足窗口时为 NaN (服务日自然如此)."""
    board = df["pctChg"] >= 9.5
    g = df.groupby("symbol", sort=False)
    d = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "date": df["date"],
            "board": board,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "pre_close": df["pre_close"],
            "pctChg": df["pctChg"],
            "amount": df["amount"],
            "circ_mv": df["circ_mv"],
            "ftr": df["free_float_turnover_rate"],
            "turn_d0": df["turnover_rate"],
            "vr_d0": df["volume_ratio"],
            "sw_l2_name": df["sw_l2_name"],
        }
    )
    d["wr1"] = g["winner_ratio"].shift(1)
    d["ret60"] = g["close"].shift(1) / g["close"].shift(61) - 1.0
    d["vr_prev"] = g["volume_ratio"].shift(1)
    d["pct_prev"] = g["pctChg"].shift(1)
    up = (df["pctChg"] > 0).fillna(False).astype(int)
    brk = (up != up.shift(1)) | (df["symbol"] != df["symbol"].shift(1))
    up_streak = up.groupby(brk.cumsum()).cumsum()
    d["streak_prev"] = up_streak.groupby(df["symbol"], sort=False).shift(1)

    d["gap"] = d["open"] / d["pre_close"] - 1.0
    limit = (d["pre_close"] * 1.10).round(2)
    d["_limit"] = limit
    seal_gap = (d["high"] - d["close"]) / d["pre_close"]
    d["seal_hard"] = ((seal_gap <= 0.002) & (d["pctChg"] >= 9.8)).astype(float)
    d.loc[d["gap"].isna() | d["pre_close"].isna() | d["high"].isna(), "seal_hard"] = (
        np.nan
    )
    d["cmv"] = d["circ_mv"]
    d["intraday"] = (d["close"] - d["open"]) / d["pre_close"]
    d["amplitude"] = (d["high"] - d["low"]) / d["pre_close"]
    d["close_ret"] = d["close"] / d["pre_close"] - 1.0

    # 事件: 首板 且 前10日无板 (rolling(10).max().shift(1), 与研究 chain1 同口径)
    d["_prior10"] = (
        d["board"]
        .astype(float)
        .groupby(df["symbol"], sort=False)
        .transform(lambda x: x.rolling(10, min_periods=1).max().shift(1))
        .fillna(0)
    )
    ev = d[d["board"] & (d["_prior10"] == 0)].copy()

    # 标签 (尾部不足 → NaN)
    rows_after = g.cumcount(ascending=False).reindex(ev.index)
    nb = g["pctChg"].shift(-1).reindex(ev.index)
    ev["next_board"] = np.where(nb.isna(), np.nan, (nb >= 9.5).astype(float))
    k2_any = pd.Series(False, index=ev.index)
    for i in range(1, 6):
        k2_any |= g["pctChg"].shift(-i).reindex(ev.index) >= 9.5
    ev["k2"] = np.where(rows_after >= 5, k2_any.astype(float), np.nan)
    k3_any = pd.Series(
        False, index=ev.index
    )  # 0923: T+3 头标签 (D0+1..D0+3 内任一板, 同 k2 式窗口3)
    for i in range(1, 4):
        k3_any |= g["pctChg"].shift(-i).reindex(ev.index) >= 9.5
    ev["k3"] = np.where(rows_after >= 3, k3_any.astype(float), np.nan)
    ev["mature10"] = rows_after >= 10  # 与研究 chain2 的 k3_ok 同口径 (10 前瞻日)
    ev["hist_ok"] = g.cumcount().reindex(ev.index) >= 61

    ev["yizi"] = ev["low"] >= ev["_limit"] - 0.005
    ev["champion"] = (ev["wr1"] >= 0.65) & ev["yizi"]
    ev["sw_l2"] = ev["sw_l2_name"].fillna("NA")

    eco = _ecology(df)
    ev = ev.merge(eco, on="date", how="left")
    ind = (
        df[board]
        .assign(sw_l2_name=df["sw_l2_name"].fillna("NA"))
        .groupby(["date", "sw_l2_name"])
        .size()
        .rename("ind_boards")
        .reset_index()
    )
    ev = ev.merge(
        ind,
        left_on=["date", "sw_l2"],
        right_on=["date", "sw_l2_name"],
        how="left",
        suffixes=("", "_ind"),
    )
    ev["ind_boards"] = ev["ind_boards"].fillna(0)
    ev["amt_rank_pct"] = ev.groupby("date")["amount"].rank(pct=True)
    return ev.reset_index(drop=True)


def train_and_save(
    out_dir=MODELS_DIR, panel_path=PANEL_V3_PATH, expect=_EXPECT_TE
) -> dict:
    """训练双头 (TR≤2024 / VA2025 早停), 落盘 booster + meta, reload 后断言 TE 复现.
    expect=None 跳过复现断言 (合成数据测试用)."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    df = load_mainboard(panel_path, tail_dates=None)
    ev = build_event_features(df)
    need6 = ["wr1", "ret60", "gap", "seal_hard", "cmv", "ftr"]
    A = ev[
        ev["hist_ok"]
        & ev["mature10"]
        & ev["k2"].notna()
        & ev["next_board"].notna()
        & ev[need6].notna().all(axis=1)
    ].copy()
    A["tr"] = A["date"] <= pd.Timestamp(TR_END)
    A["va"] = (A["date"] > pd.Timestamp(TR_END)) & (A["date"] <= pd.Timestamp(VA_END))
    A["te"] = A["date"] > pd.Timestamp(VA_END)
    cats = sorted(A["sw_l2"].unique().tolist())
    A["sw_l2"] = pd.Categorical(A["sw_l2"], categories=cats)
    X = A[FEATS + ["sw_l2"]].copy()
    for c in FEATS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 0923 修复A: lightgbm 4.7.0 sklearn shim 的 eval_set 路径早停失真 (best_iter~28, TE AUC 崩~0.51)
    # → 改原生 lgb.train 绕 shim (显式 categorical_feature), 与旧版生产 booster 逐项 Δ0.0000
    # (tmp_t/_0923_ths_ab_train.py 已验证). 参数等价映射: n_estimators→num_boost_round /
    # random_state→seed / verbose→verbosity / eval_metric="logloss"→metric="binary_logloss".
    native_params = dict(
        objective=PARAMS["objective"],
        num_leaves=PARAMS["num_leaves"],
        learning_rate=PARAMS["learning_rate"],
        min_child_samples=PARAMS["min_child_samples"],
        seed=PARAMS["random_state"],
        metric="binary_logloss",
        verbosity=PARAMS["verbose"],
    )
    models = {}
    for tgt, fname in (
        ("k2", "booster_k2.txt"),
        ("k3", "booster_k3.txt"),
        ("next_board", "booster_next.txt"),
    ):
        y = A[tgt]
        fit = (A["tr"] | A["va"]) & y.notna()
        m_tr = (fit & A["tr"]).to_numpy()
        m_va = (fit & A["va"]).to_numpy()
        dtr = lgb.Dataset(
            X[m_tr], y[m_tr], categorical_feature=["sw_l2"], free_raw_data=False
        )
        dva = lgb.Dataset(
            X[m_va],
            y[m_va],
            reference=dtr,
            categorical_feature=["sw_l2"],
            free_raw_data=False,
        )
        mdl = lgb.train(
            native_params,
            dtr,
            num_boost_round=PARAMS["n_estimators"],
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        mdl.save_model(str(out_dir / fname))
        models[tgt] = lgb.Booster(model_file=str(out_dir / fname))

    # reload 复现断言 (防两源分裂: 训练口径≠落盘口径即 fail)
    ev_te = A[A["te"]].copy()
    Xte = ev_te[FEATS + ["sw_l2"]].copy()
    Xte["sw_l2"] = pd.Categorical(Xte["sw_l2"], categories=cats)
    for c in FEATS:
        Xte[c] = pd.to_numeric(Xte[c], errors="coerce")
    p_k2 = models["k2"].predict(Xte)
    p_k3 = models["k3"].predict(Xte)
    p_nb = models["next_board"].predict(Xte)
    ev_te = ev_te.assign(p_k2=p_k2, p_k3=p_k3, p_nb=p_nb)
    hit1 = hit3 = tot1 = k3_hit1 = 0
    for _, grp in ev_te.groupby("date"):
        top = grp.sort_values("p_k2", ascending=False).head(min(3, len(grp)))
        hit3 += top["k2"].sum()
        tot1 += len(top)
        hit1 += grp.sort_values("p_k2", ascending=False).head(1)["k2"].sum()
        k3_hit1 += grp.sort_values("p_k3", ascending=False).head(1)["k3"].sum()
    k2_1 = 100 * hit1 / max(ev_te["date"].nunique(), 1)
    k2_3 = 100 * hit3 / tot1
    k3_1 = 100 * k3_hit1 / max(ev_te["date"].nunique(), 1)
    auc_next = roc_auc_score(ev_te["next_board"], p_nb)
    auc_k3 = (
        roc_auc_score(ev_te["k3"], p_k3) if ev_te["k3"].nunique() > 1 else float("nan")
    )
    meta = {
        "features": FEATS,
        "sw_l2_categories": cats,
        "train_end": VA_END,
        "params": {k: v for k, v in PARAMS.items() if k != "verbose"},
        "te_top1_k3": round(k3_1, 1),
        "te_auc_k3": round(auc_k3, 4),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"[firstboard] TE 复现: k2@1={k2_1:.1f} k2@3={k2_3:.1f} auc_next={auc_next:.4f} "
        f"k3@1={k3_1:.1f} auc_k3={auc_k3:.4f} n={len(ev_te)}"
    )
    if expect is not None:
        drift = (
            abs(k2_1 - expect["k2@1"]) > expect["tol_hits"]
            or abs(k2_3 - expect["k2@3"]) > expect["tol_hits"]
            or abs(auc_next - expect["auc_next"]) > expect["tol_auc"]
            or abs(k3_1 - expect["k3@1"]) > expect["tol_k3_hits"]
            or (
                not np.isnan(auc_k3)
                and abs(auc_k3 - expect["auc_k3"]) > expect["tol_k3_auc"]
            )
        )
        if drift:
            raise RuntimeError(
                f"TE 复现漂移: k2@1={k2_1:.1f}(期望{expect['k2@1']}) "
                f"k2@3={k2_3:.1f}(期望{expect['k2@3']}) auc_next={auc_next:.4f}"
                f"(期望{expect['auc_next']}) k3@1={k3_1:.1f}(期望{expect['k3@1']}) "
                f"auc_k3={auc_k3:.4f}(期望{expect['auc_k3']}) — 训练口径与验收线分裂, 禁上线"
            )
    return {
        "n_train": int((A["tr"] | A["va"]).sum()),
        "n_te": int(A["te"].sum()),
        "k2@1": k2_1,
        "k2@3": k2_3,
        "auc_next": auc_next,
        "k3@1": k3_1,
        "auc_k3": auc_k3,
    }


def load_models(models_dir=MODELS_DIR):
    import lightgbm as lgb

    models_dir = Path(models_dir)
    meta = json.loads((models_dir / "meta.json").read_text(encoding="utf-8"))
    return (
        lgb.Booster(model_file=str(models_dir / "booster_k2.txt")),
        lgb.Booster(model_file=str(models_dir / "booster_k3.txt")),
        lgb.Booster(model_file=str(models_dir / "booster_next.txt")),
        meta,
    )


def _predict(ev: pd.DataFrame, boosters, meta) -> pd.DataFrame:
    feats = meta["features"]
    X = ev[feats].apply(pd.to_numeric, errors="coerce")
    X["sw_l2"] = pd.Categorical(ev["sw_l2"], categories=meta["sw_l2_categories"])
    X = X[feats + ["sw_l2"]]
    ev = ev.copy()
    ok = (
        X.notna().all(axis=1).to_numpy()
    )  # 含未见过的 sw_l2 (Categorical 落 NaN → 跳过)
    ev["p_k2"] = np.nan
    ev["p_k3"] = np.nan
    ev["p_next"] = np.nan
    if ok.any():
        ev.loc[ok, "p_k2"] = boosters[0].predict(X[ok])
        ev.loc[ok, "p_k3"] = boosters[1].predict(X[ok])
        ev.loc[ok, "p_next"] = boosters[2].predict(X[ok])
    return ev


def _calib_tier(p_k3: pd.Series) -> pd.Series:
    """p_k3 → 校准档位文案 (_CALIB_TIERS 分桶; NaN → 空)."""
    tier = np.select(
        [p_k3 >= lo for lo, _ in _CALIB_TIERS[:-1]],
        [lab for _, lab in _CALIB_TIERS[:-1]],
        default=_CALIB_TIERS[-1][1],
    )
    return pd.Series(np.where(p_k3.notna(), tier, ""), index=p_k3.index)


_NAME_CACHE = None  # symbol→名称 模块级缓存 (tushare stock_basic 只读)


def _symbol_names() -> dict:
    """名称纯展示列: 拉不到就给空表, 勿阻页 (页建失败只丢两页的契约不变).
    铁规: 只读 ts.get_token(), 禁 ts.set_token() (会清空本地 tk.csv)."""
    global _NAME_CACHE
    if _NAME_CACHE is None:
        try:
            import tushare as ts

            pro = ts.pro_api(ts.get_token())
            sb = pro.stock_basic(fields="symbol,name")
            _NAME_CACHE = dict(zip(sb["symbol"].astype(str), sb["name"].astype(str)))
        except Exception:
            _NAME_CACHE = {}
    return _NAME_CACHE


_LHB_DAILY_CACHE = None  # (symbol,date) 日聚合 模块级缓存; 空 DataFrame = 无席位数据


def _agg_seat_daily(st: pd.DataFrame) -> pd.DataFrame:
    """LHB 席位明细原始行 → (symbol, date) 日聚合 (与 W13/W14 研究 tmp_t/_0923_w14_seat_ab.py 同口径).

    net_sum 已除 1e8 (元→亿); hotseat_n = 高频游资行数(席×榜面); inst_any = 机构专用在场;
    主板过滤 00/60 与 load_mainboard 一致."""
    s = pd.DataFrame(
        {
            "symbol": st["ts_code"].astype(str).str[:6],
            "date": pd.to_datetime(st["trade_date"].astype(str), format="%Y%m%d"),
            "net_buy": pd.to_numeric(st["net_buy"], errors="coerce").fillna(0.0),
            "hotseat": (st["category"].astype(str) == "高频游资").astype(float),
            "inst": (st["category"].astype(str) == "机构专用").astype(float),
        }
    )
    s = s[s["symbol"].str[:2].isin(["00", "60"])]
    daily = s.groupby(["symbol", "date"], as_index=False).agg(
        n_rows=("net_buy", "size"),
        net_sum=("net_buy", "sum"),
        hotseat_n=("hotseat", "sum"),
        inst_any=("inst", "max"),
    )
    daily["net_sum"] /= 1e8
    return daily


def _lhb_daily() -> pd.DataFrame:
    global _LHB_DAILY_CACHE
    if _LHB_DAILY_CACHE is None:
        try:
            _LHB_DAILY_CACHE = _agg_seat_daily(
                pq.read_table(str(LHB_SEAT_PARQUET)).to_pandas()
            )
        except Exception:
            _LHB_DAILY_CACHE = pd.DataFrame()  # 缺数据 → 空表, 标注列留空勿阻页
    return _LHB_DAILY_CACHE


def _lhb_annotations(df: pd.DataFrame, symbols, d0=None) -> dict[str, dict[str, float]]:
    """点名行 LHB 席位标注 (窗 = 面板日历 D-20..D0 共 LHB_PRE_WIN 日, 与 W14 rolling(21,min1) 等价).

    返回 symbol → {pre_days, pre_net, pre_hot, pre_inst, d0_hot, d0_net};
    无 LHB 数据的 symbol 不在 dict → 页面留空 (观察清单, 绝不删行/过滤)."""
    daily = _lhb_daily()
    if daily.empty or len(symbols) == 0:
        return {}
    cal = pd.DatetimeIndex(sorted(df["date"].unique()))
    d0 = cal[-1] if d0 is None else pd.Timestamp(d0)
    loc = cal.get_loc(d0)
    win = set(cal[max(0, loc - LHB_PRE_WIN + 1) : loc + 1])
    w = daily[
        daily["date"].isin(win)
        & daily["symbol"].isin(set(pd.Index(symbols).astype(str)))
    ]
    out: dict[str, dict[str, float]] = {}
    for sym, grp in w.groupby("symbol"):
        d0row = grp[grp["date"] == d0]
        out[sym] = {
            "pre_days": float(len(grp)),
            "pre_net": float(grp["net_sum"].sum()),
            "pre_hot": float(grp["hotseat_n"].sum()),
            "pre_inst": float(grp["inst_any"].max()),
            "d0_hot": float(d0row["hotseat_n"].sum()),
            "d0_net": float(d0row["net_sum"].sum()),
        }
    return out


def serve_firstboard(
    df: pd.DataFrame | None = None, models_dir=MODELS_DIR, max_rows: int = FB_MAX_ROWS
) -> pd.DataFrame:
    """当日首板点名页 v2 (0923 用户令): 全量当日事件按 T+3板概率(p_k3)降序 (中文列).

    不再 Top-N 截断 — 超 max_rows 截 max_rows 行, 冠军格★行截断豁免 (W19 并集规则保留);
    一字/次日一字风险 = 『买不到』提示列, 非否决; 校准档位 = p_k3 分桶 TE 实测板率;
    0923 尾部追加 LHB 席位级纯标注列 (W14 无模型增量 → 只标注; 无数据留空, 绝不删行)."""
    if df is None:
        df = load_mainboard()
    d0 = df["date"].max()
    ev = build_event_features(df)
    today = ev[ev["date"] == d0].copy()
    try:
        b_k2, b_k3, b_next, meta = load_models(models_dir)
        today = _predict(today, (b_k2, b_k3, b_next), meta)
    except FileNotFoundError:
        log.error(
            "首板模型缺失 (%s): p_k3 全 NaN → 排名/校准档位 将为空, 本页无效 (页面看似正常)",
            models_dir,
        )
        today = today.assign(p_k2=np.nan, p_k3=np.nan, p_next=np.nan)
    today = today.sort_values("p_k3", ascending=False, na_position="last").reset_index(
        drop=True
    )
    if len(today) > max_rows:  # 截断但冠军格★行保证入选 (W19 并集规则)
        keep = list(range(max_rows)) + [
            i for i in today.index[today["champion"]] if i >= max_rows
        ]
        today = today.loc[keep].reset_index(drop=True)
    names = _symbol_names() if len(today) else {}
    rank = today["p_k3"].notna().cumsum().where(today["p_k3"].notna(), np.nan)
    page = pd.DataFrame(
        {
            "排名": rank,
            "代码": today["symbol"],
            "名称": today["symbol"].map(names).fillna(""),
            "T+3板概率": today["p_k3"],
            "T+5板概率": today["p_k2"],
            "校准档位": _calib_tier(today["p_k3"]),
            "冠军格": np.where(today["champion"], "★", ""),
            "一字": np.where(today["yizi"], "一字", ""),
            "次日一字风险": np.where(
                today["yizi"] & (today["p_k3"] >= K3_RISK_TH), "风险", ""
            ),
            "板前获利盘": today["wr1"],
            "板块涨停数": today["ind_boards"],
            "昨日晋级率": today["promo_yd"],
            "市场涨停数": today["mkt_boards_n"],
        }
    )
    # 0923 LHB 席位级标注列 (纯标注, 页尾追加; 构建失败只丢标注不丢页 — 同名称列契约)
    try:
        ann = _lhb_annotations(df, today["symbol"], d0)
    except Exception:
        ann = {}
    adf = pd.DataFrame.from_dict(ann, orient="index")

    def _ann(key: str) -> np.ndarray:
        if key in adf.columns:
            return today["symbol"].map(adf[key]).to_numpy(dtype=float)
        return np.full(len(today), np.nan)

    inst = _ann("pre_inst")
    page["LHB上榜日数(20日)"] = _ann("pre_days")
    page["LHB净买亿(20日)"] = _ann("pre_net")
    page["游资席位数(20日)"] = _ann("pre_hot")
    page["机构在场(20日)"] = np.where(np.isnan(inst) | (inst <= 0), "", "在")
    page["D0游资席位"] = _ann("d0_hot")
    page["D0净买亿"] = _ann("d0_net")
    return page


def build_sig(df: pd.DataFrame) -> pd.DataFrame:
    """深睡签名 + 点火旗 (向量化, 与 watchW1/preW7 同口径).
    返回带 sig/fire_t1/fire_t2/wr20/t5t20 列的 df (索引对齐输入)."""
    sym = df["symbol"]
    board = df["pctChg"] >= 9.5
    bigup = board.astype("float64")

    def gshift(s, k):
        return s.groupby(sym, observed=True).shift(k)

    def groll(s, win, minp, how="mean"):
        return s.groupby(sym, observed=True).transform(
            lambda x: getattr(x.rolling(win, min_periods=minp), how)()
        )

    cum = bigup.groupby(sym, observed=True).cumsum()
    close = df["close"]
    pf = (df["pctChg"] >= 4.0) & (df["volume_ratio"] >= 1.8)
    P = pd.Series(False, index=df.index)
    for k in range(10, 41):
        nocross = ((cum - gshift(cum, k)) == 0).fillna(False)
        hold = (close >= 0.9 * gshift(close, k)).fillna(False)
        P |= (gshift(pf.astype("float64"), k) > 0.5).fillna(False) & nocross & hold
    wr20 = df["winner_ratio"] - gshift(df["winner_ratio"], 20)
    ret20 = close / gshift(close, 20) - 1.0
    C = ((wr20 >= 0.10) & (ret20.abs() <= 0.06)).fillna(False)
    t5 = groll(df["turnover_rate"], 5, 5, "mean")
    t20 = groll(df["turnover_rate"], 20, 20, "mean")
    S = ((t5 <= 1.2) & (t5 <= 0.75 * t20)).fillna(False)
    nb10 = (
        groll(bigup, 10, 10, "sum")
        .shift(0)
        .groupby(sym, observed=True)
        .shift(1)
        .fillna(0)
        == 0
    )
    sig = (P & C & S & nb10 & ~board).fillna(False)

    ge2 = df["pctChg"] >= 2
    prev5_ge2 = (
        gshift(ge2.astype("float64"), 1)
        .groupby(sym, observed=True)
        .transform(lambda x: x.rolling(5, min_periods=1).max())
        .fillna(0)
        > 0
    )
    fire_t1 = ge2 & ~prev5_ge2
    fire_t2 = df["pctChg"].between(2, 7)

    net = pd.to_numeric(df["lhb_net_buy"], errors="coerce")
    inst = pd.to_numeric(df["lhb_inst_buy"], errors="coerce")
    bothf = ((net > 0) & (inst > 0)).astype(float)
    lhb20both = (
        bothf.groupby(sym, observed=True)
        .transform(lambda x: x.shift(1).rolling(20, min_periods=1).max())
        .fillna(0)
        > 0
    )

    out = df[
        [
            "symbol",
            "date",
            "pctChg",
            "close",
            "winner_ratio",
            "volume_ratio",
            "turnover_rate",
        ]
    ].copy()
    out["sig"] = sig
    out["fire_t1"] = fire_t1
    out["fire_t2"] = fire_t2
    out["wr20"] = wr20
    out["t5"] = t5
    out["t5t20"] = t5 / t20
    out["lhb20_both"] = lhb20both
    return out


def serve_preboard(
    df: pd.DataFrame | None = None, record_csv: Path | None = None
) -> pd.DataFrame:
    """滚动20日板前哨监视名单 — 历史命中日志 (近20日每个 sig 日一行, 每股可多行; 中文列).

    次数10日/次数20日 = 近 COUNT_WIN/20 日 sig=True 天数 (上下文列, 不参与筛选);
    v4 (0923 用户令): Excel 只留 今日点火旗≠'' 股的全部命中行 (次数完全退出筛选, 其余整组只在后台表);
    点火旗/通道优先/当日涨幅 = 今日 d0 口径 (停牌缺今日行 → ''/False/NaN);
    获利盘/wr20/t5/t5t20 = 命中日当天画像; record_csv = 每股一行的汇总后台表 (WORM)."""
    if df is None:
        df = load_mainboard()
    d0 = df["date"].max()
    s = build_sig(df)
    dates = sorted(df["date"].unique())
    last20, cw = set(dates[-20:]), set(dates[-COUNT_WIN:])

    w = s[s["date"].isin(last20)]
    hw = w[w["sig"]]
    today = s[s["date"] == d0].set_index("symbol")

    # 每股一行汇总 (后台表数据层)
    if hw.empty:
        summary = pd.DataFrame(
            {
                "symbol": pd.Series(dtype=object),
                "首次击中": pd.Series(dtype="datetime64[ns]"),
                "最近击中": pd.Series(dtype="datetime64[ns]"),
                "次数10日": pd.Series(dtype=int),
                "次数20日": pd.Series(dtype=int),
            }
        )
    else:
        summary = pd.concat(
            [
                hw.groupby("symbol")["date"].agg(首次击中="min", 最近击中="max"),
                hw[hw["date"].isin(cw)].groupby("symbol").size().rename("次数10日"),
                hw.groupby("symbol").size().rename("次数20日"),
            ],
            axis=1,
        ).reset_index()
        summary["次数10日"] = summary["次数10日"].fillna(0).astype(int)
        summary["次数20日"] = summary["次数20日"].fillna(0).astype(int)
    summary = summary.join(
        today[["pctChg", "winner_ratio", "fire_t1", "fire_t2", "lhb20_both"]].rename(
            columns={"pctChg": "今日涨幅", "winner_ratio": "今日获利盘"}
        ),
        on="symbol",
    )
    f1 = summary["fire_t1"].fillna(False).astype(bool)
    f2 = summary["fire_t2"].fillna(False).astype(bool)
    summary["今日点火旗"] = np.where(
        f1 & f2, "T1+T2", np.where(f1, "T1", np.where(f2, "T2", ""))
    )
    summary["_pass"] = (summary["今日获利盘"] >= 0.65) & summary["lhb20_both"].fillna(
        False
    ).astype(bool)

    if record_csv is not None:
        rec = pd.DataFrame(
            {
                "代码": summary["symbol"],
                "首次击中": summary["首次击中"].dt.strftime("%Y-%m-%d"),
                "最近击中": summary["最近击中"].dt.strftime("%Y-%m-%d"),
                "次数10日": summary["次数10日"],
                "次数20日": summary["次数20日"],
                "今日点火旗": summary["今日点火旗"],
                "今日涨幅": summary["今日涨幅"] / 100.0,
                "今日获利盘": summary["今日获利盘"],
                "通道优先": np.where(summary["_pass"], "优先", ""),
            }
        )
        p = Path(record_csv)
        if p.exists():  # WORM: 同日重跑不覆盖, 文件名插时间戳
            p = p.with_name(f"{p.stem}__{datetime.now().strftime('%H%M%S')}{p.suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        rec.to_csv(p, index=False, encoding="utf-8-sig")

    # Excel 页 (v4 0923 用户令): 仅 今日点火旗≠'' 的股 × 每个 last20 命中日各一行; 次数退出筛选只作列
    sumi = summary.set_index("symbol")
    show = sumi.index[sumi["今日点火旗"] != ""]
    page = hw[hw["symbol"].isin(show)].copy()
    page["次数10日"] = page["symbol"].map(sumi["次数10日"])
    page["次数20日"] = page["symbol"].map(sumi["次数20日"])
    page = page.join(
        today[["pctChg", "winner_ratio", "fire_t1", "fire_t2", "lhb20_both"]].rename(
            columns={
                "pctChg": "_pct_t",
                "winner_ratio": "_wr_t",
                "fire_t1": "_f1_t",
                "fire_t2": "_f2_t",
                "lhb20_both": "_lhb_t",
            }
        ),
        on="symbol",
    )
    pf1 = page["_f1_t"].fillna(False).astype(bool)
    pf2 = page["_f2_t"].fillna(False).astype(bool)
    page["点火旗"] = np.where(
        pf1 & pf2, "T1+T2", np.where(pf1, "T1", np.where(pf2, "T2", ""))
    )
    page["_pass"] = (page["_wr_t"] >= 0.65) & page["_lhb_t"].fillna(False).astype(bool)
    page["_fr"] = page["点火旗"].map({"T1+T2": 3, "T1": 2, "T2": 1, "": 0}).fillna(0)
    page = page.sort_values(
        ["_pass", "_fr", "date", "次数10日"], ascending=False
    ).reset_index(drop=True)
    return pd.DataFrame(
        {
            "通道优先": np.where(page["_pass"], "优先", ""),
            "点火旗": page["点火旗"].to_numpy(),
            "击中日期": page["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "次数10日": page["次数10日"].to_numpy(),
            "次数20日": page["次数20日"].to_numpy(),
            "代码": page["symbol"].to_numpy(),
            "当日涨幅": page["_pct_t"].to_numpy() / 100.0,
            "获利盘": page["winner_ratio"].to_numpy(),
            "20日获利盘Δ": page["wr20"].to_numpy(),
            "5日均换手": page["t5"].to_numpy(),
            "5日/20日换手比": page["t5t20"].to_numpy(),
        }
    )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="训练+落盘+TE复现校验")
    args = ap.parse_args()
    if args.train:
        res = train_and_save()
        print(json.dumps(res, ensure_ascii=False))
        return 0
    df = load_mainboard()
    fb = serve_firstboard(df)
    d0 = df["date"].max()
    record_csv = Path(STOCK_LIST_DIR) / f"preboard_watch_hits_{d0:%Y%m%d}.csv"
    pb = serve_preboard(df, record_csv=record_csv)
    print(f"=== 首板点名页 ({len(fb)}) ===")
    print(fb.to_string(index=False))
    print(f"\n=== 板前哨页 ({len(pb)}, 仅今日点火股; 后台全量表含其余股) ===")
    print(pb.to_string(index=False))
    print(f"\n后台全量表: {record_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
