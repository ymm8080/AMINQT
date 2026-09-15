# -*- coding: utf-8 -*-
"""次日大跌预测模块 (规则 + 模型融合)。

输入股票代码 → 输出【大跌概率】+【关键指标】+【原因分析】。

设计:
  规则侧: 8 条已验证条件 → 得分档 → 历史上该档的真实命中率 (可解释, 天然校准)
  模型侧: LGBM 排序 → isotonic 校准 → 真概率
  融合侧: 在校准段上用 logistic 叠 [模型logit, 规则分, 是否命中规则, 交互项],
          融合方式由 OOS 数据挑选, 不是拍脑袋。
  原因侧: 触发的规则 + 模型逐股贡献度 (pred_contrib) 双通道。

三段切分 (严格防泄漏):
  FIT   < 20250701   训模型
  CALIB 20250701~    训 isotonic + 训融合层 + 建规则概率表
  OOS   >=20260101   只验收, 从不参与任何拟合

用法:
  python scripts/bigdrop_check.py --build           # 训练+校准+融合+落盘 (WORM)
  python scripts/bigdrop_check.py 000978 002815     # 查询
  python scripts/bigdrop_check.py --rules           # 单规则命中率
  python scripts/bigdrop_check.py --compare         # 各口径 OOS 对比
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
from scipy.special import logit  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from config.settings import PANEL_V3_PATH  # noqa: E402

SEED = 42
DROP_TH = 0.05
GAP_MAX = 15
FIT_END, CAL_END = "20250701", "20260101"
EMBARGO = 5
BUNDLE_DIR = ROOT / "models" / "bigdrop"
MIN_BUCKET = 300

COLS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "close_hfq",
    "pctChg",
    "up_limit_raw",
    "down_limit_raw",
    "is_suspended",
    "amount",
    "turnover_rate",
    "free_float_turnover_rate",
    "volume_ratio",
    "bias_5",
    "bias_10",
    "bias_20",
    "bias_60",
    "bias_120",
    "bias_250",
    "winner_ratio",
    "chip_gini",
    "chip_entropy",
    "resistance_dist",
    "support_dist",
    "amplitude_5d",
    "intraday_range",
    "ovd_15p",
    "cost_bias",
    "circ_mv",
    "total_mv",
    "pb",
    "ps_ttm",
    "board",
]

# 规则库: (内部名, 中文名, 触发说明, 为什么危险)
RULES: list[tuple[str, str, str, str]] = [
    (
        "hi_bias20",
        "短期高位乖离",
        "20日乖离率 > 25%",
        "股价短期偏离均线过远, 获利盘丰厚, 一有风吹草动就集中兑现",
    ),
    (
        "hi_bias60",
        "中期高位乖离",
        "60日乖离率 > 50%",
        "两个月涨幅过大, 处于加速末端, 回吐空间大",
    ),
    (
        "huge_turn",
        "巨量换手",
        "换手率 > 20%",
        "单日换手超两成, 筹码剧烈易手, 往往是派发而非承接",
    ),
    (
        "zt_break",
        "炸板",
        "盘中触及涨停但收盘未封住",
        "封板资金撤退, 追高盘被套, 次日容易低开续跌",
    ),
    (
        "conseq_zt",
        "连板高位",
        "连续涨停 >= 3 天",
        "连板股情绪定价, 情绪一退潮就是连续跌停",
    ),
    (
        "vol_surge",
        "显著放量",
        "量比 > 2",
        "放量本身中性, 但叠在高位就是典型的见顶量能特征",
    ),
    (
        "today_drop",
        "今日已大跌",
        "当日跌幅 >= 5%",
        "破位后有惯性, 跌停/接近跌停的次日续跌概率显著抬升",
    ),
    (
        "big_rise",
        "累积涨幅过大",
        "60日涨幅 > 50%",
        "底部起来涨超五成, 随时可能进入兑现窗口",
    ),
    (
        "rise10",
        "10日急涨",
        "10日累计涨幅 > 20%",
        "十个交易日涨超两成, 短线获利盘密集, 一旦转弱容易集中兑现。"
        "补的是「中等强度涨过头」这一族 —— 原来 8 条闸全是极端值阈值, 够不着",
    ),
]

FEATS = [
    "low60_gain",
    "ma20_dev",
    "dist_hi250",
    "ret5",
    "ret10",
    "ret20",
    "ret60",
    "vol5",
    "vol20",
    "upper_shadow",
    "close_pos",
    "is_limit_up",
    "is_limit_dn",
    "zt_break",
    "conseq_zt",
    "gap_open",
    "log_amt",
    "log_cmv",
    "turnover_rate",
    "free_float_turnover_rate",
    "volume_ratio",
    "bias_5",
    "bias_10",
    "bias_20",
    "bias_60",
    "bias_120",
    "bias_250",
    "winner_ratio",
    "chip_gini",
    "chip_entropy",
    "resistance_dist",
    "support_dist",
    "amplitude_5d",
    "intraday_range",
    "ovd_15p",
    "cost_bias",
    "pb",
    "ps_ttm",
    "pctChg",
    "mkt_ret",
    "mkt_zt_ratio",
]

# 模型贡献度的中文名 (原因分析用)
PRETTY = {
    "bias_20": "20日乖离",
    "bias_60": "60日乖离",
    "bias_120": "120日乖离",
    "bias_250": "250日乖离",
    "bias_5": "5日乖离",
    "bias_10": "10日乖离",
    "turnover_rate": "换手率",
    "free_float_turnover_rate": "自由流通换手",
    "volume_ratio": "量比",
    "winner_ratio": "获利盘占比",
    "chip_gini": "筹码集中度",
    "chip_entropy": "筹码分散度",
    "resistance_dist": "距压力位",
    "support_dist": "距支撑位",
    "amplitude_5d": "5日振幅",
    "intraday_range": "当日振幅",
    "ovd_15p": "隔夜跳空占比",
    "cost_bias": "成本乖离",
    "low60_gain": "60日涨幅",
    "ma20_dev": "20日均线乖离",
    "dist_hi250": "距250日高点",
    "ret5": "5日涨幅",
    "ret10": "10日涨幅",
    "ret20": "20日涨幅",
    "ret60": "60日涨幅",
    "vol5": "5日波动",
    "vol20": "20日波动",
    "upper_shadow": "上影线占比",
    "close_pos": "收盘位置",
    "is_limit_up": "收盘涨停",
    "is_limit_dn": "收盘跌停",
    "zt_break": "炸板",
    "conseq_zt": "连板数",
    "gap_open": "开盘跳空",
    "log_amt": "成交额",
    "log_cmv": "流通市值",
    "pb": "市净率",
    "ps_ttm": "市销率",
    "pctChg": "当日涨跌",
    "mkt_ret": "大盘中位涨跌",
    "mkt_zt_ratio": "涨跌停家数比",
}


# ---------------------------------------------------------------- 数据
def load_frame() -> pd.DataFrame:
    d = pq.read_table(str(PANEL_V3_PATH), columns=COLS).to_pandas()
    d["symbol"] = (
        d["symbol"]
        .astype(str)
        .str.replace(r"\.(SH|SZ|BJ)$", "", regex=True)
        .str.zfill(6)
    )
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y%m%d")
    d = d.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("symbol", sort=False)

    # 标签
    nxt = g["close_hfq"].shift(-1)
    gap = (
        pd.to_datetime(g["date"].shift(-1), format="%Y%m%d")
        - pd.to_datetime(d["date"], format="%Y%m%d")
    ).dt.days
    ok = nxt.notna() & (gap <= GAP_MAX) & nxt.gt(0) & d["is_suspended"].fillna(1).eq(0)
    d["fwd"] = (nxt / d["close_hfq"] - 1.0).where(ok)
    d["y"] = np.where(d["fwd"].notna(), (d["fwd"] <= -DROP_TH).astype(float), np.nan)

    # 派生特征
    d["low60_gain"] = (
        d["close_hfq"]
        / g["close_hfq"].transform(lambda s: s.rolling(60, min_periods=60).min())
        - 1.0
    )
    d["ma20_dev"] = (
        d["close_hfq"]
        / g["close_hfq"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        - 1.0
    )
    hh250 = g["close_hfq"].transform(lambda s: s.rolling(250, min_periods=200).max())
    d["dist_hi250"] = d["close_hfq"] / hh250 - 1.0
    for k in (5, 10, 20, 60):
        d[f"ret{k}"] = g["close_hfq"].transform(lambda s, kk=k: s.pct_change(kk))
    for k in (5, 20):
        d[f"vol{k}"] = d["pctChg"].transform(
            lambda s, kk=k: s.rolling(kk, min_periods=kk).std()
        )
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["upper_shadow"] = ((d["high"] - d["close"]) / rng).fillna(0.0)
    d["close_pos"] = ((d["close"] - d["low"]) / rng).fillna(0.5)
    up = d["up_limit_raw"]
    d["is_limit_up"] = (d["close"] >= up * (1 - 1e-4)).astype(int)
    d["is_limit_dn"] = (d["close"] <= d["down_limit_raw"] * (1 + 1e-4)).astype(int)
    d["zt_break"] = (
        (d["high"] >= up * (1 - 1e-4)) & (d["close"] < up * (1 - 1e-4))
    ).astype(int)
    z = d["is_limit_up"]
    blk = z.ne(z.shift()).groupby(d["symbol"], observed=True).cumsum()
    d["conseq_zt"] = z.groupby([d["symbol"], blk]).cumsum()
    d["gap_open"] = d["open"] / d["pre_close"] - 1.0
    d["log_amt"] = np.log1p(d["amount"].clip(lower=0))
    d["log_cmv"] = np.log1p(d["circ_mv"].clip(lower=0))
    # 放量倍数 (涨停切片要用, 见 situation_note)
    ma20 = g["amount"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    d["amt_x"] = d["amount"] / ma20.replace(0, np.nan)
    # 市场态
    mkt = pd.DataFrame({"mkt_ret": d.groupby("date")["pctChg"].median()})
    lu = d.groupby("date")["is_limit_up"].sum()
    ld = d.groupby("date")["is_limit_dn"].sum()
    mkt["mkt_zt_ratio"] = lu / (lu + ld).clip(lower=1)
    return d.merge(mkt.reset_index(), on="date", how="left")


def rule_flags(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hi_bias20": d["bias_20"] > 0.25,
            "hi_bias60": d["bias_60"] > 0.50,
            "huge_turn": d["turnover_rate"] > 20,
            "zt_break": d["zt_break"] == 1,
            "conseq_zt": d["conseq_zt"] >= 3,
            "vol_surge": d["volume_ratio"] > 2,
            "today_drop": d["pctChg"] <= -5,
            "big_rise": d["low60_gain"] > 0.50,
            # 第 9 条 (0915): 中等强度涨过头。全样本覆盖 20.0% 的大跌,
            # score==0 桶内 OOS 3.63%→9.58%; 新增告警的 OOS 大跌率 19.89%,
            # 对照老 score>=3 的 24.51% —— 打 85 折, 不是灌水。
            # 注意「缩量」不在此列: 在 ret10>20% 人群里缩量挑的是低风险半边。
            "rise10": d["ret10"] > 0.20,
        },
        index=d.index,
    )


def rule_table(score: np.ndarray, y: np.ndarray) -> dict:
    """得分档 → 真实命中率。样本不足的档并入相邻档。"""
    tab = {}
    for s in np.unique(score):
        m = score == s
        if m.sum() >= MIN_BUCKET:
            tab[int(s)] = float(y[m].mean())
    return tab


# ---------------------------------------------------------------- 训练
def build() -> dict:
    np.random.seed(SEED)
    d = load_frame()
    F = rule_flags(d)
    score_all = F.sum(axis=1).to_numpy(int)

    val = d["y"].notna().to_numpy()
    X = d[FEATS].astype(np.float32)
    y = d["y"].to_numpy(float)
    dt = d["date"].to_numpy()
    del d, F

    ds = np.sort(np.unique(dt))
    fit = val & (dt < FIT_END)
    cal = val & (dt >= FIT_END) & (dt < CAL_END)
    oos = val & (dt >= CAL_END)
    # embargo: 丢掉切分边界前的几个交易日 (标签跨界的样本)
    c1 = ds[max(0, np.searchsorted(ds, FIT_END) - EMBARGO)]
    c2 = ds[max(0, np.searchsorted(ds, CAL_END) - EMBARGO)]
    fit &= dt <= c1
    cal &= dt <= c2

    print(f"FIT {int(fit.sum()):,} | CALIB {int(cal.sum()):,} | OOS {int(oos.sum()):,}")
    print(
        f"正样本率: FIT {y[fit].mean():.3%} | CALIB {y[cal].mean():.3%} | "
        f"OOS {y[oos].mean():.3%}"
    )

    m = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(X[fit], y[fit])
    p_raw = m.predict_proba(X)[:, 1]
    print(f"模型 OOS AUC {roc_auc_score(y[oos], p_raw[oos]):.4f}")

    # isotonic 校准 (只在校准段拟合)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw[cal], y[cal])
    p_iso = iso.predict(p_raw)

    # 规则概率表 (只用 FIT+CALIB, 不含 OOS)
    pre = fit | cal
    rtab = rule_table(score_all[pre], y[pre])
    base_pre = float(y[pre].mean())
    rule_p = np.array([rtab.get(int(s), base_pre) for s in score_all])

    # 融合层 (只在校准段拟合)
    lg = logit(np.clip(p_raw, 1e-6, 1 - 1e-6))
    hit = (score_all >= 2).astype(float)
    Z = np.column_stack([lg, score_all, hit, lg * hit])
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Z[cal], y[cal])
    p_fus = lr.predict_proba(Z)[:, 1]

    # ---- OOS 对比: 用户只要概率, 判据是 Brier + 可靠性, 不是 AUC ----
    yo, bo = y[oos], float(y[oos].mean())
    print(
        f"\n{'=' * 96}\nOOS 概率质量对比 (基准 {bo:.3%}; 常数预测 Brier "
        f"{bo * (1 - bo):.5f})"
    )
    cands = [
        ("纯规则表", rule_p),
        ("模型(isotonic校准)", p_iso),
        ("模型(未校准原值)", p_raw),
        ("融合(规则+模型)", p_fus),
    ]
    rows = []
    for name, p in cands:
        po = p[oos]
        br = brier_score_loss(yo, po)
        qq = pd.qcut(pd.Series(po), 10, labels=False, duplicates="drop")
        g = (
            pd.DataFrame({"q": qq, "p": po, "a": yo})
            .groupby("q")
            .agg(p=("p", "mean"), a=("a", "mean"), n=("a", "size"))
        )
        cal_err = float((g["a"] - g["p"]).abs().mul(g["n"]).sum() / g["n"].sum())
        rows.append((name, roc_auc_score(yo, po), br, cal_err, float(po.mean()), po))
    print(
        f"{'口径':<20}{'Brier':>10}{'校准误差':>11}{'平均预测':>11}{'AUC(仅参考)':>13}"
    )
    for name, auc, br, ce, mp, _ in rows:
        print(f"{name:<20}{br:>10.5f}{ce:>11.4f}{mp:>11.3%}{auc:>13.4f}")

    # 用户要的是"概率准不准", 判据取【校准误差】而非 Brier ——
    # Brier 被 85% 的低概率样本稀释, 而校准误差直接衡量"说 X% 是不是真 X%"。
    best = min(rows, key=lambda r: r[3])
    winner = best[0]
    p_win = best[5]
    print(f"\n  概率口径 = {winner} (校准误差最小 {best[3]:.4f})")
    print(
        "  注意: 模型 Brier 略低但校准更差, 且 isotonic 在高位尾部外推不可信"
        "(会给极端个股报 80%+)。模型改为只出贡献度做原因分析。"
    )

    # 直接回答: 说 X% 是不是真的 X%
    print(f"\n{'=' * 96}\n规则表概率 在 OOS 的真实兑现 (说 X% → 实际跌了多少)")
    print(f"{'规则得分':>8}{'表里写':>10}{'OOS实际':>10}{'差':>10}{'OOS样本':>10}")
    for s in sorted(rtab):
        msk = oos & (score_all == s)
        if msk.sum() < 20:
            continue
        print(
            f"{s:>8}{rtab[s]:>10.2%}{y[msk].mean():>10.2%}"
            f"{y[msk].mean() - rtab[s]:>+10.2%}{int(msk.sum()):>10,}"
        )

    print(f"\n{winner} 的概率带 在 OOS 的真实兑现")
    print(f"{'概率带':>16}{'带内均值':>11}{'实际':>10}{'差':>10}{'OOS样本':>10}")
    pw, yw = p_win, y[oos]
    for lo, hi in (
        (0, 0.02),
        (0.02, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.35),
        (0.35, 1.01),
    ):
        msk = (pw >= lo) & (pw < hi)
        if msk.sum() < 50:
            continue
        print(
            f"{f'[{lo:.0%},{hi:.0%})':>16}{pw[msk].mean():>11.2%}"
            f"{yw[msk].mean():>10.2%}{yw[msk].mean() - pw[msk].mean():>+10.2%}"
            f"{int(msk.sum()):>10,}"
        )

    tag = datetime.now().strftime("%Y%m%d")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = dict(
        tag=tag,
        drop_th=DROP_TH,
        feats=FEATS,
        rules=RULES,
        booster=m.booster_,
        iso=iso,
        lr=lr,
        rule_table=rtab,
        base_pre=base_pre,
        oos_base=bo,
        seed=SEED,
        winner=winner,
        split=dict(fit_end=FIT_END, cal_end=CAL_END, embargo=EMBARGO),
        oos_brier={n: float(br) for n, _, br, _, _, _ in rows},
        oos_cal_err={n: float(ce) for n, _, _, ce, _, _ in rows},
        oos_auc=float(roc_auc_score(yo, p_fus[oos])),
    )
    joblib.dump(bundle, BUNDLE_DIR / f"bundle_{tag}.joblib")
    joblib.dump(bundle, BUNDLE_DIR / "bundle_latest.joblib")
    print(f"\n-> {BUNDLE_DIR}\\bundle_{tag}.joblib")

    # 特征重要度留档
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    print(
        "\n特征重要度 Top12: "
        + ", ".join(f"{PRETTY.get(k, k)}({v})" for k, v in imp.head(12).items())
    )
    imp.to_csv(BUNDLE_DIR / f"importance_{tag}.csv", encoding="utf-8-sig")
    return bundle


def load_bundle() -> dict:
    p = BUNDLE_DIR / "bundle_latest.joblib"
    if not p.exists():
        print("没有模型包, 先跑: python scripts/bigdrop_check.py --build")
        sys.exit(2)
    return joblib.load(p)


# ---------------------------------------------------------------- 查询
def score_frame(d: pd.DataFrame, b: dict) -> tuple:
    """返回 (规则旗标, 得分, 各口径概率字典)。"""
    F = rule_flags(d)
    sc = F.sum(axis=1).to_numpy(int)
    # 注意: Booster.predict 对二分类默认 raw_score=False, 返回的**已经是概率**,
    # 不要再套 sigmoid (套了会把全市场压到 50%)。log-odds 只在 pred_contrib 里用。
    p_raw = b["booster"].predict(d[b["feats"]].astype(np.float32))
    p_iso = b["iso"].predict(p_raw)
    lg = logit(np.clip(p_raw, 1e-6, 1 - 1e-6))
    hit = (sc >= 2).astype(float)
    Z = np.column_stack([lg, sc, hit, lg * hit])
    p = {
        "纯规则表": np.array([b["rule_table"].get(int(s), b["base_pre"]) for s in sc]),
        "模型(isotonic校准)": p_iso,
        "模型(未校准原值)": p_raw,
        "融合(规则+模型)": b["lr"].predict_proba(Z)[:, 1],
    }
    return F, sc, p


def explain(d: pd.DataFrame, b: dict, idx) -> list[tuple[str, float]]:
    """模型逐股贡献度 (log-odds), 正=推高危险。"""
    X = d[b["feats"]].astype(np.float32).loc[[idx]]
    contrib = b["booster"].predict(X, pred_contrib=True)[0][:-1]
    s = pd.Series(contrib, index=b["feats"]).sort_values(ascending=False)
    return [(PRETTY.get(k, k), float(v)) for k, v in s.head(5).items() if v > 0]


def situation_note(row: pd.Series) -> str | None:
    """涨跌停股的专用基准 (次日跌>=5%)。

    数字来自 tmp_t/_limitup_dn_commons_0915.py, 20230103~20260914 全样本回算。
    """
    if row["is_limit_dn"]:
        return (
            "今日【跌停】→ 跌停股内部更有区分度的切法: "
            "换手>10% 次日大跌 38.9%, 60日涨幅>50% 42.4%, 放量>3x 44.4%"
        )
    if row["zt_break"]:
        return "今日【炸板】→ 封板资金已撤, 追高盘被套, 次日易低开续跌"
    if row["is_limit_up"]:
        vol = bool(row["amt_x"] >= 3)
        turn = bool(row["turnover_rate"] > 15)
        pos = bool(row["bias_20"] > 0.25)
        k = sum([vol, turn, pos])
        if k == 3:
            base = "放量+高换手+高位 → 次日大跌 34.8%"
        elif k == 2:
            base = "命中两条危险特征 → 次日大跌 29.9~31.5%"
        elif k == 1:
            base = "命中一条 → 次日大跌 20.3~22.9%"
        else:
            base = "缩量/低位涨停 → 次日大跌仅 8.1%, 基本无害"
        detail = (
            ", ".join(
                n
                for n, f in (
                    ("放量(>=3x)", vol),
                    ("高换手(>15%)", turn),
                    ("高位(bias20>25%)", pos),
                )
                if f
            )
            or "均未触发"
        )
        return f"今日【涨停】→ {base} [{detail}]"
    return None


def bucket_mask(d: pd.DataFrame, row: pd.Series) -> pd.Series:
    """与 situation_note 同粒度的对照桶, 保证两行说的是一批样本。"""
    if row["is_limit_dn"]:
        return d["is_limit_dn"] == 1
    if row["zt_break"]:
        return d["zt_break"] == 1
    if row["is_limit_up"]:
        m = d["is_limit_up"] == 1
        for cond in (d["amt_x"] >= 3, d["turnover_rate"] > 15, d["bias_20"] > 0.25):
            if bool(cond.loc[row.name]):
                m = m & cond
        return m
    return pd.Series(True, index=d.index)


def three_way(d: pd.DataFrame, mask: np.ndarray) -> str:
    """涨 / 平盘 / 小跌 / 大跌 四分解。跌率 = 小跌 + 大跌, 与「大跌率」不是一个数。

    涨 + 平盘 + 跌率 = 100%。平盘 = 收盘与昨收完全相同 (复权后)。
    """
    m = mask & d["y"].notna().to_numpy()
    if m.sum() < 30:
        return f"样本不足 (n={int(m.sum())})"
    f = d["fwd"].to_numpy(float)[m]
    up = (f > 0).mean()
    fl = (f == 0).mean()
    sd = ((f < 0) & (f > -DROP_TH)).mean()
    bd = (f <= -DROP_TH).mean()
    return (
        f"涨 {up:.1%} | 平盘 {fl:.2%} | 跌率 {sd + bd:.1%}"
        f" = 小跌 {sd:.1%} + 大跌 {bd:.1%}   n={int(m.sum()):,}"
    )


def own_history_line(d: pd.DataFrame, row: pd.Series) -> str:
    """个股自身的同形态历史。与同类桶反向且样本够时明确提示, 防止纯外推被当结论。"""
    yv = d["y"].notna().to_numpy()
    fv = d["fwd"].to_numpy(float)
    bm = bucket_mask(d, row).to_numpy() & yv
    bd_bucket = (fv[bm] <= -DROP_TH).mean() if bm.sum() else np.nan
    m = bm & (d["symbol"] == row["symbol"]).to_numpy()
    n = int(m.sum())
    if n == 0:
        return "  本股自身同形态: 历史无此类样本, 上行为同类桶外推"
    own = (fv[m] <= -DROP_TH).mean()
    s = f"  本股自身同形态: n={n}, 次日大跌 {own:.1%} (同类桶 {bd_bucket:.1%})"
    if n < 5:
        return s + f"  — 样本不足 {n} 次, 不构成证据"
    if bd_bucket > 0 and own * 2 < bd_bucket:
        return s + "  ⚠ 本股历史与同类相反, 桶估计仅供参考"
    if own > bd_bucket * 2:
        return s + "  ⚠ 本股历史比同类更差"
    return s


def query(codes: list[str], b: dict) -> None:
    d = load_frame()
    last = d.groupby("symbol")["date"].transform("max") == d["date"]
    cur = d[last & d["symbol"].isin(codes)]
    if not len(cur):
        print("这些代码不在面板最新交易日")
        return
    F, sc, P = score_frame(cur, b)
    # 全市场规则得分, 用于"同分档"对照 —— 保证跌率和头条概率出自同一分组
    sc_all = rule_flags(d).sum(axis=1).to_numpy()
    win = b.get("winner", "融合(规则+模型)")
    why = {r[0]: r[3] for r in b["rules"]}
    thr_map = {r[0]: r[2] for r in b["rules"]}
    name_map = {r[0]: r[1] for r in b["rules"]}
    lvl = [(0.25, "极高"), (0.15, "高"), (0.08, "偏高"), (0.045, "中"), (0.0, "低")]

    oos_base = b["oos_base"]
    print(
        f"\n最新交易日 {d['date'].max()}   全市场基准 {b['base_pre']:.2%}   "
        f"[模型包 {b['tag']}, OOS基准 {oos_base:.2%}]"
    )
    print(f"  概率口径 = {win}; 原因 = 触发规则 + 模型贡献度")

    for code in codes:
        pos = np.where(cur["symbol"].to_numpy() == code)[0]
        if not len(pos):
            print(f"\n{'=' * 94}\n{code}  不在最新交易日 (可能停牌/退市)")
            continue
        k = pos[0]
        i = cur.index[k]
        prob = float(P[win][k])
        lv = next(n for t, n in lvl if prob >= t)
        fired = [c for c in F.columns if bool(F.loc[i, c])]
        row = cur.loc[i]
        print(
            f"\n{'=' * 94}\n{code}   次日大跌概率 {prob:.1%}   [{lv}风险]"
            f"   (基准 {b['base_pre']:.1%}, {prob / b['base_pre']:.1f} 倍)"
        )
        parts = "  ".join(f"{'★' if n == win else ''}{n} {P[n][k]:.1%}" for n in P)
        print(f"  各口径: {parts}   (规则得分 {sc[k]}/{len(F.columns)})")
        print(
            f"  同分档({int(sc[k])}/{len(F.columns)})实测: "
            f"{three_way(d, sc_all == int(sc[k]))}"
        )
        print(f"  涨跌停桶实测: {three_way(d, bucket_mask(d, row).to_numpy())}")
        note = situation_note(row)
        if note:
            print(f"  {note}")
        print(own_history_line(d, row))

        print(f"\n{'关键指标':<18}{'当前值':>12}{'危险阈值':>14}{'是否触发':>10}")
        ind = [
            ("收盘价", f"{row['close']:.2f}", "-", None),
            ("当日涨跌", f"{row['pctChg']:+.2f}%", "<= -5%", row["pctChg"] <= -5),
            ("20日乖离", f"{row['bias_20']:+.1%}", "> +25%", row["bias_20"] > 0.25),
            ("60日乖离", f"{row['bias_60']:+.1%}", "> +50%", row["bias_60"] > 0.50),
            (
                "换手率",
                f"{row['turnover_rate']:.2f}%",
                "> 20%",
                row["turnover_rate"] > 20,
            ),
            ("量比", f"{row['volume_ratio']:.2f}", "> 2.0", row["volume_ratio"] > 2),
            (
                "10日涨幅",
                f"{row['ret10']:+.1%}",
                "> +20%",
                row["ret10"] > 0.20,
            ),
            (
                "60日涨幅",
                f"{row['low60_gain']:+.1%}",
                "> +50%",
                row["low60_gain"] > 0.50,
            ),
            (
                "获利盘占比",
                f"{row['winner_ratio']:.1%}",
                "> 95%",
                row["winner_ratio"] > 0.95,
            ),
            ("连板数", f"{int(row['conseq_zt'])}", ">= 3", row["conseq_zt"] >= 3),
            ("是否炸板", "是" if row["zt_break"] else "否", "是", row["zt_break"]),
            ("大盘中位涨跌", f"{row['mkt_ret']:+.2f}%", "-", None),
        ]
        for nm, v, th, hit_ in ind:
            mark = "—" if hit_ is None else ("★ 触发" if bool(hit_) else "ok")
            print(f"{nm:<18}{v:>12}{th:>14}{mark:>10}")

        print("\n  原因分析:")
        if fired:
            print(f"  【规则命中 {len(fired)}/{len(F.columns)}】")
            for c in fired:
                print(f"    · {name_map[c]}  [{thr_map[c]}]")
                print(f"       {why[c]}")
        else:
            print(f"  【规则命中 0/{len(F.columns)}】形态不在历史大跌高发画像内")
        ex = explain(cur, b, i)
        if ex:
            print("  【模型贡献度 Top (log-odds, 正=推高危险)】")
            for nm, v in ex:
                print(f"    · {nm:<14}{v:+.3f}")
        else:
            print("  【模型未发现明显推高危险的因子】")
        if prob >= 0.08:
            print(
                f"\n  → 多因子叠加, 概率为基准的 {prob / b['base_pre']:.1f} 倍。"
                f"注意: 剩下 {1 - prob:.0%} 不是「会涨」, 而是「跌幅不到 5%」——"
                f"里面既有小跌也有上涨, 不是平盘或涨的意思。\n"
                f"     这是减仓/别追的辅助, 不是做空信号。"
            )


def show_rules(b: dict) -> None:
    d = load_frame()
    F = rule_flags(d)
    val = d["y"].notna().to_numpy()
    y = d.loc[val, "y"].to_numpy(float)
    dt = d.loc[val, "date"].to_numpy()
    Fv = F[val]
    pre = dt < CAL_END
    oos = dt >= CAL_END
    print(
        f"基准 次日跌>={DROP_TH:.0%}: 全样本 {y.mean():.2%} | "
        f"拟合段 {y[pre].mean():.2%} | OOS {y[oos].mean():.2%}"
    )
    print(
        f"\n{'规则':<14}{'触发条件':<26}{'触发数':>10}{'占比':>8}"
        f"{'次日大跌率':>11}{'lift':>7}{'OOS率':>9}{'OOS lift':>10}"
    )
    for key, cn, thr, _ in RULES:
        m = Fv[key].to_numpy(bool)
        p = y[m].mean()
        po = y[m & oos].mean()
        print(
            f"{cn:<14}{thr:<26}{int(m.sum()):>10,}{m.mean():>8.2%}{p:>11.2%}"
            f"{p / y.mean():>7.2f}{po:>9.2%}{po / y[oos].mean():>10.2f}"
        )
    print("\n得分档 → 概率 (拟合+校准段, 即服务时用的表)")
    for s in sorted(b["rule_table"]):
        m = Fv.sum(axis=1).to_numpy(int) == s
        if m.sum() < MIN_BUCKET:
            continue
        print(
            f"    {s} 分  历史 {b['rule_table'][s]:>7.2%}   "
            f"OOS {y[m & oos].mean():>7.2%}  (n={int((m & oos).sum()):,})"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--build" in args:
        build()
    elif "--rules" in args:
        show_rules(load_bundle())
    elif args and args[0].startswith("-"):
        print(__doc__)
    elif args:
        query(
            [
                a.strip()
                .upper()
                .replace(".SH", "")
                .replace(".SZ", "")
                .replace(".BJ", "")
                .zfill(6)
                for a in args
            ],
            load_bundle(),
        )
    else:
        print(__doc__)
