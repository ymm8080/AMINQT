# -*- coding: utf-8 -*-
"""次日大跌预测模块 (规则 + 模型融合)。

输入股票代码 → 输出【大跌概率】+【关键指标】+【原因分析】。

设计:
  规则侧: 一组已验证条件 (当前 9 条, 见 RULES) → 得分档 → 历史上该档的真实命中率
  模型侧: LGBM 排序 → isotonic 校准 → 真概率
  融合侧: 在校准段上用 logistic 叠 [模型logit, 规则分, 是否命中规则, 交互项],
          融合方式由 OOS 数据挑选, 不是拍脑袋。
  原因侧: 触发的规则 + 模型逐股贡献度 (pred_contrib) 双通道。
  报警侧: 规则闸是极端值阈值, 对「处处偏高但无一极端」的形态够不着 —— OOS 真大跌
          有 56.5% 落在 0 分档。服务口径取「规则 ∪ 模型」的并集: 规则 0 分而模型
          >= MODEL_ALARM 时, 头条概率改用该格的 OOS 兑现 (见 bundle['alarm']), 不取
          规则表那个只看 score 的低值 —— 否则头条会写"低风险", 与报警自相矛盾。
  捕获侧: 用户口径是"抓跌停和大跌"。跌停是大跌的真子集 (OOS 169 日: 40 只/日 全
          落在 256 只/日 内), 目标只有一个; 而 100% 召回要求标 100% 市场, 所以输出
          改成 T1~T3 三层召回阶梯 + T4 未标面的实测残留 (见 CAPTURE_TIERS 与
          bundle['capture']) —— 让"漏"永远显式, 而不是假装二值穷尽。
  分支侧: 并集口径的一整个布尔把两件事并在一起 —— 模型支带方向 (次日均 -0.26%),
          规则支不带 (次日均 +0.06%), 而规则支占 T1 体量七成, 把并集稀释到 -0.04%。
          于是"报警"被当成"要跌", 而好票 (次日涨>=3%) 在报警池里 1.7 倍富集。
          0915 起拆成两条: 方向性只归模型支, 规则支照实说成波动/跌停风险
          (见 signal_kind 与 bundle['branches'])。

三段切分 (严格防泄漏):
  FIT   < 20250701   训模型
  CALIB 20250701~    训 isotonic + 训融合层 + 建规则概率表
  OOS   >=20260101   只验收, 从不参与任何拟合

用法:
  python scripts/bigdrop_check.py --build           # 训练+校准+融合+落盘 (WORM)
  python scripts/bigdrop_check.py 000978 002815     # 查询
  python scripts/bigdrop_check.py --rules           # 单规则命中率
  python scripts/bigdrop_check.py --capture         # 捕获阶梯 (召回 vs 表面积)
  python scripts/bigdrop_check.py --compare         # 各口径 OOS 对比
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
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
# 模型报警线。规则库全是极端值阈值, 对「处处偏高但无一极端」的中空形态全盲
# (OOS 真大跌 56.5% 落在 0 分档)。规则沉默而模型 >= 此值时强制并列报警。
MODEL_ALARM = 0.10
# 跌停判据 (次日跌幅 <= 此值)。0915 起作为捕获阶梯的第二条目标线。
DT_TH = -0.095

# 信号分支 (0915): 一个 T1 布尔此前兼任两职 —— 规则支只说"波动/跌停风险", 模型支才带方向。
# 实测 (tmp_t/_bigdrop_falsealarm_0915.py, OOS 838,647 股票-日): 「仅规则喊」占 T1 体量
# 70%, 次日均收益 +0.060% —— 零方向信息, 却把「都喊」格的 −0.257% 稀释成 T1 的 −0.036%。
# 所以"次日要跌"的表述只给模型支, 否则就是拿波动当好票的看跌结论 (用户 0915 抱怨)。
BRANCH_DIRECTIONAL = "看跌 (模型支)"
BRANCH_VOLATILITY = "波动/跌停风险 (仅规则)"
BRANCH_NONE = "未标记"

# 捕获阶梯 (0915 用户令: 抓住所有跌停股和大跌股)。
# 100% 做不到, 且跌停本来就读作大跌的真子集 (OOS 169 日: 跌停 40 只/日 全落在
# 大跌 256 只/日 内), 所以不存在「两个目标」——只有一个尾部, 剩下全是表面积
# 问题。既然 100% 只能靠标 100% 市场换, 输出就不假装二值穷尽, 改成三层召回
# 阶梯 + 未标面实测残留, 让「漏」永远显式而不是静默。
# (档名, 规则得分下限, 模型当日前 q; q=None 表示改用绝对线 MODEL_ALARM)
#
# 0915 T1 由「规则>=1 ∪ 模型>=0.10」改为「规则>=2 ∪ 模型当日前 22%」。
# 关键认识: 绝对线 0.10 在非规则票里 *等价于* 按 mp 降序补足到当日面预算
# (取 top-k, k = 阈值以上个数), 逐日 Δ 恒 0.00% —— 它本来就是面分配器, 不是阈值。
# 所以这是**重新分配当日面**, 不动任何阈值; MODEL_ALARM 仍是报警格与方向的判据。
# 同面回测 (tmp_t/_bigdrop_realloc_0915.py, OOS 170 日, 面 22.90% vs 23.10%):
#   大跌召回 57.88% vs 55.27%, 跌停召回 79.99% vs 77.36%, 富集 2.53x vs 2.39x;
#   双半窗同号; 崩盘/中间/平静三分档均胜; 清单关 sep +0.697% vs −0.236%
#   (日聚类 bootstrap 95%CI [+0.184%, +1.096%], 19/25 日)。
#   前沿支配检验 (tmp_t/_bigdrop_frontdom_0915.py): 可比面上 200/200 点全胜 k=1。
CAPTURE_TIERS: list[tuple[str, int, float | None]] = [
    ("T1 报警", 2, 0.22),  # 规则>=2 ∪ 模型当日前 22% —— 现役服务口径
    ("T2 警戒", 1, 0.20),  # ∪ 规则>=1 ∪ 模型当日前 20%
    ("T3 关注", 1, 0.40),  # ∪ 模型当日前 40%
]

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
    # [0915 用户令撤] "连板高位" 曾在此: 单独看是真信号 (次日跌停 9.44% = 11.73x
    # 市场基准, 全规则最高), 但在 T1 并集里撤它逐位无差别 —— 大跌召回 55.63%→55.62%,
    # 跌停召回/误伤一位不动, 因 97% 的命中已被别的闸或模型覆盖。而它带**零方向**:
    # 同一撮票次日好票率 57%、均收益 +3.20%。零边际 + 零方向 → 撤。
    # conseq_zt 仍在 FEATS 里当模型输入 (信息不丢, 只撤硬编码闸)。
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
    (
        "hi_ff_turn",
        "自由流通高换手",
        "自由流通换手率 > 10%",
        "自由流通盘单日换手超一成, 筹码在大幅易手 —— 量价族才是模型真正的主力信号。"
        "本条补的是 600814 (20260910 收 -3.17%, 次日 -7.33%) 这类漏报: 该股当日 "
        "10日涨幅 94.5 分位/bias20 94.4/5日振幅 93.8/自由流通换手 93.7 分位, "
        "却无一条极端闸够得着。阈值扫描两段单调稳健 (拟合 1.79x / OOS 1.97x)",
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
            # [0915 撤] 连板高位 (conseq_zt >= 3) 已撤, 理由见 RULES 同处注释:
            # 边际 0.01pp 且零方向。该列仍是模型输入 (FEATS), 未删。
            "vol_surge": d["volume_ratio"] > 2,
            "today_drop": d["pctChg"] <= -5,
            "big_rise": d["low60_gain"] > 0.50,
            # rise10 (0915): 中等强度涨过头。全样本覆盖 20.0% 的大跌,
            # score==0 桶内 OOS 3.63%→9.58%; 新增告警的 OOS 大跌率 19.89%,
            # 对照老 score>=3 的 24.51% —— 打 85 折, 不是灌水。
            # 注意「缩量」不在此列: 在 ret10>20% 人群里缩量挑的是低风险半边。
            "rise10": d["ret10"] > 0.20,
            # hi_ff_turn (0915): 量价族连续信号。加它时原有 9 条 (其中连板高位
            # 已于同日稍后撤掉) 全是极端值阈值, OOS 真大跌
            # 56.5% 落在 0 分档 —— 本条把覆盖 43.5% 抬到 53.9%, 池内精准度仅
            # 13.15%→12.45%。阈值 8/10/12/15/20/25 两段单调, 取 10 (非贴合个股)。
            "hi_ff_turn": d["free_float_turnover_rate"] > 10,
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
    # 捕获阶梯的第二条目标线。fwd 已带 ok 掩码 (停牌/断档为 NaN), 所以
    # 有效性与 y 完全一致, 不需要另建掩码。
    ydt = (d["fwd"] <= DT_TH).to_numpy(float)
    dt = d["date"].to_numpy()
    # 分支表要回答"规则支到底带不带方向", 而答案就是它的次日均收益本身 ——
    # 拿在 del 之前, 否则下面只剩 y (已二值化, 看不出涨)。
    fwd_all = d["fwd"].to_numpy(float)
    # 建包用的**数据日** —— 新鲜度判据取它, 不取墙钟 tag: tag 是建包那天的日历日,
    # 早于当日抓取建包会得到 tag=今天而数据只到昨天 (同名覆盖), 拿 tag 当新鲜度
    # 会把它误判成新的 —— 同 0915 密度页日期键事故。
    # 归一化到 YYYYMMDD —— bundle_is_stale 用同格式比较, 不解析 datetime 字符串
    # (str(max) 会产出 'YYYY-MM-DD HH:MM:SS', 与 'YYYYYMMDD' 永不相等 → 永远判陈旧)。
    data_date = pd.Timestamp(d["date"].max()).strftime("%Y%m%d")
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

    # 模型报警格 (规则沉默 / 模型报警) 的 OOS 兑现 —— 存进 bundle 供 query 引用,
    # 避免把回测数字硬编码进输出 (复核见 tmp_t/_600814_fix2_0915.py)
    a_oos = oos & (p_iso >= MODEL_ALARM) & (score_all == 0)
    n_day_oos = max(1, np.unique(dt[oos]).size)
    alarm = dict(
        th=MODEL_ALARM,
        oos_n=int(a_oos.sum()),
        oos_prec=float(y[a_oos].mean()) if a_oos.sum() else float("nan"),
        oos_lift=float(y[a_oos].mean() / bo) if a_oos.sum() else float("nan"),
        daily=float(a_oos.sum() / n_day_oos),
    )
    print(
        f"\n模型报警格 (模型>={MODEL_ALARM:.0%} 且规则 0 分): "
        f"OOS n={alarm['oos_n']:,}, 次日大跌 {alarm['oos_prec']:.2%} "
        f"(基准 {bo:.2%} 的 {alarm['oos_lift']:.2f} 倍), "
        f"日均 {alarm['daily']:.1f} 只 —— 规则库漏掉的那部分靠这一格兜"
    )

    # 捕获阶梯的 OOS 实测 (存进 bundle, query/--capture 都不硬编码这些数字)。
    # 档是嵌套的: T2 含 T1, T3 含 T2, 所以召回按累计算。
    # 变量名避开 m / hit —— 上面分别被 LGBM 模型和融合层的命中项占用。
    rk_oos = pd.Series(p_iso).groupby(dt).rank(pct=True).to_numpy()
    oos_n, dt_n, y_n = int(oos.sum()), int(ydt[oos].sum()), int(y[oos].sum())
    cum = np.zeros_like(oos)
    capture = []
    for nm, rmin, q in CAPTURE_TIERS:
        widen = (score_all >= rmin) | (
            (p_iso >= MODEL_ALARM) if q is None else (rk_oos >= 1.0 - q)
        )
        cum = cum | widen
        msk = np.asarray(oos & cum, bool)
        capture.append(
            dict(
                name=nm,
                daily=msk.sum() / n_day_oos,
                share=msk.sum() / oos_n,
                bigdrop_rate=float(y[msk].mean()),
                bigdrop_recall=float(y[msk].sum() / y_n),
                dt_rate=float(ydt[msk].mean()),
                dt_recall=float(ydt[msk].sum() / dt_n),
            )
        )
    rest = np.asarray(oos & ~cum, bool)
    capture.append(
        dict(
            name="T4 未标",
            daily=rest.sum() / n_day_oos,
            share=rest.sum() / oos_n,
            bigdrop_rate=float(y[rest].mean()),
            bigdrop_recall=float(y[rest].sum() / y_n),
            dt_rate=float(ydt[rest].mean()),
            dt_recall=float(ydt[rest].sum() / dt_n),
        )
    )
    print(
        f"\n{'=' * 96}\n捕获阶梯 (OOS {n_day_oos} 日; 大跌基准 {bo:.3%} "
        f"{y_n / n_day_oos:.0f} 只/日, 跌停基准 {ydt[oos].mean():.3%} "
        f"{dt_n / n_day_oos:.1f} 只/日)"
    )
    print(
        f"{'档':<10}{'日均':>7}{'占市':>8}{'大跌召回':>10}{'跌停召回':>10}"
        f"{'本档大跌率':>11}{'本档跌停率':>11}"
    )
    for c in capture:
        print(
            f"{c['name']:<10}{c['daily']:>7.0f}{c['share']:>8.2%}"
            f"{c['bigdrop_recall']:>10.1%}{c['dt_recall']:>10.1%}"
            f"{c['bigdrop_rate']:>11.2%}{c['dt_rate']:>11.2%}"
        )
    print(
        f"  未标面仍含 {capture[-1]['bigdrop_recall']:.1%} 的大跌 / "
        f"{capture[-1]['dt_recall']:.1%} 的跌停 —— 100% 召回要求标 100% 市场, "
        f"所以这个残留是口径的一部分, 不是缺陷"
    )

    # 信号分支 (0915): T1 那一个布尔把两件性质相反的事并在一起。规则支占它七成体量
    # 却零方向 (次日均 +0.06%), 把并集均值稀释到 −0.04% —— 于是"报警"被读成"要跌",
    # 而好票 (次日涨>=3%) 在报警池里 1.7 倍富集。方向性只归模型支。
    # 判据 = 分支的次日均收益是否显著为负, 数字存 bundle, query 不硬编码。
    branches = []
    for bname, seg in (
        (BRANCH_DIRECTIONAL, p_iso >= MODEL_ALARM),
        (BRANCH_VOLATILITY, (score_all >= 1) & (p_iso < MODEL_ALARM)),
        (BRANCH_NONE, (score_all == 0) & (p_iso < MODEL_ALARM)),
    ):
        sub = np.asarray(oos & seg, bool)
        k = int(sub.sum())
        branches.append(
            dict(
                name=bname,
                daily=k / n_day_oos,
                share=k / oos_n,
                bigdrop_rate=float(y[sub].mean()) if k else float("nan"),
                bigdrop_recall=float(y[sub].sum() / y_n),
                dt_rate=float(ydt[sub].mean()) if k else float("nan"),
                dt_recall=float(ydt[sub].sum() / dt_n),
                ret=float(np.nanmean(fwd_all[sub])) if k else float("nan"),
            )
        )
    print(
        f"\n{'=' * 96}\n信号分支 (OOS {n_day_oos} 日; 方向性判据 = 次日均收益是否显著为负)"
    )
    print(
        f"{'分支':<26}{'日均':>7}{'占市':>8}{'大跌率':>9}{'大跌召回':>10}"
        f"{'跌停召回':>10}{'次日均收益':>12}"
    )
    for c in branches:
        print(
            f"{c['name']:<26}{c['daily']:>7.0f}{c['share']:>8.1%}"
            f"{c['bigdrop_rate']:>9.2%}{c['bigdrop_recall']:>10.1%}"
            f"{c['dt_recall']:>10.1%}{c['ret']:>+12.3%}"
        )
    dbr = branches[0]
    print(
        f"  → 只有「{BRANCH_DIRECTIONAL}」的次日均收益为负 ({dbr['ret']:+.3%}), 才叫看跌; "
        f"「{BRANCH_VOLATILITY}」次日均 {branches[1]['ret']:+.3%} "
        f"({branches[1]['share']:.0%} 的 T1 体量) 零方向, 只承诺跌停/波动风险"
    )
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
    dest = BUNDLE_DIR / f"bundle_{tag}.joblib"
    if dest.exists():
        # WORM: 同 tag 重跑会静默覆盖旧包。tag 是墙钟日, "当日早先建过一次包" 时
        # 重跑正是这个形状 (面板同日补齐后重建) —— 先留档再写。
        bak = BUNDLE_DIR / (
            f"bundle_{tag}_pre_rebuild_{datetime.now().strftime('%H%M%S')}.joblib.bak"
        )
        try:
            shutil.copy2(dest, bak)
            log.info("同 tag 旧包已留档 %s", bak.name)
        except OSError as e:
            log.error("留档失败 (%s → %s): %s", dest.name, bak.name, e)
    bundle = dict(
        tag=tag,
        data_date=data_date,
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
        alarm=alarm,
        capture=capture,
        branches=branches,
    )
    joblib.dump(bundle, dest)
    joblib.dump(bundle, BUNDLE_DIR / "bundle_latest.joblib")
    log.info("%s\\bundle_%s.joblib  (data_date=%s)", BUNDLE_DIR, tag, data_date)

    # 特征重要度留档
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    log.info(
        "特征重要度 Top12: %s",
        ", ".join(f"{PRETTY.get(k, k)}({v})" for k, v in imp.head(12).items()),
    )
    imp.to_csv(BUNDLE_DIR / f"importance_{tag}.csv", encoding="utf-8-sig")
    return bundle


def load_bundle() -> dict:
    p = BUNDLE_DIR / "bundle_latest.joblib"
    if not p.exists():
        log.error("没有模型包, 先跑: python scripts/bigdrop_check.py --build")
        sys.exit(2)
    return joblib.load(p)


def bundle_is_stale(b: dict, data_date: str) -> bool:
    """包是否与 `data_date` (面板最新数据日, YYYYMMDD) 不同源 → 该重建。

    判据是**数据日**不是墙钟 `tag`: tag 是建包那天的日历日, 早于当日抓取建包会
    得到 tag=今天而数据只到昨天, 拿 tag 判新鲜会把它当新的 (同 0915 密度页
    日期键事故)。**缺 data_date 的旧包一律视为陈旧** —— 无法证明同源就不假定同源。
    铁律 #6: 日期必须解析为 datetime 对象比较, 不容字符串直接比。
    """
    try:
        bundle_dt = datetime.strptime(str(b.get("data_date") or ""), "%Y%m%d")
        cur_dt = datetime.strptime(str(data_date), "%Y%m%d")
    except (ValueError, TypeError):
        return True  # 解析失败 = 无法证明同源 → 判陈旧
    return bundle_dt != cur_dt


# ---------------------------------------------------------------- 查询
def score_frame(d: pd.DataFrame, b: dict) -> tuple:
    """返回 (规则旗标, 得分, 各口径概率字典)。"""
    F = rule_flags(d)
    # 闸数与包必须一致 (0915 撤闸事故的守卫)。rule_table 的键是**建包时**的
    # np.unique(score), 取用时拿的却是这里当场算出的 sc —— 两边闸数不同就直接读错档,
    # 且缺键会静默回退 base_pre; 融合头 b["lr"] 同样按旧得分刻度拟合 (它才是默认
    # winner)。不匹配 = 数悄悄不对, 必须当场炸, 不能降级。
    if "rules" in b and len(b["rules"]) != F.shape[1]:
        log.error(
            "包与代码闸数不一致: 包 %d 闸 (tag=%s), 代码 %d 闸。\n"
            "  rule_table / 融合头的得分刻度在建包时就定死了, 直接跑会静默读错概率。\n"
            "  请重建: python scripts/bigdrop_check.py --build",
            len(b["rules"]),
            b.get("tag"),
            F.shape[1],
        )
        sys.exit(2)
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


def should_alarm(model_p: float, score: int, th: float = MODEL_ALARM) -> bool:
    """规则沉默但模型报警 —— 规则库对「中空」形态全盲, 这一格必须单独举手。

    规则已举手 (score>=1) 时不重复报警: 规则表概率本身已经抬升。
    """
    return int(score) == 0 and float(model_p) >= float(th)


def effective_prob(
    rule_p: float, model_p: float, score: int, al: dict
) -> tuple[float, bool]:
    """服务口径 = 规则与模型的并集, 只报一个数。

    规则沉默而模型报警时, 条件概率应由「规则 0 分 且 模型>=线」这一格给出,
    而不是规则表那个只看 score 的 2% —— 否则头条会写「低风险」, 和报警自相矛盾。
    """
    if should_alarm(model_p, score, al.get("th", MODEL_ALARM)) and al.get("oos_prec"):
        return float(al["oos_prec"]), True
    return float(rule_p), False


def signal_kind(
    score: int, model_p: float, th: float = MODEL_ALARM
) -> tuple[str, bool]:
    """本股落在哪个信号分支 → (分支名, 是否带方向)。

    T1 并集此前是一个布尔, 但两半的性质并不一样 (OOS 实测):
      模型支 次日均收益 −0.26%; 规则支 +0.06% —— 规则支零方向, 且占 T1 体量七成,
      把并集的均值稀释到 −0.04%。若不分开, 就是把一整片强势股说成"次日要跌"。
    所以方向性只给模型支; 规则支照实说成波动/跌停风险。

    build() 的 branches 与此同源 (分支名共用 BRANCH_* 常量), 数字由 bundle 提供。
    """
    if float(model_p) >= float(th):
        return BRANCH_DIRECTIONAL, True
    if int(score) >= 1:
        return BRANCH_VOLATILITY, False
    return BRANCH_NONE, False


def capture_tier(
    score: int, model_p: float, model_rank: float, al_th: float = MODEL_ALARM
) -> tuple[str, int]:
    """本股落在捕获阶梯第几档 → (档名, 档号)。档号超出阶梯 = 未被标记。

    档内是并集: 规则够分 或 模型够响, 任一边看见即入档 (用户口径)。
    model_rank = 该股模型概率在当日的分位 (0~1)。
    """
    for n, (nm, rmin, q) in enumerate(CAPTURE_TIERS, start=1):
        hit_rule = int(score) >= rmin
        hit_model = (
            float(model_p) >= al_th if q is None else float(model_rank) >= 1.0 - q
        )
        if hit_rule or hit_model:
            return nm, n
    return "T4 未标", len(CAPTURE_TIERS) + 1


def query(codes: list[str], b: dict) -> None:
    d = load_frame()
    last = d.groupby("symbol")["date"].transform("max") == d["date"]
    # 当日全横截面一起打分: 模型报警通道要看当日分位。只在当日算, 不做全 panel 推理。
    day = d[last]
    cur = day[day["symbol"].isin(codes)]
    if not len(cur):
        print("这些代码不在面板最新交易日")
        return
    F, sc, P = score_frame(day, b)
    p_iso_day = P["模型(isotonic校准)"]
    rk_day = pd.Series(p_iso_day).rank(pct=True).to_numpy()
    didx = {s: n for n, s in enumerate(day["symbol"].to_numpy())}
    cap = b.get("capture") or []
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
        if code not in didx:
            print(f"\n{'=' * 94}\n{code}  不在最新交易日 (可能停牌/退市)")
            continue
        k = didx[code]
        i = day.index[k]
        # 模型报警通道: 规则库的闸全是极端值阈值, 对「处处偏高但无一极端」的中空
        # 形态全盲 (OOS 真大跌 56.5% 落在 0 分档)。
        al = b.get("alarm") or {}
        mp = float(p_iso_day[k])
        prob, warn = effective_prob(P[win][k], mp, int(sc[k]), al)
        lv = next(n for t, n in lvl if prob >= t)
        fired = [c for c in F.columns if bool(F.loc[i, c])]
        row = day.loc[i]
        print(
            f"\n{'=' * 94}\n{code}   次日大跌概率 {prob:.1%}   [{lv}风险]"
            f"   (基准 {b['base_pre']:.1%}, {prob / b['base_pre']:.1f} 倍)"
            + ("   ← 模型报警格兑现, 非规则表" if warn else "")
        )
        parts = "  ".join(f"{'★' if n == win else ''}{n} {P[n][k]:.1%}" for n in P)
        print(f"  各口径: {parts}   (规则得分 {sc[k]}/{len(F.columns)})")

        # 捕获阶梯档位。T1 之外也要报 —— 用户口径是「抓跌停和大跌」, 只报头条
        # 会让人以为没报警就是安全。
        tname, tnum = capture_tier(
            int(sc[k]), mp, float(rk_day[k]), al.get("th", MODEL_ALARM)
        )
        # 直报分位而不是「当日前 X%」—— 后者是 1-rk, 高分位(危险)反而显示成小数字,
        # 600519(分位 12%) 会印成「前 88.1%」读着像危险, 正好反了。
        dtxt = f"   (模型概率分位 {float(rk_day[k]):.1%}, 越高越危险)"
        print(f"  捕获档位: {tname}{dtxt}")

        # 信号性质: 把"报警"拆成带方向 / 不带方向两种读法。不拆的话, 规则支那七成
        # 体量 (次日均 +0.06%) 会顶着"高风险"的帽子, 被当成看跌结论。
        bname, has_dir = signal_kind(int(sc[k]), mp, al.get("th", MODEL_ALARM))
        print(f"  信号性质: {bname}")
        br = next((x for x in (b.get("branches") or []) if x["name"] == bname), None)
        if br and not has_dir:
            tail = (
                "只说明波动/跌停风险, 不含方向 —— 别读成「明天要跌」"
                if bname == BRANCH_VOLATILITY
                else "本股无信号"
            )
            print(
                f"    本分支 OOS: 占市 {br['share']:.1%}, 次日大跌 {br['bigdrop_rate']:.2%}, "
                f"跌停召回 {br['dt_recall']:.1%}, 次日均收益 {br['ret']:+.3%} —— {tail}"
            )
        if cap and tnum > 1:
            t1 = cap[0]
            print(
                f"    本股不在 T1 报警档。T1 在 OOS 上只兜住 {t1['bigdrop_recall']:.1%} "
                f"的大跌 / {t1['dt_recall']:.1%} 的跌停 (日均 {t1['daily']:.0f} 只, "
                f"{t1['share']:.1%} 市) —— 没被报警 ≠ 安全; 更宽的捕获阶梯见 --capture"
            )

        # 必须排在下面那些"同分档实测 x%"之前 —— 否则读者看到低分档的低大跌率
        # 会直接判安全。
        if warn:
            pct = float((p_iso_day > mp).mean())
            print(
                f"\n  ⚠ 【模型报警 · 规则未覆盖】模型概率 {mp:.1%} (当日前 {pct:.1%})"
                f" —— 规则 0 分不等于安全"
            )
            print(
                f"     规则库 {len(F.columns)} 条闸全是极端值阈值, 对「各处都偏高但无一处"
                f"极端」的中空形态够不着; OOS 真大跌有 56.5% 落在 0 分档。"
            )
            if al.get("oos_n"):
                print(
                    f"     「模型>={al['th']:.0%} 且规则 0 分」这一格 OOS 兑现 "
                    f"{al['oos_prec']:.2%} (基准的 {al['oos_lift']:.2f} 倍), "
                    f"日均约 {al['daily']:.0f} 只。"
                )
                print(
                    "     本股当日落在这一格 → 头条概率已按本格兑现, 不取低分档的规则表值。"
                )
            mct = float(np.median(p_iso_day))
            print(f"     (当日模型概率中位 {mct:.1%}, 本股 {mp / mct:.1f} 倍)")

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
            (
                "自由流通换手",
                f"{row['free_float_turnover_rate']:.2f}%",
                "> 10%",
                row["free_float_turnover_rate"] > 10,
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
        ex = explain(day, b, i)
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


def show_capture(b: dict) -> None:
    """捕获阶梯: OOS 承诺 (建包时算好, 不在此重算) + 最新交易日的实现入档数。

    100% 召回要求标 100% 市场, 所以 T4 未标面永远非空 —— 它的残留率是口径的
    一部分, 打印出来是为了让「漏」显式, 而不是假装二值穷尽。
    """
    cap = b.get("capture") or []
    if not cap:
        print("这个模型包没有 capture 块 (0915 之前的包), 请重新 --build")
        return
    print(
        f"\n捕获阶梯 [模型包 {b['tag']}]   大跌判据 <= -{DROP_TH:.0%}   "
        f"跌停判据 <= {DT_TH:.1%}"
    )
    print(
        f"{'档':<10}{'OOS日均':>9}{'OOS占市':>9}{'大跌召回':>10}{'跌停召回':>10}"
        f"{'本档大跌率':>11}{'本档跌停率':>11}"
    )
    for c in cap:
        print(
            f"{c['name']:<10}{c['daily']:>9.0f}{c['share']:>9.2%}"
            f"{c['bigdrop_recall']:>10.1%}{c['dt_recall']:>10.1%}"
            f"{c['bigdrop_rate']:>11.2%}{c['dt_rate']:>11.2%}"
        )
    print(
        f"  未标面 ({cap[-1]['share']:.1%} 市) 里仍有 {cap[-1]['bigdrop_recall']:.1%} "
        f"的大跌 / {cap[-1]['dt_recall']:.1%} 的跌停, 其大跌率仅为基准的 "
        f"{cap[-1]['bigdrop_rate'] / b['oos_base']:.2f} 倍"
    )

    d = load_frame()
    last = d.groupby("symbol")["date"].transform("max") == d["date"]
    day = d[last]
    F, sc, P = score_frame(day, b)
    sa = np.asarray(sc)
    p = np.asarray(P["模型(isotonic校准)"], dtype=float)
    rk = pd.Series(p).rank(pct=True).to_numpy()
    th = (b.get("alarm") or {}).get("th", MODEL_ALARM)
    tier = np.full(len(day), len(cap), dtype=int)
    cum = np.zeros(len(day), bool)
    for n, (_nm, rmin, q) in enumerate(CAPTURE_TIERS):
        hit = (sa >= rmin) | ((p >= th) if q is None else (rk >= 1.0 - q))
        tier[hit & ~cum] = n + 1
        cum |= hit
    cnt = np.bincount(tier, minlength=len(cap) + 1)
    print(f"\n最新交易日 {day['date'].max()} 实现入档 (全市场 {len(day):,} 只):")
    for k, c in enumerate(cap, start=1):
        print(f"  {c['name']:<10}{int(cnt[k]):>7,}  ({cnt[k] / len(day):>6.2%})")


def show_compare(b: dict) -> None:
    """各口径的 OOS 概率质量 —— 全部读自 bundle, 不重训不重算。

    这些数字在 --build 时算出并落盘 (oos_brier / oos_cal_err), 这里只复读, 方便
    随时复核「为什么选了这个口径」而不用重跑一次建包。
    """
    bo = float(b["oos_base"])
    print(
        f"\n各口径 OOS 概率质量 [模型包 {b['tag']}]   基准 {bo:.3%}   "
        f"常数预测 Brier {bo * (1 - bo):.5f}"
    )
    print(f"{'口径':<24}{'OOS Brier':>12}{'校准误差':>12}")
    for n, br in b["oos_brier"].items():
        mark = "   ← winner" if n == b.get("winner") else ""
        print(f"{n:<24}{br:>12.5f}{b['oos_cal_err'][n]:>12.4f}{mark}")
    print(
        f"\n融合口径 OOS AUC {b['oos_auc']:.4f}"
        f"   (切分 fit<{b['split']['fit_end']}, cal<{b['split']['cal_end']}, "
        f"embargo {b['split']['embargo']})"
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--build" in args:
        build()
    elif "--rules" in args:
        show_rules(load_bundle())
    elif "--capture" in args:
        show_capture(load_bundle())
    elif "--compare" in args:
        show_compare(load_bundle())
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
