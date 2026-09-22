"""Global configuration: paths, stock pool, date ranges, execution mode.

Secrets (iFinD credentials) are loaded from environment / .env — never
hardcoded. Date handling uses datetime objects, not string comparison.
"""

import os
from datetime import date
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = PROJECT_ROOT.parent / "PARQUET"  # 主数据目录 (V3 面板所在)
RAW_DIR = DATA_DIR / "raw"
INTRADAY_DIR = DATA_DIR / "intraday"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "app" / "models" / "trained"

# Non-parquet data outputs (logs, reports, JSON state, CSV exports, etc.)
# are kept outside the repo data/ directory so that data/ contains only
# parquet-format analytical datasets.  Override via AMINQT_DATA_OTHERS env.
DATA_OTHERS_DIR = Path(
    os.getenv("AMINQT_DATA_OTHERS", str(PROJECT_ROOT.parent / "DATA OTHERS"))
)

# 所有回测报告 (并行 PIPELINE / LHB v2 / BT v3 等) 集中落盘处 (2026-08-04 用户).
BACKTEST_RESULT_DIR = DATA_OTHERS_DIR / "BACKTESTING RESULT"

# 每日操作输出 (预测短名单 / 股票清单) — 用户指定落盘处 (2026-08-05).
DAILY_OPERATION_DIR = Path(
    os.getenv("AMINQT_DAILY_OPERATION", str(PROJECT_ROOT.parent / "DAILY OPERATION"))
)
STOCK_LIST_DIR = DAILY_OPERATION_DIR / "STOCK LIST"

for _d in (
    RAW_DIR,
    INTRADAY_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    DATA_OTHERS_DIR,
    BACKTEST_RESULT_DIR,
    STOCK_LIST_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ── 预测稳定性: 输出级时间平滑 (2026-08-06, 对齐 parallel _shortlist_t5_t10) ──
# 同一只股票相邻交易日预测/概率剧变 → 每股 forecast 列 = 近 K 个可用交易日 raw 预测的
# 衰减加权均值 (w_k = α·(1-α)^k, 归一化, gap-robust). α 越大越信任今日. 历史底稿 WORM
# 落盘 legacy_preds_raw_<date>__<module>.csv (模块标签见 module-tag 约定).
LEGACY_SMOOTH_ENABLED = True
LEGACY_SMOOTH_ALPHA = 0.35
LEGACY_SMOOTH_K = 12

# ── legacy 双板特征构建子进程并行 (2026-08-21, 08-25 开启) ──
# 语义: main/dual 帧股票集不相交 + build 无状态纯函数 → 拆板并行结果与串行逐字节一致
# (tests/test_parallel_feat_worker.py 验证). 省 ≈ 整个 dual 构建时间 (串行=main+dual,
# 并行=max(main,dual)). 2026-08-25 真面板 300d 对照: 逐字节一致, 2109s→1472s (1.43x).
LEGACY_PARALLEL_FEATURES = True

# ── legacy 超额标签 (2026-08-29 行情重审后 dual 采纳, main 维持关闭) ──
# 训练目标 label_pm_{3,5,10}d_net 板内按日去均值 (学"跑赢同板市场"); 重训时近
# LEGACY_MKT_EXPECT_WINDOW 个已实现决策日的板内等权日均值作为常数存进 bundle,
# 推理端 pred_ret_{k}d 加回常数复原绝对口径 → 闸/清单/概率语义不变, 只有日内
# 排名变化. dual 采纳依据 (08-29 行情重审): 125d 全窗 +1.37pp/日双半窗正 + 4 段
# 跌市全正 (+1.47)/涨市中性; main 拒收: 跌市桶合计 -1.82pp 单一事件伪影.
# 切换由 TOP10 第二票闸在纯 OOS 60d 头对头裁决, 劣化包不进生产.
LEGACY_EXCESS_LABEL_BOARDS: tuple = ("dual",)
LEGACY_MKT_EXPECT_WINDOW = 60

# ── legacy TOP10 第二票 (2026-08-29 用户批准: 切换闸 = IC 闸 + TOP10 非劣闸) ──
# [09-02 终审改判] caliber="final_list_tool": 判决权移交已验证的终榜回放工具
# (scripts/_dual_pkg_finaltop_compare.py, 对拍过真实交付清单), retrain 脚本在
# IC 过闸后调工具对拍 current vs 新包, 套新判据 (全窗≥0 且双半≥tol_half 且
# 胜率≥win_rate_min) 才晋升. 判什么就交付什么 — 工具判的就是将上生产的那个包.
# 起因: 裸头闸错杀实锤 — main_20260902 裸头 -0.31pp 判 FAIL, 终榜口径
# +0.55pp/日 胜率 62.5% (WORM _dual_pkg_finaltop_compare_20260902_201413.json);
# 49 日裸头信噪比不足以区分改进与构建混沌 (±2pp/日). 回退 caliber="raw_head".
LEGACY_TOP10_SECOND_VOTE = {
    "enable": True,
    "caliber": "final_list_tool",  # final_list_tool | raw_head (旧裸头判据回退口)
    "tol_half": -0.005,  # [09-02] 前后半非劣容差 -0.002→-0.005 (/日):
    #   旧值曾卡死 +1.02pp/日 全窗正改进 (09-01 09:36 main 后半 -0.38pp 0.18pp 技术性)
    "win_rate_min": 0.5,  # 配对胜率下限 (终榜口径新判据)
    "eval_days": 48,  # 终榜回放窗口 (T+10 标签成熟上限)
    "min_days": 10,  # 可比日下限, 不足视为无判词 (保留旧包)
    "top_n": 10,
    # [09-02 防坏签] 构建间抽签稳定性: 构建混沌 ±2pp/日意味着"今天这一抽"可能
    # 是幸运签 (dual 09-02 坏签/09-01 中性签同输入互证). 开启后 retrain 工具闸
    # 把近 stability_window_days 日内最新 stability_max_extra 个同板构建包作为
    # 额外臂一起对拍, 晋升 = 新包自身 PASS 且 全部抽签严格多数 PASS —
    # 单抽幸运签不再晋升, 过程级坏签 (多数抽签差) 也被拦. 无额外臂时退化为
    # 单抽判词 (与未开启一致).
    "stability_enable": True,
    "stability_max_extra": 2,
    "stability_window_days": 3,
    # [09-01] 多 seed 集成判词 (仅 raw_head 口径): 修的是种子轴 ±0.4pp, 而构建间
    # 混沌 ±2pp/日才是主体 (09-02 终审). final_list_tool 口径下工具只判将交付的
    # 单包 (不重训 seeds), 更快也更对口. 回退 raw_head 时恢复生效.
    "multi_seed_enable": True,
    "multi_seed_seeds": [42, 43, 44],
    "multi_seed_agg": "median",  # median | mean
}

# ── legacy E1 q50 分位头多 seed 中位集成 (2026-09-02 A/B 判词: 默认关闭) ─────
# E7 闸3 读 pred_q50_3d/5d (>0); retrain_20260903_ms 实测 q50 早停 1 树, 地板重训
# 后贴零摆动随机翻闸 (002295 q50_3d=-0.07%). 尝试与 LEGACY_TOP10_SECOND_VOTE
# multi_seed 同机制: q50 按 seeds 独立重训 (各成员含 es+地板), 推理取中位.
# A/B 实证 (生产同配方重建, test 段 54 日; WORM q50_ensemble_ab_20260901_192538
# main / q50_ensemble_ab_20260901_210132 dual): 成员符号分歧真实 (main
# 8.5-10.7% / dual 3.9-7.9%), 翻闸 99%+ 落贴零带 (<0.5pp) — 机制确实只仲裁
# 掷硬币区, 但仲裁方向不稳定:
#   main 全窗 3d +1.13pp 纯前半驱动: 前半 +1.83pp / 后半 -3.15pp (5d +2.76/-3.98)
#   dual 全窗 3d -2.26pp / 5d -1.25pp, 两半一致负 (-2.45/-1.38; -1.16/-1.33)
# 判词: 中位投票只提供"稳定"不提供"信号"; 贴零带 q50 本身无可靠方向, seed 集成
# 救不了 (与 10d_reg multi_seed 不同: 那里判词来自 ±0.04/日 噪声包裹的真信号,
# 这里贴零带信号量≈0). 代码路径+单测保留 (models[q]/ensemble_members 双向兼容
# 已验证), 重开 enable 须带子窗稳定证据. 成本提醒: 开启 ≈ +4-5 分钟/全量重训.
QUANTILE_ENSEMBLE = {
    "enable": False,
    "quantiles": [0.50],
    "seeds": [42, 43, 44],
}

# ── V3 Panel (single source of truth) ────────────────────────
# Override via PANEL_PATH env var; defaults to D:/AMINQT/PARQUET/ directory.
# _daily_fetch.py writes here → all read paths must resolve to the same file.
PANEL_V3_PATH = Path(
    os.getenv("PANEL_PATH", str(PARQUET_DIR / "panel_full_enriched_v3.parquet"))
)
PANEL_V3_FALLBACK = DATA_DIR / "panel_full_enriched_v2.parquet"

# ── V3 CYQ 基础列删减 (2026-08-02 A/B/C 决策) ────────────────
# benefit_part 已并入 winner_ratio (别名); cost_50pct 因 cost_bias 依赖而保留.
CYQ_BASE_KEEP = [
    "winner_ratio",
    "avg_cost",
    "pct_90_high",
    "pct_90_con",
    "cost_50pct",
    "cost_95pct",
    "weight_avg",
]
CYQ_BASE_DELETE = [
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
    "pct_90_low",
    "cost_5pct",
    "cost_15pct",
    "cost_85pct",
]


def data_others_path(path: str | Path) -> Path:
    """Return the DATA_OTHERS location for a non-parquet path.

    Paths that start with ``data/`` are mapped to ``DATA_OTHERS_DIR`` so that
    ``data/`` contains only parquet-format analytical datasets. Parquet paths
    are rejected because they must remain in ``DATA_DIR``.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        raise ValueError(f"parquet files must stay in data/: {path}")
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    return DATA_OTHERS_DIR.joinpath(*parts)


def data_path(path: str | Path) -> Path:
    """Return the location under DATA_DIR for a parquet dataset path."""
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    return DATA_DIR.joinpath(*parts)


# ── Data source: "ifind" | "akshare" (akshare = fallback/dev) ──
DATA_SOURCE = os.getenv("AMINQT_DATA_SOURCE", "akshare")

# iFinD credentials (env only)
IFIND_USER = os.getenv("IFIND_USER", "")
IFIND_PASSWORD = os.getenv("IFIND_PASSWORD", "")

# Tushare token (env only)
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ── Stock pool (5-symbol test set; expand later) ─────────────
STOCK_LIST = ["000001", "000002", "600519", "000858", "600036"]

# ── Date ranges (Phase 3 split: train 18-20, val 21, test 22-24) ──
DATA_START = date(2018, 1, 1)
DATA_END = date(2024, 12, 31)
TRAIN_END = date(2020, 12, 31)
VAL_END = date(2021, 12, 31)
TEST_START = date(2022, 1, 1)

# Anti-crawl delay between symbol fetches
DOWNLOAD_SLEEP_SEC = 0.5


class ExecutionMode(str, Enum):
    """M3 execution modes."""

    AUTO = "auto"  # granted: orders sent to broker directly
    MANUAL = "manual"  # pop-up recommendation only, user confirms


# ── Execution ─────────────────────────────────────────────────
EXECUTION_MODE = ExecutionMode(os.getenv("AMINQT_EXEC_MODE", "manual"))
EXECUTION_BROKER = os.getenv("AMINQT_BROKER", "sim")  # "sim" | "xt"

# ── Risk filter hard constraints (Phase 4) ────────────────────
MIN_AMOUNT = 50_000_000  # 成交额 >= 5000万
PRICE_LIMIT_PCT = 9.5  # |涨跌幅| <= 9.5%
MAX_ACCOUNT_DRAWDOWN_PCT = 3.0  # 账户回撤 > 3% → 返回空列表

# ── V3 入库扫描 (ingest gate, 2026-08-03) ─────────────────────
# _daily_fetch.py 追加当日行前扫描: ST/*ST 股 或 上市 < INGEST_MIN_LIST_DAYS 个交易日
# 不进入 V3 面板 (universe 在入口收敛). 交易日数按面板唯一 date 列 (交易日历)
# 计算 stock_basic.list_date → trade_date 的交易日计数 (searchsorted 向量化).
INGEST_MIN_LIST_DAYS = 150

# ── 北交所剔除 (2026-09-22 用户令) ─────────────────────────────
# 08-16 宇宙解冻 (stock_basic 全市场) 后北交所随大闸入板, 但模型零 BJ 训练样本
# (30% 涨跌幅制度外 + 面板 08-14 起才有 BJ 行) → 打分=无验证外推。入库与
# build_universe 双咽喉剔此前缀; parallel load_panel 另有 0/3/6 白名单兜底。
EXCLUDE_BJ_CODE_PREFIXES = ("92", "43", "83", "87")

# ── 收盘时刻闸 (缺省取数日, 2026-09-18) ────────────────────────
# _daily_fetch.py 不传日期参数时 (计划任务路径), 取数日 = 最近一个**已收盘**的
# 开市日. 若"今天"是开市日但当前小时 < 本值, 说明当天还没收盘 → 退到上一开市日.
# 病根: 原缺省用 datetime.now() 墙钟日, 而 StartWhenAvailable 恰在凌晨补跑
# (09-17 电量休眠事故场景) → 周一凌晨会把尚未发生的"周一"行写进面板, 此后
# panel_max_date 谎报最新日, 全链新鲜度判据被污染. A 股 15:00 收盘, 留到 16:00.
MARKET_CLOSE_HOUR = 16

# ── KIMI LHB v2.0 spec 参数 (龙虎榜稀疏特征: 半衰期/情境权重/记忆下限) ──
# 见 REFERENCE/.../FEATURE/kimi LHB_v2.0_设计文档.md §3.1/§3.3/§4
LHB_V2_SPEC = {
    "h_inst": 8,  # 机构半衰期 (spec 建议 7-10)
    "h_top": 6,  # 顶级游资半衰期 (5-7)
    "h_quant": 4,  # 量化席位半衰期 (3-5)
    "h_retail": 4,  # 散户/混合半衰期 (3-5)
    "h_sell": 5,  # 抛压记忆半衰期
    "h_sellbuy": 5,  # 买卖比半衰期
    "h_conboard": 3,  # 连板衰减记忆半衰期 (§4.3)
    "w_limit_up": 1.5,  # 涨停日卖出情境权重 (§2.5)
    "w_limit_down": 1.2,  # 跌停日卖出情境权重
    "w_up5": 1.3,  # 大涨日(>5%)卖出权重
    "w_down5": 1.1,  # 大跌日(<-5%)卖出权重
    "w_flat": 1.0,  # 平盘日卖出权重
    "f_min_ratio": 0.1,  # 最小记忆值 = max(0, 历史均值×比例) (§3.3)
    "lock_thresh": 0.3,  # 机构锁仓信号阈值 F_inst > 0.3 (§4.5)
    "overheat_penalty": 0.7,  # 过热惩罚因子 (C5d≥阈值时正向资金流×0.7) (§5.2)
    "limit_up_tol": 0.001,  # 判定涨停: close >= up_limit_raw×(1−tol)
    "limit_down_tol": 0.001,  # 判定跌停: close <= down_limit_raw×(1+tol)
    "eps": 1e-6,  # 除零保护
    "circ_mv_unit": 1e4,  # 面板 circ_mv 单位 万元 → 元
}

# ── LHB v2.0 训练/评估配置 (spec §5.3 选择性偏差: 仅上榜股票池) ──
# 见 REFERENCE/.../FEATURE/kimi LHB_v2.0_设计文档.md §5.3/§6.1
LHB_V2_EVAL = {
    "horizons": [1, 3, 5],  # 标签: t+1 开盘买入 → t+1/t+3/t+5 收盘 (T+1 模拟)
    "split_ratio": 0.8,  # 时间切分: 前 80% 上榜日训练, 后 20% 评估
    "quantile": 0.2,  # 多空分位 (预测 top/bottom 20%)
    "min_ic_obs": 20,  # 单特征 IC 至少需要的日期数
    "min_ic_n": 10,  # 单日 spearman 至少需要的股票数
    "lgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# ── 大宗交易 v3 (dim33) 训练/评估配置 (用户定案 2026-08-03) ──
# 用户要求: dim33 4 个稳定负向 EWMA 特征须在"相关数据集"(大宗事件池)上训练/评估,
# 不在全市场面板上评估 (全市场 98.8% 行零值, 稀释稀疏事件信号). 协议同 LHB v2 §5.3.
BT_V3_EVAL = {
    "horizons": [1, 3, 5],  # 标签: t+1 开盘买入 → t+1/t+3/t+5 收盘 (T+1 模拟, hfq)
    "split_ratio": 0.8,  # 时间切分: 前 80% 事件日训练, 后 20% 评估 (仅事件池内)
    "quantile": 0.2,  # 多空分位 (预测 top/bottom 20%)
    "min_ic_obs": 20,  # 单特征 IC 至少需要的日期数
    "min_ic_n": 10,  # 单日 spearman 至少需要的股票数
    "lgb": {
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# ── LHB v2.0 TOP10 周频选股评估 (用户目标: 每周输出 10 只买入标的) ──
LHB_V2_TOP10 = {
    "top_n": 10,  # 每期入选股票数
    "cost_commission": 0.00025,  # 佣金 万2.5 (单边)
    "cost_stamp": 0.0005,  # 印花税 0.05% (仅卖出)
    "cost_slippage": 0.0010,  # 滑点 0.10% (单边)
    "exclude_st": True,  # 剔除 ST (基座 is_st 待修, 当前可能不触发)
}

# ── 每日短名单评分权重 (2026-08-05 用户: 先给每股预测(涨幅+达到概率), 再按权重打分, 再排名) ──
# 每股综合分 score_w = Σ_h horizon_w[h] × ( gain_w × 涨幅min-max归一化 + prob_w × 达到概率min-max归一化 )
# 涨幅与达到概率都按当日全部入选股 min-max 归一化到 0~1 (同量纲, 真 50:50; 概率用原始值≈常数会使排名被涨幅主导).
# 排名 = 综合分降序. 主视界 = T+3 (2026-08-05 用户定案: 短持 3 天, 3d 权重最高 0.40).
# 达到概率 = P(已实现 MFE ≥ 该预期涨幅) — 真实口径 (2026-08-05 用户定案, 见 _shortlist_t5_t10.py).
SHORTLIST_SCORE = {
    "horizon_w": {"3d": 0.40, "5d": 0.25, "10d": 0.10},  # 主视界 T+3 (短持)
    "gain_w": 0.50,  # 预期涨幅权重
    "prob_w": 0.50,  # 达到概率(convincing rate)权重
    # 入选门 = 纯 T+3 门 (2026-08-09 删 2d 视界: 原 T+2/T+3 联合门退化)
    # 保留 ⇔ T+3 可兑现净预期涨幅 (pred_ret_3d, close-to-close, 成本已扣) > t3_min
    # 2026-08-10: 门基准从 pred_mag_3d (MFE 最大浮盈, 虚高) 改为 pred_ret_3d (c2c 实得),
    # 与 legacy 收益闸口径一致 — 只放行"值得买入、有可兑现预期收益"的个股.
    # 2026-08-14: t3_min 扫描 (_diag_t3min_sweep_20260814, 250d OOS) 定案分板块阈值 —
    #   dual 0.50% 真实赢 (命中10d 66→68%, 实得10d +7.08→+8.01%, ≥+10% 35→38%,
    #   4 子窗全 ≥ 基线, 只砍弱市空仓日); main 无档赢基线 (0.25% +1pp命中/+0.16pp实得
    #   但出股 -18% 不值, 更高档实得反降) → 保持 0.
    # 2026-08-21: V3 扩建(4960只)后重扫 (_diag_t3min_sweep_250d_20260821) — 甜点下移:
    #   dual 0.5% 已输基线 (实得10d +3.68% vs +3.90%, 命中 56% vs 57%); 0.25% 仅
    #   +0.17pp/+1pp 且 2/4 子窗反输+砍13%出股 → 未过 ≥3/4 子窗纪律 → dual 回 0;
    #   main 仍无档赢基线保持 0. 用户 08-21 拍板: dual 改回 0.
    "select_gate": {
        # T+3 c2c 净预期涨幅下限: 分板块 dict (main=0 / dual=0), 或全局 float (向后兼容)
        "t3_min": {"main": 0.0, "dual": 0.0},
    },
}

# ── legacy 入选闸分板块参数 (2026-08-14 全量 250d OOS 定案, _diag_legacy_hitrate_topn) ──
# prob_margin: E7 概率闸 (compound_prob > base_rate) 上再收紧: dual +0.08 →
#   命中 62.2→66.3% / 实得 +5.84→+6.66% (出票 3.2→2.9 票/日).
#   2026-08-24 final p_reg 重扫 (WORM legacy_margin_sweep_preg_20260824_200719):
#   p_reg 修复 main 概率分辨率 (旧 p_cls 主板反预测/塌缩 → 无法评估保持 0) 后,
#   main +0.08 (top-10 +3.53→+3.81%, 池 469→210/日 无碍) / dual +0.10
#   (+8.63→+9.20%, 命中 +1.4pp, 容量 31→33 天<10 票 几乎免费); 4 子窗稳定.
#   勿再收至 0.15+: dual 池重尾 (中位 16/日) 使 47/117 天 <10 票, 清单填不满.
# pain_max: [E2] 痛苦闸分板块上限: dual 0.5→0.4 (叠加边际后 命中 66.3→75.3% /
#   实得 +6.66→+7.39%, 出票日 177→76 宁缺毋滥); 0.3 过严崩 (59.5%), 勿再收;
#   main 2026-08-24 125d 回放定案: 撤闸 (0.5→1.0 等效关) — pain_off top5 命中
#   50.3→52.2% / 实得 +3.53→+5.03% / ≥10% 23.5→27.4%, 出票天数不变 (99/125).
LEGACY_ENTRY_GATE = {
    "prob_margin": {"main": 0.08, "dual": 0.10},
    "pain_max": {"main": 1.0, "dual": 0.4},
    # pain 软区上限 (2026-08-31): pain 在 (pain_max, pain_soft_max] 仍硬剔, 但记入
    # gate_audit → 交付"仅供参考"节 (信息不丢, 闸不松动; 0.4 为 08-14 实证档不动)
    "pain_soft_max": {"main": 1.0, "dual": 0.5},
    # 闸3 q50 符号闸 (2026-09-02 三臂回放判死, gate3_arms_20260902_012918):
    #   条件在基线池 (prob边际+compound>0+pain) 上 top-5/top-10 实得两板块 4/4 子窗
    #   全负 (main sign +10.47% vs 撤闸 +27.07%, dual +4.30% vs +20.79%), 且池级
    #   误杀真赢家 main 56.3% / dual 92.5%; 分位闸 (τ=0.5/0.6/0.7) main 2/4 不过窗.
    #   → 默认撤闸 (q50 列仍透传清单供参考); 重开须带子窗稳定证据.
    "q50_sign_gate": False,
}

# ── 板块 制度自适应门 (2026-08-05 用户: 砍板块不能写死, 按最新市场数据动态决定) ──
# 每日生成短名单时, 用最新 OOS 市场数据 (固定视界净收益 label_pm) 判定每个板块当前
# 是否值得保留: base_ev = 当日市场全池 (board 全部股票) 平均净收益 (闭眼全买基准),
# top_ev = 当日交付桶 (已交付 T-10) 平均净收益; 每日 α = top_ev − base_ev,
# 汇总 = 全 OOS 窗均日 α; 保留 ⇔ 主视界 α > margin. 判定板块级 (2026-08-23 重锚,
# 不再分系统).
# 2026-08-23 重锚: 旧实现 base_ev 误用"交付池自身平均"当基线 (best-vs-own-avg, 结构性
# 近 0, 几乎永不通过 — 用户: "基本每天都不通过"); 现改为 config 注释本意的市场全池基线.
# 用相对基线的 alpha 而非绝对胜率门槛 — 弱市全市场基率仅 37-46%, 绝对 55% 不可达.
# 2026-08-05 用户定案: 保留判定**只看主视界** (primary_horizon), 不许用长视界兜底把没优势的
# 板块硬留下来 — "如果没有优势就不要入选" → 全不过线则空仓观望.
# 2026-08-09 用户最终定案: **不再整组剔除** — STOCK LIST 显示全部候选 (主板照常出单),
# 但制度门仍按 10d 主视界计算, 每行标注 过门=是/未过; 未过门板块在报告里明确注明
# "不建议今日买入 <板块>股票".
REGIME_GATE = {
    "enable": True,  # 2026-08-09 用户: 计算 10d 门但不整组剔除 — 清单显示 ALL 候选, 未过门个股标注 过门=未过
    "margin": 0.01,  # 至少跑赢当日市场全池基线 1pp (净收益) 才算"有优势"
    "primary_horizon": "10d",  # 2026-08-09 用户: 制度门用 10d 主视界 (与 10d 排名键一致)
    "horizons": ("3d", "5d", "10d"),  # 仅 SUMMARY 展示用 (判定只认 primary_horizon)
    "fallback": "none",  # 无优势 → 空仓观望, 不输出 (旧 "best" 兜底已废, 用户否决)
}

# ── 并行真模型概率闸 (2026-08-15 定案, _diag_parallel_gbm_signal/_wf 250d OOS) ──
# 全局 LGBM 概率头 (mfe_3d>=abs_target 二分类) + 边际闸:
#   保留 ⇔ pred_prob > base_rate + margin (legacy 配方, 扩窗训练).
# base_rate = 最近 base_rate_days 个可观测日 mfe 达标率均值 (无前瞻: 当日可观测的
# mfe 只到 latest-4, 故取可观测尾部; 与回测"近20可观测日"同语义).
# 回测: dual 命中 68→70% / 实得 +8.06→+8.82%, main 60→61% / +3.63→+4.08%,
#   双板 4/4 子窗实得赢 (扩窗训练; trailing 242d 数据饥饿勿用).
# 闸在 t3 门后、pred_mag_10d TOP-5 排名前; bundle 缺失/过旧 → fail-open 不杀清单.
PROB_GATE = {
    "enable": True,  # False → 闸关闭 (概率头照常训练但不拦截)
    "margin": 0.08,  # 边际 (legacy 配方, 平台中段, 勿扫; margin_mode=fixed 时的静态档)
    # 自适应 margin (2026-09-08 移植 legacy 引擎 app/core/gate_margin.py, 用户令
    # 硬指标全自适应化). 起步 "fixed" = 零行为变化; 125d AB 回放 (静态 0.08 vs
    # rolling_q) 过闸后翻 "rolling_q" (回退改回 "fixed" 一键还原).
    "margin_mode": "fixed",
    "margin_q": 0.90,  # 目标保留参与池前 ~10% (与 legacy 同档)
    "margin_min": 0.03,  # 地板: 差日子不许放低于 base+3% 的票 (低于静态档)
    "margin_max": 0.15,  # 顶: 静态 0.08 的约 2x 封顶
    "spread_lookback_days": 20,  # spread 池化窗 (交易日, 严格 < 当日)
    "gate_margin_dir": DATA_DIR
    / "gate_margin_parallel",  # spreads/decision WORM (与 legacy 池隔离)
    "base_rate_days": 20,  # base_rate 观测窗 (交易日)
    "abs_target": 0.03,  # 概率头目标: mfe_3d >= 3%
    # 半衰期集成 (2026-09-03 用户定案 ensB3): 逐档训练 bundle (文件名 <board>_prob_hl<hl>_),
    # serving 概率取算术均值, 排名键 mag×prob_avg 不变. 125d 配对 A/B (单 seed,
    # 复刻头口径): ensB3 main +0.56pp/日 dual +0.62pp/日 双板双半窗全正, 全面优于
    # 单档 (hl7 +0.54) 与含 60 的组合. 注意 A/B 为复刻头 (0.06/mfe_10d), 非生产配方.
    "half_lives": [7, 15, 30],
    "refit_every_days": 21,  # 训练脚本: bundle 年龄 < 此值 → skip (交易日, 逐档判断)
    "max_stale_days": 42,  # 短名单侧: bundle 年龄 > 此值 → 大声警告 + 闸失效 (fail-open)
    "model_dir": DATA_DIR
    / "prob_head",  # WORM bundle 目录 (<board>_prob_hl<hl>_<ts>.joblib)
}

# ── legacy 并行式概率闸 (2026-08-15 定案, 2026-08-16 已接线) ──
# legacy cls 概率头太粗 (闸内 22 唯一值) → blend 排名键 A/B 证伪; 用户定案建并行式
# 全局 LGBM 概率头 (mfe_3d>=abs_target 二分类), 配方镜像 PROB_GATE (扩窗训练, 250d OOS 定案).
# 训练 = scripts/_train_legacy_prob_head.py (2026-08-16 首训完成, bundle 在 model_dir);
# 接线 = daily_pipeline._prob_gate_inputs 组装输入 + list_generator.emit 调用 (已 landed).
# 排名键保持纯 pred_ret_10d (blend 证伪).
LEGACY_PROB_GATE = {
    "enable": True,  # False → 闸关闭 (概率头照常训练但不拦截)
    "gated_boards": ["main"],  # 08-22 定案: dual 全档有害→撤闸, 仅 main 过闸
    "margin": 0.22,  # 08-22 125d 重扫: main 0.22 (top-10 +5.08%, 3/3 子窗; 0.5 不可达)
    # 自适应 margin (2026-08-31): 静态 0.22 在模型/市场分布漂移后不可达 → main 板
    # 08-22 起每日全灭 (08-30 剔 70/70, 08-31 剔 109/109, 08-20 后 main 再未出票).
    # rolling_q: margin_t = clip(Q_q(近 N 日参与闸逐股 spread=pred_prob-base_rate), min, max)
    #   无历史 → 当日截面 bootstrap (top-q% 语义, 无未来数据); 均无 → 回退 margin (fail-open)
    #   连续 3 决策日 100% 剔除 → 熔断放开至 margin_min; "fixed" → 回退静态档 (一键还原)
    "margin_mode": "rolling_q",
    "margin_q": 0.90,  # 目标保留参与池前 ~10%
    "margin_min": 0.05,  # 地板: 差日子不许放低于 base+5% 的票
    "margin_max": 0.25,  # 顶: 不超旧静态档上限
    "spread_lookback_days": 20,  # spread 池化窗 (交易日, 严格 < 当日)
    "gate_margin_dir": DATA_DIR
    / "gate_margin",  # spreads/decision WORM 状态 (逐日文件)
    "base_rate_days": 20,  # base_rate 观测窗 (交易日)
    "abs_target": 0.03,  # 概率头目标: mfe_3d >= 3%
    "refit_every_days": 21,  # 训练脚本: bundle 年龄 < 此值 → skip (交易日)
    "max_stale_days": 42,  # 短名单侧: bundle 年龄 > 此值 → 大声警告 + 闸失效 (fail-open)
    "model_dir": DATA_DIR
    / "prob_head_legacy",  # WORM bundle 目录 (<board>_prob_<ts>.joblib)
    # ③+④ (08-22 定案): lr 0.03 + n800 + early stop + 地板 50 (WORM 153820)
    "es": True,  # 训练时早停 (训练窗尾隔离 val_days 交易日做验证集)
    "es_patience": 50,  # early_stopping 耐心
    "val_days": 30,  # 验证集 = 训练尾 30 个交易日
    "es_floor": 50,  # 早停树数 < 此地板 → 固定 floor 树重训 (防 dual 塌缩)
}

# ── legacy 清单板块白名单 (2026-09-10 用户令一次性判断: "两模型都换模型..如果DUAL太差就不进TOP10混排") ──
# dual_20260910 OOS IC -0.1232 史上最差 → dual 板票不进 TOP10 混排 (预测/候选落盘/
# drift 监控照常, 仅 emit 前过滤). board 取值实为 main/GEM/STAR (dual板行=GEM/STAR,
# 无"dual"字面值). 回退(恢复双创进混排): 改回 ("main", "GEM", "STAR").
LEGACY_LIST_BOARDS = ("main",)

# ── LEGACY 交付选择栈 (09-07 用户令 "FIRST UPDATE 密度MODULE" + "YOU MAKE DECISION") ──
# mode="prob10_pull" = 纯 prob_up_10d 板内降序 + 回撤闸, 跳过 E7/prob_gate/幅度键;
#   交付帽 (TOP_N=15 全局 + D18 降仓) 与风险扫描不变。
#   依据 (tmp_t/_rankkey_prob_vs_pred_0907.py, 125d 02-09..08-17 同窗同口径不去重):
#   E7 闸内换排名键 = 空操作 (B1≡A1 逐字节, mag≡prob 池内同序); 胜出来自撤闸 —
#   纯prob10+回撤闸 13.8只/日 44.2%/+6.42pp/赢家6.12/日 vs 旧栈(E7+pred键+wr5)
#   11.0只/日 33.2%/+3.77pp/赢家3.66/日, 全轴胜 (与 09-05 终表独立互证)。
#   wr5 派发闸在该臂毁值 (被切票赢率 47% > 留守 42.8%, 一并摘除 — 与 09-05 三线
#   拍板的冲突已亮, 密度/PARALLEL 两线 wr5 不动)。
# mode="e7_pred" = 旧栈回退 (E7 准入 + LEGACY_PROB_GATE + pred_ret_10d 排名)。
LEGACY_SELECTION = {
    "enable": True,
    "mode": "prob10_pull",  # 回退改 "e7_pred"
    "pull_min": -1.0,  # [0913 用户令撤回撤闸] -1.0 等效关闭; 原档 -0.10, 恢复改回
    "pull_flag_max": -0.10,  # [0913 撤删改标] 原闸档降为标注线: pull 低于此值标"回撤"不删
    # [0913 数据驱动定案] 趋势闸: prob10_pull 排序后先滤 MA10↑ 再截板内 top_n (补位).
    # 扫描 (tmp_t/_trend_gate_sweep_0913.py, 22 夜存档 0807-0904): riser5 54.1% vs
    # 53.6% (无闸), fail_trend 17.7%→10%; 5D 头换键/益盟线做键/(5D+益盟)组合键全劣.
    # "ma10_up" 开 / "off" 关.
    "trend_gate": "ma10_up",
    "board_top_n": 10,  # 每板初选数 (全局帽 TOP_N 再截, 真删不补齐)
}

# ── prob_up 来源头 (2026-09-12 用户拍板: dual 切 cls → 当夜再定为每训动态) ──
# [0912 夜用户令] cls vs reg 不固定: 每次重训 argmax 自选 serving 头, 判定落
# bundle["prob_source"] + data/others/head_choice_ledger.csv (双管线四格
# 统一台账, 只追加)。[0913 用户令] 自选判据 IC→TOP10 净 (argmax(top10_net_reg,
# top10_net_cls)): 判定对象=TOP10, IC 与 TOP10 可背离 (#11 WORM main reg 修复
# 臂 gw +.081>cls +.058 但 TOP10 −.006 vs +.050)。
# "auto" = 动态自选 (默认, 平手取 reg=历史默认头);
# 显式 "reg"/"cls" = 强制回滚旋钮 (闸判与推理都按强制头, 压过 bundle 记录)。
# 旧 bundle 无 prob_source 键 → 按 LEGACY_PROB_SOURCE_FALLBACK 回退 = auto 前
# 各板静态默认 (main=reg 残差, dual=cls)。不硬性提名 cls: main 何时切 cls 由
# 下次重训的 argmax 按该批 OOS 自判 (0912 夜 #11 WORM main cls TOP10 净 +.050
# vs reg 最好修复臂 −.006 → 下次大概率自选 cls, 但由管线说了算)。
# 同一旋钮驱动两处: V35Predictor.prob_up_kd 源 + DualTrackTrainer.validate_oos
# 闸头 (cls 经 predict_proba 取 Rank IC)。
LEGACY_PROB_SOURCE = {
    "main": "auto",
    "dual": "auto",
}
LEGACY_PROB_SOURCE_FALLBACK = {
    "main": "reg",  # reg 残差派生概率 1-F_e(CLS_THRESHOLD-pred)
    "dual": "cls",  # Platt 校准 cls predict_proba
}

# ── LEGACY 超参 json 旋钮源 (0913 #8 自适应再扫, rank_source 式三件套) ──
# 周末自适应再扫 (TOP10 实净判官 + PARAM_ADMISSION 护栏) 赢家落 WORM json
# (data/others/param_source/), dual_track_trainer.model_params 解析时叠覆盖 —
# 机器换 incumbent 免 commit. "auto" = 最新 json (trained_through 距今 ≤
# LEGACY_PARAM_MAX_STALE_DAYS); "code" = 强制代码表 (回退旋钮).
# 无 json / 过旧 / 异常 → 代码表 NUM_LEAVES_OVERRIDE/PARAMS_OVERRIDE (fail-open).
# 回退三层: 显式旋钮 > 删最新 json 回上一版 > 代码表兜底.
LEGACY_PARAM_SOURCE = {
    "main": "auto",
    "dual": "auto",
}
LEGACY_PARAM_MAX_STALE_DAYS = 45

# ── 横盘提示 (2026-08-19 用户定案 legacy+parallel; 0922 改中文口径 + genious 接入) ──
# 用户线索: 300911 连续入选短名单 (模型已识别) + 价格横盘洗盘 (10 日涨幅≈0) → 终将突破.
# 250d 检验 (_diag_stall_regime, 2026-08-19): 入选+滞涨+近20日入选≥3 全窗 63.2%/+5.88%,
# 但决定性变量是市场状态 — 强市日 80.5%/+12.35% vs 弱市日 23.5%/-8.94%; 低基线日
# (base_prod<中位 0.732) 82.9%/+13.28% vs 高基线日 -4.40%. 2025 vs 2026 差异 = 市场状态
# 分布差异 (2025 弱市日 64% vs 2026 强市日 64%), 非组合本身. → 交付层打标仅限低基线日.
# 0922 三档消融 (tmp_t/_0922_stall_ablation_replay.py): 频次条件 (近20日入选≥3) 配对差
# f10 仅 +0.28%/rnet −0.01% = 死重 → 已砍; 保留 日闸(冷静市)+横盘 两块真边际.
STALL_MARKER = {
    "ret_10d": 0.02,  # 近 10 日涨幅 < 2% = 横盘
    "window_days": 20,  # "近20日入选次数" 参考列统计窗口 (已不参与判定)
    "base_rate_max": 0.732,  # 当日 base_rate (dual 池 mfe_3d≥3% 达标率) < 此值 = 冷静市
    # 才标记 (250d replay base_prod 中位数, PIT 历史常数)
    "limit_ret_by_board": {  # 涨停判定: T-1 涨幅 ≥ 此值 = 涨停 (2026-08-19 第六轮
        "MAIN": 0.095,  # 追涨停 890d 均值 -0.82% 中位 -4.82% → 清单打标不追)
        "GEM": 0.195,
        "STAR": 0.195,
    },
}

# ── 明日开盘勿买标注 (2026-09-22 用户: "只要告诉我这股第二天开盘不要买") ──
# 三臂研究 (tmp_t/_0922_gkdzA/B/C_*, 校准 tmp_t/_0922_gkdzD_rule_calibration):
# trap = T+1 gap≥1% 且 walk≤−1%。规则 (并集, T 日收盘 PIT):
#   1) T 日涨幅 ≥ pct_rise (含涨停) — A臂梯度王: 大涨[7,9.8) 条件低走 50.4%;
#   2) amplitude_5d 当日截面分位 ≥ amp_rank 且 winner_ratio ≥ winner_min
#      (热股签名 × 深获利盘, B臂 top 特征簇)。
# 实证: 全期 trap 10.4% (2.9x 基率 3.55%), 2026H2 11.3% (3.2x); 清单域
# trap|flag 17~44% vs 未标 1~9%。语义 = 执行时点纪律: 勿开盘追, 等回落或尾盘确认。
# 删票闸 (0922 精确性回测 tmp_t/_0922_openbuy_btA/B/C/D 定案): 清单域 trap|flag
# 20.9% vs 未标 5.1% (4.1x), 标了开盘买 o2c −0.47% vs +0.38% → legacy/genious
# 采纳删票 (legacy o2c +0.24→+0.45%, genious +0.31→+1.01% 胜率 42→58%);
# parallel 删了更差 (被删 +0.30% vs 递补 −0.02%) → 只标注。过涨闸豁免
# (用户令 "过了涨闸就不要删了"; 结构性近乎互斥, 3.5y 仅 1~2 只)。清单不截断
# → 删票即天然递补。
OPEN_BUY_RISK = {
    "pct_rise": 7.0,  # T 日涨幅阈值 (pctChg, 百分数单位)
    "amp_rank": 0.8,  # amplitude_5d 当日全市场截面分位阈值
    "winner_min": 0.8,  # 获利盘水位下限 (深获利)
    "col": "明日开盘",
    "flag_text": "勿买·防高开低走",
    # 删票闸分线开关 (0922 A/B): True = flagged 且未过涨闸 → 从清单删除
    "kill": {"legacy": True, "parallel": False, "genious": True},
    "gate_col": "涨闸",  # 涨闸列名 (值 过闸=豁免删票); 无该列时用 _gate_exempt 复算
}

# ── 短名单迟滞滞留 (2026-08-26 用户定案 "清单加迟滞降换手") ──
# 昨日上榜股今日跌出 TOP-10 但仍在板内前 band_factor×10 名 → 滞留 (keep_flag="滞留",
# 排序沉底). 只降清单换手 (预测小幅回落即被换出 → 名单天天变), 不改新选股.
# TOP-10 新选不变 (2026-08-23 定案), 滞留是额外标注行.
# 2026-09-06 用户拍板 "去掉21-25" → 关闭: >20 行的唯一来源就是滞留追加, 关掉后
# 交付恒 ≤20 行 (每板块 TOP-10, 再过派发闸); 梯位复盘 15 晚行序 21-25 档全线弱于
# 前 20. 回退改 True 即恢复 08-26 降换手机制.
SHORTLIST_HYSTERESIS = {
    "enable": False,
    "band_factor": 2.0,  # 滞留带 = 板内排名 ≤ 10 × 此值
    "max_keep": 3,  # 每板块最多滞留数 (防爆清单)
}

# ── PARALLEL 短名单筹码水位标注 (2026-09-09 用户拍板 "派发不删, 清单标注") ──
# 沿革: 09-05 三线统一删派发 → 09-07 125d 回放证伪撤闸 → 09-09 改标注 (wr5<0 派发)
# → 0922 用户令改获利盘水位轴: chip_wr<0.5 → chip_flag="低获利，涨" / ≥0.5 →
# "高获利，跌" (66 张清单回测: 水平是唯一有信息的筹码轴, Δ族判死), 不删票;
# chip_wr5 数值列 0922 午后随用户令从交付清单退役。本开关控制标注接线
# (enable=False = 不标注)。
PARALLEL_CHIP_GATE = {
    "enable": True,
}

# ── PARALLEL 趋势闸 (2026-09-14 用户终令 "还是保持MA10 加CRASH8") ──
# 起因: 0913 夜清单 16 只中 15 只派发标注 (中位 chip_wr5 −29.5%), MA10↑ 仅
# 1/16 — 且唯一存活的 300808 是 09-11 单日 −18.3% 的见顶崩盘票 (用户点名)。
# 用户判词: "不是回调, 是见顶后跌了很久, 我不要这种票"; "起码基本MA10必须上升"。
# 二问 "MA5>MA10 会更好吗" 回放后 (数字见下) 用户三令终判: **保持 MA10↑+crash8**。
# 回放 (tmp_t/_trend_gate_parallel_sweep_0914.py, 21 夜 0806-0904, WORM
# diag/trend_gate_parallel_sweep_0914_*.json; 判官=riser5 用户5D/净5):
#   main:  无闸 60.0%/+1.49% | MA10↑&c8 56.7%/+1.10% | MA5>10&c8 57.4%/+1.47%
#          MA5>10&近5日涨&c8 57.1%/+1.21%, 净10 +3.48% 全变体最优 (无闸 +2.30%)
#   dual:  无闸 42.4%/−0.33% | MA10↑&c8 35.7%/−0.88% | MA5>10&近5日涨&c8 40.0%/−0.52%
#   当夜 16 只 FADING 票存活: MA10↑ 1 (300808, 被 crash8 杀→实际0) / MA5>10 2 /
#   MA5>10&近5日涨 0。MA5 族数值更优但把回调反弹全杀 (清单转向追涨风格) —
#   用户选择保留回调票风格, 判词记录备将来翻案 (风格令优先于均值)。
# 闸位 = rank_and_truncate 前 (先滤后截补位); 面板读不到 → 整体跳过
# (fail-open); 个股面板缺行 → NaN 比较判杀 (同 LEGACY)。
# [0914 逐股赢家统计后按板拍板] (tmp_t/_trend_gate_winner_stats_0914.py, WORM
# diag/trend_gate_winner_stats_0914_20260914_005517.json):
#   main: MA10↑+c8 10d 不亏 (+2.50%>无闸+2.30%, 大赢家 21>19) → 风格令原样保留;
#     5d 短视界亏赢家 (≥+5% 1.86→1.43/夜, 均值 +1.66→+1.25%) = 已接受风格代价。
#     数值最优为 crash8 单闸 (5d 上涨率 60.3% 持平无闸, ≥+10% 12→14 反增) — 回退档。
#   dual: MA10↑ 双视界全亏 (5d −0.74% 全档最差且大跌率 25.7%>无闸 21.9%;
#     10d 大赢家 27→18) → 撤 MA10 只留 crash 守卫 (off 档)。
#   MA5 族: main 10d 数值最优 (+3.71%, 大赢家 1.53/夜) 但杀回调反弹风格, 弃用留旋钮。
# [0914 续 — 用户终令 "不接受左侧"] 起因: 用户看当天清单后判 "parallel选的都是
# 已经大跌的股。。" / "我需要右侧交易，不能接受这种继续下跌的股" / "不接受左侧"。
# 取证 (tmp_t/_par_0914_leftside_check.py): 09-14 清单 main 段 r10 全正 (+0.3~+6.3%),
# dual 段 (rank 12-20) 深左 — r5 −3~−11%, r10 全负 −3.0~−19.8%, r60 全负, 距60日高
# −8.3~−44.6%。根因 = dual 档 mode="off" 只留 crash8, 无趋势闸 → 左票直通。
# 判决闸 = 益盟长线↑ (fail_trend = 入选时长线下降占比 = 用户主诉 FADING 票占比)。
# 回放 (tmp_t/_trend_gate_parallel_sweep_0914.py, 21 夜 0806-0904 全池, WORM
# diag/trend_gate_parallel_sweep_0914_20260915_0031*.json; 判官同前):
#            riser5   净5       riser10  净10      fail
#   main  无闸       60.0%  +1.49%   51.4%   +2.10%   16.2%
#         MA10↑&c8(旧) 56.7% +1.10%   51.3%   +2.21%   11.0%
#         长线↑&c8    58.1%  +1.57%   52.9%   +2.47%    0.0%   ← 新档 (五轴全面胜旧档)
#   dual  无闸       42.4%  −0.33%   34.3%   +0.11%   16.2%
#         crash8 单闸(旧) 41.0% −0.42% 34.3%  +0.27%   16.7%
#         长线↑&c8    39.0%  −0.65%   33.0%   +0.22%    0.0%
#         长线↑&中线↑&c8 40.2% −0.41% 36.0%   +0.52%    0.0%  ← 新档
#   dual 取 ym_ml: 5d 净与旧档持平 (−0.01pp), 10d 两轴胜 (riser +1.7pp / 净 +0.25pp),
#   而模型自身预测的就是 10d。main 取 ym_long (ym_ml 在 main 上 riser5 掉 9pp, 弃)。
#   闸后余量充裕 (全池中位 main 1505 / dual 582 只, 最少夜 10 / 4 只, 21 夜零空票夜)。
#   ★ 口径警示: fail_trend=0 是**构造性**的 (闸键即长线斜率), 非独立发现 — 真证据
#   是其余四轴不掉。ym 用面板**原始价** (非 hfq) = 回放口径。样本仅 21 夜 =
#   parallel_preds_raw 全部可得历史 (0806 起), 250d OOS 做不了。
# 回退: enable=False 全关; mode 全局字符串 = 两板同档一键整体切; 分板 dict
# 按板覆写。旧档 = {"main": "ma10_up", "dual": "off"}; ma5gt10_up5/ma5_gt_10
# 为已测旋钮, 勿再重扫。
PARALLEL_TREND_GATE = {
    "enable": True,
    # 按板分档: {"board": "ym_long"|"ym_ml"|"ma10_up"|"ma5gt10_up5"|"ma5_gt_10"|"off"};
    # 字符串 = 全局同档 (整体回退口)
    "mode": {"main": "ym_long", "dual": "ym_ml"},
    "crash_max_1d": -0.08,  # 评分日单日跌幅下限; None = 关守卫
    # ── 益盟长/中线 (ym 档的闸键) ──
    # raw34 = (close − 34日高)/(34日高 − 34日低), 值域 [−1,0]; 长线 = raw34 的 19日均
    # +100, 中线 = raw34 的 EWM(span=4) +100 (adjust=False)。
    "ym_span": 34,
    "ym_long_span": 19,
    "ym_mid_span": 4,
    # 34+19 需 ~52 交易日历史; 旧 45 天窗会截断 raw34 的极值窗 → 静默偏置
    "ym_lookback_days": 120,
    # ── 水平闸: 60日净涨下限 (0915 自我纠错补) ──
    # 为什么必须补: 用户主诉是 "已经大跌的股" = **水平**, 而上面的 ym 档闸键是
    # **斜率**。实盘 09-14 新清单 16 只里 13 只 r60<0, 而且**榜首就是深跌股**
    # (605588 −29.9%/605186 −26.8%/603262 −42.8%/688010 −44.2%/300321 −41.5%)
    # → 斜率闸**没解决主诉**。注: 上列是 **close_hfq** 口径; 原始价口径会夸大跌幅
    # (001268 hfq −12.3% 而原始价 −37.7%, 除权/分红污染) — 故水平闸必须用 hfq。
    # 机制 (逐日实证):
    # raw34 的 34 日高**滚出窗口**时 top 下降 → raw34 机械上升 (688010 收盘
    # 25.37→25.40 几乎不动, top 31.00→28.57, 长线 +0.0095 = 纯滚窗假升); 19 日均
    # 窗滚出极负值也能让长线升 (001268 当日跌 −0.9% 而长线 +0.0288)。
    # 回放 (tmp_t/_trend_gate_level_sweep_0915.py, 17 夜 0806-0904; 判官老四轴 +
    # 新水平轴 "picks 里 r60<0 占比 / r60 中位"; WORM diag/trend_gate_level_sweep_0915_*.json):
    #                 riser5   净5      riser10  净10    r60<0占比  r60中位
    #   main 无闸       58.8%  +1.18%   61.8%   +2.30%    86.5%   −11.1%
    #        ym_long&c8  58.2%  +1.14%   62.9%   +2.65%    84.1%   −11.1%  ← 改前 (斜率闸)
    #        ★ym+r60>−10%&c8 59.4% +1.79% 66.5%  +3.52%    67.1%    −2.3%  ← 改后
    #   dual 无闸       45.3%  +0.10%   41.8%   +0.42%    84.7%   −13.4%
    #        ym_ml&c8    42.6%  +0.15%   43.2%   +0.69%    88.2%   −15.1%  ← 改前 (水平更差!)
    #        ★ym_ml+r60>−10%&c8 48.4% +0.74% 44.4% +0.67%  69.7%    −3.5%  ← 改后
    #   ★ "完全没跌" 做不到: r60>0 硬闸把边打死 (dual 净5 −0.82%/净10 −1.92%,
    #   riser5 45.3→33.1%)。可行的是**下限**(跌幅 ≤10%), 不是清零。
    #   ★ 样本仅 17 夜 (需 60 交易日历史, 比斜率闸的 21 夜窗少 4 夜); 阈值在
    #   {0,−10,−15,−20} 里扫出 −10% 最优, 属 4 选 1 → 有过拟合风险, 但 −10% 同时
    #   是整数直觉档且非刀锋 (相邻档单调)。
    #   ★ 代价 (dual 尤重): 存活池中位 582→208 只, 且 17 夜里 **1 夜空票** (~6%) —
    #   与用户 "少过滤多出票" 偏好冲突, 但右侧优先是用户 x3 明令。main 只 1517→753。
    #   r60 用 close_hfq (真收益, 除权不污染); ym 仍用原始价 — 各有各的回放口径。
    "r60_max": -0.10,
}

# ── parallel 检查点备份保留数 (2026-09-18) ──
# _refresh_parallel_checkpoints.py 每次重建都把旧检查点改名为 .stale_<ts> 保留
# 可回溯, 但过去从不回收: 每个备份 main 2.9GB / dual 1.8GB, 每天一对 ~4.8GB,
# 8 天累积 38GB 直到 D: 归零 → 9/17 抓数写面板时 Errno 28 崩, 整条交付链哑火。
# 保留最近 N 份供短期回滚, 更早的自动删除。设 -1 表示不清理 (旧行为, 不推荐)。
STALE_CHECKPOINT_KEEP = 2

# ── 链路狙击交付层 (2026-09-14 用户令: 产 GENIOUS Excel, 交易日 20:30 自动跑) ──
# 纯规则层, 不吃任何模型产物 (只要面板 OHLCV + winner_ratio)。三触发器 T1洗盘日 /
# T2翻转日 / T3状态点火, 见 app/pipeline1/kongduo_triggers.py。
# 口径警示: 用面板**原始 close/high/low** (非 close_hfq) = 回放口径, 换 hfq 会让整张判词表
# 失效; 代价是除权日污染 r20/r60/r120。扣 0.7% 往返费后火群整体≈0, 钱只在分层头部
# (CH1 +2.30% / CH2 +6.38% / CH3 +10.10%)。逐日 cap 已判死 (68.6% 的火在洪峰日), 榜长不限。
GENIOUS = {
    "enable": True,
    "filename_prefix": "GENIOUS",
    "lookback_days": 400,  # r120 + SIG 34+6 rolling 需 ~250 交易日历史, 留余量
    # ── T1 洗盘日: 空图标 × r20>0 × 获利盘>wr_min × 60日筹码升>wr_rise_min × MA10↑ × 乖离≤x ──
    "wr_min": 0.6,
    "wr_rise_min": 0.05,
    "t1_ma10_dev_max": 1.05,  # close/MA10 上限; == 阈值仍触发 (<=)
    # ── T2 翻转日: 前 t2_kong_lookback 日出现过空图标 × 今日多图标上穿 × 涨幅>t2_gain_min ──
    "t2_gain_min": 0.02,
    "t2_kong_lookback": 10,
    # ── T3 状态点火: 多头态(非交叉) × 前5日回撤≥3% × 涨幅>t3_gain_min × 昨乖离≤x ──
    "t3_pb5_max": -0.03,  # 前5日回撤上限 (pb5 已含 shift, 因果)
    "t3_gain_min": 0.05,
    "t3_ma10_dev_max": 1.05,
    # ── 类型分桶 (r60): A深跌反转 <= r60_deep; C强牛深回调 r120>+40% 且 r60<=-5% ──
    "r60_deep": -0.30,
    "r120_base": 0.0,
    # ── 冠军段阈值 ──
    "r20_min": 0.05,  # CH1 T1长基: r20 > +5%
    "vr_quiet": 1.5,  # CH3 T3深跌缩量: vr <= 1.5
    "t2b_ext_prev_max": 0.95,  # CH2B T2稳健: 昨乖离 < 0.95
    # T1余 走 **T1宽** (剔 r120>+40% 与 r60<-5%): 不加这两个闸 8.4 → 10.4 票/日, 胜率掉 1.6pp
    "t1_wide_r120_max": 0.40,
    "t1_wide_r60_min": -0.05,
    # ── 观察带四臂 (底是 T2余/T3余, 不是裸全池 — 否则温火臂会吞掉任何 r120<=0 的当日上涨票) ──
    # 外闸分族 (续9 TierC 三剔毒): T2臂 = T2宽 三剔毒 (r60/vr/昨乖离); T3臂 = T3宽 只剔 r60。
    "band_vr_max": 2.0,  # 温火臂缩量闸
    "band_r60_max": 0.20,
    "band2_vr_outer_max": 3.0,
    "band2_ext10p_max": 1.05,
    "band3_vr_outer_max": None,  # T3臂不设 vr 外闸
    "band_t2_lo": 0.02,
    "band_t2_hi": 0.05,
    "band_t3_lo": 0.05,
    "band_t3_hi": 0.07,
    "limit_up": 0.09,  # 涨停臂: 涨幅 > 9%
    # ── Sheet2 排序 ──
    # 主键 = 观察分 (日内截面 z 的 -(band20+ext10+winner_ratio+r60+r120)): 全 896 日实测
    # IC +0.077 (t 8.7), 五分位单调, 前半/后半 0.079/0.076。层序**不作主键** (层内 IC -0.025)。
    "sheet2_top_n": 20,  # Sheet2 截断供阅读; 不截断全量留第三张表
    # S-L 标准化「正→负→正」形态 (0914 续18 全 897 日实测): 两个观察列 SL翻正年龄 / SL洗盘天数
    # 已写进 Sheet2, 可在 Excel 里自行按它排序。作排序主键**实测弱**: 翻正年龄 IC 0.013 (t 1.5)
    # 且不单调; 加进观察分只动 +0.001 (噪声)。翻正月占 27.6% — 零轴约 3.6 天穿一次, 是快振荡不是
    # regime 翻转; 且 80.6% 与 T2 图标翻转重叠 (点二列相关 0.274) = 既有触发的代理。
    # 洗盘时长**无可靠梯度**: 2-3天桶最好 (1.62%/52.5%), >=20天 58.9% 但仅 343 行且 >=30天 掉回
    # 46.9% (小样本假象)。翻正行整体 1.30%/51.4% vs 非翻正 0.83%/49.7% — 有溢价, 但当排序键无效。
    # 故默认仍用观察分; 要形态优先可把本键改成 "SL翻正"。
    "sheet2_sort_key": "观察分",  # "观察分" (默认, 实测最优) | "SL翻正"
    # ── 大涨三条件闸: **只标注, 不筛表** (2026-09-14 用户令) ──
    # 用户 0914 定调: "反正我就是要抓大涨" + "抓大涨是》=20%" + "不行。不做左侧交易";
    # 稍后改口: "或者你只在现在的 STOCKLIST 上加上 A+C+D 的标记" ⇒ 名单一只不删, 只多一列。
    # 闸 = 三条件同时成立, 缺一不可:
    #   A 低动量     r10 <= dir_gate_mom_max    ("我是想看筹码MA10上升 + 低动量")
    #   C 右侧拐头    r5 > 0                     (近端已翻上 — "不做左侧交易" 的落地)
    #   D 缩量       vr <= dir_gate_vr_max       (引擎口径 vr = 今日量/前5日均量, 已含 shift(1))
    # 目标 = fwd10 >= 20% 的命中率 lift。**引擎真实口径**实测 (基数 = 层内全样本):
    #   冠军四段 ≥20%  全1.07x | 末500 1.56x | ★末250 1.94x | 末125 1.98x
    #   冠军四段 ≥10%  全1.30x | 末500 1.87x | ★末250 2.05x | 末125 2.00x
    #   火群     ≥20%  全1.08x | 末500 1.08x | ★末250 1.70x | 末125 1.69x
    #   火群     ≥10%  全1.27x | 末500 1.39x | ★末250 1.96x | 末125 1.90x
    #   末250 绝对值: 冠军四段 +9.00% 胜81.3% ≥10% 53.0% ≥20% 12.7% (1.3行/日, 基准 19.2);
    #                 火群    +3.88% 胜60.4% ≥10% 28.6% ≥20%  6.6% (3.3行/日, 基准 70.1)。
    #   对照: 四条件闸 (含 B) 冠军末250 只有 0.5行/日 — 摘掉 B 后票数 2.6 倍, lift 还更高。
    # ★ 16 组合全枚举判词见 tmp_t/_bigrise_4cond_combos_0914.out。要点:
    #   C 是那把钥匙 (冠军末250 C单条件 ≥20% 已 1.40x), D 次之 (1.22x);
    #   **A 与 C/D 是互补的制度对冲** — A 的边在老数据 (火群全样本 ≥20% 1.33x → 末250 只剩
    #   1.04x), C/D 的边在近端 (火群全样本 0.86x → 末125 1.14x), 所以两条腿都留, 逐窗最稳。
    # ★ 不接的项 (均实测过):
    #   B 筹码MA10上行 (控盘A08的MA10十日斜率>0, 原四条件闸第2条) — **0914 用户拍板摘掉**。
    #     它是四条件里最弱的: 冠军末250 单条件 ≥20% 仅 0.95x; 往 C+D 上加它 2.09x→1.75x,
    #     往 A+C+D 上加它 1.94x→1.25x, 每加一次都掉; 观察池/火群上是中性 (1.0x 上下)。
    #     旧 A+B 合起来 0.90x, 比空集还低。降级为**展示列** (表内仍有「控盘MA10斜率」)。
    #   G 集中度变化 (Δ10W>0) 能把 ≥20% 推到 1.53~1.93x, 但**两只种子案例全被筛掉** ⇒ 不接,
    #     集中度Δ10 只作展示列。方向本身是**反直觉的强**: 变发散 1.19~1.48x vs 变集中 0.62x。
    #   E/F 获利盘 >= 20 / >= 30 (用户预设 "主力筹码20，30") 实测无独立增益, 不并进闸; 获利盘列
    #     仍留在表内可自查。
    # ★ 两个种子案例 000978 / 000823 本闸全抓到 (0824/0826 与 0826), 与四条件闸同。
    # ★ **不过滤任何一行** (0914 用户令): 三张表全量 + 一张「大涨闸」列 (过闸/被拦)。
    #   理由: 冠军四段 19.2票/日 里只有 1.3票/日 过闸 (93% 被拦) — 一筛冠军表就剩 1 行,
    #   观察池 Top20 更会被掏空 (最近20日冠军四段过闸合计18只, 中位0)。它当**窄名单标记**用。
    #   注: B 在观察池上**反而是加分** (末250 ≥10% 1.51x→1.93x, ≥20% 1.20x→1.45x), 与冠军段
    #   相反 —— 但两处都不足以当筛子; 三张表的「大涨闸」列一律用同一 ACD 口径, 保证列义一致。
    # 回退: dir_gate=False (全部标「过闸」, 列仍在)。
    "dir_gate": True,
    # 0918 终版: 过闸 = 原现役闸(r10≤0) ∪ 缺口档(r10∈(0,mom_max] 且 r5≤0.08) — 三档
    # 胜率>70%; 温和档 (r5>0.08) 与高动量都不算过闸。mom_max = 缺口档 r10 上限。
    "dir_gate_mom_max": 0.05,
    "dir_gate_vr_max": 1.0,  # D 缩量: 量比上限
    # ── CH4 中线缩量横盘 (2026-09-19 用户令: "不想要深跌股, 想要中线在涨+短线加速+缩量回调") ──
    # 规则 = r60>ch4_r60_min (强中线) & ma10up (短线加速) & vr<=ch4_vr_max (极缩量)
    # & dd5<ch4_dd5_max (微回调横盘: 收盘距5日高点<1%, 即没实际跌出去)。dd5 列在
    # compute_features 里新增。
    # ★消融 (tmp_t/_ch4_ablation_0919, f5 跨月): 基格 n=636/+6.66%/50.9%。
    #   -r120 → n=1055/+6.76%/51.0%: **不承重已删除** (强中线+缩量+横盘的保护下,
    #   长线反不反无信息; 删了多 400 票、期望持平略升)。残余次要: -r60 -2.3pp。
    #   硬墙 = -vr → n=7207/+2.19%/39.4% 与 -dd5 → n=7323/+1.65%/39.3% (面膨胀10x
    #   进来的是"回调未到位"的高分化群, 期望胜率双掉) — vr+dd5 就是"回调到位"状态本身。
    # 邻域无悬崖 (subagent 36 组合 +2.3%~+7.9%); 分半一致; 主板主导且剔
    # top10 票仍 +4.9%。"加速后回调" 变体族全与基格重叠 ≥84% — 无独立新格。
    # ★0919 再收 vr: 0.7→0.5 — vr 轴单调 (0.7→+6.8%/51.0%, 0.5→+9.3%/57.6%, t30分位
    # 0.32→+11.8%/62.1%); 逐月 9/12 月期望+4~+18%, 半分一致 (57.4/57.9)。
    # 202607 单月 19% 胜率 = 月末大盘跳水黑天鹅 (21票16止损), 非格子结构病。
    # 与现役涨闸几乎互斥 (∩全年仅14票) — 闸照常标注记账。
    # 回退: ch4_r60_min 调极大值 → 永不命中, 零行为。
    "ch4_r60_min": 0.25,
    "ch4_vr_max": 0.5,
    "ch4_dd5_max": 0.01,
    # ── 同花顺自选股推送 (2026-09-14 用户令: "出了 GENIOUS 列表 AUTO PUSH TO THS") ──
    # 出 Excel 后落一张 genious_stocklist_{date}__{HHMMSS}.csv 边车 (WORM), 再交给
    # scripts/_ths_watchlist_push.py 当第三个源推。推 Sheet1 冠军四段前 push_top_n 名
    # (与 parallel/legacy 的 TOP10 口径一致); Sheet2 观察池**不推** — 那是观察名单不是买入名单。
    "push_to_ths": True,
    "push_top_n": 10,
}

# ── 量价删查线 (2026-09-08 用户拍板 "GO WITH 1") ──
# 交付清单内删 amt_agree10 (10日量价配合度) 最高档, 真删不补齐。回放证据
# (tmp_t/_envrecheck_vpkill_0908.py, 池=2026-08-05..09-07 两线交付 13日396票):
# 删 top20% 净 +0.62pp/10日, 被删组 FWD −2.13%, 零误杀大赢 (FWD≥15%),
# 留存 P90 10.6%→11.9%; legacy +0.87pp / parallel +0.39pp。
# 样本警示: 12/13 日在 trend20>0 牛市区, 弱市缺读数 (唯一弱市日净 −0.74pp)。
# 因果: 只用 ≤清单日的 close_hfq/amount。接线 legacy+parallel 两线交付前;
# 密度线无回放证据不接。回退 enable=False。
AMT_AGREE_GATE = {
    "enable": True,
    "del_top": 0.2,  # 清单内按因子降序删最高档比例 (k=max(1,ceil(n可算×比例)))
    "window": 10,  # 量价配合度滚动窗 (交易日)
}

# ── 冲高回落闸 (2026-09-09 用户: "TOP10里如果有冲高回落票需要给提示或者删除") ──
# 回放 (_fade_top10_replay.py, 21 清单日 280 票): fade_score = vol20/turn20/r20/
# pos250 全市场日截面 pct-rank 复合 (≤清单日收盘可知)。≥0.75 的票 (8.9%): 执行日
# open→close −1.03% vs 留守 +0.30~+0.56%, 上涨率 32% vs 55%, 5日 −1.40% vs +0.45%
# [09-09 当日撤删线] 复核 (tmp_t/_fade_misfire_0909.py): ① 半窗翻转 — 删除区 5日
# 优势全集中在前半 (08-06..19: −2.62% vs 留守 −0.42%), 后半 (08-24..09-07: +1.66%
# vs +1.35%) 边际消失; ② 大赢家误杀 — ret5≥10% 的票 18.2% 落删除区; ③ 09-09 实盘:
# 删除区 3 只中 603186 +10% 涨停 / 601869 +6.0% (用户: "删掉的好票很多, 2个强势股")。
# → kill_enable=False 只留 fade_flag 提示列; ≥40 清单日复跑 replay, 若前后半窗
# 一致再议重开。总闸回退 enable=False。
FADE_GATE = {
    "enable": True,
    "kill_enable": False,  # 体质分删线 09-09 撤 (半窗翻转+误杀强势股); 重开须双半窗一致证据
    "kill_score": 0.75,  # 状态型体质分删除线 (全市场 pct-rank 复合)
    "flag_enable": True,  # 事件型 (昨日冲高回落) 提示列开关
    # 09-09 用户澄清 "要预测是否会冲高回落, 非记录昨日事件": fade_score 本就是
    # t-1 预测体质分, 加 fade_risk 人读档位列 (高档日内冲高回落概率≈1/3 vs 基线~23%)
    "risk_hi": 0.75,
    "risk_mid": 0.5,
}

# ── 闸准入协议 (2026-09-09 用户: "先判断是否应该用闸, 闸是否有正效果再行动,
#    这个应该写进PIPELINE并验证") ──
# 起因: FADE_GATE 删线凭 21 清单日回放均值直接接线, 当日复核发现半窗翻转
# (前半 +2.2pp / 后半 −0.3pp) + 大赢家误杀富集 (ret5≥10% 落删除区 18.2% vs
# 基础删除率 8.9% ≈ 2.0x) + 实盘误杀涨停股 → 同日撤线 (528c50bd)。
# 教训: 均值级回放证据不够。此后任何删票/拦截闸:
#   ① 接线前必须 scripts/_gate_admission.py --gate <name> 判 PASS;
#   ② 已上线闸每夜由 run_daily_automation gate_audit 步日审 (2026-09-09 用户指令
#      "FADE线能DAILY来重审嘛", 原 Sunday 周审保留作冗余; 告警式, 不自动改闸)。
# 判据 (交付清单回放, 删除区 vs 留存, ret5 = T+1收→T+5收):
#   - 样本充分: 清单日 ≥ min_days 且删除区成熟票 ≥ min_killed, 否则 INSUFFICIENT
#   - 全窗效果: 留存−删除 ret5 差 ≥ min_edge_pp
#   - 半窗稳定: 按清单日对半, 前后半各自差 > half_tol_pp
#   - 赢家误杀: P(删除|ret5≥winner_ret5) / P(删除) ≤ max_leak_enrich
#     (比例口径与删除带宽无关, 1=随机基线)
GATE_ADMISSION = {
    "min_days": 40,  # ≥40 清单日 (既有复验标准, fade 21 日接线即违此线)
    "min_killed": 20,  # 删除区成熟票下限 (ret5 可结算)
    "min_edge_pp": 0.5,  # 全窗留存−删除差下限 (百分点)
    "half_tol_pp": 0.0,  # 双半窗各自差须严格 > 此值 (百分点)
    "winner_ret5": 0.10,  # 大赢家定义 (ret5 ≥ 10%)
    "max_leak_enrich": 1.5,  # 大赢家落删除区比例/总删除率 上限
}

# ── 10d 回归头秩目标变换 (2026-09-10, 688228 案) ──
# 起因: 四连阴缩量下跌股 688228 被 10d 头预测 +5.27%/10d 入 TOP8。审计
# (tmp_t/_pred10d_downtrend_audit_0910.py, 13 成熟日 26430 行, T+1→T+10 匹配视界):
# 纯幅度 Huber 目标下 pred10>+6% 桶承诺 +11.2~11.5% 实得 −0.2~+1.5% (校准缺口
# −9.7~−11.8pp); 排名键按幅度取头部恰好取到最虚高的承诺 (rank≤10 实得≈0 vs
# rank 11-30 下跌股 +2.32%)。用户指令: 修预测精度, 不加闸。
# 机制: reg 头改训 per-date 截面百分位 (0..1, rank pct of net label), 预测输出
# 经桶中位映射 (训练段按预测分桶取 label 中位数, np.maximum.accumulate 单调化)
# 回收益语义 — 下游 E7/prob 残差/排名键语义不变, 顶部承诺被结构性压回可兑现水平,
# 树的优化目标从"追彩票尾"变为"排序明日截面"。
# [0912 泛化] 从 10d 单头扩到全部 reg 幅度头: dual 3d/5d/10d_reg OOS IC −0.11~−0.14
# 三头全负 (cls 同窗全正, top10 实净 +55%) — huber 在双创/科创重尾标签
# (|lab|>0.20@10d: STAR 11.26% vs main 4.49%, 峰度 303) 上把分裂容量耗在月度翻转
# 的尾部方差; rank 目标 = 评测口径本身 (评估即 rank IC)。main 板 reg 头同修同判
# (用户指令 0912)。boards/kinds 分格门控, 默认全关 = 零行为变化; A/B 拍板后开启。
LEGACY_REG_RANK_TARGET = {
    "enable": False,  # 生产默认关; 影子重训 A/B 拍板后按板开启
    "bins": 20,  # 校准映射分桶数 (预测百分位 → 桶内 label 中位数)
    "boards": ["main", "dual"],
    "kinds": ["3d_reg", "5d_reg", "10d_reg"],
}

# ── [2026-09-12] 尾部样本加权: 爆发猎杀 (002848 池内 347 名案) ──
# L2 拟合 fwd10 条件均值 → 全市场 90% 小波动样本主导 loss → 模型"说小声话",
# 将爆发的票预测幅度系统性压低 (见 +3.8% 实际 +10%)。修法: 训练时给实际 fwd10
# 落在 top_q 分位的样本 ×weight — 尾部错判代价放大, 目标定义/排名键/闸全不动。
# 只作用 reg 幅度头 (3d/5d/10d_reg, 与时间衰减权重相乘); cls/pain 头不动。
# [0912 夜 A/B 判 FAIL 已关] main 板 60d OOS tail 臂隔离: reg_IC 0.0808→0.0284,
# top10 −1.16pp, 爆发重叠 3.41→2.96。⚠[0912 更正] 交付清单排序键 prob_up_10d =
# reg 残差概率 (predictor 08-24 起 p_reg 优先, Platt cls 仅补 NaN 行) → 清单本就
# 由 reg 头驱动, tail 臂伤 reg = 直接伤交付头, FAIL 更实锤 (原"够不着"推断作废)。
# 第三次修幅度头失败 (前两次: 0910 rank 目标/decay5)。
# 复测 = enable=True (数学保留, 见 tests/test_legacy_tail_weight.py)。
LEGACY_TAIL_WEIGHT = {
    "enable": False,
    "top_q": 0.9,  # 训练段 label 分位阈值 (top 10% 实际涨幅样本)
    "weight": 4.0,  # 尾部样本权重倍数 (与 decay_sample_weights 相乘)
    "kinds": ("3d_reg", "5d_reg", "10d_reg"),
}

# ── [2026-09-12] per-head 特征集: reg/cls 头分开训练, 各头族挂独有特征 ──
# 8 格矩阵 A/B 判词分格执行 (WORM cls_top10_0912_{main,dual} + ab_families_0912_{main,dual}):
#   main reg += SL斜率20   : reg IC +2.27pp/top10 +0.30pp/lift 4.08→8.31; cls top10 −1.46pp → 只入 reg
#   main cls += quality_factor: cls IC +0.75pp/top10 +0.19pp 双正; reg top10 −3.52pp → 只入 cls
#   dual cls += quality_factor + ps_ttm: top10 +2.88/+1.46pp (base 0.54% 低基数已注)
# dual 不挂 SL斜率20 per-head — 已在 dual force_include 共享集内 (feature_selector)。
# 训练列清单铁律: 帧内缺失的 extra 由 kind_feature_cols 剔除+告警 (模型列数=bundle 声明),
# quality_factor 由训练/推理端同公式合成 (feature_selector.add_quality_factor)。
LEGACY_HEAD_EXTRA_COLS = {
    "main": {
        "reg": ["SL斜率20"],
        "cls": ["quality_factor"],
    },
    "dual": {
        "cls": ["quality_factor", "ps_ttm"],
    },
}

# ── [2026-09-12] PARALLEL per-head 特征集 (8格矩阵, 结构先行; 内容随 #11 A/B 判词填) ──
# 板×头: mag = 幅度头 pool 追加列 (app.pipeline_parallel.config.effective_pool,
#   作用于交付打分链: backtest.write_system_lists / build_merged_shortlist +
#   交付端 _panel_per_stock / _anchor_frame; 诊断类 run_system/分档/last_days
#   仍用 spec 原池). prob = 概率头 LGBM feat_cols 追加列 — 概率头是宽集
#   (feature_cols 全数值列自动收录), 故 prob extras 只对**面板外合成列**有意义
#   (现支持 quality_factor, 公式=feature_selector.add_quality_factor,
#   train/serve 同源合成). 默认全空 = 当前产线行为零变化.
# 列缺失: mag 由 pool_score 跳过并再归一化; prob 合成失败保持缺失 →
#   predict 缺列 raise (schema 漂移大声失败, fail-loud).
PARALLEL_HEAD_EXTRA_COLS = {
    "main": {"mag": [], "prob": []},
    "dual": {"mag": [], "prob": []},
}

# ── parallel 概率展示层再校准 (2026-08-29 用户批准) ──
# 08-29 实测交付概率高估 (pred_prob_10d 均值 55.8% vs MFE>6% 实得 27.5%, +28pp;
# tmp_t/_rebase_diag_0829.py). 展示层每板块每视界乘一个收敛因子 = 实得命中率/预测
# 均值, 按成熟日数收缩 (N<min_matured 时按比例靠近 1) — 板内常数乘法, 不改排序
# (排名键/闸/EMA 均在再校准之前, raw 历史 WORM 文件保持原值).
PARALLEL_PROB_RECAL = {
    "enable": True,
    "min_matured": 20,  # 成熟日达到该值后因子全额生效
    "window_days": 42,  # 只用近 window_days 个自然日的历史清单
    "factor_bounds": [0.2, 1.5],  # 因子安全夹
}

# ── 跨模块影子排名 (2026-08-26 用户批准, 纯记录零交付风险) ──
# legacy 与 parallel 交付清单按板块并池, 各模块在自己板内名次百分位归一后加权混排,
# 影子 TOP-N 只落盘不交付. 交集≈0 时 blend≈0.5×自有分 → 实质=两榜按各自强度交错.
# 数据源 (全部 WORM 已有文件): data/lists/list_<D>*.parquet (legacy, 键=prob_up) +
# STOCK LIST parallel_shortlist_<D>__*.csv (键=rank_blend, 多版本 keep-last).
XMODULE_SHADOW = {
    # 2026-09-02 积累样本终审判死 (21 日影子, T+3 c2c, _shadow_xmodule_eval_*.json):
    # main shadow 输 parallel -0.18pp/日 (win 0.12), dual 输 -1.14pp/日 (win 0.38);
    # 两模块交集≈0 (dual=0.00/日) → 混排=对半分席稀释强臂, 非集成. 勿再翻回.
    "enable": False,
    "weights": {"legacy": 0.5, "parallel": 0.5},
    "top_n": 10,
    "out_root": "shadow",  # 相对 DATA_OTHERS_DIR, 影子清单不进 STOCK LIST 交付目录
}

# ── SLOW BULL 长持影子单 (2026-09-07 用户命名+拍板; 前身=动量带×grind 长持格,
#    功能位承接旧 app/pipeline_parallel SLOW_BULL_* 暂停模块, 那套代码维持暂停) ──
# 125d 回放过闸: 全闸 tr40 +3.49% 双半窗正; 加宽度>MA60自适应闸后 +8.21% 四半窗超基线.
# 铁律: 闸基准必须自适应 (滚动统计量), 禁止固定绝对阈值 (水平版≥0.45 已证是坑).
SLOW_BULL = {
    "enable": True,
    "band_lo": 0.90,  # 20日动量截面分位下界
    "band_hi": 0.995,  # 上界 (排彩票极端动量尾)
    "grind_max1d": 0.08,  # 近20日最大单日涨幅上限 (grind 定义)
    "pull_min": -1.0,  # [0913 用户令撤回撤闸] -1.0 等效关闭; 原档 -0.10, 恢复改回
    "breadth_ma": 60,  # 宽度闸: breadth > 其60日滚动均值
    "mom_lo": 0.60,  # v4 格内选股: mom__r 格内分位 (60%, 90%] 区 — 剂量曲线确定性中段峰
    "mom_hi": 0.90,  #   (格内90-100%过热尾=反信号: 赢率46%/中位-0.8 纯右尾驱动)
    "vol_half": "low",  # v4 极致确定性拍板 (09-07"我要极致确定性"+"低波半"): 只取格内低波半
    #   (mom(60,90]×低波半: 5.3只/日 赢59% 中位+4.1% 大亏7% 125+9.5%;
    #    高波半=只有均值没确定性: 赢率46-48%/中位负/大亏12-15%)
    "trail_stop": 0.08,  # 退出: 8% 跟踪止损
    "hold_days": 40,  # 持有上限 (交易日)
    "out_root": "shadow",  # 相对 DATA_OTHERS_DIR, 影子清单不进 STOCK LIST 交付目录
}

# ── legacy 幅度/校准漂移监控 (2026-08-17 幅度, 2026-08-24 加 ECE 校准) ──
# 08-17 诊断: pred_ret_10d 系统高估 (main 均值 +4.03% vs 实现 +1.10%, dual +6.59% vs
# +1.32%), 偏差随时间扩大 → 生产 "pred>0" 闸 100% 空转. 修漂移 = 重训 (周频已做),
# 监控 = 每日全池 pred 均值 vs T+10 净实现均值偏差 (scripts/_monitor_legacy_drift.py),
# 滚动窗均值超阈值 → WORM 报告 + 日志告警 (提醒提前重训).
# 08-24 加 p_reg 校准检查: prob_up_10d = P(gross_10d > 0.5%) (label_engine
# CLS_THRESHOLD, GROSS 研究口径). ECE 用全池分位桶 (prob 集中 [0.25,0.55],
# 等宽桶失真), 事件 = gross_cc > 0.5% ⟺ realized_net > 0.5% - cost (防 cost 偏差
# 假触发; 与回放参照同口径). 滚动窗 ECE 超阈值 → 提示重训. 阈值见下方 calibration,
# 勿当定案.
DRIFT_MONITOR = {
    "window_days": 42,  # 滚动偏差窗 (交易日)
    "min_matured_days": 20,  # 成熟日少于该值 → 不出告警 (积累期)
    "bias_threshold": {"main": 0.04, "dual": 0.07},  # 初始值, 勿当定案
    "cost": 0.0020,  # 往返成本 (与诊断回放一致)
    "buy_lag": 1,  # 决策日后第 1 个交易日收盘买入
    "sell_lag": 11,  # = buy_lag + label_horizon(10)
    "calibration": {
        "cls_threshold": 0.005,  # label_engine CLS_THRESHOLD (gross +0.5%)
        "n_bins": 5,  # 校准分位桶数 (保证每桶有样本)
        # 初始阈值锚定当前 bundle 诊断基线 (legacy_prob_head_replay_20260816 回放
        # picks 子集 ECE: main 17.4% / dual 23.0%, 含 open基 label vs close基
        # realized 隔夜缺口常数偏移) + 小余量 → 只对真实新漂移告警. 成熟日积累后复核.
        "ece_threshold": {"main": 0.20, "dual": 0.25},
    },
    # 09-03 赢家泄漏复盘新增: 排名键 (pred_ret_10d) 赢家判别 AUC 月度监控.
    # winner_leak_20260903: 6 月断裂 0.66→0.52 抛硬币且未恢复, 此节保证同型断裂
    # 2 个月内自动发现. 全池口径 (candidates 无 base_rate, 无法复现 E7 池),
    # 绝对值与复盘回放不可比, 只看趋势. 连续 N 个成熟月 < 阈值 → 报警.
    "winner_auc": {
        "win_t": 0.05,  # 赢家 = T+10 净 ≥ 5% (复盘主口径)
        "threshold": 0.55,  # 报警线 (复盘: 好行情 0.60-0.67, 断裂后 ~0.52)
        "consecutive_months": 2,  # 连续 N 个成熟月低于阈值才报警
        "min_days": 8,  # 月成熟 AUC 日数下限 (不足 = 该月不参与判定)
    },
}

# ── parallel dual 幅度漂移监控 (2026-08-18) ──
# 与 legacy 同构: 每日短名单 pick 的 mfe_10d (校准预期幅度) vs label_pm_10d_net 实现
# 偏差, 滚动窗超阈值 → 告警 (特征冻结后更依赖此监控, 季度重选前唯一漂移信号).
# 数据源: BACKTEST_RESULT_DIR/*/last_*_days_picks_dual.csv (WORM, as-predicted 去重)
# realized: 刷新后的 dual 检查点 label_pm_10d_net (与校准目标同口径, 无 lag/cost 歧义).
PARALLEL_DRIFT_MONITOR = {
    "window_days": 42,  # 滚动偏差窗 (交易日)
    "min_matured_days": 20,  # 成熟日少于该值 → 不出告警 (积累期)
    "bias_threshold": {"dual": 0.07},  # 初始值 (沿用 legacy dual), 勿当定案
    "run_root": "BACKTESTING RESULT",  # 相对 DATA_OTHERS_DIR 的 run_dir 根
    "checkpoint_dual": "_diag_stage_dual_3y.parquet",  # 相对 DATA_DIR, refresh 步骤更新
    # 08-26 ECE 校准节: pred_prob_10d 事件 = net MFE(盘中) > mfe_threshold
    "calibration": {
        "enable": True,
        "mfe_threshold": 0.06,  # = _shortlist_t5_t10.ABS_TARGET["10d"]
        "n_bins": 5,
        "cost": 0.0030,  # COST 0.0013 + 2×滑点中档 (adv20 分层取代表值)
        "horizon": 10,  # 成熟需 T+1+10 交易日
        "ece_threshold": {"main": 0.20, "dual": 0.25},  # 沿用 legacy 初始值
    },
}

# ── 重训内存独占闸 (2026-08-15 用户定案, 代码强制"重训期间不跑其他重活") ──
# 08-14 教训: 重训 + 250d 复验/扫描并发 → RAM 挤兑 → 训练 8.4h 零模型完成.
# ram_guard.check_startup_gate: 启动时可用物理内存低于下限 → 拒绝启动 (exit 2);
# ram_guard.start_monitor: 运行期每 poll_s 采样, 低于下限 → 每段挤兑一条 WARNING.
RETRAIN_RAM_GUARD_MIN_FREE_GB = 2.0  # 启动闸下限 (可用物理内存, GB)
RETRAIN_RAM_GUARD_POLL_S = 30  # 运行期警报采样间隔 (秒)
# 进度静默判死阈值: 非 ram_guard 日志静默超此值 → stall_watchdog 硬退 rc=86,
# 交外层队列/手工冷却重试 (08-14 / 09-10 a1 两起换页泥潭 6h+ 零进度人工击杀).
# 45min = 1.8x 健康跑最大静默 25.1min (2026-09-10 clean run 校准).
RETRAIN_STALL_KILL_MIN = 45
