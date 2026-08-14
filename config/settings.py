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
PARQUET_DIR = PROJECT_ROOT.parent / "PARQUET"   # 主数据目录 (V3 面板所在)
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

for _d in (RAW_DIR, INTRADAY_DIR, PROCESSED_DIR, MODEL_DIR, DATA_OTHERS_DIR,
           BACKTEST_RESULT_DIR, STOCK_LIST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 预测稳定性: 输出级时间平滑 (2026-08-06, 对齐 parallel _shortlist_t5_t10) ──
# 同一只股票相邻交易日预测/概率剧变 → 每股 forecast 列 = 近 K 个可用交易日 raw 预测的
# 衰减加权均值 (w_k = α·(1-α)^k, 归一化, gap-robust). α 越大越信任今日. 历史底稿 WORM
# 落盘 legacy_preds_raw_<date>__<module>.csv (模块标签见 module-tag 约定).
LEGACY_SMOOTH_ENABLED = True
LEGACY_SMOOTH_ALPHA = 0.35
LEGACY_SMOOTH_K = 12

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
    AUTO = "auto"      # granted: orders sent to broker directly
    MANUAL = "manual"  # pop-up recommendation only, user confirms


# ── Execution ─────────────────────────────────────────────────
EXECUTION_MODE = ExecutionMode(os.getenv("AMINQT_EXEC_MODE", "manual"))
EXECUTION_BROKER = os.getenv("AMINQT_BROKER", "sim")  # "sim" | "xt"

# ── Risk filter hard constraints (Phase 4) ────────────────────
MIN_AMOUNT = 50_000_000         # 成交额 >= 5000万
PRICE_LIMIT_PCT = 9.5           # |涨跌幅| <= 9.5%
MAX_ACCOUNT_DRAWDOWN_PCT = 3.0  # 账户回撤 > 3% → 返回空列表

# ── V3 入库扫描 (ingest gate, 2026-08-03) ─────────────────────
# _daily_fetch.py 追加当日行前扫描: ST/*ST 股 或 上市 < INGEST_MIN_LIST_DAYS 个交易日
# 不进入 V3 面板 (universe 在入口收敛). 交易日数按面板唯一 date 列 (交易日历)
# 计算 stock_basic.list_date → trade_date 的交易日计数 (searchsorted 向量化).
INGEST_MIN_LIST_DAYS = 150

# ── KIMI LHB v2.0 spec 参数 (龙虎榜稀疏特征: 半衰期/情境权重/记忆下限) ──
# 见 REFERENCE/.../FEATURE/kimi LHB_v2.0_设计文档.md §3.1/§3.3/§4
LHB_V2_SPEC = {
    "h_inst": 8,          # 机构半衰期 (spec 建议 7-10)
    "h_top": 6,           # 顶级游资半衰期 (5-7)
    "h_quant": 4,         # 量化席位半衰期 (3-5)
    "h_retail": 4,        # 散户/混合半衰期 (3-5)
    "h_sell": 5,          # 抛压记忆半衰期
    "h_sellbuy": 5,       # 买卖比半衰期
    "h_conboard": 3,      # 连板衰减记忆半衰期 (§4.3)
    "w_limit_up": 1.5,    # 涨停日卖出情境权重 (§2.5)
    "w_limit_down": 1.2,  # 跌停日卖出情境权重
    "w_up5": 1.3,         # 大涨日(>5%)卖出权重
    "w_down5": 1.1,       # 大跌日(<-5%)卖出权重
    "w_flat": 1.0,        # 平盘日卖出权重
    "f_min_ratio": 0.1,   # 最小记忆值 = max(0, 历史均值×比例) (§3.3)
    "lock_thresh": 0.3,   # 机构锁仓信号阈值 F_inst > 0.3 (§4.5)
    "overheat_penalty": 0.7,  # 过热惩罚因子 (C5d≥阈值时正向资金流×0.7) (§5.2)
    "limit_up_tol": 0.001,   # 判定涨停: close >= up_limit_raw×(1−tol)
    "limit_down_tol": 0.001, # 判定跌停: close <= down_limit_raw×(1+tol)
    "eps": 1e-6,          # 除零保护
    "circ_mv_unit": 1e4,  # 面板 circ_mv 单位 万元 → 元
}

# ── LHB v2.0 训练/评估配置 (spec §5.3 选择性偏差: 仅上榜股票池) ──
# 见 REFERENCE/.../FEATURE/kimi LHB_v2.0_设计文档.md §5.3/§6.1
LHB_V2_EVAL = {
    "horizons": [1, 3, 5],   # 标签: t+1 开盘买入 → t+1/t+3/t+5 收盘 (T+1 模拟)
    "split_ratio": 0.8,      # 时间切分: 前 80% 上榜日训练, 后 20% 评估
    "quantile": 0.2,         # 多空分位 (预测 top/bottom 20%)
    "min_ic_obs": 20,        # 单特征 IC 至少需要的日期数
    "min_ic_n": 10,          # 单日 spearman 至少需要的股票数
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
    "horizons": [1, 3, 5],   # 标签: t+1 开盘买入 → t+1/t+3/t+5 收盘 (T+1 模拟, hfq)
    "split_ratio": 0.8,      # 时间切分: 前 80% 事件日训练, 后 20% 评估 (仅事件池内)
    "quantile": 0.2,         # 多空分位 (预测 top/bottom 20%)
    "min_ic_obs": 20,        # 单特征 IC 至少需要的日期数
    "min_ic_n": 10,          # 单日 spearman 至少需要的股票数
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
    "top_n": 10,                  # 每期入选股票数
    "cost_commission": 0.00025,   # 佣金 万2.5 (单边)
    "cost_stamp": 0.0005,         # 印花税 0.05% (仅卖出)
    "cost_slippage": 0.0010,      # 滑点 0.10% (单边)
    "exclude_st": True,           # 剔除 ST (基座 is_st 待修, 当前可能不触发)
}

# ── 每日短名单评分权重 (2026-08-05 用户: 先给每股预测(涨幅+达到概率), 再按权重打分, 再排名) ──
# 每股综合分 score_w = Σ_h horizon_w[h] × ( gain_w × 涨幅min-max归一化 + prob_w × 达到概率min-max归一化 )
# 涨幅与达到概率都按当日全部入选股 min-max 归一化到 0~1 (同量纲, 真 50:50; 概率用原始值≈常数会使排名被涨幅主导).
# 排名 = 综合分降序. 主视界 = T+3 (2026-08-05 用户定案: 短持 3 天, 3d 权重最高 0.40).
# 达到概率 = P(已实现 MFE ≥ 该预期涨幅) — 真实口径 (2026-08-05 用户定案, 见 _shortlist_t5_t10.py).
SHORTLIST_SCORE = {
    "horizon_w": {"3d": 0.40, "5d": 0.25, "10d": 0.10},  # 主视界 T+3 (短持)
    "gain_w": 0.50,   # 预期涨幅权重
    "prob_w": 0.50,   # 达到概率(convincing rate)权重
    # 入选门 = 纯 T+3 门 (2026-08-09 删 2d 视界: 原 T+2/T+3 联合门退化)
    # 保留 ⇔ T+3 可兑现净预期涨幅 (pred_ret_3d, close-to-close, 成本已扣) > t3_min
    # 2026-08-10: 门基准从 pred_mag_3d (MFE 最大浮盈, 虚高) 改为 pred_ret_3d (c2c 实得),
    # 与 legacy 收益闸口径一致 — 只放行"值得买入、有可兑现预期收益"的个股.
    # 2026-08-14: t3_min 扫描 (_diag_t3min_sweep_20260814, 250d OOS) 定案分板块阈值 —
    #   dual 0.50% 真实赢 (命中10d 66→68%, 实得10d +7.08→+8.01%, ≥+10% 35→38%,
    #   4 子窗全 ≥ 基线, 只砍弱市空仓日); main 无档赢基线 (0.25% +1pp命中/+0.16pp实得
    #   但出股 -18% 不值, 更高档实得反降) → 保持 0.
    "select_gate": {
        # T+3 c2c 净预期涨幅下限: 分板块 dict (main=0 / dual=0.5%), 或全局 float (向后兼容)
        "t3_min": {"main": 0.0, "dual": 0.005},
    },
}

# ── legacy 入选闸分板块参数 (2026-08-14 全量 250d OOS 定案, _diag_legacy_hitrate_topn) ──
# prob_margin: E7 概率闸 (compound_prob > base_rate) 上再收紧: dual +0.08 →
#   命中 62.2→66.3% / 实得 +5.84→+6.66% (出票 3.2→2.9 票/日); main 坍缩期无法评估保持 0.
# pain_max: [E2] 痛苦闸分板块上限: dual 0.5→0.4 (叠加边际后 命中 66.3→75.3% /
#   实得 +6.66→+7.39%, 出票日 177→76 宁缺毋滥); 0.3 过严崩 (59.5%), 勿再收;
#   main 坍缩期无法评估保持 0.5.
LEGACY_ENTRY_GATE = {
    "prob_margin": {"main": 0.0, "dual": 0.08},
    "pain_max": {"main": 0.5, "dual": 0.4},
}

# ── 板块×系统 制度自适应门 (2026-08-05 用户: 砍板块不能写死, 按最新市场数据动态决定) ──
# 每日生成短名单时, 用最新 OOS 市场数据 (固定视界净收益 label_pm) 判定每个 (board, system)
# 组合当前是否值得保留: base_ev = 池内全部行平均净收益 (闭眼全买基准),
# top_ev = 高分 top_quantile 选股平均净收益; 保留 ⇔ top_ev > base_ev + margin.
# 用相对基线的 alpha 而非绝对胜率门槛 — 弱市全市场基率仅 37-46%, 绝对 55% 不可达.
# 2026-08-05 用户定案: 保留判定**只看主视界** (primary_horizon), 不许用长视界兜底把没优势的
# 组合硬留下来 — "如果没有优势就不要入选" → 全不过线则空仓观望.
# 2026-08-09 用户最终定案: **不再整组剔除** — STOCK LIST 显示全部候选 (主板照常出单),
# 但制度门仍按 10d 主视界计算, 每行标注 过门=是/未过; 未过门组合在报告里明确注明
# "不建议今日买入 <板块>股票". 试过 3d 与 10d: 今日 main 在 3d/5d/10d 全视界 α 均跑不赢
# 池基线 (T+10 更是 -2.03%/-0.78%), 无论哪种门主板都会被标注未过.
REGIME_GATE = {
    "enable": True,   # 2026-08-09 用户: 计算 10d 门但不整组剔除 — 清单显示 ALL 候选, 未过门个股标注 过门=未过
    "top_quantile": 0.20,       # 高分 top-quantile = "系统选出的股"
    "margin": 0.01,             # 至少跑赢池基线 1pp (净收益) 才算"有优势"
    "primary_horizon": "10d",   # 2026-08-09 用户: 制度门用 10d 主视界 (与 10d 排名键一致)
    "horizons": ("3d", "5d", "10d"),  # 仅 SUMMARY 展示用 (判定只认 primary_horizon)
    "fallback": "none",         # 无优势 → 空仓观望, 不输出 (旧 "best" 兜底已废, 用户否决)
}
