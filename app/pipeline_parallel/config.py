"""PIPELINE 并行多系统配置 (2026-08-04).

三套并行系统定义 + 双头 (幅度+概率) 验收口径. 与 app/pipeline1/ 完全隔离:
本包只读数据/配置, 不 import pipeline1 的选股/训练逻辑.

特征池来源 (WORM 裁决落盘):
  - data/_sniper_pool_decision_20260804_v2.json  → 狙击系统特征池
  - data/_sniper_pool_decision_20260804.json     → 融合系统特征池
验收口径 (2026-08-04 用户): 任一视界 胜率>=55% 且 平均净收益>0 → 保留.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ── 双头验收阈值 ──
# 验收绝对阈值 (默认/主板; 2026-08-10 重锚: 须 ≥ 随机且盈利)
MIN_WINRATE = 0.50  # 上涨概率下限 (≥ 随机 50%)
MIN_MAG = 0.01  # 平均净收益下限 (≥ 1%)

# OOS 样本外窗口 (2026-08-04 用户: "BACKTESTING CONSISTS OF 6M, 3M, 10D").
# 6m (≈126 交易日, 主验收) + 3m (≈63 交易日) 两个聚合回测;
# 10d 额外给出逐日股票清单 + 每日期回测结果 (last_days_report).
# 2026-08-04 用户澄清: 10D 窗必须跨末 15 个交易日 (非 10) —
#   买在 T+1 → T+5 需未来 6 日价 (末 15 日内有 10 日可测),
#   T+10 需未来 11 日价 (末 15 日内有 5 日可测), 同一测试日 (选股日) 共用.
#   若只用末 10 日, T+10 可测日数=0, 该视界在 10D 窗内全为 NaN.
OOS_WINDOWS: dict[str, int] = {
    "6m": 6 * 21,
    "3m": 3 * 21,
    "10d": 15,
}

# 板块拆分 (2026-08-04 用户: MAIN/DUAL 分开回测, 幅度阈值不同).
# 代码前缀: 60xxxxx 沪主板 / 00xxxxx 深主板 → main; 30xxxxx 创业板 / 68xxxxx 科创 → dual.
BOARD_PREFIXES: dict[str, tuple[str, ...]] = {
    "main": ("60", "00"),
    "dual": ("30", "68"),
}
# 每板块双头验收 (2026-08-10 重锚): 08-07 c2c 切换 (28a00bc4) 把验收标签从 MFE 换成
# label_pm_{h}d_net, 但阈值仍是 MFE 时代绝对值 (MFE 全池基准胜率~90%/幅度+8%,
# c2c 基准胜率~45%/幅度~0) → 旧 0.55/3-4% 绝对闸结构上不可达, 全部 6m OOS 误报"未过".
# 重锚为交易者诚实基准: 胜率 ≥ 50% (至少赢过随机抛硬币/闭眼全买) 且 平均净收益 ≥ 1%.
# 不换回 MFE — MFE 是窗口内最高价"触摸天花板", 实得卖不到, 用它会虚高过闸.
# dual 幅度阈值仍高于 main (2026-08-04 用户: 20cm 潜力更高), 但都远低于旧 MFE 锚.
# 验收 = 任一视界 胜率>=min_winrate 且 幅度>min_mag → 保留.
BOARD_THRESHOLDS: dict[str, dict] = {
    "main": {"min_winrate": 0.50, "min_mag": 0.01, "label": "主板"},
    "dual": {"min_winrate": 0.50, "min_mag": 0.015, "label": "创业板+科创板"},
}


def board_of(symbol) -> str:
    """按代码前缀判板块; 未知前缀归 main."""
    p = str(symbol)[:2]
    for b, prefs in BOARD_PREFIXES.items():
        if p in prefs:
            return b
    return "main"


# 目标口径 (2026-08-04 用户): MFE = 持有期内**最大涨幅** (潜在最优离场),
# 非目标日收盘收益. 列名 label_mfe_{h}d_net, 由 backtest.add_mfe_labels 补算.
# 两套系统统一测 T+3/T+5/T+10 三视界矩阵 (2026-08-09 删 2d 视界).
HORIZONS: tuple[str, ...] = ("3d", "5d", "10d")
# 2026-08-07 用户: 并行全模块验收改 close-to-close (可兑现收益), 非 MFE 触摸天花板
C2C_LABELS: tuple[str, ...] = tuple(f"label_pm_{h}_net" for h in HORIZONS)

# ── LEGACY 叠加权重 (2026-08-05 正交性实证, _diag_overlay_orthogonality) ──
# OOS 6m: main 上纯 prob 单用最好 (池=prob 弱版, 0.5/0.5 叠加稀释) → 偏 prob;
# dual 上叠加在幅度 (MFE) 赢 → 保持对半 (pv_corr_5 在 dual 叠加口径有增量).
OVERLAY_WEIGHTS: dict[str, dict[str, float]] = {
    "main": {"w_pool": 0.2, "w_prob": 0.8},
    "dual": {"w_pool": 0.5, "w_prob": 0.5},
}

# ── ADX 慢牛系统 (SLOW_BULL) 常量 (2026-08-05, ADX 设计文档 v1.0) ──
# 持有 2-8 周 (10-40 交易日) → 长视界验收, 匹配文档目标 (累计 50%-150%).
SLOW_BULL_HORIZONS: tuple[str, ...] = ("10d", "20d", "40d")
# add_mfe_labels 用全系统视界并集 (sniper/fusion 3/5/10 + slow_bull 10/20/40)
ALL_HORIZON_INTS: tuple[int, ...] = (3, 5, 10, 20, 40)
# 慢牛模块版本戳 (清单文件名 module 后缀; 规则系统无训练模型 → 用设计文档版本号)
SLOW_BULL_VERSION: str = "v1_0"

# ADX 硬门槛/打分阈值 (文档 §2.2/§2.3/§3/§4, 2026-08-05 落地)
ADX_SPEC: dict = {
    "adx_period": 14,  # ADX/+DI/-DI 周期 (Wilder's, EMA 平滑)
    "adx_min": 25.0,  # 门槛二: ADX 下限 (趋势强劲)
    "adx_rise_lookback": 5,  # ADX 近 N 日上升判定窗口
    "ma_bias_max": 0.05,  # 门槛一: ma5 与 ma10 乖离率上限
    "slope_lookback": 3,  # MA 斜率判定窗口 (交易日)
    "amplitude_20_max": 0.06,  # 门槛三: 20日均振幅上限
    "max_drop_20_max": 0.05,  # 门槛三: 20日最大单日跌幅上限
    "vol_ratio_max": 3.0,  # 门槛四: 昨日量比上限
    "turnover_min": 3.0,  # 门槛四: 换手率下限 (面板 turnover_rate 单位 = 百分数, 3.0=3%)
    "turnover_max": 15.0,  # 门槛四: 换手率上限 (15% 涨停附近过热排除)
    "dev5_max": 0.08,  # 不买: 偏离 ma5 > 8%
    "vol_spike_up_max": 0.05,  # 不买: 放量上涨 > 5%
    "adx_optimal_max": 40.0,  # 打分 adx_score 上限 (25-40 最佳, >40 过热不再加分)
    "big_drop_sell": 0.07,  # 卖出: 单日放量大跌 > 7%
    "tp_gain": 0.80,  # 卖出: 累计涨幅 > 80% 且 ADX 顶背离
    "rps_lookback": 60,  # RPS 涨幅窗口 (交易日)
    "sharpe_lookback": 20,  # 夏普窗口 (交易日)
    "below_ma5_days": 3,  # 卖出: 连续 N 日收于 ma5 下方
    "turnover_spike_win": 20,  # 卖出: 换手突增至近 N 日最高
    "adx_broken_min": 20.0,  # 卖出: ADX 跌破此值 → 趋势衰竭 (文档 §4.2)
    "vol_spike_ratio_min": 1.5,  # 放量判定: 量比下限 (不买追高 / 卖出放量)
    "shrink_vol_ratio_max": 0.8,  # 买入: 缩量回调量比上限 (文档 §3.1)
    "small_candle_max": 0.03,  # 买入: 小阴线实体上限 (缩量回调)
    "pullback_tol": 0.01,  # 买入: 回踩 ma5 容差 (低吸判定)
    "tp_high_window": 60,  # tp_80_div 代理: 近 N 日新高窗口 (无成本时)
    "tp_high_near": 0.02,  # tp_80_div 代理: 距 N 日高点容忍距离
}

# ADX 打分因子权重 (文档 §2.3 表2; 北向资金 10% 数据停更 2024-08 → 缺列自动跳过并再归一化)
# 因子列来源: adx_score/ma_tightness/sharpe_20/rps_60/pv_corr_5 = indicators.prepare_adx 计算;
# margin_balance_chg_5d = 面板 dim24 已有列; pct_70_con = cyq_panel.parquet 补列.
ADX_SCORE_WEIGHTS: dict[str, float] = {
    "adx_score": 0.20,  # ADX 值 (25-40 最佳)
    "ma_tightness": 0.15,  # 均线排列紧密度 (ma5-ma20 间距)
    "sharpe_20": 0.15,  # 20日夏普
    "rps_60": 0.15,  # RPS 相对强度 (vs 全市场 60日)
    "pv_corr_5": 0.10,  # 量价相关系数 (5日)
    "margin_balance_chg_5d": 0.10,  # 融资余额变化 (5日)
    "pct_70_con": 0.05,  # 筹码集中度 pct_70
}


@dataclass(frozen=True)
class SystemSpec:
    """一套并行系统的完整定义."""

    name: str
    desc: str
    pool: tuple[str, ...]  # 特征池 (截面 TOP-N 打分用)
    top_n: int  # 主输出档位 (狙击 5 / 融合 10 / 慢牛 20)
    top_n_alt: int  # 附档位 (狙击 3)
    horizons: tuple[str, ...]  # 验收视界, 按裁决优先级排列
    labels: tuple[str, ...]  # 对应 label_pm_{h}d_net 列名
    enabled: bool = True
    gate: str | None = None  # 硬门槛函数名 (screener.GATES); None → 纯池打分
    pool_weights: dict | None = None  # 打分权重 (pool_score weights); None → 等权
    notes: tuple[str, ...] = field(default_factory=tuple)


# 狙击系统: T+1 买, 目标 MFE (持有期内最大涨幅), 统一测 T+3/T+5/T+10 三视界;
# 裁决优先级 3d > 5d > 10d (2026-08-09 删 2d 视界).
# 池 = v2 裁决 pool_top5 (5 特征) + pool_top3_extra (ret_reversal_5d).
SNIPER = SystemSpec(
    name="sniper",
    desc="狙击: 每日 3-5 只, T+1 买, 目标 MFE, 三视界 T+3/5/10 任一过双头即保留 (3d>5d>10d)",
    pool=(
        "amihud_illiq",
        "small_mv_premium",
        "amihud_illiquidity",
        "VAR51",
        "ret_reversal_5d",
        "pv_corr_5",
    ),
    top_n=5,
    top_n_alt=3,
    horizons=HORIZONS,
    labels=C2C_LABELS,
    notes=(
        "核心 3 特征 (amihud_illiq/small_mv_premium/amihud_illiquidity) 3d 过, 快进快出可用",
        "VAR51 / ret_reversal_5d 长视界出边 — 须持有多天才兑现",
        "limit_dist_pct TOP-5 全视界不过 → 只进融合池",
        "2026-08-05 池内相关 OOS 边际: +pv_corr_5 → dual 全视界 Δwr +1.7~3.6% (5d +3.6% 最强)",
        "2026-08-08 c2c LOO 审计 (250/300/200d): 剔除 down_gap_pct → 双板 STABLE_WIN +0.45~+0.72pp (MFE 选入但 c2c 不兑现)",
        "目标 = MFE (窗口内最高价可兑现的最大收益), 非目标日收盘",
    ),
)

# 融合系统: 大仓位, 持有 3-5 天; 目标 MFE, 三视界 T+3/5/10 (2026-08-09 删 2d).
# 池 = 6 独立特征 (去重后).
FUSION = SystemSpec(
    name="fusion",
    desc="融合: 大仓位, 持有 3-5 天, 目标 MFE, TOP-10 三视界 T+3/5/10 双头",
    pool=(
        "amihud_illiquidity",
        "VAR51",
        "limit_dist_pct",
        "ret_reversal_5d",
        "small_mv_premium",
        "pv_corr_5",
    ),
    top_n=10,
    top_n_alt=10,
    horizons=HORIZONS,
    labels=C2C_LABELS,
    notes=(
        "limit_dist_pct 长视界出边 — 融合方案须容忍较长持有",
        "small_mv_premium 高风险档 — 仓位纪律必需",
        "2026-08-05 池内相关 OOS 边际: +pv_corr_5 → dual 全视界 Δwr +1.7~2.2%",
        "2026-08-08 c2c LOO 审计 (250/300/200d): 剔除 down_gap_pct → 双板 STABLE_WIN +0.45~+0.72pp (MFE 选入但 c2c 不兑现)",
        "目标 = MFE (窗口内最高价可兑现的最大收益), 非目标日收盘",
    ),
)

# 慢牛系统 (2026-08-05 ADX 设计文档 v1.0 落地): 硬门槛先行 → 加权打分 Top-20 池,
# 均线低吸买入, 破均线/ADX 衰竭卖出, 移动止盈, 持有 2-8 周.
# 独立 70% 资金仓 (文档第五章: 70% 慢牛 + 30% 狙击), 不并入狙击/融合短名单.
SLOW_BULL = SystemSpec(
    name="slow_bull",
    desc="慢牛: ADX+均线硬门槛 → 加权打分 Top-20 池, 均线低吸, 破均线/ADX衰竭卖出, 持有 2-8 周",
    pool=tuple(ADX_SCORE_WEIGHTS),
    top_n=20,
    top_n_alt=10,
    horizons=SLOW_BULL_HORIZONS,
    labels=tuple(f"label_pm_{h}_net" for h in SLOW_BULL_HORIZONS),
    gate="slow_bull",
    pool_weights=dict(ADX_SCORE_WEIGHTS),
    notes=(
        "硬门槛 (ADX>25 且 +DI>-DI, 均线多头, 低波动, 量价健康) 全部满足才进打分池",
        "打分因子缺列自动跳过 + 权重再归一化 (北向资金 10% 停更 2024-08 → 跳过)",
        "目标 = MFE (持有期内最大涨幅), 长视界 T+10/20/40 匹配 2-8 周持有",
    ),
)

SYSTEMS: dict[str, SystemSpec] = {s.name: s for s in (SNIPER, FUSION, SLOW_BULL)}

# ── SLOW_BULL 市场状态条件退出 (2026-08-06) ──
# 依据 data/_diag_slowbull_stability_* + _diag_slowbull_regime_*: trail8 是趋势跟随
# 放大器 (上升段 +2~4pp, 下行段 -1.4~-5.2pp); 下行段池子所有退出都亏 (cur -0.68%/
# dual -0.86%) → 默认下行不开仓 (no_open), 符合用户"预期收益不够高就不开仓".
# 市场代理 = 面板每日全部股票 close_hfq 中位数 (PIT, 自洽无外部依赖).
SLOW_BULL_REGIME: dict = {
    "def": "A",  # A: 代理 > MA20 (经典短趋势) / B: > MA60 / C: 20日动量为正
    "ma_window": 20,  # 代理 MA 窗口 (交易日)
    "trail_pct": 0.08,  # 上升段退出: 收盘自峰值回落 8% 走 (trail8)
    "hard_stop": 0.92,  # 上升段硬止损 -8%
    "max_hold": 40,  # 最长持有 (交易日)
    "down_mode": "no_open",  # 下行段: no_open=不开仓 / cur=仍出候选但按现行退出
}

# ── SLOW_BULL 模块暂停交付 (2026-08-26) ──
# 08-25 三模块诊断: 慢牛确诊"选错股" — main 池 3 日 -4.1%/胜率 16%, 落后全市场基准
# 2.9pp, 11 个可评日天天跑输, 且 07-31~08-05 基准 +4.0% 段仍 -6.8pp (动量风格回杀,
# 非熊市背锅); trail8 只救退出救不了入场 (仍 -4.5%). 用户批准暂停交付候选.
# 恢复 = paused 改 False; 建议先重构入场信号并通过 250d OOS 验收再开.
SLOW_BULL_PAUSE: dict = {
    "paused": True,
    "reason": (
        "模块暂停交付 (2026-08-26): 选股信号失效 — 3日实得 -4.1% vs 全市场基准 "
        "落后 2.9pp, 连续跑输; 恢复需重构入场信号并通过 250d OOS 验收"
    ),
}

# ── SLOW_BULL rps_60 第二道门 (2026-08-08, factor_gradient + rps_gate 诊断) ──
# gate 内单因子梯度: rps_60 截面分位是唯一双板单调的"守得住"预测因子 (main d10-d1
# +4.08pp / dual +5.71pp, spearman 0.73/0.65). 策略模拟显示该信号只在 dual 未被
# 收割: main 合成 score 含 rps_60 权重 0.15 已部分捕获 → 硬门槛反覆盖合成择优
# (main 0.5 门 −0.06pp, 非单调); dual 池薄 (~6/日) 合成 score 近乎 no-op → 门槛直接
# +0.80pp (OOS 250d, 4 档单调, 3/4 季度正, 下跌季 Q4 最强 +2.17pp, 横盘季 Q3 唯一负
# −0.46pp 噪声级). 集中度无塌缩 (dual 门后仍 447 只/250d). 符合用户"预期收益不够高
# 就不开仓". floor 为 gate 内日截面 rps_60 百分位下限.
SLOW_BULL_RPS_GATE: dict = {
    "enabled": True,
    "floor": 0.50,
    "boards": {"main": False, "dual": True},  # main 合成 score 已捕获 → 不启
}

# ── SLOW_BULL 排名键 (2026-08-08, rank_ab 诊断) ──
# 受控 A/B (gate∩上升段→Top-N→trail8 op_rule 实得, OOS 250d): dual 用 rps_60 排名
# 全档赢合成 score (top10 +1.15pp / top5 +0.52pp / top3 +1.19pp, 3/4 季度稳定, 下跌季
# 最强), 机制同门: dual 池薄合成 score 近乎 no-op → rps 信号未收割; main 合成 score
# 含 rps_60 权重 0.15 已部分捕获 → 纯 rps 排名反更差 (+0.39pp 输 @top10, 且季度方向
# 不净). top_n 按板收紧: 现状 top-20 全收 (池 ~6-13/日 < 20) 排名不生效; dual 收紧到
# 10 让排名在深池日生效 (门后 ~6/日, 仅深池日裁到 top-10), main 保持 20.
SLOW_BULL_RANK: dict = {
    "key": "rps_60",  # 排名键: dual=rps_60 相对强度; main=合成 score (boards 关)
    "boards": {"main": False, "dual": True},
    "top_n": {"main": 20, "dual": 10},
}


@dataclass(frozen=True)
class PanelSource:
    """行集来源 (快速路径: 复用 3y 诊断检查点, 不重建面板).

    window_days: 读取时只保留末 N 个交易日的行 (2026-08-10 消融定案).
      并行规则系统对历史长度无状态 — 打分只看当日横截面 + 固定窗口滚动特征,
      消融实测末 242 交易日与全量 726 交易日的验收判定 108/108 完全一致,
      内存 3y=5.35GB → 1y=2.22GB (省 58.5%). 设 726 恢复 3y.
      检查点文件本身保持 3y 不动 (诊断脚本共用), 仅在 load_panel 读取时 pyarrow 过滤.
    """

    main_checkpoint: str = os.path.join("data", "_diag_stage_main_3y.parquet")
    dual_checkpoint: str = os.path.join("data", "_diag_stage_dual_3y.parquet")
    window_days: int = 242  # 1y ≈ 242 交易日; 726 = 3y (消融已验证等价, 内存省 58.5%)
    # 与生产行集一致 (run_train → features.build → labels → mask), 由 _finalize_slice 补 10d 净标签


PANEL = PanelSource()

# ── mag_10d 校准参数 (2026-08-07 定案, diag_10d_param_sweep_nl_20260807_200540) ──
# 并行系统 (除慢牛) 短名单排名 = 每股收缩回归 score→label_pm_10d_net (T+10
# close-to-close 校准幅度) 的全板块日截面降序 → TOP5/TOP10. 拟合窗 [D-cal_n, D)
# 只用**已实现**标签: 行 t 的卖价 close_hfq[T+11] 须在决策日 D 之前已打印 → 拟合边界
# 比 D 提前 realized_drop = buy_lag + label_horizon = 11 个交易日 (无前瞻, 铁律).
# 无前瞻扫描 (横截面按日期裁 11 交易日, 250d OOS) 定案:
#   cal_n=21 (10 已实现日) + per_stock_min_n=50 (=纯板块横截面 OLS) 双板块最优;
#   每股回归无前瞻下样本 ~10-31 太噪 (minn=15/20 为负), psw 完全 no-op, kappa=10 最优.
MAG10D_CAL = {
    "cal_n": 21,  # 校准窗 (交易日; cal_n 扫描最优 = 短窗, 对近期 regime 适应快)
    "per_stock_window": 130,  # 每股自用最近交易日 (no-op, 纯横截面下不触发)
    "per_stock_min_n": 50,  # 每股回归最小样本, 不足回退横截面; =50 强制纯横截面 OLS
    "shrink_kappa": 10.0,  # 收缩强度 (λ = take/(take+κ))
    "score_col": "score",  # 校准输入: score = max(sniper, fusion) 池分
    "target_col": "label_pm_10d_net",  # 校准目标: T+10 close-to-close 净幅度
    "cross_min_n": 50,  # 横截面最小样本, 不足该板块当日不出票
    "buy_lag": 1,  # 买在 close[T+1] (相对决策日 D)
    "label_horizon": 10,  # 视界 10 交易日
}
