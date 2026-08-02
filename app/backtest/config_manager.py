# -*- coding: utf-8 -*-
"""模块1: ConfigManager — 配置管理.

所有参数从 config.yaml 读取, 禁止魔法数字.
BacktestConfig dataclass 定义所有回测参数.
"""

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置 (所有参数从 config.yaml 读取).

    价格精度: 所有金额用整数分(fen), 禁止float累积.
    """

    # ── 资金 ──
    initial_capital: float = 100000.0

    # ── 触发 ──
    trigger_pct: float = 0.03

    # ── 滑点 (basis points, 1bp = 0.01%) ──
    slippage_buy_bp: int = 10       # 买入 0.1%
    slippage_sell_moo_bp: int = 10  # 正常卖出 0.1%
    slippage_sell_stop_bp: int = 30  # 止损/止盈卖出 0.3%

    # ── 佣金 ──
    commission_rate: float = 0.00025  # 万2.5
    min_commission: float = 5.0       # 最低5元
    stamp_tax_rate: float = 0.001     # 千1, 仅卖出

    # ── 风控 (日内规则V5.2融合, ATR自适应) ──
    # S1 止损: max(固定值, -1.5×ATR_pct), 分主板/双创
    stop_loss_main: float = -0.04       # 主板固定止损 -4%
    stop_loss_dual: float = -0.06       # 双创固定止损 -6%
    stop_loss_atr_mult: float = 1.5     # ATR倍数
    stop_loss_atr_floor: float = 1.2    # 噪音带断言: abs(stop) >= 1.2×ATR_pct
    # S2 移动止盈: 浮盈≥3%激活, 回撤max(3%, 1.0×ATR_pct)
    trailing_stop: bool = True
    trailing_stop_activate: float = 0.03  # 浮盈3%激活
    trailing_stop_min_pct: float = 0.03   # 回撤下限3%
    trailing_stop_atr_mult: float = 1.0   # 回撤ATR倍数
    # S5a 时间止损: 满2日且2日收益<20日中位数
    time_stop_days: int = 2               # 时间止损持有天数
    time_stop_use_median: bool = True     # 用20日中位数( False=固定1%)
    time_stop_fixed_threshold: float = 0.01  # 固定阈值1% (use_median=False时)
    # S5b 持仓到期
    holding_period: int = 2               # 持有期(交易日), 0=使用horizon参数
    # 日保险丝: μ-2σ双轨 + 固定兜底
    daily_fuse_fixed: float = -0.04       # 固定日亏损兜底 -4%
    daily_fuse_use_sigma: bool = True     # 启用2σ自适应
    daily_fuse_sigma: float = 2.0         # σ倍数
    daily_fuse_window: int = 20           # 滚动窗口
    # 系统停机线
    system_halt_drawdown: float = -0.15   # 总资金回撤≥15%停机
    # 连续亏损
    consecutive_loss_limit: int = 3       # 连续亏损交易日限制
    consecutive_loss_cooldown: int = 5    # 冷却天数
    # ATR参数
    atr_period: int = 14                  # ATR计算周期(交易日)

    # ── 买入过滤 (日内规则融合) ──
    max_gain_pct: float = 0.07            # B3 追高过滤: 涨幅≤7%
    min_net_edge: float = 0.005           # B5 净收益闸门: 0.5%
    stop_distance_atr_mult: float = 1.2   # B7 止损距离否决: <1.2×ATR放弃

    # ── V5.0 新增参数 ──
    volume_confirm_ratio: float = 1.5     # F2.17 成交量确认: T+1量 >= T量*1.5
    market_drop_limit: float = -0.02      # F2.18 大盘暴跌停买: 跌幅>=2%
    down_limit_max_days: int = 3          # F2.19 连续跌停强制平仓: 3日
    max_swap_per_day: int = 2             # 每日换仓上限
    swap_threshold: float = 0.01          # 换仓收益阈值: 1%

    # ── 选股 ──
    prob_threshold: float = 0.55
    position_mode: str = "squad"      # squad / sniper / sniper_max
    filter_st: bool = True
    filter_trend: bool = False        # 默认关闭

    # ── 资金管理 ──
    cash_interest_rate: float = 0.003  # 年化0.3%现金利息
    min_tradeable: int = 2             # 最小可交易候选数
    volume_limit_pct: float = 0.10     # 单笔买入≤当日成交额10%
    min_position_value: float = 20000  # 最小交易金额

    # ── 信号评估 ──
    signal_horizons: list[int] = field(default_factory=lambda: [1, 2, 4])
    signal_simulate_trigger: bool = True
    signal_k: int = 5


class ConfigManager:
    """配置管理器.

    从 YAML 文件加载配置, 生成 BacktestConfig dataclass.
    提供配置哈希用于审计.
    """

    @staticmethod
    def load(path: str = "config.yaml") -> BacktestConfig:
        """从 YAML 文件加载配置.

        Args:
            path: 配置文件路径.

        Returns:
            BacktestConfig 实例.

        Raises:
            FileNotFoundError: 配置文件不存在.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("配置文件不存在: %s, 使用默认配置", path)
            return BacktestConfig()

        with open(p, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        # 读取 backtest 段
        bt_section = raw.get("backtest", {})
        sig_section = raw.get("signal_eval", {})

        kwargs: dict[str, Any] = {}
        for key in bt_section:
            if hasattr(BacktestConfig, key):
                kwargs[key] = bt_section[key]

        # 信号评估参数
        if "horizons" in sig_section:
            kwargs["signal_horizons"] = sig_section["horizons"]
        if "simulate_trigger" in sig_section:
            kwargs["signal_simulate_trigger"] = sig_section["simulate_trigger"]
        if "k" in sig_section:
            kwargs["signal_k"] = sig_section["k"]

        config = BacktestConfig(**kwargs)
        logger.info("配置加载完成: %s, hash=%s", path, ConfigManager.hash(config))
        return config

    @staticmethod
    def hash(config: BacktestConfig) -> str:
        """计算配置的 SHA256 哈希 (用于审计).

        Args:
            config: BacktestConfig 实例.

        Returns:
            "sha256:" 前缀的哈希字符串.
        """
        d = asdict(config)
        # 序列化为确定性字符串
        raw = repr(sorted(d.items())).encode("utf-8")
        h = hashlib.sha256(raw).hexdigest()[:16]
        return f"sha256:{h}"
