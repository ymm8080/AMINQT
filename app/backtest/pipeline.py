"""模块8: BacktestPipeline — 回测全流程编排.

编排顺序:
    1. 加载配置 (ConfigManager)
    2. 加载数据 (DataLoader)
    3. 校验数据 (DataValidator)
    4. 信号评估 (SignalEvaluator)
    5. 运行回测 (BacktestEngine) — 支持 Squad + Sniper 双模式
    6. 对比分析 (ComparativeAnalyzer)
    7. 生成报告 (ReportGenerator)

所有步骤 try-except 包裹, 失败时记录日志并返回错误信息.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.backtest.comparative_analyzer import ComparativeAnalyzer
from app.backtest.config_manager import BacktestConfig, ConfigManager
from app.backtest.data_loader import DataLoader
from app.backtest.data_validator import DataValidator
from app.backtest.engine import BacktestEngine
from app.backtest.report_generator import ReportGenerator
from app.backtest.signal_evaluator import SignalEvaluator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """回测管线运行结果."""

    success: bool
    config: BacktestConfig
    squad_metrics: dict[str, Any] = field(default_factory=dict)
    sniper_metrics: dict[str, Any] = field(default_factory=dict)
    squad_result_df: pd.DataFrame | None = None
    sniper_result_df: pd.DataFrame | None = None
    squad_trades_df: pd.DataFrame | None = None
    sniper_trades_df: pd.DataFrame | None = None
    signal_report: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    report_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BacktestPipeline:
    """回测全流程编排器.

    串联所有模块, 从配置加载到报告生成, 一键运行.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        pred_path: str = "",
        price_path: str = "",
        benchmark_path: str | None = None,
        market_path: str | None = None,
        output_dir: str = "reports",
        modes: list[str] | None = None,
        horizon: int = 2,
    ):
        """初始化回测管线.

        Args:
            config_path: 配置文件路径.
            pred_path: 预测表路径.
            price_path: 行情表路径.
            benchmark_path: 基准表路径 (可选).
            market_path: 大盘指数表路径 (V5.0, 可选).
            output_dir: 报告输出目录.
            modes: 回测模式列表, 默认 ["squad", "sniper"].
            horizon: 持有期 (1, 2, 4).
        """
        self.config_path = config_path
        self.pred_path = pred_path
        self.price_path = price_path
        self.benchmark_path = benchmark_path
        self.market_path = market_path
        self.output_dir = output_dir
        self.modes = modes or ["squad", "sniper"]
        self.horizon = horizon

        self.config: BacktestConfig | None = None
        self.loader: DataLoader | None = None
        self.pred_df: pd.DataFrame | None = None
        self.price_df: pd.DataFrame | None = None
        self.benchmark_df: pd.DataFrame | None = None
        self.market_df: pd.DataFrame | None = None
        self.trade_dates: list[pd.Timestamp] = []
        self.data_version_hash: str = ""

        logger.info(
            "BacktestPipeline 初始化: modes=%s, horizon=%d",
            self.modes,
            self.horizon,
        )

    def run(self) -> PipelineResult:
        """运行完整回测管线.

        Returns:
            PipelineResult: 包含所有模式结果、信号报告、对比分析、报告路径.
        """
        errors: list[str] = []

        # ── Step 1: 加载配置 ──
        try:
            self.config = ConfigManager.load(self.config_path)
            logger.info("Step 1 配置加载完成: hash=%s", ConfigManager.hash(self.config))
        except Exception as e:
            msg = f"Step 1 配置加载失败: {e}"
            logger.error(msg)
            return PipelineResult(success=False, config=BacktestConfig(), errors=[msg])

        # ── Step 2: 加载数据 ──
        try:
            self.loader = DataLoader(
                pred_path=self.pred_path,
                price_path=self.price_path,
                benchmark_path=self.benchmark_path,
                market_path=self.market_path,
                config=self.config,
            )
            self.pred_df, self.price_df, self.benchmark_df, self.market_df = (
                self.loader.load()
            )
            self.trade_dates = self.loader.get_trade_dates()
            self.data_version_hash = self.loader.get_data_version_hash()
            logger.info("Step 2 数据加载完成: %d 交易日", len(self.trade_dates))
        except Exception as e:
            msg = f"Step 2 数据加载失败: {e}"
            logger.error(msg)
            return PipelineResult(success=False, config=self.config, errors=[msg])

        # ── Step 3: 数据校验 ──
        try:
            validator = DataValidator(
                pred_df=self.pred_df,
                price_df=self.price_df,
                market_df=self.market_df,
            )
            validation_issues = validator.run_all_checks()
            if validation_issues:
                for code, detail in validation_issues:
                    logger.warning("数据校验: %s %s", code, detail)
                # E-level 致命错误阻止回测 (E002/E003/E004 为致命)
                fatal = [
                    f"{code} {detail}"
                    for code, detail in validation_issues
                    if code in ("E002", "E003", "E004")
                ]
                if fatal:
                    msg = f"Step 3 数据校验发现致命错误: {fatal}"
                    logger.error(msg)
                    return PipelineResult(
                        success=False, config=self.config, errors=[msg]
                    )
            logger.info("Step 3 数据校验完成")
        except Exception as e:
            msg = f"Step 3 数据校验异常: {e}"
            logger.error(msg)
            errors.append(msg)

        # ── Step 4: 信号评估 ──
        signal_report: dict[str, Any] | None = None
        try:
            evaluator = SignalEvaluator(
                pred_df=self.pred_df,
                price_df=self.price_df,
                trade_dates=self.trade_dates,
                config=self.config,
            )
            signal_report = evaluator.run_full_report()
            logger.info("Step 4 信号评估完成")
        except Exception as e:
            msg = f"Step 4 信号评估失败: {e}"
            logger.error(msg)
            errors.append(msg)

        # ── Step 5: 运行回测 (多模式) ──
        mode_results: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
        for mode in self.modes:
            try:
                result_df, trades_df, metrics = self._run_single_mode(mode)
                mode_results[mode] = (result_df, trades_df, metrics)
                logger.info(
                    "Step 5 回测完成 [%s]: return=%.4%%, sharpe=%.2f",
                    mode,
                    metrics.get("total_return", 0) * 100,
                    metrics.get("sharpe_ratio", 0),
                )
            except Exception as e:
                msg = f"Step 5 回测失败 [{mode}]: {e}"
                logger.error(msg)
                errors.append(msg)

        if not mode_results:
            return PipelineResult(
                success=False,
                config=self.config,
                errors=errors,
                signal_report=signal_report,
            )

        # ── Step 6: 对比分析 (如果有两个以上模式) ──
        comparison: dict[str, Any] | None = None
        if (
            len(mode_results) >= 2
            and "squad" in mode_results
            and "sniper" in mode_results
        ):
            try:
                s_res, s_trades, s_metrics = mode_results["squad"]
                n_res, n_trades, n_metrics = mode_results["sniper"]
                analyzer = ComparativeAnalyzer(
                    squad_result=s_res,
                    sniper_result=n_res,
                    squad_trades=s_trades,
                    sniper_trades=n_trades,
                    squad_metrics=s_metrics,
                    sniper_metrics=n_metrics,
                    benchmark_df=self.benchmark_df,
                )
                comparison = analyzer.generate_comparison_report()
                logger.info("Step 6 对比分析完成: %s", comparison.get("recommendation"))
            except Exception as e:
                msg = f"Step 6 对比分析失败: {e}"
                logger.error(msg)
                errors.append(msg)

        # ── Step 7: 生成报告 ──
        report_paths: list[str] = []
        try:
            reporter = ReportGenerator(
                config=self.config,
                output_dir=self.output_dir,
            )
            for mode, (result_df, trades_df, metrics) in mode_results.items():
                basepath = reporter.generate(
                    mode_name=mode,
                    result_df=result_df,
                    trades_df=trades_df,
                    metrics=metrics,
                    signal_report=signal_report,
                    comparison=comparison,
                    data_version_hash=self.data_version_hash,
                )
                report_paths.append(basepath)
            logger.info("Step 7 报告生成完成: %d 份", len(report_paths))
        except Exception as e:
            msg = f"Step 7 报告生成失败: {e}"
            logger.error(msg)
            errors.append(msg)

        # ── 组装结果 ──
        result = PipelineResult(
            success=len(errors) == 0,
            config=self.config,
            signal_report=signal_report,
            comparison=comparison,
            report_paths=report_paths,
            errors=errors,
        )
        if "squad" in mode_results:
            _, result.squad_trades_df, result.squad_metrics = mode_results["squad"]
            result.squad_result_df = mode_results["squad"][0]
        if "sniper" in mode_results:
            _, result.sniper_trades_df, result.sniper_metrics = mode_results["sniper"]
            result.sniper_result_df = mode_results["sniper"][0]

        return result

    def _run_single_mode(
        self, mode: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """运行单个模式的回测.

        Args:
            mode: 持仓模式 (squad / sniper / sniper_max).

        Returns:
            (result_df, trades_df, metrics).
        """
        assert self.config is not None
        assert self.pred_df is not None
        assert self.price_df is not None

        # 为每个模式创建独立的配置副本
        from dataclasses import replace

        mode_config = replace(self.config, position_mode=mode)

        engine = BacktestEngine(
            config=mode_config,
            pred_df=self.pred_df,
            price_df=self.price_df,
            trade_dates=self.trade_dates,
            data_version_hash=self.data_version_hash,
            market_df=self.market_df,
        )

        result_df = engine.run(horizon=self.horizon)
        trades_df = engine.get_trades()
        metrics = engine.get_metrics()

        return result_df, trades_df, metrics
