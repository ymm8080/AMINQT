"""CLI 入口: 运行 A股日线回测.

用法:
    python scripts/run_backtest.py --config config/backtest_config.yaml
    python scripts/run_backtest.py --pred data/pred.csv --price data/price.csv
    python scripts/run_backtest.py --mode sniper --horizon 2
"""

import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.backtest.pipeline import BacktestPipeline  # noqa: E402


def setup_logging(verbose: bool = False) -> None:
    """配置日志."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="A股日线回测系统 (V5.0 + V5.2)",
    )
    parser.add_argument("--config", "-c", default="config/backtest_config.yaml")
    parser.add_argument("--pred", help="预测表路径")
    parser.add_argument("--price", help="行情表路径")
    parser.add_argument("--benchmark", help="基准表路径")
    parser.add_argument("--market", help="大盘指数表路径")
    parser.add_argument("--output", "-o", default="reports")
    parser.add_argument("--mode", "-m", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> int:
    """主函数."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("run_backtest")

    logger.info("=" * 60)
    logger.info("A股日线回测系统 V5.0+V5.2")
    logger.info("=" * 60)

    import yaml

    config_data = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    data_section = config_data.get("data", {})
    output_section = config_data.get("output", {})

    pred_path = args.pred or data_section.get("pred_path", "")
    price_path = args.price or data_section.get("price_path", "")
    benchmark_path = args.benchmark or data_section.get("benchmark_path")
    market_path = args.market or data_section.get("market_path")
    output_dir = args.output or output_section.get("report_dir", "reports")
    modes = args.mode or output_section.get("modes", ["squad", "sniper"])
    horizon = args.horizon

    if not pred_path or not price_path:
        logger.error("必须指定预测表和行情表路径")
        return 1

    logger.info("预测表: %s", pred_path)
    logger.info("行情表: %s", price_path)
    logger.info("模式: %s", modes)
    logger.info("持有期: %d", horizon)

    try:
        pipeline = BacktestPipeline(
            config_path=args.config,
            pred_path=pred_path,
            price_path=price_path,
            benchmark_path=benchmark_path,
            market_path=market_path,
            output_dir=output_dir,
            modes=modes,
            horizon=horizon,
        )
        result = pipeline.run()

        if result.success:
            logger.info("回测完成!")
            for mode in modes:
                metrics = (
                    result.squad_metrics if mode == "squad" else result.sniper_metrics
                )
                if metrics:
                    logger.info(
                        "  [%s] 收益=%.2f%% 夏普=%.2f 回撤=%.2f%% 胜率=%.2f%%",
                        mode.upper(),
                        metrics.get("total_return", 0) * 100,
                        metrics.get("sharpe_ratio", 0),
                        metrics.get("max_drawdown", 0) * 100,
                        metrics.get("win_rate", 0) * 100,
                    )
            for p in result.report_paths:
                logger.info("报告: %s.{json,txt,html}", p)
            return 0
        else:
            for err in result.errors:
                logger.error("  %s", err)
            return 1

    except Exception as e:
        logger.error("回测失败: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
