"""模块7: ReportGenerator — 回测报告生成器.

生成文本/JSON/HTML 格式的回测报告, 包含:
    1. 绩效概览 (收益率/夏普/回撤/胜率)
    2. 信号评估报告
    3. 对比分析 (Squad vs Sniper)
    4. 交易明细摘要
    5. 审计信息 (配置哈希/数据版本哈希)
    6. V5.0 Approximate Mode 免责声明

所有报告 append-only, 严禁覆盖已生成报告 (铁律10).
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.backtest.config_manager import BacktestConfig, ConfigManager

logger = logging.getLogger(__name__)

_APPROX_MODE_DISCLAIMER = (
    "⚠ V5.0 Approximate Mode: 日线数据无法精确模拟10:30时间窗口, "
    "实际收益可能低于回测. 成交量确认/大盘过滤均为日频代理."
)


class ReportGenerator:
    """回测报告生成器.

    生成文本、JSON、HTML 三种格式的报告.
    """

    def __init__(
        self,
        config: BacktestConfig,
        output_dir: str = "reports",
    ):
        """初始化报告生成器.

        Args:
            config: 回测配置实例.
            output_dir: 报告输出目录.
        """
        self.config = config
        self.output_dir = output_dir
        self.config_hash = ConfigManager.hash(config)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        logger.info("报告生成器初始化: output_dir=%s", output_dir)

    def generate(
        self,
        mode_name: str,
        result_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: dict[str, Any],
        signal_report: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        holdings_history: pd.DataFrame | None = None,
        data_version_hash: str = "",
    ) -> str:
        """生成完整回测报告 (文本 + JSON + HTML).

        Args:
            mode_name: 模式名称 (squad / sniper / sniper_max).
            result_df: 每日 NAV 记录.
            trades_df: 交易明细.
            metrics: 绩效指标字典.
            signal_report: 信号评估报告 (可选).
            comparison: 对比分析结果 (可选).
            holdings_history: 每日持仓历史 (可选).
            data_version_hash: 数据版本哈希.

        Returns:
            生成的报告文件路径 (不含扩展名).
        """
        basename = f"backtest_{mode_name}_{self._timestamp}"
        basepath = os.path.join(self.output_dir, basename)

        # JSON 报告 (结构化数据, 便于程序读取)
        json_path = basepath + ".json"
        self._write_json(
            json_path,
            mode_name,
            result_df,
            trades_df,
            metrics,
            signal_report,
            comparison,
            data_version_hash,
        )

        # 文本报告 (人类可读)
        txt_path = basepath + ".txt"
        self._write_text(
            txt_path,
            mode_name,
            result_df,
            trades_df,
            metrics,
            signal_report,
            comparison,
            data_version_hash,
        )

        # HTML 报告 (可视化)
        html_path = basepath + ".html"
        self._write_html(
            html_path,
            mode_name,
            result_df,
            trades_df,
            metrics,
            signal_report,
            comparison,
            data_version_hash,
        )

        logger.info("报告生成完成: %s.{json,txt,html}", basepath)
        return basepath

    def _write_json(
        self,
        path: str,
        mode_name: str,
        result_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: dict[str, Any],
        signal_report: dict | None,
        comparison: dict | None,
        data_version_hash: str,
    ) -> None:
        """写入 JSON 格式报告."""
        try:
            report: dict[str, Any] = {
                "meta": {
                    "mode": mode_name,
                    "timestamp": self._timestamp,
                    "config_hash": self.config_hash,
                    "data_version_hash": data_version_hash,
                    "disclaimer": _APPROX_MODE_DISCLAIMER,
                },
                "metrics": self._sanitize_for_json(metrics),
                "signal_report": self._sanitize_for_json(signal_report)
                if signal_report
                else None,
                "comparison": self._sanitize_for_json(comparison)
                if comparison
                else None,
                "trades_summary": self._trades_summary(trades_df),
                "nav_summary": self._nav_summary(result_df),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            logger.info("JSON 报告: %s", path)
        except Exception as e:
            logger.error("JSON 报告写入失败: %s", e)

    def _write_text(
        self,
        path: str,
        mode_name: str,
        result_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: dict[str, Any],
        signal_report: dict | None,
        comparison: dict | None,
        data_version_hash: str,
    ) -> None:
        """写入纯文本格式报告."""
        lines: list[str] = []
        sep = "=" * 70

        lines.append(sep)
        lines.append(f"  A股日线回测报告 — {mode_name.upper()}")
        lines.append(f"  生成时间: {self._timestamp}")
        lines.append(sep)
        lines.append("")

        # 1. 绩效概览
        lines.append("─" * 50)
        lines.append("  1. 绩效概览")
        lines.append("─" * 50)
        lines.append(f"  总收益率:       {metrics.get('total_return', 0):.4%}")
        lines.append(f"  年化收益率:     {metrics.get('annual_return', 0):.4%}")
        lines.append(f"  年化波动率:     {metrics.get('annual_volatility', 0):.4%}")
        lines.append(f"  夏普比率:       {metrics.get('sharpe_ratio', 0):.4f}")
        lines.append(f"  最大回撤:       {metrics.get('max_drawdown', 0):.4%}")
        lines.append(f"  回撤持续期:     {metrics.get('max_drawdown_duration', 0)} 天")
        lines.append(f"  Calmar 比率:    {metrics.get('calmar_ratio', 0):.4f}")
        lines.append(f"  胜率:           {metrics.get('win_rate', 0):.4%}")
        lines.append(f"  盈亏比:         {metrics.get('profit_loss_ratio', 0):.4f}")
        lines.append(f"  平均盈利:       {metrics.get('avg_win', 0):.2f}")
        lines.append(f"  平均亏损:       {metrics.get('avg_loss', 0):.2f}")
        lines.append(f"  最大单笔盈利:   {metrics.get('max_win', 0):.2f}")
        lines.append(f"  最大单笔亏损:   {metrics.get('max_loss', 0):.2f}")
        lines.append(f"  总交易次数:     {metrics.get('num_trades', 0)}")
        lines.append(f"  盈利交易:       {metrics.get('num_winning_trades', 0)}")
        lines.append(f"  亏损交易:       {metrics.get('num_losing_trades', 0)}")
        lines.append(f"  日可交易率:     {metrics.get('daily_tradeable_rate', 0):.4%}")
        lines.append("")

        # 2. 交易明细摘要
        lines.append("─" * 50)
        lines.append("  2. 交易明细摘要")
        lines.append("─" * 50)
        ts = self._trades_summary(trades_df)
        lines.append(f"  总交易数:       {ts['total_trades']}")
        lines.append(f"  买入交易:       {ts['buy_trades']}")
        lines.append(f"  卖出交易:       {ts['sell_trades']}")
        lines.append(f"  换仓交易:       {ts.get('swap_trades', 0)}")
        if ts["total_trades"] > 0 and not trades_df.empty:
            lines.append("")
            lines.append("  最近 10 笔交易:")
            lines.append(
                f"  {'日期':<12} {'股票':<8} {'买价':>8} {'卖价':>8} {'盈亏%':>8} {'原因':<12}"
            )
            recent = trades_df.tail(10)
            for _, t in recent.iterrows():
                lines.append(
                    f"  {str(t.get('exit_date', ''))[:10]:<12} "
                    f"{str(t.get('stock', '')):<8} "
                    f"{t.get('entry_price', 0):>8.2f} "
                    f"{t.get('exit_price', 0):>8.2f} "
                    f"{t.get('pnl_pct', 0):>8.2%} "
                    f"{str(t.get('exit_reason', '')):<12}"
                )
        lines.append("")

        # 3. NAV 曲线摘要
        lines.append("─" * 50)
        lines.append("  3. 资金曲线摘要")
        lines.append("─" * 50)
        ns = self._nav_summary(result_df)
        lines.append(f"  初始资金:       {self.config.initial_capital:.2f}")
        lines.append(f"  最终 NAV:       {ns['final_nav']:.2f}")
        lines.append(f"  交易天数:       {ns['num_days']}")
        lines.append(f"  最高 NAV:       {ns['max_nav']:.2f}")
        lines.append(f"  最低 NAV:       {ns['min_nav']:.2f}")
        lines.append("")

        # 4. 信号评估
        if signal_report:
            lines.append("─" * 50)
            lines.append("  4. 信号评估")
            lines.append("─" * 50)
            for horizon_key, sr in signal_report.items():
                lines.append(f"  [{horizon_key.upper()}]")
                lines.append(f"    Rank IC 均值:   {sr.get('rank_ic_mean', 0):.4f}")
                lines.append(f"    Rank IC 标准差: {sr.get('rank_ic_std', 0):.4f}")
                lines.append(f"    Rank IR:        {sr.get('rank_ir', 0):.4f}")
                lines.append(f"    命中率:         {sr.get('hit_rate', 0):.4%}")
                lines.append(
                    f"    成交量确认率:   {sr.get('volume_confirmation_rate', 0):.4%}"
                )
                ts_data = sr.get("trigger_stats", {})
                if ts_data:
                    lines.append(f"    触发统计:       {ts_data}")
                lines.append("")

        # 5. 对比分析
        if comparison:
            lines.append("─" * 50)
            lines.append("  5. 对比分析 (Squad vs Sniper)")
            lines.append("─" * 50)
            lines.append(
                f"  集中度风险系数:   {comparison.get('concentration_risk_ratio', 0):.4f}"
            )
            lines.append(f"  Squad 夏普:       {comparison.get('squad_sharpe', 0):.4f}")
            lines.append(
                f"  Sniper 夏普:      {comparison.get('sniper_sharpe', 0):.4f}"
            )
            lines.append(f"  Squad 最大回撤:   {comparison.get('squad_max_dd', 0):.4%}")
            lines.append(
                f"  Sniper 最大回撤:  {comparison.get('sniper_max_dd', 0):.4%}"
            )
            lines.append(
                f"  Squad 总收益:     {comparison.get('squad_total_return', 0):.4%}"
            )
            lines.append(
                f"  Sniper 总收益:    {comparison.get('sniper_total_return', 0):.4%}"
            )
            lines.append(
                f"  建议:             {comparison.get('recommendation', 'N/A')}"
            )
            lines.append("")

        # 6. 审计信息
        lines.append("─" * 50)
        lines.append("  6. 审计信息")
        lines.append("─" * 50)
        lines.append(f"  配置哈希:         {self.config_hash}")
        lines.append(f"  数据版本哈希:     {data_version_hash}")
        lines.append(f"  生成时间戳:       {self._timestamp}")
        lines.append("")

        # 7. 免责声明
        lines.append("─" * 50)
        lines.append("  7. 免责声明")
        lines.append("─" * 50)
        lines.append(f"  {_APPROX_MODE_DISCLAIMER}")
        lines.append("  本报告仅为量化研究用途, 非投资建议, 非交易指令.")
        lines.append("")
        lines.append(sep)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info("文本报告: %s", path)
        except Exception as e:
            logger.error("文本报告写入失败: %s", e)

    def _write_html(
        self,
        path: str,
        mode_name: str,
        result_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: dict[str, Any],
        signal_report: dict | None,
        comparison: dict | None,
        data_version_hash: str,
    ) -> None:
        """写入 HTML 格式报告 (含内嵌 CSS, 无外部依赖)."""
        try:
            nav_rows = ""
            if not result_df.empty and "nav" in result_df.columns:
                cols = ["date", "nav"] if "date" in result_df.columns else ["nav"]
                nav_data = result_df[cols].copy()
                if "date" in nav_data.columns:
                    nav_data["date"] = nav_data["date"].astype(str).str[:10]
                for _, row in nav_data.iterrows():
                    d = row.get("date", "") if "date" in nav_data.columns else ""
                    nav_rows += (
                        f"      <tr><td>{d}</td><td>{row['nav']:.2f}</td></tr>\n"
                    )

            trade_rows = ""
            if not trades_df.empty:
                recent = trades_df.tail(20)
                for _, t in recent.iterrows():
                    pnl_color = "green" if t.get("pnl", 0) >= 0 else "red"
                    trade_rows += (
                        f"      <tr><td>{str(t.get('exit_date', ''))[:10]}</td>"
                        f"<td>{t.get('stock', '')}</td>"
                        f"<td>{t.get('entry_price', 0):.2f}</td>"
                        f"<td>{t.get('exit_price', 0):.2f}</td>"
                        f"<td style='color:{pnl_color}'>{t.get('pnl_pct', 0):.2%}</td>"
                        f"<td>{t.get('exit_reason', '')}</td></tr>\n"
                    )

            signal_html = ""
            if signal_report:
                signal_html = "<h2>4. 信号评估</h2>\n"
                for hk, sr in signal_report.items():
                    signal_html += (
                        f"<h3>{hk.upper()}</h3>\n"
                        f"<table><tr><th>指标</th><th>值</th></tr>"
                        f"<tr><td>Rank IC 均值</td><td>{sr.get('rank_ic_mean', 0):.4f}</td></tr>"
                        f"<tr><td>Rank IR</td><td>{sr.get('rank_ir', 0):.4f}</td></tr>"
                        f"<tr><td>命中率</td><td>{sr.get('hit_rate', 0):.4%}</td></tr>"
                        f"<tr><td>成交量确认率</td><td>{sr.get('volume_confirmation_rate', 0):.4%}</td></tr>"
                        f"</table>\n"
                    )

            comparison_html = ""
            if comparison:
                comparison_html = (
                    "<h2>5. 对比分析</h2>\n"
                    f"<table><tr><th>指标</th><th>Squad</th><th>Sniper</th></tr>"
                    f"<tr><td>夏普比率</td><td>{comparison.get('squad_sharpe', 0):.4f}</td>"
                    f"<td>{comparison.get('sniper_sharpe', 0):.4f}</td></tr>"
                    f"<tr><td>最大回撤</td><td>{comparison.get('squad_max_dd', 0):.4%}</td>"
                    f"<td>{comparison.get('sniper_max_dd', 0):.4%}</td></tr>"
                    f"<tr><td>总收益率</td><td>{comparison.get('squad_total_return', 0):.4%}</td>"
                    f"<td>{comparison.get('sniper_total_return', 0):.4%}</td></tr>"
                    f"</table>\n"
                    f"<p><strong>建议:</strong> {comparison.get('recommendation', 'N/A')}</p>\n"
                )

            html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>回测报告 — {mode_name.upper()}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
  h2 {{ color: #283593; margin-top: 30px; }}
  h3 {{ color: #3949ab; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #e8eaf6; font-weight: bold; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .metric-card {{ display: inline-block; background: white; border-radius: 8px;
    padding: 15px; margin: 5px; min-width: 180px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .metric-value {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
  .metric-label {{ font-size: 12px; color: #666; }}
  .positive {{ color: #2e7d32; }}
  .negative {{ color: #c62828; }}
  .disclaimer {{ background: #fff3e0; border-left: 4px solid #ff9800;
    padding: 12px; margin: 20px 0; font-size: 14px; }}
  .audit {{ background: #e3f2fd; padding: 12px; border-radius: 4px;
    font-family: monospace; font-size: 12px; margin: 20px 0; }}
</style>
</head>
<body>
<h1>A股日线回测报告 — {mode_name.upper()}</h1>
<p>生成时间: {self._timestamp}</p>

<h2>1. 绩效概览</h2>
<div>
  <div class="metric-card"><div class="metric-value">{metrics.get("total_return", 0):.2%}</div><div class="metric-label">总收益率</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get("annual_return", 0):.2%}</div><div class="metric-label">年化收益率</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get("sharpe_ratio", 0):.2f}</div><div class="metric-label">夏普比率</div></div>
  <div class="metric-card"><div class="metric-value {("negative" if metrics.get("max_drawdown", 0) < 0 else "")}">{metrics.get("max_drawdown", 0):.2%}</div><div class="metric-label">最大回撤</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get("win_rate", 0):.2%}</div><div class="metric-label">胜率</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get("calmar_ratio", 0):.2f}</div><div class="metric-label">Calmar</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get("num_trades", 0)}</div><div class="metric-label">总交易数</div></div>
  <div class="metric-card"><div class="metric-value">{metrics.get("daily_tradeable_rate", 0):.2%}</div><div class="metric-label">日可交易率</div></div>
</div>

<h2>2. 交易明细 (最近20笔)</h2>
<table>
<tr><th>卖出日期</th><th>股票</th><th>买价</th><th>卖价</th><th>盈亏%</th><th>退出原因</th></tr>
{trade_rows}
</table>

<h2>3. 资金曲线</h2>
<table>
<tr><th>日期</th><th>NAV</th></tr>
{nav_rows}
</table>

{signal_html}
{comparison_html}

<div class="disclaimer">
  <strong>⚠ V5.0 Approximate Mode 免责声明</strong><br>
  {_APPROX_MODE_DISCLAIMER}<br>
  本报告仅为量化研究用途, 非投资建议, 非交易指令.
</div>

<div class="audit">
  <strong>审计信息</strong><br>
  配置哈希: {self.config_hash}<br>
  数据版本哈希: {data_version_hash}<br>
  生成时间戳: {self._timestamp}
</div>

</body>
</html>"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("HTML 报告: %s", path)
        except Exception as e:
            logger.error("HTML 报告写入失败: %s", e)

    @staticmethod
    def _trades_summary(trades_df: pd.DataFrame) -> dict[str, Any]:
        """生成交易明细摘要."""
        if trades_df is None or trades_df.empty:
            return {
                "total_trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "swap_trades": 0,
            }
        total = len(trades_df)
        swap = (
            int(trades_df.get("is_swap", pd.Series(dtype=int)).sum())
            if "is_swap" in trades_df.columns
            else 0
        )
        return {
            "total_trades": total,
            "buy_trades": total,  # 每条记录是一笔完整交易 (买+卖)
            "sell_trades": total,
            "swap_trades": swap,
        }

    @staticmethod
    def _nav_summary(result_df: pd.DataFrame) -> dict[str, Any]:
        """生成 NAV 曲线摘要."""
        if result_df is None or result_df.empty:
            return {"final_nav": 0.0, "num_days": 0, "max_nav": 0.0, "min_nav": 0.0}
        nav = result_df["nav"]
        return {
            "final_nav": float(nav.iloc[-1]),
            "num_days": len(result_df),
            "max_nav": float(nav.max()),
            "min_nav": float(nav.min()),
        }

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """递归将 numpy 类型转为 Python 原生类型, 以便 JSON 序列化."""
        if isinstance(obj, dict):
            return {k: ReportGenerator._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ReportGenerator._sanitize_for_json(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return obj
