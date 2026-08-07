"""
页面 4: 配置中心 (P10)
============================
- 规则引擎 Config 在线编辑 ([TUNABLE] 参数可编辑 + 边界提示) [E1]
- selection/trading/backtest YAML 配置编辑/保存/校验 [E2]
- 调参报告查看 (tuning_report.json)
- 因子参考表 (V3.5 14 维 + 个股公告因子)
- 一键应用调参结果到规则引擎/V5.2配置 [F3]
"""

from __future__ import annotations

import streamlit as st

from app.rules.config import TUNABLE_BOUNDS, Config

from . import data_service as ds

CONFIG_PATHS = {
    "selection": "config/selection_config.yaml",
    "trading": "config/trading_config.yaml",
    "backtest": "config/backtest_config.yaml",
}

FACTOR_DIMS = [
    ("① 价量动能", "MACD/RSI/KDJ/60日乖离/量价背离"),
    ("② 波动率", "ATR_pct / 布林带宽"),
    ("③ 基本面", "PE_log/PB/净利营收增速 (announce_date PIT)"),
    ("④ 板块效应", "板块涨停家数/板块收益 (历史快照)"),
    ("⑤ 筹码分布", "集中度/获利盘 (shift 1)"),
    ("⑥ 个股公告因子", "announce_score: 公告/业绩预告/解禁/分红等事件评分"),
    ("⑦ 涨停基因", "10/20日涨停天数/炸板率/连板高度0-4"),
    ("⑧ 日历-月份", "月份分类"),
    ("⑨ 自定义公式", "4 同花顺公式 (已审计, NECESSARY INDICATOR 复刻)"),
    ("⑩ 资金流", "主力净流入/超大单 (shift 1, 单一数据源)"),
    ("⑪ 连板/清单", "is_in_yesterday_list (Holding Bonus)"),
    ("⑫ 均线系统", "5/10/20/60/120/250 距离 + 排列"),
    ("⑬ 日历-长假", "days_to/after_holiday, is_pre/post"),
    ("⑭ 全市场情绪", "两市成交额 + 5d/20d 比值 + 涨跌停家数"),
]


def render() -> None:
    st.header("配置中心")
    tab_rules, tab_yaml, tab_report, tab_factors, tab_apply = st.tabs(
        ["规则参数", "YAML 配置", "调参报告", "因子参考", "应用调参"]
    )

    # ---------- Tab 1: 规则引擎 Config (可编辑) ----------
    with tab_rules:
        st.subheader("规则引擎参数 ([TUNABLE] 可回测调优)")
        st.caption("可直接编辑后保存, 或通过 回测中心 → 参数调优 自动写回")
        cfg = Config()
        cols = st.columns(3)
        edited_values = {}
        for i, (name, (lo, hi, step)) in enumerate(sorted(TUNABLE_BOUNDS.items())):
            with cols[i % 3]:
                edited_values[name] = st.number_input(
                    f"{name} [{lo}~{hi}]",
                    value=float(getattr(cfg, name)),
                    key=f"cfg_{name}",
                    disabled=False,
                    help=f"可编辑; 范围 [{lo}, {hi}] 步长 {step}",
                )
        col_save, col_reset = st.columns(2)
        if col_save.button("💾 保存参数到 Config", type="primary", key="save_rules"):
            try:
                cfg2 = Config()
                for name, val in edited_values.items():
                    if name in TUNABLE_BOUNDS and hasattr(cfg2, name):
                        setattr(cfg2, name, type(getattr(cfg2, name))(val))
                st.success("规则参数已写入 Config (内存级)")
                st.caption("持久化需在 YAML 配置 Tab 手动保存 selection/trading 配置")
            except Exception as exc:
                st.error(f"保存失败: {exc}")
        if col_reset.button("🔄 重置为默认值", key="reset_rules"):
            st.rerun()

    # ---------- Tab 2: YAML 配置 ----------
    with tab_yaml:
        for label, path in CONFIG_PATHS.items():
            with st.expander(f"{label}: {path}"):
                data = ds.load_yaml(path)
                text = st.text_area(
                    "YAML", value=_to_yaml(data), height=240, key=f"yaml_{label}"
                )
                if st.button("保存", key=f"save_{label}"):
                    try:
                        import yaml

                        ds.save_yaml(yaml.safe_load(text), path)
                        st.success("已保存")
                    except Exception as exc:
                        st.error(f"YAML 校验失败: {exc}")

    # ---------- Tab 3: 调参报告 ----------
    with tab_report:
        report = ds.load_tuning_report()
        if report is None:
            st.info("暂无调参报告 — 在 回测中心 执行 参数调优 后生成")
        else:
            _render_tuning_report(report)

    # ---------- Tab 4: 因子参考 ----------
    with tab_factors:
        st.subheader("V3.5 特征维度 (14 维 + 公告因子)")
        st.dataframe(
            {"维度": [d for d, _ in FACTOR_DIMS], "组成": [c for _, c in FACTOR_DIMS]},
            use_container_width=True,
        )
        st.caption(
            "因子集不固定: 每滚动重训窗口由 ICScreener 重算 (三标签并集 + 分类AUC), "
            "当期清单见 data/factor_registry/"
        )

    # ---------- Tab 5: 应用调参结果 (F3) ----------
    with tab_apply:
        _render_apply_tab()


def _to_yaml(data: dict) -> str:
    import yaml

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False) if data else ""


def _render_tuning_report(report: dict) -> None:
    """用用户能理解的语言展示调参报告."""
    best = report.get("best_params", {})
    fallback = report.get("fallback_to_default", False)
    train = report.get("train_score", 0.0)
    oos = report.get("oos_score", 0.0)

    st.subheader("调参结论")
    if fallback:
        st.warning(
            "⚠️ 调参结果在样本外 (OOS) 表现不如默认参数, 系统已自动回退到默认值。"
            "建议不要硬调参数, 优先检查数据质量或特征是否有泄漏。"
        )
    else:
        st.success("✅ 调参结果在样本外 (OOS) 验证通过, 推荐参数如下。")

    if best:
        st.markdown("**推荐参数:**")
        for k, v in best.items():
            st.markdown(f"- `{k}`: {v}")
    else:
        st.markdown("**推荐参数: 使用默认值**")

    st.markdown(f"**训练段评分:** {train:+.2%} | **样本外 (OOS) 评分:** {oos:+.2%}")
    st.caption(
        "训练段评分 = 用历史数据训练时表现最好的参数组合; "
        "OOS 评分 = 用最近一段未参与调参的数据验证, 防止过拟合。"
    )

    leaderboard = report.get("leaderboard", [])
    if leaderboard:
        with st.expander("查看 TOP 参数组合明细"):
            for i, (params, score) in enumerate(leaderboard, 1):
                st.markdown(f"**第 {i} 名** 训练段评分 {score:+.2%}")
                st.json(params)

    st.json(report)


def _render_apply_tab() -> None:
    """一键应用调参结果到规则引擎 Config 和 V5.2 BacktestConfig."""
    st.subheader("一键应用调参结果")
    st.caption("从 tuning_report.json 读取最佳参数, 写回规则引擎 Config")

    report = ds.load_tuning_report()
    if report is None:
        st.info("暂无调参报告 — 请先在 回测中心 → 参数调优 执行")
        return

    if report.get("fallback_to_default"):
        st.warning(
            "⚠ 上次调参结果在 OOS 验证未通过, 已回退默认值。"
            "建议检查数据质量或特征是否有泄漏, 不要硬调参数。"
        )
    else:
        st.success("✅ 上次调参结果在 OOS 验证通过, 推荐参数如下:")

    best = report.get("best_params", {})
    if best:
        st.markdown("**推荐参数:**")
        for k, v in best.items():
            st.markdown(f"- `{k}`: {v}")
    else:
        st.markdown("**推荐参数: 使用默认值**")
    st.caption(
        f"训练段评分: {report.get('train_score', 0):+.4f} | "
        f"OOS 评分: {report.get('oos_score', 0):+.4f} | "
        f"引擎: {report.get('engine', 'v35')}"
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 写回规则引擎 Config", type="primary", key="apply_rules"):
            try:
                from app.pipeline1.param_tuner import ParamTuner

                cfg = Config()
                ParamTuner.apply_to_config(best, cfg)
                st.success("已写入规则引擎 Config (内存级)")
                st.caption("持久化需在 YAML 配置 Tab 手动保存")
            except Exception as exc:
                st.error(f"写回失败: {exc}")
    with col2:
        if st.button("✅ 写回 V5.2 BacktestConfig", key="apply_v52"):
            try:
                from app.backtest.config_manager import ConfigManager
                from app.pipeline1.param_tuner import ParamTuner

                v52_config = ConfigManager.load("config/backtest_config.yaml")
                ParamTuner.apply_to_v52_config(best, v52_config)
                st.success("已写入 V5.2 BacktestConfig (内存级)")
                st.caption("持久化需在 YAML 配置 Tab 保存 backtest 配置")
            except Exception as exc:
                st.error(f"写回失败: {exc}")


if __name__ == "__main__":
    render()
