# -*- coding: utf-8 -*-
"""
页面 4: 配置中心 (P10)
============================
- 规则引擎 Config 在线编辑 ([TUNABLE] 参数 + 边界提示)
- selection/trading YAML 配置编辑/保存/校验
- 调参报告查看 (tuning_report.json)
- 因子参考表 (V3.5 14 维度)
"""

from __future__ import annotations

import streamlit as st

from app.rules.config import TUNABLE_BOUNDS, Config

from . import data_service as ds

CONFIG_PATHS = {
    "selection": "config/selection_config.yaml",
    "trading": "config/trading_config.yaml",
}

FACTOR_DIMS = [
    ("① 价量动能", "MACD/RSI/KDJ/60日乖离/量价背离"),
    ("② 波动率", "ATR_pct / 布林带宽"),
    ("③ 基本面", "PE_log/PB/净利营收增速 (announce_date PIT)"),
    ("④ 板块效应", "板块涨停家数/板块收益 (历史快照)"),
    ("⑤ 筹码分布", "集中度/获利盘 (shift 1)"),
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
    tab_rules, tab_yaml, tab_report, tab_factors = st.tabs(
        ["规则参数", "YAML 配置", "调参报告", "因子参考"]
    )

    # ---------- Tab 1: 规则引擎 Config ----------
    with tab_rules:
        st.subheader("规则引擎参数 ([TUNABLE] 可回测调优)")
        st.caption(
            "初始值 = 用户规则预设; 边界 = TUNABLE_BOUNDS; "
            "调优流程见 回测中心 → 参数调优"
        )
        cfg = Config()
        cols = st.columns(3)
        for i, (name, (lo, hi, step)) in enumerate(sorted(TUNABLE_BOUNDS.items())):
            with cols[i % 3]:
                st.number_input(
                    f"{name} [{lo}~{hi}]",
                    value=float(getattr(cfg, name)),
                    key=f"cfg_{name}",
                    disabled=True,
                    help="P2 定稿后开放在线编辑; 当前经 回测中心→调参 写回",
                )
        st.info(
            "在线写回将在 Pipeline-2 定稿后开放; 当前流程: 回测中心调参 → tuning_report.json → apply_to_config"
        )

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
            st.json(report)

    # ---------- Tab 4: 因子参考 ----------
    with tab_factors:
        st.subheader("V3.5 特征维度 (14 维)")
        st.dataframe(
            {"维度": [d for d, _ in FACTOR_DIMS], "组成": [c for _, c in FACTOR_DIMS]},
            use_container_width=True,
        )
        st.caption(
            "因子集不固定: 每滚动重训窗口由 ICScreener 重算 (三标签并集 + 分类AUC), "
            "当期清单见 data/factor_registry/"
        )


def _to_yaml(data: dict) -> str:
    import yaml

    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False) if data else ""
