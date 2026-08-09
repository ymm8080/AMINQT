"""
页面 5: 数据管理 — v3 面板数据获取/刷新/状态监控
=====================================================
- 显示 v3 面板当前状态 (形状/日期范围/股票数)
- 各数据源覆盖率柱状图
- 日期范围选择器 + 数据源选择
- 触发数据获取管道
- 显示管道运行进度/结果
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.data_fetch_pipeline import (  # noqa: E402
    ALL_SOURCES,
    SOURCE_GROUPS,
    SOURCE_MARKERS,
    V3_PATH,
    compute_coverage,
    load_v3,
)

logger = logging.getLogger(__name__)

# 数据源分组 (UI 显示用)
SOURCE_GROUPS_CN = {k: f"{v}" for k, v in SOURCE_GROUPS.items()}

# 数据源颜色
SOURCE_COLORS = {
    "daily_basic": "#1f77b4",  # 蓝
    "stk_limit": "#ff7f0e",  # 橙
    "margin": "#2ca02c",  # 绿
    "northbound": "#d62728",  # 红
    "lhb": "#9467bd",  # 紫
    "fina_indicator": "#8c564b",  # 棕
    "holdertrade": "#7f7f7f",  # 灰
    "sector_index": "#bcbd22",  # 黄绿
    "cyq_tushare": "#17becf",  # 青
    "block_trade": "#e377c2",  # 粉
    "announcement": "#c5b0d5",  # 淡紫
}

# 覆盖率图补充源 (面板现算, 非数据获取管道源)
_EXTRA_COVERAGE = {
    "block_trade": ("bt_count", "大宗交易"),
    "announcement": ("announce_date", "公告数据"),
}

# 覆盖率图展示顺序 (不含股东户数)
_COVERAGE_ORDER = [
    "daily_basic",
    "stk_limit",
    "margin",
    "northbound",
    "lhb",
    "fina_indicator",
    "holdertrade",
    "cyq_tushare",
    "sector_index",
    "block_trade",
    "announcement",
]


def _check_v3_exists() -> bool:
    """检查 v3 面板是否存在."""
    return V3_PATH.exists()


def _build_coverage(v3: pd.DataFrame) -> dict:
    """面板覆盖率: 数据源源标记列 + 补充源 (大宗交易/公告) 现算."""
    cov = compute_coverage(v3)
    total = len(v3)
    for src, (marker, _label) in _EXTRA_COVERAGE.items():
        if marker in v3.columns:
            non_na = int(v3[marker].notna().sum())
            cov[src] = {
                "marker": marker,
                "non_na": non_na,
                "total": total,
                "coverage_pct": round(non_na / total * 100, 1),
                "has_data": non_na > 0,
            }
        else:
            cov[src] = {
                "marker": marker,
                "non_na": 0,
                "total": total,
                "coverage_pct": 0.0,
                "has_data": False,
                "missing": True,
            }
    return cov


def _get_panel_info() -> dict | None:
    """获取 v3 面板基本信息."""
    if not _check_v3_exists():
        return None
    try:
        v3 = load_v3()
        return {
            "shape": v3.shape,
            "symbols": int(v3["symbol"].nunique()),
            "date_min": v3["date"].min(),
            "date_max": v3["date"].max(),
            "columns": sorted(v3.columns.tolist()),
            "coverage": _build_coverage(v3),
        }
    except Exception as exc:
        st.error(f"读取 v3 面板失败: {exc}")
        return None


def _render_status_card(info: dict) -> None:
    """渲染面板状态卡片 (字体缩小, 仅限本卡片)."""
    st.markdown(
        """
        <style>
        .st-key-panel_overview [data-testid="stMetricValue"] { font-size: 1.5rem; }
        .st-key-panel_overview [data-testid="stMetricLabel"] { font-size: 0.8rem; }
        .st-key-panel_overview [data-testid="stMetricDelta"] { font-size: 0.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="panel_overview"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("股票数量", f"{info['symbols']:,}")
        with col2:
            st.metric("数据行数", f"{info['shape'][0]:,}")
        with col3:
            st.metric("特征列数", info["shape"][1])
        with col4:
            st.metric(
                "日期范围",
                f"{info['date_min'].strftime('%Y%m%d')} ~ {info['date_max'].strftime('%Y%m%d')}",
            )


def _render_coverage_chart(info: dict) -> None:
    """渲染覆盖率柱状图."""
    coverage = info.get("coverage", {})
    if not coverage:
        st.info("暂无覆盖率数据")
        return

    rows = []
    for src in _COVERAGE_ORDER:
        data = coverage.get(src)
        if data is None:
            continue
        label = SOURCE_GROUPS.get(src, _EXTRA_COVERAGE.get(src, (None, src))[1])
        rows.append(
            {
                "数据源": label,
                "覆盖率 (%)": data.get("coverage_pct", 0),
                "color": SOURCE_COLORS.get(src, "#636efa"),
            }
        )
    df = pd.DataFrame(rows)

    fig = px.bar(
        df,
        x="数据源",
        y="覆盖率 (%)",
        color="数据源",
        color_discrete_map={
            SOURCE_GROUPS.get(src, _EXTRA_COVERAGE.get(src, (None, src))[1]): color
            for src, color in SOURCE_COLORS.items()
        },
        text="覆盖率 (%)",
        height=400,
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, 105],
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 80},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_coverage_detail(info: dict) -> None:
    """渲染覆盖率明细表."""
    coverage = info.get("coverage", {})
    if not coverage:
        return

    rows = []
    for src in _COVERAGE_ORDER:
        data = coverage.get(src)
        if data is None:
            continue
        rows.append(
            {
                "数据源": SOURCE_GROUPS.get(
                    src, _EXTRA_COVERAGE.get(src, (None, src))[1]
                ),
                "标识列": data.get("marker", "-"),
                "非空行数": f"{data.get('non_na', 0):,}",
                "总行数": f"{data.get('total', 0):,}",
                "覆盖率": f"{data.get('coverage_pct', 0):.1f}%",
                "状态": "✅ 正常"
                if data.get("coverage_pct", 0) > 50
                else "⚠️ 不足"
                if data.get("coverage_pct", 0) > 0
                else "❌ 缺失",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _run_pipeline(
    start_date: str,
    end_date: str,
    sources: list[str],
    refresh: bool,
    progress_callback: callable,
) -> dict:
    """在子线程中运行数据获取管道."""
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "data_fetch_pipeline.py"),
        "--start",
        start_date,
        "--end",
        end_date,
    ]
    if sources:
        cmd.extend(["--sources"] + sources)
    if refresh:
        cmd.append("--refresh")

    # 运行管道并捕获输出
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 小时超时
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "管道运行超时 (1小时)",
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
        }


def _render_run_pipeline_tab() -> None:
    """渲染运行管道页签."""
    st.subheader("运行数据获取管道")

    # 日期范围
    today = datetime.now(timezone(timedelta(hours=8)))
    default_start = today - timedelta(days=365 * 3)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "起始日期",
            value=default_start,
            min_value=datetime(2000, 1, 1),
            max_value=today,
        )
    with col2:
        end_date = st.date_input(
            "截止日期",
            value=today,
            min_value=datetime(2000, 1, 1),
            max_value=today,
        )

    if start_date > end_date:
        st.error("起始日期不能晚于截止日期")
        return

    # 数据源选择
    st.subheader("选择数据源")
    source_options = {}
    for src in ALL_SOURCES:
        label = SOURCE_GROUPS.get(src, src)
        source_options[src] = st.checkbox(label, value=True, key=f"src_{src}")

    selected_sources = [src for src, checked in source_options.items() if checked]
    if not selected_sources:
        st.warning("请至少选择一个数据源")
        return

    # 高级选项
    with st.expander("高级选项"):
        refresh = st.checkbox("强制刷新缓存 (重新拉取所有数据)", value=False)
        st.checkbox("保存前备份旧文件", value=True)

    # 运行按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    run_clicked = col1.button(
        "🚀 运行数据获取",
        type="primary",
        use_container_width=True,
        disabled=not selected_sources,
    )
    report_clicked = col2.button(
        "📊 仅查看覆盖率报告",
        use_container_width=True,
    )

    # 运行结果
    if run_clicked or report_clicked:
        with st.spinner("管道运行中..." if run_clicked else "生成报告中..."):
            progress_bar = st.progress(0, text="正在运行...")
            status_text = st.empty()

            # 模拟进度 (子进程无法直接获取实时进度)
            def _update_progress():
                for i in range(1, 100):
                    time.sleep(0.1)
                    progress_bar.progress(i, text=f"运行中... {i}%")

            import time

            if run_clicked:
                # 启动进度动画
                progress_thread = threading.Thread(target=_update_progress, daemon=True)
                progress_thread.start()

                result = _run_pipeline(
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    sources=selected_sources,
                    refresh=refresh,
                    progress_callback=None,
                )

                progress_bar.progress(100, text="完成")
                status_text.success("管道运行完成!")

                # 显示结果
                with st.expander("查看输出日志", expanded=True):
                    if result["stdout"]:
                        st.text(result["stdout"][:5000])
                    if result["stderr"]:
                        st.error(result["stderr"][:2000])
                    if result["returncode"] != 0:
                        st.error(f"管道返回错误码: {result['returncode']}")
                    else:
                        st.success("管道运行成功")

                # 刷新页面状态
                st.rerun()
            else:
                # 仅报告
                panel = load_v3()
                coverage = compute_coverage(panel)
                _render_coverage_detail({"coverage": coverage})


def render() -> None:
    st.header("数据管理 · v3 面板数据获取")

    tab_status, tab_fetch = st.tabs(["📊 面板状态", "🚀 数据获取"])

    # ---------- Tab 1: 面板状态 ----------
    with tab_status:
        if not _check_v3_exists():
            st.warning("v3 面板不存在! 请先运行数据获取管道生成面板。")
            return

        info = _get_panel_info()
        if info is None:
            return

        st.subheader("面板概览")
        _render_status_card(info)

        st.divider()
        st.subheader("数据源覆盖率")
        _render_coverage_chart(info)

        with st.expander("查看覆盖率明细"):
            _render_coverage_detail(info)

        with st.expander("查看所有列名"):
            cols_per_row = 4
            for i in range(0, len(info["columns"]), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col_name in enumerate(info["columns"][i : i + cols_per_row]):
                    with cols[j]:
                        is_na_high = any(
                            info["coverage"].get(src, {}).get("coverage_pct", 100) < 50
                            for src in SOURCE_MARKERS
                            if SOURCE_MARKERS[src] == col_name
                        )
                        if is_na_high:
                            st.markdown(f"⚠️ `{col_name}`")
                        else:
                            st.markdown(f"`{col_name}`")

    # ---------- Tab 2: 数据获取 ----------
    with tab_fetch:
        if not _check_v3_exists():
            st.info("v3 面板尚不存在。首次运行将自动创建面板并填充数据。")
        _render_run_pipeline_tab()


if __name__ == "__main__":
    render()
