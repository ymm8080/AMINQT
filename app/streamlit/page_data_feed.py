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


# 面板列名 → (数据源分类, 中文说明)
COLUMN_CN: dict[str, tuple[str, str]] = {
    # ── 基础行情 / 元数据 ──
    "symbol": ("基础行情", "股票代码 (6位数字)"),
    "date": ("基础行情", "交易日期"),
    "open": ("基础行情", "开盘价 (不复权)"),
    "high": ("基础行情", "最高价 (不复权)"),
    "low": ("基础行情", "最低价 (不复权)"),
    "close": ("基础行情", "收盘价 (不复权)"),
    "open_hfq": ("基础行情", "开盘价 (后复权)"),
    "high_hfq": ("基础行情", "最高价 (后复权)"),
    "low_hfq": ("基础行情", "最低价 (后复权)"),
    "close_hfq": ("基础行情", "收盘价 (后复权, 特征/收益计算用)"),
    "volume": ("基础行情", "成交量 (股或手, 视股票惯例)"),
    "amount": ("基础行情", "成交额 (元)"),
    "pre_close": ("基础行情", "昨收价"),
    "is_suspended": ("基础行情", "是否停牌"),
    "board": ("基础行情", "交易板块 (main=主板, dual=双创)"),
    "industry": ("基础行情", "所属行业"),
    "sw_l1_name": ("申万行业", "申万一级行业"),
    "sw_l2_name": ("申万行业", "申万二级行业"),
    "sw_l3_name": ("申万行业", "申万三级行业"),
    # ── 每日估值 daily_basic ──
    "free_float_turnover_rate": ("每日估值", "自由流通股换手率 (%)"),
    "turnover_rate": ("每日估值", "换手率 (%)"),
    "pe_ttm": ("每日估值", "市盈率 TTM"),
    "pb": ("每日估值", "市净率"),
    "ps_ttm": ("每日估值", "市销率 TTM"),
    "total_mv": ("每日估值", "总市值 (元)"),
    "circ_mv": ("每日估值", "流通市值 (元)"),
    "total_share": ("每日估值", "总股本 (股)"),
    "float_share": ("每日估值", "流通股本 (股)"),
    "free_share": ("每日估值", "自由流通股本 (股)"),
    "volume_ratio": ("每日估值", "量比"),
    "dv_ratio": ("每日估值", "股息率 (%)"),
    "dv_ttm": ("每日估值", "股息率 TTM (%)"),
    # ── 涨跌停价格 stk_limit ──
    "up_limit_raw": ("涨跌停价格", "涨停价"),
    "down_limit_raw": ("涨跌停价格", "跌停价"),
    # ── 筹码分布 CYQ ──
    "winner_ratio": ("筹码分布", "获利盘比例 (现价下方筹码占比)"),
    "pct_90_high": ("筹码分布", "90%成本区间上沿"),
    "pct_90_con": ("筹码分布", "90%筹码集中度"),
    "weight_avg": ("筹码分布", "筹码加权平均成本"),
    "cost_50pct": ("筹码分布", "50%筹码成本线 (中位成本)"),
    "cost_95pct": ("筹码分布", "95%筹码成本线 (高位成本)"),
    "peak_price": ("筹码分布", "筹码峰价位 (成本最集中价位)"),
    "chip_entropy": ("筹码分布", "筹码分布熵 (分散程度)"),
    "chip_skew_dist": ("筹码分布", "筹码分布偏度"),
    "peak_roc_5d": ("筹码分布", "筹码峰5日变化率"),
    "peak_roc_20d": ("筹码分布", "筹码峰20日变化率"),
    "cost_bias": ("筹码分布", "现价偏离中位成本幅度"),
    "conc_trend_20d": ("筹码分布", "筹码集中度20日趋势"),
    "conc_90_industry_rank": ("筹码分布", "90%集中度行业排名"),
    "chip_gini": ("筹码分布", "筹码基尼系数 (集中/均衡)"),
    "resistance_dist": ("筹码分布", "距上方压力位距离 (按现价归一)"),
    "support_dist": ("筹码分布", "距下方支撑位距离 (按现价归一)"),
    # ── 技术特征 (价量动能) ──
    "bias_5": ("技术特征", "5日乖离率 (收盘价偏离5日均线幅度)"),
    "bias_10": ("技术特征", "10日乖离率"),
    "bias_20": ("技术特征", "20日乖离率"),
    "bias_60": ("技术特征", "60日乖离率"),
    "bias_120": ("技术特征", "120日乖离率"),
    "bias_250": ("技术特征", "250日乖离率"),
    "bias_5_20_cross": ("技术特征", "5日/20日乖离率交叉信号 (金叉/死叉)"),
    "bias_20_60_cross": ("技术特征", "20日/60日乖离率交叉信号"),
    "ma_vol_ratio_5_20": ("技术特征", "量比 (5日均量/20日均量)"),
    "amplitude_5d": ("技术特征", "5日平均振幅"),
    "pctChg": ("技术特征", "涨跌幅 (%)"),
    "intraday_range": ("技术特征", "日内振幅 ((最高-最低)/昨收)"),
    "vol_surge": ("技术特征", "量能异动 ((当日量-20日均量)/20日标准差)"),
    "amt_surge": ("技术特征", "成交额异动 ((当日额-20日均额)/20日标准差)"),
    # ── 财务指标 fina_indicator ──
    "announce_date": ("财务指标", "财报公告日 (PIT, 财务数据可用日)"),
    "roe": ("财务指标", "净资产收益率 ROE"),
    "roe_deducted": ("财务指标", "扣非后 ROE"),
    "roa": ("财务指标", "总资产收益率 ROA"),
    "gross_margin": ("财务指标", "毛利率"),
    "rev_yoy": ("财务指标", "营业收入同比增速"),
    "debt_ratio": ("财务指标", "资产负债率"),
    "current_ratio": ("财务指标", "流动比率"),
    "asset_turnover": ("财务指标", "总资产周转率"),
    "ar_turnover": ("财务指标", "应收账款周转率"),
    "inventory_turnover": ("财务指标", "存货周转率"),
    "ocf_to_or": ("财务指标", "经营现金流/营业收入"),
    "net_margin": ("财务指标", "净利率"),
    "eps_yoy": ("财务指标", "每股收益同比增速"),
    "profit_yoy": ("财务指标", "净利润同比增速"),
    "ocfps": ("财务指标", "每股经营现金流"),
    "revenue_ps": ("财务指标", "每股营业收入"),
    "bps": ("财务指标", "每股净资产"),
    "eps": ("财务指标", "每股收益"),
    "dt_eps": ("财务指标", "稀释每股收益"),
    "roe_yoy": ("财务指标", "ROE 同比变化"),
    "q_roe": ("财务指标", "单季度 ROE"),
    "q_ocf_to_sales": ("财务指标", "单季度经营现金流/营业收入"),
    # ── 融资融券 margin ──
    "margin_balance": ("融资融券", "融资余额"),
    "short_balance": ("融资融券", "融券余额"),
    "margin_buy_amt": ("融资融券", "融资买入额"),
    "short_sell_vol": ("融资融券", "融券卖出量"),
    # ── 龙虎榜 lhb ──
    "lhb_net_buy": ("龙虎榜", "龙虎榜净买入额"),
    "lhb_buy_amt": ("龙虎榜", "龙虎榜买入总额"),
    "lhb_sell_amt": ("龙虎榜", "龙虎榜卖出总额"),
    "lhb_inst_buy": ("龙虎榜席位", "机构席位买入额"),
    "lhb_inst_sell": ("龙虎榜席位", "机构席位卖出额"),
    "lhb_top_buy": ("龙虎榜席位", "顶级游资席位买入额"),
    "lhb_top_sell": ("龙虎榜席位", "顶级游资席位卖出额"),
    "lhb_quant_buy": ("龙虎榜席位", "量化席位买入额"),
    "lhb_quant_sell": ("龙虎榜席位", "量化席位卖出额"),
    "lhb_retail_buy": ("龙虎榜席位", "散户席位买入额"),
    "lhb_retail_sell": ("龙虎榜席位", "散户席位卖出额"),
    # ── 股东增减持 holdertrade ──
    "sh_change_vol": ("股东增减持", "股东增减持变动股数"),
    "sh_change_amt_total": ("股东增减持", "股东增减持变动金额"),
    "sh_net_change_sign": ("股东增减持", "股东净增减持方向 (+增/-减)"),
    "sh_net_sign": ("股东增减持", "股东增减持方向标记 (+1增/-1减/0无)"),
    "sh_net_ratio": ("股东增减持", "股东净增减持比例"),
    "sh_g_ratio": ("股东增减持", "高管增减持比例"),
    "sh_p_ratio": ("股东增减持", "个人股东增减持比例"),
    "sh_c_ratio": ("股东增减持", "公司/法人增减持比例"),
    "sh_evt_start_date": ("股东增减持", "增减持事件开始日"),
    "sh_evt_end_date": ("股东增减持", "增减持事件结束日"),
    # ── 大宗交易 block_trade ──
    "bt_count": ("大宗交易", "大宗交易笔数"),
    "bt_disc_raw": ("大宗交易", "大宗交易折价率 (负向信号)"),
    "bt_inst_absorb": ("大宗交易", "机构接盘吸收度"),
    "bt_amt_ratio_float_mv": ("大宗交易", "大宗成交额/流通市值"),
    # ── 申万行业指数 sector_index ──
    "sw_index_close": ("申万行业", "申万行业指数收盘"),
    "sw_index_vol": ("申万行业", "申万行业指数成交量"),
    "sw_ret_1d": ("申万行业", "申万行业指数1日涨跌幅"),
}


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
            rows = []
            for col_name in info["columns"]:
                group, desc = COLUMN_CN.get(col_name, ("其他", "-"))
                is_na_high = any(
                    info["coverage"].get(src, {}).get("coverage_pct", 100) < 50
                    for src in SOURCE_MARKERS
                    if SOURCE_MARKERS[src] == col_name
                )
                rows.append(
                    {
                        "列名": f"⚠️ {col_name}" if is_na_high else col_name,
                        "中文说明": desc,
                        "数据源分类": group,
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "列名": st.column_config.TextColumn("列名", width="medium"),
                    "中文说明": st.column_config.TextColumn("中文说明", width="large"),
                    "数据源分类": st.column_config.TextColumn(
                        "数据源分类", width="small"
                    ),
                },
            )
            st.caption("⚠️ = 对应数据源覆盖率 < 50%，列数据可能缺失")

    # ---------- Tab 2: 数据获取 ----------
    with tab_fetch:
        if not _check_v3_exists():
            st.info("v3 面板尚不存在。首次运行将自动创建面板并填充数据。")
        _render_run_pipeline_tab()


if __name__ == "__main__":
    render()
