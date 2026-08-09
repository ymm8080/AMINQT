"""训练/预测档案 (只读) — 从持久化存储回看历史, 不现场重算.

数据源 (仓库外 / 落盘 WORM, 防 automation git 误删):
  - 回测历史: BACKTEST_RESULT_DIR (每趟=日期子目录, 含 backtest.json)
  - 交付清单: STOCK_LIST_DIR (预测/短名单交付)
  - 模块绩效: 交付清单 × V3 面板 close_hfq → 已实现收益/命中率

本页只读: 模块绩效按交付清单+收盘价现算, 其余纯回看, 不重算训练/不写盘.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH

logger = logging.getLogger(__name__)


def _safe(fn, default=None, **kwargs):
    """任一只读源失败都不让整页崩溃 (缺失目录 / 损坏文件 → 提示)."""
    try:
        return fn(**kwargs)
    except Exception as exc:
        logger.warning("档案读取失败: %s", exc, exc_info=True)
        st.warning(f"数据源读取失败 (跳过): {exc}")
        return default


# ───────────────────────── 回测历史 ─────────────────────────
def _render_backtests() -> None:
    from app.streamlit import bt_report as btr

    base = BACKTEST_RESULT_DIR
    if not base.is_dir():
        st.info(f"回测目录不存在: {base}")
        return
    all_runs = btr.list_runs(str(base))
    runs = [r for r in all_runs if (base / r["ts"] / "backtest.json").is_file()]
    if not runs:
        st.info("回测目录暂无含 backtest.json 的 run")
        if all_runs:
            st.dataframe(pd.DataFrame(all_runs), use_container_width=True)
        return
    st.subheader("回测 run 列表")
    st.dataframe(pd.DataFrame(runs), use_container_width=True)

    sel_ts = st.selectbox("选择回测 run", [r["ts"] for r in runs], key="archive_bt_run")
    if not sel_ts:
        return
    d = btr.load_run_json(sel_ts, str(base))
    if d is None:
        st.warning("backtest.json 读取失败或损坏")
        return
    _render_bt_run(d, btr)
    with st.expander("原始 backtest.json"):
        st.json(d)


def _render_bt_run(d: dict, btr) -> None:
    """结构化渲染单个回测 run (结论/逐视界/系统/个股), 旧 schema 防御."""
    boards = btr.list_boards(d)
    if not boards:
        st.info("该 run 无板块数据")
        return
    concl = btr.parse_conclusion_summary(d)
    if not concl.empty:
        st.subheader("结论判定 (cuts)")
        st.dataframe(concl, use_container_width=True)
    for board in boards:
        st.subheader(f"板块: {board}")
        ov = btr.parse_board_overview(d, board)
        cuts = ov.get("cuts")
        impr = ov.get("improvements")
        if isinstance(cuts, dict) or impr:
            col1, col2 = st.columns(2)
            if isinstance(cuts, dict):
                with col1:
                    st.markdown("**Cuts 判定**")
                    for cut, c in cuts.items():
                        if not isinstance(c, dict):
                            continue
                        tag = "保留" if c.get("kept") else "剔除"
                        st.write(
                            f"{cut} → {tag} (wr {c.get('winrate')} / mag {c.get('mag')})"
                        )
            if impr:
                with col2:
                    st.markdown("**改进建议**")
                    for line in impr:
                        st.write(f"- {line}")
        ph = btr.parse_per_horizon(d, board, "top5")
        if not ph.empty:
            st.markdown("**TOP-5 逐视界胜率 vs 基线 (OOS)**")
            st.bar_chart(
                ph.set_index("horizon")[["winrate", "base_winrate"]], height=280
            )
        sysdf = btr.parse_systems(d, board)
        if not sysdf.empty:
            st.markdown("**系统对比 (OOS.6m)**")
            st.dataframe(sysdf, use_container_width=True)
        picks = btr.parse_picks(d, board)
        if not picks.empty:
            st.markdown("**近日入选个股**")
            st.dataframe(picks, use_container_width=True)


# ───────────────────────── 个股预测查询 / 日期清单 ─────────────────────────
def _fmt_gain(v) -> str:
    if v is None or pd.isna(v):
        return ""
    try:
        return f"{float(v):+.1%}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_prob(v) -> str:
    if v is None or pd.isna(v):
        return ""
    try:
        return f"{float(v):.0%}"
    except (ValueError, TypeError):
        return str(v)


def _render_stock_query() -> None:
    """输入股票代码 (可多只) → 最近 5 个交易日预测 + 概率 (含模块来源)."""
    from app.streamlit import data_service as ds

    st.subheader("个股预测查询 (最近 5 个交易日)")
    st.caption(
        "数据源: STOCK_LIST_DIR 预测文件 (文件名带 日期+模块 双标识). "
        "支持同时输入多只, 用 逗号/空格 分隔. "
        "模块列 = 来源系统·模型标签; 系统列 = parallel 的融合/狙击子系统. "
        "legacy `prob_up`=上涨概率, `pred_ret`=预期涨幅; "
        "parallel `pred_prob`=达到概率, `pred_mag`=预期幅度. "
        "`na` = 无模块标签 (早期产出)."
    )
    raw = st.text_input(
        "股票代码 (多只用 逗号/空格 分隔)",
        value="600519, 001283, 300750",
        key="stock_query_symbol",
    )
    symbols = [s.strip() for s in re.split(r"[,，\s]+", raw) if s.strip()]
    if not symbols:
        st.info("请输入至少一个 6 位股票代码")
        return
    frames, missing = [], []
    for sym in symbols:
        hist = _safe(ds.load_stock_prediction_history, symbol=sym)
        if hist is None or hist.empty:
            missing.append(sym)
        else:
            frames.append(hist)
    if not frames:
        st.info(
            f"以下代码最近 5 个交易日无预测记录 (仅列出被预测/入选的股票): {', '.join(missing)}"
        )
        return
    hist = pd.concat(frames, ignore_index=True)
    disp = pd.DataFrame()
    disp["代码"] = hist["symbol"]
    disp["日期"] = hist["date"]
    disp["模块"] = hist["family"] + "·" + hist["module"]
    disp["板块"] = hist["board"].fillna("")
    disp["系统"] = hist["system"].fillna("")
    disp["rk"] = hist["rk"]
    disp["score"] = hist["score"]
    for h in ("3d", "5d", "10d"):
        disp[f"涨{h}"] = hist[f"gain_{h}"].map(_fmt_gain)
        disp[f"概率{h}"] = hist[f"prob_{h}"].map(_fmt_prob)
    disp = disp.sort_values(["日期", "代码"], ascending=[False, True])
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        f"共 {len(hist)} 条预测 ({hist['symbol'].nunique()} 只股票 · "
        f"{hist['date'].nunique()} 个交易日 · {hist['family'].nunique()} 类模块). "
        "同日期不同模块 = 各系统对同一股票各自的预测."
    )
    if missing:
        st.caption(f"以下代码无预测记录: {', '.join(missing)}")


def _render_date_list() -> None:
    """按日期看交付清单预测明细 — 日期/模块均可多选, 选择即自动刷新."""
    from app.streamlit import data_service as ds

    st.subheader("股票清单预测明细 (按日期)")
    st.caption(
        "选择日期 (可多选) 与模块 (可多选), 数据随选择自动刷新. "
        "模块列 = 来源系统·模型标签; 系统列 = parallel 的融合/狙击子系统. "
        "legacy `prob_up`=上涨概率 / `pred_ret`=预期涨幅; "
        "parallel `pred_prob`=达到概率 / `pred_mag`=预期幅度."
    )
    avail = _safe(ds.list_prediction_dates)
    if not avail:
        st.info("STOCK_LIST_DIR 暂无预测文件")
        return
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    default_dates = [today] if today in avail else [avail[0]]
    if today not in avail:
        st.caption(f"今天 {today} 暂无预测文件, 已选最近可用日期 {avail[0]}")
    sel_dates = st.multiselect(
        "选择日期 (可多选)",
        avail,
        default=default_dates,
        key="archive_date_multi",
    )
    if not sel_dates:
        st.info("未选择日期, 无数据显示")
        return
    df = _safe(ds.load_stock_list_on_dates, dates=sel_dates)
    if df is None or df.empty:
        st.info("所选日期均无预测记录")
        return
    df["_mod"] = df["family"] + "·" + df["module"].astype(str)
    mod_opts = sorted(df["_mod"].unique())
    # key 绑定所选日期: 换日期即重建模块选项并默认全选, 新日期数据立即可见
    chosen = st.multiselect(
        "模块筛选 (可多选)",
        mod_opts,
        default=mod_opts,
        key="archive_module_multi_" + "_".join(sorted(sel_dates)),
    )
    if not chosen:
        st.info("未选择任何模块, 无数据显示")
        return
    df = df[df["_mod"].isin(chosen)].drop(columns=["_mod"])
    disp = pd.DataFrame()
    disp["日期"] = df["date"]
    disp["代码"] = df["symbol"]
    disp["模块"] = df["family"] + "·" + df["module"]
    disp["板块"] = df["board"].fillna("")
    disp["系统"] = df["system"].fillna("")
    disp["rk"] = df["rk"]
    disp["score"] = df["score"]
    for h in ("3d", "5d", "10d"):
        disp[f"涨{h}"] = df[f"gain_{h}"].map(_fmt_gain)
        disp[f"概率{h}"] = df[f"prob_{h}"].map(_fmt_prob)
    disp = disp.sort_values(["日期", "模块", "rk"], na_position="last")
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(
        f"共 {len(df)} 条预测 ({df['date'].nunique()} 个交易日 · "
        f"{df['symbol'].nunique()} 只股票 · {df['family'].nunique()} 类模块). "
        "同日不同模块 = 各系统各自的预测."
    )


@st.cache_data(show_spinner=False, ttl=600)
def _load_close_panel() -> pd.DataFrame | None:
    """只读 V3 面板 date/symbol/close_hfq 三列 (已实现收益对齐, 避免整面板 OOM)."""
    if not os.path.exists(PANEL_V3_PATH):
        return None
    try:
        return pd.read_parquet(PANEL_V3_PATH, columns=["date", "symbol", "close_hfq"])
    except Exception:
        logger.warning(
            "面板收盘列读取失败 (跳过已实现收益): %s", PANEL_V3_PATH, exc_info=True
        )
        return None


def _render_module_perf() -> None:
    """模块绩效: 只交付短名单, 下拉选模型 → 该模型已实现收益/命中率 (只读)."""
    from app.pipeline1.model_meta import load_modules
    from app.pipeline1.model_meta import module_id as current_module_id
    from app.streamlit import module_perf as mp

    picks = mp.load_module_picks()
    if picks is None or picks.empty:
        st.info("暂无预测清单数据 (STOCK_LIST_DIR 为空)")
        return
    panel = _load_close_panel()
    if panel is None or panel.empty:
        st.info("缺少 V3 面板收盘数据, 无法计算已实现收益")
        return
    realized = mp.compute_realized_returns(picks, panel)
    if realized is None or realized.empty or "real_3d" not in realized.columns:
        st.info("已实现收益不可用 (清单日期太新 / 面板缺失)")
        return

    # 只交付短名单 (legacy/parallel/slow_bull), 不含全市场底稿
    delivered = mp.filter_scope(realized, "交付短名单")
    if delivered is None or delivered.empty:
        st.info("交付短名单无数据")
        return

    # 当前模型 (current_meta.json), 在交付清单里标记 (当前)
    mods = load_modules()
    cur_tag = current_module_id(mods) if mods else None
    if cur_tag:
        st.caption(f"当前模型: {cur_tag}")
        cur_ids = sorted(
            delivered.loc[
                delivered["module"].astype(str) == cur_tag, "module_id"
            ].unique()
        )
    else:
        st.caption("当前模型: 未设置 (current_meta.json 缺失或损坏)")
        cur_ids = []

    # 模型下拉: 每个模型族最近 5 个版本 + 当前模型置顶
    options = mp.recent_module_ids_per_model(delivered, n=5)
    for cid in cur_ids:
        if cid not in options:
            options.insert(0, cid)
    if not options:
        st.info("暂无可用模型 (交付清单无模块标签)")
        return
    default_idx = options.index(cur_ids[0]) if cur_ids else 0
    sel = st.selectbox(
        "选择模型",
        options,
        index=default_idx,
        format_func=lambda mid: f"{mid} (当前)" if mid in cur_ids else mid,
    )

    # 评估窗口 (选股日): 对所选模型一致应用, 保证跨模型可比
    avail_dates = sorted(delivered["date"].dropna().unique())
    if len(avail_dates) > 1:
        w_start, w_end = st.select_slider(
            "评估窗口 (选股日)",
            options=avail_dates,
            value=(avail_dates[0], avail_dates[-1]),
            key="mp_eval_window",
        )
        delivered = delivered[
            (delivered["date"] >= w_start) & (delivered["date"] <= w_end)
        ]
        st.caption(
            f"评估窗口: {w_start} ~ {w_end} (共 {delivered['date'].nunique()} 个选股日)"
        )
    else:
        st.caption(
            f"评估窗口: {avail_dates[0] if avail_dates else '无'} (仅 1 个选股日)"
        )

    topk = st.selectbox(
        "Top-N (按每日模块内排名)", [5, 10, 20, "全部"], index=1, key="mp_topk"
    )
    n = None if topk == "全部" else int(topk)
    sub = mp.top_k(delivered[delivered["module_id"] == sel].copy(), n)
    if sub is None or sub.empty:
        st.info("该模型无已实现收益数据")
        return

    rows = []
    for h in mp.HORIZONS:
        summary = mp.perf_summary(sub, h)
        if summary is None or summary.empty:
            continue
        r = summary.iloc[0][["n", "hit_rate", "mean", "median", "p10", "p90"]].to_dict()
        r["horizon"] = h
        rows.append(r)
    if not rows:
        st.info("该模型清单日期太新, 暂无已实现收益")
        return
    out = pd.DataFrame(rows)[
        ["horizon", "n", "hit_rate", "mean", "median", "p10", "p90"]
    ]
    out["horizon"] = out["horizon"].map({"3d": "T+3", "5d": "T+5", "10d": "T+10"})
    out["hit_rate"] = out["hit_rate"].map(lambda v: f"{v:.1%}")
    for c in ("mean", "median", "p10", "p90"):
        out[c] = out[c].map(lambda v: f"{v:+.2%}")
    st.subheader(f"{sel} 已实现收益 / 命中率")
    st.dataframe(out, use_container_width=True, hide_index=True)


# ───────────────────────── 页面 ─────────────────────────
def render() -> None:
    st.title("训练/预测档案 (只读)")
    st.caption(
        "从持久化存储读取历史训练/预测结果, 供存储与回看. "
        "模块绩效按落盘清单+收盘价现算, 其余只读回看."
    )
    (
        tab_module_perf,
        tab_bt,
        tab_query,
        tab_datelist,
    ) = st.tabs(
        [
            "模块绩效",
            "回测历史",
            "个股预测查询",
            "日期清单",
        ]
    )
    with tab_module_perf:
        _render_module_perf()
    with tab_bt:
        _render_backtests()
    with tab_query:
        _render_stock_query()
    with tab_datelist:
        _render_date_list()


if __name__ == "__main__":
    render()
