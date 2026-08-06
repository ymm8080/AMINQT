# -*- coding: utf-8 -*-
"""训练/预测档案 (只读) — 从持久化存储回看历史, 不现场重算.

数据源 (仓库外 / 落盘 WORM, 防 automation git 误删):
  - 模型档案: models/pipeline1/current_meta.json + *.pkl
  - 每日预测: PredictionDB (SQLite, DATA_OTHERS/data/predictions.db)
  - 回测历史: BACKTEST_RESULT_DIR (每趟=日期子目录, 含 backtest.json)
  - 落盘清单: STOCK_LIST_DIR (交付) + data/lists (仓库内, 易被清理)

本页全部为只读查询, 无任何重算/写入.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from config.settings import BACKTEST_RESULT_DIR, STOCK_LIST_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = "models/pipeline1"


def _safe(fn, default=None):
    """任一只读源失败都不让整页崩溃 (缺失目录 / 损坏文件 → 提示)."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("档案读取失败: %s", exc, exc_info=True)
        st.warning(f"数据源读取失败 (跳过): {exc}")
        return default


# ───────────────────────── 模型档案 ─────────────────────────
def _render_models() -> None:
    st.subheader("当前生效模型 (current_meta.json)")
    from app.pipeline1.model_meta import load_modules

    mods = _safe(load_modules)
    if mods:
        rows = [
            {
                "board": board,
                "tag": info.get("tag", ""),
                "file": info.get("file", ""),
                "updated": info.get("updated", ""),
            }
            for board, info in mods.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info(
            "current_meta.json 缺失或为空 — 尚无模型被提升为当前 (运行 train_predict_bt3y 后自动提升)"
        )

    st.subheader("模型包文件")
    if not os.path.isdir(MODEL_DIR):
        st.info(f"模型目录不存在: {MODEL_DIR}")
        return
    files = []
    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pkl"):
            continue
        p = os.path.join(MODEL_DIR, fname)
        s = os.stat(p)
        files.append(
            {
                "file": fname,
                "size_MB": round(s.st_size / 1e6, 1),
                "mtime": datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    if not files:
        st.info("模型目录为空")
        return
    st.dataframe(
        pd.DataFrame(files).sort_values("mtime", ascending=False),
        use_container_width=True,
    )


# ───────────────────────── 每日预测 ─────────────────────────
def _render_predictions() -> None:
    from app.pipeline1.prediction_db import PredictionDB

    db = PredictionDB()
    runs = _safe(db.list_runs, limit=60)
    if not runs:
        st.info("预测DB暂无运行记录")
        return
    st.subheader("运行历史")
    st.dataframe(pd.DataFrame(runs), use_container_width=True)

    st.subheader("预测质量趋势 (方向准确率)")
    qual = _safe(db.all_quality, limit=60)
    if qual:
        df = pd.DataFrame(qual).sort_values("date")
        st.line_chart(df.set_index("date")["direction_accuracy"])
        st.dataframe(df, use_container_width=True)

    st.subheader("单日详情")
    dates = list(runs[0]["date"]) if isinstance(runs[0], dict) else runs
    date_sel = st.selectbox(
        "选择运行日期", sorted(dates, reverse=True), key="archive_pred_date"
    )
    if date_sel:
        run = _safe(db.get_run, date_sel)
        if run and run.get("stocks"):
            st.dataframe(pd.DataFrame(run["stocks"]), use_container_width=True)


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
            st.markdown("**TOP-5 逐视界胜率 vs 基线**")
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


# ───────────────────────── 落盘清单 ─────────────────────────
def _list_dir(d: str, label: str) -> None:
    if not os.path.isdir(d):
        st.info(f"{label}: 目录不存在")
        return
    files = []
    for fname in sorted(os.listdir(d), reverse=True)[:50]:
        p = os.path.join(d, fname)
        if not os.path.isfile(p):
            continue
        s = os.stat(p)
        files.append(
            {
                "file": fname,
                "KB": round(s.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    if files:
        st.dataframe(pd.DataFrame(files), use_container_width=True)
    else:
        st.info(f"{label}: 暂无文件")


def _render_lists() -> None:
    st.subheader("交付清单 (STOCK_LIST_DIR, 仓库外持久)")
    _list_dir(str(STOCK_LIST_DIR), "STOCK LIST")
    st.subheader("data/lists (仓库内, 可能被 automation 清理)")
    _list_dir(os.path.join("data", "lists"), "data/lists")


# ───────────────────────── 页面 ─────────────────────────
def render() -> None:
    st.title("训练/预测档案 (只读)")
    st.caption(
        "从持久化存储读取历史训练/预测结果, 供存储与回看. "
        "数据源: 模型包 / 预测DB / 回测报告 / 落盘清单. 本页不做任何重算."
    )
    tab_models, tab_pred, tab_bt, tab_lists = st.tabs(
        ["模型档案", "每日预测", "回测历史", "落盘清单"]
    )
    with tab_models:
        _render_models()
    with tab_pred:
        _render_predictions()
    with tab_bt:
        _render_backtests()
    with tab_lists:
        _render_lists()


if __name__ == "__main__":
    render()
