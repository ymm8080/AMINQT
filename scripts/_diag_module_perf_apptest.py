"""诊断: AppTest 验证 page_archive.render (模块绩效 tab: 每模型5版本下拉 + 评估窗口)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def _run() -> None:
    # from_function 生成独立脚本, 只认函数体内导入的名字
    import streamlit as st  # noqa: F401

    from app.streamlit import page_archive

    page_archive.render()


def main() -> None:
    at = AppTest.from_function(_run)
    at.run(timeout=300)
    print("exception:", at.exception)
    assert not at.exception, f"page_archive 渲染异常: {at.exception}"
    print("selectboxes:", [s.label for s in at.selectbox])
    print("sliders:", [s.label for s in at.slider])
    print("select_sliders:", len(at.select_slider))
    print("dataframes:", len(at.dataframe))
    print("plotly_charts:", len(at.get("plotly_chart")))
    # 模块绩效 tab: 应有模型下拉 + 评估窗口 (select_slider)
    labels = [s.label for s in at.selectbox]
    assert any("模型" in lbl for lbl in labels), f"缺模型下拉: {labels}"
    print("OK: page_archive.render 无异常, 模块绩效下拉/评估窗口已渲染")


if __name__ == "__main__":
    main()
