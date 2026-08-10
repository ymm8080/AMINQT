"""诊断: AppTest 验证 预测评估中心 个股预测 UI (股票输入 + 预测基准日)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def _run() -> None:
    import streamlit as st  # noqa: F401

    from app.streamlit import page_eval

    page_eval.render()


def main() -> None:
    at = AppTest.from_function(_run)
    at.run(timeout=300)
    assert not at.exception, f"page_eval 渲染异常: {at.exception}"
    print("exception:", at.exception)

    # 1. 预测模式 radio + 使用最新数据 checkbox 存在
    radios = [r.label for r in at.radio]
    print("radios:", radios)
    assert any("预测模式" in r for r in radios), f"缺预测模式: {radios}"
    chk = [c.label for c in at.checkbox]
    print("checkboxes:", chk)
    assert any("使用最新数据" in c for c in chk), f"缺 使用最新数据: {chk}"

    # 2. 切到 指定股票 → 股票输入框出现
    mode = next(r for r in at.radio if "预测模式" in r.label)
    mode.set_value("指定股票").run(timeout=300)
    assert not at.exception, f"指定股票模式异常: {at.exception}"
    texts = [t.label for t in at.text_input]
    print("text_inputs:", texts)
    assert any("股票代码" in t for t in texts), f"缺股票输入: {texts}"

    # 3. 取消 使用最新数据 → 预测基准日 date_input 出现
    chk_use = next(c for c in at.checkbox if "使用最新数据" in c.label)
    chk_use.uncheck().run(timeout=300)
    assert not at.exception, f"取消使用最新数据异常: {at.exception}"
    dates = [d.label for d in at.date_input]
    print("date_inputs:", dates)
    assert any("预测基准日" in d for d in dates), f"缺 预测基准日: {dates}"

    print("OK: 个股预测 UI (股票输入 + 预测基准日) 渲染无异常")


if __name__ == "__main__":
    main()
