"""诊断: AppTest 验证 page_trading.render (持仓置顶)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def _run() -> None:
    import streamlit as st  # noqa: F401

    from app.streamlit import page_trading

    page_trading.render()


def main() -> None:
    at = AppTest.from_function(_run)
    at.run(timeout=300)
    print("exception:", at.exception)
    assert not at.exception, f"page_trading 渲染异常: {at.exception}"
    print("headers:", [h.value for h in at.subheader])
    print("metrics:", [m.label for m in at.metric])
    # 持仓应出现在顶部 (subheader 顺序靠前, 早于 行情/信号)
    hdrs = [h.value for h in at.subheader]
    assert "持仓" in hdrs, f"缺持仓区块: {hdrs}"
    pos_idx = hdrs.index("持仓")
    print(f"持仓 subheader 位置: {pos_idx}/{len(hdrs)}")
    assert any("行情" in h for h in hdrs[pos_idx:]), "持仓应位于行情之前 (置顶)"
    print("OK: page_trading.render 无异常, 持仓已置顶")


if __name__ == "__main__":
    main()
