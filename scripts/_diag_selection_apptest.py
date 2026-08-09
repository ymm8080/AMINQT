"""诊断: AppTest 验证 page_selection.render (data_editor 复选框改日内买入)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest


def _run() -> None:
    # from_function 生成独立脚本, 只认函数体内导入的名字
    import streamlit as st  # noqa: F401

    from app.streamlit import page_selection

    page_selection.render()


def main() -> None:
    at = AppTest.from_function(_run)
    at.run(timeout=300)
    print("exception:", at.exception)
    assert not at.exception, f"page_selection 渲染异常: {at.exception}"
    # st.data_editor / st.dataframe 都以 dataframe 元素暴露
    print("dataframes:", len(at.dataframe))
    for i, d in enumerate(at.dataframe):
        try:
            cols = list(d.value.columns)
            print(f"  df[{i}] cols: {cols}")
        except Exception as exc:
            print(f"  df[{i}] value 读取失败: {exc}")
    print("buttons:", [b.label for b in at.button])
    assert len(at.dataframe) >= 1, "缺少选股池表格"
    pool = at.dataframe[0].value
    cols = list(pool.columns)
    assert "名称" in cols or "name" in cols, f"pool 缺名称列: {cols}"
    assert "模型" in cols and "入选" in cols, f"pool 缺模型/入选列: {cols}"
    # 名称应已填充真实名称 (非 "-")
    names = pool["name"] if "name" in cols else pool["名称"]
    filled = (names.astype(str) != "-").sum()
    print(f"pool 行数: {len(pool)}, 名称已填充: {filled}/{len(pool)}")
    assert filled == len(pool), "存在名称为空行"
    # 预期列: 只要 3/5/10d, 无 1d
    for c in ("3d 预期", "5d 预期", "10d 预期"):
        assert c in cols, f"pool 缺 {c}: {cols}"
    assert "1d 预期" not in cols and "pred_ret_1d" not in cols, (
        f"pool 不应含 1d 预期: {cols}"
    )
    # 模型 = 官方交付短名单的来源模型 (family·module)
    models = pool["模型"].dropna().astype(str)
    assert len(models) > 0, "模型列全空"
    print("模型样例:", models.unique()[:5])
    print("入选样例:", pool["入选"].dropna().astype(str).unique()[:5])
    print(
        "预期样例(3d/5d/10d):",
        pool["3d 预期"].head(2).tolist(),
        pool["5d 预期"].head(2).tolist(),
        pool["10d 预期"].head(2).tolist(),
    )
    # 个股明细顶部: 模型推荐 strip (STOCK LIST 真实 3/5/10d, 两行 markdown)
    mds = [m.value for m in at.markdown]
    recs = [m for m in mds if "模型推荐" in m]
    line = next((m for m in mds if "**3d**" in m and "**10d**" in m), "")
    print("模型推荐 strip:", recs[:1], "| 预期行:", line[:80])
    assert recs, "缺模型推荐 strip (各股明细)"
    assert "**3d**" in line and "**10d**" in line, f"strip 缺 3/10d: {line}"
    print("OK: page_selection.render 无异常, 模型/入选/3d5d10d预期 已渲染, 无1d")


if __name__ == "__main__":
    main()
