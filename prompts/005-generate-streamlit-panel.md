---
prompt_id: 005
title: 生成 Streamlit 研究面板
model_tested: Claude Sonnet 4.5
date: 2026-07-16
version: 1.0
input: {stock_code, date_range}
output: {streamlit_app.py}
known_limitations: "需 Plotly 依赖；大数据量交互可能卡顿"
---

# Prompt 005: 生成 Streamlit 研究面板

## 指令

生成 `app/streamlit_app.py`：

1. **左侧输入框**：股票代码（如 600519）、日期范围
2. **中间主图**：用 Plotly 绘制 K 线图，叠加均线 (MA5, MA20, MA60)
3. **副图1**：MACD 指标 (DIF, DEA, 柱状图)
4. **副图2**：在 K 线图上用红色/绿色箭头标记金叉/死叉信号点

## 验证标准

```bash
streamlit run app/streamlit_app.py
# 浏览器打开后能看到完整的可交互图形
```

## 合规检查

- [ ] K 线图 + 均线叠加
- [ ] MACD 副图
- [ ] 金叉/死叉标记
- [ ] 使用 Plotly（不是 matplotlib）
- [ ] 文件头 `# -*- coding: utf-8 -*-`
- [ ] 使用 `logging`（调试信息）
- [ ] `try-except` 错误捕获
