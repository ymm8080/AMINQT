---
prompt_id: 001
title: 生成数据下载脚本
model_tested: Claude Sonnet 4.5
date: 2026-07-16
version: 1.0
input: {stock_list, data_source, output_dir}
output: {download_script.py, csv_files}
known_limitations: "akshare 反爬限速，全市场下载需分批；iFinD 需凭据"
---

# Prompt 001: 生成数据下载脚本

## 指令

生成 `scripts/download_data.py`，要求：

1. 定义股票池 `STOCK_LIST = ['000001','000002','600519','000858','600036']`（先测试5只）
2. 使用 `ak.stock_zh_a_hist(symbol, period='daily', adjust='qfq')` 下载
3. **读取后立即重命名中文列名**（见 `rules/020-data-format-mapping.md`）：
   - `日期→date, 开盘→open, 收盘→close, 最高→high, 最低→low, 成交量→volume, 成交额→amount`
4. 保存至 `data/raw/{code}.csv`
5. 包含 `time.sleep(0.5)` 防止反爬
6. 包含 `try-except` 错误捕获
7. 使用 `logging` 记录日志（不用 print）

## 验证标准

运行 `python scripts/download_data.py`：
- 终端无报错
- `data/raw/` 下出现5个CSV文件
- 每个CSV列名为英文 (`date, open, close, high, low, volume, amount`)

## 合规检查

- [ ] 文件头 `# -*- coding: utf-8 -*-`
- [ ] 函数有 docstring
- [ ] 使用 `logging` 而非 `print`
- [ ] 包含 `try-except`
- [ ] 包含 `time.sleep(0.5)` 反爬延时
- [ ] 列名映射在适配器内部完成
- [ ] 路径使用 `pathlib.Path` 或 `os.path.join()`
