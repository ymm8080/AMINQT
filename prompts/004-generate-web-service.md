---
prompt_id: 004
title: 生成 Web 服务 + 风控过滤器
model_tested: Claude Sonnet 4.5
date: 2026-07-16
version: 1.0
input: {stock_pool, model_path, risk_config}
output: {routes.py, main.py, risk_filter.py}
known_limitations: "内存缓存全市场数据；选股响应须<5秒"
---

# Prompt 004: 生成 Web 服务 + 风控过滤器

## 指令

### 1. 生成 `app/core/risk_filter.py`

实现 `apply_filters(candidates, account_drawdown_pct)`：
- 剔除成交额 < 5000万
- 剔除 |涨跌幅| > 9.5%
- 账户回撤 > 3% → 返回空列表
- 熔断后需人工恢复（不得自动解除）

### 2. 生成 `app/api/routes.py` 和 `app/main.py`

- `POST /api/v1/select`：遍历股票池 → 计算特征 → 模型推理 → 风控过滤 → 返回 Top 10 JSON
- APScheduler：每天 14:50 (Asia/Shanghai) 自动执行 select，结果存入内存缓存
- **启动时全量加载 CSV 到内存** `Dict[str, pd.DataFrame]`（ADR-003）
- `POST /api/v1/cache/refresh`：刷新内存缓存
- `POST /api/v1/recover`：人工恢复熔断

## 验证标准

```bash
uvicorn app.main:app --reload
curl http://127.0.0.1:8000/docs
curl -X POST http://127.0.0.1:8000/api/v1/select  # < 5秒返回 JSON
```

## 合规检查

- [ ] 启动时全量加载内存缓存
- [ ] 选股从内存读取（不逐个文件读取）
- [ ] 风控三层过滤
- [ ] 熔断后人工恢复
- [ ] APScheduler 14:50 Asia/Shanghai
- [ ] 文件头 `# -*- coding: utf-8 -*-`
- [ ] 函数有 docstring
- [ ] 使用 `logging`
- [ ] `try-except` 错误捕获
- [ ] 参见 `rules/030-risk-filter-hard-constraints.md` 和 `skills/risk-circuit-breaker.md`
