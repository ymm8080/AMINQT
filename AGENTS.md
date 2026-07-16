# AGENTS.md

> A股图形因子量化交易系统 (AMINQT)
> Project version: v0.1 | Last updated: 2026-07-16

## Project Identity

A股图形因子量化交易系统，基于同花顺(iFinD)量化指标选股。
核心能力：K线图形因子提取 → LSTM/LightGBM 模型推理 → 风控过滤 → 自动/手动交易执行。
技术栈：Python 3.9+, FastAPI, APScheduler, PyTorch, akshare/iFinD, miniQMT(xtquant), Streamlit。

## Global Response Standards (Anti-Sycophancy + Token Efficiency)

**Anti-sycophancy:**
1. Tag claims: `[KNOWN]` · `[COMPUTED]` · `[INFERRED]` · `[GUESS]` · `[COMMON]` · `[FRAME]`
2. CONFIDENCE: HIGH ≥80% · MED 50–80% · LOW 20–50% · VERY LOW <20% · UNKNOWN. `[FRAME]` real-world and `[GUESS]` cap at LOW.
3. Don't know? First line "I don't know." No fabricating.
4. Sycophancy red flags → cut specifics, add `[GUESS]`, or say "I don't know."
5. Post-hoc → mark `[INFERRED, post-hoc]`.
6. No fabricated citations.
7. Rules broken? Append `[RULES I BROKE]: which, where, why.`

**Token efficiency (applies everywhere):**
8. Be concise. No greetings, closings, or praising.
9. No verbose disclaimers.
10. One answer per question — no restating the question.
11. Prefer Grep/Glob over reading entire files to locate relevant code first.

Applies globally to every response — research, analysis, problem-solving, code review, Q&A.

## Iron Rules (量化交易铁律)

1. **未来函数绝对禁止**: 计算第 t 天的指标时，严禁使用 t 以后的数据。使用 `pandas.rolling/expanding`，严禁 `shift(-k)`。
2. **实盘资金安全**: AUTO 模式下单前，必须经过 `risk_filter.apply_filters()` 硬约束过滤。账户回撤 > 3% → 立即停止下单。
3. **LLM 交易禁令**: LLM 不得直接发出实盘交易指令。LLM 只建议；用户或风控引擎执行。
4. **T+1 限制**: 当日买入的股票不能在当日卖出。`executor` 必须强制执行此规则。
5. **数据主权**: 凭据只从环境变量/.env 加载，严禁硬编码。iFinD 密码不入代码。
6. **时区**: 所有时间戳存储为 UTC+8 (Asia/Shanghai)；日期操作使用 `datetime` 对象，严禁字符串直接比较。
7. **路径跨平台**: 所有文件路径使用 `os.path.join()` 或 `pathlib.Path`，确保 Windows/Linux 兼容。
8. **除零防护**: 所有除法运算使用 `safe_divide()`，分母为 0 时返回 0（防 `inf`）。
9. **NaN 处理**: 特征矩阵输入模型前必须执行 `np.nan_to_num(X)`。
10. **审计日志不可变**: 交易日志 append-only，严禁 DELETE 或 UPDATE。
11. **变更前备份**: 修改模型权重、配置或数据库 schema 前，必须先备份。
12. **模拟器优先**: 无法连接券商时，必须生成模拟执行器 (`SimExecutor`)，仅打印指令，不影响系统其他模块。

### File Placement Decision Tree
1. 核心业务逻辑？ → `app/core/`
2. API 路由？ → `app/api/`
3. 模型定义/训练？ → `app/models/`
4. 日内学习/规则？ → `app/pattern/`, `app/rules/`
5. 数据适配器？ → `data/adapters/`
6. 交易执行？ → `services/`
7. 脚本工具？ → `scripts/`
8. 文档？ → `docs/` 或根目录编号文件
9. 测试？ → `tests/` 与源码对应
10. ADR？ → `10_adr/`
11. 运维手册？ → `03_operations/`
12. 部署清单？ → `02_deployment/`
13. 不确定？ → ASK before creating

### Post-Creation Checklist
After creating any file:
- [ ] 正确目录（按上方决策树）
- [ ] 文件头 `# -*- coding: utf-8 -*-`
- [ ] 函数有 docstring（Google/Numpy 风格）
- [ ] 使用 `logging` 而非 `print`（除模拟执行器）
- [ ] try-except 错误捕获
- [ ] 无硬编码凭据
- [ ] 测试已编写（如适用）
- [ ] ADR 已创建（如涉及架构决策）

## System Architecture

### 核心组件与服务端口
| 组件 | 端口 | 职责 |
|------|------|------|
| FastAPI | 8000 | Web API, 选股接口 |
| APScheduler | — | 每日 14:50 自动选股 |
| Streamlit | 8501 | 研究面板, K线可视化 |
| miniQMT (xtquant) | — | 券商交易接口 |
| akshare/iFinD | — | 行情数据源 |

### 源码结构
```
app/
  ├── api/routes.py        — FastAPI 路由
  ├── core/
  │   ├── data_loader.py   — 数据加载 (CSV内存缓存)
  │   ├── factor_engine.py — 图形因子工程 (MACD/KDJ/BOLL/RSI + 衍生特征)
  │   └── risk_filter.py   — 硬约束风控过滤
  ├── models/
  │   ├── lstm_model.py    — LSTM 模型定义
  │   └── trained/         — 训练权重 (.pth/.pkl)
  ├── pattern/intraday_learner.py — 日内模式学习
  ├── rules/rule_engine.py       — 交易规则引擎
  ├── streamlit_app.py    — 可视化研究面板
  └── main.py             — FastAPI 入口
config/settings.py         — 全局配置
data/
  ├── adapters/           — 数据源适配器 (akshare, iFinD)
  ├── raw/                — 原始CSV
  ├── intraday/           — 日内数据
  └── processed/          — 处理后数据
services/
  ├── executor_base.py    — 执行器基类
  ├── sim_executor.py     — 模拟执行器
  └── xt_executor.py      — miniQMT 实盘执行器
scripts/                   — 下载/训练/测试脚本
tests/                     — pytest 测试套件
```

### 关键技术模式
1. **数据源策略模式**: `data/adapters/` (akshare 适配器, iFinD 适配器) — 通过 `DATA_SOURCE` 环境变量切换
2. **执行器策略模式**: `services/` (SimExecutor, XtExecutor) — 通过 `EXECUTION_BROKER` 切换
3. **未来函数防护**: `factor_engine` 使用 `rolling/expanding` 计算指标，严禁前瞻偏差
4. **硬约束风控**: `risk_filter` — 成交额/涨跌幅/回撤三层过滤
5. **内存缓存选股**: FastAPI 启动时全量 CSV 加载入 `Dict[str, pd.DataFrame]`，选股 < 5s

## Critical Protocols

### 数据格式映射 (必须严格遵守)
akshare 下载的数据列名为**中文**，读取后必须立即重命名：

| akshare 原始列名 | 重命名后 |
|:---|:---|
| `日期` | `date` |
| `开盘` | `open` |
| `收盘` | `close` |
| `最高` | `high` |
| `最低` | `low` |
| `成交量` | `volume` |
| `成交额` | `amount` |

### 选股流程
1. 遍历股票池 → 从内存缓存读取 DataFrame
2. 调用 `factor_engine.build_features(df)` → 特征矩阵 X
3. 调用 `model_predict.predict(X)` → 预测得分
4. 按 score 排序取 Top-N → 候选列表
5. 调用 `risk_filter.apply_filters(candidates)` → 过滤后列表
6. 返回 JSON (或推送到交易执行器)

### 训练数据切分
- 训练集: 2018-01-01 ~ 2020-12-31
- 验证集: 2021-01-01 ~ 2021-12-31
- 测试集: 2022-01-01 ~ 2024-12-31
- 严格按时间切分，严禁随机打乱（防止时间泄漏）

## Development Standards

### Code
- Python 3.9+ for all modules
- Conventional Commits for git messages
- Test coverage minimum: 80%
- Zero linting errors on commit (ruff for Python)
- 文件头 `# -*- coding: utf-8 -*-`
- 函数 docstring（Google/Numpy 风格，说明参数/返回值/异常）
- `import logging`，关键步骤打印 info 日志，报错打印 error 日志
- 所有代码必须包含 `try-except` 错误捕获
- 所有涉及日期的操作使用 `datetime` 对象
- 所有数据路径使用 `os.path.join()` 或 `pathlib.Path`

### Language Boundaries (STRICT)
- 后端/API: Python 3.9+ only
- 前端面板: Streamlit + Plotly (Python)
- 数据脚本: Python only
- 测试: pytest

### Risk Filter Hard Constraints
| 约束 | 阈值 | 动作 |
|------|------|------|
| 成交额 | < 5000万 | 剔除 |
| 涨跌幅 | |x| > 9.5% | 剔除 |
| 账户回撤 | > 3% | 返回空列表（停止下单） |

### Documentation
- 文档随代码变更同步更新
- 架构变更需创建 ADR (见 `10_adr/`)
- 运维手册在部署前更新
- 每次会话结束更新 MEMORY.md

## Anti-Patterns (PROHIBITED)

- ❌ 严禁使用未来函数 (`shift(-k)` 或未来数据计算指标)
- ❌ 严禁跳过 `risk_filter` 直接下单
- ❌ 严禁声称"完成"而未运行验证 (见 `verify-before-done` skill)
- ❌ 严禁硬编码凭据 (iFinD 密码等)
- ❌ 严禁随机打乱时序数据 (训练集必须按时间切分)
- ❌ 严禁逐个文件读取选股 (必须内存缓存)
- ❌ 严禁字符串直接比较日期 (必须用 `datetime`)
- ❌ 严禁当日卖出当日买入股票 (T+1)
- ❌ 严禁跳过 `try-except` 错误捕获
- ❌ 严禁修改数据源 SDK (包装, 不分叉)

## Session Workflow

### Start
1. Read this AGENTS.md (project context)
2. Read ARCHITECTURE + IMPLEMENTATION PLAN + PROMPT_CONTENT
3. Check for relevant patterns/pitfalls

### During
1. Follow verification-before-done rules — never claim "done" without running verification commands and showing evidence
2. Reference existing patterns before creating new solutions
3. Use compressed communication (concise, no filler)
4. Create ADRs for architecture decisions
5. Follow phased implementation plan — each phase must pass validation

### End
1. Update memory with new learnings
2. Update AGENTS.md if architecture changed
3. Verify files updated successfully

## Emergency Procedures

### 模型推理异常
```bash
# 检查模型权重是否存在
ls -la app/models/trained/
# 重新加载模型
python -c "from app.models.lstm_model import LSTMModel; m = LSTMModel(); print(m)"
```

### 数据下载失败
```bash
# 检查 akshare 连接
python -c "import akshare as ak; df = ak.stock_zh_a_hist('600519', period='daily', adjust='qfq'); print(df.shape)"
# 检查磁盘空间
df -h data/
```

### 实盘交易熔断
```python
# 账户回撤 > 3% → risk_filter 返回空列表
# AUTO 模式下立即停止下单
# MANUAL 模式下弹出告警
```

## Quick Commands
```bash
# 启动 API 服务
uvicorn app.main:app --reload

# 启动研究面板
streamlit run app/streamlit_app.py

# 下载测试数据
python scripts/download_data.py

# 训练模型
python scripts/train_model.py

# 运行测试
pytest tests/ -v

# 代码检查
ruff check app/ scripts/ services/
```
