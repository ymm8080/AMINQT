# AMINQT 全项目审计报告

> 审计时间: 2026-08-07
> 审计范围: 全部源码、配置、CI/CD、测试、依赖、安全
> 审计工具: 静态代码分析 + ruff + git history + 手动审查

---

## 一、执行摘要

AMINQT 是一个 A 股图形因子量化交易系统, 基于 K 线图形因子提取 → LSTM/LightGBM 模型推理 → 风控过滤 → 自动/手动交易执行的架构。项目整体架构设计优秀, 量化安全意识极强 (未来函数防护、T+1 强制、风控硬约束、WORM 审计日志), 但存在 **代码组织混乱** (340 个脚本散落在根目录和 scripts/)、**API 无认证**、**pickle 反序列化风险** 等问题需要修复。

**总体评分: 7.2 / 10**

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | 9/10 | 分层清晰, 策略模式到位, 文档完善 |
| 量化安全 | 9.5/10 | 未来函数防护、T+1、风控硬约束均为业界最佳实践 |
| 代码质量 | 6/10 | 340 个散落脚本, 12 个 lint 错误, 大量临时文件 |
| 安全性 | 6.5/10 | API 无认证, pickle 风险, 但凭据管理到位 |
| 测试覆盖 | 7.5/10 | 85 个测试文件, CI 80% 覆盖率门, 但缺安全扫描 |
| CI/CD | 7.5/10 | 完善的 PR Gate + Auto-Fix + AI Review |
| 依赖管理 | 5.5/10 | 无版本锁定, 缺失依赖, 重复条目 |

---

## 二、关键发现 (按严重程度排序)

### CRITICAL (必须立即修复)

#### C1. `.env` 文件包含真实 Tushare API Token

- **文件**: `.env` 第 17 行
- **问题**: `TUSHARE_TOKEN=ff1a00b005486505d1bdd87c72d63206d72f6a3ec0cdc062ec867a96`
- **影响**: 任何获得文件系统访问权限的人可使用该 token 调用 Tushare API, 消耗积分或泄露数据
- **缓解**: `.env` 已在 `.gitignore` 中, 且从未提交到 git 历史 (已验证)
- **建议**:
  1. 轮换 (regenerate) Tushare token
  2. 确保 `.env.example` 不含真实 token (当前 `.env` 本身就是模板格式但填入了真实值)
  3. 考虑使用系统密钥管理 (Windows Credential Manager / Vault)

#### C2. API 无认证 — 所有 FastAPI 路由公开可访问

- **文件**: `app/api/routes.py`, `app/api/frontier_routes.py`
- **问题**: 所有 API 端点 (`/api/v1/select`, `/api/v1/execute`, `/api/frontier/*`) 无任何认证/授权机制
- **影响**:
  - `/api/v1/execute` 可被调用提交交易指令 (虽然当前是 stub)
  - `/api/frontier/pipeline/trigger` 可远程触发管道脚本执行
  - `/api/frontier/pipeline/append-daily` 可远程修改面板数据
- **建议**:
  1. 添加 FastAPI 认证中间件 (API Key / JWT / OAuth2)
  2. 至少对写入类端点 (POST) 实施认证
  3. 生产环境绑定 localhost 或内网 IP

#### C3. `pickle.load()` 反序列化 — 任意代码执行风险

- **文件**:
  - `app/pipeline1/dual_track_trainer.py:678` — `pickle.load(fh)`
  - `app/pipeline1/checkpoint.py:186` — `pickle.load(fh)`
  - `app/models/model_zoo.py:468` — `pickle.load(f)`
  - `app/models/model_zoo.py:497` — `pickle.load(f)`
- **问题**: `pickle.load()` 可执行任意 Python 代码。若模型文件被篡改, 攻击者可获得完整代码执行权限
- **影响**: 模型文件 (`.pkl`) 如果来源不可信或被篡改, 可导致系统被完全接管
- **建议**:
  1. 改用 `joblib.load()` 配合 `allow_pickle=False` (sklearn 模型)
  2. 对 torch 模型使用 `torch.load(weights_only=True)` (当前 `model_zoo.py:492` 用了 `weights_only=False`)
  3. 添加模型文件完整性校验 (SHA256 hash 签名)
  4. 限制模型文件目录权限

---

### HIGH (应尽快修复)

#### H1. 根目录脚本泛滥 — 78 个 Python 文件散落在项目根

- **位置**: 项目根目录
- **问题**: 78 个 `.py` 文件 + 大量 `_*.txt` / `_*.log` 临时文件直接放在根目录
- **示例**: `_abc_3y_final.py`, `_apply_fixes.py`, `_compare_models.py`, `_daily_fetch.py`, `_quick_predict.py` 等
- **影响**:
  - 严重影响项目可维护性和新人 onboarding
  - 根目录 `.py` 文件会被 Python 解释器当作顶层模块, 可能与包名冲突
  - 大量临时文件 (`_test_*.txt`, `_r*.txt`) 污染工作区
- **建议**:
  1. 将 `_daily_fetch.py` 等核心脚本移入 `scripts/`
  2. 将 `_abc_*.py`, `_apply_*.py`, `_compare_models.py` 等实验性脚本移入 `scripts/experiments/` 或删除
  3. 清理所有 `_*.txt`, `_*.log` 临时文件 (`.gitignore` 已忽略大部分, 但文件仍在磁盘)
  4. 根目录只保留 `AGENTS.md`, `README.md`, `pyproject.toml`, `requirements.txt` 等标准文件

#### H2. `scripts/` 目录脚本过多 — 262 个文件无子目录组织

- **位置**: `scripts/` 目录
- **问题**: 262 个 Python 脚本平铺在 `scripts/` 下, 无子目录分类
- **影响**: 难以查找和维护, 功能重叠风险高
- **建议**: 按 功能/阶段 分子目录: `scripts/data/`, `scripts/training/`, `scripts/diag/`, `scripts/eval/` 等

#### H3. 12 个 Ruff Lint 错误

- **位置**: `scripts/` 目录下的文件
- **错误类型**: E741 (歧义变量名 `l`)
- **文件**: `scripts/_diag_main_retrain_ic.py`, `scripts/_diag_price_verdict.py` 等
- **影响**: 违反 `AGENTS.md` 中 "Zero linting errors on commit" 的规定
- **建议**: 运行 `ruff check --fix scripts/` 自动修复

#### H4. 大量裸 `except:` 捕获

- **位置**: `scripts/` 目录 (15+ 处)
- **示例**:
  ```python
  except: pass          # _ic_eval_fast.py:22
  except:  # noqa: E722  # scripts/full_fetch_and_eval.py:140
  ```
- **影响**: 吞没所有异常 (包括 `KeyboardInterrupt`, `SystemExit`), 隐藏真实错误
- **建议**: 改为 `except Exception:` 并记录日志

#### H5. 核心功能未实现 — `factor_engine.build_features()` 抛 `NotImplementedError`

- **文件**: `app/core/factor_engine.py:41`
- **问题**: `AGENTS.md` 中描述的核心组件 "图形因子工程 (MACD/KDJ/BOLL/RSI + 衍生特征)" 在 `factor_engine.py` 中未实现
- **影响**: `app/core/` 下的原始因子引擎是空壳, 实际因子计算在 `app/pipeline1/feature_engine_v35.py` 中
- **建议**: 要么实现 `build_features()`, 要么标记为 deprecated 并指向 `feature_engine_v35.py`

---

### MEDIUM (应计划修复)

#### M1. FastAPI 使用已弃用的 `@app.on_event("startup")`

- **文件**: `app/main.py:33`
- **问题**: `@app.on_event("startup")` 在 Starlette 0.36+ / FastAPI 0.110+ 中已弃用
- **建议**: 改用 `lifespan` 上下文管理器:
  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # startup
      ...
      yield
      # shutdown
  app = FastAPI(lifespan=lifespan)
  ```

#### M2. CORS 配置允许所有方法和头部

- **文件**: `app/main.py:18-28`
- **问题**: `allow_methods=["*"]` 和 `allow_headers=["*"]` 过于宽松
- **影响**: 虽然 origins 限制在 localhost, 但生产部署时若忘记限制 origins, 所有 HTTP 方法和头部都被允许
- **建议**: 明确列出允许的方法 (`GET`, `POST`) 和头部 (`Content-Type`)

#### M3. `requirements.txt` 无版本锁定

- **文件**: `requirements.txt`
- **问题**: 所有依赖使用 `>=` 最低版本, 无上限锁定
- **影响**: `pip install -r requirements.txt` 可能安装不兼容版本, CI 和本地环境不一致
- **建议**: 生成 `requirements.lock.txt` 或使用 `pip-compile` / `uv lock` 锁定完整版本

#### M4. 依赖缺失 — `tushare` 和 `pyarrow` 未在 requirements.txt 中

- **文件**: `_daily_fetch.py` 使用 `import tushare as ts` 和 `import pyarrow as pa`
- **问题**: 这两个包未在 `requirements.txt` 中列出
- **影响**: 全新环境安装后 `_daily_fetch.py` 会 `ImportError`
- **建议**: 添加 `tushare` 和 `pyarrow>=14.0` 到 `requirements.txt`

#### M5. `requirements.txt` 中 `httpx` 重复

- **文件**: `requirements.txt` 第 31 行和第 35 行
- **问题**: `httpx>=0.27` (MCP 用) 和 `httpx>=0.25` (Test 用) 重复
- **建议**: 合并为一行 `httpx>=0.27`

#### M6. Ruff 规则过于宽松

- **文件**: `pyproject.toml`
- **问题**: 只启用了 `E4, E7, E9, F` 四组基本规则
- **缺失**: 未启用 `B` (flake8-bugbear), `S` (bandit 安全), `UP` (pyupgrade), `I` (isort) 等有用规则
- **建议**: 启用更多规则集, 至少添加 `B` 和 `I`

#### M7. CI Python 版本与 `AGENTS.md` 不一致

- **文件**: `.github/workflows/ci.yml` (Python 3.11) vs `AGENTS.md` (Python 3.9+)
- **问题**: CI 使用 3.11, 但项目声明支持 3.9+, 可能在 3.9 上有语法兼容问题 (如 `str | None` 类型注解在 3.9 需要 `from __future__ import annotations`)
- **建议**: 统一为 Python 3.11+ 或在 CI 中添加 3.9 兼容测试

#### M8. 无安全扫描 CI 步骤

- **文件**: `.github/workflows/ci.yml`
- **问题**: CI 缺少 `pip audit` / `safety` 依赖安全扫描
- **建议**: 添加 `pip install pip-audit && pip-audit` 步骤

#### M9. `torch.load(weights_only=False)` 安全风险

- **文件**: `app/models/model_zoo.py:492`
- **问题**: `weights_only=False` 允许加载任意 Python 对象, 同 pickle 风险
- **建议**: 改用 `weights_only=True` (PyTorch 2.0+ 默认推荐)

#### M10. subprocess 传参虽用列表但仍需输入校验

- **文件**: `app/api/frontier_routes.py:623` — `_subprocess.run(["python", script_path] + args, ...)`
- **问题**: `req.trade_date` 作为命令行参数传入, 虽然使用了列表形式 (非 `shell=True`), 但未校验格式
- **影响**: 恶意输入虽无法注入 shell 命令, 但可传入异常参数导致脚本行为异常
- **建议**: 添加 `trade_date` 格式校验 (正则 `^\d{8}$`)

---

### LOW (建议改进)

#### L1. 硬编码券商路径

- **文件**: `config/trading_config.yaml:10` — `qmt_path: "D:\\渤海证券QMT\\userdata_mini"`
- **问题**: 本地路径硬编码在配置文件中
- **建议**: 移入 `.env` 环境变量

#### L2. APScheduler 任务为空壳

- **文件**: `app/main.py:41-42`
- **问题**: 定时任务只执行 `logger.info("scheduled select tick")`, 无实际选股逻辑
- **建议**: 实现 TODO 或标记为 Phase 4 待实现

#### L3. API 路由 `/select` 和 `/execute` 为 stub

- **文件**: `app/api/routes.py:24-39`
- **问题**: 核心选股和执行端点返回占位数据
- **建议**: 实现 或 标记为 `@router.get("/select", include_in_schema=False)` 暂时从 API 文档隐藏

#### L4. CI 无依赖缓存

- **文件**: `.github/workflows/ci.yml`
- **问题**: 每次 CI 都重新 `pip install -r requirements.txt`, 浪费时间
- **建议**: 使用 `actions/setup-python@v5` 的 `cache: pip` 参数

#### L5. 前端无安全头部

- **文件**: `app/main.py`
- **问题**: 未设置 `X-Frame-Options`, `X-Content-Type-Options`, `Strict-Transport-Security` 等安全头部
- **建议**: 添加 `fastapi.middleware.TrustedHostMiddleware` 和安全头部中间件

---

## 三、亮点 (做得好的方面)

### 架构与设计

1. **分层架构清晰**: `app/core/` (业务逻辑) / `app/api/` (路由) / `app/models/` (模型) / `services/` (交易执行) / `config/` (配置) — 符合 `AGENTS.md` 中的文件放置决策树
2. **策略模式到位**: 数据源策略 (`akshare` / `iFinD` 可切换) + 执行器策略 (`SimExecutor` / `XtExecutor` 可切换)
3. **`AGENTS.md` 项目规范**: 铁律 12 条、反模式列表、文件放置决策树、创建后检查清单 — 业界少见的完善规范
4. **ADR 记录**: `10_adr/` 下有 4 个架构决策记录, 决策有迹可循

### 量化安全 (项目最大亮点)

5. **未来函数双防线** (`app/pipeline1/leakage_audit.py`):
   - 防线 1: 源码静态扫描 — 正则匹配 `ZIG/PEAK/TROUGH/shift(-k)/REF(X,-k)` 等未来函数模式
   - 防线 2: IC 上限哨兵 — 任一特征 |Rank IC| > 0.15 触发泄漏复核
6. **T+1 强制执行**: `XtExecutor._today_bought` 集合 + `OrderManager.check_t1_sell()` 双重保障
7. **风控硬约束** (`app/core/risk_filter.py`): 成交额 < 5000 万 → 剔除; |涨跌幅| > 9.5% → 剔除; 账户回撤 > 3% → 返回空列表 (熔断)
8. **AUTO 模式风控门** (`services/executor_base.py:69-100`): AUTO 下单前必须携带 `amount` + `pct_change` 元数据, 缺失则拒绝 (fail-safe)
9. **WORM 审计日志** (`services/order_manager.py:36-46`): 交易日志 JSONL append-only, 不可修改
10. **时间切分防泄漏**: 训练 2018-2020 / 验证 2021 / 测试 2022-2024, 严禁随机打乱

### 测试与 CI/CD

11. **85 个测试文件**: 覆盖核心模块 (风控、执行器、因子引擎、回测、管道等)
12. **CI 80% 覆盖率门**: `--cov-fail-under=80` 强制保持覆盖率
13. **PR Gate 工作流**: 自动 `request-changes` 阻止合并, CI 通过后 `auto-approve` 自动解除
14. **Auto-Fix 工作流**: PR 自动运行 `ruff format + ruff check --fix` 并提交修复
15. **DeepSeek AI PR Review**: 使用 AI 自动审查 PR 代码 (8 维度量化规则检查)
16. **Parquet-only 数据门**: CI 强制 `data/` 目录只包含 parquet 文件

### 凭据管理

17. **`.env` 正确 gitignore**: 已验证 `.env` 从未进入 git 历史
18. **凭据从环境变量加载**: `IFIND_USER`, `IFIND_PASSWORD`, `TUSHARE_TOKEN`, `DEEPSEEK_API_KEY` 全部从 `os.getenv()` 读取
19. **配置文件不含密钥**: `llm_config.yaml` 明确注释 "API key 只存 .env"
20. **DeepSeek 熔断器**: 连续失败 3 次 → 熔断 30 分钟; 日 token 限额 500K

---

## 四、统计数据

| 指标 | 数值 |
|------|------|
| 根目录 Python 文件 | 78 |
| `scripts/` 目录 Python 文件 | 262 |
| `app/` 目录 Python 文件 | ~80 |
| `tests/` 目录测试文件 | 85 |
| Ruff lint 错误 | 12 |
| 裸 `except:` (scripts/) | 15+ |
| `pickle.load()` 调用 | 4 |
| CI 工作流文件 | 5 |
| ADR 文件 | 4 + 1 模板 |
| 配置 YAML 文件 | 7 |
| `AGENTS.md` 铁律 | 12 |

---

## 五、修复优先级路线图

### Phase 1: 紧急修复 (1-2 天)
1. [C1] 轮换 Tushare token
2. [C2] 添加 API 认证中间件 (至少 API Key)
3. [C3] 替换 `pickle.load()` → `joblib.load()` / `torch.load(weights_only=True)`
4. [H3] 修复 12 个 ruff 错误

### Phase 2: 代码整理 (3-5 天)
5. [H1] 清理根目录 78 个脚本 → 移入 `scripts/` 或删除
6. [H2] `scripts/` 目录分子目录
7. [H4] 修复所有裸 `except:`
8. 清理 `_*.txt` / `_*.log` 临时文件

### Phase 3: 依赖与配置 (2-3 天)
9. [M3] 生成 `requirements.lock.txt` 版本锁定
10. [M4] 添加缺失依赖 (`tushare`, `pyarrow`)
11. [M5] 合并重复 `httpx`
12. [M6] 扩展 ruff 规则集
13. [M7] 统一 Python 版本要求

### Phase 4: CI/CD 加固 (1-2 天)
14. [M8] 添加 `pip-audit` 安全扫描步骤
15. [L4] 添加 pip 缓存
16. [M10] 添加 API 输入校验

### Phase 5: 架构优化 (长期)
17. [H5] 实现或废弃 `factor_engine.build_features()`
18. [L2] 实现 APScheduler 选股任务
19. [L3] 实现 API `/select` 和 `/execute` 端点
20. [L5] 添加安全 HTTP 头部

---

## 六、审计方法说明

本审计基于以下方法:
1. **静态代码分析**: 阅读核心源码 (`app/`, `services/`, `config/`) 理解架构
2. **模式搜索**: 使用 `grep` 搜索安全漏洞模式 (硬编码凭据、`eval`/`exec`、`shift(-k)`、裸 `except`、`pickle.load`)
3. **Git 历史分析**: 验证 `.env` 和 Tushare token 从未提交到版本控制
4. **配置审查**: 检查 `.gitignore`、`pyproject.toml`、`requirements.txt`、CI 工作流、YAML 配置
5. **Ruff 静态分析**: 运行 `ruff check` 获取 lint 错误统计
6. **依赖审查**: 检查 `requirements.txt` 和 `package.json` 的版本管理
7. **测试审查**: 检查测试文件数量、覆盖率门、测试配置
8. **CI/CD 审查**: 审查 5 个 GitHub Actions 工作流的安全性和完整性

> **免责声明**: 本审计为代码审查, 非渗透测试。建议使用 `pip-audit`、`trivy`、`bandit` 等工具进行自动化安全扫描作为补充。
