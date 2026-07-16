# 项目规则索引 (Rules Index)

> 所有规则以 IDE 无关的 `.md` 文件存放于此目录。
> AI 助手在会话开始时应读取本索引，按需加载相关规则。

## 规则列表

| 文件 | 范围 | 说明 |
|------|------|------|
| `000-global-iron-rules.md` | 全局铁律 | 未来函数禁止、资金安全、T+1、审计日志、变更前备份 |
| `010-future-function-prevention.md` | 因子工程 | 未来函数防护的具体写法和验证方法 |
| `020-data-format-mapping.md` | 数据加载 | akshare 中文列名映射、适配器接口、内存缓存 |
| `030-risk-filter-hard-constraints.md` | 风控执行 | 三层风控过滤、执行模式、模拟执行器、审计日志 |
| `040-model-training-standards.md` | 模型训练 | 数据切分、LSTM 规范、训练循环、特征矩阵 |
| `050-code-conventions.md` | 代码规范 | 文件头、docstring、日志、错误捕获、日期/路径处理 |
| `060-anti-sycophancy.md` | AI 行为 | 声明标签、置信度分级、反讨好检测 |
| `070-karpathy-guidelines.md` | AI 编码行为 | Karpathy 12 条 LLM 编码指南（始终生效，错误率 41%→3%） |
| `080-gsd-workflow.md` | 交付流程 | GSD 快速交付：先跑通再优化，Phase 分阶段递进 |
| `090-compressed-communication.md` | 沟通模式 | 压缩沟通：直接答案、列表优于段落、代码优于描述 |
| `100-alert-priority-and-audit.md` | 告警合规 | 告警分级 P0/P1/P2、告警冷却、审计日志保留 180 天 |
| `110-operational-limits.md` | 运维限流 | 内存缓存监控、日志轮转、部署防护、下载限流 |
| `120-idempotency-patterns.md` | 交易幂等 | 重复下单防护、调度器重复执行防护、API 幂等键 |

## alwaysApply 规则（始终生效）

以下规则标记为始终生效（对标机器人项目 `alwaysApply: true`）：

| 规则 | 机器人项目对应 | 说明 |
|------|---------------|------|
| `000-global-iron-rules.md` | `000-global-iron-rules.mdc` | 全局铁律 |
| `070-karpathy-guidelines.md` | `karpathy-guidelines.mdc` | LLM 编码行为 |
| `080-gsd-workflow.md` | `gsd-workflow.mdc` | 快速交付流程 |
| `090-compressed-communication.md` | `compressed-communication.mdc` | 压缩沟通 |
| `100-alert-priority-and-audit.md` | `080-enterprise-policies.mdc` | 告警+合规 (精简适配) |
| `110-operational-limits.md` | `090-operational-limits.mdc` | 运维限流 (精简适配) |

## 使用方式

1. **会话开始时**: 读取本索引 + `AGENTS.md`
2. **按需加载**: 根据当前任务触发条件，加载对应规则
3. **规则冲突时**: 以 `000-global-iron-rules.md` 为准
4. **新增规则**: 在此目录创建 `.md` 文件，更新本索引

## 规则来源

本套规则从 `D:\EWM ROBOT\ROBOTIC PLATFORM CODES` 项目的以下模式中提炼并适配：
- 铁律体系 (Iron Rules) → 量化交易铁律
- 安全 PLC 硬下限 → 风控硬约束
- WORM 黑匣子审计 → 交易审计日志
- 熔断/降级模式 → 账户回撤熔断
- 反讨好协议 → AI 声明标签与置信度
- 代码规范 → Python 量化代码规范
- GSD 工作流 → Phase 分阶段交付
- 压缩沟通 → 量化场景快速沟通
- 企业通知矩阵 → 告警分级 (删除了飞书/短信/2FA)
- Node-RED/Docker 限流 → 内存缓存/日志轮转/下载限流 (删除了 Node-RED/Docker/SQLite WAL)
- Karpathy 12 条 → LLM 编码行为指南
