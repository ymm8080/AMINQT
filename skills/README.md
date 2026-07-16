# 项目技能索引 (Skills Index)

> 所有技能以 IDE 无关的 `.md` 文件存放于此目录。
> AI 助手在执行相关任务时应加载对应技能。

## 技能列表

| 文件 | 触发条件 | 说明 |
|------|----------|------|
| `verify-before-done.md` | 声称"完成"前 | 必须运行验证命令并展示证据 |
| `quant-factor-engineering.md` | 编写因子工程代码 | 技术指标、衍生特征、滑动窗口的完整规范 |
| `risk-circuit-breaker.md` | 编写风控/执行代码 | 三层熔断、T+1、审计日志、执行模式安全 |
| `quant-data-pipeline.md` | 编写数据加载/下载代码 | akshare列名映射、增量更新、完整性校验、内存缓存 |
| `graph-to-vector.md` | K线图形向量化 | 四条路径: 滑动窗口矩阵、CNN图像嵌入、形态二值向量、股票关系图嵌入 |
| `superpowers.md` | 新功能/重构/架构设计前 | 8 维编码前检查清单（架构/测试/安全/性能/数据流/部署/可观测性） |
| `diagnose.md` | bug 排查、异常诊断 | 系统化调试 5 步法（复现→收集→隔离→修→验） |
| `careful.md` | 实盘交易/风控/金融计算 | 关键系统双重检查清单 |
| `grill-me.md` | 需求不明确/新功能开发 | 编码前追问需求，防止误解 |
| `memory-manager.md` | 完成重要工作后/会话结束 | 跨会话记忆管理（Pattern/Pitfall/Decision 格式） |

## 使用方式

1. **按需加载**: 根据当前任务的触发条件，读取对应技能文件
2. **遵循检查清单**: 技能文件末尾有检查清单，逐项确认
3. **验证后声称完成**: 遵循 `verify-before-done` 的验证流程

## 技能来源

本套技能从 `D:\EWM ROBOT\ROBOTIC PLATFORM CODES` 项目的以下模式中提炼并适配：

| 机器人项目模式 | → | 量化项目技能 |
|:---|:---|:---|
| `verify-before-done` skill | → | `verify-before-done.md` (验证流程适配量化场景) |
| `SafetyPlc` 硬下限 + `Watchdog` 熔断 | → | `risk-circuit-breaker.md` (三层风控 + 熔断恢复) |
| `WormBlackbox` 因果存证 | → | `risk-circuit-breaker.md` 中的审计日志 (append-only + fsync) |
| `FailoverDegrade` 降级模式 | → | `risk-circuit-breaker.md` 中的执行器降级 (sim 优先) |
| `VersionRouter` N-1 兼容 | → | `risk-circuit-breaker.md` 中的 T+1 检查器 |
| 核心模块分层设计 | → | `quant-factor-engineering.md` 因子工程完整规范 |
