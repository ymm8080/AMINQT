# Skill: memory-manager (跨会话记忆管理)

> **来源**: EWM Robot 项目 `memory-manager.md`
> **用途**: 管理跨会话 AI 记忆，记录 Pattern/Pitfall/Decision，避免重复踩坑
> **触发**: 完成重要工作后、发现模式/陷阱时、会话结束时

## 记忆文件

### AGENTS.md（项目上下文，已有）
- 当前架构、活跃决策、关键协议
- 会话开始时读取
- 架构变更时更新

### MEMORY.md（跨会话记忆，待创建）
- Pattern: 可靠的解决方案
- Pitfall: 踩过的坑及修复
- Decision: 为什么选 X 不选 Y
- Workflow: 重复执行的操作流程
- Session History: 最近工作摘要

## 何时记忆

| 场景 | 记忆类型 |
|------|----------|
| 发现非显而易见的 bug 及修复 | Pitfall |
| 找到可靠可复用的方案 | Pattern |
| 做了有权衡的架构决策 | Decision |
| 同一任务重复 3+ 次 | Workflow |
| 环境变化（依赖升级等） | Context |

## 记忆格式

### Pattern
```markdown
### Pattern XXX: [标题]
**发现**: YYYY-MM-DD
**适用于**: [范围]
**模式**: [一句话描述]

# 错误写法
[code]

# 正确写法
[code]

**原因**: [为什么]
```

### Pitfall
```markdown
### Pitfall XXX: [标题]
**日期**: YYYY-MM-DD
**症状**: [出了什么问题]
**根因**: [为什么]
**修复**: [怎么解决]

[code]

**教训**: [关键收获]
**预防**: [如何避免]
```

### Decision
```markdown
### Decision XXX: [标题]
**日期**: YYYY-MM-DD
**ADR**: ADR-XXX 或 "Pending"
**决策**: [选了什么]
**原因**: [为什么]
**权衡**: ✅ Pro / ❌ Con
**备选**: [考虑过什么，为什么放弃]
```

## 会话工作流

### 开始
1. 读 `AGENTS.md`
2. 读 `MEMORY.md`（如有）
3. 查找相关 Pattern/Pitfall

### 结束
1. 识别本次值得记忆的内容
2. 按格式写入 `MEMORY.md`
3. 如架构变更，更新 `AGENTS.md`
4. 验证文件已更新

## Token 预算

| 层级 | 大小 | 加载策略 |
|------|------|----------|
| Tier 1: AGENTS.md | ~8K tokens | 始终加载 |
| Tier 2: MEMORY.md 章节 | ~12K tokens | 相关时加载 |
| Tier 3: 完整 MEMORY.md | ~20K tokens | 按需搜索 |

超预算时：清理 90 天以上条目，合并相关条目。

## 量化记忆示例

### Pattern: safe_divide 防除零
```markdown
### Pattern 001: safe_divide 防除零
**发现**: 2026-07-16
**适用于**: factor_engine, risk_filter

np.divide(num, den, out=np.zeros_like(num), where=den!=0)
```

### Pitfall: akshare 列名中文
```markdown
### Pitfall 001: akshare 列名中文导致 KeyError
**症状**: df['close'] → KeyError
**根因**: akshare 返回中文列名
**修复**: 读取后立即 rename(columns=COL_MAP)
**预防**: 适配器内部完成映射
```
