# -*- coding: utf-8 -*-
# CLAUDE.md — Claude Code 协作约定

机器学习PIPELINE 特征因子大多看连续交易日时序变化，而不是看某一天的截面

所有特征因子数据都需要日频，或者分钟级别得全量数据。
如果取不到就弹出信息让我确认是放弃还是继续寻找数据源

> **主上下文在 `AGENTS.md`**。本文件不重复其内容，仅补充 Claude Code 专属约定。
> 会话开始时：先读 `AGENTS.md`（铁律/架构/协议），再读本文件。

## Agent skills

### Issue tracker
GitHub Issues (`gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels
Default 5 canonical labels. See `docs/agents/triage-labels.md`.

### Domain docs
Single-context: ADRs in `10_adr/`, no `CONTEXT.md` yet. See `docs/agents/domain.md`.

## 始终生效的技能 (alwaysApply)

### Karpathy LLM 编码指南
`karpathy-guidelines` 技能始终生效。核心 4 条 + 扩展 8 条见 `rules/070-karpathy-guidelines.md`。

简版：
1. **编码前先思考** — 显式声明假设，不隐藏困惑
2. **简洁优先** — 最少代码解决问题，不做推测性开发
3. **外科手术式修改** — 只动该动的，不"改进"相邻代码
4. **目标驱动执行** — 定义成功标准，循环直到验证通过
5. LLM 只做判断，确定性逻辑用代码
6. 硬性 Token 预算
7. 暴露冲突，不要平均
8. 写之前先读
9. 测试验证意图，不只验证行为
10. 每个重要步骤后检查点
11. 匹配代码库惯例
12. 失败要大声

### Matt Pocock 工程技能
`setup-matt-pocock-skills` 已配置完成。相关技能（`to-tickets`, `triage`, `to-spec`, `qa` 等）读取 `docs/agents/*.md` 获取 issue tracker / triage 标签 / domain docs 配置。

## 工具链

| 工具 | 用途 |
|------|------|
| Claude Code | 主力 AI 编码（读本文件） |
| Cursor | IDE + AI（读 `AGENTS.md`） |
| `gh` CLI | GitHub Issues / PR 操作 |
| `ruff` | Python lint |
| `pytest` | 测试 |

## Sub-Agent 模块

| 模块 | 模型 | 协议 | 工具 |
|------|------|------|------|
| `deepseek-sub-agent` | DeepSeek V4 Flash | MCP stdio | 6 个工具: `deepseek_ask`, `deepseek_review_code`, `deepseek_analyze_sentiment`, `deepseek_explain_signal`, `deepseek_factor_hypothesis`, `deepseek_diagnose` |

- 注册在 `.claude/mcp.json`，Claude Code 启动时自动发现
- 需在 `.env` 中设置 `DEEPSEEK_API_KEY`
- 详见 `docs/agents/deepseek-sub-agent.md`

## 文件优先级（冲突时）

1. `AGENTS.md` — 项目铁律、架构、协议（最高优先级）
2. `rules/*.md` — 详细规则（`000-global-iron-rules.md` 为铁律准绳）
3. 本文件 — Claude Code 专属约定 + 技能配置
4. `docs/agents/*.md` — 技能运行时配置

