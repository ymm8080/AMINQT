# DeepSeek V4 Flash — Claude Code Sub-Agent Module

> 将 DeepSeek V4 Flash 作为 Claude Code 的子代理，通过 MCP 协议暴露 6 个专用工具。
> 注册在 `.claude/mcp.json`，Claude Code 启动时自动发现并加载。

## 架构

```
Claude Code  ↔  MCP stdio  ↔  services/deepseek_sub_agent.py  ↔  DeepSeek API
     │                                                                │
     │  工具调用 (deepseek_*)                                      deepseek-v4-flash
     └───────────────────────────────────────────────────────────────┘
```

## 可用工具

| 工具 | 用途 | 典型场景 |
|------|------|----------|
| `deepseek_ask` | 通用问答 | 任意推理/分析需求 |
| `deepseek_review_code` | 8 维度代码审查 | PR review、代码合规检查 |
| `deepseek_analyze_sentiment` | 市场情绪分析 | 新闻/公告情绪 bull/neutral/bear |
| `deepseek_explain_signal` | 交易信号解释 | 解释模型输出信号逻辑 |
| `deepseek_factor_hypothesis` | 因子假设生成 | 探索性因子研究 |
| `deepseek_diagnose` | 系统诊断 | 错误日志根因分析 |

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | (必填) | DeepSeek API key，存 `.env` |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 模型名称 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MAX_TOKENS` | `8192` | 最大输出 Token |
| `DEEPSEEK_TIMEOUT_SEC` | `60` | 请求超时秒数 |

### .env 模板

```bash
# .env
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 使用方式

### 在 Claude Code 中调用

MCP 工具自动注册，直接使用自然语言：

```
请用 deepseek 审查这段代码...
用 deepseek 分析一下这个新闻的情绪...
请 deepseek 解释这个信号的逻辑...
```

### 手动测试

```bash
cd d:\AMINQT\AMINQT CODES
python -c "
import asyncio
from services.deepseek_sub_agent import main
asyncio.run(main())
"
```

## 工具输出格式

所有工具默认返回 JSON 格式（代码审查、情绪分析、信号解释、因子假设、诊断），
通用问答 `deepseek_ask` 返回纯文本。

## 故障排除

| 症状 | 原因 | 修复 |
|------|------|------|
| `DEEPSEEK_API_KEY not set` | .env 缺少 key | 添加 `DEEPSEEK_API_KEY` 到 `.env` |
| 工具返回空 | 网络问题 | 检查 `DEEPSEEK_BASE_URL` 可达性 |
| 超时 | 请求过大 | 减少输入内容或增大 `DEEPSEEK_TIMEOUT_SEC` |
| `ModuleNotFoundError: mcp` | MCP SDK 未安装 | `pip install mcp` |