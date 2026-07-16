# Prompts — 版本管理

> **来源**: 从 EWM Robot 项目 `prompts/README.md` 的版本管理模式适配
> **设计原则**: 文档即代码，git 管理 + model_tested 元数据

## 规则

1. **每个 prompt 有 YAML frontmatter**，包含 `model_tested`, `date`, `version`, `known_limitations`
2. **版本升级** 当模型变更或 prompt 逻辑变更时
3. **不删除旧版本** — 归档为 `v1.0_20260716/001_xxx.md`
4. **用目标模型测试** 后再提交

## 当前 Prompts

| ID | 标题 | 测试模型 | 版本 | 日期 |
|----|------|----------|------|------|
| 001 | 生成数据下载脚本 | Claude Sonnet 4.5 | 1.0 | 2026-07-16 |
| 002 | 生成因子引擎模块 | Claude Sonnet 4.5 | 1.0 | 2026-07-16 |
| 003 | 生成 LSTM 模型 + 训练脚本 | Claude Sonnet 4.5 | 1.0 | 2026-07-16 |
| 004 | 生成 Web 服务 + 风控过滤器 | Claude Sonnet 4.5 | 1.0 | 2026-07-16 |
| 005 | 生成 Streamlit 研究面板 | Claude Sonnet 4.5 | 1.0 | 2026-07-16 |
| 006 | 生成模拟执行器 | Claude Sonnet 4.5 | 1.0 | 2026-07-16 |

## 新增 Prompt 模板

```yaml
---
prompt_id: 007
title: 简短描述
model_tested: <model_name>
date: <YYYY-MM-DD>
version: 1.0
input: {param1, param2}
output: {output1, output2}
known_limitations: "已知限制"
---
```

## Git 工作流

```bash
# 所有 prompt 纳入 git 版本控制
git add prompts/
git commit -m "prompts: add v1.1 of 003-lstm with updated model metadata"
```
