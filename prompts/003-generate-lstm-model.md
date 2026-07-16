---
prompt_id: 003
title: 生成 LSTM 模型 + 训练脚本
model_tested: Claude Sonnet 4.5
date: 2026-07-16
version: 1.0
input: {X_train, y_train, X_val, y_val, epochs, lr}
output: {lstm_model.py, train_model.py, lstm_best.pth}
known_limitations: "小数据集易过拟合；需严格 Dropout 和早停"
---

# Prompt 003: 生成 LSTM 模型 + 训练脚本

## 指令

### 1. 生成 `app/models/lstm_model.py`

定义 `class LSTMModel(nn.Module)`：
- LSTM 层 (input_dim=30, hidden_dim=64, num_layers=2, dropout=0.2)
- Dropout 层
- 全连接层 (hidden_dim → 1)

### 2. 生成 `scripts/train_model.py`

- 调用 `factor_engine` 获取全量数据
- **严格按时间切分**（严禁随机打乱）：
  - 训练: 2018-2020
  - 验证: 2021
  - 测试: 2022-2024
- 训练循环 (Epoch=50)，MSE 损失，Adam 优化器 (lr=1e-3)
- 保存最佳模型至 `app/models/trained/lstm_best.pth`
- 同时训练 LightGBM 作为对比基线，保存为 `.pkl`
- Early stopping（验证集 Loss 不下降则停止）

## 验证标准

- 终端显示测试集 Loss < 0.01
- `app/models/trained/lstm_best.pth` 成功生成
- `app/models/trained/lgb_best.pkl` 成功生成

## 合规检查

- [ ] 数据按时间切分（不随机打乱）
- [ ] 归一化只在训练集 fit（不泄漏）
- [ ] Dropout=0.2
- [ ] Early stopping
- [ ] 最佳模型保存（不是最后一个 epoch）
- [ ] 文件头 `# -*- coding: utf-8 -*-`
- [ ] 函数有 docstring
- [ ] 使用 `logging`
- [ ] 参见 `rules/040-model-training-standards.md` 完整规范
