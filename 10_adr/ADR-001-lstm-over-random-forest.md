# ADR-001: LSTM 时序模型优于随机森林

**状态**: Accepted  
**日期**: 2026-07-16

## 背景 (Context)

系统需要从日 K 线图形中提取有效因子并预测未来 5 日收益率。核心选择是：
- K 线数据本质上是**时间序列**，前后日有关联性
- 需要捕捉 20 天窗口内的**时序模式**（趋势、拐点）
- 传统机器学习模型（如随机森林）将特征视为独立维度，忽略时序结构

## 决策 (Decision)

选择 **LSTM (长短期记忆网络)** 作为主模型，LightGBM 作为对比基线。

架构：
1. LSTM 输入: `(batch, 20_days, 30_features)` — 保留时序结构
2. LSTM 隐藏层: 64 维, 2 层, Dropout=0.2
3. 全连接输出: 未来 5 日收益率预测
4. 同时训练 LightGBM 作为对比基线

## 备选方案 (Alternatives Considered)

1. **随机森林**: 特征独立假设，忽略时序 — 放弃
2. **ARIMA**: 线性假设，无法捕捉非线性图形模式 — 放弃
3. **Transformer**: 序列建模能力强，但数据量不足时易过拟合 — 备选 (未来数据充足时升级)
4. **LSTM (选中)**: 时序建模成熟，小数据量表现稳定，社区资源丰富

## 影响 (Consequences)

### 正面影响
- 保留 K 线的时序结构，捕捉趋势和拐点
- LSTM 适合 20 天窗口的短序列建模
- 与 LightGBM 对比可验证时序模型的价值

### 负面影响
- LSTM 训练速度慢于 LightGBM
- 超参调优复杂 (hidden_dim, num_layers, dropout, lr)
- 小数据集易过拟合，需严格 Dropout 和早停

### 缓解措施
- Early stopping (验证集 Loss 不下降则停止)
- Dropout=0.2 防过拟合
- 保存最佳模型 (`lstm_best.pth`)，不是最后一个 epoch

## 合规要求 (Compliance)

- 模型定义: `app/models/lstm_model.py`
- 训练脚本: `scripts/train_model.py`
- 权重保存: `app/models/trained/lstm_best.pth`
- 验证标准: 测试集 MSE Loss < 0.01
- 数据切分: 训练 2018-2020, 验证 2021, 测试 2022-2024

## 参考 (References)

- ARCHITECTURE §5 技术栈: PyTorch (LSTM), LightGBM (对比基线)
- PROMPT_CONTENT §2.A 未来函数防止
