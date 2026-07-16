# 040 — 模型训练标准 (Model Training Standards)

> **触发条件**: 编写或修改 `lstm_model.py`、`train_model.py`、任何模型训练/评估代码时

## 数据切分规则

**严格按时间切分，严禁随机打乱**（防止时间泄漏）：

| 集合 | 时间范围 | 用途 |
|------|----------|------|
| 训练集 | 2018-01-01 ~ 2020-12-31 | 模型拟合 |
| 验证集 | 2021-01-01 ~ 2021-12-31 | 超参调优 |
| 测试集 | 2022-01-01 ~ 2024-12-31 | 最终评估 |

配置项 (`config/settings.py`)：
```python
DATA_START = date(2018, 1, 1)
TRAIN_END = date(2020, 12, 31)
VAL_END = date(2021, 12, 31)
TEST_START = date(2022, 1, 1)
```

## LSTM 模型规范

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, 20, 30) → (batch, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 取最后一天
```

## 训练循环规范

- Epoch = 50
- 损失函数 = MSE
- 优化器 = Adam (lr=1e-3)
- 保存最佳模型到 `app/models/trained/lstm_best.pth`
- 同时训练 LightGBM 作为对比基线，保存为 `.pkl`
- 验证标准：测试集 Loss < 0.01

## 特征矩阵规范

| 维度 | 值 | 说明 |
|------|-----|------|
| 样本数 | N | 随股票数和时间窗口变化 |
| 时间窗口 | 20 | `WINDOW_DAYS = 20` |
| 特征数 | ≥ 25 | `HORIZON_DAYS = 5` (未来5日收益率) |

**必须包含的技术指标**：
- MACD (DIF, DEA, BAR)
- KDJ (K, D, J)
- BOLL (上轨, 中轨, 下轨)
- RSI

**必须包含的衍生特征**：
- `(close - DIF) / DIF` — 股价与指标偏离度
- `close / MA5 - 1` — 乖离率
- 过去5日指标的线性斜率

## NaN 与除零处理

```python
# 除零防护
def safe_divide(numerator, denominator):
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

# NaN 替换
X = np.nan_to_num(X)  # 特征矩阵输入模型前必须执行
```
