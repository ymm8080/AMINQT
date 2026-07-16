# Skill: verify-before-done (完成前验证)

> **来源**: 从 EWM Robot 项目同名技能适配
> **用途**: 声称"完成"前，必须运行验证命令并展示证据

## 核心规则

**绝对禁止在未运行验证的情况下声称"完成"。**

声称完成前必须：
1. 运行验证命令
2. 展示输出给用户
3. 确认结果符合要求
4. 然后才可以说"完成"

## 验证命令

### 代码修改后
```bash
# 1. Lint 检查
ruff check app/ scripts/ services/ tests/

# 2. 类型检查
python -m py_compile app/main.py

# 3. 单元测试
pytest tests/ -v
```

### 数据下载后
```bash
# 验证 CSV 文件存在且有数据
ls -la data/raw/
python -c "
import pandas as pd
df = pd.read_csv('data/raw/600519.csv')
assert len(df) > 100, '数据行数不足'
assert 'date' in df.columns, '缺少 date 列'
print(f'OK: {len(df)} rows')
"
```

### 模型训练后
```bash
# 验证权重文件存在
ls -la app/models/trained/lstm_best.pth

# 验证模型可加载
python -c "
from app.models.lstm_model import LSTMModel
import torch
m = LSTMModel()
m.load_state_dict(torch.load('app/models/trained/lstm_best.pth'))
print('模型加载成功')
"

# 验证 Loss < 0.01
# 从训练日志中提取
grep "val_loss" logs/train.log | tail -1
```

### 选股接口
```bash
# 验证响应时间 < 5 秒
time curl -X POST http://127.0.0.1:8000/api/v1/select

# 验证返回 JSON
curl -s -X POST http://127.0.0.1:8000/api/v1/select | python -m json.tool
```

### API 服务启动
```bash
# 验证服务可访问
curl http://127.0.0.1:8000/docs

# 验证健康检查（如有）
curl http://127.0.0.1:8000/health
```

### Streamlit 面板
```bash
# 验证面板可访问
curl http://localhost:8501

# 手动验证: 浏览器打开后能看到 K线图和指标
```

## 验证报告格式

完成任务时，附带验证报告：

```
## 验证结果

| 验证项 | 命令 | 结果 |
|--------|------|------|
| Lint | `ruff check app/` | ✅ 零错误 |
| Tests | `pytest tests/ -v` | ✅ 全部通过 |
| 模型加载 | `python -c "..."` | ✅ 成功 |
| API 响应 | `curl ...` | ✅ < 5s, JSON 正常 |
```

## 违反规则

如果声称"完成"但未展示验证证据：
- 视为违反铁律
- 追加 `[RULES I BROKE]: verify-before-done — 未运行验证命令`
