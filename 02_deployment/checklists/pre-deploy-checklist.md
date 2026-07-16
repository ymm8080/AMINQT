# 部署前检查清单 (Pre-Deployment Checklist)

> **位置**: `02_deployment/checklists/pre-deploy-checklist.md`
> **最后更新**: 2026-07-16

## 环境检查

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| Python 版本 | `python --version` | 3.9+ |
| 磁盘空间 | `df -h .` / `Get-PSDrive .` | > 1GB 可用 |
| 内存 | `systeminfo` / `free -h` | > 2GB 可用 |
| 依赖安装 | `pip install -r requirements.txt` | 无报错 |
| ruff 检查 | `ruff check app/ scripts/ services/` | 零错误 |
| pytest | `pytest tests/ -v` | 全部通过 |

## 数据检查

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| CSV 文件存在 | `ls data/raw/*.csv \| wc -l` | ≥ 5 个文件 |
| 数据完整性 | `python -c "import pandas as pd; print(pd.read_csv('data/raw/600519.csv').shape)"` | > 1000 行 |
| 列名正确 | `python -c "import pandas as pd; print(list(pd.read_csv('data/raw/600519.csv').columns))"` | 含 date, open, close... |
| 日内数据 | `ls data/intraday/` | 目录存在 |

## 模型检查

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| LSTM 权重 | `ls -la app/models/trained/lstm_best.pth` | 文件存在 |
| LightGBM 权重 | `ls -la app/models/trained/lgb_best.pkl` | 文件存在 |
| 模型加载 | `python -c "from app.models.lstm_model import LSTMModel; import torch; m=LSTMModel(); m.load_state_dict(torch.load('app/models/trained/lstm_best.pth')); print('OK')"` | OK |

## 配置检查

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| .env 文件 | `cat .env` | 存在且非空 |
| 执行模式 | `echo $AMINQT_EXEC_MODE` | manual (安全优先) |
| 券商模式 | `echo $AMINQT_BROKER` | sim (安全优先) |
| 数据源 | `echo $AMINQT_DATA_SOURCE` | akshare 或 ifind |

## 安全检查

| 检查项 | 预期结果 |
|--------|----------|
| 无硬编码凭据 | 代码中无密码/API Key |
| .env 不入 git | `.gitignore` 包含 `.env` |
| 审计日志目录可写 | `logs/` 目录存在且有写权限 |
| 备份目录存在 | `data/backup/`, `app/models/trained/backup/` |

## 功能验证

```bash
# 1. 启动 API
uvicorn app.main:app --reload &

# 2. 验证 API 文档可访问
curl http://127.0.0.1:8000/docs
# 预期: Swagger UI 页面

# 3. 验证选股功能
curl -X POST http://127.0.0.1:8000/api/v1/select
# 预期: 5秒内返回 JSON

# 4. 验证调度器
grep "调度器" logs/aminqt.log
# 预期: "调度器启动 (每日 14:50 Asia/Shanghai)"

# 5. 启动 Streamlit
streamlit run app/streamlit_app.py &
# 预期: 浏览器打开可交互图表

# 6. 停止服务
kill %1 %2
```

## 上线决策

- [ ] 所有环境检查通过
- [ ] 所有数据检查通过
- [ ] 所有模型检查通过
- [ ] 所有配置检查通过
- [ ] 所有安全检查通过
- [ ] 所有功能验证通过
- [ ] 执行模式确认为 manual（首次上线）
- [ ] 券商模式确认为 sim（首次上线）
- [ ] 备份已创建
- [ ] 审计日志目录就绪

**所有项打勾后方可上线。**
