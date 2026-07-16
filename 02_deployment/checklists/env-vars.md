# 环境变量清单 (Environment Variables)

> **位置**: `02_deployment/checklists/env-vars.md`
> **最后更新**: 2026-07-16

## 环境变量一览

| 变量名 | 默认值 | 可选值 | 说明 |
|--------|--------|--------|------|
| `AMINQT_DATA_SOURCE` | `akshare` | `akshare` \| `ifind` | 数据源选择 |
| `AMINQT_EXEC_MODE` | `manual` | `manual` \| `auto` | 执行模式 (手动/自动) |
| `AMINQT_BROKER` | `sim` | `sim` \| `xt` | 券商接口 (模拟/实盘) |
| `IFIND_USER` | (空) | — | iFinD 用户名 |
| `IFIND_PASSWORD` | (空) | — | iFinD 密码 |

## .env 文件模板

```bash
# .env (此文件不入 git)
AMINQT_DATA_SOURCE=akshare
AMINQT_EXEC_MODE=manual
AMINQT_BROKER=sim

# iFinD 凭据 (仅使用 iFinD 数据源时需要)
# IFIND_USER=your_username
# IFIND_PASSWORD=your_password
```

## 安全规则

1. **.env 不入 git**: `.gitignore` 必须包含 `.env`
2. **凭据不从代码读取**: 所有凭据通过 `os.getenv()` 获取
3. **默认安全**: 所有执行相关变量默认为最安全选项 (`manual`, `sim`)
4. **实盘切换需显式操作**: 从 `sim` 切到 `xt` 需修改环境变量并重启

## 切换到实盘

```bash
# 1. 修改 .env
# AMINQT_EXEC_MODE=auto
# AMINQT_BROKER=xt

# 2. 重启服务
# 确认所有部署前检查通过后执行

# 3. 验证券商连接
python -c "
from services.xt_executor import XtExecutor
ex = XtExecutor()
print('持仓:', ex.get_positions())
"

# 4. 切回模拟（如需紧急降级）
# 修改 .env: AMINQT_BROKER=sim
# 重启服务
```
