# ADR-002: akshare 作为主数据源 (iFinD 备用)

**状态**: Accepted  
**日期**: 2026-07-16

## 背景 (Context)

系统需要下载 A 股历史日 K 线数据。数据源选择考虑：
- 数据质量和完整性
- 免费 vs 付费
- API 稳定性和限速
- 是否需要实盘行情

## 决策 (Decision)

选择 **akshare** 作为开发/测试阶段的主数据源，**iFinD (同花顺)** 作为生产/实盘数据源。

通过**策略模式**（数据适配器）实现数据源切换：
- `data/adapters/akshare_adapter.py` — akshare 实现
- `data/adapters/ifind_adapter.py` — iFinD 实现
- `data/adapters/base.py` — 抽象基类
- 环境变量 `AMINQT_DATA_SOURCE=akshare|ifind` 控制切换

## 备选方案 (Alternatives Considered)

1. **Tushare**: 免费 token 有积分限制，高频调用受限 — 备用
2. **Baostock**: 数据更新慢，不如 akshare 全面 — 放弃
3. **akshare (选中)**: 完全免费，无需 token，接口丰富 — 主数据源
4. **iFinD (选中)**: 同花顺专业数据，质量高，需付费 — 生产数据源

## 影响 (Consequences)

### 正面影响
- 开发阶段零成本（akshare 免费）
- 适配器模式解耦数据源，切换零代码改动
- iFinD 保证实盘数据质量

### 负面影响
- akshare 反爬限速，需 `time.sleep(0.5)`
- akshare 列名为中文，需重命名映射（见规则 020）
- iFinD 需要凭据管理（环境变量）

### 缓解措施
- 反爬延时: `DOWNLOAD_SLEEP_SEC = 0.5`
- 列名映射: 适配器内部完成，核心代码无感知
- 凭据: `os.getenv("IFIND_USER")`, `os.getenv("IFIND_PASSWORD")`
- 内存缓存: 启动时全量加载 CSV，选股不触发网络请求

## 合规要求 (Compliance)

- 适配器基类: `data/adapters/base.py`
- akshare 适配器: `data/adapters/akshare_adapter.py`
- iFinD 适配器: `data/adapters/ifind_adapter.py`
- 环境变量: `AMINQT_DATA_SOURCE` (默认 `akshare`)
- 列名映射: 见 `rules/020-data-format-mapping.md`

## 参考 (References)

- ARCHITECTURE §5 技术栈: akshare (免费) / Tushare (备用)
- PROMPT_CONTENT §1 数据格式映射
