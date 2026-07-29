# Data Issues Log — 2026-07-28

## Panel v3 State (post-backfill)

| Metric | Value |
|---|---|
| Rows | 2,711,084 |
| Symbols | 3,244 |
| Columns | 83 |
| Size | 253 MB |
| Date range | 2023-01-03 ~ 2026-07-28 |
| Stocks >=600 days | 3,158 / 3,244 (97.3%) |
| Median days/stock | 863 |

---

## Open Issues

### 1. 🔴 fina_indicator 列稀疏 (roe 93.5% NaN, eps_yoy 99.1% NaN)

- **范围**: 所有基本面列 (roe, roa, gross_margin, eps_yoy, profit_yoy, net_margin 等)
- **根因**: fina cache 仅 2,582 个文件，其中只有 28 个包含完整列 (basic_eps_yoy, netprofit_yoy, netprofit_margin)。旧 cache 文件缺少这些列。
- **修复方案**: 从 Tushare Pro 全量重建 fina_indicator cache（~3,200 只股票），确保包含完整列集。需要 Tushare token (已有 2000+ 积分)。
- **阻塞**: Tushare api.waditu.com 不稳定（间歇性超时）

### 2. 🔴 margin / northbound / lhb / holdertrade 高 NaN

- **margin_balance**: 96.5% NaN — 仅两融标的
- **northbound**: 100% NaN — 仅沪深港通标的，且需按日期广播
- **lhb_net_buy**: 99.7% NaN — 仅龙虎榜上榜标的
- **sh_net_change_sign**: 99.9% NaN — 仅股东增减持标的
- **说明**: 这些是结构性稀疏，不是 bug。但 2023 年回填日期完全无覆盖，因为 alt data cache 仅覆盖 2024-2026。
- **修复方案**: 如需要 2023 年 alt data，需重新拉取各数据源的 2023 年数据。

### 3. 🟡 3 只股票 fina 拉取失败 (Tushare 超时)

- **股票**: 300148, 300252, 300390
- **原因**: Tushare api.waditu.com 读取超时
- **修复**: 重试即可（间歇性网络问题）

### 4. 🟡 86 只股票 <600 天

- **范围**: 155-598 天
- **说明**: 2024-2025 年新上市 IPO (001xxx, 301xxx 系列)，上市时间不足 600 个交易日。非数据缺失，属正常。
- **无修复必要**

### 5. 🟢 sector_index 2023 年 SW 指数数据缺口

- **sw_ret_1d NaN**: 14.3% (主要是部分行业映射缺失 + 2023 年早期日期)
- **当前状态**: 已从 sw_all cache (1999-2026) 合并，覆盖率 ~86%
- **修复方案**: 补全 SW→东财行业映射（部分行业名不匹配）

### 6. 🟢 daily_basic / stk_limit / cyq 稀疏

- **daily_basic (pe_ttm)**: 82.1% NaN — 旧 cache 格式不统一
- **stk_limit (up_limit_raw)**: 76.2% NaN
- **cyq (benefit_part)**: 83.3% NaN
- **说明**: 回填的 2023 年日期无这些数据。cache 仅覆盖 2024-2026。
- **修复方案**: 同问题 2，扩展 alt data cache 到 2023 年

---

## 数据源可用性状态

| 数据源 | 当前状态 | 备注 |
|---|---|---|
| Baostock | ✅ 可用 | 用于 OHLCV backfill，无频率限制 |
| Tushare Pro | ⚠️ 不稳定 | api.waditu.com 间歇性超时 |
| EastMoney (akshare) | ❌ IP被封 | 东财拒绝连接 |
| SW sector cache | ✅ 完整 | 1999-2026 申万指数数据 |

---

## 已完成修复 (今日)

1. ✅ 去重: 删除 1,407 行重复 (symbol, date)
2. ✅ 600天回填: Baostock 拉取 2,979 只股票，仅 4 只失败
3. ✅ board/turnover_rate NaN 填充
4. ✅ SW 行业分类: Tushare index_classify → 28 个申万一级行业
5. ✅ sector_index 合并: sw_ret_1d, sw_index_close, sw_index_vol
6. ✅ eps_yoy, profit_yoy, net_margin 列添加（但稀疏）
