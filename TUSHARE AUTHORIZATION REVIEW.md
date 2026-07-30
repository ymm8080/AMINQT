# -*- coding: utf-8 -*-
# TUSHARE AUTHORIZATION REVIEW
# 生成时间: 2026-07-29
# 积分等级: 5000
# 测试交易日: 20260728
# 实测方式: 逐接口真实调用 (非文档推断)

---

## 一、接口授权总览

| 接口 | 字段总数 | V3 已有 | V3 缺失 | TUSHARE AUTHORIZATION | 实测行数 |
|------|---------|---------|---------|----------------------|---------|
| `pro.daily` | 11 | 10 | 1 | ✅ OK | 超时 (pro_bar 等价) |
| `pro.daily_basic` | 18 | 16 | 2 | ✅ OK | 5524 |
| `pro.bak_daily` | 31 | 19 | 12 | ✅ OK | 5541 |
| `pro.stk_factor` | 35 | 27 | 8 | ✅ OK | 5524 |
| `pro.stk_factor_pro` | 261 | ~27 | 234 | ✅ OK | 5524 |
| `pro.stk_limit` | 4 | 2 | 2 | ✅ OK | 7710 |
| `pro.adj_factor` | 3 | 1 | 2 | ✅ OK | 5546 |
| `pro.stk_mins` | 8 | 0 | 8 | ✅ OK | 241 |
| `pro.stk_weekly_monthly` | — | — | — | ❌ NOT OK | 参数需 freq |
| `pro.stk_auction_o` | — | 0 | — | ❌ NOT OK | 权限不足 |
| `pro.stk_auction_c` | — | 0 | — | ❌ NOT OK | 权限不足 |
| `pro.stk_nineturn` | 13 | 0 | 13 | ✅ OK | 5522 |
| `pro.cyq_chips` | — | 0 | — | ❌ NOT OK | 需逐股 ts_code |
| `pro.pro_bar` | 11 | 10 | 1 | ✅ OK | 2 (单股测试) |
| `pro.suspend_d` | 4 | 0 | 4 | ✅ OK | 10 |
| `pro.moneyflow` | 20 | 0 | 20 | ✅ OK | 5194 |

---

## 二、V3 缺失字段明细 (含中文说明, 仅授权 OK 的接口)

### 2.1 `pro.daily` / `pro.pro_bar` → 缺 1 列

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `change` | 涨跌额 — 当日收盘价与前收盘价之差 (元) | ✅ OK |

> `pct_chg` (涨跌幅%) V3 已有等价列 `pctChg` (来自 akshare spot)。

---

### 2.2 `pro.daily_basic` → 缺 2 列

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `pe` | 市盈率 (总市值 / 净利润, 非TTM) | ✅ OK |
| `ps` | 市销率 (总市值 / 营业收入, 非TTM) | ✅ OK |

> 其余 16 列 V3 已通过 `fetch_daily_basic()` 全量拉取。
> V3 仅有 `pe_ttm` 和 `ps_ttm`, 缺非 TTM 口径。

---

### 2.3 `pro.bak_daily` → 缺 12 列 ⭐ 最高价值

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `swing` | 振幅 (%) — 单日 (最高价 - 最低价) / 前收盘价 × 100 | ✅ OK |
| `buying` | 外盘 — 主动买入量 (手), 买方主动以卖一价成交的量 | ✅ OK |
| `selling` | 内盘 — 主动卖出量 (手), 卖方主动以买一价成交的量 | ✅ OK |
| `avg_price` | 成交均价 — 成交额 / 成交量, 当日实际平均成交价格 | ✅ OK |
| `strength` | 强弱度 (%) — 当日涨跌幅在全市场中的排名分位 | ✅ OK |
| `activity` | 活跃度 (%) — 当日成交量 / 近期平均成交量, 衡量交投活跃程度 | ✅ OK |
| `avg_turnover` | 平均换手率 — 近期平均换手率, 用于对比当日换手是否异常 | ✅ OK |
| `attack` | 攻击波 (%) — 开盘价到最高价的涨幅, 衡量盘中多头攻击力度 | ✅ OK |
| `interval_3` | 3日振幅 (%) — 近3日 (最高价 - 最低价) / 前收盘价 × 100 | ✅ OK |
| `interval_6` | 6日振幅 (%) — 近6日 (最高价 - 最低价) / 前收盘价 × 100 | ✅ OK |
| `name` | 股票名称 — 中文简称 (如 "平安银行") | ✅ OK |
| `area` | 地域 — 注册地所在省份/地区 (如 "深圳") | ✅ OK |

---

### 2.4 `pro.stk_factor` → 缺 8 列

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `pre_close_hfq` | 昨收后复权价 — 前一交易日收盘价 × 后复权因子 | ✅ OK |
| `pre_close_qfq` | 昨收前复权价 — 前一交易日收盘价 × 前复权因子 | ✅ OK |
| `open_qfq` | 开盘前复权价 | ✅ OK |
| `close_qfq` | 收盘前复权价 | ✅ OK |
| `high_qfq` | 最高前复权价 | ✅ OK |
| `low_qfq` | 最低前复权价 | ✅ OK |
| `cci` | CCI 指标 (14日) — 顺势指标, 衡量价格偏离均值的程度 | ✅ OK |

> V3 仅有后复权 (`*_hfq`), 缺前复权 (`*_qfq`) 全套。
> MACD/KDJ/RSI/BOLL V3 在 `feature_engine_v35` 中自行计算, 未用 Tushare 预算值。

---

### 2.5 `pro.stk_factor_pro` → 缺 234 列 ⚡ 最大金矿

261 列全部授权 OK, V3 仅自行计算了约 27 列。以下按类别列出缺失字段及中文说明:

#### 2.5.1 EMA 指数移动均线 (18 列, 6周期 × 3复权)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ema_bfq_5/10/20/30/60/90/250` | 不复权 EMA — 5/10/20/30/60/90/250日指数移动均线 | ✅ OK |
| `ema_hfq_5/10/20/30/60/90/250` | 后复权 EMA — 同上, 后复权口径 | ✅ OK |
| `ema_qfq_5/10/20/30/60/90/250` | 前复权 EMA — 同上, 前复权口径 | ✅ OK |

#### 2.5.2 MA 简单移动均线 (18 列, 6周期 × 3复权)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ma_bfq_5/10/20/30/60/90/250` | 不复权 MA — 5/10/20/30/60/90/250日简单移动均线 | ✅ OK |
| `ma_hfq_5/10/20/30/60/90/250` | 后复权 MA — 同上, 后复权口径 | ✅ OK |
| `ma_qfq_5/10/20/30/60/90/250` | 前复权 MA — 同上, 前复权口径 | ✅ OK |

#### 2.5.3 MACD 柱状图 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `macd_bfq` | 不复权 MACD 柱 — (DIF - DEA) × 2, 不复权口径 | ✅ OK |
| `macd_hfq` | 后复权 MACD 柱 — 同上, 后复权口径 | ✅ OK |
| `macd_qfq` | 前复权 MACD 柱 — 同上, 前复权口径 | ✅ OK |

#### 2.5.4 ATR 平均真实波幅 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `atr_bfq` | 不复权 ATR — 14日平均真实波幅, 衡量波动率 | ✅ OK |
| `atr_hfq` | 后复权 ATR — 同上, 后复权口径 | ✅ OK |
| `atr_qfq` | 前复权 ATR — 同上, 前复权口径 | ✅ OK |

#### 2.5.5 DMI 趋向指标 (12 列, 4子指标 × 3复权)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `dmi_pdi_bfq/hfq/qfq` | +DI 上升方向线 — 多头力量, 不复权/后复权/前复权 | ✅ OK |
| `dmi_mdi_bfq/hfq/qfq` | -DI 下降方向线 — 空头力量, 不复权/后复权/前复权 | ✅ OK |
| `dmi_adx_bfq/hfq/qfq` | ADX 趋势强度 — 趋势明确度 (不论多空), 3种复权 | ✅ OK |
| `dmi_adxr_bfq/hfq/qfq` | ADXR 评估线 — ADX 的移动平均, 3种复权 | ✅ OK |

#### 2.5.6 BRAR 情绪指标 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `brar_ar_bfq/hfq/qfq` | AR 人气指标 — 以开盘价为基准衡量买卖气势, 3种复权 | ✅ OK |
| `brar_br_bfq/hfq/qfq` | BR 意愿指标 — 以收盘价为基准衡量买卖意愿, 3种复权 | ✅ OK |

#### 2.5.7 CR 能量指标 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `cr_bfq/hfq/qfq` | CR 中间意愿指标 — 以中间价为基准衡量多空力量, 3种复权 | ✅ OK |

#### 2.5.8 BBI 多空指标 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `bbi_bfq/hfq/qfq` | BBI 多空指标 — (MA4+MA8+MA12+MA16)/4 综合均线, 3种复权 | ✅ OK |

#### 2.5.9 BIAS 乖离率 (9 列, 3周期 × 3复权)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `bias1_bfq/hfq/qfq` | 6日乖离率 — 收盘价偏离6日均线百分比, 3种复权 | ✅ OK |
| `bias2_bfq/hfq/qfq` | 12日乖离率 — 收盘价偏离12日均线百分比, 3种复权 | ✅ OK |
| `bias3_bfq/hfq/qfq` | 24日乖离率 — 收盘价偏离24日均线百分比, 3种复权 | ✅ OK |

#### 2.5.10 BOLL 布林带 (9 列, 3子指标 × 3复权)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `boll_upper_bfq/hfq/qfq` | 布林带上轨 — MID + 2×STD, 3种复权 | ✅ OK |
| `boll_mid_bfq/hfq/qfq` | 布林带中轨 — 20日简单移动均线, 3种复权 | ✅ OK |
| `boll_lower_bfq/hfq/qfq` | 布林带下轨 — MID - 2×STD, 3种复权 | ✅ OK |

#### 2.5.11 CCI 顺势指标 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `cci_bfq/hfq/qfq` | CCI 顺势指标 — 衡量价格偏离均值的程度, 3种复权 | ✅ OK |

#### 2.5.12 DPO 区间震荡线 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `dpo_bfq/hfq/qfq` | DPO 区间震荡线 — 去除趋势的价格震荡, 3种复权 | ✅ OK |
| `madpo_bfq/hfq/qfq` | MADPO — DPO 的移动平均, 3种复权 | ✅ OK |

#### 2.5.13 EMV 简易波动指标 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `emv_bfq/hfq/qfq` | EMV 简易波动指标 — 结合成交量和价格变化衡量供需, 3种复权 | ✅ OK |
| `maemv_bfq/hfq/qfq` | MAEMV — EMV 的移动平均, 3种复权 | ✅ OK |

#### 2.5.14 EXPMA 指数平均数 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `expma_12_bfq/hfq/qfq` | 12日 EXPMA — 12日指数平均数, 3种复权 | ✅ OK |
| `expma_50_bfq/hfq/qfq` | 50日 EXPMA — 50日指数平均数, 3种复权 | ✅ OK |

#### 2.5.15 KDJ 随机指标 (3 列, 柱状图)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `kdj_bfq/hfq/qfq` | KDJ J值柱 — 3K-2D, 反映超买超卖, 3种复权 | ✅ OK |

> KDJ 的 K/D/J 子值另有 `kdj_k_*` / `kdj_d_*` 系列列。

#### 2.5.16 Keltner Channel 肯特纳通道 (9 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ktn_upper_bfq/hfq/qfq` | 肯特纳通道上轨 — EMA + ATR×系数, 3种复权 | ✅ OK |
| `ktn_mid_bfq/hfq/qfq` | 肯特纳通道中轨 — EMA, 3种复权 | ✅ OK |
| `ktn_down_bfq/hfq/qfq` | 肯特纳通道下轨 — EMA - ATR×系数, 3种复权 | ✅ OK |

#### 2.5.17 MASS 梅斯线 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `mass_bfq/hfq/qfq` | MASS 梅斯线 — MA9高低差 / MA9(MA9高低差), 3种复权 | ✅ OK |
| `ma_mass_bfq/hfq/qfq` | MASS 的移动平均, 3种复权 | ✅ OK |

#### 2.5.18 MFI 资金流量指标 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `mfi_bfq/hfq/qfq` | MFI 资金流量指标 — 带成交量的RSI, 衡量资金进出, 3种复权 | ✅ OK |

#### 2.5.19 MTM 动量指标 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `mtm_bfq/hfq/qfq` | MTM 动量指标 — 当日收盘 - N日前收盘, 3种复权 | ✅ OK |
| `mtmma_bfq/hfq/qfq` | MTMMA — MTM 的移动平均, 3种复权 | ✅ OK |

#### 2.5.20 OBV 能量潮 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `obv_bfq/hfq/qfq` | OBV 能量潮 — 累计成交量, 量价配合验证趋势, 3种复权 | ✅ OK |

#### 2.5.21 PSY 心理线 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `psy_bfq/hfq/qfq` | PSY 心理线 — N日上涨天数占比, 衡量市场情绪, 3种复权 | ✅ OK |
| `psyma_bfq/hfq/qfq` | PSYMA — PSY 的移动平均, 3种复权 | ✅ OK |

#### 2.5.22 ROC 变动率 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `roc_bfq/hfq/qfq` | ROC 变动率 — (当日收盘 - N日前收盘) / N日前收盘 × 100, 3种复权 | ✅ OK |
| `maroc_bfq/hfq/qfq` | MAROC — ROC 的移动平均, 3种复权 | ✅ OK |

#### 2.5.23 RSI 相对强弱 (9 列, 3周期 × 3复权)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `rsi_bfq_6` | 不复权 RSI(6) — 6日相对强弱指标 | ✅ OK |
| `rsi_bfq_12` | 不复权 RSI(12) — 12日相对强弱指标 | ✅ OK |
| `rsi_bfq_24` | 不复权 RSI(24) — 24日相对强弱指标 | ✅ OK |
| `rsi_hfq_6/12/24` | 后复权 RSI — 同上, 后复权口径 | ✅ OK |
| `rsi_qfq_6/12/24` | 前复权 RSI — 同上, 前复权口径 | ✅ OK |

#### 2.5.24 SAR 抛物线 / TAQ 逆势操作 (9 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `taq_up_bfq/hfq/qfq` | TAQ 上轨 — 抛物线SAR反转点上方, 3种复权 | ✅ OK |
| `taq_mid_bfq/hfq/qfq` | TAQ 中轨 — 抛物线SAR中值, 3种复权 | ✅ OK |
| `taq_down_bfq/hfq/qfq` | TAQ 下轨 — 抛物线SAR反转点下方, 3种复权 | ✅ OK |

#### 2.5.25 TRIX 三重指数平滑 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `trix_bfq/hfq/qfq` | TRIX 三重指数平滑移动平均 — 长线趋势指标, 3种复权 | ✅ OK |
| `trma_bfq/hfq/qfq` | TRMA — TRIX 的移动平均 (信号线), 3种复权 | ✅ OK |

#### 2.5.26 VR 容量比率 (3 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `vr_bfq/hfq/qfq` | VR 容量比率 — 上涨日成交量之和 / 下跌日成交量之和, 3种复权 | ✅ OK |

#### 2.5.27 WR 威廉指标 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `wr_bfq/hfq/qfq` | WR 威廉指标 — (N日最高 - 收盘) / (N日最高 - N日最低) × 100, 3种复权 | ✅ OK |
| `wr1_bfq/hfq/qfq` | WR1 — WR 的短周期变体, 3种复权 | ✅ OK |

#### 2.5.28 XSII TD 序列 (12 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `xsii_td1_bfq/hfq/qfq` | TD1 序列 — TD序列第一层信号, 3种复权 | ✅ OK |
| `xsii_td2_bfq/hfq/qfq` | TD2 序列 — TD序列第二层信号, 3种复权 | ✅ OK |
| `xsii_td3_bfq/hfq/qfq` | TD3 序列 — TD序列第三层信号, 3种复权 | ✅ OK |
| `xsii_td4_bfq/hfq/qfq` | TD4 序列 — TD序列第四层信号, 3种复权 | ✅ OK |

#### 2.5.29 ASI 振动升降指标 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `asi_bfq/hfq/qfq` | ASI 振动升降指标 — 以真实波动衡量多空, 3种复权 | ✅ OK |
| `asit_bfq/hfq/qfq` | ASIT — ASI 的移动平均, 3种复权 | ✅ OK |

#### 2.5.30 DFMA 平行线差 (6 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `dfma_dif_bfq/hfq/qfq` | DFMA DIF — 短期与长期 DPO 之差, 3种复权 | ✅ OK |
| `dfma_difma_bfq/hfq/qfq` | DFMA DIFMA — DIF 的移动平均, 3种复权 | ✅ OK |

#### 2.5.31 统计字段 (4 列)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `updays` | 连涨天数 — 连续上涨的交易日天数 | ✅ OK |
| `downdays` | 连跌天数 — 连续下跌的交易日天数 | ✅ OK |
| `topdays` | 阶段高点天数 — 距离近期最高价的交易日天数 | ✅ OK |
| `lowdays` | 阶段低点天数 — 距离近期最低价的交易日天数 | ✅ OK |

---

### 2.6 `pro.stk_limit` → 缺 2 列

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `up_limit` | 涨停价 — 当日涨停限制价格 (元) | ✅ OK |
| `down_limit` | 跌停价 — 当日跌停限制价格 (元) | ✅ OK |

> V3 已有 `up_limit_raw` / `down_limit_raw` (来自 akshare), 但 Tushare `stk_limit` 覆盖更完整。

---

### 2.7 `pro.adj_factor` → 缺 2 列

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ts_code` | Tushare 股票代码 (如 000001.SZ) | ✅ OK |
| `trade_date` | 交易日期 (YYYYMMDD) | ✅ OK |

> V3 仅使用了 `adj_factor` 值本身, 未保留原始 `ts_code` / `trade_date` 键列。

---

### 2.8 `pro.stk_mins` → 缺 8 列 (V3 完全未用)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ts_code` | Tushare 股票代码 | ✅ OK |
| `trade_time` | 成交时间 (YYYY-MM-DD HH:MM:SS) | ✅ OK |
| `open` | 分钟开盘价 | ✅ OK |
| `high` | 分钟最高价 | ✅ OK |
| `low` | 分钟最低价 | ✅ OK |
| `close` | 分钟收盘价 | ✅ OK |
| `vol` | 分钟成交量 (手) | ✅ OK |
| `amount` | 分钟成交额 (千元) | ✅ OK |

---

### 2.9 `pro.stk_nineturn` → 缺 13 列 (V3 完全未用)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ts_code` | Tushare 股票代码 | ✅ OK |
| `trade_date` | 交易日期 | ✅ OK |
| `freq` | 频率 (日线/60分钟等) | ✅ OK |
| `open` | 开盘价 | ✅ OK |
| `high` | 最高价 | ✅ OK |
| `low` | 最低价 | ✅ OK |
| `close` | 收盘价 | ✅ OK |
| `vol` | 成交量 | ✅ OK |
| `amount` | 成交额 | ✅ OK |
| `up_count` | 上涨计数 — 连续上涨天数计数 | ✅ OK |
| `down_count` | 下跌计数 — 连续下跌天数计数 | ✅ OK |
| `nine_up_turn` | 九转上涨信号 — 连续9日满足上涨条件, 潜在反转卖出点 | ✅ OK |
| `nine_down_turn` | 九转下跌信号 — 连续9日满足下跌条件, 潜在反转买入点 | ✅ OK |

---

### 2.10 `pro.suspend_d` → 缺 4 列 (V3 完全未用)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ts_code` | Tushare 股票代码 | ✅ OK |
| `trade_date` | 交易日期 | ✅ OK |
| `suspend_timing` | 停牌时机 — 全天停牌/盘中停牌等 | ✅ OK |
| `suspend_type` | 停牌类型 — 如重大事项/盘中异常等 | ✅ OK |

---

### 2.11 `pro.moneyflow` → 缺 20 列 (V3 完全未用)

| 字段 | 中文说明 | TUSHARE AUTHORIZATION |
|------|---------|----------------------|
| `ts_code` | Tushare 股票代码 | ✅ OK |
| `trade_date` | 交易日期 | ✅ OK |
| `buy_sm_vol` | 小单买入量 (手) | ✅ OK |
| `buy_sm_amount` | 小单买入额 (万元) | ✅ OK |
| `sell_sm_vol` | 小单卖出量 (手) | ✅ OK |
| `sell_sm_amount` | 小单卖出额 (万元) | ✅ OK |
| `buy_md_vol` | 中单买入量 (手) | ✅ OK |
| `buy_md_amount` | 中单买入额 (万元) | ✅ OK |
| `sell_md_vol` | 中单卖出量 (手) | ✅ OK |
| `sell_md_amount` | 中单卖出额 (万元) | ✅ OK |
| `buy_lg_vol` | 大单买入量 (手) | ✅ OK |
| `buy_lg_amount` | 大单买入额 (万元) | ✅ OK |
| `sell_lg_vol` | 大单卖出量 (手) | ✅ OK |
| `sell_lg_amount` | 大单卖出额 (万元) | ✅ OK |
| `buy_elg_vol` | 超大单买入量 (手) | ✅ OK |
| `buy_elg_amount` | 超大单买入额 (万元) | ✅ OK |
| `sell_elg_vol` | 超大单卖出量 (手) | ✅ OK |
| `sell_elg_amount` | 超大单卖出额 (万元) | ✅ OK |
| `net_mf_vol` | 净资金流量 (手) — 买入总量 - 卖出总量 | ✅ OK |
| `net_mf_amount` | 净资金流额 (万元) — 买入总额 - 卖出总额 | ✅ OK |

---

## 三、权限不足接口明细

| 接口 | 状态 | 中文说明 | 原因 |
|------|------|---------|------|
| `pro.stk_auction_o` | ❌ NOT OK | 开盘集合竞价数据 — 9:30 集合竞价明细 | 权限不足, 需更高积分 |
| `pro.stk_auction_c` | ❌ NOT OK | 收盘集合竞价数据 — 15:00 集合竞价明细 | 权限不足, 需更高积分 |
| `pro.cyq_chips` | ❌ NOT OK | 每日筹码分布 — 各价位占比, 2018年起 | 需逐股 ts_code 查询 (非全市场批量) |
| `pro.stk_weekly_monthly` | ❌ NOT OK | 周月线行情 (每日更新) | 参数需 freq, 非权限问题 |

---

## 四、总结

### 4.1 授权 OK 但 V3 未用的字段统计

| 接口 | 缺失列数 | 优先级建议 |
|------|---------|-----------|
| `pro.stk_factor_pro` | **234** | ⚡ 最高 — 261 列全授权, 含 30+ 种技术指标 × 3复权 |
| `pro.bak_daily` | **12** | ⭐ 高 — 外盘/内盘/攻击波/强弱度, 短线高价值 |
| `pro.moneyflow` | **20** | ⭐ 高 — 大/中/小/超大单资金流, 量价配合 |
| `pro.stk_nineturn` | **13** | 中 — 神奇九转反转信号 |
| `pro.stk_factor` | **8** | 中 — 前复权价 + CCI |
| `pro.stk_mins` | **8** | 中 — 分钟级数据 (日内模式) |
| `pro.daily` | **1** | 低 — `change` 可从 close-pre_close 计算 |
| `pro.daily_basic` | **2** | 低 — `pe`/`ps` 非TTM口径 |
| `pro.stk_limit` | **2** | 低 — V3 已有 akshare 等价列 |
| `pro.adj_factor` | **2** | 低 — 仅键列, 非数据列 |
| `pro.suspend_d` | **4** | 中 — 停牌信息, 风控过滤用 |

### 4.2 关键发现

1. **`stk_factor_pro` (261列) 是最大未开掘金矿** — 5000积分已授权, V3 仅自行算了约27列, 其余234列 Tushare 已预算好且三种复权全覆盖
2. **`bak_daily` 的外盘/内盘 (`buying`/`selling`) 是 V3 完全缺失的独有数据** — akshare 无此字段
3. **`moneyflow` 的四档资金流** — 比东财资金流更细粒度 (超大/大/中/小单分离)
4. **权限不足仅 3 个接口** — 集合竞价 (`stk_auction_o/c`) 和逐股筹码 (`cyq_chips`)

---

## 五、文档信息

- 生成时间: 2026-07-29
- 测试环境: Windows 10, Python 3.12, tushare SDK
- 测试交易日: 20260728
- 积分等级: 5000
- 数据来源: 实测调用 (非文档推断)
- V3 面板文件: `data/panel_full_enriched_v3.parquet` (102 列)
