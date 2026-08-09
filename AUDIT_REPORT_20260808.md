# AMINQT 深度审计报告 — 2026-08-08

> 审计方式: 5 路并行子智能体 (前瞻偏差 / 资金数值铁律 / 数据配置合规 / 安全复验 / 测试代码质量) + 关键项人工读码复核
> 基线: 对比 2026-08-07 AUDIT_REPORT.md 复验 + 首次深挖 ML 管道 (pipeline1 / pipeline_parallel / backtest)
> 范围: 659 个 .py, ~140k 行, 869 tracked files

---

## 一、CRITICAL (3)

**C1. 真实 Tushare token 已入 git 历史 — 旧报告"从未提交"结论错误**
`AUDIT_REPORT.md:34` 曾明文写入真实 token (`86237a96` 提交引入, 仍存于 `5a47a966`/`86237a96`/`7608b0ef` 历史)。
`.env` 现已清空 (`TUSHARE_TOKEN=` 空值) 不缓解历史泄露。
→ 已办: 报告中 token 已打码 (2026-08-08)。待办: **轮换 token** (Tushare 后台 regen) + 决定是否 purge 历史。

**C2. 回测卖出口径"当日收盘裁决 + 当日开盘成交"未来泄漏 — 已修复**
`app/pipeline1/backtest_v35.py` 默认 `exec_session="AM"` (:50):
- 旧: 止损/移动止盈/概率衰减/到期四类退出用**当日** close/high 裁决, 却按**当日开盘**成交 (`_exec_sell_price` :126) → 先见收盘再回开盘逃顶, 系统性高估回测收益。
- 新: 卖出裁决改用 **T-1 收盘** 信息, 当日开盘成交; 同步修掉 section 2 预算估值用当日收盘的同类泄漏 (等权仓位估值改用 T-1 收盘)。
- 影响面: 生产链路 `frontier_routes` / `param_tuner` / `page_backtest` / `page_eval` 全走此引擎。
- 验证: `test_backtest_tuner` + `test_end_to_end_v35` + `test_pipeline1_v35/v38/v38_appendix_de/rule_engine_v2/sell_signal` 全过; 新增回归测试 `test_am_exit_uses_t1_close_no_same_day_lookahead`。

**C3. API 认证"代码层有、运行层关"**
`app/main.py:27-78` X-API-Key 中间件已加, 但 `.env:21 AMINQT_API_KEY=` 为空 → 运行期认证实际禁用。
`/api/frontier/pipeline/trigger`、`append-daily`、`/api/v1/execute` 仍可匿名调用 (本地绑定缓解)。
→ 已办: `monitor_catpaw.py:37` + `catpaw_monitor_config.json` API 启动改绑 `127.0.0.1` (消费方全为本地 automation/dashboard/Vite)。残留: `AMINQT_API_KEY` 仍空 → 本地匿名可用; streamlit 8501 仍绑 0.0.0.0 (未纳 C3 范围, 如不需局域网访问建议同改)。

## 二、HIGH (6)

| # | 位置 | 问题 |
|---|------|------|
| H1 | `app/backtest/engine.py:503-506` | 持仓替换: 用当日收盘算盈亏决定替换, 替换卖出却按当日开盘成交 (乐观方向) — **已修**: 盈亏裁决改用 T-1 收盘 (替换卖按当日开盘成交, 当日收盘裁决即前视); T-1 缺行 (次新/停牌) 回退当日开盘价 (同为开盘已知信息, 非前视). 回归测试 `test_conflict_replacement_uses_prev_close_not_today_close` |
| H2 | `app/backtest/data_loader.py:160-173`, `app/pipeline1/data_supply.py`, `app/intraday/v51/data_5min.py:22-49` | OHLCV 校验孤岛 — **已修**: `validate_ohlcv` 下沉到 `data_supply.fetch_daily/fetch_history` (缓存命中+新拉取双路径, 失败以 DataSupplyError 上抛) 与 `data_5min.load` (唯一咽喉, 双路径都过); backtest 路径已有 DataValidator E003/E004 优雅降级, 不重复加严格 raise |
| H3 | `scripts/build_features.py:173,184`, `train_predict_main.py:153,428`, `train_predict_dual.py:94` | `np.random.choice` 抽样本股无 seed → 训练样本不可复现 (LGBM 有 random_state=42, 入口抽样无) — **已修**: 3 脚本入口抽样前加 `np.random.seed(42)` |
| H4 | `mask_recent_days=6`(20+处), `CLS_THRESHOLD=0.005`(8处), `cleaning_pipeline.py:33-41` 涨跌停硬编码 0.10/0.20+"2020-08-24", `OOS_DAYS=250`(6处) | — **已修** (生产路径): `training_config.yaml labels.mask_recent_days/cls_threshold` + `trading_config.yaml limit_rules`; `label_engine`(CLS_THRESHOLD/MASK_RECENT_DAYS 单一真相源) + `train_runner`/`build_features` 引用 config, `cleaning_pipeline.get_limit_pct` 读 config, 行为全不变 (69 测试过)。审计"20+/8+/6处"高估: 多为 eval/诊断脚本 (mask 大量 `days=6` 硬编码于 scripts/eval_*.py, OOS_DAYS=250 全在 `scripts/_diag_*` 孤儿脚本), 未扫 (属 M6 清理范畴) |
| H5 | `_daily_fetch.py:685-694` | V3 面板单文件就地原子覆盖, 无日期分片, 非严格 WORM |
| H6 | `.gitignore` | — **已修**: 补 `_*.txt`/`predictions*.csv`/`result_*.json`/`filtered_candidates.csv`/`pipeline1_result.txt` 规则; `git rm --cached` 7 个已入库结果文件 (保留磁盘) |

## 三、MEDIUM (选列)

- M1 `app/backtest/engine.py:452-459` 跌停顺延用当日收盘判当日开盘 (保守方向, 不虚增收益) — **已评估, 暂缓**: 属真实但保守的口径 (低估收益而非虚增); 修复需区分 MOO 开盘卖出 (开盘已知信息, 应改 T-1 跌停) 与收盘/触发卖出 (当日跌停不可卖, 当日收盘判合理), 改动面跨多调用方, 暂缓处理
- M2 `app/pipeline_parallel/backtest.py:414-447` 慢牛回测 k=1 同日判退出违背 T+1; 止损 low 触发按 close 成交 (盘内乐观) [部分 GUESS]
- M3 `paper_trading.py:157` 资金用 float (Decimal 铁律仅覆盖实盘 `order_manager`/`executor_base`); `paper_trading.py:157` 价格=0 时除零→inf — **已修**: 建仓零/负价剔除 + 估值除零防护 (`cost>0` 才除, 否则按 0), 新增回归测试 `test_zero_cost_price_guard`. **已定案**: 回测引擎 (`backtest_v35`/`intraday/v51/backtest_engine`) 保持 float — 回测是模拟非真钱, `engine.py` 用整数分记账已规避累积误差, 转 Decimal 是纯性能回归; Decimal 铁律覆盖的实盘路径 (`order_manager`/`executor_base`/`core/backtest_engine`) 已核实正确
- M4 `feature_engine_v35.py:1056-1058` + `cleaning_pipeline.py:178` limit_pct 逐行推导, 全面板数百万行非向量化 — **已修**: 新增 `limit_pct_series` (pandas map + numpy where), 接入两处生产路径, 与逐行语义全等 (回归测试 `test_limit_pct_series_matches_scalar`); eval 脚本仍逐行 (非生产链)
- M5 `app/utils/safe_load.py:63` 内层仍裸 `pickle.load` (可审计非防 RCE); scripts/ 7+ 脚本仍裸用 (不在生产链) — **已修**: `_SafeUnpickler` find_class 拦截已知 RCE gadget 模块 (os/subprocess/builtins/codecs/ctypes 等), 45 个生产 bundle 加载正常, 恶意 payload 全拒 (`tests/test_safe_load.py`); scripts/ 裸用不在生产链, 未逐一迁移 (避免诊断脚本 churn)
- M6 死代码: `risk_overlays.py` 零引用函数; 6 个根目录孤儿脚本 — **已修**: 删除 risk_overlays 7 个零引用函数 + 3 个连带孤儿 helper + 仅它们用的 P23.4 常量 (审计"4 个"低估, 实为 10 个; 生产路径/测试引用的保留); 删除 6 个孤儿脚本 (`_gate_d.py`/`_build_features.py`/`_ic_eval_fast.py`/`_predict_today.py`/`_select_features_main.py`/`_main_list_gen.py`), 全代码库零引用已核实
- M7 潜在逻辑缺陷: `backtest_v35.py` `pos["high_hfq"]` 买入后从不更新 → "移动止盈"实际为"相对成本的回撤止盈", 未真正随高点移动 — **已修**: 决策日棘轮推进 `pos["high_hfq"] = max(high_hfq, T-1 bar hfq 高点)` (无未来函数: 决策只用 T-1 数据), 回撤改从持仓最高点测量; 新增回归测试 `test_trailing_stop_uses_running_high_m7` (冲高+10%→回撤4.5%仍盈利, 旧代码不触发/新代码触发)

## 四、亮点 (未坏)

- 特征/标签/信号构造链路全干净: `feature_engine_v35`/`panel_builder`(merge_asof backward)/`label_engine`/`indicators`/`signals`/`scoring`/`data_5min` 均无未来函数
- 三套回测成本 (佣金+滑点+印花税) 全计入; `engine.py` 用整数分记账规避 float 累积
- 实盘链路 Decimal 正确 (`order_manager`/`executor_base`/`xt_executor`)
- 1261 个测试 / 0 收集错误 / 8 个铁律模块全有测试 / 无空壳断言
- 裸 `except:` 清零 (app/scripts 0 处); M1 lifespan / M2 CORS / M10 subprocess 白名单均已修
- ruff 12 错 → 剩 1 错 (`config/__init__.py` UP009); 主数据链路 Parquet 全覆盖
- 凭据管理: `.env` 未被 git 跟踪; web/ 前端无硬编码密钥; eval/exec 不在生产路径

## 五、修复优先级

| 优先级 | 事项 | 状态 |
|---|---|---|
| P0 | 轮换 Tushare token + 清 `AUDIT_REPORT.md:34` | 文件已打码, token 轮换待用户 |
| P0 | 修 `backtest_v35` AM 卖出口径 (C2) | **已修 + 回归测试** |
| P1 | 设 `AMINQT_API_KEY` 或锁 localhost | **已锁 localhost** (`monitor_catpaw` + json) |
| P1 | OHLCV 校验下沉主链路 (H2) | **已修**: `data_supply.fetch_daily/fetch_history` + `data_5min.load` 接入 `validate_ohlcv` (backtest 路径已有 DataValidator 覆盖, 未重复加) |
| P1 | 训练脚本补 seed (H3) | **已修** |
| P2 | 硬编码阈值进 config (H4) | **已修** (生产路径; eval/诊断脚本未扫) |
| P2 | V3 面板分区 (H5) + gitignore 补 rules/移除已入库结果 (H6) | H6 **已修**; H5 待办 |
| P3 | 回测引擎 float→Decimal (定案保持 float); limit_pct 向量化; 删死代码/孤儿脚本 | **paper_trading 除零 / M4 向量化 / M6 死代码清理已修** |
| P3 | 评估 `engine.py` 替换路径与跌停顺延口径 (H1/M1) | H1 **已修**; M1 已评估暂缓 (保守口径, 需跨调用方改造) |
