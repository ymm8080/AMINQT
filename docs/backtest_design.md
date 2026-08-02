# A股日线回测系统设计文档 (基于实际代码)

> **版本**: V5.0 + V5.2 | **最后更新**: 2026-08-02 | **基于代码**: `app/backtest/`

---

## 1. 系统概览

### 1.1 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    BacktestPipeline (模块8)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Config   │→│ Data     │→│ Data     │→│ Signal   │       │
│  │ Manager  │  │ Loader   │  │ Validator│  │ Evaluator│       │
│  │ (模块1)  │  │ (模块3)  │  │ (模块2)  │  │ (模块4)  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       │                                           │             │
│       ▼                                           ▼             │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              BacktestEngine (模块5)                    │      │
│  │  Squad(5只/30%) │ Sniper(2只/60%) │ SniperMax(1只/100%)│      │
│  └──────────────────────────────────────────────────────┘      │
│       │                                           │             │
│       ▼                                           ▼             │
│  ┌──────────────────┐  ┌──────────────────────────────┐       │
│  │ Comparative      │→│ ReportGenerator (模块7)        │       │
│  │ Analyzer (模块6) │  │ JSON + TXT + HTML            │       │
│  └──────────────────┘  └──────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 文件清单

| 模块 | 文件 | 职责 |
|------|------|------|
| 模块1 | `app/backtest/config_manager.py` | 配置管理: YAML→BacktestConfig dataclass, SHA256哈希 |
| 模块2 | `app/backtest/data_validator.py` | 数据校验: E001-E010 错误码, 未复权检测, PIT警告 |
| 模块3 | `app/backtest/data_loader.py` | 数据加载: CSV/Parquet, 列校验, 对齐, 20日均额, 交易日历 |
| 模块4 | `app/backtest/signal_evaluator.py` | 信号评估: Rank IC, 命中率, 触发率, 校准, 跳空风险 |
| 模块5 | `app/backtest/engine.py` | 核心引擎: Fen精度, T+1交收, V5.2 ATR风控, V5.0过滤 |
| 模块6 | `app/backtest/comparative_analyzer.py` | 对比分析: Squad vs Sniper, 集中度风险, Jensen Alpha |
| 模块7 | `app/backtest/report_generator.py` | 报告生成: JSON/TXT/HTML三格式, 审计信息, 免责声明 |
| 模块8 | `app/backtest/pipeline.py` | 管线编排: 串联1-7, try-except包裹, PipelineResult |
| CLI | `scripts/run_backtest.py` | 命令行入口: argparse, YAML路径读取, 日志配置 |
| 配置 | `config/backtest_config.yaml` | YAML配置模板: 所有参数集中管理 |
| 测试 | `tests/test_backtest_v5.py` | 18个单元测试: Config/Validator/Engine/Report/Pipeline |

---

## 2. 模块1: ConfigManager

### 2.1 BacktestConfig 参数全表

```python
@dataclass
class BacktestConfig:
    # ── 资金 ──
    initial_capital: float = 100000.0
    # ── 触发 ──
    trigger_pct: float = 0.03              # 3% 突破触发
    # ── 滑点 (basis points) ──
    slippage_buy_bp: int = 10              # 买入 0.1%
    slippage_sell_moo_bp: int = 10         # 正常卖出 0.1%
    slippage_sell_stop_bp: int = 30        # 止损卖出 0.3%
    # ── 佣金 ──
    commission_rate: float = 0.00025       # 万2.5
    min_commission: float = 5.0            # 最低5元
    stamp_tax_rate: float = 0.001          # 千1, 仅卖出
    # ── V5.2 风控 (ATR自适应) ──
    stop_loss_main: float = -0.04          # 主板 -4%
    stop_loss_dual: float = -0.06          # 双创 -6%
    stop_loss_atr_mult: float = 1.5
    stop_loss_atr_floor: float = 1.2
    trailing_stop: bool = True
    trailing_stop_activate: float = 0.03
    trailing_stop_min_pct: float = 0.03
    trailing_stop_atr_mult: float = 1.0
    time_stop_days: int = 2
    time_stop_use_median: bool = True
    time_stop_fixed_threshold: float = 0.01
    holding_period: int = 2
    daily_fuse_fixed: float = -0.04
    daily_fuse_use_sigma: bool = True
    daily_fuse_sigma: float = 2.0
    daily_fuse_window: int = 20
    system_halt_drawdown: float = -0.15
    consecutive_loss_limit: int = 3
    consecutive_loss_cooldown: int = 5
    atr_period: int = 14
    # ── 买入过滤 ──
    max_gain_pct: float = 0.07
    min_net_edge: float = 0.005
    stop_distance_atr_mult: float = 1.2
    # ── V5.0 新增 ──
    volume_confirm_ratio: float = 1.5
    market_drop_limit: float = -0.02
    down_limit_max_days: int = 3
    max_swap_per_day: int = 2
    swap_threshold: float = 0.01
    # ── 选股 ──
    prob_threshold: float = 0.55
    position_mode: str = "squad"
    filter_st: bool = True
    filter_trend: bool = False
    # ── 资金管理 ──
    cash_interest_rate: float = 0.003
    min_tradeable: int = 2
    volume_limit_pct: float = 0.10
    min_position_value: float = 20000
    # ── 信号评估 ──
    signal_horizons: list[int] = [1, 2, 4]
    signal_simulate_trigger: bool = True
    signal_k: int = 5
```

### 2.2 ConfigManager 方法

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `load(path)` | YAML路径 | `BacktestConfig` | 读取YAML, 映射backtest/signal_eval段 |
| `hash(config)` | `BacktestConfig` | `"sha256:xxxx"` | SHA256前16位, 用于审计复现 |

---

## 3. 模块2: DataValidator

### 3.1 错误码表

| 码 | 含义 | 级别 | 处理 |
|----|------|------|------|
| E001 | 未复权价格检测 | 警告 | 日志记录 |
| E002 | 缺失必要列 | 致命 | 终止回测 |
| E003 | close ≤ 0 | 致命 | 删除行 |
| E004 | high < low / open越界 | 致命 | 删除行 |
| E005 | 停牌推断异常 | 警告 | 标记is_halt |
| E006 | 次新股未过滤 | 警告 | 过滤<60日 |
| E007 | PIT数据警告 | 警告 | 日志记录 |
| E008 | 基准数据缺失 | 警告 | 降级运行 |
| E009 | 成交量数据缺失 (V5.0) | 警告 | 跳过成交量确认 |
| E010 | 大盘指数缺失 (V5.0) | 警告 | 跳过大盘过滤 |

### 3.2 校验方法

- `validate_prices()`: 检查close≤0, high<low, open越界, 停牌推断
- `validate_predictions()`: 检查prob_up_h{1,2,4}范围[0,1]
- `check_adjusted_prices()`: 检测单日跌幅>18%且次日恢复 (未复权特征)
- `check_pit_data()`: 检查board/is_st/is_halt是否在T日快照
- `infer_halt_status()`: volume≤0或<1000 → is_halt=1
- `filter_ipo_stocks(min_listing_days=60)`: 过滤上市<60交易日

---

## 4. 模块3: DataLoader

### 4.1 必要列

**预测表** (`pred_df`): `date, stock, score_h1, prob_up_h1, pred_ret_h1, score_h2, prob_up_h2, pred_ret_h2, score_h4, prob_up_h4, pred_ret_h4, board`

**行情表** (`price_df`): `date, stock, open, high, low, close, volume, amount, up_limit, down_limit, is_st, is_halt, pre_close, circ_mv`

**大盘指数表** (`market_df`, V5.0): `date, index_close`

### 4.2 加载流程

1. 读取CSV/Parquet (按扩展名自动判断)
2. 校验必要列
3. 统一日期为 `pd.Timestamp`, 股票代码为 `str`
4. 删除预测表中行情缺失的行
5. 按 `(date, stock)` 强制排序 (可重复性)
6. 预计算 `avg_amount_20d` (20日日均成交额)
7. 构建交易日历 (sorted unique dates, 非Timedelta)
8. 生成数据版本哈希 `sha256(rows+dates+cols)[:16]`

---

## 5. 模块4: SignalEvaluator

### 5.1 评估指标

| 指标 | 方法 | 说明 |
|------|------|------|
| Rank IC | `calc_rank_ic()` | 每日Top5内部Spearman相关, 分母=触发价×滑点 |
| Rank IR | IC均值/IC标准差 | 信号稳定性 |
| 命中率 | `calc_topk_hit_rate()` | Top1/2/3/5的actual_ret>0比例 |
| 触发率 | `calc_trigger_stats()` | 触发率/开盘触发率/盘中触发率 |
| 预测偏差 | `calc_prediction_bias()` | MAE/RMSE/WMAPE/Bias/Correlation |
| 概率校准 | `calc_prob_calibration()` | 分桶: <0.50, 0.50-0.55, 0.55-0.60, 0.60-0.70, 0.70+ |
| 跳空风险 | `calc_gap_risk()` | TopK平均隔夜跳空收益率 |
| 成交量确认率 | `calc_volume_confirmation_rate()` | V5.0: T+1量≥T量×1.5的比例 |

### 5.2 触发价定义

```python
trigger_price = open_price * (1 + trigger_pct)  # = open * 1.03
triggered = (high_price >= trigger_price)
entry_price = trigger_price * (1 + slippage_buy)
actual_ret = close_price / entry_price - 1.0
```

---

## 6. 模块5: BacktestEngine (核心)

### 6.1 价格精度: Fen (整数分)

所有金额在引擎内部用整数分(fen), 禁止float累积误差:

```python
_price_to_fen(10.05) → 1005
_fen_to_yuan(1005) → 10.05
_calc_trigger_price(open_fen) → int(open_fen * 103 / 100 + 0.5)
_calc_entry_price(trigger_fen, bp) → int(trigger_fen * (1000+bp) / 1000 + 0.5)
```

### 6.2 数据结构

**Trade**: entry_date, exit_date, stock, entry_price_fen, exit_price_fen, quantity, pnl_fen, pnl_pct, exit_reason, horizon, mode, prob_up, pred_ret, score, max_profit_pct, commission_entry_fen, commission_exit_fen, stamp_tax_fen, **is_swap** (V5.0)

**Holding**: stock, entry_date, entry_price_fen, quantity, horizon, mode, prob_up, pred_ret, score, days_held, max_close_fen, stop_loss_triggered, stop_profit_triggered, **stop_price_fen** (V5.2), **atr_pct** (V5.2), **board** (V5.2), **median_2d_return** (V5.2), **down_limit_days** (V5.0)

### 6.3 仓位模式

| 模式 | 选股数 | 单只上限 | 说明 |
|------|--------|---------|------|
| Squad | 5 | 30% | 分散持仓, Softmax加权 |
| Sniper | 2 | 60% | 集中持仓 |
| SniperMax | 1 | 100% | 满仓单只 |

资金分配: Softmax(prob×2.0) → 截断到上限 → 一次重分配

### 6.4 回测主循环 (13步)

```
For each trade_date (i=1..N):
  1. 开盘前风控 (基于昨日收盘): S1止损/S2移动止盈/S5a时间止损
  2. 开盘执行风控卖出 (MOO, 检查跌停/停牌)
  3. [占位] 到期卖出在步骤5处理
  4. 买入:
     a. 读取T日信号, 按score排序, 过滤prob<阈值
     b. 买入资格检查: 停牌/ST/涨停/流动性/流通市值/追高
     c. 触发价计算: trigger = open × 1.03
     d. 检查: high ≥ trigger (Approximate Mode)
     e. V5.2: 预计算ATR/板块/中位数, 计算止损价
     f. B7止损距离否决: 距stop < 1.2×ATR → 放弃
     g. 持仓冲突处理 (Minmin修正)
     h. 资金分配 + 执行买入 (100股取整, 成交量≤10%)
  4a. 系统停机: 清仓所有可卖持仓
  5. 收盘到期卖出 (S5b, 检查跌停→顺延)
  6. 更新持仓: days_held++, max_close_fen更新 (用收盘价)
  7. 计算市值与NAV
  8. 当日盈亏检查:
     - 连续亏损按交易日计 (非笔数)
     - V5.2日保险丝: 2σ自适应 + 4%固定兜底 (双轨)
     - V5.2系统停机线: 总回撤≥15%
  9. T+1交收: frozen_cash → available_cash
  10. 现金计息: 年化0.3%
  11. 冷却倒计时
  12. 对账: cash≥0, frozen≥0
  13. 记录: daily_records + holdings_history
```

### 6.5 V5.2 ATR 自适应风控

**S1 动态止损**:
```
stop_pct = max(固定值, -1.5×ATR_pct)
  主板: max(-0.04, -1.5×ATR_pct)
  双创: max(-0.06, -1.5×ATR_pct)
噪音带断言: abs(stop_pct) ≥ 1.2×ATR_pct (否则调宽)
stop_price_fen = int(entry_fen × (1 + stop_pct))
```

**S2 移动止盈**:
```
激活: max_close_fen > entry 且 profit_pct ≥ 3%
回撤: retrace = (max_close_fen - close) / max_close_fen
阈值: max(3%, 1.0×ATR_pct)
触发: retrace ≥ 阈值
```

**S5a 时间止损**:
```
条件: days_held ≥ 2 且 2日收益 < 基准
基准: 20日滚动2日收益中位数 (个股自比) 或固定1%
```

### 6.6 V5.0 过滤器 (日频代理)

| 过滤器 | 规则 | 日频代理 | 偏差 |
|--------|------|---------|------|
| F2.17 成交量确认 | T+1量 ≥ T量×1.5 | 全日量对比 | 乐观 (尾盘放量) |
| F2.18 大盘暴跌 | T+1跌≥2%停买 | **T收盘 vs T-1收盘** | 保守 (无前视) |
| F2.19 跌停强平 | 连续3日跌停 | 全日close≤down_limit | 无偏差 |

**大盘过滤日频代理**: 原始规则用T+1开盘 vs T收盘, 但日频数据无法在买入时获取T+1开盘价. 改为保守方案: T日收盘 vs T-1日收盘, 跌>2%→T+1不买. 无前视偏差 (铁律1).

### 6.7 持仓冲突处理 (Minmin 修正)

```
若候选股已在持仓中:
  new_expected = candidate.pred_ret
  holding_expected = holding.pred_ret + current_pnl_pct
  if new_expected > holding_expected + 1%: → 卖旧买新 (replace_signal)
  elif holding亏损 and new_expected > 0: → 替换亏损 (replace_loss)
  else: → 保留旧持仓, 跳过新信号
```
原则: 可控风险下总收益最大化.

### 6.8 T+1 交收机制

```
卖出所得 → frozen_cash (当日不可用)
次日开盘: frozen_cash → available_cash (解冻)
```
保证: 当日买入的股票不能在当日卖出 (铁律4).

### 6.9 绩效指标

| 指标 | 计算 |
|------|------|
| total_return | (final_nav - initial) / initial |
| annual_return | (1+total)^(252/days) - 1 |
| sharpe_ratio | mean(daily_ret)/std(daily_ret) × √252 |
| max_drawdown | min(equity/cummax - 1) |
| calmar_ratio | annual_return / abs(max_dd) |
| win_rate | wins / total_trades |
| profit_loss_ratio | avg_win / avg_loss |
| daily_tradeable_rate | tradeable_days / total_days |

---

## 7. 模块6: ComparativeAnalyzer

| 方法 | 说明 |
|------|------|
| `calc_concentration_risk_ratio()` | Sniper最大回撤 / Squad最大回撤 |
| `calc_jensen_alpha(result, benchmark)` | Alpha = 策略年化 - (rf + β×(基准年化-rf)) |
| `compare_nav_curves()` | 最终NAV对比 |
| `generate_comparison_report()` | 汇总报告 |

建议规则: 集中度>2且Sniper夏普<Squad → 分散; Sniper夏普>Squad×1.2且集中度<1.5 → 集中; 否则收紧止损.

---

## 8. 模块7: ReportGenerator

### 8.1 输出格式

| 格式 | 扩展名 | 用途 |
|------|--------|------|
| JSON | .json | 程序读取, 结构化数据 |
| TXT | .txt | 人类可读, 终端查看 |
| HTML | .html | 可视化, 浏览器打开 |

### 8.2 报告内容

1. 绩效概览 (8个指标卡片)
2. 交易明细 (最近20笔)
3. 资金曲线 (每日NAV)
4. 信号评估 (H1/H2/H4)
5. 对比分析 (Squad vs Sniper)
6. 审计信息 (config_hash + data_version_hash)
7. 免责声明 (Approximate Mode日频代理说明)

文件命名: `reports/backtest_{mode}_{timestamp}.{json,txt,html}`

---

## 9. 模块8: BacktestPipeline

### 9.1 编排流程

```
Step 1: ConfigManager.load(config_path) → BacktestConfig
Step 2: DataLoader.load() → (pred_df, price_df, benchmark_df, market_df)
Step 3: DataValidator.validate_all() → errors[] (E级致命→终止)
Step 4: SignalEvaluator.run_full_report() → signal_report
Step 5: for mode in modes: BacktestEngine.run() → result_df, trades_df, metrics
Step 6: ComparativeAnalyzer.generate_comparison_report() → comparison
Step 7: ReportGenerator.generate() → report_paths[]
```

### 9.2 PipelineResult 结构

```python
@dataclass
class PipelineResult:
    success: bool
    config: BacktestConfig
    squad_metrics: Dict
    sniper_metrics: Dict
    squad_result_df: pd.DataFrame
    sniper_result_df: pd.DataFrame
    squad_trades_df: pd.DataFrame
    sniper_trades_df: pd.DataFrame
    signal_report: Dict
    comparison: Dict
    report_paths: List[str]
    errors: List[str]
```

---

## 10. CLI 入口

```bash
# 基本用法
python scripts/run_backtest.py --config config/backtest_config.yaml

# 指定数据和模式
python scripts/run_backtest.py \
  --pred data/processed/predictions.csv \
  --price data/processed/prices.csv \
  --market data/processed/market_index.csv \
  --mode squad sniper --horizon 2

# 详细日志
python scripts/run_backtest.py -v
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/backtest_config.yaml` | 配置文件 |
| `--pred` | (YAML) | 预测表路径 |
| `--price` | (YAML) | 行情表路径 |
| `--benchmark` | None | 基准表 |
| `--market` | None | 大盘指数表 |
| `--output` | `reports` | 输出目录 |
| `--mode` | `[squad, sniper]` | 回测模式 |
| `--horizon` | `2` | 持有期 1/2/4 |
| `--verbose` | False | DEBUG日志 |

---

## 11. 测试

`tests/test_backtest_v5.py` — 18个单元测试:

| 测试类 | 数量 | 覆盖 |
|--------|------|------|
| TestConfigManager | 7 | 默认配置/哈希/YAML加载/V5.0参数/V5.2参数 |
| TestReportGenerator | 3 | TXT报告/JSON序列化/空交易 |
| TestBacktestEngine | 6 | 初始化/run/metrics/market_drop/Holding字段/Trade字段 |
| TestPipeline | 1 | 初始化 |

当前状态: **18/18 passed, ruff All checks passed**

---

## 12. 关键设计决策

### 12.1 Approximate Mode (废除 Strict Mode)

日线数据无法判断最高价出现在10:30前还是后. 仅保留 Approximate Mode: 买入条件 High ≥ Trigger Price, 报告标注"日线数据无法精确模拟10:30时间窗口".

### 12.2 Fen 精度 (整数分)

所有金额在引擎内部用整数分, 避免 float 累积误差.

### 12.3 无前视偏差

- 信号T日生成, T+1日执行
- 风控基于T日收盘 (非T+1)
- 大盘过滤: T vs T-1收盘 (非T+1开盘 vs T收盘)
- 交易日历: sorted unique dates (非Timedelta)

### 12.4 移动止盈用收盘价

`max_close_fen` 使用最高收盘价, 非日内最高价. 避免涨停日"一tick"极限价格.

---

## 13. 双向同步机制

本设计文档有两份完全相同的副本, 修改任一文件后运行同步脚本即可:

```bash
python scripts/sync_backtest_design.py
```

- **副本1**: `docs/backtest_design.md` (项目代码库)
- **副本2**: `d:/AMINQT/REFERENCE/Design All/Function Spec/BACKTESTING/backtest_design.md`

同步逻辑: 比较两文件修改时间, 较新的覆盖较旧的.

---
**测试同步**: 此行由副本A添加, 同步后应出现在副本B.

---
**反向测试**: 此行由副本B添加, 同步后应出现在副本A.
