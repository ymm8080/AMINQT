# 融合系统 PRD / Function Spec

> **系统**: Fusion（融合）| **模块**: `app/pipeline_parallel/fusion.py`（新建，从 `backtest.py` 拆出）
> **版本**: V1.0 | **最后更新**: 2026-08-05 | **依据代码**: `app/pipeline_parallel/{backtest,scoring,config}.py`
> **目标口径**: MFE（持有期内最大涨幅，2026-08-04 用户需求）| **验收**: 只看 OOS

---

## 1. 定位

融合系统是并行 PIPELINE（`app/pipeline_parallel/`）三大系统之一：**大仓位、持有 3-5 天、TOP-10**。

- **选股数**: 主档 TOP-10，附档 TOP-10（同档位，无拆档）
- **持有**: T+1 买，持有 3-5 天，目标 = 持有期内最高价可兑现的最大收益（MFE）
- **验收视界**: T+2 / T+3 / T+5 / T+10 四视界（2026-08-04 用户加 T+2/T+3）
- **定位**（与狙击对比）: 融合=大仓、容忍较长持有（`limit_dist_pct` 长视界出边）；狙击=小仓机动快出。两系统共现股 = 最高确定性，大仓。

## 2. 模块边界（拆自 backtest.py）

| 现有代码 | 归属 | 说明 |
|----------|------|------|
| `config.FUSION` | fusion 模块 | 特征池 / TOP-N / 视界定义，迁入本模块 |
| `scoring.pool_score / select_topn` | 共享 `scoring.py` | 截面合成打分 + 每日 TOP-N，两系统共用，不改 |
| `backtest.run_system` | 共享引擎 | 全窗/OOS 双头回测骨架，两系统共用 |
| `backtest.build_merged_shortlist` | 融合模块（合并入口） | 狙击TOP-5 ∪ 融合TOP-10 去重，共现优先，分数降序，截 top_n |
| `backtest.run_all` 中 `systems["fusion"]` 段 | fusion 模块 | 编排 + 落盘，迁入本模块 |

本模块只读数据/配置，不 import `app/pipeline1/` 训练/选股逻辑。

## 3. 输入

| 输入 | 来源 | 说明 |
|------|------|------|
| 行集 `work` | `load_panel()` | main/dual 3y 检查点 + `label_mfe_{2,3,5,10}d_net` + 可交易性门 + `board` 列 |
| `SystemSpec` | 本模块 | 特征池 / TOP-N / 视界（见 §4） |
| 板块阈值 | `config.BOARD_THRESHOLDS` | main 0.55/0.03，dual 0.55/0.04 |
| OOS 窗 | `config.OOS_WINDOWS` | 6m=126d（主验收）/ 3m=63d / 10d=15d |

## 4. 系统定义（SystemSpec）

```python
FUSION = SystemSpec(
    name="fusion",
    desc="融合: 大仓位, 持有 3-5 天, 目标 MFE, TOP-10 四视界 T+2/3/5/10 双头",
    pool=("amihud_illiquidity", "VAR51", "down_gap_pct", "limit_dist_pct",
          "ret_reversal_5d", "small_mv_premium"),
    top_n=10,
    top_n_alt=10,
    horizons=("3d", "2d", "5d", "10d"),
    labels=("label_mfe_3d_net", "label_mfe_2d_net",
            "label_mfe_5d_net", "label_mfe_10d_net"),
    notes=(...),   # 见 §4.2
)
```

### 4.1 特征池（6 个独立特征，去重后）

| 特征 | 语义 | 已知行为（config notes） |
|------|------|------|
| `amihud_illiquidity` | 非流动性（长窗） | 池内 |
| `VAR51` | 风险价值 | 长视界出边 |
| `down_gap_pct` | 低开缺口 | 池内 |
| `limit_dist_pct` | 距涨跌停距离 | **长视界出边 → 融合方案须容忍较长持有** |
| `ret_reversal_5d` | 5日反转 | 长视界出边 |
| `small_mv_premium` | 小市值溢价 | **高风险档 → 仓位纪律必需** |

### 4.2 已知边界（notes）

- `limit_dist_pct` 在狙击池 TOP-5 全视界不过，但入融合池 → 融合**必须容忍较长持有**
- `small_mv_premium` 高风险档 → 大仓有波动放大风险，**仓位纪律（共现大仓/单系统小仓）必需**

## 5. 选股流程（纯特征，无前瞻）

每 (symbol, date) 行、每板块独立执行：

```
1. cross_rank: 每日期截面分位数排名 (升序 0~1)         # scoring.cross_rank
2. pool_score: 池特征等权平均 → 合成池分 score          # 缺列自动跳过; 全缺列 → 抛错
3. select_topn: 每日期按 score 降序取 TOP-10 (主=附)
4. 双头验收: 对选中切片逐视界量 MFE 幅度+胜率, 与窗口无条件基准对比
```

- 板块必须**分开**：截面排名不能混板（2026-08-05 bug 记录：未过滤 → main 名单混入双创股）
- 纯向量化（`groupby("date").rank` / `head`），禁 for 循环遍历股票（铁律）
- `score` 只由当日截面分位合成，不使用任何未来信息 → 无 look-ahead bias

## 6. 目标口径（MFE）

`label_mfe_{k}d_net = max(high_hfq[T+2 .. T+1+k]) / close_hfq[T+1] - 1 - COST - 2*滑点`

- 买价 = `close_hfq[T+1]`（T+1 收盘已持有，无法兑现 T+1 盘中最高）
- 窗口 = `max(high[T+2 .. T+1+k])`，含目标日
- `skipna=False`：窗口内缺未来价（尾段/停牌）→ NaN（保守，同生产口径）
- 成本口径 = 生产（`COST + 2×分层滑点`，adv20 分层）

## 7. 验收标准

**双头通过**（`dual_head_ok`）: `n ≥ 5` 且 `winrate ≥ min_winrate` 且 `mag > min_mag`。

| 板块 | min_winrate | min_mag | 锚定基准 |
|------|------------|---------|---------|
| main（60/00） | 0.55 | 0.03 | main 无条件 T+2 幅度 +2.96% |
| dual（30/68） | 0.55 | 0.04 | dual 无条件 T+2 幅度 +4.25% |

**保留判定（只看 OOS）**:
- OOS 主窗 = 6m（126 交易日）；另报 3m / 10d
- 任一视界双头通过（主档 TOP-10，附档=同档）→ 系统保留
- **FULL 全窗永不用于验收**，仅作参考（`kept=None`，2026-08-04 用户铁律）
- **TOP-10 分档**（`rank_bands`）: 整组 all10 vs 前5 first5 vs 后5 last5 — 回答"TOP-10 是否赢 TOP-5，还是仅前 5 赢"。若 `last5.mag < first5.mag` → 改进点"可考虑仅买 TOP-5"

## 8. 合并短名单（买入名单，本模块核心出口）

用户 2026-08-04："WHAT WE EVALUATING IS ON FINAL SHORT LIST" — 验收/买入都基于**最终合并短名单**，不是两系统分开 15 只。

`build_merged_shortlist(work, top_n, mask)`:
```
狙击TOP-5 ∪ 融合TOP-10  →  按 (date, symbol) 去重
  systems   = "+".join(去重后的系统名)          # "fusion+sniper" / "fusion" / "sniper"
  co_occur  = systems 含 "+"                    # 双系统一致 = 最高确定性
  score     = max(两系统各自合成池分)
排序: date → co_occur(降) → score(降)  →  rk=每日 1..top_n
```

**仓位纪律**（结论 §9 输出）:
- 共现股（`fusion+sniper`）= 双系统一致 → **大仓**
- 单系统股 → **小仓**

## 9. 输出（WORM）

| 文件 | 内容 |
|------|------|
| `<BACKTEST_RESULT_DIR>/<ts>/backtest.json` | 完整报告 + conclusion |
| `backtest.log` / `conclusion.txt` | 人类可读结论 |
| `stocks_{board}_fusion_{full,oos}.csv` | 每 OOS 日选股清单（date/symbol/score/rk + `label_mfe_*_net`） |
| `stocks_merged_oos_{board}.csv` | 狙击TOP-5 ∪ 融合TOP-10 去重合并 OOS 买入名单 |
| `last_{15}_days_picks_{board}.csv` | 末 15 交易日逐日融合 TOP-10 实际名单 + 各视界 MFE |
| `shortlist_{board}.csv` | 今日短名单（T-10 档，含 `cut` 列区分；T-5 与 T-10 同文件） |

- 全部带日期后缀目录/文件名，旧文件不覆盖（WORM 铁律）
- 预测/短名单另写 `config.settings.STOCK_LIST_DIR`（`D:\AMINQT\DAILY OPERATION\STOCK LIST\`）

## 10. 函数规格（独立模块 `fusion.py`）

```python
FUSION: SystemSpec                       # §4 系统定义（迁入）

def select_daily(day: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame
    """当日截面 TOP-10 选股. 输入单日/板块切片; 返回 date/symbol/score/rk (rk=降序1..N)."""

def run(work: pd.DataFrame, oos_windows: dict[str, int],
        bcrit: tuple[float, float]) -> dict
    """全窗参考 + 各 OOS 窗: 每窗独立选股+逐视界双头, 返回 {full, oos, kept}.
       full['kept']=None; oos[lab]['kept'] = 任一视界通过."""

def rank_bands(work: pd.DataFrame, top_n: int,
               bands: tuple[tuple[str, int, int], ...],
               mask, crit) -> dict
    """TOP-10 拆 all10/first5/last5 逐档双头, 供 §7 分档对比."""

def build_merged_shortlist(work: pd.DataFrame, top_n: int,
                           mask=None) -> pd.DataFrame
    """§8 合并短名单: 狙击TOP-5 ∪ 融合TOP-10 去重, 共现优先, 分数降序, 截 top_n.
       返回 date/symbol/systems/co_occur/score/rk."""

def evaluate_merged(work: pd.DataFrame, top_n: int,
                    mask=None, crit=None) -> dict
    """对合并最终短名单 (TOP-5/TOP-10) 逐视界量双头, 基准取窗口无条件 (验收用)."""

def export(work: pd.DataFrame, run_dir: Path, board: str) -> list[str]
    """stocks_<board>_fusion_<full|oos>.csv WORM 落盘."""

def shortlist(day: pd.DataFrame, oos_ph: dict) -> pd.DataFrame
    """今日融合 TOP-10 短名单 + est_wr / prob_{h} / exp_{h} 校准列.
       共现股 (fusion+sniper) 逐视界取两系统胜率较高者 (双系统一致=更高确定性)."""
```

依赖：`scoring.{cross_rank, pool_score, select_topn, measure_dual_head, dual_head_ok}`、`backtest.{add_mfe_labels, _baseline, run_system, _dual_per_horizon}`、`config.{BOARD_THRESHOLDS, OOS_WINDOWS}`、`sniper.SNIPER`（合并时取狙击 TOP-5）。

## 11. 铁律合规

- **无前瞻**：选股只用当日截面特征；MFE 标签仅供训练/验收，不用于特征
- **成本**：MFE 净收益已含 COST + 2×滑点
- **可交易性门**：剔除近 20 交易日有行比例 < 80% 的慢性停牌股（PIT，无前瞻）
- **板块隔离**：main/dual 分开回测与排名，阈值不同
- **验收只看 OOS**：FULL 永不用于保留判定

## 12. 测试计划（TDD，与交易规则同步）

| 用例 | 断言 |
|------|------|
| `select_daily` 每日期恰好 TOP-10，按 score 降序 | rk==1..10，无重复 |
| 板块隔离：main 名单不含 30/68 前缀 | 断言 |
| `build_merged_shortlist` 共现优先、组内 score 降序、截 top_n | co_occur 全在单系统前 |
| 共现股 `systems=="fusion+sniper"`，score=两系统 max | 断言 |
| 双头通过条件 | n<5 → False；winrate/阈值边界 |
| OOS 保留判定只用 OOS | full['kept'] is None |
| 分档对比：last5.mag < first5.mag → 改进点"仅买 TOP-5" | 断言 |
| 无前瞻：score 与未来标签无引用 | 构造未来 NaN 用例 |

---

## 13. 关联

- [[sniper_system_prd]]（狙击系统 PRD）— 共现大仓
- `docs/backtest_design.md` — 旧版 Squad/Sniper/SniperMax 仓位模式（本系统为并行版融合大仓）
