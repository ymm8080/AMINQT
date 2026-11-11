# 定时任务 (Scheduled Tasks / Cron Jobs)

> **位置**: `03_operations/runbooks/scheduled-tasks.md`
> **最后更新**: 2026-11-11

## Windows 任务计划程序

| 任务名 | 计划 | 命令 | 说明 |
|--------|------|------|------|
| AMINQT-Data-Download | 每日 17:00 | `python scripts/download_data.py` | 收盘后下载最新日K数据 |
| AMINQT-Backup | 每日 02:00 | `python scripts/backup.py` | 模型权重+数据备份 |
| AMINQT-Backup-Cleanup | 每日 02:45 | `powershell Get-ChildItem logs/backup -Filter *.tar.gz \| Where-Object {$_.CreationTime -lt (Get-Date).AddDays(-30)} \| Remove-Item` | 清理30天以上备份 |
| AMINQT-Genious-2030 | 周一~五 20:30 (+08:00) | `python scripts/_genious_excel.py` | GENIOUS Excel 交付 (冠军四段+观察池); 注册: `scripts/setup_genious_task.ps1`; 状态: `logs/genious_{date}.state.json` |
| AMINQT-Genious-Watchdog-2110 | 周一~五 21:10 (+08:00) | `python scripts/_genious_backfill_missing.py` | 缺文件/缺页看门狗: 扫近 5 交易日, 缺 `GENIOUS_{date}.xlsx` 或三页不全 (冠军四段/首板点名/板前哨) 则补跑 (--no-push); 面板落后交付日先无参自愈 `_daily_fetch.py` 再补 (10-26 抓取 OOM 半途死 → 面板缺当日 → 只出冠军页的场景); 让路存活 GENIOUS/夜链; 心跳 `logs/genious_watchdog.log`; 注册: `scripts/setup_genious_watchdog_task.ps1`。堵 2026-10-22/10-26 漏交付 |
| AMINQT-LegacyAutomation-2130 | 每日 21:30 (+08:00) | `python scripts/run_legacy_automation.py` | LEGACY 夜链 (~2.5h, 周五 retrain); 注册: `scripts/setup_legacy_automation.ps1` |
| AMINQT-ParallelAutomation-2330 | 每日 23:30 (+08:00) | `python scripts/run_parallel_automation.py` | PARALLEL 夜链 (~2.5h, 含 prob10dens/stocklist_combined/ths_push); 注册: `scripts/setup_parallel_automation.ps1` |
| AMINQT-Combined-Watchdog-0830 | 周一~五 08:30 (+08:00) | `python scripts/_stocklist_combined_backfill.py` | stocklist_combined 缺文件看门狗: 扫近 5 交易日, 有 LEGACY 清单且缺 combined 则补产 (缺源跳页口径); 夜链整链未跑/中途死也兜底; 让路存活夜链/combined; 注册: `scripts/setup_combined_watchdog_task.ps1`; 心跳 `logs/combined_watchdog.log`。堵 2026-10-30 起 refresh OOM 连锁断供 (10-29 最后一次, 11-11 解耦+看门狗双修) |
| AMINQT-Champ4-Push-0745 | 周一~五 07:45 (+08:00) | `python scripts/_ths_champ4_push.py` | GENIOUS「冠军四段」页每日推同花顺自选 (2026-11-11 用户令; 独立旗标 `GENIOUS.push_champ4_to_ths=True` 放行, 不受 11-03 停推总闸影响 — 板前哨等其余源仍停); 幂等: 该日结果单全 landed 跳过, blocked/manual 新槽位重推; 双触发=交付当晚 spawn + 本任务晨间兜底; 注册: `scripts/setup_champ4_push_task.ps1`; 心跳 `logs/champ4_push.log` |

三条夜链守卫互斥 (`scripts/_run_guard.py` 哨兵扫描), **严禁同时手工启动两条链** — 2026-10-23
双链互等死锁事故。GENIOUS 及其看门狗不在哨兵表内, 但看门狗自带让路闸。
**stocklist_combined 每日必产 (2026-11-11 用户令 "fix to have daily creation")**: combined 在 PARALLEL 链内已解耦 deliver_parallel (LEGACY 源独立健康, PARALLEL/密度/SLOW_BULL 缺源本来就跳页) — parallel 侧 refresh OOM 等失败不再拖垮 combined;
夜链整链未跑/中途死由 AMINQT-Combined-Watchdog-0830 次晨扫近 5 交易日补缺。
**冠军四段每日推送 (2026-11-11 用户令 "auto push to ths for genious file 冠军四段 ON DAILY BASIS")**: 新增 `scripts/_ths_champ4_push.py` — 源=交付层 xlsx「冠军四段」页全票 (不做板前哨的次数双非零过滤), 独立旗标 `GENIOUS.push_champ4_to_ths=True` 放行, 不受 11-03 "stop pushing to ths" 总闸影响 (板前哨/影子单等其余源仍全停, 闸见 `_ths_watchlist_push._push_allowed`);
双触发: GENIOUS 交付后当晚 spawn + AMINQT-Champ4-Push-0745 晨间幂等重推 (全 landed 跳过; blocked/manual 按 新小时槽位重推 — 用户在场 fail-closed 次晨自愈)。

### 设置方式
```powershell
# 创建数据下载任务
schtasks /create /tn "AMINQT-Data-Download" /tr "python D:\AMINQT\AMINQT CODES\scripts\download_data.py" /sc daily /st 17:00

# 创建备份任务
schtasks /create /tn "AMINQT-Backup" /tr "python D:\AMINQT\AMINQT CODES\scripts\backup.py" /sc daily /st 02:00

# 查看任务
schtasks /query /tn "AMINQT-*"

# 立即运行
schtasks /run /tn "AMINQT-Data-Download"

# 查看上次运行时间
schtasks /query /tn "AMINQT-Data-Download" /v
```

## APScheduler 内置调度

| 计划 | 触发 | 机制 | 说明 |
|------|------|------|------|
| 每日 14:50 | Asia/Shanghai | APScheduler cron | 收盘前自动选股 |
| 启动时 | 一次性 | FastAPI startup | 加载内存缓存 |

### 配置位置
```python
# app/main.py
sched.add_job(
    select_stocks,           # 选股函数
    "cron",
    hour=14, minute=50,
    timezone="Asia/Shanghai"
)
```

### 验证
```bash
# 检查调度器日志
grep "调度器\|Scheduler" logs/aminqt.log

# 手动触发选股
curl -X POST http://127.0.0.1:8000/api/v1/select
```

## 交易时段参考 (Asia/Shanghai)

| 时段 | 时间 | 操作 |
|------|------|------|
| 盘前 | 09:15-09:25 | 集合竞价 |
| 上午盘 | 09:30-11:30 | 连续竞价 |
| 午休 | 11:30-13:00 | 休市 |
| 下午盘 | 13:00-15:00 | 连续竞价 |
| 收盘 | 15:00 | 收盘 |
| **选股触发** | **14:50** | 收盘前自动选股 |
| 数据下载 | 17:00 | 收盘后下载最新数据 |
