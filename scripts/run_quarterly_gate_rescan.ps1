# 季度闸甜点重扫 — 计划任务 AMINQT-QuarterlyGateRescan 调用 (用户 2026-08-14 定案)
# 每 13 周周六 06:45 (系统时间) 重跑 legacy dual 闸参数诊断: scripts/_diag_legacy_hitrate_topn.py
# 只做诊断不改生产 (WORM 落盘 DATA_OTHERS/diag/), 调整须另行人工定案 — 见 memory quarterly-gate-rescan-requirement
$ErrorActionPreference = 'Continue'
$root = 'D:\AMINQT\AMINQT CODES'
$py = 'C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe'
$logDir = 'D:\AMINQT\DATA OTHERS\diag'
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ('quarterly_scan_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $py (Join-Path $root 'scripts\_diag_legacy_hitrate_topn.py') *> $log
