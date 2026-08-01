<#
.SYNOPSIS
    Register two scheduled tasks for AMINQT data pipelines.

.DESCRIPTION
    Pipeline 1: 22:00 daily - market data (OHLCV, PE/PB, margin, northbound, lhb, stk_limit)
    Pipeline 2: 22:00 daily - announcement data (fina_indicator, holdertrade, holdernumber, anns_d)
    两个管道统一在 22:00 运行 (2026-08-01 起): 行情数据 16:00 已结算完毕,
    当日公告 22:00 基本发布完成 (公告脚本拉取当日+前日窗口).

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (weekend skip is also
    handled inside the Python scripts).
#>

$ErrorActionPreference = "Continue"   # schtasks /query 找不到任务时 stderr 在 PS5.1 Stop 模式下会变成终止错误

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "python"

# --- Pipeline 1: 22:00 market data ---
$task1Name = "AMINQT-MarketData-22h"

# Remove existing task if present
schtasks /query /tn $task1Name 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing existing task: $task1Name"
    schtasks /delete /tn $task1Name /f
}

# Create task: Monday-Friday 22:00
# 注意: 必须用 --% 停止符 + \" 转义, 否则 PS 5.1 会吞掉路径中的引号导致 /tr 解析失败
schtasks --% /create /tn "AMINQT-MarketData-22h" /tr "python \"D:\AMINQT\AMINQT CODES\scripts\run_daily_market_pipeline.py\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 22:00 /f

Write-Host "Created: $task1Name (Mon-Fri 22:00)" -ForegroundColor Green

# --- Pipeline 2: 22:00 announcement data ---
$task2Name = "AMINQT-Announcement-22h"

# Remove existing task if present
schtasks /query /tn $task2Name 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing existing task: $task2Name"
    schtasks /delete /tn $task2Name /f
}

# Create task: Monday-Friday 22:00
schtasks --% /create /tn "AMINQT-Announcement-22h" /tr "python \"D:\AMINQT\AMINQT CODES\scripts\run_announcement_pipeline.py\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 22:00 /f

Write-Host "Created: $task2Name (Mon-Fri 22:00)" -ForegroundColor Green

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Summary ===" -ForegroundColor Cyan
schtasks /query /tn $task1Name /fo LIST 2>$null | Select-String "TaskName|Status|Next Run"
schtasks /query /tn $task2Name /fo LIST 2>$null | Select-String "TaskName|Status|Next Run"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn AMINQT-MarketData-22h"
Write-Host "  schtasks /run /tn AMINQT-Announcement-22h"
Write-Host ""
Write-Host "View task details:" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn AMINQT-MarketData-22h /v"
Write-Host "  schtasks /query /tn AMINQT-Announcement-22h /v"
