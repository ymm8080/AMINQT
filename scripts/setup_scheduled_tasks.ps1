<#
.SYNOPSIS
    Register two scheduled tasks for AMINQT data pipelines.

.DESCRIPTION
    Pipeline 1: 16:00 daily - market data (OHLCV, PE/PB, margin, northbound, lhb, stk_limit)
    Pipeline 2: 08:00 daily - announcement data (fina_indicator, holdertrade, holdernumber, anns_d)

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (weekend skip is also
    handled inside the Python scripts).
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "python"

# --- Pipeline 1: 16:00 market data ---
$task1Name = "AMINQT-MarketData-16h"
$task1Cmd = "$python `"$projectRoot\scripts\run_daily_market_pipeline.py`""

# Remove existing task if present
schtasks /query /tn $task1Name 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing existing task: $task1Name"
    schtasks /delete /tn $task1Name /f
}

# Create task: Monday-Friday 16:00
schtasks /create `
    /tn $task1Name `
    /tr $task1Cmd `
    /sc weekly `
    /d MON,TUE,WED,THU,FRI `
    /st 16:00 `
    /f

Write-Host "Created: $task1Name (Mon-Fri 16:00)" -ForegroundColor Green

# --- Pipeline 2: 08:00 announcement data ---
$task2Name = "AMINQT-Announcement-08h"
$task2Cmd = "$python `"$projectRoot\scripts\run_announcement_pipeline.py`""

# Remove existing task if present
schtasks /query /tn $task2Name 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Removing existing task: $task2Name"
    schtasks /delete /tn $task2Name /f
}

# Create task: Monday-Friday 08:00
schtasks /create `
    /tn $task2Name `
    /tr $task2Cmd `
    /sc weekly `
    /d MON,TUE,WED,THU,FRI `
    /st 08:00 `
    /f

Write-Host "Created: $task2Name (Mon-Fri 08:00)" -ForegroundColor Green

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Summary ===" -ForegroundColor Cyan
schtasks /query /tn "AMINQT-*" /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Last Run"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn AMINQT-MarketData-16h"
Write-Host "  schtasks /run /tn AMINQT-Announcement-08h"
Write-Host ""
Write-Host "View task details:" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn AMINQT-MarketData-16h /v"
Write-Host "  schtasks /query /tn AMINQT-Announcement-08h /v"
