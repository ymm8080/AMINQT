<#
.SYNOPSIS
    Register two scheduled tasks for AMINQT data pipelines.

.DESCRIPTION
    Pipeline 1: 22:00 daily (Asia/Shanghai) - market data (OHLCV, PE/PB, margin, northbound, lhb, stk_limit)
    Pipeline 2: 22:00 daily (Asia/Shanghai) - announcement data (fina_indicator, holdertrade, holdernumber, anns_d)
    两个管道统一在北京时间 22:00 运行 (2026-08-01 起): 行情数据 16:00 已结算完毕,
    当日公告 22:00 基本发布完成 (公告脚本拉取当日+前日窗口).

    任务使用 XML 注册并显式指定 +08:00 时区, 因此无论本机系统时区如何设置,
    触发时间均为北京时间 22:00.

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (weekend skip is also
    handled inside the Python scripts).
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "python"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")

function New-AmqTaskXml {
    param(
        [string]$Description,
        [string]$ScriptPath
    )

    $arguments = '"{0}"' -f $ScriptPath

    @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>$Description</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>${startDate}T22:00:00${timeZoneOffset}</StartBoundary>
      <ScheduleByWeek>
        <DaysOfWeek>
          <Monday/>
          <Tuesday/>
          <Wednesday/>
          <Thursday/>
          <Friday/>
        </DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
    </Principal>
  </Principals>
  <Settings>
    <Enabled>true</Enabled>
    <AllowStartOnDemand>true</AllowStartOnDemand>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$python</Command>
      <Arguments>$arguments</Arguments>
      <WorkingDirectory>$projectRoot</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
}

function Register-AmqTask {
    param(
        [string]$Name,
        [string]$ScriptPath,
        [string]$Description
    )

    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue

    $xml = New-AmqTaskXml -Description $Description -ScriptPath $ScriptPath
    $tempFile = [System.IO.Path]::GetTempFileName() + ".xml"
    $xml | Set-Content -Path $tempFile -Encoding Unicode

    try {
        Register-ScheduledTask -TaskName $Name -Xml $xml -Force | Out-Null
        Write-Host "Created: $Name (Mon-Fri 22:00 Asia/Shanghai)" -ForegroundColor Green
    }
    finally {
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
    }
}

# --- Pipeline 1: 22:00 market data ---
Register-AmqTask `
    -Name "AMINQT-MarketData-22h" `
    -ScriptPath "$projectRoot\scripts\run_daily_market_pipeline.py" `
    -Description "AMINQT market data pipeline at 22:00 Asia/Shanghai (OHLCV, PE/PB, margin, northbound, lhb, stk_limit)"

# --- Pipeline 2: 22:00 announcement data ---
Register-AmqTask `
    -Name "AMINQT-Announcement-22h" `
    -ScriptPath "$projectRoot\scripts\run_announcement_pipeline.py" `
    -Description "AMINQT announcement data pipeline at 22:00 Asia/Shanghai (fina_indicator, holdertrade, holdernumber, anns_d)"

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Summary ===" -ForegroundColor Cyan
schtasks /query /tn "AMINQT-MarketData-22h" /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Start Time|Time Zone"
schtasks /query /tn "AMINQT-Announcement-22h" /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Start Time|Time Zone"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn AMINQT-MarketData-22h"
Write-Host "  schtasks /run /tn AMINQT-Announcement-22h"
Write-Host ""
Write-Host "View task details:" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn AMINQT-MarketData-22h /v"
Write-Host "  schtasks /query /tn AMINQT-Announcement-22h /v"
