<#
.SYNOPSIS
    Register two scheduled tasks for AMINQT data pipelines.

.DESCRIPTION
    Pipeline 1: 19:15 daily (Asia/Shanghai) - _daily_fetch.py (appends today's data to V3 panel)
    Pipeline 2: 19:45 daily (Asia/Shanghai) - announcement data + V3 panel holdertrade update
    Pipeline 2 必须在 Pipeline 1 之后运行 (依赖今日行已写入面板).
    (2026-08-12 用户改档: 抓取 19:15 / 公告 19:45 / 自动化 20:15 北京.)

    任务使用 XML 注册并显式指定 +08:00 时区, 因此无论本机系统时区如何设置,
    触发时间均为北京时间 (19:15 / 19:45).

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (weekend skip is also
    handled inside the Python scripts).
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")

function New-AmqTaskXml {
    param(
        [string]$Description,
        [string]$ScriptPath,
        [string]$TriggerTime = "19:15:00"
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
      <StartBoundary>${startDate}T${TriggerTime}${timeZoneOffset}</StartBoundary>
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
        [string]$Description,
        [string]$TriggerTime = "19:15:00"
    )

    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue

    $xml = New-AmqTaskXml -Description $Description -ScriptPath $ScriptPath -TriggerTime $TriggerTime
    $tempFile = [System.IO.Path]::GetTempFileName() + ".xml"
    $xml | Set-Content -Path $tempFile -Encoding Unicode

    try {
        Register-ScheduledTask -TaskName $Name -Xml $xml -Force | Out-Null
        Write-Host "Created: $Name (Mon-Fri ${TriggerTime} Asia/Shanghai)" -ForegroundColor Green
    }
    finally {
        Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
    }
}

# --- Pipeline 1: 19:15 daily fetch (append to V3 panel) ---
Register-AmqTask `
    -Name "AMINQT-MarketData-22h" `
    -ScriptPath "$projectRoot\_daily_fetch.py" `
    -TriggerTime "19:15:00" `
    -Description "AMINQT daily fetch at 19:15 Asia/Shanghai — appends one day to V3 panel (_daily_fetch.py: OHLCV, adj, daily_basic, stk_limit, moneyflow, cyq_perf, margin, lhb + derived features)"

# --- Pipeline 2: 19:45 announcement data + V3 panel holdertrade update ---
Register-AmqTask `
    -Name "AMINQT-Announcement-22h" `
    -ScriptPath "$projectRoot\scripts\run_announcement_pipeline.py" `
    -Description "AMINQT announcement pipeline at 19:45 Asia/Shanghai — fetches fina_indicator/holdertrade/holdernumber/anns_d + updates V3 panel holdertrade columns (runs after Pipeline 1)" `
    -TriggerTime "19:45:00"

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
