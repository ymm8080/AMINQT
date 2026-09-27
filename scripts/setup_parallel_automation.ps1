<#
.SYNOPSIS
    Register the AMINQT PARALLEL-only automation scheduled task.

.DESCRIPTION
    接替已停用的 AMINQT-DailyAutomation-2330, 在同一 23:30 时间槽只跑 PARALLEL 模块。
    Orchestrates (scripts/run_parallel_automation.py):
      [refresh]  parallel 3y checkpoints rebuild
      [parallel] sniper/fusion/slow_bull regenerate
    任务用 XML 注册并显式指定 +08:00, 因此无论本机系统时区如何设置, 触发时间均为北京时间。

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (Asia/Shanghai +08:00).
    Manual trigger: schtasks /run /tn AMINQT-ParallelAutomation-2330
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
        [string]$TriggerTime = "23:30:00"
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
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
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
        [string]$TriggerTime = "23:30:00"
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

# --- PARALLEL-only automation: 23:30 daily, 接替已停用的 AMINQT-DailyAutomation-2330 ---
Register-AmqTask `
    -Name "AMINQT-ParallelAutomation-2330" `
    -ScriptPath "$projectRoot\scripts\run_parallel_automation.py" `
    -Description "AMINQT PARALLEL-only automation at 23:30 Asia/Shanghai - parallel sniper/fusion/slow_bull regenerate. Replaces the disabled AMINQT-DailyAutomation-2330 at the same time slot. Requires 19:15 fetch to have run." `
    -TriggerTime "23:30:00"

# --- 旧主链必须保持停用 ---
# 两条链同槽 23:30 触发时, 后启动的那条会在启动守卫看到对方的重活进程 -> skipped,
# 等于当晚没跑; 且两条链并发跑 refresh/parallel = 双建 OOM. 故此处只告警不代劳 --
# 停不停是用户的决定.
$legacyTask = Get-ScheduledTask -TaskName "AMINQT-DailyAutomation-2330" -ErrorAction SilentlyContinue
if ($null -ne $legacyTask -and $legacyTask.State -ne "Disabled") {
    Write-Host ""
    Write-Host "[WARN] AMINQT-DailyAutomation-2330 仍处于 $($legacyTask.State), 与本链同槽." -ForegroundColor Yellow
    Write-Host "       若不禁用, 两条链 23:30 互撞, 本链会被守卫判 skipped." -ForegroundColor Yellow
    Write-Host "       禁用: Disable-ScheduledTask -TaskName 'AMINQT-DailyAutomation-2330'" -ForegroundColor Yellow
}

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Summary ===" -ForegroundColor Cyan
schtasks /query /tn "AMINQT-ParallelAutomation-2330" /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Start Time|Time Zone"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn AMINQT-ParallelAutomation-2330"
Write-Host ""
Write-Host "View task details:" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn AMINQT-ParallelAutomation-2330 /v"
