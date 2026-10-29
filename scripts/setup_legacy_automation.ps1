<#
.SYNOPSIS
    Register the AMINQT LEGACY-only automation scheduled task (21:30 Asia/Shanghai).

.DESCRIPTION
    2026-10-17 用户令: 恢复 LEGACY 日常管道并挂在正确时间槽, 保证
    LEGACY / PARALLEL / GENIOUS / 密度四模块次日 08:00 前全部交付。
    夜间节奏 (Mon-Fri +08:00):
      19:15 fetch + 19:45 announcements (上游, 不变)
      20:30 GENIOUS     (实测 ~10-25min 收工)
      21:30 LEGACY      (本任务, ~2.5h; 周五含 retrain ~+2h)
      23:30 PARALLEL    (guard 见本链哨兵会等本链退出; 周五 ~02:00 清场后 ~04:00 收工)
    Orchestrates (scripts/run_legacy_automation.py):
      [cyq/freshness/canary] 预检  [retrain] 周五 legacy 全量重训
      [legacy_prob_head/legacy/deliver] 清单交付  [drift/gate_audit] 监控
    跨链步骤 (密度/combined/THS推送) 在 PARALLEL 链尾, 不在本任务。
    任务用 XML 注册并显式指定 +08:00, 因此无论本机系统时区如何设置, 触发时间均为北京时间。

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (Asia/Shanghai +08:00).
    Manual trigger: schtasks /run /tn AMINQT-LegacyAutomation-2130
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
        [string]$TriggerTime = "21:30:00"
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
        [string]$TriggerTime = "21:30:00"
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

# --- LEGACY-only automation: 21:30 daily, 20:30 GENIOUS 之后 / 23:30 PARALLEL 之前 ---
Register-AmqTask `
    -Name "AMINQT-LegacyAutomation-2130" `
    -ScriptPath "$projectRoot\scripts\run_legacy_automation.py" `
    -Description "AMINQT LEGACY-only automation at 21:30 Asia/Shanghai - legacy preflight (cyq/freshness/canary) + Friday full retrain + legacy prob head + stock list delivery + drift/gate monitors. Cross-module steps (density/combined/THS push) run at the tail of the 23:30 PARALLEL chain. Requires 19:15 fetch to have run." `
    -TriggerTime "21:30:00"

# --- 旧全量主链必须保持停用 (0928 用户令; 2026-10-17 起其职责由本链+PARALLEL链分担) ---
$oldTask = Get-ScheduledTask -TaskName "AMINQT-DailyAutomation-2330" -ErrorAction SilentlyContinue
if ($null -ne $oldTask -and $oldTask.State -ne "Disabled") {
    Write-Host ""
    Write-Host "[WARN] AMINQT-DailyAutomation-2330 is $($oldTask.State), not Disabled." -ForegroundColor Yellow
    Write-Host "       It duplicates LEGACY+PARALLEL work across the two new chains." -ForegroundColor Yellow
    Write-Host "       Disable: Disable-ScheduledTask -TaskName 'AMINQT-DailyAutomation-2330'" -ForegroundColor Yellow
}

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Summary ===" -ForegroundColor Cyan
schtasks /query /tn "AMINQT-LegacyAutomation-2130" /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Start Time|Time Zone"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn AMINQT-LegacyAutomation-2130"
Write-Host ""
Write-Host "View task details:" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn AMINQT-LegacyAutomation-2130 /v"
