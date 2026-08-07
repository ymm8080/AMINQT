<#
.SYNOPSIS
    Register the AMINQT four-module daily automation scheduled task.

.DESCRIPTION
    Runs after the 22:00 fetch + 22:40 announcement pipelines so today's V3 panel
    data is in place. Orchestrates (scripts/run_daily_automation.py):
      [refresh]  parallel 3y checkpoints rebuild
      [retrain]  legacy main+dual weekly full retrain (Fridays only, OOS gate)
      [parallel] sniper/fusion/slow_bull regenerate
      [legacy]   legacy stock list generation
      [deliver]  legacy list delivery to STOCK_LIST_DIR
    "Push to dashboard" = artifacts land in dashboard-read WORM dirs automatically.

.NOTES
    Run as Administrator. Tasks run Monday-Friday only (Asia/Shanghai +08:00).
    Manual trigger: schtasks /run /tn AMINQT-DailyAutomation-2330
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "python"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")

function New-AmqTaskXml {
    param(
        [string]$Description,
        [string]$ScriptPath,
        [string]$TriggerTime = "22:00:00"
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
        [string]$TriggerTime = "22:00:00"
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

# --- Four-module automation: 23:30 daily, after 22:00 fetch + 22:40 announcement ---
Register-AmqTask `
    -Name "AMINQT-DailyAutomation-2330" `
    -ScriptPath "$projectRoot\scripts\run_daily_automation.py" `
    -Description "AMINQT four-module daily automation at 23:30 Asia/Shanghai — legacy main/dual weekly retrain (Fri) + parallel sniper/fusion regenerate + legacy stock list generation and delivery. Requires 22:00 fetch and 22:40 announcement to have run." `
    -TriggerTime "23:30:00"

# --- Summary ---
Write-Host ""
Write-Host "=== Scheduled Tasks Summary ===" -ForegroundColor Cyan
schtasks /query /tn "AMINQT-DailyAutomation-2330" /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Start Time|Time Zone"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn AMINQT-DailyAutomation-2330"
Write-Host ""
Write-Host "View task details:" -ForegroundColor Yellow
Write-Host "  schtasks /query /tn AMINQT-DailyAutomation-2330 /v"
