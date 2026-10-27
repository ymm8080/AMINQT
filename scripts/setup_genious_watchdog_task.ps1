<#
.SYNOPSIS
    Register the GENIOUS missing-file watchdog: 21:10 Asia/Shanghai, Mon-Fri.

.DESCRIPTION
    scripts\_genious_backfill_missing.py 的计划任务: 扫最近 5 个交易日的
    GENIOUS_{date}.xlsx, 缺谁补谁 (顺序, --no-push)。堵 2026-10-22/10-26 两次
    漏交付: 20:30 主任务未触发或失败后, 错过日期无人补。

    21:10 定位: 晚于 GENIOUS (20:30, 实测 ~10min 收尾) 早于 LEGACY (21:30);
    脚本自身另有让路闸 — 存活的 _genious_excel.py 或任一夜链哨兵 → 本轮退出。
    StartWhenAvailable=true: 开机补跑时也能补历史缺日。

.NOTES
    Run as Administrator. Mon-Fri only; 交易日/日历判定在脚本内部 (日历不可得
    fail-open 退出)。心跳与补跑结果: logs\genious_watchdog.log。

    手动验证 (只报缺哪些, 不补):
    C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe `
        "D:\AMINQT\AMINQT CODES\scripts\_genious_backfill_missing.py" --dry-run
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")
$taskName = "AMINQT-Genious-Watchdog-2110"
$scriptPath = "$projectRoot\scripts\_genious_backfill_missing.py"
$triggerTime = "21:10:00"

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AMINQT GENIOUS watchdog at 21:10 Asia/Shanghai - 扫近 5 交易日缺 GENIOUS_{date}.xlsx 则顺序补跑 (--no-push); 让路存活 GENIOUS/夜链; 日历不可得 fail-open</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>${startDate}T${triggerTime}${timeZoneOffset}</StartBoundary>
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
      <Arguments>"$scriptPath"</Arguments>
      <WorkingDirectory>$projectRoot</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

$tempFile = [System.IO.Path]::GetTempFileName() + ".xml"
$xml | Set-Content -Path $tempFile -Encoding Unicode
try {
    Register-ScheduledTask -TaskName $taskName -Xml $xml -Force | Out-Null
    Write-Host "Created: $taskName (Mon-Fri 21:10 Asia/Shanghai)" -ForegroundColor Green
}
finally {
    Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
}

Write-Host ""
schtasks /query /tn $taskName /fo LIST 2>$null | Select-String "TaskName|Status|Next Run"

Write-Host ""
Write-Host "Rollback:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
