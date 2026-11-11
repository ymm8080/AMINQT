<#
.SYNOPSIS
    Register the stocklist_combined missing-file watchdog: 08:30 Asia/Shanghai, Mon-Fri.

.DESCRIPTION
    scripts\_stocklist_combined_backfill.py 的计划任务: 扫最近 5 个交易日,
    有 legacy_stocklist_{date} 清单且缺 stocklist_combined_{date}*.xlsx 则补产
    (缺源跳页口径)。背景: 2026-10-30 起 PARALLEL 夜链 refresh OOM 连锁
    deliver_parallel 失败, combined 断供至 11-10; 2026-11-11 combined 已在链内
    解耦 deliver_parallel, 本看门狗兜"夜链整链没跑/中途死"的洞。

    08:30 定位: 三条夜链 (LEGACY 21:30 / PARALLEL 23:30) 凌晨已收尾, 开盘
    (09:30) 前交付; 脚本自身另有让路闸 — 存活夜链哨兵或 _stocklist_combined.py
    → 本轮退出。StartWhenAvailable=true: 开机补跑时也能补历史缺日。

.NOTES
    Run as Administrator. Mon-Fri only; 交易日/日历判定在脚本内部 (日历不可得
    fail-open 退出)。心跳与补跑结果: logs\combined_watchdog.log。

    手动验证 (只报缺哪些, 不补):
    C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe `
        "D:\AMINQT\AMINQT CODES\scripts\_stocklist_combined_backfill.py" --dry-run
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")
$taskName = "AMINQT-Combined-Watchdog-0830"
$scriptPath = "$projectRoot\scripts\_stocklist_combined_backfill.py"
$triggerTime = "08:30:00"

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AMINQT stocklist_combined watchdog at 08:30 Asia/Shanghai - 扫近 5 交易日, 有 LEGACY 清单且缺 stocklist_combined 则补产 (缺源跳页口径); 让路存活夜链/combined; 日历不可得 fail-open</Description>
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
    Write-Host "Created: $taskName (Mon-Fri 08:30 Asia/Shanghai)" -ForegroundColor Green
}
finally {
    Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
}

Write-Host ""
schtasks /query /tn $taskName /fo LIST 2>$null | Select-String "TaskName|Status|Next Run"

Write-Host ""
Write-Host "Rollback:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
