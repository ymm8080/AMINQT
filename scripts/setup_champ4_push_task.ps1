<#
.SYNOPSIS
    Register the GENIOUS 冠军四段 daily THS push: 07:45 Asia/Shanghai, Mon-Fri.

.DESCRIPTION
    scripts\_ths_champ4_push.py 的计划任务: 把最新交付日 GENIOUS_{date}.xlsx 的
    「冠军四段」页推入同花顺自选股 (2026-11-11 用户令 "auto push to ths for
    genious file 冠军四段 ON DAILY BASIS"; 独立旗标 GENIOUS.push_champ4_to_ths,
    不受 2026-11-03 停推总闸影响)。

    双触发: GENIOUS 交付后当晚 spawn (20:3x) + 本任务次晨 07:45 重推兜底 —
    幂等: 该日结果单全 landed 则跳过; blocked/manual (用户在场 fail-closed 等)
    则按新小时槽位重推。07:45 定位: 开盘 (09:30) 前、用户落座前窗口; 机器未开机
    由 StartWhenAvailable 开机补跑。

.NOTES
    Run as Administrator. Mon-Fri only。心跳与逐码终态: logs\champ4_push.log 与
    STOCK LIST\ths_push_result_{date}__{HH}__genious__champ4.csv。

    手动验证 (只列码单不推):
    C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe `
        "D:\AMINQT\AMINQT CODES\scripts\_ths_champ4_push.py" --dry-run
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")
$taskName = "AMINQT-Champ4-Push-0745"
$scriptPath = "$projectRoot\scripts\_ths_champ4_push.py"
$triggerTime = "07:45:00"

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AMINQT GENIOUS 冠军四段 daily THS push at 07:45 Asia/Shanghai - 最新交付日 champ4 页推入自选 (独立旗标 push_champ4_to_ths, 幂等: 全 landed 跳过); 让路空闲闸 fail-closed 由本任务重推</Description>
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
    Write-Host "Created: $taskName (Mon-Fri 07:45 Asia/Shanghai)" -ForegroundColor Green
}
finally {
    Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
}

Write-Host ""
schtasks /query /tn $taskName /fo LIST 2>$null | Select-String "TaskName|Status|Next Run"

Write-Host ""
Write-Host "Rollback:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
