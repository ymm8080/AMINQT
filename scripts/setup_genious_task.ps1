<#
.SYNOPSIS
    Register the GENIOUS delivery task: 20:30 Asia/Shanghai, Mon-Fri.

.DESCRIPTION
    链路狙击规则层 (app/pipeline1/kongduo_triggers.py) 的 Excel 交付链, 20:30 北京跑。
    纯规则层只吃面板 OHLCV + winner_ratio, 不依赖任何模型训练产物。
    依赖 _daily_fetch.py (19:15) 已把当日行写入 V3 面板 → 20:30 留 75min 余量。
    脚本自身有新鲜度校验 + 有限等待 + 内存自愈, 面板缺当日行时不会产旧数据清单。

    任务用 XML 注册并显式指定 +08:00, 因此无论本机系统时区如何设置, 触发时间均为北京时间。

.NOTES
    Run as Administrator. Mon-Fri only (非交易日由脚本内部 exit 0 跳过)。
    独立任务, 不接 run_daily_automation.py (该链 Disabled 且正在重建)。
    日志 logs\genious_{date}.log / 状态 logs\genious_{date}.state.json。
#>

$ErrorActionPreference = "Stop"

$projectRoot = "D:\AMINQT\AMINQT CODES"
$python = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$timeZoneOffset = "+08:00"  # Asia/Shanghai
$startDate = (Get-Date -Format "yyyy-MM-dd")
$taskName = "AMINQT-Genious-2030"
$scriptPath = "$projectRoot\scripts\_genious_excel.py"
$triggerTime = "20:30:00"

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AMINQT GENIOUS at 20:30 Asia/Shanghai - 链路狙击规则层 Excel 交付 (冠军四段 + 观察池), 全池 OHLCV+winner_ratio 纯规则, 不截断</Description>
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
    <Enabled>true</Enabled>
    <AllowStartOnDemand>true</AllowStartOnDemand>
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
    Write-Host "Created: $taskName (Mon-Fri 20:30 Asia/Shanghai)" -ForegroundColor Green
}
finally {
    Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
}

Write-Host ""
schtasks /query /tn $taskName /fo LIST 2>$null | Select-String "TaskName|Status|Next Run|Start Time|Time Zone"

Write-Host ""
Write-Host "Manual trigger:" -ForegroundColor Yellow
Write-Host "  schtasks /run /tn $taskName"
Write-Host ""
Write-Host "Verify output:" -ForegroundColor Yellow
Write-Host "  logs\genious_{date}.log + logs\genious_{date}.state.json"
Write-Host "  D:\AMINQT\DAILY OPERATION\STOCK LIST\GENIOUS_{date}.xlsx"
Write-Host ""
Write-Host "Rollback (停交付):" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName $taskName -Confirm:`$false"
Write-Host "  或在 config\settings.py 里把 GENIOUS['enable'] 置 False"
