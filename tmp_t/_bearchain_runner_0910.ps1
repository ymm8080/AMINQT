# 看涨回填成功完成后自动接力看跌回填 (成功判据: bull state last_done = 20260909)
$ErrorActionPreference = 'Continue'
$bullPid = 15744
$bullState = 'D:\AMINQT\AMINQT CODES\tmp_t\ths_bull_daily_0910\_state.json'
$runlog = 'D:\AMINQT\AMINQT CODES\tmp_t\_bearchain_0910.log'
$bearPy = 'D:\AMINQT\AMINQT CODES\tmp_t\_bear_backfill3y_0910.py'
$bearOut = 'D:\AMINQT\AMINQT CODES\tmp_t\_bear_backfill_0910.out'
$bearErr = 'D:\AMINQT\AMINQT CODES\tmp_t\_bear_backfill_0910.err'

"runner start $(Get-Date -Format s)" | Out-File $runlog -Encoding utf8
while (Get-Process -Id $bullPid -ErrorAction SilentlyContinue) { Start-Sleep 60 }
"bull PID $bullPid exited $(Get-Date -Format s)" | Out-File $runlog -Append -Encoding utf8
Start-Sleep 20

$lastDone = ''
try { $lastDone = (Get-Content $bullState -Raw | ConvertFrom-Json).last_done } catch {}
"bull last_done=$lastDone" | Out-File $runlog -Append -Encoding utf8

if ($lastDone -ne '20260909') {
    "BULL-DID-NOT-FINISH: 不接力看跌 (避免双打同一出口), 请人工核查bull后手动启动" |
        Out-File $runlog -Append -Encoding utf8
    exit 1
}

Start-Sleep 30
$p = Start-Process -FilePath python -ArgumentList "`"$bearPy`"" `
    -RedirectStandardOutput $bearOut -RedirectStandardError $bearErr `
    -WindowStyle Hidden -PassThru
"bear launched pid=$($p.Id) $(Get-Date -Format s)" | Out-File $runlog -Append -Encoding utf8
