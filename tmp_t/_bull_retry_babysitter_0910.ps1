# Bull retry babysitter: cooldown -> retry -> gapfill -> chain bear. Events go to _bearchain_0910.log
# (ASCII only: PS 5.1 reads BOM-less ps1 as ANSI/GBK, UTF-8 Chinese breaks parsing)
$ErrorActionPreference = 'Continue'
$log = 'D:\AMINQT\AMINQT CODES\tmp_t\_bearchain_0910.log'
$bullPy = 'D:\AMINQT\AMINQT CODES\tmp_t\_ths_backfill3y_0910.py'
$gapPy = 'D:\AMINQT\AMINQT CODES\tmp_t\_bull_gapfill_0910.py'
$bearPy = 'D:\AMINQT\AMINQT CODES\tmp_t\_bear_backfill3y_0910.py'
$state = 'D:\AMINQT\AMINQT CODES\tmp_t\ths_bull_daily_0910\_state.json'
$td = 'D:\AMINQT\AMINQT CODES\tmp_t'

function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File $log -Append -Encoding utf8 }

Log "babysitter start (403 ban at 09:37, initial cooldown 45min)"
Start-Sleep -Seconds 2700

$ok = $false
foreach ($n in 1..6) {
    Log "bull retry $n launched"
    $p = Start-Process -FilePath python -ArgumentList "`"$bullPy`"" `
        -RedirectStandardOutput "$td\_bull_retry${n}_0910.out" `
        -RedirectStandardError "$td\_bull_retry${n}_0910.err" `
        -WindowStyle Hidden -PassThru -Wait
    $lastDone = ''
    try { $lastDone = (Get-Content $state -Raw | ConvertFrom-Json).last_done } catch {}
    Log "bull retry $n exit=$($p.ExitCode) last_done=$lastDone"
    if ($lastDone -eq '20260909') { $ok = $true; break }
    if ($n -lt 6) { Log "cooldown 45min before retry $($n+1)"; Start-Sleep -Seconds 2700 }
}
if (-not $ok) { Log "BULL-ALL-RETRIES-FAILED: need egress switch, bear NOT started"; exit 1 }

foreach ($n in 1..3) {
    Log "gapfill attempt $n launched"
    $g = Start-Process -FilePath python -ArgumentList "`"$gapPy`"" `
        -RedirectStandardOutput "$td\_bull_gapfill_run${n}_0910.out" `
        -RedirectStandardError "$td\_bull_gapfill_run${n}_0910.err" `
        -WindowStyle Hidden -PassThru -Wait
    Log "gapfill attempt $n exit=$($g.ExitCode)"
    if ($g.ExitCode -eq 0) { break }
    if ($n -lt 3) { Log "cooldown 45min"; Start-Sleep -Seconds 2700 }
}

Start-Sleep -Seconds 30
$b = Start-Process -FilePath python -ArgumentList "`"$bearPy`"" `
    -RedirectStandardOutput "$td\_bear_backfill_0910.out" `
    -RedirectStandardError "$td\_bear_backfill_0910.err" `
    -WindowStyle Hidden -PassThru
Log "bear launched pid=$($b.Id)"
