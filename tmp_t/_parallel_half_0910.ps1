# parallel half babysitter 2026-09-10 (ASCII only)
# Waits for fast chain (daily_automation_20260910.state.json) to reach terminal state,
# then runs parallel refresh -> prob head FORCE retrain -> inference -> deliver -> push -> combined.
# Fail-fast: any step rc != 0 stops the sequence and writes failed state.

$ErrorActionPreference = "Continue"
$root = "D:\AMINQT\AMINQT CODES"
$py = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$stateFile = "$root\tmp_t\_parallel_half_0910.state.txt"
$log = "$root\tmp_t\_parallel_half_0910.out"
$chainState = "$root\logs\daily_automation_20260910.state.json"

function Write-State($s) { Set-Content -Path $stateFile -Value $s -Encoding ascii }
function Log($m) { $ts = Get-Date -Format "HH:mm:ss"; Add-Content -Path $log -Value "[$ts] $m" -Encoding utf8 }

Write-State "waiting_chain"
Log "babysitter start; polling chain state"

$deadline = (Get-Date).AddHours(3)
$chainDone = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path $chainState) {
        try {
            $j = Get-Content $chainState -Raw | ConvertFrom-Json
            if ($j.status -ne "running") { $chainDone = $true; Log ("chain terminal: " + $j.status); break }
        } catch { }
    }
    Start-Sleep -Seconds 30
}
if (-not $chainDone) { Write-State "failed_wait_chain_timeout"; Log "chain did not reach terminal state in 3h"; exit 2 }

Start-Sleep -Seconds 10

$steps = @(
    @{ name = "refresh";          args = @("-u", "scripts/_refresh_parallel_checkpoints.py"); timeout = 10800 },
    @{ name = "prob_head";        args = @("-u", "scripts/_train_parallel_prob_head.py", "--force"); timeout = 3600 },
    @{ name = "parallel";         args = @("-u", "-m", "app.pipeline_parallel.runner"); timeout = 3600 },
    @{ name = "deliver_parallel"; args = @("-u", "scripts/_shortlist_t5_t10.py", "20260910"); timeout = 1800 },
    @{ name = "ths_push";         args = @("-u", "scripts/_ths_watchlist_push.py", "20260910"); timeout = 2400 },
    @{ name = "final_stocklist";  args = @("-u", "scripts/_final_stocklist.py", "20260910"); timeout = 900 },
    @{ name = "stocklist_combined"; args = @("-u", "scripts/_stocklist_combined.py", "20260910"); timeout = 900 }
)

foreach ($st in $steps) {
    Write-State ("running_" + $st.name)
    Log ("start " + $st.name)
    $p = Start-Process -FilePath $py -ArgumentList (("-X", "utf8") + $st.args) -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$root\tmp_t\_parallel_half_0910.$($st.name).out" -RedirectStandardError "$root\tmp_t\_parallel_half_0910.$($st.name).err"
    $exited = $p.WaitForExit($st.timeout * 1000)
    if (-not $exited) { Stop-Process -Id $p.Id -Force; Write-State ("failed_" + $st.name + "_timeout"); Log ($st.name + " TIMEOUT after " + $st.timeout + "s"); exit 3 }
    if ($p.ExitCode -ne 0) { Write-State ("failed_" + $st.name + "_rc" + $p.ExitCode); Log ($st.name + " FAILED rc=" + $p.ExitCode); exit 4 }
    Log ("ok " + $st.name + " rc=0")
}

Write-State "ok"
Log "all steps ok"
exit 0
