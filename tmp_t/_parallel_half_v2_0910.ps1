# parallel half babysitter v2 2026-09-10 (ASCII only)
# v2: NO THS PUSH anywhere (user order 2026-09-10 "no pushes needed").
# Adopts running refresh (python PID 5072), then runs:
#   prob head FORCE retrain -> inference -> deliver_parallel -> final_stocklist -> stocklist_combined
# Fail-fast: any step rc != 0 stops the sequence and writes failed state.

$ErrorActionPreference = "Continue"
$root = "D:\AMINQT\AMINQT CODES"
$py = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$stateFile = "$root\tmp_t\_parallel_half_v2_0910.state.txt"
$log = "$root\tmp_t\_parallel_half_v2_0910.out"
$refreshPid = 5072

function Write-State($s) { Set-Content -Path $stateFile -Value $s -Encoding ascii }
function Log($m) { $ts = Get-Date -Format "HH:mm:ss"; Add-Content -Path $log -Value "[$ts] $m" -Encoding utf8 }

Write-State "waiting_refresh"
Log "v2 babysitter start; adopting refresh PID $refreshPid (no push edition)"

# Wait for the already-running refresh process to exit (2h deadline).
$deadline = (Get-Date).AddHours(2)
while (Get-Process -Id $refreshPid -ErrorAction SilentlyContinue) {
    if ((Get-Date) -gt $deadline) { Write-State "failed_wait_refresh_timeout"; Log "refresh PID $refreshPid still alive after 2h"; exit 2 }
    Start-Sleep -Seconds 30
}
Log "refresh PID $refreshPid gone; 30s settle buffer"
Start-Sleep -Seconds 30

$steps = @(
    @{ name = "prob_head";        args = @("-u", "scripts/_train_parallel_prob_head.py", "--force"); timeout = 3600 },
    @{ name = "parallel";         args = @("-u", "-m", "app.pipeline_parallel.runner"); timeout = 3600 },
    @{ name = "deliver_parallel"; args = @("-u", "scripts/_shortlist_t5_t10.py", "20260910"); timeout = 1800 },
    @{ name = "final_stocklist";  args = @("-u", "scripts/_final_stocklist.py", "20260910"); timeout = 900 },
    @{ name = "stocklist_combined"; args = @("-u", "scripts/_stocklist_combined.py", "20260910"); timeout = 900 }
)

foreach ($st in $steps) {
    Write-State ("running_" + $st.name)
    Log ("start " + $st.name)
    $p = Start-Process -FilePath $py -ArgumentList (("-X", "utf8") + $st.args) -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$root\tmp_t\_parallel_half_v2_0910.$($st.name).out" -RedirectStandardError "$root\tmp_t\_parallel_half_v2_0910.$($st.name).err"
    $exited = $p.WaitForExit($st.timeout * 1000)
    if (-not $exited) { Stop-Process -Id $p.Id -Force; Write-State ("failed_" + $st.name + "_timeout"); Log ($st.name + " TIMEOUT after " + $st.timeout + "s"); exit 3 }
    if ($p.ExitCode -ne 0) { Write-State ("failed_" + $st.name + "_rc" + $p.ExitCode); Log ($st.name + " FAILED rc=" + $p.ExitCode); exit 4 }
    Log ("ok " + $st.name + " rc=0")
}

Write-State "ok"
Log "all steps ok (no push)"
exit 0
