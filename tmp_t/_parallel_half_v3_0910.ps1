# parallel half babysitter v3 2026-09-10 (ASCII only)
# NO THS PUSH anywhere (user order 2026-09-10).
# Waits for the night queue (_night_queue_0910.py, PID 26548: retrain -> predict chain -> prob A/B)
# to fully exit, then waits for free RAM, then runs:
#   parallel inference -> deliver_parallel -> final_stocklist -> stocklist_combined
# Fix vs v2: parameterless WaitForExit() after the timed wait so ExitCode is populated
# (PS 5.1 + redirected streams leaves ExitCode null otherwise -> v2 falsely failed prob_head).

$ErrorActionPreference = "Continue"
$root = "D:\AMINQT\AMINQT CODES"
$py = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$stateFile = "$root\tmp_t\_parallel_half_v3_0910.state.txt"
$log = "$root\tmp_t\_parallel_half_v3_0910.out"
$queuePid = 26548

function Write-State($s) { Set-Content -Path $stateFile -Value $s -Encoding ascii }
function Log($m) {
    $ts = Get-Date -Format "HH:mm:ss"
    [System.IO.File]::AppendAllText($log, "[$ts] $m`r`n", [System.Text.Encoding]::UTF8)
}

Write-State "waiting_queue"
Log "v3 babysitter start; waiting for night queue PID $queuePid (no push edition)"

$deadline = (Get-Date).AddHours(7)
while (Get-Process -Id $queuePid -ErrorAction SilentlyContinue) {
    if ((Get-Date) -gt $deadline) { Write-State "failed_wait_queue_timeout"; Log "night queue PID $queuePid still alive after 7h"; exit 2 }
    Start-Sleep -Seconds 60
}
Log "night queue gone; waiting for >=4GB free RAM"
Write-State "waiting_ram"
$ramDeadline = (Get-Date).AddHours(1)
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = [Math]::Round($os.FreePhysicalMemory / 1MB, 1)
    if ($freeGB -ge 4.0) { Log "free RAM ${freeGB}GB ok"; break }
    if ((Get-Date) -gt $ramDeadline) { Write-State "failed_wait_ram_timeout"; Log "free RAM only ${freeGB}GB after 1h"; exit 5 }
    Start-Sleep -Seconds 30
}

Log "120s settle buffer"
Start-Sleep -Seconds 120

$steps = @(
    @{ name = "parallel";         args = @("-u", "-m", "app.pipeline_parallel.runner"); timeout = 3600 },
    @{ name = "deliver_parallel"; args = @("-u", "scripts/_shortlist_t5_t10.py", "20260910"); timeout = 1800 },
    @{ name = "final_stocklist";  args = @("-u", "scripts/_final_stocklist.py", "20260910"); timeout = 900 },
    @{ name = "stocklist_combined"; args = @("-u", "scripts/_stocklist_combined.py", "20260910"); timeout = 900 }
)

# If the night chain already built stocklist_combined BEFORE our parallel shortlist
# existed (chain runs ahead of us), its PARALLEL page is missing. Archive it (WORM:
# rename, never delete) so the combined step below rebuilds with all pages.
$sld = "D:\AMINQT\DAILY OPERATION\STOCK LIST"
$xlsx = Join-Path $sld "stocklist_combined_20260910.xlsx"
$psl = Get-ChildItem -Path $sld -Filter "parallel_shortlist_20260910__*.csv" -ErrorAction SilentlyContinue | Select-Object -First 1
if ((Test-Path $xlsx) -and $psl -and ((Get-Item $xlsx).LastWriteTime -lt $psl.LastWriteTime)) {
    $arch = Join-Path $sld ("stocklist_combined_20260910__pre_parallel" + (Get-Date -Format "HHmmss") + ".xlsx")
    Move-Item -Path $xlsx -Destination $arch -Force
    Log ("archived stale combined (pre-parallel) -> " + (Split-Path -Leaf $arch))
}

foreach ($st in $steps) {
    Write-State ("running_" + $st.name)
    Log ("start " + $st.name)
    $p = Start-Process -FilePath $py -ArgumentList (("-X", "utf8") + $st.args) -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$root\tmp_t\_parallel_half_v3_0910.$($st.name).out" -RedirectStandardError "$root\tmp_t\_parallel_half_v3_0910.$($st.name).err"
    $exited = $p.WaitForExit($st.timeout * 1000)
    if (-not $exited) { Stop-Process -Id $p.Id -Force; Write-State ("failed_" + $st.name + "_timeout"); Log ($st.name + " TIMEOUT after " + $st.timeout + "s"); exit 3 }
    $null = $p.WaitForExit()
    $rc = $p.ExitCode
    Log ($st.name + " exited rc=" + $rc)
    if ($rc -ne 0) { Write-State ("failed_" + $st.name + "_rc" + $rc); Log ($st.name + " FAILED rc=" + $rc); exit 4 }
    Log ("ok " + $st.name)
}

Write-State "ok"
Log "all steps ok (no push)"
exit 0
