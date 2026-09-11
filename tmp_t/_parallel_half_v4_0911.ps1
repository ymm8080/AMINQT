# parallel half babysitter v4 2026-09-11 (ASCII only)
# NO THS PUSH anywhere (no-push flag still in place).
# v3 died: RAM gate (4GB) passed at 06:19 while THS bull A/B #1 (started 06:16:39)
# was still ramping to 6.2GB WS -> parallel runner hard-killed 06:26, rc blank.
# v4 fix: wait for ALL _ths_ab_minibacktest processes to EXIT FIRST (cmdline match,
# not fixed PID), then free RAM >= 6GB, then 180s settle, then:
#   parallel inference -> deliver_parallel -> final_stocklist -> stocklist_combined

$ErrorActionPreference = "Continue"
$root = "D:\AMINQT\AMINQT CODES"
$py = "C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
$stateFile = "$root\tmp_t\_parallel_half_v4_0911.state.txt"
$log = "$root\tmp_t\_parallel_half_v4_0911.out"

function Write-State($s) { Set-Content -Path $stateFile -Value $s -Encoding ascii }
function Log($m) {
    $ts = Get-Date -Format "HH:mm:ss"
    [System.IO.File]::AppendAllText($log, "[$ts] $m`r`n", [System.Text.Encoding]::UTF8)
}

function Get-AbProcs {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "_ths_ab_minibacktest" }
}

Write-State "waiting_ab1"
Log "v4 start; waiting for THS bull A/B #1 (_ths_ab_minibacktest) to exit"

$deadline = (Get-Date).AddHours(4)
while ($true) {
    $ab = Get-AbProcs
    if (-not $ab) { Log "A/B #1 gone"; break }
    if ((Get-Date) -gt $deadline) { Write-State "failed_wait_ab1_timeout"; Log "A/B #1 still alive after 4h"; exit 2 }
    Start-Sleep -Seconds 60
}

Log "waiting for >=6GB free RAM"
Write-State "waiting_ram"
$ramDeadline = (Get-Date).AddHours(1)
while ($true) {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeGB = [Math]::Round($os.FreePhysicalMemory / 1MB, 1)
    if ($freeGB -ge 6.0) { Log "free RAM ${freeGB}GB ok"; break }
    if ((Get-Date) -gt $ramDeadline) { Write-State "failed_wait_ram_timeout"; Log "free RAM only ${freeGB}GB after 1h"; exit 5 }
    Start-Sleep -Seconds 30
}

Log "180s settle buffer"
Start-Sleep -Seconds 180

$steps = @(
    @{ name = "parallel";         args = @("-u", "-m", "app.pipeline_parallel.runner"); timeout = 3600 },
    @{ name = "deliver_parallel"; args = @("-u", "scripts/_shortlist_t5_t10.py", "20260910"); timeout = 1800 },
    @{ name = "final_stocklist";  args = @("-u", "scripts/_final_stocklist.py", "20260910"); timeout = 900 },
    @{ name = "stocklist_combined"; args = @("-u", "scripts/_stocklist_combined.py", "20260910"); timeout = 900 }
)

# WORM: if a combined xlsx already exists and is older than our soon-to-be-written
# parallel shortlist, archive it so the combined step rebuilds with all pages.
$sld = "D:\AMINQT\DAILY OPERATION\STOCK LIST"
$xlsx = Join-Path $sld "stocklist_combined_20260910.xlsx"
$psl = Get-ChildItem -Path $sld -Filter "parallel_shortlist_20260910__*.csv" -ErrorAction SilentlyContinue | Select-Object -First 1
if ((Test-Path $xlsx) -and $psl -and ((Get-Item $xlsx).LastWriteTime -lt $psl.LastWriteTime)) {
    $arch = Join-Path $sld ("stocklist_combined_20260910__pre_parallel_v4_" + (Get-Date -Format "HHmmss") + ".xlsx")
    Move-Item -Path $xlsx -Destination $arch -Force
    Log ("archived stale combined (pre-parallel) -> " + (Split-Path -Leaf $arch))
}

foreach ($st in $steps) {
    Write-State ("running_" + $st.name)
    Log ("start " + $st.name)
    $p = Start-Process -FilePath $py -ArgumentList (("-X", "utf8") + $st.args) -WorkingDirectory $root -NoNewWindow -PassThru -RedirectStandardOutput "$root\tmp_t\_parallel_half_v4_0911.$($st.name).out" -RedirectStandardError "$root\tmp_t\_parallel_half_v4_0911.$($st.name).err"
    $exited = $p.WaitForExit($st.timeout * 1000)
    if (-not $exited) { Stop-Process -Id $p.Id -Force; Write-State ("failed_" + $st.name + "_timeout"); Log ($st.name + " TIMEOUT after " + $st.timeout + "s"); exit 3 }
    $null = $p.WaitForExit()
    $rc = $p.ExitCode
    Log ($st.name + " exited rc=" + $rc)
    if ($null -eq $rc -or $rc -ne 0) { Write-State ("failed_" + $st.name + "_rc" + $rc); Log ($st.name + " FAILED rc=" + $rc); exit 4 }
    Log ("ok " + $st.name)
}

Write-State "ok"
Log "all steps ok (no push)"
exit 0
