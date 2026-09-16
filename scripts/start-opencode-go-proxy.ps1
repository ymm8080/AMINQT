# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Start the LiteLLM transfer proxy: Claude Code (Anthropic) -> OpenCode Go (OpenAI) -> glm-5.3-flash.

.DESCRIPTION
    Claude Code speaks Anthropic /v1/messages. OpenCode Go serves glm-5.3-flash only on
    /v1/chat/completions (its /messages path 500s for GLM). This proxy bridges the two.

    Key is read from the project .env (OPENCODE_API_KEY) — never hardcoded.

    TRAPS AVOIDED:
    - Does NOT use Start-Process -RedirectStandardOutput (kills python children on PS 5.1);
      redirects via cmd.exe instead.
    - LITELLM_LOCAL_MODEL_COST_MAP stops the proxy stalling on a remote cost-map fetch.

.PARAMETER Port
    Listen port. Default 8788.

.PARAMETER Stop
    Stop a running proxy instead of starting one.
#>

param(
    [int]$Port = 8788,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$ConfigPath  = Join-Path $PSScriptRoot "litellm-opencode-go.yaml"
$LitellmExe  = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\Scripts\litellm.exe"
$LogDir      = Join-Path $ProjectRoot "tmp_t"
$LogPath     = Join-Path $LogDir "litellm_opencode_go_$Port.log"

# ── Locate whatever is listening on $Port ──
function Get-ProxyProc {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conn) { return Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue }
    return $null
}

if ($Stop) {
    $p = Get-ProxyProc
    if ($p) { Stop-Process -Id $p.Id -Force; Write-Host "[OK] stopped pid $($p.Id) on port $Port" -ForegroundColor Green }
    else    { Write-Host "[--] nothing listening on $Port" -ForegroundColor DarkGray }
    exit 0
}

if (Get-ProxyProc) {
    Write-Host "[--] proxy already listening on $Port" -ForegroundColor DarkGray
    exit 0
}

# ── Key from .env ──
$envLine = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^OPENCODE_API_KEY=" } | Select-Object -First 1
if (!$envLine) {
    Write-Host "[ERR] .env 未设置 OPENCODE_API_KEY — 中止" -ForegroundColor Red
    exit 1
}
$env:OPENCODE_API_KEY = ($envLine -split "=", 2)[1].Trim()

if (!(Test-Path $LitellmExe)) {
    Write-Host "[ERR] 找不到 litellm.exe: $LitellmExe" -ForegroundColor Red
    exit 1
}
if (!(Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"

# cmd.exe /c with a spaced path mangles quoting, so stage a .bat and launch that.
# OPENCODE_API_KEY is NOT written into the .bat — it is inherited from this process env.
$batPath = Join-Path $LogDir "_litellm_run_$Port.bat"
$bat = @"
@echo off
cd /d "$ProjectRoot"
set LITELLM_LOCAL_MODEL_COST_MAP=True
"$LitellmExe" --config "$ConfigPath" --host 127.0.0.1 --port $Port > "$LogPath" 2>&1
"@
Set-Content -Path $batPath -Value $bat -Encoding ASCII
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$batPath`"" -WindowStyle Hidden

Write-Host "[..] starting proxy on 127.0.0.1:$Port, log: $LogPath" -ForegroundColor Cyan
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    if (Get-ProxyProc) {
        Write-Host "[OK] proxy listening on 127.0.0.1:$Port" -ForegroundColor Green
        exit 0
    }
}
Write-Host "[ERR] proxy failed to bind $Port in 20s — check $LogPath" -ForegroundColor Red
exit 1
