# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Switch Claude Code back to GLM-5.3 (Zhipu/BigModel).

.DESCRIPTION
    Updates BOTH user-level and project-level settings.json to use GLM-5.3 via Zhipu.
    Cleans up DeepSeek ANTHROPIC_AUTH_TOKEN to avoid conflicts.

    TRAPS AVOIDED:
    - Uses ANTHROPIC_API_KEY (not ANTHROPIC_AUTH_TOKEN) for Zhipu
    - Updates BOTH settings files
    - All models set to glm-5.3
    - Removes stale ANTHROPIC_AUTH_TOKEN to prevent auth conflict
#>

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────
$BASE_URL         = "https://open.bigmodel.cn/api/anthropic"
$MODEL            = "glm-5.3"

# ── Paths ──────────────────────────────────────────────────────
$UserSettings     = Join-Path $env:USERPROFILE ".claude\settings.json"
$ProjectRoot      = "d:\AMINQT\AMINQT CODES"
$ProjectSettings  = Join-Path $ProjectRoot ".claude\settings.json"

# ── 密钥从项目 .env 读取 (CLAUDE.md 约定) — 勿硬编码密钥进脚本 ──
#    (GitHub push protection 会拦截含密钥的提交)
$GLM_API_KEY = $null
$envLine = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^GLM_API_KEY=" } | Select-Object -First 1
if ($envLine) { $GLM_API_KEY = ($envLine -split "=", 2)[1].Trim() }
if ([string]::IsNullOrEmpty($GLM_API_KEY)) {
    Write-Host "[ERR] .env 未设置 GLM_API_KEY — 中止" -ForegroundColor Red
    exit 1
}

function Set-Settings {
    param(
        [string]$Path,
        [string]$Label
    )

    $dir = Split-Path $Path -Parent
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $json = if (Test-Path $Path) {
        Get-Content $Path -Raw | ConvertFrom-Json -AsHashtable
    } else {
        @{}
    }

    if (!$json.ContainsKey("env")) {
        $json["env"] = @{}
    }

    # ── Remove conflicting / stale keys ──
    $staleKeys = @(
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL"
    )
    foreach ($k in $staleKeys) {
        if ($json["env"].ContainsKey($k)) {
            $json["env"].Remove($k)
        }
    }

    # ── Set GLM config ──
    $json["env"]["ANTHROPIC_BASE_URL"]              = $BASE_URL
    $json["env"]["ANTHROPIC_API_KEY"]               = $GLM_API_KEY
    $json["env"]["ANTHROPIC_MODEL"]                 = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"]    = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]  = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"]   = $MODEL
    $json["env"]["CLAUDE_CODE_SUBAGENT_MODEL"]      = $MODEL
    $json["env"]["CLAUDE_CODE_EFFORT_LEVEL"]        = "max"
    $json["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "1000000"

    if ($json.ContainsKey("model")) {
        $json["model"] = $MODEL
    }

    $json | ConvertTo-Json -Depth 10 | Set-Content -Path $Path -Encoding UTF8
    Write-Host "[OK] $Label -> $Path" -ForegroundColor Green
}

# ── Execute ────────────────────────────────────────────────────
Set-Settings -Path $UserSettings -Label "User-level "
Set-Settings -Path $ProjectSettings -Label "Project-level"

Write-Host ""
Write-Host "Done. Restart Claude Code to apply." -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify with: claude --version && claude" -ForegroundColor Gray
