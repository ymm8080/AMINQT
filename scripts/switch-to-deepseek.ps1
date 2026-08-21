# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Switch Claude Code to DeepSeek V4 Flash — all modules.

.DESCRIPTION
    Updates BOTH user-level and project-level settings.json to use DeepSeek V4 Flash.
    Cleans up old ANTHROPIC_API_KEY / GLM configs to avoid auth conflicts.

    TRAPS AVOIDED:
    - Uses ANTHROPIC_AUTH_TOKEN (not ANTHROPIC_API_KEY) per DeepSeek docs
    - Updates BOTH C:\Users\<user>\.claude\settings.json AND project .claude\settings.json
    - All models set to deepseek-v4-flash (not pro, not mixed)
    - Removes stale ANTHROPIC_API_KEY to prevent auth conflict
#>

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────
$BASE_URL            = "https://api.deepseek.com/anthropic"
$MODEL               = "deepseek-v4-flash"

# ── Paths ──────────────────────────────────────────────────────
$UserSettings  = Join-Path $env:USERPROFILE ".claude\settings.json"
$ProjectRoot   = "d:\AMINQT\AMINQT CODES"
$ProjectSettings = Join-Path $ProjectRoot ".claude\settings.json"

# ── 密钥从项目 .env 读取 (CLAUDE.md 约定) — 勿硬编码密钥进脚本 ──
#    (GitHub push protection 会拦截含密钥的提交)
$DEEPSEEK_API_KEY = $null
$envLine = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^DEEPSEEK_API_KEY=" } | Select-Object -First 1
if ($envLine) { $DEEPSEEK_API_KEY = ($envLine -split "=", 2)[1].Trim() }
if ([string]::IsNullOrEmpty($DEEPSEEK_API_KEY)) {
    Write-Host "[ERR] .env 未设置 DEEPSEEK_API_KEY — 中止" -ForegroundColor Red
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

    # Load existing or start fresh
    $json = if (Test-Path $Path) {
        Get-Content $Path -Raw | ConvertFrom-Json -AsHashtable
    } else {
        @{}
    }

    # Ensure env block exists
    if (!$json.ContainsKey("env")) {
        $json["env"] = @{}
    }

    # ── Remove conflicting / stale keys ──
    $staleKeys = @(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
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

    # ── Set DeepSeek config ──
    $json["env"]["ANTHROPIC_BASE_URL"]              = $BASE_URL
    $json["env"]["ANTHROPIC_AUTH_TOKEN"]            = $DEEPSEEK_API_KEY
    $json["env"]["ANTHROPIC_MODEL"]                 = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"]    = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]  = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"]   = $MODEL
    $json["env"]["CLAUDE_CODE_SUBAGENT_MODEL"]      = $MODEL
    $json["env"]["CLAUDE_CODE_EFFORT_LEVEL"]        = "max"
    $json["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "786432"

    # Set top-level model if exists
    if ($json.ContainsKey("model")) {
        $json["model"] = $MODEL
    }

    # ── Write back ──
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
