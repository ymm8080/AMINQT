# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Switch Claude Code backend to OpenCode Zen (opencode.ai).

.DESCRIPTION
    Updates BOTH user-level and project-level settings.json + Windows User registry
    to route through OpenCode Zen's Anthropic-compatible endpoint.

    ENDPOINT MAPPING (docs 2026-09: https://opencode.ai/docs/zen/):
    - Zen full messages path = https://opencode.ai/zen/v1/messages
    - Claude Code appends /v1/messages itself, so ANTHROPIC_BASE_URL = https://opencode.ai/zen
    - Only /messages (Anthropic-compat) models can drive Claude Code:
        Claude family (claude-sonnet-4-5/4-6/5, claude-opus-4-5..5, claude-fable-5, claude-haiku-4-5)
        + Qwen (qwen3.5-plus, qwen3.6-plus, qwen3.7-plus, qwen3.7-max)
    - GLM / DeepSeek / Kimi / free models on Zen are OpenAI-compat (/chat/completions)
      -> NOT usable as ANTHROPIC_BASE_URL for Claude Code.

    TRAPS AVOIDED (mirror of switch-to-glm.ps1):
    - Key read from project .env OPENCODE_API_KEY, never hardcoded
    - Updates BOTH settings files (user env block + project modelOverrides)
    - Removes stale ANTHROPIC_AUTH_TOKEN / OPENAI_* to prevent auth conflict
    - PS 5.1 compatible (no ConvertFrom-Json -AsHashtable)

    IF 401 AFTER SWITCH: some gateways want Authorization Bearer instead of x-api-key.
    Fallback: rename ANTHROPIC_API_KEY -> ANTHROPIC_AUTH_TOKEN in the env blocks below.

    IF 404 AFTER SWITCH: try BASE_URL = "https://opencode.ai/zen/v1" instead.

.PARAMETER MODEL
    Zen model ID. Default claude-sonnet-4-5 ($3/$15 per 1M; >200K ctx bills $6/$22.50).
    Cheap alternatives: qwen3.7-plus ($0.40/$1.60), claude-haiku-4-5 ($1/$5).
#>

param(
    [string]$MODEL = "claude-sonnet-4-5"
)

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────
$BASE_URL = "https://opencode.ai/zen"

# ── Paths ──────────────────────────────────────────────────────
$UserSettings    = Join-Path $env:USERPROFILE ".claude\settings.json"
$ProjectRoot     = "d:\AMINQT\AMINQT CODES"
# 项目级写 settings.local.json (gitignored 实际生效文件), 勿写 settings.json (共享/已提交)
$ProjectSettings = Join-Path $ProjectRoot ".claude\settings.local.json"

# ── 密钥从项目 .env 读取 — 勿硬编码密钥进脚本 ──
$ZEN_API_KEY = $null
$envLine = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^OPENCODE_API_KEY=" } | Select-Object -First 1
if ($envLine) { $ZEN_API_KEY = ($envLine -split "=", 2)[1].Trim() }
if ([string]::IsNullOrEmpty($ZEN_API_KEY)) {
    Write-Host "[ERR] .env 未设置 OPENCODE_API_KEY — 到 opencode.ai Zen 登录后取 key, 加一行 OPENCODE_API_KEY=... 到项目 .env" -ForegroundColor Red
    exit 1
}

# ── PS 5.1 兼容: 深转 PSCustomObject → Hashtable ──
function ConvertTo-HashtableDeep {
    param($InputObject)
    if ($InputObject -is [System.Management.Automation.PSCustomObject]) {
        $result = @{}
        $InputObject.PSObject.Properties | ForEach-Object { $result[$_.Name] = ConvertTo-HashtableDeep $_.Value }
        return $result
    } elseif ($InputObject -is [System.Collections.IEnumerable] -and $InputObject -isnot [string]) {
        return @($InputObject | ForEach-Object { ConvertTo-HashtableDeep $_ })
    } else {
        return $InputObject
    }
}

function Read-SettingsJson {
    param([string]$Path)
    if (!(Test-Path $Path)) { return @{} }
    $raw = Get-Content $Path -Raw
    if ($PSVersionTable.PSVersion.Major -ge 7) {
        return ($raw | ConvertFrom-Json -AsHashtable)
    }
    return (ConvertTo-HashtableDeep ($raw | ConvertFrom-Json))
}

function Set-Settings {
    param(
        [string]$Path,
        [string]$Label,
        [switch]$ApplyOverrides
    )

    $dir = Split-Path $Path -Parent
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    $json = Read-SettingsJson -Path $Path

    if (!$json.ContainsKey("env")) {
        $json["env"] = @{}
    }

    # ── Remove conflicting / stale keys (GLM/DeepSeek 遗留) ──
    $staleKeys = @(
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL"
    )
    foreach ($k in $staleKeys) {
        if ($json["env"].ContainsKey($k)) {
            $json["env"].Remove($k)
        }
    }

    # ── Set Zen config ──
    $json["env"]["ANTHROPIC_BASE_URL"]              = $BASE_URL
    $json["env"]["ANTHROPIC_API_KEY"]               = $ZEN_API_KEY
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

    # ── modelOverrides: claude-* 显式模型引用档位 (切档改两处勿漏) ──
    if ($ApplyOverrides) {
        if (!$json.ContainsKey("modelOverrides")) {
            $json["modelOverrides"] = @{}
        }
        $json["modelOverrides"]["claude-opus-4-8"]             = $MODEL
        $json["modelOverrides"]["claude-sonnet-4-6"]           = $MODEL
        $json["modelOverrides"]["claude-haiku-4-5-20251001"]   = $MODEL
        $json["modelOverrides"]["claude-sonnet-4-5-20250929"]  = $MODEL
        $json["modelOverrides"]["claude-opus-4-20250514"]      = $MODEL
    }

    $json | ConvertTo-Json -Depth 10 | Set-Content -Path $Path -Encoding UTF8
    Write-Host "[OK] $Label -> $Path" -ForegroundColor Green
}

# ── Sync Windows User-Registry Env (priority > settings.json) ──
function Set-UserEnvRegistry {
    $registryVars = [ordered]@{
        "ANTHROPIC_BASE_URL"              = $BASE_URL
        "ANTHROPIC_API_KEY"               = $ZEN_API_KEY
        "ANTHROPIC_MODEL"                 = $MODEL
        "ANTHROPIC_DEFAULT_OPUS_MODEL"    = $MODEL
        "ANTHROPIC_DEFAULT_SONNET_MODEL"  = $MODEL
        "ANTHROPIC_DEFAULT_HAIKU_MODEL"   = $MODEL
        "CLAUDE_CODE_SUBAGENT_MODEL"      = $MODEL
        "CLAUDE_CODE_EFFORT_LEVEL"        = "max"
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW" = "1000000"
    }

    foreach ($kv in $registryVars.GetEnumerator()) {
        $existing = [Environment]::GetEnvironmentVariable($kv.Key, "User")
        if ($existing -ne $kv.Value) {
            [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, "User")
            $short = if ($kv.Value.Length -gt 8) { $kv.Value.Substring(0, 8) } else { $kv.Value }
            Write-Host "[REG] $($kv.Key) = $short..." -ForegroundColor DarkYellow
        }
    }

    $legacyKeys = @("ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
    foreach ($legacyKey in $legacyKeys) {
        $existing = [Environment]::GetEnvironmentVariable($legacyKey, "User")
        if ($existing) {
            [Environment]::SetEnvironmentVariable($legacyKey, $null, "User")
            Write-Host "[DEL-REG] $legacyKey (removed stale conflict)" -ForegroundColor DarkGray
        }
    }
}

# ── Execute ────────────────────────────────────────────────────
Set-Settings -Path $UserSettings -Label "User-level " -ApplyOverrides
Set-Settings -Path $ProjectSettings -Label "Project-level" -ApplyOverrides
Set-UserEnvRegistry

Write-Host ""
Write-Host "Done. Restart Claude Code to apply." -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify first call: curl with x-api-key header against $BASE_URL/v1/messages" -ForegroundColor Gray
Write-Host "If 401 -> use ANTHROPIC_AUTH_TOKEN instead of ANTHROPIC_API_KEY (see header notes)"
Write-Host "If 404 -> change BASE_URL to '$BASE_URL/v1' (see header notes)"
