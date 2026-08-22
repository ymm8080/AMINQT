# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Switch Claude Code to DeepSeek V4 Flash — all modules.

.DESCRIPTION
    Updates BOTH user-level and project-level settings.json to use DeepSeek V4 Flash.
    Cleans up old ANTHROPIC_API_KEY / GLM configs to avoid auth conflicts.
    Also syncs Windows user-registry env vars (higher priority than settings.json).

    FIXES (v2):
    - PS 5.1 safe: uses ConvertFrom-Json -AsHashtable (PS 6+) with explicit PSCustomObject fallback
    - Removed stale $k stray references from foreach scopes
    - Verifies .env DEEPSEEK_API_KEY before edit (no hardcode)
    - Logs each stale-key removal (audit trace)
#>

$ErrorActionPreference = "Stop"

# ── Configuration ─────────────────────────────────────────────────────────────
$BASE_URL = "https://api.deepseek.com/anthropic"
$MODEL    = "deepseek-v4-flash"

# ── Paths ─────────────────────────────────────────────────────────────────────
$UserSettings   = Join-Path $env:USERPROFILE     ".claude\settings.json"
$ProjectRoot    = "d:\AMINQT\AMINQT CODES"
$ProjectSettings = Join-Path $ProjectRoot         ".claude\settings.local.json"

# ── Read DEEPSEEK_API_KEY from project .env (no hardcoding) ───────────────────
$DEEPSEEK_API_KEY = $null
$envLine = Get-Content (Join-Path $ProjectRoot ".env") -ErrorAction SilentlyContinue |
    Where-Object { $_ -match "^DEEPSEEK_API_KEY=" } | Select-Object -First 1
if ($envLine) { $DEEPSEEK_API_KEY = ($envLine -split "=", 2)[1].Trim() }
if ([string]::IsNullOrEmpty($DEEPSEEK_API_KEY)) {
    Write-Host "[ERR] .env missing DEEPSEEK_API_KEY — abort" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] DEEPSEEK_API_KEY loaded from .env ($($DEEPSEEK_API_KEY.Substring(0,8))...)" -ForegroundColor Green

# ── Recursive PSCustomObject → Hashtable converter (PS 5.1 safe) ──────────────
function ConvertTo-HashtableDeep {
    param($InputObject)
    if ($null -eq $InputObject) { return $null }
    if ($InputObject -is [System.Management.Automation.PSCustomObject]) {
        $ht = @{}
        foreach ($prop in $InputObject.PSObject.Properties) {
            $ht[$prop.Name] = ConvertTo-HashtableDeep $prop.Value
        }
        return $ht
    }
    if ($InputObject -is [System.Collections.IEnumerable] -and $InputObject -isnot [string]) {
        return @($InputObject | ForEach-Object { ConvertTo-HashtableDeep $_ })
    }
    return $InputObject
}

# ── Read JSON as hashtable (PS 5.1+ safe) ─────────────────────────────────────
function Read-SettingsJson {
    param([string]$Path)
    if (!(Test-Path $Path)) { return @{} }
    $raw = Get-Content $Path -Raw -ErrorAction SilentlyContinue
    if ([string]::IsNullOrWhiteSpace($raw)) { return @{} }
    if ($PSVersionTable.PSVersion.Major -ge 6) {
        return ($raw | ConvertFrom-Json -AsHashtable)
    }
    # PS 5.1 fallback
    return (ConvertTo-HashtableDeep ($raw | ConvertFrom-Json))
}

# ── Write settings file ───────────────────────────────────────────────────────
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

    # Ensure env block
    if (!$json.ContainsKey("env")) {
        $json["env"] = @{}
    }

    # ── Remove stale keys (with logging) ───────────────────────────────────────
    $staleKeys = @(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL"
    )
    foreach ($staleKey in $staleKeys) {
        if ($json["env"].ContainsKey($staleKey)) {
            $json["env"].Remove($staleKey)
            Write-Host "[DEL] $staleKey" -ForegroundColor DarkGray
        }
    }

    # ── Set DeepSeek env vars ──────────────────────────────────────────────────
    $json["env"]["ANTHROPIC_BASE_URL"]              = $BASE_URL
    $json["env"]["ANTHROPIC_AUTH_TOKEN"]            = $DEEPSEEK_API_KEY
    $json["env"]["ANTHROPIC_MODEL"]                 = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"]    = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"]  = $MODEL
    $json["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"]   = $MODEL
    $json["env"]["CLAUDE_CODE_SUBAGENT_MODEL"]      = $MODEL
    $json["env"]["CLAUDE_CODE_EFFORT_LEVEL"]        = "max"
    $json["env"]["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = "786432"

    # Top-level model field
    if ($json.ContainsKey("model")) {
        $json["model"] = $MODEL
    }

    # modelOverrides for explicit Claude model references
    if ($ApplyOverrides) {
        if (!$json.ContainsKey("modelOverrides")) {
            $json["modelOverrides"] = @{}
        }
        $json["modelOverrides"]["claude-opus-4-8"]           = $MODEL
        $json["modelOverrides"]["claude-sonnet-4-6"]         = $MODEL
        $json["modelOverrides"]["claude-haiku-4-5-20251001"] = $MODEL
    }

    # Write back
    $json | ConvertTo-Json -Depth 10 | Set-Content -Path $Path -Encoding UTF8
    Write-Host "[OK] $Label -> $Path" -ForegroundColor Green
}

# ── Sync Windows User-Registry Env (priority > settings.json) ────────────────
function Set-UserEnvRegistry {
    $registryVars = [ordered]@{
        "ANTHROPIC_BASE_URL"              = $BASE_URL
        "ANTHROPIC_AUTH_TOKEN"            = $DEEPSEEK_API_KEY
        "ANTHROPIC_MODEL"                 = $MODEL
        "ANTHROPIC_DEFAULT_OPUS_MODEL"    = $MODEL
        "ANTHROPIC_DEFAULT_SONNET_MODEL"  = $MODEL
        "ANTHROPIC_DEFAULT_HAIKU_MODEL"   = $MODEL
        "CLAUDE_CODE_SUBAGENT_MODEL"      = $MODEL
        "CLAUDE_CODE_EFFORT_LEVEL"        = "max"
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW" = "786432"
    }

    foreach ($kv in $registryVars.GetEnumerator()) {
        $existing = [Environment]::GetEnvironmentVariable($kv.Key, "User")
        if ($existing -ne $kv.Value) {
            [Environment]::SetEnvironmentVariable($kv.Key, $kv.Value, "User")
            Write-Host "[REG] $($kv.Key) = $($kv.Value)" -ForegroundColor DarkYellow
        }
    }

    # Remove conflicting legacy keys from registry
    $legacyKeys = @(
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL"
    )
    foreach ($legacyKey in $legacyKeys) {
        $existing = [Environment]::GetEnvironmentVariable($legacyKey, "User")
        if ($existing) {
            [Environment]::SetEnvironmentVariable($legacyKey, $null, "User")
            Write-Host "[DEL-REG] $legacyKey (was: $existing)" -ForegroundColor DarkGray
        }
    }
}

# ── Execute ───────────────────────────────────────────────────────────────────
Set-Settings -Path $UserSettings    -Label "User-level   "
Set-Settings -Path $ProjectSettings -Label "Project-level" -ApplyOverrides
Set-UserEnvRegistry

Write-Host ""
Write-Host "Done. Restart Claude Code to apply." -ForegroundColor Cyan
Write-Host ""
Write-Host "Verify with: claude --version" -ForegroundColor Gray
