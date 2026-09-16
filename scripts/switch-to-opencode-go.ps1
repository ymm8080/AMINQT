# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Switch Claude Code backend to OpenCode Go (subscription plan) running glm-5.3-flash.

.DESCRIPTION
    Unlike Zen, OpenCode Go serves glm-5.3-flash ONLY over the OpenAI wire format
    (https://opencode.ai/zen/go/v1/chat/completions). Its Anthropic /messages path
    returns HTTP 500 for GLM, so a translating proxy is mandatory. This script points
    Claude Code at the local LiteLLM proxy started by start-opencode-go-proxy.ps1.

    Endpoint mapping:
      Claude Code -> http://127.0.0.1:8788/v1/messages   (Anthropic format)
      LiteLLM     -> https://opencode.ai/zen/go/v1/chat/completions  (OpenAI format)

    TRAPS AVOIDED:
    - Writes settings WITHOUT a UTF-8 BOM (PS 5.1's `Set-Content -Encoding UTF8` adds
      one; a BOM breaks JSON readers, notably cc-switch: "expected value at line 1
      column 1"). Uses .NET WriteAllText with UTF8Encoding($false).
    - Updates BOTH settings files (user env block + project modelOverrides).
    - Removes cc-switch leftovers: ANTHROPIC_*_MODEL_NAME and the PROXY_MANAGED token.
    - The proxy needs no auth (localhost only); a placeholder token is set because
      Claude Code refuses to start without one.

.PARAMETER MODEL
    Model name as declared in litellm-opencode-go.yaml. Default glm-5.3-flash.

.PARAMETER Port
    Proxy port. Must match the port used by start-opencode-go-proxy.ps1. Default 8788.
#>

param(
    [string]$MODEL = "glm-5.3-flash",
    [int]$Port = 8788
)

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────
$BASE_URL = "http://127.0.0.1:$Port"
$PROXY_TOKEN = "sk-litellm-local"   # proxy is unauthenticated localhost; placeholder only

# ── Paths ──────────────────────────────────────────────────────
$UserSettings    = Join-Path $env:USERPROFILE ".claude\settings.json"
$ProjectRoot     = Split-Path $PSScriptRoot -Parent
# 项目级写 settings.local.json (gitignored 实际生效文件), 勿写 settings.json (共享/已提交)
$ProjectSettings = Join-Path $ProjectRoot ".claude\settings.local.json"

# ── PS 5.1 兼容: 5.1 无 ConvertFrom-Json -AsHashtable, 深转 PSCustomObject → Hashtable ──
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
    if ([string]::IsNullOrWhiteSpace($raw)) { return @{} }
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

    # ── Remove conflicting / stale keys (DeepSeek / GLM / cc-switch 遗留) ──
    $staleKeys = @(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
        "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL"
    )
    foreach ($k in $staleKeys) {
        if ($json["env"].ContainsKey($k)) {
            $json["env"].Remove($k)
        }
    }

    # ── Set proxy config ──
    $json["env"]["ANTHROPIC_BASE_URL"]              = $BASE_URL
    $json["env"]["ANTHROPIC_AUTH_TOKEN"]            = $PROXY_TOKEN
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

    # ── modelOverrides: claude-* 显式模型引用档位 (记忆: 切档改两处勿漏) ──
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

    # ── BOM-free UTF-8 write (PS 5.1 Set-Content -Encoding UTF8 emits a BOM) ──
    $out = $json | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($Path, $out, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "[OK] $Label -> $Path" -ForegroundColor Green
}

# ── Sync Windows User-Registry Env (priority > settings.json) ──
function Set-UserEnvRegistry {
    $registryVars = [ordered]@{
        "ANTHROPIC_BASE_URL"              = $BASE_URL
        "ANTHROPIC_AUTH_TOKEN"            = $PROXY_TOKEN
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
            Write-Host "[REG] $($kv.Key) = $($kv.Value)" -ForegroundColor DarkYellow
        }
    }

    $legacyKeys = @(
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
        "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL"
    )
    foreach ($legacyKey in $legacyKeys) {
        $existing = [Environment]::GetEnvironmentVariable($legacyKey, "User")
        if ($existing) {
            [Environment]::SetEnvironmentVariable($legacyKey, $null, "User")
            Write-Host "[DEL-REG] $legacyKey (removed stale conflict)" -ForegroundColor DarkGray
        }
    }
}

# ── Ensure the proxy is up ─────────────────────────────────────
$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (!$listening) {
    Write-Host "[..] proxy not listening on $Port — starting it" -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "start-opencode-go-proxy.ps1") -Port $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERR] proxy failed to start — aborting" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[--] proxy already listening on $Port" -ForegroundColor DarkGray
}

# ── Execute ────────────────────────────────────────────────────
Set-Settings -Path $UserSettings -Label "User-level " -ApplyOverrides
Set-Settings -Path $ProjectSettings -Label "Project-level" -ApplyOverrides
Set-UserEnvRegistry

Write-Host ""
Write-Host "Done. Restart Claude Code to apply." -ForegroundColor Cyan
Write-Host "Model: $MODEL via OpenCode Go (proxy at $BASE_URL)" -ForegroundColor Gray
