<#
.SYNOPSIS
  AMINQT 今日工作简报生成器
  Adapted from EWM Robot project append-today-brief.ps1

.DESCRIPTION
  Collects git diff + transcript changes, generates a Chinese daily brief.
  Output: D:\AMINQT\daily-briefs\aminqt-today-brief-YYYYMMDD.md
#>
param([string]$Date)

$R = "D:\AMINQT\AMINQT CODES"
if ($Date) {
    $targetDate = [datetime]::ParseExact($Date, "yyyy-MM-dd", $null)
} else {
    $targetDate = Get-Date
}
$DS = $targetDate.ToString("yyyyMMdd")
$DF = $targetDate.ToString("yyyy-MM-dd")
$TS = (Get-Date).ToString("HH:mm:ss")

$brDir = "D:\AMINQT\daily-briefs"
if (-not (Test-Path $brDir)) { New-Item -ItemType Directory -Path $brDir -Force | Out-Null }
$brFile = Join-Path $brDir ("aminqt-today-brief-$DS.md")

# --- Collect changed files from git ---
$FE = New-Object System.Collections.ArrayList
$seen = @{}

Push-Location $R
$gd = git diff --name-status HEAD 2>$null
if ($gd) {
    foreach ($ln in $gd) {
        if ($ln -match "^([AMDR])\s+(.+)") {
            $p = $Matches[2].Trim() -replace '\\', '/'
            if (-not $seen.ContainsKey($p)) {
                $seen[$p] = $true
                $code = $Matches[1]
                $action = switch ($code) {
                    "A" { "CREATE" }
                    "M" { "MODIFY" }
                    "D" { "DELETE" }
                    "R" { "REWRITE" }
                    default { "MODIFY" }
                }
                [void]$FE.Add([PSCustomObject]@{T=""; P=$p; A=$action})
            }
        }
    }
}
# Untracked files
$uf = git ls-files --others --exclude-standard 2>$null
if ($uf) {
    foreach ($u in ($uf -split "`n")) {
        $u = $u.Trim()
        if ($u -and -not $seen.ContainsKey($u)) {
            $seen[$u] = $true
            [void]$FE.Add([PSCustomObject]@{T=""; P=($u -replace '\\','/'); A="CREATE"})
        }
    }
}
# Committed on target date
$committedRaw = git log --since="$DF 00:00:00" --until="$DF 23:59:59" --name-status --pretty=format:"" 2>$null
if ($committedRaw) {
    foreach ($ln in ($committedRaw -split "`n")) {
        $ln = $ln.Trim()
        if ($ln -match "^([AMDR])\t(.+)") {
            $code = $Matches[1]
            $parts = $Matches[2] -split "`t"
            $p = ($parts[-1]).Trim() -replace '\\', '/'
            if (-not $seen.ContainsKey($p)) {
                $seen[$p] = $true
                $action = switch ($code) {
                    "A" { "CREATE" }
                    "M" { "MODIFY" }
                    "D" { "DELETE" }
                    "R" { "REWRITE" }
                    default { "MODIFY" }
                }
                [void]$FE.Add([PSCustomObject]@{T=""; P=$p; A=$action})
            }
        }
    }
}
Pop-Location

# Filter: skip deleted and trivial files
$sigFE = New-Object System.Collections.ArrayList
$trimmed = New-Object System.Collections.ArrayList
foreach ($fe in $FE) {
    $isTrivial = $fe.A -eq "DELETE" -or $fe.P -match '\.gitkeep|__pycache__|\.pyc'
    if (-not $isTrivial -and $fe.A -ne "DELETE") {
        $resolved = if ([System.IO.Path]::IsPathRooted($fe.P)) { $fe.P } else { Join-Path $R $fe.P }
        if (-not (Test-Path $resolved)) { continue }
        if (-not $fe.T) {
            $mtime = (Get-Item $resolved).LastWriteTime
            $todayEnd = $targetDate.Date.AddDays(1)
            if ($mtime -lt $targetDate.Date -or $mtime -ge $todayEnd) { continue }
        }
    }
    if ($isTrivial) { [void]$trimmed.Add($fe) } else { [void]$sigFE.Add($fe) }
}
$cnt = $sigFE.Count
$trivialCnt = $trimmed.Count

# --- No changes case ---
if ($cnt -eq 0) {
    $O = New-Object System.Collections.Generic.List[string]
    $O.Add("# AMINQT 今日工作简报")
    $O.Add("")
    $O.Add("> **日期：** $DF")
    $O.Add("> **项目：** A股图形因子量化交易系统")
    $O.Add("> **总文件数：** 0 个")
    $O.Add("")
    $O.Add("## 今日无文件变更")
    $O.Add("")
    $O.Add("---")
    $O.Add("")
    $O.Add("*文档生成：$DF $TS*")
    $nl = [System.Environment]::NewLine
    [System.IO.File]::WriteAllText($brFile, ($O -join $nl), [System.Text.Encoding]::UTF8)
    Write-Host "Generated: $brFile (no changes)"
    exit 0
}

# --- File description helper ---
function gDetail($p, $a) {
    $fullPath = if ([System.IO.Path]::IsPathRooted($p)) { $p } else { Join-Path $R $p }
    try {
        if (Test-Path $fullPath) {
            $ext = [System.IO.Path]::GetExtension($p).ToLower()
            if ($ext -eq '.py') {
                $lines = Get-Content $fullPath -Encoding UTF8 -TotalCount 80 -ErrorAction SilentlyContinue
                $inDoc = $false; $docParts = @()
                foreach ($ln in $lines) {
                    $t = $ln.Trim()
                    if ($t -match '^"""') {
                        if (-not $inDoc) { $inDoc = $true; $remain = $t -replace '^"""',''
                            if ($remain -and $remain -notmatch '^"""' -and $remain.Length -gt 0) { $docParts += $remain } }
                        else { break }
                    } elseif ($inDoc) {
                        if ($t -match '"""') { break }
                        $docParts += $t
                    }
                }
                if ($docParts.Count -gt 0) {
                    return "Python: " + (($docParts -join ' ') -replace '\s+',' ').Trim()
                }
                $members = @()
                foreach ($ln in $lines) {
                    if ($ln -match '^\s*(async\s+)?def\s+(\w+)\s*\(') { $members += "$($Matches[2])()" }
                    if ($ln -match '^\s*class\s+(\w+)') { $members += "class:$($Matches[1])" }
                }
                if ($members.Count -gt 0) { return "Python: $($members -join ', ')" }
            }
            if ($ext -eq '.md') {
                $lines = Get-Content $fullPath -Encoding UTF8 -TotalCount 10 -ErrorAction SilentlyContinue
                foreach ($ln in $lines) {
                    if ($ln -match '^#\s+(.+)$') { return "文档: $($Matches[1].Trim())" }
                }
            }
            if ($ext -eq '.json') { return "配置: $([System.IO.Path]::GetFileName($p))" }
            if ($ext -in '.yml','.yaml') { return "YAML: $([System.IO.Path]::GetFileName($p))" }
            if ($ext -in '.sh','.ps1','.bat') {
                $lines = Get-Content $fullPath -Encoding UTF8 -TotalCount 5 -ErrorAction SilentlyContinue
                $cmts = @()
                foreach ($ln in $lines) { if ($ln -match '^#\s*(.+)') { $cmts += $Matches[1].Trim() } }
                if ($cmts.Count -gt 0) { return "脚本: $($cmts -join ' ')" }
            }
        }
    } catch {}
    # Fallback: path-based
    $n = [System.IO.Path]::GetFileName($p)
    if ($p -match 'rules/') { return "AI规则: $n" }
    if ($p -match 'skills/') { return "AI技能: $n" }
    if ($p -match 'app/core/') { return "核心模块: $n" }
    if ($p -match 'app/models/') { return "模型: $n" }
    if ($p -match 'app/api/') { return "API路由: $n" }
    if ($p -match 'app/pattern/') { return "模式学习: $n" }
    if ($p -match 'app/rules/') { return "规则引擎: $n" }
    if ($p -match 'data/adapters/') { return "数据适配器: $n" }
    if ($p -match 'services/') { return "交易执行: $n" }
    if ($p -match 'scripts/') { return "脚本: $n" }
    if ($p -match 'tests/') { return "测试: $n" }
    if ($p -match '10_adr/') { return "ADR: $n" }
    if ($p -match '03_operations/') { return "运维手册: $n" }
    if ($p -match '02_deployment/') { return "部署清单: $n" }
    if ($p -match 'prompts/') { return "生成指令: $n" }
    return "$n"
}

function gBrief($p) {
    if ($p -match 'rules/') { return "AI规则" }
    if ($p -match 'skills/') { return "AI技能" }
    if ($p -match 'app/core/') { return "核心引擎" }
    if ($p -match 'app/models/') { return "模型" }
    if ($p -match 'app/api/') { return "API" }
    if ($p -match 'data/') { return "数据" }
    if ($p -match 'services/') { return "执行器" }
    if ($p -match 'scripts/') { return "脚本" }
    if ($p -match 'tests/') { return "测试" }
    if ($p -match '10_adr/') { return "ADR" }
    if ($p -match '03_operations/') { return "运维" }
    if ($p -match '02_deployment/') { return "部署" }
    if ($p -match 'prompts/') { return "Prompts" }
    if ($p -match 'AGENTS') { return "项目配置" }
    $e = [System.IO.Path]::GetExtension($p).ToLower()
    if ($e -eq '.md') { return "文档" }
    if ($e -eq '.py') { return "Python" }
    if ($e -in '.json','.yml','.yaml') { return "配置" }
    return "文件"
}

# --- Phase mapping ---
function MapPhase($path) {
    if ($path -match 'data/|^scripts/download') { return @{P="Phase 1"; L="数据底座"; S="1.x 数据下载"} }
    if ($path -match 'app/core/factor') { return @{P="Phase 2"; L="因子工程"; S="2.x 因子引擎"} }
    if ($path -match 'app/core/data_loader|data/adapters') { return @{P="Phase 1"; L="数据底座"; S="1.x 数据加载"} }
    if ($path -match 'app/models/|scripts/train') { return @{P="Phase 3"; L="模型训练"; S="3.x LSTM/LightGBM"} }
    if ($path -match 'app/api/|app/main|risk_filter|app/rules/') { return @{P="Phase 4"; L="Web服务+风控"; S="4.x API+风控"} }
    if ($path -match 'streamlit|plotly') { return @{P="Phase 5"; L="可视化面板"; S="5.x Streamlit"} }
    if ($path -match 'services/|executor|xtquant') { return @{P="Phase 6"; L="交易执行"; S="6.x 执行器"} }
    if ($path -match 'rules/|skills/|10_adr/|03_operations/|02_deployment/|prompts/|AGENTS') { return @{P="Phase 0"; L="项目治理"; S="0.x 规则与文档"} }
    if ($path -match 'tests/') { return @{P="持续"; L="测试"; S="测试维护"} }
    return @{P="持续"; L="其他"; S="其他变更"}
}

# --- Build phase groups ---
$phaseGroups = @{}
$planPhases = @()
foreach ($fe in $sigFE) {
    $pm = MapPhase $fe.P
    $key = "$($pm.P)|$($pm.L)|$($pm.S)"
    if (-not $phaseGroups.ContainsKey($key)) {
        $phaseGroups[$key] = New-Object System.Collections.ArrayList
        $planPhases += [PSCustomObject]@{P=$pm.P; L=$pm.L; S=$pm.S; K=$key}
    }
    $dt = gDetail $fe.P $fe.A
    $br = gBrief $fe.P
    [void]$phaseGroups[$key].Add([PSCustomObject]@{P=$fe.P; A=$fe.A; B=$br; D=$dt})
}

# Deduplicate phases
$seenK = @{}; $uniq = @()
foreach ($pp in $planPhases) { if (-not $seenK.ContainsKey($pp.K)) { $seenK[$pp.K]=$true; $uniq += $pp } }
$order = @{"Phase 0"=0; "Phase 1"=1; "Phase 2"=2; "Phase 3"=3; "Phase 4"=4; "Phase 5"=5; "Phase 6"=6; "持续"=99}
$uniq = $uniq | Sort-Object { $order[$_.P] }

# --- Build markdown ---
$O = New-Object System.Collections.Generic.List[string]

$O.Add("# AMINQT 今日工作简报")
$O.Add("")
$O.Add("> **日期：** $DF")
$O.Add("> **项目：** A股图形因子量化交易系统")
$O.Add("> **根目录：** ``$R``")
$O.Add("> **总文件数：** $($cnt + $trivialCnt) 个（新建 + 修改）")
$O.Add("")
$O.Add("---")
$O.Add("")

# Overview
$O.Add("## 今日工作总览")
$O.Add("")
$O.Add("| 功能领域 | 主要变更 | 文件数 |")
$O.Add("|----------|----------|--------|")

$areaGroups = @{}
foreach ($fe in $sigFE) {
    $br = gBrief $fe.P
    if (-not $areaGroups.ContainsKey($br)) { $areaGroups[$br] = New-Object System.Collections.ArrayList }
    [void]$areaGroups[$br].Add($fe)
}
foreach ($ak in ($areaGroups.Keys | Sort-Object)) {
    $ag = $areaGroups[$ak]
    $names = @()
    foreach ($ff in $ag) { $names += [System.IO.Path]::GetFileName($ff.P) }
    $fileStr = $names -join "、"
    if ($fileStr.Length -gt 60) { $fileStr = $fileStr.Substring(0,57) + "..." }
    $O.Add("| **$ak** | $fileStr | $($ag.Count) |")
}
if ($trivialCnt -gt 0) { $O.Add("| 清理维护 | 删除废弃文件 | $trivialCnt |") }
$O.Add("| **合计** | — | **$($cnt + $trivialCnt)** |")
$O.Add("")
$O.Add("---")
$O.Add("")

# Phase details
$curP = ""
foreach ($pp in $uniq) {
    $files = $phaseGroups[$pp.K]
    if ($pp.P -ne $curP) {
        $curP = $pp.P
        $O.Add("## $pp.P — $($pp.L)")
        $O.Add("")
    }
    $O.Add("### $($pp.S)")
    $O.Add("")
    $O.Add("| 文件路径 | 类型 | 说明 |")
    $O.Add("|----------|------|------|")
    foreach ($ff in $files) {
        $icon = switch ($ff.A) { "CREATE" { "🆕" } "MODIFY" { "✅" } "REWRITE" { "🔄" } default { "📄" } }
        $label = switch ($ff.A) { "CREATE" { "新建" } "MODIFY" { "修改" } "REWRITE" { "重写" } default { $ff.A } }
        $desc = $ff.D
        if ($desc.Length -gt 60) { $desc = $desc.Substring(0,57) + "..." }
        $O.Add("| ``$($ff.P)`` | $icon $label | $desc |")
    }
    $O.Add("")
}

# Key files
$O.Add("## 关键文件速查")
$O.Add("")
$O.Add("| 用途 | 路径 |")
$O.Add("|------|------|")
$O.Add("| AI 项目配置 | ``AGENTS.md`` |")
$O.Add("| 开发总计划 | ``../IMPLEMENTATION PLAN`` |")
$O.Add("| 架构文档 | ``../ARCHITECTURE`` |")
$O.Add("| 需求文档 | ``../PROMPT_CONTENT`` |")
$O.Add("| AI 规则目录 | ``rules/`` |")
$O.Add("| AI 技能目录 | ``skills/`` |")
$O.Add("| ADR 目录 | ``10_adr/`` |")
$O.Add("")

# Next actions
$O.Add("## 后续行动")
$O.Add("")
$naList = @()
$naList += "【日常】执行 git commit 提交变更并推送"
$naList += "【日常】运行 ruff check app/ scripts/ services/ tests/ 确认零错误"
$naList += "【日常】运行 pytest tests/ -v 确认全部通过"
$naList += "【Phase 1】验证 data/raw/ 下 CSV 文件完整且列名正确"
$naList += "【Phase 2】验证 factor_engine X.shape[2] >= 25 且无 NaN"
$naList += "【Phase 3】验证模型权重 lstm_best.pth 存在且 Loss < 0.01"
$naList += "【Phase 4】验证 select 接口响应 < 5s"
$naList += "【Phase 5】验证 Streamlit 面板可交互"
$naList += "【Phase 6】验证 SimExecutor T+1 检查和审计日志"
$i = 1; foreach ($item in $naList) { $O.Add("$i. $item"); $i++ }
$O.Add("")

$O.Add("---")
$O.Add("")
$O.Add("*文档生成：$DF $TS | 由 AMINQT 自动汇总*")
$O.Add("")

$nl = [System.Environment]::NewLine
[System.IO.File]::WriteAllText($brFile, ($O -join $nl), [System.Text.Encoding]::UTF8)

Write-Host "Generated: $brFile"
Write-Host "  Files: $cnt significant + $trivialCnt trivial = $($cnt + $trivialCnt) total"
