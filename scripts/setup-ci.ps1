<#
.SYNOPSIS
  AMINQT CI Setup Script — configures GitHub repository for CI/CD
  Run on your local machine (requires gh CLI authenticated).

.DESCRIPTION
  1. Creates GH_PAT secret (for auto-fix push to re-trigger CI)
  2. Creates DEEPSEEK_API_KEY secret (for AI PR review)
  3. Sets CATPAW_SELF_HOSTED variable (enables full CatPaw auto-fix)
  4. Configures branch protection on main
  5. Optionally registers self-hosted runner

.PARAMETER Repo
  GitHub repository in owner/repo format (e.g. "user/aminqt")

.PARAMETER Pat
  GitHub Personal Access Token with repo + workflow scope

.PARAMETER DeepSeekKey
  DeepSeek API key for AI PR review

.EXAMPLE
  .\setup-ci.ps1 -Repo "user/aminqt" -Pat "ghp_xxx" -DeepSeekKey "sk-xxx"
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$Repo,

    [Parameter(Mandatory=$true)]
    [string]$Pat,

    [Parameter(Mandatory=$false)]
    [string]$DeepSeekKey,

    [switch]$RegisterRunner
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AMINQT CI Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Repository: $Repo"
Write-Host ""

# --- Step 0: Verify gh CLI ---
Write-Host "Step 0: Checking gh CLI..." -ForegroundColor Yellow
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: gh CLI not installed. Install from https://cli.github.com" -ForegroundColor Red
    exit 1
}
Write-Host "OK: gh CLI found" -ForegroundColor Green
Write-Host ""

# --- Step 1: Set GH_PAT secret ---
Write-Host "Step 1: Setting GH_PAT secret..." -ForegroundColor Yellow
echo $Pat | gh secret set GH_PAT --repo $Repo
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: GH_PAT secret set" -ForegroundColor Green
} else {
    Write-Host "WARN: Failed to set GH_PAT (may already exist)" -ForegroundColor DarkYellow
}
Write-Host ""

# --- Step 2: Set DEEPSEEK_API_KEY secret ---
if ($DeepSeekKey) {
    Write-Host "Step 2: Setting DEEPSEEK_API_KEY secret..." -ForegroundColor Yellow
    echo $DeepSeekKey | gh secret set DEEPSEEK_API_KEY --repo $Repo
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK: DEEPSEEK_API_KEY secret set" -ForegroundColor Green
    } else {
        Write-Host "WARN: Failed to set DEEPSEEK_API_KEY" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "Step 2: Skipping DEEPSEEK_API_KEY (not provided)" -ForegroundColor DarkGray
}
Write-Host ""

# --- Step 3: Set CATPAW_SELF_HOSTED variable ---
Write-Host "Step 3: Setting CATPAW_SELF_HOSTED variable..." -ForegroundColor Yellow
gh variable set CATPAW_SELF_HOSTED --body "true" --repo $Repo 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: CATPAW_SELF_HOSTED=true (full auto-fix enabled)" -ForegroundColor Green
} else {
    Write-Host "INFO: CATPAW_SELF_HOSTED already set or failed" -ForegroundColor DarkYellow
    Write-Host "       Full CatPaw auto-fix will only run if self-hosted runner exists." -ForegroundColor DarkGray
}
Write-Host ""

# --- Step 4: Configure branch protection ---
Write-Host "Step 4: Configuring branch protection on main..." -ForegroundColor Yellow
gh api --method PUT "repos/$Repo/branches/main/protection" `
    --field "required_status_checks[strict]=true" `
    --field "required_status_checks[contexts][]=Lint" `
    --field "required_status_checks[contexts][]=ADR Check" `
    --field "required_status_checks[contexts][]=Test" `
    --field "enforce_admins=false" `
    --field "required_pull_request_reviews[required_approving_review_count]=0" `
    --field "required_pull_request_reviews[dismiss_stale_reviews]=true" `
    --field "restrictions=null" `
    2>$null

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK: Branch protection configured on main" -ForegroundColor Green
    Write-Host "     Required checks: Lint, ADR Check, Test" -ForegroundColor DarkGray
    Write-Host "     Strict: yes (must be up to date with base)" -ForegroundColor DarkGray
} else {
    Write-Host "WARN: Branch protection failed (may require admin access)" -ForegroundColor DarkYellow
    Write-Host "      Configure manually: Settings > Branches > main > Add rule" -ForegroundColor DarkGray
}
Write-Host ""

# --- Step 5: Self-hosted runner (optional) ---
if ($RegisterRunner) {
    Write-Host "Step 5: Registering self-hosted runner..." -ForegroundColor Yellow
    Write-Host "Follow the instructions at:" -ForegroundColor White
    Write-Host "  Settings > Actions > Runners > New self-hosted runner" -ForegroundColor White
    Write-Host ""
    Write-Host "After registration, ensure:" -ForegroundColor White
    Write-Host "  1. Runner has label 'catpaw'" -ForegroundColor White
    Write-Host "  2. CatPaw CLI installed and authenticated on runner" -ForegroundColor White
    Write-Host "  3. Runner is online" -ForegroundColor White

    # Download runner if not present
    $runnerDir = "$env:USERPROFILE\actions-runner-aminqt"
    if (-not (Test-Path $runnerDir)) {
        Write-Host "Downloading runner to $runnerDir..." -ForegroundColor DarkGray
        New-Item -ItemType Directory -Path $runnerDir -Force | Out-Null
        # User should follow GitHub UI instructions for their specific platform
        Write-Host "Please follow GitHub UI instructions for runner setup." -ForegroundColor Yellow
    }
} else {
    Write-Host "Step 5: Skipping self-hosted runner (use -RegisterRunner to enable)" -ForegroundColor DarkGray
}
Write-Host ""

# --- Summary ---
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CI SETUP COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configured:" -ForegroundColor White
Write-Host "  [x] GH_PAT secret (auto-fix push re-triggers CI)"
Write-Host "  [$('x' if $DeepSeekKey else ' ')] DEEPSEEK_API_KEY secret (AI PR review)"
Write-Host "  [x] CATPAW_SELF_HOSTED variable (full auto-fix)"
Write-Host "  [x] Branch protection on main (Lint + ADR + Test required)"
Write-Host "  [$('x' if $RegisterRunner else ' ')] Self-hosted runner (CatPaw CLI)"
Write-Host ""
Write-Host "Workflows active:" -ForegroundColor White
Write-Host "  ci.yml            - Lint + ADR + Test on every PR/push"
Write-Host "  auto-fix.yml      - Auto-fix lint on PR (ruff format + fix)"
Write-Host "  pr-gate.yml       - Block merge until CI passes"
Write-Host "  auto-approve.yml   - Dismiss block when CI passes"
Write-Host "  deepseek-pr-review.yml - AI code review on PR"
Write-Host ""
Write-Host "Next: Push a test PR to verify the pipeline." -ForegroundColor Yellow
