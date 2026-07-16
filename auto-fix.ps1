# ============================================================================
# auto-fix.sh — CatPaw Auto-Fix Loop (AMINQT)
# Prompt-script for CatPaw/Claude. Usage: bash auto-fix.sh <PR_NUMBER>
# ============================================================================
$PROJECT_ROOT = "D:\AMINQT\AMINQT CODES"
$MAX_LOOP = 6
$MAX_SESSIONS = 3
$PR_NUMBER = $args[0]

Write-Host "================================================================"
Write-Host "  auto-fix.sh v2.0 -- session-rotate (AMINQT)"
Write-Host "================================================================"
Write-Host "  Loop/session: $MAX_LOOP | Max sessions: $MAX_SESSIONS"
Write-Host "  PR: $PR_NUMBER"
Write-Host ""

if (-not $PR_NUMBER) {
    Write-Host "Run: gh pr list --state open"
    Write-Host "Then: bash auto-fix.sh <PR_NUMBER>"
    exit 1
}

Write-Host "  STEP 1: CHECK PR MERGEABLE"
Write-Host "    cd $PROJECT_ROOT"
Write-Host "    gh pr view $PR_NUMBER --json mergeable,mergeStateStatus,headRefName,baseRefName"
Write-Host "    mergeable==true AND status in [CLEAN, MERGEABLE] -> MERGE -> EXIT"
Write-Host ""
Write-Host "  STEP 2: CHECK RETRY COUNT"
Write-Host "    git log -1 --pretty=%B"
Write-Host "    Extract [auto-fix-loop N/$MAX_LOOP] [auto-fix-session S/$MAX_SESSIONS]"
Write-Host "    NEXT > $MAX_LOOP -> CLOSE session, OPEN new (S+1, N=0)"
Write-Host "    NEXT_SESSION > $MAX_SESSIONS -> EXIT (needs human)"
Write-Host ""
Write-Host "  STEP 3: CHECKOUT PR BRANCH"
Write-Host "    git fetch origin --prune"
Write-Host "    git checkout -B <headRefName> origin/<headRefName>"
Write-Host "    git merge origin/<baseRefName> --no-edit 2>`$null || true"
Write-Host ""
Write-Host "  STEP 4: COLLECT ISSUES"
Write-Host "    gh pr view $PR_NUMBER --json comments  # AI review TODO"
Write-Host "    gh pr view $PR_NUMBER --json statusCheckRollup  # CI failures"
Write-Host "    Both empty -> Sleep 45s, loop to STEP 1"
Write-Host ""
Write-Host "  STEP 5: READ AFFECTED FILES"
Write-Host "    Read each file, check rules/ for constraints"
Write-Host ""
Write-Host "  STEP 6: FIX CODE (string_replace/MultiEdit, preserve style)"
Write-Host ""
Write-Host "  STEP 7: VERIFY"
Write-Host "    cd $PROJECT_ROOT"
Write-Host "    ruff check app/ scripts/ services/ tests/"
Write-Host "    python -m pytest tests/ -q --tb=short --no-header 2>&1 | Select-Object -Last 30"
Write-Host ""
Write-Host "  STEP 8: COMMIT + PUSH"
Write-Host "    git config user.name 'CatPaw Auto-Fix'"
Write-Host "    git config user.email 'auto-fix@catpaw.local'"
Write-Host "    git add -A; git commit -m 'fix: auto-fix-loop NEXT/$MAX_LOOP session S/$MAX_SESSIONS'"
Write-Host "    git push origin <headRefName>"
Write-Host ""
Write-Host "  STEP 9: WAIT + LOOP (Sleep 45s for CI)"
Write-Host ""
Write-Host "  EXIT: 1.MERGEABLE 2.SESSIONS_EXHAUSTED 3.MERGE_CONFLICT"
