#!/usr/bin/env bash
# ============================================================================
# auto-implement-and-pr.sh â€” Implement an IMPLEMENTATION PLAN phase â†’ single PR
#
# Adapted from EWM Robot project for AMINQT A-Share Quant Platform.
# Prompt-script: echoes instructions for the CatPaw/Claude agent.
# Run either way:
#   bash auto-implement-and-pr.sh [phase]
#   cat auto-implement-and-pr.sh | claude
#
# NOTE: This is the IMPLEMENTATION stage ONLY. auto-fix.sh runs AFTER the PR
# is created and CI/AI-review have run â€” do not invoke it during implementation.
# ============================================================================

PROJECT_ROOT="D:/AMINQT/AMINQT CODES"
PLAN_FILE="D:/AMINQT/IMPLEMENTATION PLAN"
PHASE_HINT="${1:-}"

echo "================================================================"
echo "  auto-implement-and-pr.sh â€” implement plan phase â†’ PR (no merge)"
echo "================================================================"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Plan file:    ${PLAN_FILE}"
[ -n "$PHASE_HINT" ] && echo "  Phase hint:   ${PHASE_HINT}"
echo ""

# ============================================================================
# STEP 0: READ THE PLAN
# ============================================================================

echo "================================================================"
echo "  STEP 0: READ THE IMPLEMENTATION PLAN (authoritative)"
echo "================================================================"
echo "  ${PLAN_FILE}"
echo "  Read this file IN FULL before touching any source."
echo "  Also read: ARCHITECTURE, PROMPT_CONTENT, AGENTS.md"
echo ""

# ============================================================================
# IMPLEMENT-AND-PR INSTRUCTIONS FOR THE AGENT
# ============================================================================

echo "================================================================"
echo "  IMPLEMENT-AND-PR INSTRUCTIONS"
echo "================================================================"
echo ""
echo "  You are running auto-implement-and-pr: implement ONE phase of the"
echo "  IMPLEMENTATION PLAN into a single PR. Do NOT merge."
echo "  Project root: ${PROJECT_ROOT}"
echo ""

# -- STEP 1: READ THE PLAN ----------------------------------------------------

echo "  STEP 1: READ THE PLAN + PROJECT CONTEXT"
echo "    Read IN FULL:"
echo "      - \"${PLAN_FILE}\""
echo "      - D:/AMINQT/ARCHITECTURE"
echo "      - D:/AMINQT/PROMPT_CONTENT"
echo "      - ${PROJECT_ROOT}/AGENTS.md"
echo "      - ${PROJECT_ROOT}/rules/000-global-iron-rules.md (é“å¾‹)"
[ -n "$PHASE_HINT" ] && echo "    Target phase hint: ${PHASE_HINT}" \
  || echo "    Pick the next un-built phase per the plan's stated build order."
echo ""

# -- STEP 2: AVOID SUPERSESSION ----------------------------------------------

echo "  STEP 2: CONFIRM MASTER STATE + OPEN PRs (avoid superseding)"
echo "    cd \"${PROJECT_ROOT}\""
echo "    git fetch origin --prune"
echo "    git checkout main && git pull --ff-only"
echo "    gh pr list --state open --json number,title,headRefName"
echo "    If an open PR already covers this phase -> STOP, surface it."
echo ""

# -- STEP 3: IMPLEMENT -------------------------------------------------------

echo "  STEP 3: IMPLEMENT THE PHASE"
echo "    Follow the plan's instructions for this phase."
echo "    Obey iron rules (rules/000):"
echo "      - No future functions (shift(-k) forbidden)"
echo "      - risk_filter before any order"
echo "      - T+1 enforcement"
echo "      - Credentials from .env only"
echo "      - try-except on all critical blocks"
echo "      - logging not print (except SimExecutor)"
echo "      - pathlib.Path for all paths"
echo "      - datetime objects for dates"
echo ""

# -- STEP 4: VERIFY ----------------------------------------------------------

echo "  STEP 4: VERIFY"
echo "    cd \"${PROJECT_ROOT}\""
echo "    ruff check app/ scripts/ services/ tests/"
echo "    python -m pytest tests/ -q --tb=short --no-header 2>&1 | tail -30"
echo ""
echo "    Phase-specific checks:"
echo "      Phase 1: ls data/raw/*.csv | wc -l  # >= 5"
echo "      Phase 2: python scripts/test_factor.py  # X.shape[2] >= 25"
echo "      Phase 3: ls app/models/trained/lstm_best.pth  # exists"
echo "      Phase 4: curl -X POST http://127.0.0.1:8000/api/v1/select  # < 5s"
echo "      Phase 5: streamlit run app/streamlit_app.py  # opens browser"
echo "      Phase 6: python -c \"from services.sim_executor import SimExecutor; ...\""
echo ""
echo "    Red -> fix in-agent -> retry (max 3)."
echo ""

# -- STEP 5: BRANCH + PR (NO MERGE) -----------------------------------------

echo "  STEP 5: BRANCH OFF MASTER + CREATE PR (NEVER MERGE)"
echo "    cd \"${PROJECT_ROOT}\""
echo "    BRANCH=\"auto-impl/<phase-slug>\""
echo "    git checkout -B \"\$BRANCH\" main"
echo "    git add -A"
echo "    git commit -m \"feat(<phase>): <summary>\""
echo "    git push -u origin \"\$BRANCH\""
echo "    gh pr create --base master --title \"<phase title>\" --body-file .pr-body.md"
echo "    DO NOT run gh pr merge. The AI code review gate handles merge."
echo ""

# -- STEP 6: POST-PR HANDOFF -------------------------------------------------

echo "  STEP 6: POST-PR HANDOFF"
echo "    After CI has run on the new PR, hand off to the fix loop:"
echo "      bash auto-fix.sh <NEW_PR_NUMBER>"
echo "    auto-fix.sh is a POST-PR stage only â€” never during implementation."
echo ""

# ============================================================================
# EXIT CONDITIONS
# ============================================================================

echo "================================================================"
echo "  EXIT CONDITIONS"
echo "================================================================"
echo ""
echo "  1. PR CREATED: gh pr create succeeded -> hand off to auto-fix.sh -> EXIT"
echo "  2. SUPERSEDED: open PR already covers this phase -> STOP (surface it)"
echo "  3. UNRESOLVABLE FAILURES: ruff/pytest red after 3 retries -> EXIT (human)"
echo "  4. PLAN AMBIGUOUS: phase cannot be determined -> EXIT (ask human)"
echo ""
echo "================================================================"
echo "  auto-implement-and-pr.sh END"
echo "================================================================"

