# AMINQT CI/CD é…ç½®æŒ‡å—

> æœ¬æŒ‡å—å¸®åŠ©ä½ é…ç½® GitHub ä»“åº“ä»¥å¯ç”¨å®Œæ•´çš„ CI/CD ç®¡é“

## å‰ç½®æ¡ä»¶

1. GitHub ä»“åº“å·²åˆ›å»ºä¸”ä»£ç å·²æŽ¨é€
2. [gh CLI](https://cli.github.com) å·²å®‰è£…å¹¶è®¤è¯ (`gh auth login`)
3. Python 3.9+ å·²å®‰è£…

## å¿«é€Ÿé…ç½®

```powershell
# åœ¨ AMINQT CODES ç›®å½•ä¸‹è¿è¡Œ
cd "D:\AMINQT\AMINQT CODES"

# æ›¿æ¢ä¸ºä½ çš„å®žé™…å€¼
.\scripts\setup-ci.ps1 -Repo "your-username/aminqt" -Pat "ghp_your_pat_here" -DeepSeekKey "sk-your_deepseek_key"
```

## éœ€è¦é…ç½®çš„ Secrets

| Secret | ç”¨é€” | èŽ·å–æ–¹å¼ |
|--------|------|----------|
| `GH_PAT` | è®© auto-fix push èƒ½é‡æ–°è§¦å‘ CI | GitHub Settings â†’ Developer settings â†’ Personal access tokens â†’ ç”Ÿæˆ PAT (repo + workflow scope) |
| `DEEPSEEK_API_KEY` | AI ä»£ç å®¡æŸ¥ | https://platform.deepseek.com â†’ API Keys |

### æ‰‹åŠ¨è®¾ç½®ï¼ˆå¦‚æžœè„šæœ¬å¤±è´¥ï¼‰

```bash
# è®¾ç½® GH_PAT
echo "ghp_your_pat" | gh secret set GH_PAT --repo your-username/aminqt

# è®¾ç½® DEEPSEEK_API_KEY
echo "sk_your_key" | gh secret set DEEPSEEK_API_KEY --repo your-username/aminqt
```

## éœ€è¦é…ç½®çš„ Variables

| Variable | å€¼ | ç”¨é€” |
|----------|-----|------|
| `CATPAW_SELF_HOSTED` | `true` | å¯ç”¨å®Œæ•´ CatPaw AI ä¿®å¤å¾ªçŽ¯ï¼ˆéœ€è¦ self-hosted runnerï¼‰ |

```bash
gh variable set CATPAW_SELF_HOSTED --body "true" --repo your-username/aminqt
```

## Branch Protection

åœ¨ GitHub ä»“åº“è®¾ç½®ä¸­é…ç½®ï¼š

1. Settings â†’ Branches â†’ Add rule â†’ Branch name: main
2. **Require status checks to pass before merging**:
   - `Lint`
   - `ADR Check`
   - `Test`
3. **Require branches to be up to date before merging** (strict)
4. **Require pull request reviews before merging**: 0 approving reviews
5. **Dismiss stale pull request approvals when new commits are pushed**: Yes

### å‘½ä»¤è¡Œé…ç½®

```bash
gh api --method PUT "repos/your-username/aminqt/branches/main/protection" \
  --field "required_status_checks[strict]=true" \
  --field "required_status_checks[contexts][]=Lint" \
  --field "required_status_checks[contexts][]=ADR Check" \
  --field "required_status_checks[contexts][]=Test" \
  --field "enforce_admins=false" \
  --field "required_pull_request_reviews[required_approving_review_count]=0" \
  --field "restrictions=null"
```

## Self-Hosted Runnerï¼ˆå¯é€‰ï¼Œç”¨äºŽå®Œæ•´ AI ä¿®å¤ï¼‰

### 1. æ³¨å†Œ Runner

GitHub ä»“åº“ â†’ Settings â†’ Actions â†’ Runners â†’ New self-hosted runner â†’ æŒ‰ç…§å¼•å¯¼ä¸‹è½½å’Œé…ç½®

ç¡®ä¿æ·»åŠ æ ‡ç­¾ `catpaw`ã€‚

### 2. å®‰è£… CatPaw CLI

åœ¨ runner æœºå™¨ä¸Šå®‰è£…å¹¶è®¤è¯ CatPaw CLIã€‚

### 3. éªŒè¯

```bash
# æ£€æŸ¥ runner æ˜¯å¦åœ¨çº¿
gh api repos/your-username/aminqt/actions/runners

# æ£€æŸ¥å˜é‡
gh variable list --repo your-username/aminqt

# æ£€æŸ¥ secretsï¼ˆåªæ˜¾ç¤ºåç§°ï¼Œä¸æ˜¾ç¤ºå€¼ï¼‰
gh secret list --repo your-username/aminqt
```

## å·¥ä½œæµè¯´æ˜Ž

```
PR åˆ›å»º/æŽ¨é€
    â”‚
    â”œâ”€â”€> pr-gate.yml          â†’ ç«‹å³é˜»æ­¢åˆå¹¶ (request changes)
    â”‚
    â”œâ”€â”€> ci.yml               â†’ Lint (ruff) + ADR éªŒè¯ + Test (pytest)
    â”‚
    â”œâ”€â”€> auto-fix.yml
    â”‚     â”œâ”€â”€> auto-fix-lint  â†’ ruff format + ruff check --fix â†’ commit + push
    â”‚     â””â”€â”€> auto-fix-full  â†’ CatPaw AI ä¿®å¤å¾ªçŽ¯ (éœ€ self-hosted runner)
    â”‚
    â”œâ”€â”€> deepseek-pr-review.yml â†’ AI ä»£ç å®¡æŸ¥ (æ£€æŸ¥é“å¾‹è¿è§„)
    â”‚
    â””â”€â”€> auto-approve.yml    â†’ CI é€šè¿‡åŽè§£é™¤åˆå¹¶é˜»æ­¢
```

## éªŒè¯é…ç½®

```bash
# åˆ›å»ºæµ‹è¯• PR
git checkout -b test-ci
echo "# test" > test.txt
git add test.txt
git commit -m "test: CI pipeline"
git push -u origin test-ci
gh pr create --title "Test CI" --body "Testing CI pipeline"

# æ£€æŸ¥ CI è¿è¡ŒçŠ¶æ€
gh pr checks <PR_NUMBER>
```

## æ•…éšœæŽ’é™¤

| é—®é¢˜ | è§£å†³ |
|------|------|
| auto-fix push ä¸è§¦å‘ CI | è®¾ç½® `GH_PAT` secretï¼ˆé»˜è®¤ `github.token` çš„ push ä¸è§¦å‘ workflowï¼‰ |
| DeepSeek review è¶…æ—¶ | æ£€æŸ¥ `DEEPSEEK_API_KEY` æ˜¯å¦æœ‰æ•ˆ |
| auto-fix-full ä¸è¿è¡Œ | ç¡®è®¤ `CATPAW_SELF_HOSTED=true` ä¸” runner åœ¨çº¿ |
| Branch protection æŠ¥é”™ | éœ€è¦ä»“åº“ admin æƒé™ |
| Tests æ‰¾ä¸åˆ° pytest | ç¡®è®¤ `requirements.txt` åŒ…å« pytest |

