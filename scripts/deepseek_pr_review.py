"""AI-powered PR Review for AMINQT.

Called by .github/workflows/deepseek-pr-review.yml.
Reviews the PR diff using an LLM API (DeepSeek or GLM/Zhipu) and posts a
comment with findings.

Environment variables (set by the workflow):
    LLM_API_KEY: API key for the LLM provider (DeepSeek or GLM)
    GITHUB_TOKEN: GitHub token for posting comments
    GITHUB_REPOSITORY: owner/repo (e.g. "user/aminqt")
    PR_NUMBER: Pull request number
    LLM_MODEL: Model name (e.g. "glm-4.6" or "deepseek-chat")
    LLM_BASE_URL: API base URL (e.g. "https://open.bigmodel.cn/api/coding/paas/v4")
    LLM_PROVIDER: Provider name ("glm" or "deepseek"), controls param format

Legacy fallback: if LLM_* vars are absent, reads DEEPSEEK_* vars for
backward compatibility.
"""

import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The exact "no issues" example string from the system prompt. If the model
# returns this verbatim for a non-trivial diff, it parroted the example
# instead of reviewing the code.
_PARROTED_SUMMARY = "No issues found."
# Diffs shorter than this are considered trivial (e.g. whitespace-only);
# a canned "no issues" response is acceptable for them.
_PARROT_THRESHOLD = 500


def get_pr_diff(pr_number: str, repo: str, token: str) -> str:
    """Fetch PR diff via GitHub API.

    Tries the REST diff media type first; falls back to ``gh pr diff`` (which
    uses the authenticated GH_TOKEN) when the API returns 406 Not Acceptable.
    """
    # --- Strategy 1: REST API with diff media type ---
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    for accept in ("application/vnd.github.v3.diff", "application/vnd.github.diff"):
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"token {token}",
                "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                diff = resp.read().decode("utf-8", errors="replace")
                if diff.strip():
                    if len(diff) > 50000:
                        diff = diff[:50000] + "\n... [diff truncated for token budget]"
                    return diff
        except Exception as e:
            logger.error(f"Error fetching diff (Accept={accept}): {e}")

    # --- Strategy 2: gh CLI fallback (authenticated via GH_TOKEN env) ---
    try:
        env = {**os.environ, "GH_TOKEN": token}
        result = subprocess.run(
            ["gh", "pr", "diff", pr_number, "--repo", repo],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if result.returncode == 0 and result.stdout.strip():
            diff = result.stdout
            if len(diff) > 50000:
                diff = diff[:50000] + "\n... [diff truncated for token budget]"
            return diff
        logger.error(
            f"gh pr diff failed (rc={result.returncode}): {result.stderr[:200]}"
        )
    except Exception as e:
        logger.error(f"gh pr diff fallback error: {e}")

    # --- Strategy 3: List PR files API (handles PRs > 300 files) ---
    try:
        parts = []
        page = 1
        while page <= 10:  # 10 pages x 100 = 1000 files max
            files_url = (
                f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
                f"?page={page}&per_page=100"
            )
            freq = urllib.request.Request(
                files_url,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urllib.request.urlopen(freq, timeout=30) as resp:
                files = json.loads(resp.read().decode("utf-8", errors="replace"))
            if not files:
                break
            for f in files:
                patch = f.get("patch", "")
                if patch:
                    parts.append(f"--- {f.get('filename', '?')}\n{patch}")
            if len(files) < 100:
                break
            page += 1
        diff = "\n\n".join(parts)
        if diff.strip():
            if len(diff) > 30000:
                diff = diff[:30000] + "\n... [diff truncated for token budget]"
            return diff
    except Exception as e:
        logger.error(f"List PR files fallback error: {e}")

    return ""


def _sanitize_header(value: str) -> str:
    """Remove BOM and other non-latin-1 characters that break HTTP headers."""
    return (
        value.replace("\ufeff", "").encode("latin-1", errors="ignore").decode("latin-1")
    )


def _extract_json(text: str) -> dict | None:
    """Try multiple strategies to extract JSON from LLM response text."""
    if text is None:
        return None
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]

    # Strip thinking/reasoning tags (some models wrap output)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { and last } -- extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = text[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass

        # Try fixing common issues: trailing commas, unescaped newlines
        cleaned = re.sub(r",\s*}", "}", fragment)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    return None


def review_with_deepseek(
    diff: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: str = "deepseek",
) -> dict:
    """Send diff to LLM for review. Returns parsed response.

    The ``error`` key is set when the review itself failed (API error,
    parse failure, etc.) so callers can distinguish genuine "no issues"
    from a broken review that looks blank.

    Args:
        diff: PR diff text.
        api_key: LLM API key.
        model: Model name (e.g. "glm-4.6" or "deepseek-chat").
        base_url: API base URL (without ``/chat/completions`` suffix).
        provider: "deepseek" or "glm" — controls param format (thinking
            mode, response_format support).
    """
    if not diff.strip():
        return {"issues": [], "summary": "No diff to review.", "error": True}

    # api_key and base_url are already sanitized in main(); no re-sanitize needed

    system_prompt = """You are a code reviewer for a Python quant trading platform (AMINQT).
Review the PR diff and identify:
1. Future function violations: shift(-k) is FORBIDDEN in FEATURE computation \
(look-ahead bias). Label construction legitimately references future prices \
(e.g. label_kd = close[T+k]/close[T]-1) — this is NOT a violation. Functions \
named _label_reference, build_labels, or mask_suspension operate on labels, \
not features — do NOT flag them as future function violations.
2. Missing risk_filter before trading logic (not needed in label/cleaning steps)
3. Missing try-except error handling (only for network/API calls, file I/O, \
model.fit/predict, or external library calls that may raise unexpectedly; \
simple numpy/pandas operations like np.nan_to_num, boolean masking, and \
arithmetic do NOT need try-except)
4. Hardcoded credentials
5. String date comparison (must use datetime objects)
6. Missing logging (print is forbidden except SimExecutor)
7. Missing np.nan_to_num before model input (only for FEATURE matrices fed to \
model.fit/predict; label arrays (y) do NOT need nan_to_num. If np.nan_to_num \
is already called on the feature variable in the diff, do NOT flag it)
8. Division without safe_divide (zero division risk)

Respond in JSON format:
{"issues": [{"file": "...", "line": "...", "severity": "critical|warning|info", "message": "..."}], "summary": "one-line summary"}

If no issues found, your summary MUST include the diff stats (file count and
changed line count) provided in the user message, e.g.
"Reviewed 3 files / 450 changed lines, no violations of the 8 rules found."
Do NOT use the generic phrase "No issues found." — always include the counts.

Keep messages concise (one sentence per issue). Only report real violations.
"""

    # ── Thinking mode control ──────────────────────────────────
    # deepseek-v4-flash defaults to thinking mode (enabled). In thinking
    # mode the model emits reasoning_content BEFORE content. With a small
    # max_tokens budget the reasoning alone exhausts the limit, leaving
    # content empty → review appears blank.
    #
    # Code review is a checklist task; disable thinking for predictable
    # token usage and faster responses.
    # Diff stats — concrete numbers the model must acknowledge in its summary,
    # preventing it from blindly returning a canned "no issues" response.
    diff_file_count = sum(1 for line in diff.splitlines() if line.startswith("+++"))
    diff_changed_lines = sum(
        1
        for line in diff.splitlines()
        if line[:1] in ("+", "-") and line[:3] not in ("+++", "---")
    )
    diff_stats = (
        f"[Diff stats: {diff_file_count} files, {diff_changed_lines} changed lines]\n\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{diff_stats}Review this PR diff:\n\n```\n{diff}\n```",
        },
    ]

    max_tokens = 16000
    for attempt in range(3):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        # DeepSeek-specific: disable thinking mode for predictable token
        # usage. GLM (Zhipu) does not support this param.
        if provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            # glm-4.6 reasoning on a large diff routinely exceeds 60s
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                msg = result["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                parsed = _extract_json(content)
                if parsed is not None:
                    # Detect parroting: model returns the exact canned
                    # "No issues found." summary without reviewing the diff.
                    if (
                        not parsed.get("issues")
                        and parsed.get("summary", "").strip() == _PARROTED_SUMMARY
                        and len(diff) > _PARROT_THRESHOLD
                    ):
                        parsed["summary"] = (
                            "Review may be incomplete: model returned the "
                            "canned phrase 'No issues found.' without a "
                            "substantive summary for a non-trivial diff. "
                            "Re-run the review or inspect the diff manually."
                        )
                        parsed["error"] = True
                        logger.warning(
                            "Detected parroted '%s' for a %d-char diff "
                            "— flagging as suspicious.",
                            _PARROTED_SUMMARY,
                            len(diff),
                        )
                    return parsed
                finish_reason = result.get("choices", [{}])[0].get(
                    "finish_reason", "unknown"
                )
                logger.error(
                    "Could not parse review response (attempt %d). "
                    "finish_reason=%s, content (first 500): %r, "
                    "reasoning_content (first 300): %r",
                    attempt + 1,
                    finish_reason,
                    content[:500],
                    reasoning[:300],
                )
                # Retry on truncation with doubled max_tokens
                if finish_reason == "length" and attempt < 2:
                    max_tokens *= 2
                    logger.info(
                        "Retrying with max_tokens=%d due to finish_reason=length",
                        max_tokens,
                    )
                    continue
                if finish_reason == "length":
                    return {
                        "issues": [],
                        "summary": (
                            "Review failed: response truncated after "
                            "max retries (finish_reason=length). "
                            "Consider reducing diff size."
                        ),
                        "error": True,
                    }
                if not content.strip() and reasoning:
                    return {
                        "issues": [],
                        "summary": (
                            "Review failed: model returned empty content "
                            "(reasoning consumed token budget). "
                            "Consider increasing max_tokens."
                        ),
                        "error": True,
                    }
                return {
                    "issues": [],
                    "summary": "Could not parse review response.",
                    "error": True,
                }
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            logger.error("LLM API HTTP %s: %s", e.code, body)
            return {
                "issues": [],
                "summary": f"Review failed: HTTP {e.code} — {body}",
                "error": True,
            }
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return {
                "issues": [],
                "summary": f"Review failed: {e}",
                "error": True,
            }

    return {
        "issues": [],
        "summary": "Review failed: response truncated after 3 retries.",
        "error": True,
    }


def post_comment(pr_number: str, repo: str, token: str, review: dict) -> bool:
    """Post review as a formal PR review (visible in Files changed tab).

    Uses the pulls reviews API with ``event=COMMENT`` so the review
    appears in both the Conversation tab and the Files changed tab.
    The resulting review state is ``COMMENTED`` (no approve/request-changes),
    so it does not interfere with the PR Review Gate workflow.
    """
    issues = review.get("issues", [])
    summary = review.get("summary", "Review complete.")
    is_error = review.get("error", False)

    # Display name for the review comment header
    display_name = os.environ.get("LLM_PROVIDER", "deepseek").upper()
    display_model = os.environ.get("LLM_MODEL", "") or os.environ.get(
        "DEEPSEEK_MODEL", "unknown"
    )

    if is_error:
        # Review itself failed — show a visible error banner, NOT "no issues"
        body = f"""## {display_name} PR Review

**Status:** Review failed

{summary}

---
*Automated review by {display_name} ({display_model})*"""
    elif not issues:
        body = f"""## {display_name} PR Review

**Status:** No issues found.

{summary}

---
*Automated review by {display_name} ({display_model})*"""
    else:
        critical = [i for i in issues if i.get("severity") == "critical"]
        warnings = [i for i in issues if i.get("severity") == "warning"]
        info = [i for i in issues if i.get("severity") == "info"]

        lines = [f"## {display_name} PR Review", ""]
        lines.append(
            f"**Critical:** {len(critical)} | **Warnings:** {len(warnings)} "
            f"| **Info:** {len(info)}"
        )
        lines.append("")
        lines.append(f"> {summary}")
        lines.append("")

        for label, items in [
            ("### Critical Issues", critical),
            ("### Warnings", warnings),
            ("### Info", info),
        ]:
            if items:
                lines.append(label)
                for i in items:
                    lines.append(
                        f"- `{i.get('file', '?')}` L{i.get('line', '?')}: "
                        f"{i.get('message', '')}"
                    )
                lines.append("")

        marker = (
            "<!--AUTOFIX:HAS_ISSUES-->"
            if critical or warnings
            else "<!--AUTOFIX:CLEAN-->"
        )
        lines.append(marker)
        lines.append("")
        lines.append("---")
        lines.append(f"*Automated review by {display_name} ({display_model})*")

        body = "\n".join(lines)

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    data = json.dumps({"body": body, "event": "COMMENT"}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30):
            logger.info(f"Review posted on PR #{pr_number}")
            return True
    except Exception as e:
        logger.error(f"Error posting review: {e}")
        return False


def main():
    # ── LLM_* vars (new standard) with DEEPSEEK_* fallback (legacy) ──
    api_key = _sanitize_header(
        os.environ.get("LLM_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    )
    token = _sanitize_header(os.environ.get("GITHUB_TOKEN", ""))
    repo = _sanitize_header(os.environ.get("GITHUB_REPOSITORY", ""))
    pr_number = _sanitize_header(os.environ.get("PR_NUMBER", ""))
    model = _sanitize_header(
        os.environ.get("LLM_MODEL", "") or os.environ.get("DEEPSEEK_MODEL", "glm-4.6")
    )
    base_url = _sanitize_header(
        os.environ.get("LLM_BASE_URL", "")
        or os.environ.get("DEEPSEEK_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    )
    provider = _sanitize_header(
        os.environ.get("LLM_PROVIDER", "")
        or ("deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "glm")
    ).lower()

    if not all([api_key, token, repo, pr_number]):
        logger.error("Missing required environment variables")
        sys.exit(1)

    logger.info(f"Reviewing PR #{pr_number} in {repo} using {model} ({provider})")

    diff = get_pr_diff(pr_number, repo, token)
    logger.info(f"Diff length: {len(diff)} chars")

    review = review_with_deepseek(diff, api_key, model, base_url, provider)
    logger.info(f"Review result: {review.get('summary', 'N/A')}")

    post_comment(pr_number, repo, token, review)


if __name__ == "__main__":
    main()
