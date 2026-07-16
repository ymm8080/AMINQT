# -*- coding: utf-8 -*-
"""DeepSeek PR Review — AI-powered code review for AMINQT.

Called by .github/workflows/deepseek-pr-review.yml.
Reviews the PR diff using DeepSeek API and posts a comment with findings.

Environment variables (set by the workflow):
    DEEPSEEK_API_KEY: API key for DeepSeek
    GITHUB_TOKEN: GitHub token for posting comments
    GITHUB_REPOSITORY: owner/repo (e.g. "user/aminqt")
    PR_NUMBER: Pull request number
    DEEPSEEK_MODEL: Model name (e.g. "deepseek-v4-flash")
    DEEPSEEK_BASE_URL: API base URL (e.g. "https://api.deepseek.com")
"""

import json
import os
import sys
import urllib.request


def get_pr_diff(pr_number: str, repo: str, token: str) -> str:
    """Fetch PR diff via GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3.diff",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            diff = resp.read().decode("utf-8", errors="replace")
            if len(diff) > 50000:
                diff = diff[:50000] + "\n... [diff truncated for token budget]"
            return diff
    except Exception as e:
        print(f"Error fetching diff: {e}")
        return ""


def review_with_deepseek(diff: str, api_key: str, model: str, base_url: str) -> dict:
    """Send diff to DeepSeek for review. Returns parsed response."""
    if not diff.strip():
        return {"issues": [], "summary": "No diff to review."}

    system_prompt = """You are a code reviewer for a Python quant trading platform (AMINQT).
Review the PR diff and identify:
1. Future function violations (shift(-k) is FORBIDDEN)
2. Missing risk_filter before trading logic
3. Missing try-except error handling
4. Hardcoded credentials
5. String date comparison (must use datetime objects)
6. Missing logging (print is forbidden except SimExecutor)
7. Missing np.nan_to_num before model input
8. Division without safe_divide (zero division risk)

Respond in JSON format:
{"issues": [{"file": "...", "line": "...", "severity": "critical|warning|info", "message": "..."}], "summary": "one-line summary"}

If no issues found: {"issues": [], "summary": "No issues found."}
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Review this PR diff:\n\n```\n{diff}\n```"},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }

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
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            # Parse JSON from response (handle markdown code blocks)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content)
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        return {"issues": [], "summary": f"Review failed: {e}"}


def post_comment(pr_number: str, repo: str, token: str, review: dict) -> bool:
    """Post review comment on the PR."""
    issues = review.get("issues", [])
    summary = review.get("summary", "Review complete.")

    if not issues:
        body = f"""## DeepSeek PR Review

**Status:** No issues found.

{summary}

---
*Automated review by DeepSeek ({os.environ.get("DEEPSEEK_MODEL", "unknown")})*"""
    else:
        critical = [i for i in issues if i.get("severity") == "critical"]
        warnings = [i for i in issues if i.get("severity") == "warning"]
        info = [i for i in issues if i.get("severity") == "info"]

        lines = ["## DeepSeek PR Review", ""]
        lines.append(
            f"**Critical:** {len(critical)} | **Warnings:** {len(warnings)} | **Info:** {len(info)}"
        )
        lines.append("")
        lines.append(f"> {summary}")
        lines.append("")

        if critical:
            lines.append("### Critical Issues")
            for i in critical:
                lines.append(
                    f"- ``{i.get('file', '?')}`` L{i.get('line', '?')}: {i.get('message', '')}"
                )
            lines.append("")

        if warnings:
            lines.append("### Warnings")
            for i in warnings:
                lines.append(
                    f"- ``{i.get('file', '?')}`` L{i.get('line', '?')}: {i.get('message', '')}"
                )
            lines.append("")

        if info:
            lines.append("### Info")
            for i in info:
                lines.append(
                    f"- ``{i.get('file', '?')}`` L{i.get('line', '?')}: {i.get('message', '')}"
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
        lines.append(
            f"*Automated review by DeepSeek ({os.environ.get('DEEPSEEK_MODEL', 'unknown')})*"
        )

        body = "\n".join(lines)

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    data = json.dumps({"body": body}).encode("utf-8")
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
            print(f"Comment posted on PR #{pr_number}")
            return True
    except Exception as e:
        print(f"Error posting comment: {e}")
        return False


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not all([api_key, token, repo, pr_number]):
        print("Missing required environment variables")
        sys.exit(1)

    print(f"Reviewing PR #{pr_number} in {repo} using {model}")

    diff = get_pr_diff(pr_number, repo, token)
    print(f"Diff length: {len(diff)} chars")

    review = review_with_deepseek(diff, api_key, model, base_url)
    print(f"Review result: {review.get('summary', 'N/A')}")

    post_comment(pr_number, repo, token, review)


if __name__ == "__main__":
    main()
