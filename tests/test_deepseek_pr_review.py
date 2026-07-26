"""Tests for scripts/deepseek_pr_review.py — all mocked, no network.

These tests lock down the fix for the "always-blank review" bug:
  - thinking mode must be disabled (otherwise reasoning_content eats the
    token budget and content comes back empty)
  - empty-content + non-empty-reasoning must surface as an error, not
    silently render as "No issues found"
  - HTTP errors must include the response body for debugging
  - post_comment must show "Review failed" when the review errored
"""

import io
import json
from urllib.error import HTTPError


import scripts.deepseek_pr_review as dsr

# ── helpers ────────────────────────────────────────────────────────


class FakeHTTPResponse:
    """Minimal file-like object for urllib.request.urlopen context."""

    def __init__(self, body_bytes: bytes):
        self._buf = io.BytesIO(body_bytes)

    def read(self):
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _api_response(
    content: str | None,
    reasoning: str | None = "",
    finish_reason: str = "stop",
) -> bytes:
    """Build a DeepSeek chat/completions response body.

    ``content`` and ``reasoning`` can be ``None`` to simulate the API
    returning ``null`` when thinking mode exhausts the token budget.
    """
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"total_tokens": 10},
        }
    ).encode()


# ── review_with_deepseek ───────────────────────────────────────────


class TestThinkingDisabled:
    """The core fix: thinking mode must be disabled in the payload."""

    def test_payload_includes_thinking_disabled(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data.decode())
            return FakeHTTPResponse(_api_response('{"issues": [], "summary": "ok"}'))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        dsr.review_with_deepseek("some diff", "key", "deepseek-v4-flash", "https://x")

        assert captured["data"]["thinking"] == {"type": "disabled"}

    def test_max_tokens_is_16000(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data.decode())
            return FakeHTTPResponse(_api_response('{"issues": [], "summary": "ok"}'))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert captured["data"]["max_tokens"] == 16000


class TestBlankReviewBug:
    """Reproduces the original bug: thinking mode eats tokens, content
    is empty, but old code returned error=False -> blank 'No issues'."""

    def test_empty_content_with_reasoning_returns_error(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(
                _api_response(content="", reasoning="Let me think about this diff...")
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True
        assert "empty content" in result["summary"].lower()

    def test_empty_diff_returns_error(self):
        result = dsr.review_with_deepseek("", "key", "m", "https://x")
        assert result["error"] is True

    def test_unparseable_content_returns_error(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(content="not json at all"))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")
        assert result["error"] is True

    def test_none_content_with_reasoning_returns_error(self, monkeypatch):
        """API returns content=null when thinking eats the budget."""

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(
                _api_response(content=None, reasoning="Let me think...")
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True
        assert "empty content" in result["summary"].lower()

    def test_none_content_none_reasoning_returns_error(self, monkeypatch):
        """Both content and reasoning are null (edge case)."""

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(content=None, reasoning=None))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True

    def test_truncated_response_returns_error(self, monkeypatch):
        """finish_reason=length means content was cut by max_tokens."""

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(
                _api_response(content='{"issues": [', finish_reason="length")
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True

    def test_truncated_then_success_retries(self, monkeypatch):
        """First call truncated, retry with doubled max_tokens succeeds."""
        call_count = [0]

        def fake_urlopen(req, timeout=None):
            call_count[0] += 1
            data = json.loads(req.data.decode())
            if call_count[0] == 1:
                assert data["max_tokens"] == 16000
                return FakeHTTPResponse(
                    _api_response(content='{"issues": [', finish_reason="length")
                )
            else:
                assert data["max_tokens"] == 32000
                return FakeHTTPResponse(
                    _api_response('{"issues": [], "summary": "ok"}')
                )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert call_count[0] == 2
        assert "error" not in result
        assert result["summary"] == "ok"

    def test_truncated_3_times_returns_error(self, monkeypatch):
        """All 3 retries truncated — returns error."""
        call_count = [0]

        def fake_urlopen(req, timeout=None):
            call_count[0] += 1
            return FakeHTTPResponse(
                _api_response(content='{"issues": [', finish_reason="length")
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert call_count[0] == 3
        assert result["error"] is True
        assert "truncated" in result["summary"].lower()


class TestParrotingDetection:
    """The model may return the canned 'No issues found.' summary verbatim
    without reviewing the diff. This must be detected and flagged."""

    def test_parroted_summary_flagged_for_large_diff(self, monkeypatch):
        """Model returns exact 'No issues found.' for a substantial diff.

        This reproduces the blank-review bug seen on PR #19/#20:
        41K-char diff, 1.6s response, summary = 'No issues found.'
        """
        payload = {"issues": [], "summary": "No issues found."}

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(json.dumps(payload)))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("x" * 1000, "key", "m", "https://x")

        assert result["error"] is True
        assert "incomplete" in result["summary"].lower()

    def test_parroted_summary_accepted_for_tiny_diff(self, monkeypatch):
        """Tiny diff (< 500 chars) with 'No issues found.' is acceptable."""
        payload = {"issues": [], "summary": "No issues found."}

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(json.dumps(payload)))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("x" * 100, "key", "m", "https://x")

        assert "error" not in result

    def test_substantive_summary_not_flagged(self, monkeypatch):
        """A summary that is NOT the canned string must pass cleanly."""
        payload = {
            "issues": [],
            "summary": "Reviewed 3 files / 450 changed lines, no violations found.",
        }

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(json.dumps(payload)))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("x" * 50000, "key", "m", "https://x")

        assert "error" not in result

    def test_issues_with_parroted_summary_not_flagged(self, monkeypatch):
        """If the model found issues, the summary string doesn't matter."""
        payload = {
            "issues": [
                {"file": "a.py", "line": "1", "severity": "warning", "message": "x"}
            ],
            "summary": "No issues found.",
        }

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(json.dumps(payload)))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("x" * 50000, "key", "m", "https://x")

        assert "error" not in result
        assert len(result["issues"]) == 1


class TestDiffStats:
    """The user message must include diff stats so the model can't
    return a canned response without acknowledging the diff."""

    def test_diff_stats_included_in_user_message(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data.decode())
            return FakeHTTPResponse(_api_response('{"issues": [], "summary": "ok"}'))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        diff = "+++ b/file.py\n+import os\n-old_import\n"
        dsr.review_with_deepseek(diff, "key", "m", "https://x")

        user_msg = captured["data"]["messages"][1]["content"]
        assert "[Diff stats:" in user_msg
        assert "1 files" in user_msg
        assert "2 changed lines" in user_msg


class TestSuccessfulReview:
    """A valid JSON response must parse cleanly with no error flag."""

    def test_valid_json_review(self, monkeypatch):
        payload = {
            "issues": [
                {
                    "file": "app/foo.py",
                    "line": "42",
                    "severity": "warning",
                    "message": "missing try-except",
                },
            ],
            "summary": "1 warning found",
        }

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(json.dumps(payload)))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert "error" not in result
        assert len(result["issues"]) == 1
        assert result["issues"][0]["severity"] == "warning"

    def test_no_issues_review(self, monkeypatch):
        # Use a substantive summary — the old "No issues found." is now
        # flagged as parroting for non-trivial diffs.
        payload = {"issues": [], "summary": "Reviewed 1 file, no violations found."}

        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(_api_response(json.dumps(payload)))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert "error" not in result
        assert result["issues"] == []


class TestHTTPError:
    """HTTP errors must include the response body, not be swallowed."""

    def test_http_error_includes_body(self, monkeypatch):
        error_body = b'{"error": {"message": "Model Not Exist"}}'

        def fake_urlopen(req, timeout=None):
            raise HTTPError(
                url="https://x",
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=io.BytesIO(error_body),
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True
        assert "400" in result["summary"]
        assert "Model Not Exist" in result["summary"]


# ── post_comment ───────────────────────────────────────────────────


class TestPostComment:
    """Reviews must be posted as formal PR reviews (visible in Files changed).

    The pulls reviews API is used with ``event=COMMENT`` so the review
    appears in both the Conversation tab and the Files changed tab.
    """

    def test_uses_pulls_reviews_endpoint(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse(b'{"id": 1}')

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        review = {"issues": [], "summary": "ok"}
        dsr.post_comment("1", "owner/repo", "tok", review)

        assert "/pulls/1/reviews" in captured["url"]
        assert "/issues/1/comments" not in captured["url"]

    def test_event_is_commented(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse(b'{"id": 1}')

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        review = {"issues": [], "summary": "ok"}
        dsr.post_comment("1", "owner/repo", "tok", review)

        assert captured["body"]["event"] == "COMMENT"

    def test_error_review_shows_failed_banner(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse(b'{"id": 1}')

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        review = {"issues": [], "summary": "boom", "error": True}
        dsr.post_comment("1", "owner/repo", "tok", review)

        assert "Review failed" in captured["body"]["body"]
        assert "No issues found" not in captured["body"]["body"]

    def test_clean_review_shows_no_issues(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse(b'{"id": 1}')

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        review = {"issues": [], "summary": "All good."}
        dsr.post_comment("1", "owner/repo", "tok", review)

        assert "No issues found" in captured["body"]["body"]
        assert "Review failed" not in captured["body"]["body"]

    def test_review_with_issues_shows_counts(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return FakeHTTPResponse(b'{"id": 1}')

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        review = {
            "issues": [
                {"file": "a.py", "line": "1", "severity": "critical", "message": "x"},
                {"file": "b.py", "line": "2", "severity": "warning", "message": "y"},
            ],
            "summary": "2 issues",
        }
        dsr.post_comment("1", "owner/repo", "tok", review)

        body = captured["body"]["body"]
        assert "Critical:** 1" in body
        assert "Warnings:** 1" in body
        assert "<!--AUTOFIX:HAS_ISSUES-->" in body


# ── _extract_json ──────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json(self):
        result = dsr._extract_json('{"issues": [], "summary": "ok"}')
        assert result == {"issues": [], "summary": "ok"}

    def test_markdown_fenced_json(self):
        result = dsr._extract_json('```json\n{"issues": [], "summary": "ok"}\n```')
        assert result == {"issues": [], "summary": "ok"}

    def test_json_embedded_in_text(self):
        result = dsr._extract_json(
            'Here is the review:\n{"issues": [], "summary": "ok"}\nDone.'
        )
        assert result == {"issues": [], "summary": "ok"}

    def test_trailing_comma_fixed(self):
        result = dsr._extract_json('{"issues": [], "summary": "ok",}')
        assert result == {"issues": [], "summary": "ok"}

    def test_think_tags_stripped(self):
        result = dsr._extract_json(
            '<think>reasoning here</think>{"issues": [], "summary": "ok"}'
        )
        assert result == {"issues": [], "summary": "ok"}

    def test_invalid_returns_none(self):
        assert dsr._extract_json("not json at all") is None

    def test_none_input_returns_none(self):
        assert dsr._extract_json(None) is None
