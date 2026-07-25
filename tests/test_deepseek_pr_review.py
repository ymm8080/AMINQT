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

    def test_max_tokens_is_8000(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data.decode())
            return FakeHTTPResponse(_api_response('{"issues": [], "summary": "ok"}'))

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert captured["data"]["max_tokens"] == 8000


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
            return FakeHTTPResponse(
                _api_response(content=None, reasoning=None)
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True

    def test_truncated_response_returns_error(self, monkeypatch):
        """finish_reason=length means content was cut by max_tokens."""
        def fake_urlopen(req, timeout=None):
            return FakeHTTPResponse(
                _api_response(
                    content='{"issues": [', finish_reason="length"
                )
            )

        monkeypatch.setattr(dsr.urllib.request, "urlopen", fake_urlopen)

        result = dsr.review_with_deepseek("diff", "key", "m", "https://x")

        assert result["error"] is True


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
        payload = {"issues": [], "summary": "No issues found."}

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
    """Error reviews must show 'Review failed', not 'No issues found'."""

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
