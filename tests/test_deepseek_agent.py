# -*- coding: utf-8 -*-
"""Tests for app/core/deepseek_agent.py — 全部 mock, 不触网."""

import json

import pytest

import app.core.deepseek_agent as ds_module
from app.core.deepseek_agent import DeepSeekAgent

ALL_FEATURES = [
    "explain_signal", "sentiment", "factor_hypothesis", "diagnose",
    "sentiment_index", "dragon_tiger", "northbound", "concept_graph",
    "similar_kline", "block_trade", "order_book", "ai_report",
]


def make_config(**overrides):
    cfg = {
        "deepseek": {
            "enabled": True,
            "base_url": "https://api.deepseek.test",
            "model": "deepseek-chat",
            "timeout_sec": 1,
            "daily_token_limit": 500000,
            "features": {f: True for f in ALL_FEATURES},
        }
    }
    cfg["deepseek"].update(overrides)
    return cfg


class FakeResponse:
    """模拟 requests.post 响应."""

    def __init__(self, content: dict, tokens: int = 100):
        self._body = {
            "choices": [{"message": {"content": json.dumps(
                content, ensure_ascii=False)}}],
            "usage": {"total_tokens": tokens},
        }

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


@pytest.fixture
def agent():
    return DeepSeekAgent(config=make_config())


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


class TestFeatureGate:
    """功能开关 + 密钥检查."""

    def test_feature_disabled(self, agent, monkeypatch):
        called = []
        monkeypatch.setattr(ds_module.requests, "post",
                            lambda *a, **k: called.append(1))
        agent._cfg["features"]["sentiment"] = False
        result = agent.analyze_sentiment("600519", ["利好"])
        assert result == {"available": False, "reason": "feature_disabled"}
        assert called == []

    def test_global_disabled(self, monkeypatch):
        agent = DeepSeekAgent(config=make_config(enabled=False))
        monkeypatch.setattr(ds_module.requests, "post",
                            lambda *a, **k: pytest.fail("不应发起请求"))
        result = agent.explain_signal("600519", 0.9, [], {})
        assert result["available"] is False
        assert result["reason"] == "feature_disabled"

    def test_missing_api_key(self, agent, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY")
        result = agent.explain_signal("600519", 0.9, [], {})
        assert result == {"available": False, "reason": "missing_api_key"}


class TestSuccessAndCache:
    """成功调用 + 当日缓存."""

    def test_success_returns_parsed_json(self, agent, monkeypatch):
        payload = {"explanation": "放量突破", "key_drivers": ["量比"]}
        monkeypatch.setattr(ds_module.requests, "post",
                            lambda *a, **k: FakeResponse(payload))
        result = agent.explain_signal("600519", 0.9, [], {})
        assert result["available"] is True
        assert result["result"] == payload

    def test_cache_hit_avoids_second_call(self, agent, monkeypatch):
        calls = []
        payload = {"index": 66.0, "comment": "偏暖"}

        def fake_post(*a, **k):
            calls.append(1)
            return FakeResponse(payload)

        monkeypatch.setattr(ds_module.requests, "post", fake_post)
        stats = {"up": 3000, "down": 1500}
        r1 = agent.market_sentiment_index(stats)
        r2 = agent.market_sentiment_index(stats)
        assert r1 == r2 == {"available": True, "result": payload}
        assert len(calls) == 1

    def test_token_usage_recorded(self, agent, monkeypatch):
        monkeypatch.setattr(ds_module.requests, "post",
                            lambda *a, **k: FakeResponse({"analysis": "x"},
                                                         tokens=250))
        agent.interpret_dragon_tiger({"seats": []})
        assert sum(agent._tokens_used.values()) == 250


class TestDegradation:
    """超时/异常 → 降级; 熔断; 日限额."""

    def test_api_error_degrades(self, agent, monkeypatch):
        def boom(*a, **k):
            raise ConnectionError("timeout")

        monkeypatch.setattr(ds_module.requests, "post", boom)
        result = agent.diagnose_stock("600519", {})
        assert result["available"] is False
        assert result["reason"].startswith("api_error")

    def test_invalid_json_degrades(self, agent, monkeypatch):
        class BadJson(FakeResponse):
            def json(self):
                return {"choices": [{"message": {"content": "not-json"}}],
                        "usage": {}}

        monkeypatch.setattr(ds_module.requests, "post",
                            lambda *a, **k: BadJson({}))
        result = agent.generate_report("600519", {})
        assert result["available"] is False

    def test_breaker_opens_after_3_failures(self, agent, monkeypatch):
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise ConnectionError("down")

        monkeypatch.setattr(ds_module.requests, "post", boom)
        # 3 次失败 (不同功能避免干扰)
        agent.diagnose_stock("600519", {})
        agent.concept_graph("600519")
        agent.analyze_order_book({})
        assert len(calls) == 3
        # 第 4 次: 熔断, 不再发请求
        result = agent.generate_report("600519", {})
        assert result == {"available": False, "reason": "circuit_open"}
        assert len(calls) == 3

    def test_success_resets_failure_count(self, agent, monkeypatch):
        responses = iter([
            ConnectionError("x"), ConnectionError("y"),
            FakeResponse({"analysis": "ok"}), FakeResponse({"analysis": "ok2"}),
        ])

        def flaky(*a, **k):
            item = next(responses)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(ds_module.requests, "post", flaky)
        agent.diagnose_stock("600519", {})
        agent.concept_graph("600519")
        assert agent._consecutive_failures == 2
        result = agent.analyze_order_book({})
        assert result["available"] is True
        assert agent._consecutive_failures == 0

    def test_daily_token_limit(self, monkeypatch):
        agent = DeepSeekAgent(config=make_config(daily_token_limit=100))
        from datetime import date

        agent._tokens_used[date.today().isoformat()] = 100
        monkeypatch.setattr(ds_module.requests, "post",
                            lambda *a, **k: pytest.fail("不应发起请求"))
        result = agent.explain_signal("600519", 0.9, [], {})
        assert result == {"available": False,
                          "reason": "daily_token_limit_exceeded"}


class TestAllTwelveMethods:
    """12 项功能均走统一降级契约."""

    def test_all_methods_build_payload(self, agent, monkeypatch):
        seen = []

        def fake_post(url, headers=None, json=None, timeout=None):
            seen.append((url, json, timeout))
            return FakeResponse({"ok": True})

        monkeypatch.setattr(ds_module.requests, "post", fake_post)
        agent.explain_signal("s", 0.1, [], {})
        agent.analyze_sentiment("s", ["t"])
        agent.propose_factor_hypotheses({})
        agent.diagnose_stock("s", {})
        agent.market_sentiment_index({})
        agent.interpret_dragon_tiger({})
        agent.interpret_northbound({})
        agent.concept_graph("s")
        agent.similar_kline("s", window=30)
        agent.analyze_block_trade({})
        agent.analyze_order_book({})
        agent.generate_report("s", {})
        assert len(seen) == 12
        for url, body, timeout in seen:
            assert url.startswith("https://api.deepseek.test")
            assert body["model"] == "deepseek-chat"
            assert body["messages"][0]["role"] == "system"
            assert timeout == 1
