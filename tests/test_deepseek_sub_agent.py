"""Tests for services/deepseek_sub_agent.py — 测 MCP handler 业务逻辑, 全部 mock, 不触网.

测试策略: 直接调用 handle_call_tool_standalone / handle_list_tools_standalone
避开不同 MCP SDK 版本 (1.x 装饰器 / 2.x 构造器 / <1.x request_handlers) 的
dispatch 差异, 只验证 handler 本身的参数契约与错误处理。
"""

import asyncio

import services.deepseek_sub_agent as mcp_sub


def _run(coro):
    """同步 runner for async handler."""
    return asyncio.run(coro)


def test_list_tools_returns_six_tools():
    """list_tools handler 必须返回恰好 6 个注册工具."""
    result = _run(mcp_sub.handle_list_tools_standalone())
    assert len(result.tools) == 6
    names = {t.name for t in result.tools}
    assert names == {
        "deepseek_ask",
        "deepseek_review_code",
        "deepseek_analyze_sentiment",
        "deepseek_explain_signal",
        "deepseek_factor_hypothesis",
        "deepseek_diagnose",
    }


def test_call_tool_unknown_returns_error():
    """未知工具名必须返回 isError=True, 且提示 Unknown tool."""
    result = _run(mcp_sub.handle_call_tool_standalone("__no_such_tool__", {}))
    assert result.isError is True
    assert "Unknown tool" in result.content[0].text


def test_call_tool_ask_receives_arguments(monkeypatch):
    """deepseek_ask 必须真拿到 arguments 里的 prompt, 透传给 _call_deepseek."""
    captured = {}

    def fake_call(messages, system_prompt=None, temperature=0.3, response_format=None):
        captured["messages"] = messages
        return "OK"

    monkeypatch.setattr(mcp_sub, "_call_deepseek", fake_call)
    result = _run(
        mcp_sub.handle_call_tool_standalone("deepseek_ask", {"prompt": "ping"})
    )

    assert result.isError is False
    assert result.content[0].text == "OK"
    assert captured["messages"] == [{"role": "user", "content": "ping"}]


def test_call_tool_ask_uses_default_args(monkeypatch):
    """参数缺失时使用默认值 (temperature=0.3), 不抛 KeyError."""
    captured = {}

    def fake_call(messages, system_prompt=None, temperature=0.3, response_format=None):
        captured["temperature"] = temperature
        captured["system"] = system_prompt
        return "OK"

    monkeypatch.setattr(mcp_sub, "_call_deepseek", fake_call)
    result = _run(mcp_sub.handle_call_tool_standalone("deepseek_ask", {"prompt": "q"}))

    assert result.isError is False
    assert captured["temperature"] == 0.3
    assert captured["system"] is None


def test_call_tool_handles_exception(monkeypatch):
    """handler 内部异常必须被捕获并转为 isError=True, 不向外抛."""

    def fake_call(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mcp_sub, "_call_deepseek", fake_call)
    result = _run(mcp_sub.handle_call_tool_standalone("deepseek_ask", {"prompt": "q"}))

    assert result.isError is True
    assert "boom" in result.content[0].text


def test_server_object_is_ready():
    """server 对象 (被注册的 MCP server) 在 import 后必须可访问."""
    assert mcp_sub.server is not None
    assert mcp_sub.server.name == "deepseek-sub-agent"
