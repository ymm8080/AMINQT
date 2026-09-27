"""Tests for services/deepseek_sub_agent.py — 只测 MCP 派发契约, 全部 mock, 不触网."""

import asyncio

from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult

import services.deepseek_sub_agent as mcp_sub


def _dispatch(name, arguments):
    """走 SDK 真实派发路径 (request_handlers[CallToolRequest]), 不直接调 handler."""
    handler = mcp_sub.server.request_handlers[CallToolRequest]
    req = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments),
    )
    return asyncio.run(handler(req)).root


def test_call_tool_dispatch_signature():
    """回归: handler 必须是 (name, arguments) 签名.

    mcp 1.23.3 的低层 call_tool 装饰器内部执行 await func(tool_name, arguments);
    旧式单参数 (request) 签名会让每次 tools/call 抛 TypeError 并被吞成 isError 结果,
    表现为六个工具全都"不可用"而不是报错。
    """
    result = _dispatch("__no_such_tool__", {})
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert "Unknown tool" in result.content[0].text


def test_call_tool_ask_receives_arguments(monkeypatch):
    """deepseek_ask 必须真拿到 arguments 里的 prompt."""
    captured = {}

    def fake_call(messages, system_prompt=None, temperature=0.3, response_format=None):
        captured["messages"] = messages
        return "OK"

    monkeypatch.setattr(mcp_sub, "_call_deepseek", fake_call)
    result = _dispatch("deepseek_ask", {"prompt": "ping"})

    assert result.isError is False
    assert result.content[0].text == "OK"
    assert captured["messages"] == [{"role": "user", "content": "ping"}]
