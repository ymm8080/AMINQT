"""DeepSeek V4 Flash — Claude Code sub-agent MCP server.

Provides MCP tools that delegate to DeepSeek's API for specialized tasks:
code review, sentiment analysis, signal explanation, factor hypothesis,
and general reasoning.

Usage
-----
Claude Code auto-discovers via .claude/mcp.json.
Manual test:
    python services/deepseek_sub_agent.py

Env vars
--------
DEEPSEEK_API_KEY  : (required) DeepSeek API key
DEEPSEEK_MODEL    : (default: deepseek-v4-flash) model name
DEEPSEEK_BASE_URL : (default: https://api.deepseek.com) API base URL
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))
TIMEOUT_SEC = int(os.getenv("DEEPSEEK_TIMEOUT_SEC", "60"))

server = Server("deepseek-sub-agent")


# ── Helpers ───────────────────────────────────────────────────
def _call_deepseek(
    messages: list[dict],
    system_prompt: str | None = None,
    temperature: float = 0.3,
    response_format: dict | None = None,
    disable_thinking: bool = True,
) -> str:
    """Call DeepSeek chat completions API. Returns content string."""
    import httpx

    if not API_KEY:
        return "Error: DEEPSEEK_API_KEY not set. Add it to .env or environment."

    payload_messages: list[dict] = []
    if system_prompt:
        payload_messages.append({"role": "system", "content": system_prompt})
    payload_messages.extend(messages)

    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": payload_messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    if response_format:
        payload["response_format"] = response_format
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}

    try:
        with httpx.Client(timeout=TIMEOUT_SEC) as client:
            resp = client.post(
                f"{BASE_URL}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"].get("content") or ""
            return content
    except Exception as e:
        logger.error("DeepSeek API error: %s", e)
        return f"Error: {e}"


# ── Tool Definitions ──────────────────────────────────────────
TOOLS: list[Tool] = [
    Tool(
        name="deepseek_ask",
        description="通用问答：向 DeepSeek V4 Flash 提问任意问题，获得推理/分析结果",
        inputSchema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "问题内容"},
                "system_prompt": {
                    "type": "string",
                    "description": "可选的系统提示词，设定角色和语气",
                },
                "temperature": {
                    "type": "number",
                    "description": "采样温度 0-2，默认 0.3",
                },
            },
            "required": ["prompt"],
        },
    ),
    Tool(
        name="deepseek_review_code",
        description="代码审查：对代码 diff 或代码片段进行 8 维度量化规则检查",
        inputSchema={
            "type": "object",
            "properties": {
                "code_diff": {
                    "type": "string",
                    "description": "代码 diff 或完整代码片段",
                },
                "language": {
                    "type": "string",
                    "description": "编程语言 (默认 python)",
                },
            },
            "required": ["code_diff"],
        },
    ),
    Tool(
        name="deepseek_analyze_sentiment",
        description="市场情绪分析：分析新闻/公告/评论的情绪倾向，输出 bull/neutral/bear",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "待分析的文本内容"},
                "stock_code": {
                    "type": "string",
                    "description": "关联股票代码（可选）",
                },
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="deepseek_explain_signal",
        description="交易信号解释：解释模型输出的买入/卖出信号背后的逻辑",
        inputSchema={
            "type": "object",
            "properties": {
                "signal": {"type": "string", "description": "信号类型 (buy/sell/hold)"},
                "features": {
                    "type": "string",
                    "description": "当前特征值 JSON 或 key=value 列表",
                },
                "stock_code": {
                    "type": "string",
                    "description": "股票代码（可选）",
                },
            },
            "required": ["signal", "features"],
        },
    ),
    Tool(
        name="deepseek_factor_hypothesis",
        description="因子假设生成：基于当前市场数据，生成可测试的量化因子假设",
        inputSchema={
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "当前市场环境描述（板块/风格/时间段）",
                },
                "n_hypotheses": {
                    "type": "integer",
                    "description": "生成假设数量，默认 3",
                },
            },
            "required": ["context"],
        },
    ),
    Tool(
        name="deepseek_diagnose",
        description="系统诊断：分析错误日志和异常堆栈，定位根因并给出修复建议",
        inputSchema={
            "type": "object",
            "properties": {
                "error_log": {
                    "type": "string",
                    "description": "错误日志或异常堆栈",
                },
                "context": {
                    "type": "string",
                    "description": "运行上下文（模块/操作/输入数据特征）",
                },
            },
            "required": ["error_log"],
        },
    ),
]


# ── Request Handlers ──────────────────────────────────────────
@server.list_tools()
async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)


@server.call_tool()
async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    name = request.params.name
    args = request.params.arguments or {}

    try:
        if name == "deepseek_ask":
            result = _call_deepseek(
                messages=[{"role": "user", "content": args["prompt"]}],
                system_prompt=args.get("system_prompt"),
                temperature=float(args.get("temperature", 0.3)),
            )

        elif name == "deepseek_review_code":
            code = args["code_diff"]
            lang = args.get("language", "python")
            system = """You are a code reviewer for a Python quant trading platform.
Review the code and identify:
1. Future function violations (look-ahead bias in feature computation)
2. Missing risk_filter before trading logic
3. Missing try-except (network/API/file I/O only)
4. Hardcoded credentials
5. String date comparison (must use datetime objects)
6. Missing logging (print forbidden except SimExecutor)
7. Missing np.nan_to_num before model input
8. Division without safe_divide

Respond in JSON format:
{{"issues": [{{"severity": "critical|warning|info", "message": "..."}}], "summary": "..."}}"""
            result = _call_deepseek(
                messages=[
                    {
                        "role": "user",
                        "content": f"Review this {lang} code:\n\n```{lang}\n{code}\n```",
                    }
                ],
                system_prompt=system,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

        elif name == "deepseek_analyze_sentiment":
            text = args["text"]
            stock = args.get("stock_code", "")
            system = """You are a financial sentiment analyst.
Analyze the sentiment of the given text about A-share stocks.
Output JSON: {"sentiment": "bullish|neutral|bearish", "confidence": 0.0-1.0, "reasoning": "..."}"""
            user = (
                f"Analyze sentiment for stock {stock}:\n\n{text}"
                if stock
                else f"Analyze sentiment:\n\n{text}"
            )
            result = _call_deepseek(
                messages=[{"role": "user", "content": user}],
                system_prompt=system,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

        elif name == "deepseek_explain_signal":
            signal = args["signal"]
            features = args["features"]
            stock = args.get("stock_code", "")
            system = f"""You are a quant trading signal explainer.
Explain why the signal is '{signal}' based on the given features.
Output JSON: {{"signal": "{signal}", "key_factors": [...], "confidence": 0.0-1.0, "explanation": "..."}}"""
            user = (
                f"Stock: {stock}\nSignal: {signal}\nFeatures:\n{features}"
                if stock
                else f"Signal: {signal}\nFeatures:\n{features}"
            )
            result = _call_deepseek(
                messages=[{"role": "user", "content": user}],
                system_prompt=system,
                temperature=0.3,
                response_format={"type": "json_object"},
            )

        elif name == "deepseek_factor_hypothesis":
            ctx = args["context"]
            n = int(args.get("n_hypotheses", 3))
            system = f"""You are a quant factor researcher for A-share market.
Generate {n} testable factor hypotheses based on the given context.
Each hypothesis should include: factor_name, formula, hypothesis, expected_ic_sign.
Output JSON: {{"hypotheses": [{{"factor_name": "...", "formula": "...", "hypothesis": "...", "expected_ic_sign": "positive|negative"}}]}}"""
            result = _call_deepseek(
                messages=[
                    {
                        "role": "user",
                        "content": f"Market context:\n{ctx}\n\nGenerate {n} factor hypotheses.",
                    }
                ],
                system_prompt=system,
                temperature=0.5,
                response_format={"type": "json_object"},
            )

        elif name == "deepseek_diagnose":
            error_log = args["error_log"]
            ctx = args.get("context", "")
            system = """You are a senior DevOps engineer for a quant trading system.
Analyze the error log, identify root cause and suggest fixes.
Output JSON: {"root_cause": "...", "severity": "critical|warning|info", "fix": "...", "prevention": "..."}"""
            user = (
                f"Error log:\n{error_log}\n\nContext: {ctx}"
                if ctx
                else f"Error log:\n{error_log}"
            )
            result = _call_deepseek(
                messages=[{"role": "user", "content": user}],
                system_prompt=system,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

        else:
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            )

        return CallToolResult(
            content=[TextContent(type="text", text=result)],
        )

    except Exception as e:
        logger.error("Tool %s error: %s", name, e)
        return CallToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Error: {e}")],
        )


# ── Entrypoint ────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
