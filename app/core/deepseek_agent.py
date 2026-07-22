# -*- coding: utf-8 -*-
"""DeepSeek LLM AI 辅助 (P13, ARCH §5.20, DESIGN_V1 §10.2).

12 项只读智能分析功能。LLM 输出仅展示给用户, 不进入交易信号链路。
API key 只存 .env (环境变量 DEEPSEEK_API_KEY);
超时 10s / 失败 3 次熔断 30 分钟 / 日 token 限额熔断。

降级契约: 任何失败 → {"available": False, "reason": str};
成功 → {"available": True, "result": <该功能类型结果>}。
缓存: 当日内存缓存 (date + feature + payload 哈希为键); Redis 可选。
"""

import hashlib
import json
import logging
import os
import time
from datetime import date
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

_BREAKER_THRESHOLD = 3  # 连续失败次数 → 熔断
_BREAKER_OPEN_SEC = 30 * 60  # 熔断时长 30 分钟
_DEFAULT_TIMEOUT = 10  # 单次超时 10s
_DEFAULT_DAILY_TOKEN_LIMIT = 500_000

_SYSTEM_PROMPT = (
    "你是 A 股量化分析助手。仅使用用户给定数据作答, 禁止编造数字; "
    "所有数字结论必须引用输入数据中的真实值。"
    "严格按指定 JSON schema 输出, 不输出任何额外文本。"
)


class DeepSeekAgent:
    """DeepSeek LLM 辅助分析代理 — 12 项功能 (全部只读).

    功能: 推理解释/情感分析/因子假设/诊股/情绪指数/龙虎榜/北向资金/
    概念图谱/相似K线/大宗交易/盘口分析/AI研报。
    """

    def __init__(self, config: dict = None) -> None:
        """加载 llm_config.yaml; API key 从环境变量 DEEPSEEK_API_KEY 读取.

        Args:
            config: 配置 dict (完整 yaml {"deepseek": {...}} 或内层 dict);
                None 时尝试读取 config/llm_config.yaml。
        """
        if config is None:
            config = self._load_default_config()
        self.config = config or {}
        self._cfg = self.config.get("deepseek", self.config)
        # 当日缓存: {(date, feature, payload_hash): result}
        self._cache: Dict[tuple, dict] = {}
        # 日 token 用量: {date: tokens}
        self._tokens_used: Dict[str, int] = {}
        # 熔断器状态
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @staticmethod
    def _load_default_config() -> dict:
        """读取 config/llm_config.yaml; 失败返回空 dict (全部功能降级)."""
        try:
            import yaml

            path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "config",
                "llm_config.yaml",
            )
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:  # noqa: BLE001
            logger.warning("[DeepSeek] llm_config.yaml 加载失败, 全部功能降级")
            return {}

    # ── 核心调用 ──────────────────────────────────────────────────

    def _degraded(self, reason: str) -> dict:
        """构造降级结果."""
        return {"available": False, "reason": reason}

    def _feature_enabled(self, feature: str) -> bool:
        """功能开关检查 (全局 enabled + 单项 feature)."""
        if not self._cfg.get("enabled", False):
            return False
        return bool(self._cfg.get("features", {}).get(feature, False))

    def _breaker_open(self) -> bool:
        """熔断器是否处于打开状态."""
        return time.time() < self._breaker_open_until

    def _record_failure(self) -> None:
        """记录失败; 连续 3 次 → 熔断 30 分钟."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.time() + _BREAKER_OPEN_SEC
            logger.error(
                "[DeepSeek] 连续失败 %d 次, 熔断 %d 分钟",
                self._consecutive_failures,
                _BREAKER_OPEN_SEC // 60,
            )

    def _record_success(self, tokens: int) -> None:
        """记录成功 + token 用量."""
        self._consecutive_failures = 0
        today = date.today().isoformat()
        self._tokens_used[today] = self._tokens_used.get(today, 0) + max(tokens, 0)

    def _tokens_exceeded(self) -> bool:
        """日 token 限额检查."""
        limit = int(self._cfg.get("daily_token_limit", _DEFAULT_DAILY_TOKEN_LIMIT))
        return self._tokens_used.get(date.today().isoformat(), 0) >= limit

    def _call(self, feature: str, payload: dict) -> dict:
        """统一调用入口: 开关/熔断/限额/缓存/HTTP/解析/降级.

        Args:
            feature: 功能名 (对应 llm_config.yaml features 键)。
            payload: {"prompt": str, "schema": dict} — 渲染后的用户 prompt
                与期望 JSON schema。

        Returns:
            {"available": True, "result": <解析后的 JSON>} 或
            {"available": False, "reason": str}。
        """
        if not self._feature_enabled(feature):
            return self._degraded("feature_disabled")
        if self._breaker_open():
            return self._degraded("circuit_open")
        if self._tokens_exceeded():
            return self._degraded("daily_token_limit_exceeded")

        cache_key = (
            date.today().isoformat(),
            feature,
            hashlib.md5(
                json.dumps(
                    payload, sort_keys=True, ensure_ascii=False, default=str
                ).encode("utf-8")
            ).hexdigest(),
        )
        if cache_key in self._cache:
            logger.info("[DeepSeek] 缓存命中: %s", feature)
            return self._cache[cache_key]

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return self._degraded("missing_api_key")

        base_url = self._cfg.get("base_url", "https://api.deepseek.com")
        model = self._cfg.get("model", "deepseek-chat")
        timeout = float(self._cfg.get("timeout_sec", _DEFAULT_TIMEOUT))
        user_prompt = (
            f"{payload['prompt']}\n\n"
            f"输出 JSON schema: {json.dumps(payload['schema'], ensure_ascii=False)}"
        )
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            tokens = int(body.get("usage", {}).get("total_tokens", 0))
        except Exception as exc:  # noqa: BLE001 — 任何失败 → 降级
            logger.warning("[DeepSeek] %s 调用失败: %s", feature, exc)
            self._record_failure()
            return self._degraded(f"api_error: {exc}")

        self._record_success(tokens)
        result = {"available": True, "result": parsed}
        self._cache[cache_key] = result
        return result

    # ── 12 项功能 ─────────────────────────────────────────────────

    def explain_signal(
        self, symbol: str, score: float, top_factors: List[dict], shap: dict
    ) -> dict:
        """1. 推理解释: 为什么选/为什么买."""
        return self._call(
            "explain_signal",
            {
                "prompt": (
                    f"解释选股信号: 标的 {symbol}, 模型得分 {score:.4f}, "
                    f"Top-K 因子 {json.dumps(top_factors, ensure_ascii=False, default=str)}, "
                    f"SHAP 归因 {json.dumps(shap, ensure_ascii=False, default=str)}。"
                    "用中文解释为什么选/为什么买。"
                ),
                "schema": {"explanation": "str", "key_drivers": ["str"]},
            },
        )

    def analyze_sentiment(self, symbol: str, texts: List[str]) -> dict:
        """2. 情感分析: 新闻/公告/股吧 → 情感得分 (-1~1) + 摘要."""
        return self._call(
            "sentiment",
            {
                "prompt": (
                    f"对标的 {symbol} 的以下文本做情感分析, "
                    f"给出 -1~1 情感得分与摘要: {json.dumps(texts, ensure_ascii=False)}"
                ),
                "schema": {"score": "float (-1~1)", "summary": "str"},
            },
        )

    def propose_factor_hypotheses(self, ic_report: dict) -> dict:
        """3. 因子假设生成 (供 FactorDiscovery 验证)."""
        return self._call(
            "factor_hypothesis",
            {
                "prompt": (
                    "基于以下历史因子 IC 报告, 提出可量化验证的新因子假设: "
                    f"{json.dumps(ic_report, ensure_ascii=False, default=str)}"
                ),
                "schema": {
                    "hypotheses": [
                        {"name": "str", "formula": "str", "rationale": "str"}
                    ]
                },
            },
        )

    def diagnose_stock(self, symbol: str, factor_pack: dict) -> dict:
        """4. 诊股: 趋势/资金/筹码/风险多维诊断."""
        return self._call(
            "diagnose",
            {
                "prompt": (
                    f"对 {symbol} 做多维诊断 (趋势/资金/筹码/风险), "
                    f"因子数据: {json.dumps(factor_pack, ensure_ascii=False, default=str)}"
                ),
                "schema": {
                    "trend": "str",
                    "capital": "str",
                    "chips": "str",
                    "risk": "str",
                    "conclusion": "str",
                },
            },
        )

    def market_sentiment_index(self, market_stats: dict) -> dict:
        """5. 市场情绪指数 (0~100)."""
        return self._call(
            "sentiment_index",
            {
                "prompt": (
                    "基于全市场涨跌/成交/涨停数据计算市场情绪指数 (0~100): "
                    f"{json.dumps(market_stats, ensure_ascii=False, default=str)}"
                ),
                "schema": {"index": "float (0~100)", "comment": "str"},
            },
        )

    def interpret_dragon_tiger(self, lhb_data: dict) -> dict:
        """6. 龙虎榜席位解读."""
        return self._call(
            "dragon_tiger",
            {
                "prompt": (
                    "解读龙虎榜席位行为 (游资/机构/北向): "
                    f"{json.dumps(lhb_data, ensure_ascii=False, default=str)}"
                ),
                "schema": {"analysis": "str", "seat_types": ["str"]},
            },
        )

    def interpret_northbound(self, north_data: dict) -> dict:
        """7. 北向资金解读."""
        return self._call(
            "northbound",
            {
                "prompt": (
                    "解读北向资金偏好与异动: "
                    f"{json.dumps(north_data, ensure_ascii=False, default=str)}"
                ),
                "schema": {"analysis": "str", "preference": "str"},
            },
        )

    def concept_graph(self, symbol: str) -> dict:
        """8. 概念图谱 (同概念联动股)."""
        return self._call(
            "concept_graph",
            {
                "prompt": f"给出 {symbol} 所属概念/题材及同概念联动股图谱。",
                "schema": {
                    "concepts": ["str"],
                    "linked": [{"symbol": "str", "name": "str", "concept": "str"}],
                },
            },
        )

    def similar_kline(self, symbol: str, window: int = 60) -> dict:
        """9. 相似 K 线检索 + 后续走势统计."""
        return self._call(
            "similar_kline",
            {
                "prompt": (
                    f"检索与 {symbol} 最近 {window} 日 K 线形态相似的历史个股, "
                    "并给出后续走势统计。"
                ),
                "schema": {
                    "similar": [
                        {
                            "symbol": "str",
                            "similarity": "float",
                            "forward_return": "float",
                        }
                    ]
                },
            },
        )

    def analyze_block_trade(self, block_data: dict) -> dict:
        """10. 大宗交易分析."""
        return self._call(
            "block_trade",
            {
                "prompt": (
                    "分析大宗交易折溢价与买卖双方意图: "
                    f"{json.dumps(block_data, ensure_ascii=False, default=str)}"
                ),
                "schema": {"analysis": "str", "premium_rate": "float"},
            },
        )

    def analyze_order_book(self, book_snapshot: dict) -> dict:
        """11. 盘口分析 (五档 + 逐笔)."""
        return self._call(
            "order_book",
            {
                "prompt": (
                    "解读五档盘口与逐笔成交的买卖力量, 提示异动: "
                    f"{json.dumps(book_snapshot, ensure_ascii=False, default=str)}"
                ),
                "schema": {"analysis": "str", "strength": "buy|sell|neutral"},
            },
        )

    def generate_report(self, symbol: str, full_pack: dict) -> dict:
        """12. AI 研报 (亮点/风险/结论)."""
        return self._call(
            "ai_report",
            {
                "prompt": (
                    f"基于 {symbol} 综合数据包生成结构化研报 (亮点/风险/结论): "
                    f"{json.dumps(full_pack, ensure_ascii=False, default=str)}"
                ),
                "schema": {
                    "highlights": ["str"],
                    "risks": ["str"],
                    "conclusion": "str",
                },
            },
        )
