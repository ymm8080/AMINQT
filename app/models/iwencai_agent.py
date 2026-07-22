# -*- coding: utf-8 -*-
"""同花顺问财 AI 候选池生成器 (P10.5, ARCH §5.9.5, DESIGN_V1 §4 STEP1 第二步).

前置候选池生成器: 问财预筛在前, 模型打分在后。
不做"模型池 ∪ 问财池"事后合并 (v2.5 起废弃 merge_to_pipeline)。

双通道对接:
  - HTTP API 直调 (优先): requests 调问财后端 JSON 接口, 无需浏览器。
  - Playwright 降级 (备选): HTTP API 不可用时自动降级到浏览器自动化。

候选池 = ① 问财排名交集 ∩ base_pool ∩ ② 本地形态 OR
         (放量上涨缩量回踩/缩量上涨/控盘渐升/主力抢筹)
         − ③ 剔除 (放量下跌/高位巨量/脱离均线/板块退潮)。
形态计算仅用 rolling/shift, 无未来函数。
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 问财 HTTP API 配置 ─────────────────────────────────────────
_IWENCAI_API_URL = "https://www.iwencai.com/customized/chart/get-robot-data"
_IWENCAI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://www.iwencai.com/unifiedwap/result",
    "Origin": "https://www.iwencai.com",
}
_IWENCAI_TIMEOUT = 30  # 秒
_MAX_API_PERPAGE = 200  # 问财 API 单页最大行数

# ① 排名交集的 5 个 AND 条件
RANK_CONDITIONS = [
    "A股评分前200",
    "成交额前100",
    "热度主力前100",
    "主力资金流入前200",
    "股性极佳前100",
]

# 模板化查询预设
QUERY_TEMPLATES = {
    "super_strong": "超强主力 且 资金流入 且 技术突破",
    "leader": "板块龙头 且 核心股 且 主力资金流入",
    "low_position": "低位放量 且 换手率温和 且 非高位股",
}

_MIN_ROWS = 25  # 形态判断所需最少日线行数


class IwencaiAgent:
    """同花顺问财 AI — 前置候选池生成器 (HTTP API 优先 + Playwright 降级).

    定位: 候选池生成器, 非决策者 (铁律三: LLM 不直接交易)。

    数据获取双通道:
      1. _fetch_api()    — HTTP API 直调 (优先, 无需浏览器, 速度快)
      2. _fetch_playwright() — Playwright 浏览器自动化 (降级备选)
    _fetch() 统一入口: 先试 HTTP API, 失败自动降级到 Playwright。
    """

    def __init__(self, cookie_path: str = None,
                 use_api: bool = True) -> None:
        """初始化.

        Args:
            cookie_path: 问财登录态 storage_state 文件路径 (Playwright 降级用)。
            use_api: 是否优先使用 HTTP API (默认 True); False 则直接走 Playwright。
        """
        self.cookie_path = cookie_path
        self.use_api = use_api

    # ── HTTP API 直调 (优先路径) ──────────────────────────────────

    @staticmethod
    def _get_requests():
        """惰性导入 requests; 未安装 → RuntimeError."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests 未安装: pip install requests"
            ) from exc
        return requests

    @staticmethod
    def _extract_code(raw: str) -> str:
        """从问财返回的代码字段中提取 6 位股票代码.

        问财代码格式可能为 '600519' / 'SH600519' / '600519.SH' 等。
        """
        digits = re.sub(r"[^0-9]", "", str(raw))
        return digits.zfill(6) if digits else ""

    def _fetch_api(self, condition: str, top_n: int) -> List[dict]:
        """HTTP API 直调问财后端, 返回结构化结果.

        Args:
            condition: 自然语言查询条件。
            top_n: 返回上限。

        Returns:
            [{symbol, name, iwencai_score, match_reasons}]。

        Raises:
            RuntimeError: requests 未安装。
            Exception: API 请求失败 / 响应解析失败。
        """
        requests = self._get_requests()
        perpage = min(top_n, _MAX_API_PERPAGE)
        payload = {
            "question": condition,
            "perpage": perpage,
            "page": 1,
            "secondary_intent": "stock",
            "log_info": '{"input_type":"typewrite"}',
            "source": "Ths_iwencai_Xuangu",
            "version": "2.0",
            "query_area": "",
            "block_list": "",
            "add_info": json.dumps({
                "urp": {"scene": 1, "company": 1, "business": 1},
                "contentType": "json",
                "searchInfo": True,
            }),
        }
        resp = requests.post(
            _IWENCAI_API_URL, data=payload,
            headers=_IWENCAI_HEADERS, timeout=_IWENCAI_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # 问财 API 返回结构: data.result.{columns, rows} 或 data.data.result
        result = (
            data.get("data", {}).get("result")
            or data.get("result")
            or {}
        )
        if not result:
            logger.warning("[问财API] 查询 '%s' 返回空 result", condition)
            return []

        # 兼容两种返回格式: {columns, rows} 或 {result_list}
        cols = result.get("columns") or []
        raw_rows = result.get("rows") or result.get("result_list") or []

        if not cols or not raw_rows:
            logger.warning("[问财API] 查询 '%s' 无数据行", condition)
            return []

        # 列名中查找股票代码列和名称列
        code_col = next(
            (c for c in cols
             if any(k in str(c).lower() for k in ("code", "代码", "股票代码"))),
            cols[0],
        )
        name_col = next(
            (c for c in cols
             if any(k in str(c) for k in ("名称", "股票简称", "name"))),
            cols[1] if len(cols) > 1 else cols[0],
        )
        score_col = next(
            (c for c in cols
             if any(k in str(c) for k in ("评分", "score"))),
            None,
        )

        results: List[dict] = []
        for row in raw_rows[:top_n]:
            # row 可能是 list (按 cols 顺序) 或 dict
            if isinstance(row, dict):
                code = self._extract_code(row.get(code_col, ""))
                name = str(row.get(name_col, ""))
                score = (
                    float(row.get(score_col))
                    if score_col and row.get(score_col)
                    else None
                )
            elif isinstance(row, (list, tuple)):
                idx_code = cols.index(code_col) if code_col in cols else 0
                idx_name = cols.index(name_col) if name_col in cols else 1
                code = self._extract_code(
                    row[idx_code] if idx_code < len(row) else "")
                name = str(row[idx_name] if idx_name < len(row) else "")
                score = None
                if score_col and score_col in cols:
                    idx_score = cols.index(score_col)
                    if idx_score < len(row):
                        try:
                            score = float(row[idx_score])
                        except (TypeError, ValueError):
                            score = None
            else:
                continue
            if not code:
                continue
            results.append({
                "symbol": code,
                "name": name,
                "iwencai_score": score,
                "match_reasons": [condition],
            })
        logger.info(
            "[问财API] 查询 '%s' → %d 条", condition, len(results))
        return results

    # ── Playwright 降级路径 ───────────────────────────────────────

    @staticmethod
    def _get_sync_playwright():
        """惰性导入 playwright; 未安装 → RuntimeError."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "playwright 未安装: pip install playwright && "
                "playwright install chromium"
            ) from exc
        return sync_playwright

    def _fetch_playwright(self, condition: str, top_n: int) -> List[dict]:
        """Playwright 访问问财并解析结果表 (降级路径).

        Args:
            condition: 自然语言查询条件。
            top_n: 返回上限。

        Returns:
            [{symbol, name, iwencai_score, match_reasons}]。

        Raises:
            RuntimeError: playwright 未安装。
        """
        sync_playwright = self._get_sync_playwright()
        results: List[dict] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            storage = (
                self.cookie_path
                if self.cookie_path and os.path.exists(self.cookie_path)
                else None
            )
            context = browser.new_context(storage_state=storage)
            try:
                page = context.new_page()
                page.goto("https://www.iwencai.com/unifiedwap/result",
                          wait_until="domcontentloaded", timeout=30_000)
                page.fill("textarea", condition)
                page.keyboard.press("Enter")
                page.wait_for_selector("table", timeout=30_000)
                rows = page.eval_on_selector_all(
                    "table tbody tr",
                    "trs => trs.map(tr => Array.from(tr.cells)"
                    ".map(td => td.innerText.trim()))",
                )
                for row in rows[:top_n]:
                    if len(row) < 2:
                        continue
                    results.append({
                        "symbol": str(row[0]).split(".")[0].zfill(6),
                        "name": row[1],
                        "iwencai_score": None,
                        "match_reasons": [condition],
                    })
            finally:
                context.close()
                browser.close()
        logger.info(
            "[问财Playwright] 查询 '%s' → %d 条", condition, len(results))
        return results

    # ── 统一获取入口 (API 优先 → Playwright 降级) ─────────────────

    def _fetch(self, condition: str, top_n: int) -> List[dict]:
        """统一获取入口: 先试 HTTP API, 失败降级到 Playwright.

        Args:
            condition: 自然语言查询条件。
            top_n: 返回上限。

        Returns:
            [{symbol, name, iwencai_score, match_reasons}]。
        """
        if self.use_api:
            try:
                results = self._fetch_api(condition, top_n)
                if results:
                    return results
                logger.warning(
                    "[问财] API 返回空, 降级到 Playwright: '%s'",
                    condition)
            except Exception as exc:
                logger.warning(
                    "[问财] API 调用失败 (%s), 降级到 Playwright: '%s'",
                    exc, condition)
        # 降级 / 直接走 Playwright
        return self._fetch_playwright(condition, top_n)

    # ── 查询接口 ──────────────────────────────────────────────────

    def query(self, condition: str, top_n: int = 50) -> List[dict]:
        """自然语言条件查询.

        Args:
            condition: 如 'A股评分前200 且 成交额前100 且 热度主力前100'。
            top_n: 返回上限。

        Returns:
            [{symbol, name, iwencai_score, match_reasons}]。
        """
        return self._fetch(condition, top_n)

    def query_rank_intersection(self) -> List[str]:
        """① 排名条件交集 (AND):
        A股评分前200 ∩ 成交额前100 ∩ 热度主力前100 ∩
        主力资金流入前200 ∩ 股性极佳前100; 优选板块龙头核心股。
        """
        per_condition: List[set] = []
        first_order: List[str] = []
        for i, cond in enumerate(RANK_CONDITIONS):
            rows = self.query(cond, top_n=200)
            symbols = [r["symbol"] for r in rows]
            if i == 0:
                first_order = symbols
            per_condition.append(set(symbols))
        if not per_condition:
            return []
        intersection = set.intersection(*per_condition)
        ordered = [s for s in first_order if s in intersection]
        # 补足不在第一条结果顺序里的交集成员 (确定性排序)
        ordered.extend(sorted(intersection - set(ordered)))
        logger.info("[问财] 排名交集: %d 只", len(ordered))
        return ordered

    def query_by_template(self, template: str, **params) -> List[dict]:
        """模板化查询 (预设条件组合, 用户填参数).

        Args:
            template: QUERY_TEMPLATES 键名。
            **params: 附加条件参数 (追加到模板条件后)。

        Raises:
            ValueError: 未知模板。
        """
        if template not in QUERY_TEMPLATES:
            raise ValueError(
                f"未知问财模板: {template} (可选: {sorted(QUERY_TEMPLATES)})"
            )
        condition = QUERY_TEMPLATES[template]
        top_n = int(params.pop("top_n", 50))
        for key, val in params.items():
            condition += f" 且 {key}{val}"
        return self.query(condition, top_n=top_n)

    def get_market_hot(self) -> List[dict]:
        """问财热门概念/板块."""
        return self.query("今日热门概念板块 按热度降序", top_n=20)

    # ── ② 本地形态 (OR) ──────────────────────────────────────────

    @staticmethod
    def _prep(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """预计算均线 (rolling only); 数据不足返回 None."""
        if df is None or len(df) < _MIN_ROWS:
            return None
        out = df.copy()
        out["vol_ma5"] = out["volume"].rolling(5, min_periods=1).mean()
        out["vol_ma5_prev"] = out["vol_ma5"].shift(1)
        out["close_prev"] = out["close"].shift(1)
        return out

    @staticmethod
    def _pat_volume_surge_pullback(df: pd.DataFrame) -> bool:
        """放量上涨缩量回踩: 放量上涨日后, 回踩且量能逐级萎缩."""
        d = df.iloc[-10:]
        surge = (d["volume"] > 1.8 * d["vol_ma5_prev"]) & (
            d["close"] > d["close_prev"])
        surge_idx = [i for i, v in zip(d.index, surge) if v]
        if not surge_idx:
            return False
        j = surge_idx[-1]
        tail = d.loc[d.index > j]
        if not (1 <= len(tail) <= 5):
            return False
        surge_close = df.loc[j, "close"]
        surge_vol = df.loc[j, "volume"]
        pulled_back = bool(tail["close"].iloc[-1] < surge_close)
        shrinking = bool((tail["volume"] < surge_vol).all())
        return pulled_back and shrinking

    @staticmethod
    def _pat_low_volume_rise(df: pd.DataFrame) -> bool:
        """缩量上涨: 近 3 日价格上涨 + 当日量能低于 5 日均量."""
        price_up = bool(df["close"].iloc[-1] > df["close"].iloc[-4])
        low_vol = bool(df["volume"].iloc[-1] < df["vol_ma5"].iloc[-1])
        return price_up and low_vol

    @staticmethod
    def _pat_ctrl_rising(df: pd.DataFrame) -> bool:
        """控盘渐升: tech_ths_ctrl_ratio 局部低点与高点同步抬升 (2-3 周)."""
        if "tech_ths_ctrl_ratio" not in df.columns:
            return False
        r = df["tech_ths_ctrl_ratio"].iloc[-15:]
        if len(r) < 10 or r.isna().all():
            return False
        roll_min = r.rolling(3, min_periods=1).min()
        roll_max = r.rolling(3, min_periods=1).max()
        half = len(r) // 2
        eps = 1e-9  # 浮点容差: 平台期不算"渐升"
        lows_rising = bool(
            roll_min.iloc[half:].mean() > roll_min.iloc[:half].mean() + eps)
        highs_rising = bool(
            roll_max.iloc[half:].mean() > roll_max.iloc[:half].mean() + eps)
        return lows_rising and highs_rising

    @staticmethod
    def _pat_main_force_snatch(df: pd.DataFrame) -> bool:
        """主力抢筹: 近 1-2 周突发巨量 + 价格上涨."""
        d = df.iloc[-10:]
        spike = (d["volume"] > 2.5 * d["vol_ma5_prev"]) & (
            d["close"] > d["close_prev"] * 1.03)
        return bool(spike.any())

    # ── ③ 剔除条件 ────────────────────────────────────────────────

    @staticmethod
    def _exc_volume_drop(df: pd.DataFrame) -> bool:
        """放量下跌: 近 5 日存在巨量阴线."""
        d = df.iloc[-5:]
        cond = (d["volume"] > 1.8 * d["vol_ma5_prev"]) & (
            d["close"] < d["close_prev"] * 0.97)
        return bool(cond.any())

    @staticmethod
    def _exc_high_volume_top(df: pd.DataFrame) -> bool:
        """高位巨量: 价格接近 60 日新高 + 当日巨量."""
        high_60 = df["close"].rolling(60, min_periods=1).max().iloc[-1]
        near_top = bool(df["close"].iloc[-1] >= 0.95 * high_60)
        huge_vol = bool(df["volume"].iloc[-1] > 2.5 * df["vol_ma5"].iloc[-1])
        return near_top and huge_vol

    @staticmethod
    def _exc_off_ma20(df: pd.DataFrame) -> bool:
        """脱离均线: 收盘价高于 MA20 超过 15%."""
        ma20 = df["close"].rolling(20, min_periods=1).mean().iloc[-1]
        if ma20 <= 0 or np.isnan(ma20):
            return False
        return bool(df["close"].iloc[-1] > ma20 * 1.15)

    # ── 候选池构建 ────────────────────────────────────────────────

    def build_candidate_pool(self, base_pool: List[str],
                             daily_data: Dict[str, pd.DataFrame]) -> List[str]:
        """第二步完整候选池构建.

        Args:
            base_pool: 第一步基础池 (BaseLiquidityFilter 输出)。
            daily_data: {symbol: 日线 DataFrame} (本地形态过滤用,
                需含 open/high/low/close/volume, 控盘形态另需
                tech_ths_ctrl_ratio)。

        Returns:
            候选池 = ① 问财排名交集 ∩ base_pool ∩ ② 形态 OR
            (放量回踩/缩量上涨/控盘渐升/抢筹) − ③ 剔除
            (放量下跌/高位巨量/脱离均线/板块退潮)。
        """
        rank_set = set(self.query_rank_intersection())
        pool: List[str] = []
        for symbol in dict.fromkeys(base_pool):  # 去重保序
            # ① 排名交集 ∩ base_pool
            if symbol not in rank_set:
                continue
            df = self._prep(daily_data.get(symbol))
            if df is None:
                logger.info("[问财] %s 日线数据不足, 跳过", symbol)
                continue
            # ② 形态 OR
            if not (
                self._pat_volume_surge_pullback(df)
                or self._pat_low_volume_rise(df)
                or self._pat_ctrl_rising(df)
                or self._pat_main_force_snatch(df)
            ):
                continue
            # ③ 剔除
            if (
                self._exc_volume_drop(df)
                or self._exc_high_volume_top(df)
                or self._exc_off_ma20(df)
            ):
                continue
            pool.append(symbol)
        # 板块退潮剔除: 当前无板块数据源, 跳过并记录
        logger.info("[问财] 板块退潮剔除: 无板块数据, 跳过该剔除项")
        logger.info("[问财] 候选池: %d/%d", len(pool), len(base_pool))
        return pool
