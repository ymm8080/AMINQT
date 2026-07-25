"""
公告因子引擎 — 安全网 #17, PIPELINE1_V3.8 继承不变, 附录B 接入规范 (B.1-B.5)
================================================================================
分阶段实施 (P19.1b):
  阶段二 (本模块): 人工扫雷 SOP → 手动输入公告标记 → announce_score; 关键词词典
                  情感分析 (B.3), 不接问财 API.
  阶段四: 问财 API (主) + DeepSeek LLM (复核) → 全自动化 (19:00-20:00 流水线).

附录B 接入规范 (沿用 V3.6 B.1-B.5):
  B.2 公告类型: 财报/解禁/增发/重组/风险提示/其他
  B.3 情感分析口径: 关键词词典 (阶段二) + LLM 复核 (阶段四), score ∈ [-1.0, +1.0]
  B.4 事件窗口: 财报前3日~后1日 / 解禁前5日~后3日 → 禁买 (D-8, 攻击档 grade_A_entry)
  B.5 入库 schema: symbol / announce_date / announce_type / announce_score /
                  event_window_flag

排序分公式: apply_announcement(score) = score × (1 + 0.3 × announce_score);
公告驱动的板块冻结与止损驱动的板块联动熔断 (E11) 互补.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

import pandas as pd

logger = logging.getLogger(__name__)

SCORE_ADJ = 0.3  # apply_announcement: score × (1 + 0.3×announce_score)
# B.4 事件窗口 (交易日)
EARNINGS_WINDOW = (-3, 1)  # 财报: 前3日 ~ 后1日
UNLOCK_WINDOW = (-5, 3)  # 解禁: 前5日 ~ 后3日
# 板块冻结: 单日同行业利空公告 ≥ N 只 → 冻结该行业
SECTOR_FREEZE_COUNT = 3

# B.3 关键词词典 (阶段二, LLM 复核留阶段四)
POSITIVE_KEYWORDS = (
    "预增",
    "扭亏",
    "中标",
    "签约",
    "获批",
    "回购",
    "增持",
    "分红",
    "突破",
    "量产",
    "定点",
    "超预期",
    "涨价",
    "扩产",
    "重组成功",
    "注入",
)
NEGATIVE_KEYWORDS = (
    "预亏",
    "预减",
    "减持",
    "处罚",
    "立案",
    "警示",
    "退市风险",
    "质押",
    "违约",
    "诉讼",
    "冻结",
    "问询",
    "商誉减值",
    "爆雷",
    "ST",
)

ANNOUNCE_TYPES = ("财报", "解禁", "增发", "重组", "风险提示", "其他")  # B.2


@dataclass
class Announcement:
    """B.5 入库 schema."""

    symbol: str
    announce_date: str  # YYYY-MM-DD
    announce_type: str  # B.2 六类
    announce_score: float  # [-1.0, +1.0]
    event_window_flag: bool = False
    title: str = ""
    industry: str = ""  # 板块冻结用


class AnnouncementFactor:
    """公告因子引擎 (阶段二: 人工扫雷 + 关键词词典).

    用法 (每日 19:00 扫雷 SOP):
        af = AnnouncementFactor("data/announcements")
        af.add_manual_entry("600519", "2026-07-25", "财报", title="中报预增50%")
        score = af.compute_announce_score("600519", "2026-07-25")
        cands = af.attach_scores(candidates, "2026-07-25")  # 并入清单候选
    """

    def __init__(self, store_dir: str | None = None):
        self.store_dir = store_dir
        self._records: list[Announcement] = []
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
            self._load()

    # ---------------- 持久化 (B.5, 追加式 WORM) ----------------
    def _path(self) -> str:
        return os.path.join(self.store_dir, "announcements.jsonl")

    def _load(self) -> None:
        path = self._path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._records.append(Announcement(**json.loads(line)))

    def _append(self, rec: Announcement) -> None:
        self._records.append(rec)
        if self.store_dir:
            with open(self._path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # ---------------- 人工扫雷入口 (阶段二) ----------------
    def add_manual_entry(
        self,
        symbol: str,
        announce_date: str,
        announce_type: str,
        title: str = "",
        score: float | None = None,
        industry: str = "",
    ) -> Announcement:
        """人工扫雷录入: score 缺省时由关键词词典自动评分 (B.3)."""
        assert announce_type in ANNOUNCE_TYPES, f"announce_type 须为 {ANNOUNCE_TYPES}"
        if score is None:
            score = self.analyze_sentiment(title)
        score = max(-1.0, min(1.0, float(score)))
        rec = Announcement(
            symbol=str(symbol),
            announce_date=str(announce_date),
            announce_type=announce_type,
            announce_score=score,
            event_window_flag=announce_type in ("财报", "解禁"),
            title=title,
            industry=industry,
        )
        self._append(rec)
        return rec

    # ---------------- B.3 情感分析 (关键词词典) ----------------
    @staticmethod
    def analyze_sentiment(text: str) -> float:
        """关键词词典评分 ∈ [-1.0, +1.0]: 命中利好 +0.5/个, 利空 -0.5/个, clip.

        无命中 → 0.0 (中性). 阶段四由 DeepSeek LLM 复核替换.
        """
        if not text:
            return 0.0
        pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
        neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
        return max(-1.0, min(1.0, 0.5 * (pos - neg)))

    # ---------------- 查询 ----------------
    def load_announcements(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """按标的+日期区间查询 (B.5 schema)."""
        rows = [
            asdict(r)
            for r in self._records
            if r.symbol == str(symbol) and str(start) <= r.announce_date <= str(end)
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "symbol",
                "announce_date",
                "announce_type",
                "announce_score",
                "event_window_flag",
                "title",
                "industry",
            ],
        )

    def compute_announce_score(self, symbol: str, trade_date: str) -> float:
        """当日有效公告情感分 ∈ [-1, +1]: 近 5 个自然日公告按时间衰减平均."""
        symbol, trade_date = str(symbol), str(trade_date)
        td = pd.Timestamp(trade_date)
        scores = []
        for r in self._records:
            if r.symbol != symbol:
                continue
            age = (td - pd.Timestamp(r.announce_date)).days
            if 0 <= age <= 5:
                scores.append(r.announce_score * (0.8**age))  # 时间衰减
        if not scores:
            return 0.0
        return max(-1.0, min(1.0, float(sum(scores) / len(scores))))

    def apply_announcement(self, score: float, symbol: str, trade_date: str) -> float:
        """排序分公告调整: score × (1 + 0.3×announce_score) (安全网 #17)."""
        return score * (1 + SCORE_ADJ * self.compute_announce_score(symbol, trade_date))

    # ---------------- B.4 事件窗口黑名单 (D-8) ----------------
    def is_event_window(self, symbol: str, trade_date: str) -> bool:
        """财报前3日~后1日 / 解禁前5日~后3日 → 禁买 (True=事件窗口内)."""
        symbol, trade_date = str(symbol), str(trade_date)
        td = pd.Timestamp(trade_date)
        for r in self._records:
            if r.symbol != symbol or not r.event_window_flag:
                continue
            lo, hi = EARNINGS_WINDOW if r.announce_type == "财报" else UNLOCK_WINDOW
            age = (td - pd.Timestamp(r.announce_date)).days  # 负=公告前, 正=公告后
            if lo <= age <= hi:  # 财报 [-3,+1] / 解禁 [-5,+3]
                return True
        return False

    # ---------------- 板块冻结 (公告驱动) ----------------
    def get_sector_freeze(self, trade_date: str) -> set[str]:
        """公告驱动的板块冻结: 当日同行业利空 (score<0) 公告 ≥3 只 → 冻结.

        与止损驱动的板块联动熔断 (E11 sector_fuse) 互补.
        """
        trade_date = str(trade_date)
        neg: dict[str, int] = {}
        for r in self._records:
            if r.announce_date == trade_date and r.announce_score < 0 and r.industry:
                neg[r.industry] = neg.get(r.industry, 0) + 1
        frozen = {ind for ind, n in neg.items() if n >= SECTOR_FREEZE_COUNT}
        if frozen:
            logger.error(
                "公告驱动板块冻结: %s (单日利空≥%d只)",
                sorted(frozen),
                SECTOR_FREEZE_COUNT,
            )
        return frozen

    # ---------------- 清单集成 ----------------
    def attach_scores(self, candidates: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """给清单候选并入 announce_score 列 (ListGenerator.compute_scores 消费).

        事件窗口内的标的 announce_score 置 -1.0 并标记 (下游/攻击档禁买).
        """
        out = candidates.copy()
        out["announce_score"] = [
            self.compute_announce_score(s, trade_date) for s in out["symbol"]
        ]
        if len(out):
            event_mask = [self.is_event_window(s, trade_date) for s in out["symbol"]]
            out.loc[event_mask, "announce_score"] = -1.0
            out["event_window"] = event_mask
        return out

    # ---------------- 清单失效条件 #5 (公告驱动) ----------------
    def list_invalidation(
        self, trade_date: str, threshold: float = -0.5
    ) -> dict[str, str]:
        """清单失效条件 #5: 公告剔除 (与盘中条件 1-4 互补, 回测必须模拟).

        剔除规则: 事件窗口内 (B.4) 或 有效公告情感 ≤ -0.5 (利空).
        Returns:
            {symbol: 剔除原因}
        """
        trade_date = str(trade_date)
        out: dict[str, str] = {}
        for symbol in {r.symbol for r in self._records}:
            if self.is_event_window(symbol, trade_date):
                out[symbol] = "失效#5: 事件窗口禁买 (财报/解禁)"
                continue
            score = self.compute_announce_score(symbol, trade_date)
            if score <= threshold:
                out[symbol] = f"失效#5: 公告利空 score={score:.2f}"
        if out:
            logger.error("清单失效#5 (公告剔除): %s", sorted(out))
        return out

    @staticmethod
    def apply_invalidation(
        candidates: pd.DataFrame, invalidated: dict[str, str]
    ) -> pd.DataFrame:
        """按失效#5 结果从候选中剔除 (回测与实盘同口径)."""
        if not invalidated:
            return candidates
        return candidates[~candidates["symbol"].isin(invalidated)]
