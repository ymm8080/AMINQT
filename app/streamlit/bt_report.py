"""回测报告解析 (只读) — 从 backtest.json 提取结构化展示数据.

纯函数: dict in → DataFrame/dict out, 供 page_archive 回测历史 tab 渲染.
所有字段缺失防御性返回空, 旧 schema (无 conclusion) 不崩页.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = ["3d", "5d", "10d"]

_PICKS_COLS = [
    "date",
    "system",
    "rk",
    "symbol",
    "score",
    "mfe_3d",
    "mfe_5d",
    "mfe_10d",
]
_SYS_COLS = ["system", "horizon", "winrate", "mag", "n", "ok", "base_winrate"]
_PH_COLS = ["horizon", "cut", "winrate", "mag", "n", "ok", "base_winrate"]
_CONCL_COLS = [
    "board",
    "cut",
    "kept",
    "best_horizon",
    "winrate",
    "mag",
    "delta_wr",
    "baseline_wr",
    "n",
]


# ───────────────────────── 磁盘 IO (薄封装) ─────────────────────────
def list_runs(base) -> list[dict]:
    """列出回测目录下所有 run 目录 → [{ts, mtime, path}] (时间降序)."""
    if not os.path.isdir(base):
        return []
    runs = []
    for name in sorted(os.listdir(base), reverse=True):
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        mtime = ""
        try:
            mtime = datetime.fromtimestamp(os.stat(p).st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
        except OSError:
            pass
        runs.append({"ts": name, "mtime": mtime, "path": p})
    return runs


def load_run_json(ts: str, base) -> dict | None:
    """加载某 run 的 backtest.json → dict; 缺失/损坏返回 None."""
    p = os.path.join(base, ts, "backtest.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取回测 JSON 失败 %s: %s", p, exc)
        return None


# ───────────────────────── 内部防御取值 ─────────────────────────
def _boards(d: dict) -> dict:
    b = d.get("boards")
    return b if isinstance(b, dict) else {}


def _conclusion_boards(d: dict) -> dict:
    c = d.get("conclusion")
    b = c.get("boards") if isinstance(c, dict) else None
    return b if isinstance(b, dict) else {}


def list_boards(d: dict) -> list[str]:
    """返回该 run 含板块名列表 (main/dual 优先)."""
    boards = list(_boards(d).keys())
    if "main" in boards or "dual" in boards:
        return [b for b in ("main", "dual") if b in boards]
    return boards


def _base(d: dict | None, key: str) -> dict:
    v = d.get(key) if isinstance(d, dict) else None
    return v if isinstance(v, dict) else {}


# OOS 窗口 label 优先序: 常规 run 多窗 (6m/3m/10d); oos_days 单窗 run 只有 "oos".
_OOS_LABEL_PRIORITY = ("6m", "3m", "10d")


def _pick_oos_window(oos: dict) -> dict:
    """从 OOS 窗口 dict 挑主窗口: 优先 6m, 否则任取首个非空 (oos_days 单窗 run 兼容)."""
    if not isinstance(oos, dict):
        return {}
    for lab in _OOS_LABEL_PRIORITY:
        w = oos.get(lab)
        if isinstance(w, dict) and w:
            return w
    for w in oos.values():
        if isinstance(w, dict) and w:
            return w
    return {}


# ───────────────────────── 解析函数 ─────────────────────────
def parse_conclusion_summary(d: dict) -> pd.DataFrame:
    """从 conclusion.boards[*].cuts.top5/top10 提取 cut 行."""
    rows = []
    for board, bc in _conclusion_boards(d).items():
        if not isinstance(bc, dict):
            continue
        cuts = bc.get("cuts")
        if not isinstance(cuts, dict):
            continue
        for cut, c in cuts.items():
            if not isinstance(c, dict):
                continue
            rows.append(
                {
                    "board": board,
                    "cut": cut,
                    "kept": c.get("kept"),
                    "best_horizon": c.get("best_horizon"),
                    "winrate": c.get("winrate"),
                    "mag": c.get("mag"),
                    "delta_wr": c.get("delta_wr"),
                    "baseline_wr": c.get("baseline_wr"),
                    "n": c.get("n"),
                }
            )
    return pd.DataFrame(rows, columns=_CONCL_COLS)


def cut_summary(d: dict, board: str, cut: str) -> dict:
    """单板块单 cut 的结论指标 (指标卡用)."""
    bc = _conclusion_boards(d).get(board)
    cuts = bc.get("cuts") if isinstance(bc, dict) else None
    if not isinstance(cuts, dict):
        return {}
    v = cuts.get(cut)
    return v if isinstance(v, dict) else {}


def parse_board_overview(d: dict, board: str) -> dict:
    """板块概览: 结论 cuts + criteria + 窗口 + 改进建议 (指标卡/明细用)."""
    out = {"board": board}
    bc = _conclusion_boards(d).get(board)
    if isinstance(bc, dict):
        for k in ("label", "latest", "stale", "improvements", "cuts"):
            if k in bc:
                out[k] = bc[k]
    b = _boards(d).get(board)
    if isinstance(b, dict):
        for k in ("criteria", "rows", "stocks", "latest"):
            if k in b:
                out.setdefault(k, b[k])
    return out


def parse_per_horizon(d: dict, board: str, cut: str) -> pd.DataFrame:
    """板块 merged.{cut} 主 OOS 窗四视界指标 + 基线 (胜率/幅度 bar 图用)."""
    merged = _base(_boards(d).get(board), "merged")
    node = _base(merged, cut)
    oos6 = _pick_oos_window(_base(node, "oos"))
    ph = _base(oos6, "per_horizon")
    if not ph:
        return pd.DataFrame(columns=_PH_COLS)
    rows = []
    for h in HORIZONS:
        v = _base(ph, h)
        base = _base(v, "baseline")
        rows.append(
            {
                "horizon": h,
                "cut": cut,
                "winrate": v.get("winrate"),
                "mag": v.get("mag"),
                "n": v.get("n"),
                "ok": v.get("ok"),
                "base_winrate": base.get("winrate"),
            }
        )
    return pd.DataFrame(rows, columns=_PH_COLS)


def parse_systems(d: dict, board: str) -> pd.DataFrame:
    """板块三系统主 OOS 窗 primary 四视界指标 (系统对比表/图)."""
    systems = _base(_boards(d).get(board), "systems")
    rows = []
    for name, s in systems.items():
        if not isinstance(s, dict):
            continue
        oos6 = _pick_oos_window(_base(s, "oos"))
        prim = _base(oos6, "primary")
        ph = _base(prim, "per_horizon")
        if not ph:
            continue
        for h in HORIZONS:
            v = _base(ph, h)
            base = _base(v, "baseline")
            rows.append(
                {
                    "system": name,
                    "horizon": h,
                    "winrate": v.get("winrate"),
                    "mag": v.get("mag"),
                    "n": v.get("n"),
                    "ok": v.get("ok"),
                    "base_winrate": base.get("winrate"),
                }
            )
    return pd.DataFrame(rows, columns=_SYS_COLS)


def parse_picks(d: dict, board: str) -> pd.DataFrame:
    """last_days 展平 → 逐日入选个股 (date/system/rk/symbol/score/mfe_*)."""
    ld = _base(_boards(d).get(board), "last_days")
    days = ld.get("days")
    if not isinstance(days, list):
        return pd.DataFrame(columns=_PICKS_COLS)
    rows = []
    for day in days:
        if not isinstance(day, dict):
            continue
        date = day.get("date")
        for key, val in day.items():
            if key == "date" or not isinstance(val, dict):
                continue
            picks = val.get("picks")
            if not isinstance(picks, list):
                continue
            for pk in picks:
                if not isinstance(pk, dict):
                    continue
                rows.append(
                    {
                        "date": date,
                        "system": key,
                        "rk": pk.get("rk"),
                        "symbol": pk.get("symbol"),
                        "score": pk.get("score"),
                        "mfe_3d": pk.get("mfe_3d"),
                        "mfe_5d": pk.get("mfe_5d"),
                        "mfe_10d": pk.get("mfe_10d"),
                    }
                )
    return pd.DataFrame(rows, columns=_PICKS_COLS)
