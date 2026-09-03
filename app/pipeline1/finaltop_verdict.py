"""终榜口径 TOP10 第二票判词 (2026-09-02) — 纯函数.

读 tmp_t/_dual_pkg_finaltop_compare.py 回放 JSON 的 payload[board]["delta_B_vs_A"]
(B=新包, A=旧 current, 配对 Δnet3/日), 套 09-02 定案判据:

  全窗 ≥ tol_full 且 前半/后半 ≥ tol_half 且 配对胜率 ≥ win_rate_min

起因: 裸头闸错杀实锤 — main_20260902 裸头 -0.31pp 判 FAIL, 终榜口径
+0.55pp/日 胜率 62.5% (WORM _dual_pkg_finaltop_compare_20260902_201413.json)。
ok=False = 无判词 (可比日不足/缺 delta), 调用方 fail-safe 保留旧包。
"""

from __future__ import annotations


def verdict_from_payload(
    payload: dict,
    board: str,
    *,
    tol_full: float = 0.0,
    tol_half: float = -0.005,
    win_rate_min: float = 0.5,
    min_days: int = 10,
) -> dict:
    delta = (payload.get(board) or {}).get("delta_B_vs_A")
    if not isinstance(delta, dict) or not delta:
        return {"ok": False, "pass": False, "reason": "no_delta"}
    days = int(delta.get("days", 0) or 0)
    win = int(delta.get("win_days", 0) or 0)
    lose = int(delta.get("lose_days", 0) or 0)
    if days < min_days:
        return {"ok": False, "pass": False, "reason": "insufficient_days", "days": days}
    win_rate = win / (win + lose) if (win + lose) else 0.0
    d_full = float(delta["d3_full"])
    d_h1 = float(delta["d3_h1"])
    d_h2 = float(delta["d3_h2"])
    checks = {
        "full": d_full >= tol_full,
        "half1": d_h1 >= tol_half,
        "half2": d_h2 >= tol_half,
        "win_rate": win_rate >= win_rate_min,
    }
    return {
        "ok": True,
        "pass": all(checks.values()),
        "days": days,
        "d3_full": d_full,
        "d3_h1": d_h1,
        "d3_h2": d_h2,
        "win_days": win,
        "lose_days": lose,
        "win_rate": win_rate,
        "checks": checks,
    }
