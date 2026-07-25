"""
D.10 B/C 档回测裁决协议 (PIPELINE1_V3.8 附录D.10, 裁决标准事前冻结, 严禁看结果选边)
====================================================================================
核心风险: C 档 (1只×100%) 的劣势藏在罕见尾部事件中, 回测样本内大跳空次数有限,
裸回测几乎必然偏袒 C. 裁决必须包含以下反偏袒补丁:

  补丁1 尾部压力测试 (强制注入): 每 60 交易日注入 1 次隔夜跳空 -10%
    (主板跌停无法止损), 每 120 交易日注入 1 次 -15% (流动性危机);
    B/C 同位置注入 (B 档 75% 仓位缓冲), 对比注入后回撤分布.
  补丁2 最坏交易审计: 导出最差 20 笔, 逐笔确认是否含"止损无法按 -4% 执行"
    的情景 (跳空成交/跌停顺延), 按真实可执行价重算.
  补丁3 分时段稳健性: 2018熊市 / 2020.2疫情 / 2022 / 2024.1流动性危机 四段
    分别统计, 任一段最大回撤 > 15% (停机线) → C 直接出局.
  补丁4 回测区间可选规范: 裁决类对比只准用预置区间; 自定义区间仅限调试
    打 exploratory 标记; 全部运行参数与结果入 WORM 日志 (反数据窥探账本).

裁决规则 (全部满足才选 C, 任一不满足选 B):
  C 的 GT-Score (含注入) > B 的 GT-Score / C 注入后最大回撤 < 15% /
  四个分时段均无破产月份 / 最坏 20 笔审计无可执行性失真.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DD_LIMIT = 0.15  # 停机线 (D.3, 补丁3 出局线)
HARD_STOP = -0.04  # D.3 硬止损 (补丁2 可执行性基准)
STOP_TOLERANCE = 0.005  # 可执行容差: 实际亏损超 -4.5% 视为无法按止损执行
# 补丁1 注入参数
GAP_1_EVERY, GAP_1_SIZE = 60, -0.10  # 每60日 隔夜跳空-10% (主板跌停)
GAP_2_EVERY, GAP_2_SIZE = 120, -0.15  # 每120日 跳空-15% (流动性危机)

# 补丁4 预置标准区间 (裁决类对比只准用这些)
PRESET_RANGES: dict[str, tuple[str | None, str | None]] = {
    "all": (None, None),
    "bear_2018": ("2018-01-01", "2018-12-31"),
    "covid_2020": ("2020-01-20", "2020-04-30"),  # 2020.2 疫情开盘
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "liquidity_2024": ("2024-01-01", "2024-03-31"),  # 2024.1 流动性危机
}
CRISIS_SEGMENTS = ("bear_2018", "covid_2020", "bear_2022", "liquidity_2024")


# ============================================================
# 补丁4: 回测区间规范 (防 cherry-picking)
# ============================================================
def resolve_range(
    preset: str | None = None,
    custom: tuple[str, str] | None = None,
    recent_years: int | None = None,
) -> dict:
    """解析回测区间. 裁决类对比只准用预置区间; 自定义打 exploratory 标记.

    Returns:
        {'start', 'end', 'preset', 'exploratory'} exploratory=True → 不得作裁决依据.
    """
    if custom is not None:
        logger.warning("自定义回测区间 %s — exploratory 标记, 不得作裁决依据", custom)
        return {
            "start": custom[0],
            "end": custom[1],
            "preset": None,
            "exploratory": True,
        }
    if recent_years is not None:  # "近1年" 预置档
        end = pd.Timestamp.today().normalize()
        start = end - pd.DateOffset(years=recent_years)
        return {
            "start": str(start)[:10],
            "end": str(end)[:10],
            "preset": f"recent_{recent_years}y",
            "exploratory": False,
        }
    key = preset or "all"
    if key not in PRESET_RANGES:
        raise KeyError(f"未知预置区间: {key} (可选: {list(PRESET_RANGES)})")
    start, end = PRESET_RANGES[key]
    return {"start": start, "end": end, "preset": key, "exploratory": False}


# ============================================================
# 补丁1: 尾部压力测试 (强制注入)
# ============================================================
def inject_tail_shocks(
    nav: pd.Series, position_cap: float, seed: int = 42
) -> pd.Series:
    """对净值曲线注入隔夜跳空情景 (非依赖样本自然出现).

    每 60 个交易日随机注入 1 次跳空 -10%×position_cap (满仓 C 档吃满 -10%,
    B 档 75% 仓位吃 -7.5%); 每 120 个交易日注入 1 次 -15%×position_cap.

    Args:
        nav: 日净值曲线 (index 升序)
        position_cap: 档位单票仓位 (B=0.75 / C=1.00)
        seed: 随机种子 (可复现, 量化铁律)
    """
    rng = np.random.default_rng(seed)
    out = nav.astype(float).copy()
    n = len(out)
    shock_days = set()
    for anchor in range(GAP_1_EVERY, n, GAP_1_EVERY):
        shock_days.add(min(anchor + int(rng.integers(0, 10)), n - 1))
    for anchor in range(GAP_2_EVERY, n, GAP_2_EVERY):
        shock_days.add(min(anchor + int(rng.integers(0, 10)), n - 1))
    for i in sorted(shock_days):
        gap = GAP_2_SIZE if (i % GAP_2_EVERY) < 10 else GAP_1_SIZE
        loss = 1 + gap * position_cap
        out.iloc[i:] = out.iloc[i:] * loss  # 跳空永久冲击后续净值
    logger.warning(
        "补丁1 尾部注入: %d 次跳空 (cap=%.0f%%)", len(shock_days), position_cap * 100
    )
    return out


# ============================================================
# 补丁2: 最坏交易审计
# ============================================================
def worst_trades_audit(
    trades: pd.DataFrame, n: int = 20, stop: float = HARD_STOP
) -> dict:
    """导出最差 N 笔, 标记"止损无法按 -4% 执行"的情景 (跳空成交/跌停顺延).

    trades: 需含 pnl 列 (卖出交易的净盈亏).
    Returns:
        {'worst': DataFrame, 'n_unexecutable': int, 'executable_distortion': bool}
        executable_distortion=True → 存在可执行性失真, 按真实可执行价重算后才能裁决.
    """
    sells = trades[trades["pnl"].notna()].sort_values("pnl").head(n)
    unexec = sells[sells["pnl"] < stop - STOP_TOLERANCE]
    distortion = len(unexec) > 0
    if distortion:
        logger.error(
            "补丁2 最坏交易审计: %d/%d 笔亏损超止损可执行范围 (%.1f%%), "
            "含跳空/跌停顺延, 需按真实可执行价重算",
            len(unexec),
            len(sells),
            (stop - STOP_TOLERANCE) * 100,
        )
    return {
        "worst": sells,
        "n_unexecutable": len(unexec),
        "executable_distortion": distortion,
    }


# ============================================================
# 补丁3: 分时段稳健性
# ============================================================
def segment_robustness(
    segment_max_dd: dict[str, float], dd_limit: float = DD_LIMIT
) -> dict:
    """四段危机期分别统计, 任一段最大回撤 > 15% → C 直接出局.

    Args:
        segment_max_dd: {段名: 该段最大回撤 (负数)}, 段名须覆盖 CRISIS_SEGMENTS.
    """
    failures = {
        seg: dd
        for seg, dd in segment_max_dd.items()
        if seg in CRISIS_SEGMENTS and dd < -dd_limit
    }
    ok = len(failures) == 0 and all(seg in segment_max_dd for seg in CRISIS_SEGMENTS)
    if failures:
        logger.error(
            "补丁3 分时段稳健性: %s 回撤破停机线, C 档出局",
            {k: f"{v:.1%}" for k, v in failures.items()},
        )
    return {"pass": ok, "failed_segments": failures}


# ============================================================
# D.10 总裁决
# ============================================================
def adjudicate_b_vs_c(
    gt_b: float,
    gt_c: float,
    c_max_dd_injected: float,
    c_segment_dd: dict[str, float],
    c_worst_audit: dict,
) -> dict:
    """B/C 裁决 (全部满足才选 C, 任一不满足选 B; 标准事前冻结)."""
    checks = {
        "gt_score_c_wins": gt_c > gt_b,
        "c_injected_dd_ok": c_max_dd_injected > -DD_LIMIT,
        "c_segments_ok": segment_robustness(c_segment_dd)["pass"],
        "c_no_execution_distortion": not c_worst_audit["executable_distortion"],
    }
    choose_c = all(checks.values())
    logger.warning(
        "D.10 B/C 裁决: %s (checks=%s)",
        "选 C (1只×100%)" if choose_c else "选 B (1只×75%)",
        checks,
    )
    return {"choose": "C" if choose_c else "B", "checks": checks}


# ============================================================
# 回测清单失效条件模拟 (V3.8 §四 bis: 回测必须模拟清单失效条件)
# ============================================================
def simulate_invalidations(daily_lists: dict, invalidated_by_date: dict) -> dict:
    """把失效条件 (跳空/跌停/板块/公告#5) 应用到每日清单字典.

    Args:
        daily_lists: {date: DataFrame(symbol, ...)} 原始清单
        invalidated_by_date: {date: {symbol: 原因}} 失效票 (如
            AnnouncementFactor.list_invalidation 的输出, 或盘中条件 1-4 汇总)
    Returns:
        过滤后的 {date: DataFrame}; 失效票剔除留痕 (日志), 不静默.
    """
    out = {}
    for date, lst in daily_lists.items():
        bad = (invalidated_by_date or {}).get(date, {})
        if bad and len(lst):
            n_before = len(lst)
            lst = lst[~lst["symbol"].isin(bad)]
            if len(lst) < n_before:
                logger.warning(
                    "回测失效条件模拟: %s 剔除 %d 只 (%s)",
                    date,
                    n_before - len(lst),
                    sorted(bad)[:5],
                )
        out[date] = lst
    return out


# ============================================================
# 补丁4: WORM 反数据窥探账本
# ============================================================
class BacktestJournal:
    """全部回测运行参数与结果入 WORM 日志, 失败区间一并保留."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, tag: str, params: dict, metrics: dict) -> str:
        rec = {"tag": tag, "params": params, "metrics": metrics}
        path = os.path.join(self.log_dir, "backtest_runs.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return path
