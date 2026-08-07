"""_shortlist_t5_t10.py — 今日最终短名单 (狙击 TOP-5 ∪ 融合 TOP-10) 前瞻预测输出.

数据源 = 最后一次 FULL RUN 回测产物 (用户 2026-08-05: "已有 full run 基于昨日数据
做出预测, 模块特征/一切都要保留, 不得丢失"). 本脚本**不重算特征、不重新选股**,
只读 FULL RUN 落盘产物, 给每只短名单股计算**前瞻预测** (最新 score 经 OOS 校准).

  - shortlist_main.csv / shortlist_dual.csv : 权威选股结果 (score/co_occur/rk/cut/est_wr)
  - stocks_{board}_{system}_oos.csv          : OOS 逐股 评分 + 已实现前视 MFE
    → 按 score 分位桶校准 → 每股逐视界 预期涨幅(MFE) + 达到概率 (前瞻, 非历史回看)
  - backtest.json                           : 系统级 OOS 逐视界 胜率/期望 (SUMMARY 段)

**概率口径 (2026-08-06 用户定案):** 概率 = 该股**逐股自然概率** (每股唯一真值)
= P(该股该视界已实现 MFE ≥ 固定绝对目标) via 逐股 Platt 平滑校准 — 替代旧桶级共享值
(同 score 桶内每股同值, 用户: "natural, not bulk probability"). 不是 P(MFE>0) —
那曾是 85-99% 被用户否决 "incorrectly high".

**制度自适应门 (2026-08-05 用户):** 输出哪个 (板块, 系统) 组合**不写死**, 用最新 OOS
市场数据 (固定视界净收益 label_pm) 每日判定: top-quantile 选股平均净收益 > 池基线 + margin
才保留; **判定只看主视界 (T+3)** — 2026-08-05 用户否决用 T+5/T+10 兜底 ("如果没有优势就
不要入选") → 主视界无优势组合则空仓观望, 不输出清单. 见 config REGIME_GATE.

顶部 SUMMARY = 主板+双创合并的单一集成排名清单 (rank/命中模块/预期涨幅/达到概率), 直接决策;
明细表格保留每板块 T-5/T-10 逐视界.

流程 (用户): 先给每只股 预期涨幅+达到概率 (预测), 再排名 — 绝不先排名后预测.
**排名键 (2026-08-07 定案):** 每股 pred_mag (主视界 T+3 预期幅度) 降序 — 250d OOS
证明纯幅度排名 MFE 全视界赢特征/混合排名 +3.9~10.1pp, 上涨率打平. 候选池每系统特征
TOP-30 加宽 (CAND_POOL_N), 再按 pred_mag 取每板块 TOP-10; score_w 降为平局裁决/展示.

**主视界 (2026-08-05 用户定案):** T+3 (短持 3 天). 排名权重 3d=0.40 最高 (2d+3d 合计 0.65),
入选门 = **T+2/T+3 联合门** (2026-08-07 用户: "考虑 T+2,T+3 一起"; 见 config
SHORTLIST_SCORE.select_gate): 保留 ⇔ T+3 预期涨幅>0, **或** T+2 强看涨(>t2_min)且 T+3 未深度转负
(>t3_floor) — T+3 仍为首要, 但 T+2 强烈看涨的股不会因 T+3 边际转负被整只剔除.
平局裁决用 T+3. 四视界 T+2/3/5/10 预期涨幅+达到概率仍全部展示 (T+10 降为参考视界).

用法: python scripts/_shortlist_t5_t10.py [YYYYMMDD 选股日, 默认=full run 短名单日期]
可选: 传 symbol 列表 (空格分隔) 查看这些股今日预测 (未入选股模型今日无打分, 无预测).
输出: STOCK_LIST_DIR/parallel_shortlist_<date>__<module>.csv + STOCK LIST <date>__<module>.docx/.xlsx
      (module = current_meta.json 模型版本 tag, 供回归按模块分组评估)
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.pipeline1.label_engine import COST, slippage_tier
from app.pipeline_parallel.config import FUSION, HORIZONS, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import (
    DATA_DIR,
    DATA_OTHERS_DIR,
    REGIME_GATE,
    SHORTLIST_SCORE,
    STOCK_LIST_DIR,
)

try:
    from docx import Document
    from docx.shared import Pt
except Exception:  # python-docx 未安装时跳过 docx 输出
    Document = None

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
except Exception:  # openpyxl 未安装时跳过 xlsx 输出
    Workbook = None

# FULL RUN 目录: 默认自动解析最新一次并行回测 (含 shortlist_main.csv 的最晚 ts 子目录),
# 供逐日连续运行 (EMA 平滑需各日 raw 历史); 找不到才回退硬编码默认.
_FULLRUN_HARD = DATA_OTHERS_DIR / "BACKTESTING RESULT" / "20260806_144240"


def _latest_fullrun_dir() -> Path:
    base = DATA_OTHERS_DIR / "BACKTESTING RESULT"
    try:
        cands = sorted(
            (p for p in base.glob("*/") if (p / "shortlist_main.csv").exists()),
            key=lambda p: p.name,
            reverse=True,
        )
    except OSError:
        return _FULLRUN_HARD
    return cands[0] if cands else _FULLRUN_HARD


FULLRUN_DIR = _latest_fullrun_dir()
print(f"[fullrun] 使用 {FULLRUN_DIR}", flush=True)
BOARD_LABEL = {"main": "主板", "dual": "双创(创业板+科创板)"}
# 集成清单逐视界展示顺序 (时间序; HORIZONS = ("3d","2d","5d","10d"))
HORIZON_ORDER = ("2d", "3d", "5d", "10d")
# OOS score→前视涨幅校准 (前瞻预测): 每股独立连续校准 (LinearRegression score→mfe),
# 替代 CAL_BINS 桶级共享值 (2026-08-06 用户: 价格预测必须每股独立, 非 bulk);
# CAL_MIN_N 保留用于 load_system_stats/regime_gate 最少样本门.
CAL_MIN_N = 5
# [2026-08-06] 达到概率改逐股自然概率 (用户: "natural, not bulk probability"):
# 口径 = P(该股该视界达到固定绝对涨幅) — 每股唯一真值 (score 单调略增, 非桶内共享).
ABS_TARGET = {"2d": 0.02, "3d": 0.03, "5d": 0.04, "10d": 0.06}
# [2026-08-06] 价格预测=每股独立, 只用该股自己最近 ~6 个月历史 (用户定案):
# 每股取自己最近 PER_STOCK_WINDOW 交易日 (score→mfe) 拟合, 不足 PER_STOCK_MIN_N 回退横截面.
PER_STOCK_WINDOW = 130
PER_STOCK_MIN_N = 30
# [2026-08-06] 预测稳定性 (诊断 _diag_pred_decomp: 校准器逐日漂移 + score 逐日抖动同量级):
#  1) per-stock 斜率向横截面收缩 (empirical-Bayes partial pooling): λ=n/(n+SHRINK_KAPPA),
#     保持个股质心 (intercept=ȳ−slope·x̄) → 面板逐日滚动重拟合时 Δslope 大幅降低.
#  2) 输出级时间平滑 (EMA): 同日股用近 SMOOTH_K 个可用交易日 raw 预测的衰减加权均值
#     w_k=α·(1-α)^k (k=0=今日), gap-robust (缺日跳过) → 直接抑制相邻交易日预测/概率剧变.
SHRINK_KAPPA = 40
SMOOTH_ENABLED = True
SMOOTH_ALPHA = 0.35
SMOOTH_K = 12
# [2026-08-07] 短名单排名定案: 按每股 pred_mag(主视界幅度) 排名, 替代原 score_w 混合排名.
# 250d OOS (scripts/_diag_parallel_rank_compare.py + _diag_rank_strategies.py) 证明:
# 纯 pred_mag 排名 MFE 全视界赢特征排名 +3.9~10.1pp, 上涨率打平; 混合/两段式全不如纯.
# 原短名单仅 ~15 只 (特征 TOP-5∪TOP-10), pred_mag 在内重排近乎无效 → 需先把候选池加宽.
CAND_POOL_N = 30  # 候选池宽度: 每系统特征分 TOP-N → 再按 pred_mag 取 TOP-10
CAND_RANK_KEY = "pred_mag_3d"  # 排名键 = 主视界(T+3)每股预期幅度


def load_system_stats(records: dict) -> dict:
    """OOS 逐股记录 → {(board, system): {h: {"mag", "hit", "n"}}} (SUMMARY 段, 系统级汇总).

    系统级单一值 (非个股值): mag=系统期望涨幅(均值), hit=P(已实现 MFE ≥ 系统期望) —
    不是 P(MFE>0) 的虚高系统胜率 (旧 82-94% 被用户否决 "not convincing")."""
    stats = {}
    for (board, key), rec in records.items():
        if key == "both":
            continue
        per = {}
        for h in HORIZONS:
            v = rec[f"mfe_{h}"].dropna()
            if len(v) < CAL_MIN_N:
                continue
            exp = float(v.mean())
            per[h] = {"mag": exp, "hit": float((v >= exp).mean()), "n": int(len(v))}
        stats[(board, key)] = per
    return stats


# 面板固定视界净收益列 (现实"持有到收盘"口径, 制度门用): label_pm_{2,3,5,10}d_net → pm_{h}
PM_LABEL = {h: f"label_pm_{h[:-1]}d_net" for h in HORIZONS}
PM_RENAME = {v: f"pm_{h}" for h, v in PM_LABEL.items()}


def _norm_oos(df: pd.DataFrame) -> pd.DataFrame:
    """OOS 逐股记录 → 留 symbol/date + score + 各视界已实现 MFE (前视未实现行留 NaN)."""
    d = df.rename(columns={f"label_mfe_{h}_net": f"mfe_{h}" for h in HORIZONS})
    keep = ["symbol", "date", "score"] + [f"mfe_{h}" for h in HORIZONS]
    return d[keep]


def _load_panel_pm(board: str) -> pd.DataFrame:
    """投影 3y 诊断面板的 label_pm 列 (symbol+date+四视界净收益), 供制度门并入 OOS 记录."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    t = pq.read_table(str(fp), columns=["symbol", "date"] + list(PM_LABEL.values()))
    t = t.to_pandas().rename(columns=PM_RENAME)
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    return t


def load_oos_records() -> dict[tuple[str, str], pd.DataFrame]:
    """FULL RUN 的 OOS 逐股 评分 + 已实现前视 MFE + 面板固定视界净收益 label_pm
    → {(board, key): DataFrame}. key: "sniper" / "fusion" / "both"(共现=两系统合并)."""
    records = {}
    for board in ("main", "dual"):
        pm = _load_panel_pm(board)
        sn = _norm_oos(
            pd.read_csv(
                FULLRUN_DIR / f"stocks_{board}_sniper_oos.csv", dtype={"symbol": str}
            )
        )
        fu = _norm_oos(
            pd.read_csv(
                FULLRUN_DIR / f"stocks_{board}_fusion_oos.csv", dtype={"symbol": str}
            )
        )
        records[(board, "sniper")] = sn.merge(pm, on=["symbol", "date"], how="left")
        records[(board, "fusion")] = fu.merge(pm, on=["symbol", "date"], how="left")
        records[(board, "both")] = pd.concat(
            [records[(board, "sniper")], records[(board, "fusion")]], ignore_index=True
        )
    return records


def regime_gate(records: dict) -> dict:
    """制度自适应门: 用最新 OOS 市场数据判定每个 (board, system) 组合今日是否保留.

    每组合: base_ev_h = 池内全部行平均 label_pm (闭眼全买池基准);
    top_ev_h = 高分 top_quantile 选股平均 label_pm; 保留 ⇔ top_ev > base_ev + margin
    **只看主视界** (REGIME_GATE.primary_horizon) — 不许长视界兜底 (2026-08-05 用户:
    "如果没有优势就不要入选"). 全视界 alpha 仍进 per 供 SUMMARY 展示. 返回
    {(board, sys): {"active": bool, "per": {h: {...}}}}.
    """
    cfg = REGIME_GATE
    q, margin = cfg["top_quantile"], cfg["margin"]
    primary = cfg["primary_horizon"]
    out = {}
    for board in ("main", "dual"):
        for sname in ("sniper", "fusion"):
            rec = records.get((board, sname))
            if rec is None or len(rec) == 0:
                continue
            s = rec["score"]
            thr = s.quantile(1.0 - q)  # top q-quantile (高分选股)
            top_mask = s >= thr
            per = {}
            for h in cfg["horizons"]:
                col = f"pm_{h}"
                if col not in rec:
                    continue
                v = rec[col].dropna()
                if len(v) < CAL_MIN_N:
                    continue
                tv = rec.loc[top_mask, col].dropna()
                if len(tv) < CAL_MIN_N:
                    continue
                base_ev, top_ev = float(v.mean()), float(tv.mean())
                per[h] = {
                    "base_ev": base_ev,
                    "top_ev": top_ev,
                    "top_wr": float((tv > 0).mean()),
                    "alpha": top_ev - base_ev,
                }
            active = primary in per and per[primary]["alpha"] > margin
            out[(board, sname)] = {"active": active, "per": per}
    return out


def _row_active(row, gate: dict) -> bool:
    """该短名单行 (board, systems) 是否命中今日保留的组合; 共现股=任一所涉系统保留即算."""
    combos = ("sniper", "fusion") if row["systems"] == "both" else (row["systems"],)
    return any(gate.get((row["board"], s), {}).get("active", False) for s in combos)


def gate_fallback(gate: dict) -> tuple | None:
    """REGIME_GATE.fallback = "best" 时: 全组合未过线 → 返回 best-alpha 组合.

    2026-08-05 用户否决 "best" 兜底 ("如果没有优势就不要入选") — fallback="none"
    时恒返回 None → 主视界无优势即空仓观望, 不输出清单."""
    if REGIME_GATE.get("fallback") != "best":
        return None
    best, best_alpha = None, -1e9
    for (b, s), g in gate.items():
        if not g["per"]:
            continue
        a = max(p["alpha"] for p in g["per"].values())
        if a > best_alpha:
            best, best_alpha = (b, s), a
    return best


def fmt_regime(gate: dict) -> list[str]:
    """制度门判定摘要 (SUMMARY 顶部): 每组合 top vs 池基线 alpha + 保留/剔除."""
    cfg = REGIME_GATE
    lines = [
        "",
        f"[制度自适应门] 板块×系统保留判定 — 基于最新 OOS 市场数据 "
        f"(固定视界净收益 label_pm, top{int(cfg['top_quantile'] * 100)}% vs 池基线, "
        f"门槛 +{cfg['margin'] * 100:.0f}pp, 判定只看主视界 "
        f"T+{cfg['primary_horizon'][:-1]}):",
    ]
    active = []
    for b, s in (
        ("main", "sniper"),
        ("main", "fusion"),
        ("dual", "sniper"),
        ("dual", "fusion"),
    ):
        g = gate.get((b, s))
        if not g or not g["per"]:
            continue
        seg = [
            f"T+{h[:-1]} top {p['top_ev']:+.2%}(wr {p['top_wr']:.0%}) "
            f"vs 基 {p['base_ev']:+.2%} α{p['alpha']:+.2%}"
            for h, p in g["per"].items()
        ]
        if g["active"]:
            active.append(f"{b}/{s}")
        lines.append(
            f"  {b}/{s:<9}: {'; '.join(seg)}  → {'保留' if g['active'] else '剔除'}"
        )
    if not active:
        note = (
            "无 (主视界无优势组合, 空仓观望)"
            if REGIME_GATE.get("fallback") != "best"
            else "无 (弱市兜底 best-alpha)"
        )
    else:
        note = ", ".join(active)
    lines.append(f"  今日输出组合: {note}")
    return lines


def _add_mfe(df: pd.DataFrame) -> pd.DataFrame:
    """按生产口径 (backtest.add_mfe_labels) 补算四视界 MFE 净标签:
    窗口内最高价 / 买入价 - 1 - cost; 尾段缺未来价 → NaN."""
    horizons = (2, 3, 5, 10)
    g = df.groupby("symbol", sort=False)
    exec_px = g["close_hfq"].shift(-1)
    max_off = max(horizons) + 1
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, max_off + 1)],
        axis=1,
        keys=range(2, max_off + 1),
    )
    slip = df["adv20"].map(slippage_tier)
    cost_total = COST + 2 * slip
    for k in horizons:
        peak = shifts.loc[:, 2 : k + 1].max(axis=1, skipna=False)
        df[f"mfe_{k}d"] = peak / exec_px - 1 - cost_total
    return df


_PANEL_CACHE: dict = {"ready": False, "data": {}}  # _panel_per_stock 结果缓存 (被候选池+校准各调一次)


def _panel_per_stock() -> dict[tuple[str, str], pd.DataFrame]:
    """3y 面板 → 每股最近 ~PER_STOCK_WINDOW 交易日日频 (score, mfe) 序列.

    [2026-08-06 用户: 价格预测必须每股独立, 只用该股自己最近 6 个月历史]
    短名单 OOS 文件只含"每日被选股" → 每股仅零星几行, 无法做每股回归;
    面板 (每 stock × 每交易日) 才有逐股日频. 在此按生产口径重算:
      - score = 与生产一致的 6 特征截面分位 (pv_corr_5 面板同样缺列, 自动跳过)
      - mfe   = _add_mfe 口径 (窗口最高价 / 买入价 - 1 - cost, 净)
    返回 {(board, key): DataFrame[symbol, date, score, mfe_2d..mfe_10d]}.
    """
    out: dict[tuple[str, str], pd.DataFrame] = {}
    need = ["symbol", "date", "close_hfq", "high_hfq", "adv20"] + [
        c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"
    ]
    if _PANEL_CACHE.get("ready"):
        return _PANEL_CACHE["data"]
    for board in ("main", "dual"):
        fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
        dates = pd.to_datetime(
            pq.read_table(str(fp), columns=["date"]).to_pandas()["date"]
        )
        uniq = np.unique(dates.values)
        if len(uniq) < PER_STOCK_WINDOW + 12:
            continue
        cutoff = uniq[-(PER_STOCK_WINDOW + 12)]
        t = pq.read_table(
            str(fp), columns=need, filters=[("date", ">=", cutoff)]
        ).to_pandas()
        if t.empty:
            continue
        t["symbol"] = t["symbol"].astype(str)
        t = t.sort_values(["symbol", "date"]).reset_index(drop=True)
        t = _add_mfe(t)
        fr_sn = t[["symbol", "date"]].copy()
        fr_sn["score"] = pool_score(t, SNIPER.pool)
        fr_fu = t[["symbol", "date"]].copy()
        fr_fu["score"] = pool_score(t, FUSION.pool)
        for h in HORIZONS:
            fr_sn[f"mfe_{h}"] = t[f"mfe_{h}"]
            fr_fu[f"mfe_{h}"] = t[f"mfe_{h}"]
        out[(board, "sniper")] = fr_sn
        out[(board, "fusion")] = fr_fu
        # "both" = 合并短名单口径 (build_merged_shortlist: score=max 两系统分位分) →
        # 共现行也启用逐股校准, 不再回退横截面
        fr_bt = fr_sn.copy()
        fr_bt["score"] = np.maximum(fr_sn["score"], fr_fu["score"])
        out[(board, "both")] = fr_bt
    _PANEL_CACHE["ready"], _PANEL_CACHE["data"] = True, out
    return out


def expand_candidates(full: pd.DataFrame) -> pd.DataFrame:
    """FULL RUN 短名单加宽: 每系统特征分 TOP-{CAND_POOL_N} 候选 ∪ 原短名单.

    2026-08-07 定案: 250d OOS 证明"每股 pred_mag(幅度) 排名"胜过"特征分排名"
    (MFE 全视界 +3.9~10.1pp, 上涨率打平). 原短名单仅 ~15 只 (特征 TOP-5∪TOP-10),
    pred_mag 在其内重排近乎无效 → 用面板最新截面把候选池加宽到每系统特征 TOP-30,
    再由 rank_and_truncate 按 pred_mag 取 TOP-10. 沿用 _panel_per_stock 与生产一致的
    截面分 (不重算特征); 原"不重选股"约定被本定案取代 (选股宽进→预测细排).
    """
    panel = _panel_per_stock()
    rows = []
    for board in ("main", "dual"):
        sn = panel.get((board, "sniper"))
        fu = panel.get((board, "fusion"))
        if sn is None or fu is None or sn.empty or fu.empty:
            continue
        last = max(sn["date"].max(), fu["date"].max())

        def _top(pf: pd.DataFrame, last=last) -> pd.DataFrame:
            cs = pf[pf["date"] == last][["symbol", "score"]].dropna()
            return cs.sort_values("score", ascending=False).head(CAND_POOL_N)

        sn30, fu30 = _top(sn), _top(fu)
        sset, fset = set(sn30["symbol"]), set(fu30["symbol"])
        sscore = sn30.set_index("symbol")["score"]
        fscore = fu30.set_index("symbol")["score"]
        for sym in sset | fset:
            in_s, in_f = sym in sset, sym in fset
            rows.append(
                {
                    "date": str(pd.Timestamp(last).date()),
                    "board": board,
                    "symbol": sym,
                    "systems": (
                        "fusion+sniper" if in_s and in_f else ("sniper" if in_s else "fusion")
                    ),
                    "score": float(
                        max(sscore.get(sym, -1.0), fscore.get(sym, -1.0))
                    ),
                    "co_occur": bool(in_s and in_f),
                    "rk": 0,
                    "cut": "T-10",
                }
            )
    cands = pd.DataFrame(rows)
    base = ["date", "board", "symbol", "systems", "score", "co_occur", "rk", "cut"]
    keep = full[base].copy()
    keep["date"] = keep["date"].astype(str).str.slice(0, 10)
    cands["date"] = cands["date"].astype(str).str.slice(0, 10)
    merged = pd.concat([cands, keep], ignore_index=True)
    merged = merged.drop_duplicates(subset=["board", "symbol"], keep="first")
    print(
        f"[cand] 候选池: 原短名单 {len(full):,} → 加宽 {len(merged):,} "
        f"(面板候选 {len(cands):,}, 每系统特征 TOP-{CAND_POOL_N})",
        flush=True,
    )
    return merged


class _PooledReg:
    """单变量线性映射 (收缩后斜率 + 质心截距), 提供与 LinearRegression 一致的 predict.

    slope = λ·slope_per + (1-λ)·slope_cross (λ=n/(n+SHRINK_KAPPA)),
    intercept = ȳ − slope·x̄ → 拟合线过该股 (score, mfe) 质心.
    """

    __slots__ = ("coef_", "intercept_")

    def __init__(self, slope: float, intercept: float):
        self.coef_ = np.array([slope])
        self.intercept_ = intercept

    def predict(self, X) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def _fit_calibrators(records: dict) -> dict[tuple, dict]:
    """[2026-08-06] 价格预测=每股独立, 只用该股自己最近 6 个月历史 (用户定案):
    - mag: 每股自己最近 ~PER_STOCK_WINDOW 交易日拟合 LinearRegression (score→mfe),
           数据来自 3y 面板逐股日频 (_panel_per_stock), 非 OOS 稀疏清单;
           该股样本 ≥PER_STOCK_MIN_N 才启用; 不足回退板块×系统横截面回归
    - prob: 横截面 Platt (score → P(mfe_h ≥ ABS_TARGET[h])), 每股分数→唯一概率
    返回 {(board, key, h): {"prob": lr, "mag_cross": reg, "per": {symbol: reg}}}."""
    cals = {}
    panel = _panel_per_stock()
    for (board, key), rec in records.items():
        for h in HORIZONS:
            col = f"mfe_{h}"
            sub = rec[["score", col]].dropna()
            if len(sub) < 20:
                continue
            prob = LogisticRegression()
            prob.fit(
                sub[["score"]].to_numpy(),
                (sub[col] >= ABS_TARGET[h]).astype(int).to_numpy(),
            )
            cross = LinearRegression()
            cross.fit(sub[["score"]].to_numpy(), sub[col].to_numpy())
            per: dict[str, _PooledReg] = {}
            pf = panel.get((board, key))
            if pf is not None:
                for sym, g in pf.groupby("symbol"):
                    gg = (
                        g.dropna(subset=["score", col])
                        .sort_values("date")
                        .tail(PER_STOCK_WINDOW)
                    )
                    if len(gg) >= PER_STOCK_MIN_N:
                        x = gg[["score"]].to_numpy()
                        y = gg[col].to_numpy()
                        raw = LinearRegression().fit(x, y)
                        lam = len(gg) / (len(gg) + SHRINK_KAPPA)
                        slope = lam * float(raw.coef_[0]) + (1 - lam) * float(
                            cross.coef_[0]
                        )
                        per[sym] = _PooledReg(
                            slope, float(y.mean()) - slope * float(x.mean())
                        )
            cals[(board, key, h)] = {"prob": prob, "mag_cross": cross, "per": per}
    return cals


def calibrate(
    records: dict, board: str, key: str, symbol: str, score: float, cals: dict
) -> dict[str, tuple]:
    """最新 score → 逐视界 (预期涨幅, 自然达到概率).

    每股独立 (2026-08-06 用户: 价格预测必须每股独立, 只用该股自己最近 6 个月历史):
    - 预期涨幅 = 该股 score 经该股自己最近 ~6 个月 (score→mfe) 线性回归的独立值,
      该股样本不足时回退板块×系统横截面回归
    - 概率     = P(该股该视界达到固定绝对涨幅 ABS_TARGET) via 横截面 Platt (每股分数→唯一概率)
    返回 {h: (exp_mfe, hit_prob, n)}; 无校准数据 → (NaN, NaN, 0).
    """
    out = {h: (float("nan"), float("nan"), 0) for h in HORIZONS}
    rec = records.get((board, key))
    if rec is None or len(rec) == 0:
        return out
    for h in HORIZONS:
        c = cals.get((board, key, h))
        if c is None:
            continue
        mag = c["per"].get(symbol, c["mag_cross"])
        exp = float(mag.predict(np.array([[score]]))[0])
        prob = float(c["prob"].predict_proba(np.array([[score]]))[0, 1])
        out[h] = (exp, prob, int(len(rec[f"mfe_{h}"].dropna())))
    return out


def add_oos_pred(res: pd.DataFrame, records: dict) -> pd.DataFrame:
    """每只短名单股: 用最新 score 经 OOS 校准给 逐视界 前瞻 预期涨幅(MFE)+逐股自然概率."""
    cals = _fit_calibrators(records)
    out = res.copy()
    for h in HORIZONS:
        out[f"pred_mag_{h}"], out[f"pred_prob_{h}"], out[f"pred_n_{h}"] = (
            float("nan"),
            float("nan"),
            0,
        )
    for idx, r in out.iterrows():
        key = r["systems"] if r["systems"] in ("sniper", "fusion") else "both"
        cal = calibrate(
            records, r["board"], key, str(r["symbol"]), float(r["score"]), cals
        )
        for h in HORIZONS:
            mag, prob, n = cal[h]
            out.at[idx, f"pred_mag_{h}"] = mag
            out.at[idx, f"pred_prob_{h}"] = prob
            out.at[idx, f"pred_n_{h}"] = n
    # est_wr 已移除: 系统级常量 (同系统每股同值), 且是旧 P(MFE>0) 虚高口径 (用户 2026-08-05 否决)
    first = ["date", "board", "cut", "rk", "symbol", "systems", "co_occur", "score"]
    ph = [f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob", "pred_n")]
    return out[first + ph].reset_index(drop=True)


def _module_suffix(module: str) -> str:
    return f"__{module}" if module != "na" else ""


def _persist_raw_preds(res: pd.DataFrame, sel_date: pd.Timestamp, module: str) -> None:
    """WORM: 每股当日 raw 预测 (平滑前) → parallel_preds_raw_<date>__<module>.csv.

    EMA 平滑的历史底稿; 旧文件不覆盖. 同 symbol 多 cut 行去重 (预测值相同, keep 最后一行).
    """
    if not SMOOTH_ENABLED:
        return
    os.makedirs(str(STOCK_LIST_DIR), exist_ok=True)
    cols = ["date", "board", "symbol", "systems", "score"] + [
        f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob")
    ]
    d = res[cols].copy()
    d["date"] = str(sel_date.date())
    d = d.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)
    stamp = str(sel_date.date()).replace("-", "")
    fp = STOCK_LIST_DIR / f"parallel_preds_raw_{stamp}{_module_suffix(module)}.csv"
    d.to_csv(fp, index=False)
    print(f"[raw] {fp}", flush=True)


def _load_raw_history(sel_date: pd.Timestamp, module: str) -> pd.DataFrame:
    """读 <选股日> 之前的 raw 预测文件 → 长表 (symbol, hist_date, pred_mag/prob_{h})."""
    suffix = _module_suffix(module)
    today = str(sel_date.date()).replace("-", "")
    pred_cols = [f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob")]
    frames = []
    for fp in STOCK_LIST_DIR.glob("parallel_preds_raw_*.csv"):
        m = re.match(r"parallel_preds_raw_(\d{8})(.*)\.csv$", fp.name)
        if not m:
            continue
        if suffix and m.group(2) != suffix:
            continue
        if not suffix and m.group(2):
            continue  # module=na 时不混入带模块标记的历史
        if m.group(1) >= today:
            continue  # 不含今日 (今日 raw 由调用方直接给)
        d = pd.read_csv(fp, dtype={"symbol": str})
        d = d[["symbol"] + pred_cols]
        d["hist_date"] = m.group(1)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    return long.drop_duplicates(subset=["symbol", "hist_date"], keep="last")


def ema_smooth(res: pd.DataFrame, sel_date: pd.Timestamp, module: str) -> pd.DataFrame:
    """输出级时间平滑: 每股今日 pred_mag/prob = 近 SMOOTH_K 个可用交易日 raw 的衰减加权均值.

    w_k = α·(1-α)^k (k=0=今日, 越旧越轻), 截断后归一化; 缺日跳过 (gap-robust).
    只平滑 pred_mag/pred_prob (不动 score); 无历史 → 保持 raw. 直接抑制相邻交易日预测/概率剧变.
    """
    if not SMOOTH_ENABLED:
        return res
    hist = _load_raw_history(sel_date, module)
    if hist.empty:
        return res
    weights = np.array(
        [SMOOTH_ALPHA * (1 - SMOOTH_ALPHA) ** k for k in range(SMOOTH_K)]
    )
    weights /= weights.sum()
    out = res.copy()
    for sym in out["symbol"].unique():
        h = hist[hist["symbol"] == sym].sort_values("hist_date", ascending=False)
        if h.empty:
            continue
        src = h.head(SMOOTH_K - 1)  # 最多 SMOOTH_K-1 个旧日 (今日占 k=0)
        mask = out["symbol"] == sym
        for kind in ("pred_mag", "pred_prob"):
            for hz in HORIZONS:
                col = f"{kind}_{hz}"
                today_v = out.loc[mask, col]
                if today_v.empty or not np.isfinite(float(today_v.iloc[0])):
                    continue
                pairs = [
                    (w, v)
                    for w, v in zip(
                        weights,
                        [float(today_v.iloc[0])] + [float(x) for x in src[col]],
                        strict=False,
                    )
                    if np.isfinite(v)
                ]
                if not pairs:
                    continue
                ww = np.array([p[0] for p in pairs])
                ww /= ww.sum()
                out.loc[mask, col] = float(np.dot([p[1] for p in pairs], ww))
    return out


def _sel_reason(r: pd.Series) -> str:
    p = "n/a" if pd.isna(r["pred_prob_3d"]) else f"{r['pred_prob_3d']:.0%}"
    m3 = "n/a" if pd.isna(r["pred_mag_3d"]) else f"{r['pred_mag_3d']:+.1%}"
    m2 = "n/a" if pd.isna(r["pred_mag_2d"]) else f"{r['pred_mag_2d']:+.1%}"
    return f"{r['symbol']}(T+3 {m3}/{p}, T+2 {m2})"


def select_confident(res: pd.DataFrame, prob_min: float = 0.0) -> pd.DataFrame:
    """IRON RULE (用户): 只列预测上涨股. 入选门 = T+2/T+3 联合门.

    2026-08-07 用户: "考虑 T+2,T+3 一起" (301326 08-05 教训: 该股 raw score dual 第 1,
    但 T+3 预期涨幅边际转负即被整只剔除, 实际 2 天 +12%). 联合门 (config
    SHORTLIST_SCORE.select_gate):
      保留 ⇔ (T+3 > t3_min)                     # 主门: T+3 预期涨幅>0 (原硬门)
             OR (T+2 > t2_min 且 T+3 > t3_floor) # 副门: T+2 强看涨 + T+3 未深度转负
    T+3 仍为首要 (主门); 副门只救 T+3 边际转负但 T+2 强看涨的股, 不救深转负.

    概率口径 (用户 2026-08-06): 概率=逐股自然概率 (P(该股达到固定绝对目标)), 每股唯一
    真值. 原 ">60%" 门槛 (基于 P(MFE>0)≈90% 旧口径) 不可达, 故默认不设概率门槛.
    保留 prob_min 参数以便后续收紧. 主视界 T+3 (2026-08-05 用户: 短持 3 天)."""
    g3 = res["pred_mag_3d"]
    g2 = res["pred_mag_2d"]
    sg = SHORTLIST_SCORE.get("select_gate", {})
    t3_min = sg.get("t3_min", 0.0)
    t2_min = sg.get("t2_min", 0.01)
    t3_floor = sg.get("t3_floor", -0.01)
    keep = (g3 > t3_min) | ((g2 > t2_min) & (g3 > t3_floor))
    if prob_min > 0:
        keep = keep & (res["pred_prob_3d"] > prob_min)
    dropped = res[~keep]
    if len(dropped):
        prob_txt = "" if prob_min <= 0 else f" 或 达到概率≤{prob_min:.0%}"
        print(
            f"[select] 剔除 {len(dropped)} 只 (T+3≤{t3_min:.1%} 且 "
            f"(T+2≤{t2_min:.1%} 或 T+3≤{t3_floor:.1%}){prob_txt}): "
            f"{', '.join(dropped.apply(_sel_reason, axis=1))}",
            flush=True,
        )
    return res[keep].copy().reset_index(drop=True)


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    """按 SHORTLIST_SCORE 权重合成综合分 score_w (用户: 预测先行→按权重打分→再排名).

    score_w = Σ_h horizon_w[h] × (gain_w×norm_gain_h + prob_w×norm_prob_h)
    涨幅与达到概率都按当日入选股 min-max 归一化到 0~1 (同量纲, 真 50:50).
    2026-08-05 用户定案: 概率若用原始值 (0.32~0.40 近似常数), 排名被涨幅主导 (相关 0.949);
    归一化后概率真正参与排名. 排名=score_w 降序, 绝不先排名后预测.
    """
    out = df.copy()
    for h in HORIZONS:
        g = out[f"pred_mag_{h}"]
        glo, ghi = g.min(), g.max()
        out[f"norm_g_{h}"] = ((g - glo) / (ghi - glo)).fillna(0.0) if ghi > glo else 0.0
        p = out[f"pred_prob_{h}"].fillna(0.0)
        plo, phi = p.min(), p.max()
        out[f"norm_p_{h}"] = ((p - plo) / (phi - plo)) if phi > plo else 0.0
    hw, gw, pw = (
        SHORTLIST_SCORE["horizon_w"],
        SHORTLIST_SCORE["gain_w"],
        SHORTLIST_SCORE["prob_w"],
    )
    out["score_w"] = sum(
        hw[h] * (gw * out[f"norm_g_{h}"] + pw * out[f"norm_p_{h}"]) for h in HORIZONS
    )
    return out


def rank_and_truncate(res: pd.DataFrame) -> pd.DataFrame:
    """2026-08-07 定案: 短名单按 每股 pred_mag(主视界 T+3 幅度) 降序 取每板块 TOP-10.

    替代原 score_w 混合排名 (250d OOS: 纯 pred_mag 排名 MFE 全视界赢, 混合/两段式全不如).
    前 5 标记 T-5, 前 10 标记 T-10 (沿用原 cut 语义); 超 10 剔除. 排名键 = CAND_RANK_KEY.
    """
    key = CAND_RANK_KEY
    if res.empty:
        return res
    out = []
    for board in ("main", "dual"):
        b = res[res["board"] == board].sort_values(
            key, ascending=False, na_position="last"
        )
        if b.empty:
            continue
        top5 = b.head(5).copy()
        top10 = b.head(10).copy()
        out.append(
            pd.concat([top5.assign(cut="T-5"), top10.assign(cut="T-10")], ignore_index=True)
        )
    return pd.concat(out, ignore_index=True).reset_index(drop=True)


def build_merged(res: pd.DataFrame) -> pd.DataFrame:
    """合并短名单: T-5⊂T-10, 取 T-10 全集; main+dual 全局排名.

    IRON RULE (用户): 先预测(涨幅+达到概率) → 再排名.
    排名=pred_mag_3d 降序 (2026-08-07 定案: 纯幅度排名 > score_w 混合), 平局按 score_w→
    达到概率; 负涨幅绝不在正涨幅之前 (已由 select 门保证). 共现仅作平局裁决参考.
    """
    df = res[res["cut"] == "T-10"].copy()
    t5 = set(res[res["cut"] == "T-5"]["symbol"])
    df["in_t5"] = df["symbol"].isin(t5)
    df = add_score(df)
    keys = [CAND_RANK_KEY, "score_w", "pred_prob_3d"]
    df = df.sort_values(keys, ascending=False, na_position="last").reset_index(
        drop=True
    )
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def fmt_merged(m: pd.DataFrame) -> list[str]:
    """合并排名表的终端文本 (单张决策清单, 涨幅→概率降序, 前瞻逐视界)."""
    hdr = (
        f"{'#':>3} {'代码':<8} {'板块':<5} {'模块':<16} {'score':>7} "
        + "  ".join(
            f"{'T+' + h[:-1] + ' 预期涨幅(达到概率)':>20}" for h in HORIZON_ORDER
        )
        + f" {'T5':>3}"
    )
    if len(m) == 0:
        return [
            "── 合并短名单 (主板+双创合一, 共 0 只) ──",
            "  今日主视界无优势组合, 空仓观望 (见下方制度门判定)",
        ]
    lines = [
        f"── 合并短名单 (主板+双创合一, 共 {len(m)} 只; 仅正预期涨幅; "
        f"预期=该股最新score经OOS每股独立线性校准的今后涨幅(MFE), 每股唯一; "
        f"达到概率=逐股自然概率 P(该股达到固定绝对目标), 每股唯一真值, 非桶级共享; "
        f"过门=该板块×系统今日过制度门, "
        f"未过门个股不应买入) ──",
        hdr,
    ]
    for _, r in m.iterrows():
        mod = "★共现(双系统)" if bool(r["co_occur"]) else str(r["systems"])
        cols = []
        for h in HORIZON_ORDER:
            ex = (
                "n/a"
                if pd.isna(r[f"pred_mag_{h}"])
                else f"{r[f'pred_mag_{h}']:+.1%}({r[f'pred_prob_{h}']:.0%})"
            )
            cols.append(f"{ex:>18}")
        t5 = "Y" if bool(r["in_t5"]) else ""
        lines.append(
            f"{r['rank']:>3} {r['symbol']:<8} {BOARD_LABEL.get(r['board'], r['board']):<5} "
            f"{mod:<16} {r['score']:.4f} " + "  ".join(cols) + f" {t5:>3}"
        )
    return lines


def build_summary(res: pd.DataFrame, stats: dict, sel_date: pd.Timestamp) -> list[str]:
    """SUMMARY: 系统级逐视界胜率/期望 → 两系统共识 → 建议顺序."""
    lines = [
        f"── SUMMARY ── 选股日(数据) {sel_date:%Y-%m-%d}",
        "系统级 OOS 逐视界 期望涨幅(MFE)/达到概率 (同系统个股共享, 非个股值; 真实口径):",
    ]
    for board in ("main", "dual"):
        label = BOARD_LABEL.get(board, board)
        lines.append(f"  [{label}]")
        for name in ("sniper", "fusion"):
            per = stats.get((board, name), {})
            seg = "  ".join(
                f"T+{h[:-1]} {per[h]['mag']:+.1%}({per[h]['hit']:.0%})"
                for h in HORIZONS
                if h in per
            )
            lines.append(f"    {name}: {seg or 'n/a'}")
    for board in ("main", "dual"):
        b = res[res["board"] == board]
        if b.empty:
            continue
        label = BOARD_LABEL.get(board, board)
        co = b[b["co_occur"]].sort_values(["cut", CAND_RANK_KEY], ascending=[True, False])
        lines.append(
            f"\n[{label}] 两系统共识(共现)股: "
            f"{', '.join(str(s) for s in co['symbol'].unique()) if not co.empty else '无'}"
        )
        lines.append(f"  建议顺序 ({CAND_RANK_KEY} 降序: 主视界每股预期幅度):")
        for cut in ("T-5", "T-10"):
            g = b[b["cut"] == cut].sort_values(CAND_RANK_KEY, ascending=False)
            if g.empty:
                continue
            picks = " > ".join(
                f"{r['symbol']}#{i + 1}[{r['systems']}]{'★' if bool(r['co_occur']) else ''}"
                for i, (_, r) in enumerate(g.iterrows())
            )
            lines.append(f"    {cut}: {picks}")
    lines.append(
        "\n每只个股逐视界预期 = 该股最新 score 经 OOS 每股独立线性校准的 今后涨幅(MFE) "
        "(每股唯一, 非桶级共享; 前瞻, 非历史回看)."
    )
    lines.append(
        "达到概率 = 逐股自然概率 P(该股达到固定绝对目标) — 每股唯一真值, 非桶级共享 "
        "(旧 P(MFE>0) 85-99% 已废, 2026-08-05 用户定案)."
    )
    lines.append(
        "建议: 优先 score_w 高者 (预期涨幅×概率加权); 系统级期望/胜率见 SUMMARY 段."
    )
    return lines


def write_docx(
    res: pd.DataFrame,
    summary: list[str],
    sel_date: pd.Timestamp,
    path: Path,
    merged: pd.DataFrame | None = None,
    module: str = "",
) -> None:
    if Document is None:
        return
    doc = Document()
    title = f"STOCK LIST {sel_date:%Y%m%d}"
    if module:
        title += f" · module {module}"
    doc.add_heading(title, level=0)
    if merged is not None and not merged.empty:
        doc.add_heading("合并短名单 · 集成决策清单", level=1)
        doc.add_paragraph(
            "rank=score_w 降序 (预期涨幅×达到概率加权, T+3 主视界短持) · "
            "systems=命中模块(共现=双系统) · T5=该股是否入选所在板块T-5 · 仅保留 T+3 预期涨幅>0 的股 · "
            "每视界(T+2/3/5/10) 预期涨幅(MFE)=最新score经OOS每股独立线性校准的今后表现, 每股唯一; 达到概率=逐股自然概率 P(该股达到固定绝对目标), 每股唯一真值"
        )
        mcols = ["rank", "symbol", "board", "module", "score", "score_w", "in_t5"] + [
            f"{k}_{h}" for h in HORIZON_ORDER for k in ("pred_mag", "pred_prob")
        ]
        t = doc.add_table(rows=1, cols=len(mcols))
        for j, c in enumerate(mcols):
            t.rows[0].cells[j].text = c
        for _, r in merged.iterrows():
            cells = t.add_row().cells
            cells[0].text = str(r["rank"])
            cells[1].text = str(r["symbol"])
            cells[2].text = BOARD_LABEL.get(r["board"], r["board"])
            cells[3].text = str(r["systems"])
            cells[4].text = f"{float(r['score']):.4f}"
            cells[5].text = f"{float(r['score_w']):.4f}"
            cells[6].text = "Y" if bool(r["in_t5"]) else ""
            j = 7
            for h in HORIZON_ORDER:
                cells[j].text = (
                    "n/a"
                    if pd.isna(r[f"pred_mag_{h}"])
                    else f"{float(r[f'pred_mag_{h}']):+.1%}"
                )
                cells[j + 1].text = (
                    "n/a"
                    if pd.isna(r[f"pred_prob_{h}"])
                    else f"{float(r[f'pred_prob_{h}']):.0%}"
                )
                j += 2
    for line in summary:
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.size = Pt(9)
    cols = ["rank", "symbol", "module", "co_occur", "score", "score_w"] + [
        f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob")
    ]
    for board in ("main", "dual"):
        b = res[res["board"] == board]
        if b.empty:
            continue
        doc.add_heading(f"{board} ({BOARD_LABEL.get(board, board)})", level=1)
        for cut in ("T-5", "T-10"):
            g = b[b["cut"] == cut].sort_values("score_w", ascending=False)
            if g.empty:
                continue
            doc.add_paragraph(
                f"{cut} · score_w 降序(预期涨幅×达到概率加权) · 仅正预期涨幅股 · "
                f"预期涨幅(MFE)=最新score经OOS每股独立线性校准的今后表现, 每股唯一; 达到概率=逐股自然概率 P(该股达到固定绝对目标)"
            )
            t = doc.add_table(rows=1, cols=len(cols))
            for j, c in enumerate(cols):
                t.rows[0].cells[j].text = c
            for i, (_, r) in enumerate(g.iterrows()):
                cells = t.add_row().cells
                cells[0].text = str(i + 1)
                cells[1].text = str(r["symbol"])
                cells[2].text = str(r["systems"])
                cells[3].text = "★" if bool(r["co_occur"]) else ""
                cells[4].text = f"{float(r['score']):.4f}"
                cells[5].text = f"{float(r['score_w']):.4f}"
                for j, h in enumerate(HORIZONS):
                    cells[6 + 2 * j].text = (
                        "n/a"
                        if pd.isna(r[f"pred_mag_{h}"])
                        else f"{float(r[f'pred_mag_{h}']):+.1%}"
                    )
                    cells[7 + 2 * j].text = (
                        "n/a"
                        if pd.isna(r[f"pred_prob_{h}"])
                        else f"{float(r[f'pred_prob_{h}']):.0%}"
                    )
    doc.save(str(path))


def write_xlsx(
    res: pd.DataFrame,
    summary: list[str],
    sel_date: pd.Timestamp,
    path: Path,
    merged: pd.DataFrame | None = None,
    module: str = "",
) -> None:
    if Workbook is None:
        return
    wb = Workbook()
    hdr_fill = PatternFill("solid", fgColor="D9E1F2")
    bold = Font(bold=True)
    pct = "0.0%"

    def _sheet(ws, df, cols, pct_cols=()):
        ws.append(cols)
        for c in ws[1]:
            c.font, c.fill = bold, hdr_fill
        for _, r in df.iterrows():
            ws.append([r.get(c, "") for c in cols])
        for j, c in enumerate(cols, 1):
            if c in pct_cols:
                for cell in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                    v = cell[0].value
                    if isinstance(v, (int, float)):
                        cell[0].number_format = pct
            ws.column_dimensions[get_column_letter(j)].width = max(
                10, min(28, max(len(str(c)), 6))
            )
        ws.freeze_panes = "A2"

    ws = wb.active
    ws.title = "SUMMARY"
    title = f"STOCK LIST {sel_date:%Y%m%d}"
    if module:
        title += f" · module {module}"
    ws.append([title])
    ws["A1"].font = Font(bold=True, size=13)
    for line in summary:
        ws.append([line])
    ws.column_dimensions["A"].width = 110

    data_cols = [
        "date",
        "board",
        "cut",
        "rk",
        "symbol",
        "module",
        "co_occur",
        "score",
    ] + [f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob", "pred_n")]
    pct_cols = [f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob")]
    if merged is not None and not merged.empty:
        mcols = ["rank", "symbol", "board", "module", "score", "score_w", "in_t5"] + [
            f"{k}_{h}" for h in HORIZON_ORDER for k in ("pred_mag", "pred_prob")
        ]
        m = merged.copy().rename(columns={"systems": "module"})
        _sheet(
            wb.create_sheet("合并排名"),
            m[mcols],
            mcols,
            pct_cols=[
                f"{k}_{h}" for h in HORIZON_ORDER for k in ("pred_mag", "pred_prob")
            ],
        )
    for cut, title in (("T-5", "短名单 T-5"), ("T-10", "短名单 T-10")):
        r2 = res[res["cut"] == cut].copy().rename(columns={"systems": "module"})
        _sheet(wb.create_sheet(title), r2, data_cols, pct_cols)
    wb.save(str(path))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = sys.argv[1:]
    trade_date = args[0] if (args and len(args[0]) == 8 and args[0].isdigit()) else None
    watch = [a for a in args if a != trade_date]

    # 读 FULL RUN 落盘短名单 (权威选股, 不重算)
    frames = []
    for board in ("main", "dual"):
        fp = FULLRUN_DIR / f"shortlist_{board}.csv"
        if not fp.exists():
            print(f"[warn] {fp.name} 不存在, 跳过", flush=True)
            continue
        df = pd.read_csv(fp, dtype={"symbol": str})
        frames.append(df)
    if not frames:
        print("[error] FULL RUN 短名单为空", flush=True)
        return 1
    full = pd.concat(frames, ignore_index=True)
    sel_date = (
        pd.Timestamp(trade_date) if trade_date else pd.Timestamp(full["date"].max())
    )

    t0 = pd.Timestamp.now()
    from app.pipeline1.model_meta import load_modules, module_id

    module = module_id(load_modules())
    records = load_oos_records()
    gate = regime_gate(records)
    active = [f"{b}/{s}" for (b, s), g in gate.items() if g["active"]]
    print(
        f"[regime] 今日保留组合: {', '.join(active) if active else '无 (主视界无优势, 空仓)'}",
        flush=True,
    )
    cands = expand_candidates(full)
    raw_res = add_oos_pred(cands, records)
    _persist_raw_preds(raw_res, sel_date, module)
    res = select_confident(ema_smooth(raw_res, sel_date, module), prob_min=0.0)
    # 制度门 (2026-08-06 用户: 清单必须含个股明细 — 不再整组剔除→空仓,
    # 改为标注 regime_active 过门/未过门; 未过门个股**不应买入**, 见顶部说明)
    if REGIME_GATE.get("enable", True):
        active_mask = res.apply(lambda r: _row_active(r, gate), axis=1)
        if not active_mask.any():
            fb = gate_fallback(gate)
            if fb:
                print(
                    f"[regime] 全组合未过线 → 弱市兜底: 仅保留 best-alpha {fb[0]}/{fb[1]}",
                    flush=True,
                )
                active_mask = res.apply(
                    lambda r: (
                        r["board"] == fb[0]
                        and (r["systems"] == fb[1] or r["systems"] == "both")
                    ),
                    axis=1,
                )
        n_drop = int((~active_mask).sum())
        if n_drop:
            print(
                f"[regime] 制度门剔除 {n_drop} 行 (所属板块×系统今日未过线)", flush=True
            )
            res = res[active_mask].copy().reset_index(drop=True)
    res = add_score(res)
    res = rank_and_truncate(res)
    stats = load_system_stats(records)
    merged = build_merged(res)
    summary = build_summary(res, stats, sel_date)
    summary = summary[:1] + fmt_regime(gate) + summary[1:]
    print(
        f"[enrich] {len(res)} 行 ({(pd.Timestamp.now() - t0).total_seconds():.0f}s)",
        flush=True,
    )

    suffix = _module_suffix(module)
    stamp = str(sel_date.date()).replace("-", "")
    csv_path = STOCK_LIST_DIR / f"parallel_shortlist_{stamp}{suffix}.csv"
    res.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}", flush=True)
    # WORM: 同名旧文件若被 Word 锁定则换带标记的新名, 不覆盖不丢失
    docx_path = STOCK_LIST_DIR / f"STOCK LIST {stamp}{suffix}.docx"
    try:
        write_docx(res, summary, sel_date, docx_path, merged, module)
    except PermissionError:
        docx_path = STOCK_LIST_DIR / f"STOCK LIST {stamp}{suffix}_perhorizon.docx"
        write_docx(res, summary, sel_date, docx_path, merged, module)
        print(f"[warn] 原 docx 被占用, 已写 {docx_path.name}", flush=True)
    if docx_path.exists():
        print(f"[saved] {docx_path}", flush=True)
    xlsx_path = STOCK_LIST_DIR / f"STOCK LIST {stamp}{suffix}.xlsx"
    try:
        write_xlsx(res, summary, sel_date, xlsx_path, merged, module)
    except PermissionError:
        xlsx_path = STOCK_LIST_DIR / f"STOCK LIST {stamp}{suffix}_perhorizon.xlsx"
        try:
            write_xlsx(res, summary, sel_date, xlsx_path, merged, module)
            print(f"[warn] 原 xlsx 被占用, 已写 {xlsx_path.name}", flush=True)
        except PermissionError:
            ts = pd.Timestamp.now().strftime("%H%M%S")
            xlsx_path = (
                STOCK_LIST_DIR / f"STOCK LIST {stamp}{suffix}_perstock_{ts}.xlsx"
            )
            write_xlsx(res, summary, sel_date, xlsx_path, merged, module)
            print(
                f"[warn] 原 xlsx + perhorizon 均被 Excel 占用, 已写 {xlsx_path.name}",
                flush=True,
            )
    if xlsx_path.exists():
        print(f"[saved] {xlsx_path}", flush=True)

    lines = summary[:1] + fmt_merged(merged) + [""] + summary[1:]
    print("\n" + "\n".join(lines), flush=True)
    for board in ("main", "dual"):
        b = res[res["board"] == board]
        if b.empty:
            continue
        print(f"\n══ {board} ══")
        for cut, g in b.groupby("cut", sort=True):
            g = g.sort_values(CAND_RANK_KEY, ascending=False)
            print(f"── {cut} ──")
            for i, (_, r) in enumerate(g.iterrows()):
                tag = "★共现" if r["co_occur"] else "    "
                ex = "  ".join(
                    f"{h}:{r[f'pred_mag_{h}']:+.1%}/{r[f'pred_prob_{h}']:.0%}"
                    for h in HORIZONS
                )
                print(
                    f"  #{i + 1} {tag} {r['symbol']:<8} "
                    f"score={r['score']:.4f} [{ex}] ({r['systems']})"
                )

    if watch:
        print("\n══ 指定个股 ══", flush=True)
        sl = set(res["symbol"])
        for s in watch:
            if s in sl:
                row = res[(res["symbol"] == s) & (res["cut"] == "T-10")].iloc[0]
                seg = "  ".join(
                    f"T+{h[:-1]} {row[f'pred_mag_{h}']:+.1%}({row[f'pred_prob_{h}']:.0%})"
                    for h in HORIZONS
                )
                print(f"{s}: 入选短名单 | 预期涨幅(MFE)/概率: {seg}", flush=True)
            else:
                print(
                    f"{s}: 未入选 — 模型今日未对其打分, 无前瞻预测 "
                    f"(模型只对 top-N 产生预测)",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
