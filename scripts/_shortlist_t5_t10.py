# -*- coding: utf-8 -*-
"""_shortlist_t5_t10.py — 今日最终短名单 (狙击 TOP-5 ∪ 融合 TOP-10) 前瞻预测输出.

数据源 = 最后一次 FULL RUN 回测产物 (用户 2026-08-05: "已有 full run 基于昨日数据
做出预测, 模块特征/一切都要保留, 不得丢失"). 本脚本**不重算特征、不重新选股**,
只读 FULL RUN 落盘产物, 给每只短名单股计算**前瞻预测** (最新 score 经 OOS 校准).

  - shortlist_main.csv / shortlist_dual.csv : 权威选股结果 (score/co_occur/rk/cut/est_wr)
  - stocks_{board}_{system}_oos.csv          : OOS 逐股 评分 + 已实现前视 MFE
    → 按 score 分位桶校准 → 每股逐视界 预期涨幅(MFE) + 达到概率 (前瞻, 非历史回看)
  - backtest.json                           : 系统级 OOS 逐视界 胜率/期望 (SUMMARY 段)

**概率口径 (2026-08-05 用户定案):** 概率 = 该股达到其"预期涨幅"的真实概率
(= 同 score 桶内 已实现 MFE ≥ 桶期望 的样本占比). 不是 P(MFE>0) — 那曾是 85-99%
被用户否决 "incorrectly high". 真实口径下命中率仅 ~31-44%, 原 "概率>60%" 门槛不可达.

**制度自适应门 (2026-08-05 用户):** 输出哪个 (板块, 系统) 组合**不写死**, 用最新 OOS
市场数据 (固定视界净收益 label_pm) 每日判定: top-quantile 选股平均净收益 > 池基线 + margin
才保留; **判定只看主视界 (T+3)** — 2026-08-05 用户否决用 T+5/T+10 兜底 ("如果没有优势就
不要入选") → 主视界无优势组合则空仓观望, 不输出清单. 见 config REGIME_GATE.

顶部 SUMMARY = 主板+双创合并的单一集成排名清单 (rank/命中模块/预期涨幅/达到概率), 直接决策;
明细表格保留每板块 T-5/T-10 逐视界.

流程 (用户): 先给每只股 预期涨幅+达到概率 (预测), 再按权重 (SHORTLIST_SCORE) 合成综合分,
再排名 — 绝不先排名后预测.

**主视界 (2026-08-05 用户定案):** T+3 (短持 3 天). 排名权重 3d=0.40 最高 (2d+3d 合计 0.65),
入选门 = T+3 预期涨幅>0 (select_confident), 平局裁决用 T+3. 四视界 T+2/3/5/10 预期涨幅+达到概率
仍全部展示 (T+10 降为参考视界).

用法: python scripts/_shortlist_t5_t10.py [YYYYMMDD 选股日, 默认=full run 短名单日期]
可选: 传 symbol 列表 (空格分隔) 查看这些股今日预测 (未入选股模型今日无打分, 无预测).
输出: STOCK_LIST_DIR/parallel_shortlist_<date>__<module>.csv + STOCK LIST <date>__<module>.docx/.xlsx
      (module = current_meta.json 模型版本 tag, 供回归按模块分组评估)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.settings import (
    STOCK_LIST_DIR,
    DATA_OTHERS_DIR,
    DATA_DIR,
    SHORTLIST_SCORE,
    REGIME_GATE,
)
from app.pipeline_parallel.config import HORIZONS

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

# 最后一次 FULL RUN (用户: 基于昨日数据做出预测的全量回测, 产物必须保留)
FULLRUN_DIR = DATA_OTHERS_DIR / "BACKTESTING RESULT" / "20260805_005343"
BOARD_LABEL = {"main": "主板", "dual": "双创(创业板+科创板)"}
# 集成清单逐视界展示顺序 (时间序; HORIZONS = ("3d","2d","5d","10d"))
HORIZON_ORDER = ("2d", "3d", "5d", "10d")
# OOS score→前视涨幅校准 (前瞻预测): score 分位桶数 / 单桶最少已实现样本
CAL_BINS = 6
CAL_MIN_N = 5


def load_system_stats(records: dict) -> dict:
    """OOS 逐股记录 → {(board, system): {h: {"mag", "hit", "n"}}} (SUMMARY 段, 真实口径).

    与个股口径一致: mag=期望涨幅(桶期望), hit=P(已实现 MFE ≥ 期望) — 不再是 P(MFE>0)
    的虚高系统胜率 (旧 82-94% 被用户否决 "not convincing")."""
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


def calibrate(records: dict, board: str, key: str, score: float) -> dict[str, tuple]:
    """把最新 score 映射到 OOS 同分位桶的 已实现 MFE 期望 + 达到期望概率 (逐视界).

    概率 = P(已实现 MFE ≥ 桶期望) — 该股达到其"预期涨幅"的真实概率
    (用户定案: "probability to represent the possibility of stock to gain that price
    increase during the horizon"). 旧口径 P(MFE>0)=85-99% 被否决.
    返回 {h: (exp_mfe, hit_prob, n_bin)}; 桶样本不足或无校准数据 → (NaN, NaN, 0).
    """
    out = {h: (float("nan"), float("nan"), 0) for h in HORIZONS}
    rec = records.get((board, key))
    if rec is None or len(rec) == 0:
        return out
    s = rec["score"]
    edges = np.quantile(s, np.linspace(0.0, 1.0, CAL_BINS + 1))
    idx = int(np.searchsorted(edges, score, side="right")) - 1
    idx = min(max(idx, 0), CAL_BINS - 1)
    lo, hi = edges[idx], edges[idx + 1]
    mask = (s >= lo) & (s < hi)
    if idx == CAL_BINS - 1:
        mask |= s == hi
    sub = rec[mask]
    for h in HORIZONS:
        v = sub[f"mfe_{h}"].dropna()
        if len(v) < CAL_MIN_N:
            continue
        exp = float(v.mean())
        out[h] = (exp, float((v >= exp).mean()), len(v))
    return out


def add_oos_pred(res: pd.DataFrame, records: dict) -> pd.DataFrame:
    """每只短名单股: 用最新 score 经 OOS 校准给 逐视界 前瞻 预期涨幅(MFE)+达到概率."""
    out = res.copy()
    for h in HORIZONS:
        out[f"pred_mag_{h}"], out[f"pred_prob_{h}"], out[f"pred_n_{h}"] = (
            float("nan"),
            float("nan"),
            0,
        )
    for idx, r in out.iterrows():
        key = r["systems"] if r["systems"] in ("sniper", "fusion") else "both"
        cal = calibrate(records, r["board"], key, float(r["score"]))
        for h in HORIZONS:
            mag, prob, n = cal[h]
            out.at[idx, f"pred_mag_{h}"] = mag
            out.at[idx, f"pred_prob_{h}"] = prob
            out.at[idx, f"pred_n_{h}"] = n
    # est_wr 已移除: 系统级常量 (同系统每股同值), 且是旧 P(MFE>0) 虚高口径 (用户 2026-08-05 否决)
    first = ["date", "board", "cut", "rk", "symbol", "systems", "co_occur", "score"]
    ph = [f"{k}_{h}" for h in HORIZONS for k in ("pred_mag", "pred_prob", "pred_n")]
    return out[first + ph].reset_index(drop=True)


def _sel_reason(r: pd.Series) -> str:
    p = "n/a" if pd.isna(r["pred_prob_3d"]) else f"{r['pred_prob_3d']:.0%}"
    m = "n/a" if pd.isna(r["pred_mag_3d"]) else f"{r['pred_mag_3d']:+.1%}"
    return f"{r['symbol']}(T+3 达到概率 {p}, 预期涨幅 {m})"


def select_confident(res: pd.DataFrame, prob_min: float = 0.0) -> pd.DataFrame:
    """IRON RULE (用户): 只列预测上涨股 — 主视界(T+3)预期涨幅>0.

    概率口径已改 (用户 2026-08-05): 概率=达到该预期涨幅的真实概率 (P(实现MFE ≥ 桶期望)),
    实测仅 ~31-44%, 原 ">60%" 门槛 (基于 P(MFE>0)≈90% 旧口径) 不可达, 故默认不设概率门槛.
    保留 prob_min 参数以便后续收紧. 主视界 T+3 (2026-08-05 用户: 短持 3 天)."""
    keep = (res["pred_mag_3d"] > 0.0) & (res["pred_prob_3d"] > prob_min)
    dropped = res[~keep]
    if len(dropped):
        print(
            "[select] 剔除 %d 只 (T+3 预期涨幅≤0%s): %s"
            % (
                len(dropped),
                "" if prob_min <= 0 else f" 或 达到概率≤{prob_min:.0%}",
                ", ".join(dropped.apply(_sel_reason, axis=1)),
            ),
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


def build_merged(res: pd.DataFrame) -> pd.DataFrame:
    """合并短名单: T-5⊂T-10, 取 T-10 全集; main+dual 全局排名.

    IRON RULE (用户): 先预测(涨幅+达到概率) → 按权重合成 score_w → 再排名.
    排名=score_w 降序, 平局按 T+3 涨幅→达到概率; 负涨幅绝不在正涨幅之前 (已由 select 门保证).
    共现仅作平局裁决参考.
    """
    df = res[res["cut"] == "T-10"].copy()
    t5 = set(res[res["cut"] == "T-5"]["symbol"])
    df["in_t5"] = df["symbol"].isin(t5)
    df = add_score(df)
    keys = ["score_w", "pred_mag_3d", "pred_prob_3d"]
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
        f"预期=该股最新score经OOS校准的今后涨幅(MFE); 达到概率=P(实现≥该预期), 真实口径) ──",
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
        co = b[b["co_occur"]].sort_values(["cut", "score_w"], ascending=[True, False])
        lines.append(
            f"\n[{label}] 两系统共识(共现)股: "
            f"{', '.join(str(s) for s in co['symbol'].unique()) if not co.empty else '无'}"
        )
        lines.append("  建议顺序 (score_w 降序: 预期涨幅×达到概率加权):")
        for cut in ("T-5", "T-10"):
            g = b[b["cut"] == cut].sort_values("score_w", ascending=False)
            if g.empty:
                continue
            picks = " > ".join(
                f"{r['symbol']}#{i + 1}[{r['systems']}]{'★' if bool(r['co_occur']) else ''}"
                for i, (_, r) in enumerate(g.iterrows())
            )
            lines.append(f"    {cut}: {picks}")
    lines.append(
        "\n每只个股逐视界预期 = 该股最新 score 经 OOS 同分位校准的 今后涨幅(MFE) (前瞻, 非历史回看)."
    )
    lines.append(
        "达到概率 = P(已实现 MFE ≥ 该预期) — 真实口径 (旧 P(MFE>0) 85-99% 已废, 2026-08-05 用户定案)."
    )
    lines.append(
        "建议: 优先 score_w 高者 (预期涨幅×概率加权); est_wr 为系统级 OOS 参考."
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
            "每视界(T+2/3/5/10) 预期涨幅(MFE)=最新score经OOS校准的今后表现; 达到概率=P(实现≥该预期) (真实口径)"
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
                f"预期涨幅(MFE)=最新score经OOS校准的今后表现; 达到概率=P(实现≥该预期)"
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
    records = load_oos_records()
    gate = regime_gate(records)
    active = [f"{b}/{s}" for (b, s), g in gate.items() if g["active"]]
    print(
        f"[regime] 今日保留组合: {', '.join(active) if active else '无 (主视界无优势, 空仓)'}",
        flush=True,
    )
    res = select_confident(add_oos_pred(full, records), prob_min=0.0)
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
    stats = load_system_stats(records)
    merged = build_merged(res)
    summary = build_summary(res, stats, sel_date)
    summary = summary[:1] + fmt_regime(gate) + summary[1:]
    print(
        f"[enrich] {len(res)} 行 ({(pd.Timestamp.now() - t0).total_seconds():.0f}s)",
        flush=True,
    )

    from app.pipeline1.model_meta import load_modules, module_id

    module = module_id(load_modules())
    suffix = f"__{module}" if module != "na" else ""
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
        write_xlsx(res, summary, sel_date, xlsx_path, merged, module)
        print(f"[warn] 原 xlsx 被占用, 已写 {xlsx_path.name}", flush=True)
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
            g = g.sort_values("score_w", ascending=False)
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
