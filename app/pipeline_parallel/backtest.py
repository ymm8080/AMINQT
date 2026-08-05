# -*- coding: utf-8 -*-
"""PIPELINE 并行多系统 回测/验收 (2026-08-04).

目标口径 = MFE (持有期内最大涨幅, 2026-08-04 用户需求), 非目标日收盘.
对每套启用系统:
  1. 特征池合成池分 (每日期截面分位等权);
  2. 按 TOP-N / TOP-N_ALT 双档每日选股;
  3. 对选中切片逐视界 (T+2/3/5/10) 量 双头 (幅度+胜率), 与无条件基准对比;
  4. 板块拆分 (main/dual), 每板块独立阈值;
  5. 验收只看 OOS (默认末 250 交易日): 任一视界双头通过 → 系统保留;
     full 全窗仅作参考, 永不参与保留判定 (2026-08-04 用户).

输出 (WORM): <BACKTEST_RESULT_DIR>_parallel_backtest_<ts>.json + .log
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.pipeline1.label_engine import COST, slippage_tier
from app.pipeline_parallel import indicators, screener, signals
from app.pipeline_parallel.config import (
    ALL_HORIZON_INTS,
    BOARD_PREFIXES,
    BOARD_THRESHOLDS,
    FUSION,
    HORIZONS,
    MIN_MAG,
    MIN_WINRATE,
    OOS_WINDOWS,
    PANEL,
    SLOW_BULL,
    SNIPER,
    SYSTEMS,
    board_of,
)
from app.pipeline_parallel.scoring import (
    dual_head_ok,
    measure_dual_head,
    pool_score,
    select_topn,
)
from config.settings import BACKTEST_RESULT_DIR
from scripts._reclassify_all_features import _finalize_slice


def add_mfe_labels(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """补算 MFE 净标签 (2026-08-04 用户需求): 持有期内最大涨幅, 非目标日收盘.

    MFE = 持有窗口内**最高价**能兑现的最大收益 (潜在最优离场):
      label_mfe_{k}d_net = max(high_hfq[T+2 .. T+1+k]) / close_hfq[T+1] - 1 - cost
    与生产 label_pm_{k}d (close_hfq[T+1+k]/close_hfq[T+1]-1, 目标日收盘) 同时间轴,
    仅把"目标日收盘"换成"窗口内最高价"。买价仍为 close_hfq[T+1] (T+1 买),
    窗口从 T+2 起 (T+1 收盘已持有, 无法兑现 T+1 盘中最高), 含目标日 T+1+k。
    成本口径 = 生产 (COST + 2×分层滑点, adv20)。
    标签合法引用未来价 (非前瞻 bias — 仅供训练/验收, 不用于特征)。
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    exec_px = g["close_hfq"].shift(-1)  # T+1 收盘买价
    max_off = max(horizons) + 1
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, max_off + 1)],
        axis=1,
        keys=range(2, max_off + 1),
    )
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    cost_total = COST + 2 * slip
    for k in horizons:
        # skipna=False: 窗口内缺未来价 (尾段/停牌) → 标签 NaN (保守, 同生产口径)
        peak = shifts.loc[:, 2 : k + 1].max(axis=1, skipna=False)
        df[f"label_mfe_{k}d_net"] = peak / exec_px - 1 - cost_total
    del shifts
    gc.collect()
    return df


def tradability_gate(
    work: pd.DataFrame, lookback: int = 20, min_presence: float = 0.8
) -> tuple[pd.DataFrame, dict]:
    """PIT 可交易性门 (2026-08-04 用户: 剔除慢性停牌/数据中断股).

    对每 (symbol,date): 前 lookback 个交易日该股有行(在交易)的比例 < min_presence
    → 剔除 (买入即套牢/无法成交). 只用历史行 → 无前瞻偏差.
    实测: 慢性停牌股 (300642 近 33 交易日只交易 2 天) 被剔除, 正常股不受影响.
    返回 (过滤后 work, stats).
    """
    dates = np.sort(work["date"].unique())
    di = np.searchsorted(dates, work["date"].values)
    syms = work["symbol"].unique()
    sym_pos = {s: i for i, s in enumerate(syms)}
    si = np.array([sym_pos[s] for s in work["symbol"]])
    mat = np.zeros((len(syms), len(dates)), dtype=np.int64)
    np.add.at(mat, (si, di), 1)
    np.minimum(mat, 1, out=mat)  # 重复行只算在交易
    csum = np.cumsum(mat, axis=1)
    col_cur = csum[si, di]
    prev = di - lookback
    col_prev = np.zeros(len(work), dtype=np.int64)
    ok = prev >= 0
    col_prev[ok] = csum[si[ok], prev[ok]]
    count = col_cur - col_prev
    denom = np.minimum(lookback, di + 1).astype(float)
    ratio = count / denom
    keep = ratio >= min_presence
    stats = {
        "lookback": lookback,
        "min_presence": min_presence,
        "removed_rows": int((~keep).sum()),
        "removed_stocks": int(work.loc[~keep, "symbol"].nunique()),
        "kept_rows": int(keep.sum()),
        "kept_stocks": int(work.loc[keep, "symbol"].nunique()),
    }
    # 布尔索引已产生新帧; 显式 .copy() 会强制 block consolidation (整帧挤进单块,
    # 需 ~2× 表大小连续内存) → 本机 15.8GB 物理必 OOM (见 memory/machine-ram-block-consolidation).
    return work.loc[keep], stats


def load_panel() -> pd.DataFrame:
    """快速路径行集: 复用 main/dual 3y 检查点 + MFE 标签 + 可交易性门 + 板块列.

    与 _reclassify_all_features / _sniper_acceptance 完全一致的行集,
    额外加 MFE 净标签 (2026-08-04 用户: 目标是持有期内最大涨幅),
    再按 PIT 可交易性门剔除慢性停牌股, 并按代码前缀标 board 列.
    """
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        df = _finalize_slice(pd.read_parquet(ckpt))
        df = add_mfe_labels(df, horizons=ALL_HORIZON_INTS)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    work, gate = tradability_gate(work)
    work["board"] = work["symbol"].map(board_of)
    # ADX 慢牛: 一次性加指标/打分因子列 + 买卖信号列 + 硬门槛掩码 (全窗口 PIT)
    work = indicators.prepare_adx(work)
    signals.add_signal_columns(work)
    work["gate_slow_bull"] = screener.compute_gate(work, "slow_bull")
    print(
        f"可交易性门: 剔除 {gate['removed_rows']:,} 行 / "
        f"{gate['removed_stocks']} 只 (近{gate['lookback']}交易日"
        f"有行比例<{gate['min_presence']}), 保留 {gate['kept_rows']:,} 行 / "
        f"{gate['kept_stocks']} 只",
        flush=True,
    )
    print(
        f"慢牛门槛: 通过 {int(work['gate_slow_bull'].sum()):,} 行 / "
        f"{work.loc[work['gate_slow_bull'], 'symbol'].nunique():,} 只 "
        f"(每日 Top-20 池候选)",
        flush=True,
    )
    return work


def _baseline(work: pd.DataFrame, label_col: str) -> dict:
    v = work[label_col].dropna()
    return {
        "mag": float(v.mean()),
        "winrate": float((v > 0).mean()),
        "n": int(len(v)),
    }


def run_system(
    work: pd.DataFrame,
    spec,
    top_n: int,
    mask: np.ndarray | None = None,
    crit: tuple[float, float] | None = None,
) -> dict:
    """单系统单档位回测: 返回 {per_horizon, passed, baseline_delta}.

    mask: 可选的布尔数组 (对齐 work) → 只在子窗口内选股+量测 (OOS 测试).
    截面分位对日期独立, 子窗口选股与全量窗口等价, 无需重算全量.
    crit: (min_winrate, min_mag) 每板块阈值; None → 模块级默认.
    """
    min_wr, min_mag = crit if crit is not None else (MIN_WINRATE, MIN_MAG)
    if not spec.pool:
        raise ValueError(f"{spec.name} 特征池为空")
    if mask is None:
        sub = work
    else:
        sub = work[mask]
    if len(sub) == 0:
        return {"top_n": top_n, "per_horizon": {}, "passed": [], "n_picks": 0}
    # 硬门槛先行 (慢牛系统): 只对通过门槛的池打分. 优先读预计算掩码列 (全窗 PIT),
    # 缺列 (未 prepare 的通用面板) 时用 apply_gate → 无候选即空帧.
    if spec.gate:
        gc_col = f"gate_{spec.gate}"
        sub = (
            sub[sub[gc_col]]
            if gc_col in sub.columns
            else screener.apply_gate(sub, spec.gate)
        )
    if len(sub) == 0:  # 门槛后无候选 (未 prepare 面板) → 无选股, 不崩
        return {"top_n": top_n, "per_horizon": {}, "passed": [], "n_picks": 0}
    score = pool_score(sub, spec.pool, weights=spec.pool_weights)
    top = select_topn(sub, score, top_n)
    del score
    gc.collect()

    per = {}
    for h, lab in zip(spec.horizons, spec.labels):
        if lab not in sub.columns:
            per[h] = {"mag": float("nan"), "winrate": float("nan"), "n": 0, "ok": False}
            continue
        base = _baseline(sub, lab)
        sel = top.merge(sub[["symbol", "date", lab]], on=["symbol", "date"], how="left")
        m = measure_dual_head(sel, lab)
        per[h] = {
            "mag": m["mag"],
            "winrate": m["winrate"],
            "n": m["n"],
            "ok": dual_head_ok(m, min_wr, min_mag),
            "baseline": base,
            "delta_wr": (m["winrate"] - base["winrate"]) if m["n"] else None,
            "delta_mag": (m["mag"] - base["mag"]) if m["n"] else None,
        }
    passed = [h for h, r in per.items() if r.get("ok")]
    return {
        "top_n": top_n,
        "per_horizon": per,
        "passed": passed,
        "n_picks": int(len(top)),
    }


def _dual_per_horizon(
    sub: pd.DataFrame, sel: pd.DataFrame, spec, crit: tuple[float, float] | None = None
) -> dict:
    """对选中切片 sel (含 symbol/date) 逐视界量双头, 基准取 sub 窗口."""
    min_wr, min_mag = crit if crit is not None else (MIN_WINRATE, MIN_MAG)
    per = {}
    for h, lab in zip(spec.horizons, spec.labels):
        if lab not in sub.columns:
            per[h] = {"mag": float("nan"), "winrate": float("nan"), "n": 0, "ok": False}
            continue
        base = _baseline(sub, lab)
        s = sel.merge(sub[["symbol", "date", lab]], on=["symbol", "date"], how="left")
        m = measure_dual_head(s, lab)
        per[h] = {
            "mag": m["mag"],
            "winrate": m["winrate"],
            "n": m["n"],
            "ok": dual_head_ok(m, min_wr, min_mag),
            "baseline": base,
            "delta_wr": (m["winrate"] - base["winrate"]) if m["n"] else None,
        }
    return per


def rank_bands(
    work: pd.DataFrame,
    spec,
    top_n: int,
    bands: tuple[tuple[str, int, int], ...],
    mask: np.ndarray | None = None,
    crit: tuple[float, float] | None = None,
) -> dict:
    """选 TOP-N 后按每日期 score 排名分档 (如 TOP-10 拆 [1-5]/[6-10]), 逐档量双头.

    回答 (2026-08-04 用户): T+10 是否整组 TOP-10 赢 TOP-5, 还是仅 TOP-10 前 5 只赢?
    """
    if mask is None:
        sub = work
    else:
        sub = work[mask]
    score = pool_score(sub, spec.pool)
    top = select_topn(sub, score, top_n)
    if top.empty:
        return {bname: {} for bname, _, _ in bands}
    top["rk"] = (
        top.groupby("date")["score"].rank(ascending=False, method="first").astype(int)
    )
    return {
        bname: _dual_per_horizon(sub, top[top["rk"].between(lo, hi)], spec, crit)
        for bname, lo, hi in bands
    }


def run_all(
    work: pd.DataFrame,
    ts: str,
    oos_days: int | None = None,
    oos_windows: dict[str, int] | None = None,
) -> dict:
    """跑全部启用系统 → WORM JSON 结构.

    报告窗口:
      - full: 全 3y 窗 (仅参考, kept=None)
      - oos:  dict {label: 交易日数} 的样本外窗口, 每窗独立选股+量测.
        2026-08-04 用户: "BACKTESTING CONSISTS OF 6M, 3M, 10D" → 默认
        OOS_WINDOWS = {"6m":126, "3m":63, "10d":10} (config);
        传 oos_days=N 时退化为单窗 {"oos": N} (测试用).
    2026-08-04 用户: 验收只看 OOS 结果, full 永不用于保留判定.
    """
    if oos_windows is None:
        oos_windows = {"oos": oos_days} if oos_days is not None else dict(OOS_WINDOWS)
    dates = np.sort(work["date"].unique())
    if any(d >= len(dates) for d in oos_windows.values()):
        raise ValueError(f"OOS 窗口天数 {dict(oos_windows)} >= 总交易日 {len(dates)}")

    out = {
        "ts": ts,
        "objective": "MFE: 持有期内最大涨幅 (label_mfe_{h}d_net), 非目标日收盘",
        "window": {
            "full": {
                "start": str(work["date"].min()),
                "end": str(work["date"].max()),
                "trading_days": int(len(dates)),
            },
            "oos": {
                lab: {
                    "start": str(dates[-d]),
                    "end": str(work["date"].max()),
                    "trading_days": d,
                }
                for lab, d in oos_windows.items()
            },
        },
        "criteria": {
            "dual_head": "幅度>阈值 且 胜率>=阈值 (每板块不同, 见 boards[*].criteria)",
            "boards": {
                b: {k: v for k, v in t.items()} for b, t in BOARD_THRESHOLDS.items()
            },
        },
        "gate": work.attrs.get("gate"),
        "rows": int(len(work)),
        "stocks": int(work["symbol"].nunique()),
        "boards": {},
    }
    for b in BOARD_PREFIXES:
        th = BOARD_THRESHOLDS[b]
        bcrit = (th["min_winrate"], th["min_mag"])
        sub = work[work["board"] == b]
        systems = {}
        for name, spec in SYSTEMS.items():
            if not spec.enabled:
                systems[name] = {"enabled": False, "desc": spec.desc}
                continue
            full = {
                "primary": run_system(sub, spec, spec.top_n, crit=bcrit),
                "alt": run_system(sub, spec, spec.top_n_alt, crit=bcrit),
            }
            # 2026-08-04 用户: 验收只看 OOS (样本外). full 仅作参考, 永不参与保留判定.
            full["kept"] = None
            oos = {}
            for lab, d in oos_windows.items():
                bm = sub["date"].values >= dates[-d]
                oos[lab] = {
                    "primary": run_system(sub, spec, spec.top_n, bm, crit=bcrit),
                    "alt": run_system(sub, spec, spec.top_n_alt, bm, crit=bcrit),
                }
                oos[lab]["kept"] = bool(
                    oos[lab]["primary"]["passed"] or oos[lab]["alt"]["passed"]
                )
            systems[name] = {
                "enabled": True,
                "desc": spec.desc,
                "pool": list(spec.pool),
                "top_n": {"primary": spec.top_n, "alt": spec.top_n_alt},
                "full": full,
                "oos": oos,
                "notes": list(spec.notes),
            }
            gc.collect()
        out["boards"][b] = {
            "label": th["label"],
            "criteria": {"min_winrate": th["min_winrate"], "min_mag": th["min_mag"]},
            "rows": int(len(sub)),
            "stocks": int(sub["symbol"].nunique()),
            "latest": str(sub["date"].max()),
            "merged": _build_merged_eval(sub, bcrit, dates, oos_windows),
            "systems": systems,
            "compare": {
                "objective": "TOP-5(狙击池) vs TOP-10(融合池) 逐视界 + "
                "TOP-10 分档 [1-5]/[6-10]",
                "full": _build_compare(sub, None, bcrit),
                "oos": {
                    lab: _build_compare(sub, sub["date"].values >= dates[-d], bcrit)
                    for lab, d in oos_windows.items()
                },
            },
            "last_days": last_days_report(sub),
        }
        del sub
        gc.collect()
    return out


def _build_compare(
    work: pd.DataFrame,
    mask: np.ndarray | None = None,
    crit: tuple[float, float] | None = None,
) -> dict:
    """TOP-5 vs TOP-10 对比: 狙击 TOP-5, 融合 TOP-10 整组 + 前 5 + 后 5."""
    snip = rank_bands(work, SNIPER, SNIPER.top_n, (("top5", 1, 5),), mask, crit)
    fus = rank_bands(
        work,
        FUSION,
        FUSION.top_n,
        (("all10", 1, 10), ("first5", 1, 5), ("last5", 6, 10)),
        mask,
        crit,
    )
    return {"sniper_top5": snip["top5"], "fusion": fus}


def write_worm(out: dict, ts: str) -> tuple[Path, Path, Path]:
    """WORM 落盘 → <BACKTEST_RESULT_DIR>/<ts>/ 子目录 (每次回测一个日期命名目录).

    文件: backtest.json (完整报告, 含 conclusion) + backtest.log (结论置顶)
         + conclusion.txt (可分享的验收结论). 返回 (json, log, run_dir).
    """
    concl = build_conclusion(out)
    out["conclusion"] = concl
    run_dir = BACKTEST_RESULT_DIR / ts
    os.makedirs(run_dir, exist_ok=True)
    p = run_dir / "backtest.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    concl_path = run_dir / "conclusion.txt"
    with open(concl_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(format_conclusion(concl)) + "\n")
    w = out["window"]
    gate = out.get("gate") or {}
    lines = [
        f"[{ts}] PIPELINE 并行多系统回测 (3y) | 目标=MFE",
        f"行集 rows={out['rows']:,} stocks={out['stocks']:,} latest={w['full']['end']}",
        f"全窗 {w['full']['start']} → {w['full']['end']} "
        f"({w['full']['trading_days']}d)",
    ]
    for lab, ow in w["oos"].items():
        lines.append(
            f"OOS[{lab}] {ow['start']} → {ow['end']} ({ow['trading_days']} 交易日)"
        )
    if gate:
        lines.append(
            f"可交易性门: 剔除 {gate.get('removed_rows', 0):,} 行 / "
            f"{gate.get('removed_stocks', 0)} 只 "
            f"(近{gate.get('lookback', 20)}交易日有行比例"
            f"<{gate.get('min_presence', 0.8)}), "
            f"保留 {gate.get('kept_rows', 0):,} 行"
        )
    for b, bd in out["boards"].items():
        lines.append(
            f"\n=== 板块 [{b}] {bd['label']} | 行 {bd['rows']:,} "
            f"股票 {bd['stocks']:,} | 双头: 胜率>={bd['criteria']['min_winrate']} "
            f"幅度>{bd['criteria']['min_mag']} ==="
        )
        for name, s in bd["systems"].items():
            if not s.get("enabled"):
                lines.append(f"[{name}] 未启用 (占位)")
                continue
            fpr = s["full"]["primary"]
            lines.append(
                f"[{name}] 全窗参考 TOP-{s['top_n']['primary']}: "
                f"通过 {fpr['passed'] or '无'} | 选股 {fpr['n_picks']:,}"
            )
            for lab, oos in s["oos"].items():
                opr = oos["primary"]
                lines.append(
                    f"[{name}|OOS {lab}] TOP-{s['top_n']['primary']}: "
                    f"通过 {opr['passed'] or '无'} | "
                    f"选股 {opr['n_picks']:,} | 保留={oos['kept']}"
                )
                for h, r in opr["per_horizon"].items():
                    if r["n"]:
                        lines.append(
                            f"    T+{h}: 幅度={r['mag']:+.2%} "
                            f"胜率={r['winrate']:.1%} "
                            f"(Δ胜率{r['delta_wr']:+.1%}) n={r['n']}"
                        )
        c = bd["compare"]
        for lab in c["oos"]:
            lines.append(
                f"  ── 对比 (OOS {lab}): TOP-5(狙击池) vs TOP-10(融合池), 逐视界双头 ──"
            )
            for h in bd["systems"]["fusion"]["oos"][lab]["primary"]["per_horizon"]:
                r5 = (c["oos"][lab].get("sniper_top5") or {}).get(h) or {}
                fus = c["oos"][lab].get("fusion") or {}
                a10 = (fus.get("all10") or {}).get(h) or {}
                f5 = (fus.get("first5") or {}).get(h) or {}
                l5 = (fus.get("last5") or {}).get(h) or {}
                if not r5.get("n"):
                    lines.append(f"    T+{h}: n/a")
                    continue
                lines.append(
                    f"    T+{h}: 狙TOP5 wr={r5.get('winrate'):.1%}"
                    f"/mag={r5.get('mag'):+.2%} "
                    f"| 融TOP10 wr={a10.get('winrate'):.1%}"
                    f"/mag={a10.get('mag'):+.2%} "
                    f"| 融前5 wr={f5.get('winrate'):.1%}"
                    f"/mag={f5.get('mag'):+.2%} "
                    f"| 融后5 wr={l5.get('winrate'):.1%}"
                    f"/mag={l5.get('mag'):+.2%}"
                )
        lt = bd.get("last_days", {}).get("last_testable") or {}
        if lt:
            lines.append(
                "  各视界可测日期 (末 15 交易日, 同一选股日): "
                + " ".join(
                    f"{h}=至{lt[h]['last_date']}({lt[h]['n']}日)"
                    for h in ("2d", "3d", "5d", "10d")
                )
            )
        n_days = bd.get("last_days", {}).get("n_days", 15)
        lines.append(
            f"  ── 末 {n_days} 个交易日逐日: 当天 TOP-5(狙击)/TOP-10(融合) "
            f"各视界 MFE 双头图 ──"
        )
        for day in bd.get("last_days", {}).get("days", []):
            cells = []
            for name, tag in (("sniper_top5", "狙5"), ("fusion_top10", "融10")):
                fig = day[name]["figure"]
                bits = []
                for h in ("2d", "3d", "5d", "10d"):
                    f = fig.get(h, {})
                    if f.get("n"):
                        bits.append(f"{h} +{f['mag']:.2%}/{f['winrate']:.0%}")
                    else:
                        bits.append(f"{h} n/a")
                cells.append(f"{tag} " + " ".join(bits))
            lines.append(f"    {day['date']}: {' | '.join(cells)}")
    log = run_dir / "backtest.log"
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("\n".join(format_conclusion(concl) + [""] + lines) + "\n")
    return p, log, run_dir


def export_stock_lists(work: pd.DataFrame, oos_start, run_dir: Path) -> list[str]:
    """按板块落盘每系统选股清单 (full + oos, 含各视界 MFE 净标签) + 合并 OOS 名单.

    每板块每系统 stocks_<board>_<system>_<window>.csv
    (date/symbol/score/rk + label_mfe_*_net),
    以及 stocks_merged_oos_<board>.csv: 狙击 TOP-5 ∪ 融合 TOP-10 去重后的
    OOS 买入名单 (2026-08-04 用户: 至少 OOS 的选股清单; 也是"每天买什么"的可操作输出).
    work 需含 board 列 (load_panel 已标).
    返回落盘文件名列表.
    """
    os.makedirs(run_dir, exist_ok=True)
    # runner 传的是 out["window"]["oos"]["start"] (str); 面板 date 是 datetime64,
    # 直接 >= 会触发 UFuncNoLoopError (真实数据回归 2026-08-04). 统一转 Timestamp.
    oos_start = pd.Timestamp(oos_start)
    lab_cols = [f"label_mfe_{h}d_net" for h in (2, 3, 5, 10)]
    written: list[str] = []
    for b in BOARD_PREFIXES:
        bwork = work[work["board"] == b]
        if bwork.empty:
            continue
        oos_mask = bwork["date"].values >= oos_start
        oos_frames: list[pd.DataFrame] = []
        for spec in (SNIPER, FUSION):
            for tag, mask in (("full", None), ("oos", oos_mask)):
                sub = bwork if mask is None else bwork[mask]
                score = pool_score(sub, spec.pool)
                top = select_topn(sub, score, spec.top_n)
                if top.empty:
                    continue
                top["rk"] = (
                    top.groupby("date")["score"]
                    .rank(ascending=False, method="first")
                    .astype(int)
                )
                out = top.merge(
                    sub[["symbol", "date"] + lab_cols],
                    on=["symbol", "date"],
                    how="left",
                )
                out = out.sort_values(["date", "rk"])
                out.insert(0, "window", tag)
                out.insert(1, "system", spec.name)
                out.insert(2, "board", b)
                fp = run_dir / f"stocks_{b}_{spec.name}_{tag}.csv"
                out.to_csv(fp, index=False)
                written.append(fp.name)
                if tag == "oos":
                    oos_frames.append(out[["date", "symbol", "system"]])
        if oos_frames:
            merged = pd.concat(oos_frames, ignore_index=True)
            g = merged.groupby(["date", "symbol"], as_index=False).agg(
                systems=("system", lambda x: "+".join(sorted(set(x))))
            )
            g.insert(0, "board", b)
            fp = run_dir / f"stocks_merged_oos_{b}.csv"
            g.to_csv(fp, index=False)
            written.append(fp.name)

    # 慢牛系统独立导出 (70% 独立资金仓, 不并入狙击/融合 merged OOS 名单).
    # 门槛先行 → 权重打分 → Top-20; 每行附慢牛长视界 (10/20/40d) MFE 净标签.
    sb_lab_cols = [f"label_mfe_{h}d_net" for h in (10, 20, 40)]
    sb_gate_col = f"gate_{SLOW_BULL.gate}"
    for b in BOARD_PREFIXES:
        bwork = work[work["board"] == b]
        if bwork.empty:
            continue
        oos_mask = bwork["date"].values >= oos_start
        for tag, mask in (("full", None), ("oos", oos_mask)):
            sub = bwork if mask is None else bwork[mask]
            if sb_gate_col in sub.columns:
                sub = sub[sub[sb_gate_col]]
            else:
                sub = screener.apply_gate(sub, SLOW_BULL.gate)
            if len(sub) == 0:
                continue
            score = pool_score(sub, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
            top = select_topn(sub, score, SLOW_BULL.top_n)
            if top.empty:
                continue
            top["rk"] = (
                top.groupby("date")["score"]
                .rank(ascending=False, method="first")
                .astype(int)
            )
            out = top.merge(
                sub[["symbol", "date"] + sb_lab_cols], on=["symbol", "date"], how="left"
            )
            out = out.sort_values(["date", "rk"])
            out.insert(0, "window", tag)
            out.insert(1, "system", SLOW_BULL.name)
            out.insert(2, "board", b)
            fp = run_dir / f"stocks_{b}_slow_bull_{tag}.csv"
            out.to_csv(fp, index=False)
            written.append(fp.name)
    return written


def last_days_report(work: pd.DataFrame, n_days: int = 15) -> dict:
    """末 n 个交易日逐日: 该日实际选股清单 (TOP-5 狙击 / TOP-10 融合) + 各视界 MFE 双头图.

    用户需求 (2026-08-04): OOS 末段, 每天用当天实际产生的选股名单算回测数字.
    默认 n_days=15: 买在 T+1 → T+5 需未来 6 日价 (15 日内约 10 日可测),
    T+10 需未来 11 日价 (15 日内约 5 日可测), 同一测试日 (选股日) 共用.
    尾部 T+5/T+10 无未来价 → 对应视界 n=0 如实标注 (与"2026-07-31 无 T+5"口径一致).
    """
    dates = np.sort(work["date"].unique())
    last = dates[-n_days:]
    sub = work[work["date"].isin(set(last))]
    lab_cols = [f"label_mfe_{h}d_net" for h in (2, 3, 5, 10)]
    horizons = (2, 3, 5, 10)
    systems = (
        ("sniper_top5", SNIPER, SNIPER.top_n),
        ("fusion_top10", FUSION, FUSION.top_n),
    )
    days = []
    for d in last:
        day_df = sub[sub["date"] == d]
        entry = {"date": str(pd.Timestamp(d).date())}
        for name, spec, topn in systems:
            if day_df.empty:
                entry[name] = {"picks": [], "figure": {}}
                continue
            score = pool_score(day_df, spec.pool)
            top = select_topn(day_df, score, topn)
            if top.empty:
                entry[name] = {"picks": [], "figure": {}}
                continue
            top["rk"] = top["score"].rank(ascending=False, method="first").astype(int)
            top = top.sort_values("rk")
            figure = {}
            for h, lab in zip(horizons, lab_cols):
                v = work.loc[top.index, lab].dropna()
                figure[f"{h}d"] = {
                    "mag": round(float(v.mean()), 6) if len(v) else None,
                    "winrate": round(float((v > 0).mean()), 4) if len(v) else None,
                    "n": int(len(v)),
                }
            picks = []
            for row_name, row in top.iterrows():
                pick = {
                    "symbol": row["symbol"],
                    "rk": int(row["rk"]),
                    "score": round(float(row["score"]), 4),
                }
                for h, lab in zip(horizons, lab_cols):
                    val = work.loc[row_name, lab]
                    pick[f"mfe_{h}d"] = None if pd.isna(val) else round(float(val), 4)
                picks.append(pick)
            entry[name] = {"picks": picks, "figure": figure}
        days.append(entry)
    # 各视界在报告窗内的可测性: last_date = 该视界 MFE 标签非 NaN 的最后一个选股日,
    # n = 报告窗内可测(非 NaN)的天数. 用户需求 (2026-08-04): "LAST TESTABLE DATES,
    # T-5 HV 10 DATES WHILE T-10 HV 5 DATES..THAT IS LAST 15 TRADING DATES" —
    # 长视界需更远未来价 → 可测末日期更早, 可测天数更少, 但测试日(选股日)同一.
    win = set(last)
    last_testable = {}
    for h, lab in zip(horizons, lab_cols):
        v = work.loc[(work["date"].isin(win)) & work[lab].notna(), "date"]
        last_testable[f"{h}d"] = {
            "last_date": (str(pd.Timestamp(v.max()).date()) if len(v) else None),
            "n": int(v.nunique()),
        }
    return {
        "objective": "末 15 个交易日逐日: 当天 TOP-5(狙击池)/TOP-10(融合池) "
        "实际选股清单 + 各视界 MFE 双头图 (回测用当天名单)",
        "n_days": len(days),
        "days": days,
        "last_testable": last_testable,
    }


def write_last_days_csv(ld: dict, run_dir: Path, board: str = "") -> str:
    """逐日选股清单 → last_{n_days}_days_picks[_<board>].csv (date/system/rk/symbol + 各视界 mfe)."""
    rows = []
    for day in ld["days"]:
        for name in ("sniper_top5", "fusion_top10"):
            system = "sniper" if name == "sniper_top5" else "fusion"
            for p in day[name]["picks"]:
                rows.append(
                    {
                        "date": day["date"],
                        "system": system,
                        "rk": p["rk"],
                        "symbol": p["symbol"],
                        "score": p["score"],
                        "mfe_2d": p["mfe_2d"],
                        "mfe_3d": p["mfe_3d"],
                        "mfe_5d": p["mfe_5d"],
                        "mfe_10d": p["mfe_10d"],
                    }
                )
    suffix = f"_{board}" if board else ""
    fp = run_dir / f"last_{ld['n_days']}_days_picks{suffix}.csv"
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp.name


# ── 最终短名单: 合并模块 (2026-08-04 用户: 一般管道设计, 验收/买入都基于最终短名单) ──


def build_merged_shortlist(
    work: pd.DataFrame, top_n: int, mask: np.ndarray | None = None
) -> pd.DataFrame:
    """合并模块 (核心阶段): 每日期 狙击TOP-5 ∪ 融合TOP-10, 去重, 共现优先+分数降序, 截 top_n.

    最终短名单 = 实际买入名单 (用户: "WHAT WE EVALUATING IS ON FINAL SHORT LIST").
    排序: 共现 (fusion+sniper, 双系统一致=最高确定性) 优先, 组内按分数降序.
    返回长表 date/symbol/systems/co_occur/score/rk (rk 为每日排名).
    """
    sub = work if mask is None else work[mask]
    frames: list[pd.DataFrame] = []
    for spec in (SNIPER, FUSION):
        score = pool_score(sub, spec.pool)
        top = select_topn(sub, score, spec.top_n)
        if top.empty:
            continue
        top = top.copy()
        top["system"] = spec.name
        frames.append(top)
        del score
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    g = merged.groupby(["date", "symbol"], as_index=False).agg(
        systems=("system", lambda x: "+".join(sorted(set(x)))),
        score=("score", "max"),
    )
    g["co_occur"] = g["systems"].str.contains("+", regex=False)
    g = g.sort_values(["date", "co_occur", "score"], ascending=[True, False, False])
    g["rk"] = g.groupby("date").cumcount() + 1
    return g[g["rk"] <= top_n].reset_index(drop=True)


def evaluate_merged(
    work: pd.DataFrame,
    top_n: int,
    mask: np.ndarray | None = None,
    crit: tuple[float, float] | None = None,
) -> dict:
    """对合并最终短名单 (TOP-n) 逐视界量双头 (幅度+胜率), 基准取窗口无条件.

    用户 (2026-08-04): 验收/评估基于最终短名单 (T-5 / T-10).
    """
    sub = work if mask is None else work[mask]
    sl = build_merged_shortlist(sub, top_n)
    if sl.empty:
        return {
            h: {
                "mag": float("nan"),
                "winrate": float("nan"),
                "n": 0,
                "ok": False,
                "baseline": None,
                "delta_wr": None,
            }
            for h in FUSION.horizons
        }
    return _dual_per_horizon(sub, sl[["date", "symbol"]], FUSION, crit)


def _build_merged_eval(
    sub: pd.DataFrame,
    bcrit: tuple[float, float],
    dates: np.ndarray,
    oos_windows: dict[str, int],
) -> dict:
    """run_all 每板块 merged 段: T-5 / T-10 两档, full 仅参考 + 各 OOS 窗验收 (只看 OOS)."""
    merged: dict = {
        "objective": "最终短名单 (狙击TOP-5 ∪ 融合TOP-10 去重, 共现优先, 分数降序) — 验收只看 OOS",
        "cuts": {"top5": 5, "top10": 10},
    }
    for cut, n in (("top5", 5), ("top10", 10)):
        full_ph = evaluate_merged(sub, n, crit=bcrit)
        oos = {}
        for lab, d in oos_windows.items():
            mh = evaluate_merged(sub, n, sub["date"].values >= dates[-d], bcrit)
            oos[lab] = {
                "per_horizon": mh,
                "kept": bool(any(r.get("ok") for r in mh.values())),
            }
        merged[cut] = {"full": {"per_horizon": full_ph, "kept": None}, "oos": oos}
    return merged


def _primary_oos_label(out: dict) -> str:
    """验收用主 OOS 窗 = 交易日数最多的窗 (6m 默认; 单窗 --oos-days 时退化为该窗)."""
    return max(
        out["window"]["oos"], key=lambda lab: out["window"]["oos"][lab]["trading_days"]
    )


def _best_horizon(per: dict) -> tuple[str | None, dict | None]:
    """从逐视界结果里挑最佳视界: 优先已通过双头者, 再按胜率排序; 无 n>=5 则 (None,None)."""
    cand = [(h, r) for h, r in per.items() if r.get("n", 0) >= 5]
    if not cand:
        return None, None
    ok = [(h, r) for h, r in cand if r.get("ok")]
    return max(ok or cand, key=lambda t: t[1]["winrate"])


def build_conclusion(out: dict) -> dict:
    """最终短名单验收结论 (2026-08-04 用户: 报告须有结论/改进点).

    验收只看 OOS 主窗 (out 已含 top5/top10 merged 各窗结果); 产出:
      - 每板块 T-5/T-10 保留判定 + 最佳视界双头数字;
      - 每系统诊断保留判定;
      - 建议 (哪档可交易) + 共现仓位纪律;
      - 自动改进点 (数据落后 / 尾段长视界 NaN / TOP-10 后5名弱 → 只买前5).
    """
    oos_label = _primary_oos_label(out)
    global_latest = out["window"]["full"]["end"]
    boards = {}
    for b, bd in out["boards"].items():
        merged = bd.get("merged", {})
        cuts = {}
        for cut in ("top5", "top10"):
            ph = (
                merged.get(cut, {})
                .get("oos", {})
                .get(oos_label, {})
                .get("per_horizon", {})
            )
            kept = bool(
                (
                    merged.get(cut, {})
                    .get("oos", {})
                    .get(oos_label, {})
                    .get("kept", False)
                )
            )
            best_h, best = _best_horizon(ph)
            if best is None:
                cuts[cut] = {
                    "kept": kept,
                    "best_horizon": None,
                    "winrate": None,
                    "mag": None,
                    "delta_wr": None,
                    "baseline_wr": None,
                    "n": 0,
                }
                continue
            cuts[cut] = {
                "kept": kept,
                "best_horizon": best_h,
                "winrate": round(float(best["winrate"]), 4),
                "mag": round(float(best["mag"]), 4),
                "delta_wr": (
                    round(float(best["delta_wr"]), 4)
                    if best.get("delta_wr") is not None
                    else None
                ),
                "baseline_wr": (
                    round(float(best["baseline"]["winrate"]), 4)
                    if best.get("baseline")
                    else None
                ),
                "n": int(best["n"]),
            }
        systems = {
            name: bool(s.get("oos", {}).get(oos_label, {}).get("kept"))
            for name, s in bd["systems"].items()
            if s.get("enabled")
        }
        boards[b] = {
            "label": bd["label"],
            "latest": bd.get("latest"),
            "stale": bd.get("latest") is not None
            and str(bd["latest"]) < str(global_latest),
            "cuts": cuts,
            "systems": systems,
            "improvements": _conclusion_improvements(b, bd, oos_label, global_latest),
        }
    return {
        "oos_label": oos_label,
        "objective": "最终短名单 (狙击TOP-5 ∪ 融合TOP-10 去重, 共现优先, 分数降序) 双头验收 — 只看 OOS",
        "boards": boards,
        "recommendation": _conclusion_recommendation(boards),
    }


def _conclusion_recommendation(boards: dict) -> dict:
    """按各板块保留判定给出交易建议 + 共现仓位纪律."""
    kept5 = any(bd["cuts"]["top5"]["kept"] for bd in boards.values())
    kept10 = any(bd["cuts"]["top10"]["kept"] for bd in boards.values())
    if kept5 and kept10:
        rec = "T-5 与 T-10 均过验收 → 核心仓 TOP-10 + 机动仓 TOP-5 两档并存"
    elif kept10:
        rec = "仅 T-10 过验收 → 主用 TOP-10 大仓持有 3-5 天"
    elif kept5:
        rec = "仅 T-5 过验收 → 主用 TOP-5 小仓快进快出"
    else:
        rec = "两档均未过验收 → 当前池 OOS 无确定性信号, 建议回池重选或调阈值"
    return {
        "text": rec,
        "sizing": "共现股 (fusion+sniper)=双系统一致, 大仓; 单系统股=小仓",
    }


def _conclusion_improvements(
    b: str, bd: dict, oos_label: str, global_latest: str
) -> list[str]:
    """自动改进点: 数据落后 / 尾段长视界 MFE NaN / TOP-10 后5名弱 → 只买前5."""
    pts: list[str] = []
    if bd.get("latest") is not None and str(bd["latest"]) < str(global_latest):
        pts.append(
            f"数据落后: 板块 {b} 最新 {bd['latest']} < 全局最新 "
            f"{global_latest} — 短名单缺最新交易日"
        )
    lt = (bd.get("last_days", {}) or {}).get("last_testable") or {}
    nd = (bd.get("last_days", {}) or {}).get("n_days", 15)
    for h in ("5d", "10d"):
        info = lt.get(h)
        if info and int(info.get("n", 0)) < nd:
            pts.append(
                f"末段 T+{h}: 仅 {info.get('n')}/{nd} 个选股日可测 "
                f"(缺未来价 → NaN), 长视界信号可信度下降"
            )
    fus = ((bd.get("compare", {}) or {}).get("oos", {}).get(oos_label, {}) or {}).get(
        "fusion"
    ) or {}
    for h, r in (fus.get("all10") or {}).items():
        first5 = (fus.get("first5") or {}).get(h)
        last5 = (fus.get("last5") or {}).get(h)
        if (
            r.get("n", 0) >= 5
            and first5
            and last5
            and first5.get("n", 0) >= 5
            and last5.get("n", 0) >= 5
        ):
            if last5["mag"] < first5["mag"]:
                pts.append(
                    f"T+{h}: TOP-10 后5名幅度 {last5['mag']:+.2%} < "
                    f"前5名 {first5['mag']:+.2%} → 可考虑仅买 TOP-5"
                )
                break
    return pts


def format_conclusion(concl: dict) -> list[str]:
    """结论 dict → 人类可读行 (置于报告顶部 / conclusion.txt)."""
    lines = ["=" * 78, "结论 (最终短名单 双头验收 — 只看 OOS)", "=" * 78]
    lines.append(f"OOS 窗: {concl['oos_label']} | 目标: {concl['objective']}")
    for b, bd in concl["boards"].items():
        stale = " ⚠数据落后" if bd["stale"] else ""
        lines.append(f"\n[{b}] {bd['label']} | 数据最新 {bd['latest']}{stale}")
        for cut in ("top5", "top10"):
            c = bd["cuts"][cut]
            tag = "✓保留" if c["kept"] else "✗未过"
            if c["best_horizon"]:
                delta = (
                    f" (Δ胜率{c['delta_wr']:+.1%} vs 基准{c['baseline_wr']:.1%})"
                    if c["delta_wr"] is not None and c["baseline_wr"] is not None
                    else ""
                )
                lines.append(
                    f"  {cut.upper()}: {tag} | 最佳 T+{c['best_horizon']} "
                    f"胜率={c['winrate']:.1%} 幅度={c['mag']:+.2%}{delta} "
                    f"n={c['n']}"
                )
            else:
                lines.append(f"  {cut.upper()}: {tag} | 无足量数据 (n<5)")
        sysl = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in bd["systems"].items())
        lines.append(f"  系统诊断: {sysl}")
        if bd["improvements"]:
            lines.append("  改进点:")
            for p in bd["improvements"]:
                lines.append(f"    - {p}")
    lines.append(f"\n建议: {concl['recommendation']['text']}")
    lines.append(f"仓位: {concl['recommendation']['sizing']}")
    return lines


def build_daily_shortlists(
    work: pd.DataFrame, out: dict, board: str, date, top_ns: tuple[int, ...] = (5, 10)
) -> pd.DataFrame:
    """今日最终短名单 (T-5 / T-10): 合并模块 + 补逐视界期望/概率与 realized MFE.

    day 按 board 过滤 — 交叉截面排名不能混板 (2026-08-05 bug: 未过滤 → main 名单混入双创股).
    est_wr = 该股命中系统的 OOS 主窗最佳视界胜率; 共现股取两系统 max (双系统一致=更高确定性).
    prob_{h}/exp_{h} (h=2d/3d/5d/10d) = 该股命中系统 OOS 主窗该视界的胜率/平均 MFE
      (2026-08-05 用户: 每只股票需各视界期望涨幅 + 概率); 共现股逐视界取胜率较高者.
    MFE 列是已实现盈利 (需未来价) — 最新日无未来价 → NaN (今日买入名单以 score/est_wr/exp/prob 排序).
    """
    day = work[(work["date"] == date) & (work["board"] == board)]
    frames: list[pd.DataFrame] = []
    for n in top_ns:
        sl = build_merged_shortlist(day, n)
        if sl.empty:
            continue
        sl = sl.copy()
        sl["cut"] = f"T-{n}"
        frames.append(sl)
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    del frames
    oos_label = _primary_oos_label(out)
    best_wr: dict[str, float] = {}
    sys_ph: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("sniper", "fusion"):
        per = out["boards"][board]["systems"][name]["oos"][oos_label]["primary"][
            "per_horizon"
        ]
        cand = [
            (h, r["winrate"])
            for h, r in per.items()
            if r.get("n", 0) >= 5 and not pd.isna(r.get("winrate"))
        ]
        best_wr[name] = max(cand, key=lambda x: x[1])[1] if cand else float("nan")
        sys_ph[name] = {
            h: {"mag": r["mag"], "winrate": r["winrate"]}
            for h, r in per.items()
            if r.get("n", 0) >= 5
            and not pd.isna(r.get("winrate"))
            and not pd.isna(r.get("mag"))
        }

    def _est_wr(systems: str) -> float:
        if systems == "fusion+sniper":
            return max(
                best_wr.get("sniper", float("nan")), best_wr.get("fusion", float("nan"))
            )
        return best_wr.get(systems, float("nan"))

    res["est_wr"] = res["systems"].map(_est_wr)
    # 2026-08-05 用户: 每只股票需各视界期望涨幅 + 概率(置信度).
    # 期望幅度 exp_{h} = OOS 主窗该视界平均 MFE (mag); 概率 prob_{h} = OOS 胜率 (winrate).
    # 共现股逐视界取两系统中胜率较高者 (与 est_wr max 约定一致, "双系统一致=更高确定性"); n<5 → NaN.
    # 列按 HORIZONS 恒定输出 (无样本视界 → NaN), 保证 CSV schema 稳定.
    for h in HORIZONS:
        pick: dict[str, tuple[float, float]] = {}
        for systems in ("fusion+sniper", "fusion", "sniper"):
            names = ("sniper", "fusion") if systems == "fusion+sniper" else (systems,)
            cand = [
                (sys_ph[n][h]["winrate"], sys_ph[n][h]["mag"])
                for n in names
                if h in sys_ph[n]
            ]
            pick[systems] = (
                max(cand, key=lambda x: x[0]) if cand else (float("nan"), float("nan"))
            )
        res[f"prob_{h}"] = res["systems"].map(lambda s, p=pick: p[s][0])
        res[f"exp_{h}"] = res["systems"].map(lambda s, p=pick: p[s][1])
    lab_map = {f"label_mfe_{h}d_net": f"mfe_{h}d" for h in (2, 3, 5, 10)}
    mfe = day.set_index("symbol")[[c for c in lab_map]]
    for src, dst in lab_map.items():
        res[dst] = res["symbol"].map(mfe[src])
    # build_merged_shortlist 输出已含 date 列 (groupby key) → 先摘再以格式化字符串置首
    res = res.drop(columns=["date"])
    res.insert(0, "date", str(pd.Timestamp(date).date()))
    res.insert(1, "board", board)
    return res


def write_daily_shortlist(
    work: pd.DataFrame, out: dict, run_dir: Path, board: str, date=None
) -> str:
    """今日最终短名单 → shortlist_<board>.csv (T-5 与 T-10 同文件, 以 cut 列区分).

    date: 显式指定选股日 (如用户口径 08-03); None → 该板块最新交易日.
    数据落后提示见结论 improve_points (main/dual 各自 latest vs 全局最新).
    """
    if date is None:
        latest = work.loc[work["board"] == board, "date"].max()
    else:
        latest = pd.Timestamp(date)
    sl = build_daily_shortlists(work, out, board, latest)
    if sl.empty:
        return ""
    fp = run_dir / f"shortlist_{board}.csv"
    sl.to_csv(fp, index=False)
    return fp.name
