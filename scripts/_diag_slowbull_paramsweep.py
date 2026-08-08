# -*- coding: utf-8 -*-
"""慢牛 ADX_SPEC + SLOW_BULL_REGIME 单因子参数扫描 (2026-08-08).

围绕生产默认值对 ~10 个高杠杆 ADX_SPEC 阈值 + 全部 SLOW_BULL_REGIME 旋钮
逐参数双向扰动, 用"实得回测"做头对头评估 (op_rule = 上升段 trail 退出 /
下降段按 down_mode)。验收只看 OOS (末 250 交易日)。

内存 (本机 15.8GB 物理 / 44GB commit): 实得回测不用任何 label 列 → 直接以窄列集
(symbol/date/价格/量/换手/adv20/margin) 读检查点重建面板, 避开 load_panel 宽表
深拷贝 OOM (见 memory/machine-ram-block-consolidation)。选股/退出逻辑与
backtest.simulate_slowbull_realized 完全同构, 仅把 trail_pct/hard_stop/max_hold/
down_mode 参数化。

WORM: BACKTEST_RESULT_DIR/diag_slowbull_paramsweep_<ts>/paramsweep.json + summary.txt
"""

from __future__ import annotations

import gc
import json
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline_parallel import indicators, screener, signals
from app.pipeline_parallel.backtest import COST, tradability_gate, slippage_tier
from app.pipeline_parallel.config import (
    ADX_SPEC,
    PANEL,
    SLOW_BULL,
    SLOW_BULL_REGIME,
    board_of,
)
from app.pipeline_parallel.scoring import pool_score, select_topn
from config.settings import BACKTEST_RESULT_DIR

SELL_COLS = (
    "below_ma20",
    "adx_broken",
    "big_drop",
    "below_ma5_3d",
    "turnover_spike",
    "tp_80_div",
)

# 窄列集 (实得回测 + 打分所需全部列; 标签列不需要)
NEED = [
    "symbol",
    "date",
    "close_hfq",
    "high_hfq",
    "low_hfq",
    "open_hfq",
    "adv20",
    "volume",
    "turnover_rate",
    "volume_ratio",
    "margin_balance_chg_5d",
]

OOS_DAYS = 250  # 验收窗 (末 250 交易日)
OOS_CROSSCHECK_DAYS = 63  # 与先前 trail8 实得 (main +1.58/dual +1.88) 交叉核对

# ── 扫描规格 (围绕生产默认, 双向各取 1 个邻近值) ──
ADX_PERTURBS: dict = {
    "adx_min": {"vals": [22.0, 28.0]},
    "ma_bias_max": {"vals": [0.03, 0.08]},
    "amplitude_20_max": {"vals": [0.04, 0.09]},
    "max_drop_20_max": {"vals": [0.03, 0.07]},
    "turnover_min": {"vals": [2.0, 5.0]},
    "turnover_max": {"vals": [12.0, 20.0]},
    "vol_ratio_max": {"vals": [2.0, 5.0]},
    "tp_gain": {"vals": [0.60, 1.00]},
    "adx_optimal_max": {"vals": [30.0, 50.0]},
    "below_ma5_days": {"vals": [2, 4]},
}
REGIME_PERTURBS: dict = {
    "trail_pct": {"vals": [0.05, 0.12]},
    "hard_stop": {"vals": [0.90, 0.95]},
    "max_hold": {"vals": [30, 60]},
    "ma_window": {"vals": [10, 30]},
    "down_mode": {"vals": ["cur"]},
}


def build_base_panel() -> pd.DataFrame:
    """窄列集重建面板 (等价 load_panel 慢牛所需, 免标签免宽表深拷贝)."""
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        df = pd.read_parquet(ckpt, columns=NEED)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    work["board"] = work["symbol"].map(board_of)
    work, gate = tradability_gate(work)
    work = indicators.prepare_adx(work)
    signals.add_signal_columns(work)  # 生产默认信号; 后续按配置覆盖
    print(
        f"基础面板 rows={len(work):,} stocks={work['symbol'].nunique():,} "
        f"dates={work['date'].nunique():,} | 可交易门剔除 "
        f"{gate['removed_stocks']} 只",
        flush=True,
    )
    return work


def apply_config(work: pd.DataFrame, adx_spec: dict, regime_spec: dict) -> None:
    """就地按配置重算: 卖出信号列 + 硬门槛 + adx_score + 市场状态."""
    signals.sell_signals(work, adx_spec)
    work["gate_slow_bull"] = screener.slow_bull_gate(work, adx_spec)
    work["adx_score"] = work["adx"].clip(lower=0.0, upper=float(adx_spec["adx_optimal_max"]))
    signals.add_market_regime(work, regime_spec)


def gen_picks(work: pd.DataFrame, board: str) -> pd.DataFrame:
    """每日 Top-N 池 (gate 行 + 权重打分截面 TOP-N), 同 _slowbull_picks 口径."""
    wb = work[(work["board"] == board) & work["gate_slow_bull"]]
    if wb.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    score = pool_score(wb, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
    top = select_topn(wb, score, SLOW_BULL.top_n)
    return top[["symbol", "date"]].copy()


def build_arrays(work: pd.DataFrame) -> dict:
    """按 symbol/date 排序对齐的模拟数组 (any_sell 因 ADX 配置而异, 逐配置补)."""
    uniques, codes = np.unique(work["symbol"].values, return_inverse=True)
    sizes = np.bincount(codes)
    starts = np.zeros(len(uniques), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    ends = np.cumsum(sizes)
    A = {
        "sym_code": {s: int(c) for c, s in enumerate(uniques)},
        "starts": starts,
        "ends": ends,
        "dates": work["date"].values.astype("datetime64[ns]"),
        "close": work["close_hfq"].values,
        "low": work["low_hfq"].values,
        "ma20": work["ma20"].values,
        "cost": COST + 2 * np.array([slippage_tier(v) for v in work["adv20"].values]),
    }
    return A


def update_arrays(work: pd.DataFrame, A: dict) -> None:
    """按当前配置就地更新配置相关数组: any_sell + regime_lut.

    work 恒保持 [symbol,date] 排序 → 与 A 各数组同序; regime_lut 必须逐配置重建
    (apply_config 按 ma_window/down_mode 重算了 slow_bull_regime).
    """
    any_sell = np.zeros(len(work), dtype=bool)
    for c in SELL_COLS:
        any_sell |= work[c].values.astype(bool)
    A["any_sell"] = any_sell
    A["regime_lut"] = dict(
        zip(
            work["date"].values,
            work["slow_bull_regime"].values,
            strict=False,
        )
    )


def exit_rets(
    picks: pd.DataFrame,
    A: dict,
    mode: str,
    trail_pct: float,
    hard_stop: float,
    max_hold: int,
    adx_spec: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """逐 pick 模拟退出, 返回与 picks 行对齐的 (净收益, 持有天数). NaN=跳过.

    mode: cur (6 信号 + 移动止盈) / trail (收盘自峰值回落 trail_pct 走 + 硬止损).
    """
    sym_code, starts, ends = A["sym_code"], A["starts"], A["ends"]
    dates_dt, close, low, ma20 = A["dates"], A["close"], A["low"], A["ma20"]
    any_sell, cost_arr = A["any_sell"], A["cost"]
    M = int(max_hold)
    hard = float(hard_stop)
    rets = np.full(len(picks), np.nan)
    holds = np.full(len(picks), np.nan)
    for i, (sym, T) in enumerate(
        zip(picks["symbol"], picks["date"], strict=False)
    ):
        c = sym_code[str(sym)]
        lo, hi = starts[c], ends[c]
        base = lo + int(np.searchsorted(dates_dt[lo:hi], np.datetime64(T)))
        r0 = base + 1
        if r0 + M >= hi:  # 未来窗不足 → 弃 (保守, 同生产)
            continue
        entry = close[r0]
        if not np.isfinite(entry) or entry <= 0:
            continue
        cost = cost_arr[base]
        peak = 0.0
        exit_ret = None
        for k in range(1, M + 1):
            r = base + k
            if mode == "cur":
                peak = max(peak, low[r] / entry - 1.0)
                stop_hit = low[r] < signals.trailing_stop_price(
                    entry, peak, ma20[r], adx_spec
                )
                if any_sell[r] or stop_hit:
                    exit_ret = close[r] / entry - 1 - cost
                    holds[i] = k
                    break
            else:
                ret = close[r] / entry - 1.0
                peak = max(peak, ret)
                if ret <= peak - trail_pct or ret <= hard - 1.0:
                    exit_ret = close[r] / entry - 1 - cost
                    holds[i] = k
                    break
        if exit_ret is None:
            exit_ret = close[r0 + M] / entry - 1 - cost
            holds[i] = M
        rets[i] = exit_ret
    return rets, holds


def _agg(rets: np.ndarray, holds: np.ndarray, dates: np.ndarray) -> dict:
    m = np.isfinite(rets)
    rr = rets[m]
    if not len(rr):
        return {"n": 0, "realized": None, "p_win": None, "median_hold": None, "max_dd": None}
    hh = holds[m]
    order = np.argsort(dates[m], kind="stable")  # 按入场日排序 → 顺序权益曲线
    eq = np.cumprod(1 + rr[order])
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    return {
        "n": int(len(rr)),
        "realized": round(float(rr.mean()), 5),
        "p_win": round(float((rr > 0).mean()), 4),
        "median_hold": round(float(np.median(hh)), 1),
        "max_dd": round(-dd, 5),
    }


def _concentration(rets: np.ndarray) -> dict:
    rr = rets[np.isfinite(rets)]
    if not len(rr):
        return {"best_trade": None, "frac_of_total": None}
    total = float(rr.sum())
    best = float(np.max(rr))
    return {"best_trade": round(best, 4), "frac_of_total": round(best / total, 3) if total > 0 else None}


def eval_board(
    picks_full: pd.DataFrame,
    A: dict,
    adx_spec: dict,
    regime_spec: dict,
    oos_start: np.datetime64,
) -> dict:
    if picks_full.empty:
        return {"n_picks_oos": 0, "note": "无 gate 候选"}
    p = picks_full[picks_full["date"] >= oos_start].copy()
    if p.empty:
        return {"n_picks_oos": 0, "note": "OOS 无候选"}
    up = p["date"].map(A["regime_lut"]).fillna(False).values
    trail_pct, hard = regime_spec["trail_pct"], regime_spec["hard_stop"]
    M, down_mode = regime_spec["max_hold"], regime_spec["down_mode"]
    tr, th = exit_rets(p, A, "trail", trail_pct, hard, M, adx_spec)
    cr, ch = exit_rets(p, A, "cur", trail_pct, hard, M, adx_spec)
    op_r = np.full(len(p), np.nan)
    op_h = np.full(len(p), np.nan)
    op_r[up] = tr[up]
    op_h[up] = th[up]
    if down_mode == "cur":
        op_r[~up] = cr[~up]
        op_h[~up] = ch[~up]
    dates = p["date"].values
    return {
        "n_picks_oos": int(len(p)),
        "op_rule": _agg(op_r, op_h, dates),
        "split": {
            "up": _agg(tr[up], th[up], dates[up]),
            "down": _agg(cr[~up], ch[~up], dates[~up]),
        },
        "cur_all": _agg(cr, ch, dates),
        "concentration": _concentration(op_r),
    }


def build_configs() -> list[dict]:
    cfgs = [{"name": "prod_ref", "kind": "ref", "param": None, "value": None,
             "adx": deepcopy(ADX_SPEC), "regime": deepcopy(SLOW_BULL_REGIME)}]
    for param, spec in ADX_PERTURBS.items():
        prod_val = ADX_SPEC[param]
        for v in spec["vals"]:
            adx = deepcopy(ADX_SPEC)
            adx[param] = v
            cfgs.append({"name": f"{param}={v}", "kind": "adx", "param": param,
                         "value": v, "prod": prod_val, "adx": adx,
                         "regime": deepcopy(SLOW_BULL_REGIME)})
    for param, spec in REGIME_PERTURBS.items():
        prod_val = SLOW_BULL_REGIME[param]
        for v in spec["vals"]:
            reg = deepcopy(SLOW_BULL_REGIME)
            reg[param] = v
            cfgs.append({"name": f"{param}={v}", "kind": "regime", "param": param,
                         "value": v, "prod": prod_val, "adx": deepcopy(ADX_SPEC),
                         "regime": reg})
    return cfgs


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("构建基础窄面板 (免 label, 免宽表 OOM)...", flush=True)
    work = build_base_panel()
    dates = np.sort(work["date"].unique())
    oos_start = dates[-OOS_DAYS]
    oos63_start = dates[-OOS_CROSSCHECK_DAYS]
    print(
        f"全窗 {dates[0]} → {dates[-1]} ({len(dates)}d) | "
        f"OOS {OOS_DAYS}d 起 {oos_start} | 63d 起 {oos63_start}",
        flush=True,
    )
    A = build_arrays(work)  # 基础数组 (any_sell 逐配置补)
    cfgs = build_configs()
    print(f"共 {len(cfgs)} 个配置 (prod_ref + {len(cfgs) - 1} 扰动)", flush=True)

    out = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "objective": "慢牛实得回测: op_rule=上升段trail退出+下降段按down_mode; "
                     "cur_all=全部cur退出(参考). 验收只看 OOS 末 250 交易日",
        "oos_days": OOS_DAYS,
        "window": {
            "full": {"start": str(dates[0]), "end": str(dates[-1]), "n_days": int(len(dates))},
            "oos": {"start": str(pd.Timestamp(oos_start).date()), "end": str(pd.Timestamp(dates[-1]).date()), "n_days": OOS_DAYS},
            "oos63": {"start": str(pd.Timestamp(oos63_start).date()), "end": str(pd.Timestamp(dates[-1]).date()), "n_days": OOS_CROSSCHECK_DAYS},
        },
        "entry_exit": "close_hfq[T+1] 入场, 收盘判定退出, 成本=COST+2×滑点",
        "max_dd_note": "顺序权益曲线近似 (每笔=1单位, 前一笔了结后续接), 用于跨配置横向对比",
        "configs": [],
    }

    results = []
    for ci, cfg in enumerate(cfgs):
        apply_config(work, cfg["adx"], cfg["regime"])
        update_arrays(work, A)
        boards = {}
        for b in ("main", "dual"):
            pk = gen_picks(work, b)
            boards[b] = eval_board(pk, A, cfg["adx"], cfg["regime"], oos_start)
            del pk
            gc.collect()
        cfg_out = {k: cfg.get(k) for k in ("name", "kind", "param", "value", "prod")}
        cfg_out["boards"] = boards
        results.append(cfg_out)
        out["configs"].append(cfg_out)
        if cfg["name"] == "prod_ref":
            # 63d 交叉核对 (先前 trail8 实得: main +1.58% / dual +1.88%)
            ck = {"name": "prod_ref_oos63"}
            for b in ("main", "dual"):
                pk = gen_picks(work, b)
                ck[b] = eval_board(pk, A, cfg["adx"], cfg["regime"], oos63_start)
                del pk
                gc.collect()
            out["oos63_prod_ref"] = ck
        mop = boards["main"].get("op_rule", {})
        dop = boards["dual"].get("op_rule", {})
        print(f"[{ci + 1}/{len(cfgs)}] {cfg['name']:<22} "
              f"main op={mop.get('realized')} n={mop.get('n')} | "
              f"dual op={dop.get('realized')} n={dop.get('n')}", flush=True)

    del work, A
    gc.collect()

    # ── WORM 落盘 ──
    ts = out["ts"]
    run_dir = BACKTEST_RESULT_DIR / f"diag_slowbull_paramsweep_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = run_dir / "paramsweep.json"
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)

    # ── summary.txt (逐参数对比 prod_ref) ──
    ref = next(r for r in results if r["name"] == "prod_ref")
    lines = [
        "=" * 90,
        f"慢牛参数扫描 (单因子, 围绕生产默认双向扰动) | OOS 末 {OOS_DAYS} 交易日",
        "=" * 90,
        f"面板 {out['window']['full']['start']} → {out['window']['full']['end']} "
        f"({out['window']['full']['n_days']}d)",
        f"prod_ref (op_rule realized/p_win/n/max_dd, cur_all in 括号):",
    ]
    for b in ("main", "dual"):
        r = ref["boards"][b].get("op_rule", {})
        c = ref["boards"][b].get("cur_all", {})
        if r.get("realized") is None:
            lines.append(f"  [{b}] n/a")
            continue
        lines.append(
            f"  [{b}] op={r['realized'] * 100:+.2f}% wr={r['p_win']:.0%} "
            f"n={r['n']} dd={r['max_dd']:.1%} "
            f"(cur_all: {c['realized'] * 100:+.2f}% n={c['n']})"
        )
    lines.append("-" * 90)
    # 按参数分组打印
    seen = []
    for r in results:
        if r["name"] == "prod_ref":
            continue
        key = r["param"]
        if key not in seen:
            seen.append(key)
            prod_val = r.get("prod")
            lines.append(f"\n### {key}  (prod={prod_val})")
            lines.append(f"    {'cfg':<16} | {'main':>42} | {'dual':>42}")
        mb = r["boards"]["main"].get("op_rule", {})
        db = r["boards"]["dual"].get("op_rule", {})
        fmt = lambda x: (f"{x['realized'] * 100:+.2f}%/wr{x['p_win']:.0%}/"
                         f"n{x['n']}/dd{x['max_dd']:.1%}") if x.get("realized") is not None else "n/a"
        lines.append(f"    {r['name']:<16} | {fmt(mb):>42} | {fmt(db):>42}")
    lines.append("-" * 90)
    lines.append(f"落盘: {fp}")
    stxt = run_dir / "summary.txt"
    with open(stxt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n落盘目录: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
