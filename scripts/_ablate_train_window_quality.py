"""训练数据窗消融: 固定同一 OOS 日期区间, 把面板行裁剪到末 3y/2y/1y, 验证验收是否不变.

动机 (2026-08-10): 用户问"3y→2y 是否伤最终预测质量, 1y 是否大幅省内存".
并行系统打分 = 每日期截面分位 (pool_score) + ≤60d 有界特征回看 (VAR51/ret_reversal_5d/
pv_corr_5/limit_dist_pct/rps_60/ADX 均 ≤60d), 无全历史 expanding 特征进打分池;
验收只看末 6m/3m/10d OOS → 旧行理论上不参与打分.

本脚本实证 (2026-08-10 定稿):
  * 3y / 2y / 1y 均用**列子集读取** (只读验收链需要的 18 列, 从检查点 557 列缩到 18 列,
    基帧 ~30× 缩小 → 3y 全量也能跑, 根治此前 2.74GiB OOM 的 block-consolidation 陷阱),
    复制 load_panel 全部步骤, 同一批特征值 (预计算检查点), 同一固定 OOS 日期区间.
  * 自验证: 3y-reduced 与今天生产的 backtest JSON (20260810_125705, 全 557 列 run_all)
    逐位对比 (mag/winrate/n/ok) → 证明列子集读取忠实. 若忠实, 则 2y/1y 的差异是**真窗口效应**.
  * 不变性: 2y/1y vs 3y-reduced, 全 OOS (6m/3m/10d) × 全系统 × 全视界 逐位对比.

内存口径: 诚实回答"剪到 N 年省多少内存" → 用**全检查点 schema (557 列) × 末 N 天行数**
估基帧内存 (真裁剪是把存档行数减到 N 天, 列仍是全量).

诚实边界: 裁的是**已预计算特征**的检查点行集. 若真按 1y 重建特征, ≤60d 有界特征在 OOS
(末 126d) 前有 ≥124d 回看, 暖机充足 → 与本次等价; 全历史 expanding 特征 (不进打分池) 会
不同但不影响打分. 对 legacy LGBM 重训 (WINDOW_TOTAL) 不适用.

用法: python scripts/_ablate_train_window_quality.py
输出: data/_ablate_train_window_quality_<ts>.json (WORM) + 控制台对比表
"""

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from app.pipeline_parallel import indicators, screener, signals
from app.pipeline_parallel.backtest import (
    add_c2c_labels,
    add_mfe_labels,
    board_of,
    run_system,
    tradability_gate,
)
from app.pipeline_parallel.config import (
    ALL_HORIZON_INTS,
    BOARD_PREFIXES,
    BOARD_THRESHOLDS,
    OOS_WINDOWS,
    PANEL,
    SLOW_BULL_REGIME,
    SYSTEMS,
)
from app.pipeline_parallel.scoring import dual_head_ok, pool_score, select_topn

# 末 N 个交易日 (3y=726=现状全量, 2y=484, 1y=242)
WINDOWS: dict[str, int] = {"3y": 726, "2y": 484, "1y": 242}
JSON_3Y = "D:/AMINQT/DATA OTHERS/BACKTESTING RESULT/20260810_125705/backtest.json"
FULL_SCHEMA_COLS = 557  # 检查点列数 (内存口径用)

# 验收链 (load_panel→labels→gate→prepare_adx→signals→slow_bull→run_system) 实际读的
# 检查点列. pv_corr_5 由 prepare_adx 重算, pct_70_con 由 _merge_pct_70 从 cyq_panel 合入,
# 均非检查点直读 → 不在列集里. 任何遗漏由 _assert_spec_columns 大声失败兜底.
READ_COLS = [
    "symbol",
    "date",
    "close_hfq",
    "high_hfq",
    "low_hfq",
    "open_hfq",
    "volume",
    "volume_ratio",
    "turnover_rate",
    "adv20",
    "is_suspended",
    "margin_balance_chg_5d",
    "amihud_illiq",
    "small_mv_premium",
    "amihud_illiquidity",
    "VAR51",
    "ret_reversal_5d",
    "limit_dist_pct",
]


def est_gb(rows: int, cols: int) -> float:
    """粗估 float64 主导面板基帧内存 (GB)."""
    return rows * cols * 8 / 1e9


def norm_ok(per_horizon: dict, board: str) -> dict:
    """规范化每视界 {mag,winrate,n,ok}, ok 用当前板块阈值统一重算."""
    th = BOARD_THRESHOLDS[board]
    return {
        h: {
            "mag": r["mag"],
            "winrate": r["winrate"],
            "n": r["n"],
            "ok": dual_head_ok(r, th["min_winrate"], th["min_mag"]),
        }
        for h, r in per_horizon.items()
    }


def _assert_spec_columns(work: pd.DataFrame) -> None:
    """大声失败: 验收链所需列 (池/门槛/标签) 必须齐全, 防止列子集读取静默跳列."""
    needed: set[str] = {"symbol", "date"}
    for spec in SYSTEMS.values():
        if not spec.enabled:
            continue
        needed |= set(spec.pool)
        if spec.gate:
            needed.add(f"gate_{spec.gate}")
        needed |= set(spec.labels)
    missing = sorted(c for c in needed if c not in work.columns)
    if missing:
        raise SystemExit(
            f"[FATAL] 列子集读取漏列, 验收链会静默跳列 → 结果不可信: {missing}"
        )


def load_window(days: int):
    """复制 load_panel 全步骤, 但读取时 pyarrow 过滤到末 `days` 交易日 + 只读验收列."""
    from scripts._reclassify_all_features import _finalize_slice

    if days < 726:
        dts = pd.read_parquet(PANEL.main_checkpoint, columns=["date"])["date"].unique()
        cutoff = np.sort(pd.to_datetime(pd.Series(dts)))[-days]
        print(f"  load_window({days}d): cutoff={pd.Timestamp(cutoff)}", flush=True)
    else:
        cutoff = None
        print(f"  load_window({days}d): 全量 (无日期过滤)", flush=True)
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        kw = {"columns": READ_COLS}
        if cutoff is not None:
            kw["filters"] = [("date", ">=", cutoff)]
        df = pd.read_parquet(ckpt, **kw)
        df = _finalize_slice(df)
        df = add_mfe_labels(df, horizons=ALL_HORIZON_INTS)
        df = add_c2c_labels(df, horizons=ALL_HORIZON_INTS)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    work, _gate = tradability_gate(work)
    work["board"] = work["symbol"].map(board_of)
    work = indicators.prepare_adx(work)
    signals.add_signal_columns(work)
    work["gate_slow_bull"] = screener.compute_gate(work, "slow_bull")
    signals.add_market_regime(work, SLOW_BULL_REGIME)
    gc.collect()
    _assert_spec_columns(work)
    return work


def _picks(sub: pd.DataFrame, spec, top_n: int, bm: np.ndarray) -> list:
    """复刻 run_system 的选股 (门槛→池打分→TOP-N), 返回 [(symbol, date_str)] 供跨窗对比."""
    s = sub[bm]
    gc_col = f"gate_{spec.gate}" if spec.gate else None
    if gc_col and gc_col in s.columns:
        s = s[s[gc_col]]
    if s.empty:
        return []
    score = pool_score(s, spec.pool, weights=spec.pool_weights)
    top = select_topn(s, score, top_n)
    return sorted(zip(top["symbol"], top["date"].astype(str)))


def accept_board(work: pd.DataFrame, board: str, bcrit: tuple[float, float]) -> dict:
    """对单板块跑全部启用系统的全部 OOS 窗 → {system: {lab/kind: {...}, lab/kept}} + picks."""
    sub = work[work["board"] == board]
    dates = np.sort(sub["date"].unique())
    oos_cutoff = {lab: dates[-d] for lab, d in OOS_WINDOWS.items()}
    res: dict = {"rows": int(len(sub)), "picks": {}}
    for name, spec in SYSTEMS.items():
        if not spec.enabled:
            continue
        entry: dict = {}
        picks: dict = {}
        for lab, _d in OOS_WINDOWS.items():
            bm = sub["date"].values >= oos_cutoff[lab]
            for kind, tn in (("primary", spec.top_n), ("alt", spec.top_n_alt)):
                r = run_system(sub, spec, tn, bm, crit=bcrit)
                entry[f"{lab}/{kind}"] = norm_ok(r["per_horizon"], board)
                entry[f"{lab}/{kind}"]["passed"] = list(r["passed"])
                picks[f"{lab}/{kind}"] = _picks(sub, spec, tn, bm)
            entry[lab + "/kept"] = bool(
                entry[f"{lab}/primary"]["passed"] or entry[f"{lab}/alt"]["passed"]
            )
        res[name] = entry
        res["picks"][name] = picks
        gc.collect()
    return res


def extract_json_window(prod: dict, board: str, name: str, spec) -> dict:
    """从生产 JSON (全 557 列 run_all) 提取某系统全部 OOS 窗的结果, ok 用当前阈值重算."""
    rec: dict = {}
    for lab in OOS_WINDOWS:
        o = prod["boards"][board]["systems"][name].get("oos", {}).get(lab, {})
        pr = o.get("primary") or {}
        ph = pr.get("per_horizon") or {}
        if ph:
            r = norm_ok(ph, board)
            rec[f"{lab}/primary"] = r
            rec[f"{lab}/primary"]["passed"] = [h for h in spec.horizons if r[h]["ok"]]
        else:
            rec[f"{lab}/primary"] = {"passed": []}
        alt = o.get("alt") or {}
        a_ph = alt.get("per_horizon") or {}
        if a_ph:
            a = norm_ok(a_ph, board)
            rec[f"{lab}/alt"] = a
            rec[f"{lab}/alt"]["passed"] = [h for h in spec.horizons if a[h]["ok"]]
        else:
            rec[f"{lab}/alt"] = {"passed": []}
        rec[f"{lab}/kept"] = bool(
            rec[f"{lab}/primary"]["passed"] or rec[f"{lab}/alt"]["passed"]
        )
    return rec


def cell(rec: dict, lab: str, kind: str, h: str) -> dict:
    return rec.get(f"{lab}/{kind}", {}).get(h, {}) or {}


def same_cell(a: dict, b: dict) -> bool:
    if a.get("ok") != b.get("ok") or a.get("n") != b.get("n"):
        return False
    if abs((a.get("winrate") or 0) - (b.get("winrate") or 0)) > 1e-12:
        return False
    if abs((a.get("mag") or 0) - (b.get("mag") or 0)) > 1e-9:
        return False
    return True


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    if not os.path.exists(JSON_3Y):
        raise SystemExit(f"3y 基线 JSON 不存在: {JSON_3Y}")
    prod = json.load(open(JSON_3Y, encoding="utf-8"))
    print(f"3y 生产基线: {JSON_3Y} (今日 {prod['ts']})", flush=True)

    out: dict = {
        "ts": ts,
        "question": "3y→2y/1y 裁剪是否改变并行验收 (固定同一 OOS 日期区间, 同特征值, 列子集读取自验证)",
        "windows_days": WINDOWS,
        "json_3y": JSON_3Y,
        "json_3y_ts": prod["ts"],
        "rows_full_3y_postgate": prod["rows"],
        "read_cols": READ_COLS,
        "windows": {},
        "boards": {},
    }
    for b in BOARD_PREFIXES:
        out["boards"][b] = {}

    # ── ref3y_json: 生产 JSON 提取 (faithfulness 参照) ──
    for b in BOARD_PREFIXES:
        out["boards"][b]["ref3y_json"] = {}
        for name, spec in SYSTEMS.items():
            if spec.enabled:
                out["boards"][b]["ref3y_json"][name] = extract_json_window(
                    prod, b, name, spec
                )
    out["windows"]["ref3y_json"] = {"rows": prod["rows"]}

    # ── 3y / 2y / 1y: 列子集读取 + 复制 load_panel + 重跑验收 ──
    for wname, wdays in WINDOWS.items():
        print(f"\n===== {wname} (末 {wdays} 交易日) =====", flush=True)
        work = load_window(wdays)
        print(
            f"  load_window({wname}): {len(work):,} 行 / {work['symbol'].nunique():,} 只 / "
            f"{work['date'].nunique()} 交易日 / 读取 {len(READ_COLS)} 列 / "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
        out["windows"][wname] = {
            "rows": int(len(work)),
            "stocks": int(work["symbol"].nunique()),
            "days": int(work["date"].nunique()),
            "est_gb_full_schema": round(est_gb(len(work), FULL_SCHEMA_COLS), 2),
        }
        for b in BOARD_PREFIXES:
            th = BOARD_THRESHOLDS[b]
            out["boards"][b][wname] = accept_board(
                work, b, (th["min_winrate"], th["min_mag"])
            )
        del work
        gc.collect()

    # ── 控制台对比表: 各 OOS 窗 primary 各视界 ──
    print("\n\n########## OOS 对比 (primary, 同固定日期区间, ok 同阈值) ##########")
    for lab in OOS_WINDOWS:
        print(f"\n===== OOS={lab} =====")
        for b in BOARD_PREFIXES:
            print(
                f"\n--- board={b} (min_wr={BOARD_THRESHOLDS[b]['min_winrate']} "
                f"min_mag={BOARD_THRESHOLDS[b]['min_mag']}) ---"
            )
            for name, spec in SYSTEMS.items():
                if not spec.enabled:
                    continue
                for h in spec.horizons:
                    cells = {
                        wn: cell(out["boards"][b][wn][name], lab, "primary", h)
                        for wn in list(WINDOWS) + ["ref3y_json"]
                    }
                    fmt = []
                    for wn in ("ref3y_json", "3y", "2y", "1y"):
                        hr = cells[wn]
                        n = hr.get("n", 0)
                        if n < 5:
                            fmt.append(f"{wn}: n<5")
                        else:
                            fmt.append(
                                f"{wn}: wr={hr['winrate']:.1%} m={hr['mag']:+.2%} n={n}"
                            )
                    ref3 = cells["ref3y_json"]
                    base = cells["3y"]
                    faithful = same_cell(cells["3y"], ref3)
                    inv = all(same_cell(cells[wn], base) for wn in ("2y", "1y"))
                    tag = []
                    if faithful:
                        tag.append("SAME-JSON")
                    if inv:
                        tag.append("SAME-3Y")
                    print(
                        f"  {name}/{h}: "
                        + " | ".join(fmt)
                        + (f"  [{'+'.join(tag)}]" if tag else "  [DIFF]")
                    )
            print(
                "  kept:",
                {
                    wn: {
                        sname: out["boards"][b][wn][sname].get(lab + "/kept")
                        for sname in SYSTEMS
                        if SYSTEMS[sname].enabled
                    }
                    for wn in ("ref3y_json", "3y", "2y", "1y")
                },
            )

    # ── 不变性判定: faithfulness (3y-reduced vs JSON) + invariance (2y/1y vs 3y-reduced) ──
    # picks 对比 (决定性): 同日期选出同股票 → 预测本身不变 (量测差异=窗尾标签缺失).
    faith_diffs, inv_diffs, picks_diffs = [], [], []
    for b in BOARD_PREFIXES:
        for name, spec in SYSTEMS.items():
            if not spec.enabled:
                continue
            for lab in OOS_WINDOWS:
                for kind in ("primary", "alt"):
                    for h in spec.horizons:
                        ref3 = cell(out["boards"][b]["ref3y_json"][name], lab, kind, h)
                        base = cell(out["boards"][b]["3y"][name], lab, kind, h)
                        if not same_cell(base, ref3):
                            faith_diffs.append((b, name, lab, kind, h, "3y-vs-json"))
                        for wn in ("2y", "1y"):
                            cur = cell(out["boards"][b][wn][name], lab, kind, h)
                            if not same_cell(cur, base):
                                inv_diffs.append((b, name, lab, kind, h, wn))
                    p3 = out["boards"][b]["3y"]["picks"][name][f"{lab}/{kind}"]
                    for wn in ("2y", "1y"):
                        pwn = out["boards"][b][wn]["picks"][name][f"{lab}/{kind}"]
                        if p3 != pwn:
                            picks_diffs.append(
                                (b, name, lab, kind, wn, len(p3), len(pwn))
                            )

    def dump_diff(d):
        b, name, lab, kind, h, wn = d
        return {
            "board": b,
            "system": name,
            "oos": lab,
            "kind": kind,
            "horizon": h,
            "window": wn,
            "ref": {"ok": None, "n": None},
            "cur": {"ok": None, "n": None},
        }

    out["verdict"] = {
        "faithful_3y_reduced_vs_json": len(faith_diffs) == 0,
        "picks_identical": len(picks_diffs) == 0,
        "measured_quality_invariant": len(inv_diffs) == 0,
        "n_faith_diffs": len(faith_diffs),
        "n_inv_diffs": len(inv_diffs),
        "n_picks_diffs": len(picks_diffs),
        "faith_diffs": [dump_diff(d) for d in faith_diffs[:50]],
        "inv_diffs": [dump_diff(d) for d in inv_diffs[:50]],
        "picks_diffs": [
            {
                "board": d[0],
                "system": d[1],
                "oos": d[2],
                "kind": d[3],
                "window": d[4],
                "n_3y": d[5],
                "n_wn": d[6],
            }
            for d in picks_diffs[:50]
        ],
        "note": "faithful = 列子集读取 vs 生产 JSON 逐位一致 (方法学自验证). "
        "picks_identical = 同日期选出同股票 → 预测本身对窗宽不变 (决定性). "
        "measured_quality_invariant = 量测数值逐位一致; 若 False, 差异来自窗尾标签缺失 "
        "(1y/2y 窗末尾 max_horizon+1 天无未来数据 → 标签 NaN → n 变小), 非质量变化. "
        "真 1y 重建特征时 ≤60d 特征暖机充足仍等价; legacy LGBM 重训不适用本结论.",
    }
    outp = os.path.join("data", f"_ablate_train_window_quality_{ts}.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # ── 选股集合对比 (决定性) ──
    print("\n\n########## 选股集合 (symbol×date) 跨窗对比 ##########")
    for b in BOARD_PREFIXES:
        for name, spec in SYSTEMS.items():
            if not spec.enabled:
                continue
            for lab in OOS_WINDOWS:
                for kind in ("primary", "alt"):
                    p3 = out["boards"][b]["3y"]["picks"][name][f"{lab}/{kind}"]
                    rows = [
                        (
                            wn,
                            len(out["boards"][b][wn]["picks"][name][f"{lab}/{kind}"]),
                            out["boards"][b][wn]["picks"][name][f"{lab}/{kind}"] == p3,
                        )
                        for wn in ("2y", "1y")
                    ]
                    tags = "  ".join(
                        f"{wn}: n={n} {'SAME' if ok else 'DIFF'}" for wn, n, ok in rows
                    )
                    print(f"  {b}/{name}/{lab}/{kind}: n3y={len(p3)} | {tags}")

    # ── 内存口径 ──
    print("\n\n########## 内存口径 (全检查点 schema 557 列 × 末 N 天行数) ##########")
    base3 = out["windows"]["3y"]["est_gb_full_schema"]
    for wn in ("3y", "2y", "1y"):
        gb = out["windows"][wn]["est_gb_full_schema"]
        print(
            f"  {wn}: {out['windows'][wn]['rows']:>10,} 行 ≈ {gb:6.2f} GB "
            f"({gb / base3 * 100:5.1f}% of 3y)"
        )

    print(f"\nWORM 落盘: {outp}")
    print(
        f"verdict: faithful={out['verdict']['faithful_3y_reduced_vs_json']} "
        f"| picks_identical={out['verdict']['picks_identical']} "
        f"| measured_quality_invariant={out['verdict']['measured_quality_invariant']} "
        f"| n_faith_diffs={len(faith_diffs)} n_inv_diffs={len(inv_diffs)} "
        f"n_picks_diffs={len(picks_diffs)}"
    )
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
