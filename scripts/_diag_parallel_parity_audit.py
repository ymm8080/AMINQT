"""_diag_parallel_parity_audit.py — parallel 模块 train/infer 特征对齐审计 (2026-09-03).

背景: legacy brute 推理缺失修复 (+60% 真赢家) 证明"训练看得见、推理喂不对"类
缺口是 TOP10 最大杠杆. parallel 概率头训练/推理同源 (都读 _diag_stage_{board}_3y
.parquet, 缺列 predict 直接 raise), 故 legacy 式硬缺口结构性不存在; 真正未审过的
风险面是**检查点尾段特征健康度** (09-02 新鲜度审计只查了 max(date), 没查逐列):

  训练史有值 → 推理尾段 NaN/停更/塌缩/漂移 = 模型按历史分布学的分裂, 推理喂的是
  另一个分布. prob 闸 + blend 排名键 (mag × prob) 全部吃这个检查点.

审计口径 (对每板最新 prob bundle 的 feat_cols 逐列):
  hist = 训练行 (date <= bundle.trained_through), tail = 最后 21 交易日 (refit 窗).
  - schema_missing: bundle feat_cols 不在检查点 schema (闸已死, predict 会 raise)
  - silent_stop:    hist 非空率 >= 0.9 且 tail <= 0.1 (曾稠密 → 现停更)
  - nonnull_drop:   hist >= 0.5 且掉幅 >= 0.30 (显著变稀)
  - const_tail:     tail 非空 >= 0.5 且 std == 0 (塌缩成常数)
  - zero_collapse:  hist 零率 <= 0.05 且 tail >= 0.9 (塌缩成零)
  - drift5:         tail 非空 >= 0.3 且 |tail中位 - hist中位| >= 5*hist_std (分布漂移)
  - inf_tail:       tail 出现 ±inf (LGBM 推理会炸, 闸死)
  另: pred_ 列 (blend 排名输入) tail 非空率 + 最后非空日; bundle 年龄 (交易日).

只读, 不改任何数据/模型. WORM: DATA OTHERS/diag/parallel_parity_audit_<ts>.json + .parquet
用法: python scripts/_diag_parallel_parity_audit.py [--tail 21]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel import prob_head
from config.settings import DATA_DIR, data_others_path

CHUNK = 48  # 每次读的列数 (控内存, 1.6M 行 × 48 列 float64 ≈ 0.6GB 峰值)


def _col_stats(
    v: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, float, float]:
    """单列单段统计: (nonnull率, 零率, std, 中位, inf率). 空段 → (0, nan, 0, nan, 0)."""
    x = v[mask]
    if x.size == 0:
        return 0.0, float("nan"), 0.0, float("nan"), 0.0
    fin = np.isfinite(x)
    inf_rate = float((~fin).sum() - np.isnan(x).sum()) / x.size
    n = x[fin]
    nonnull = float(n.size) / x.size
    if n.size:
        return (
            nonnull,
            float((n == 0).mean()),
            float(n.std()),
            float(np.median(n)),
            inf_rate,
        )
    return nonnull, float("nan"), 0.0, float("nan"), inf_rate


def audit_board(board: str, tail_days: int) -> dict:
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    schema = pq.read_schema(str(fp)).names
    b = prob_head.load_latest(board)
    if b is None:
        return {"board": board, "fatal": "无概率头 bundle"}
    feat_cols = list(b["feat_cols"])
    schema_missing = [c for c in feat_cols if c not in schema]

    dates = pd.to_datetime(
        pq.read_table(str(fp), columns=["date"]).to_pandas()["date"]
    ).to_numpy()
    uniq = np.unique(dates)
    if len(uniq) < tail_days + 5:
        return {"board": board, "fatal": f"检查点日期不足 ({len(uniq)})"}
    tt = np.datetime64(pd.Timestamp(str(b["trained_through"])))
    tail_start = uniq[-tail_days]
    hist_mask = dates <= tt
    tail_mask = dates >= tail_start
    age = prob_head.bundle_age_trading_days(uniq, str(b["trained_through"]))

    rows: list[dict] = []
    flagged: dict[str, list[str]] = {}
    cols_to_scan = [c for c in feat_cols if c in schema]
    for i in range(0, len(cols_to_scan), CHUNK):
        chunk = cols_to_scan[i : i + CHUNK]
        tbl = pq.read_table(str(fp), columns=chunk)
        for c in chunk:
            v = np.asarray(
                tbl.column(c).to_numpy(zero_copy_only=False), dtype="float32"
            )
            hn, hz, hs, hm, _ = _col_stats(v, hist_mask)
            tn, tz, ts_, tm, tinf = _col_stats(v, tail_mask)
            flags = []
            if hn >= 0.90 and tn <= 0.10:
                flags.append("silent_stop")
            if hn >= 0.50 and (hn - tn) >= 0.30 and "silent_stop" not in flags:
                flags.append("nonnull_drop")
            if tn >= 0.50 and ts_ == 0.0:
                flags.append("const_tail")
            if (
                (not np.isnan(hz))
                and hz <= 0.05
                and (not np.isnan(tz))
                and tz >= 0.90
                and tn >= 0.50
            ):
                flags.append("zero_collapse")
            if hs > 0 and tn >= 0.30 and (not np.isnan(tm)) and abs(tm - hm) >= 5 * hs:
                flags.append("drift5")
            if tinf > 0:
                flags.append("inf_tail")
            for f in flags:
                flagged.setdefault(f, []).append(c)
            if flags or tn < 0.99:
                rows.append(
                    {
                        "board": board,
                        "col": c,
                        "hist_nonnull": hn,
                        "tail_nonnull": tn,
                        "hist_zero": hz,
                        "tail_zero": tz,
                        "hist_std": hs,
                        "tail_std": ts_,
                        "hist_med": hm,
                        "tail_med": tm,
                        "tail_inf": tinf,
                        "flags": ";".join(flags),
                    }
                )
        del tbl

    # pred_ 列 = blend 排名输入, tail 非空率 + 最后非空日
    pred_cols = [c for c in schema if c.startswith("pred_")]
    ptbl = pq.read_table(str(fp), columns=["date"] + pred_cols).to_pandas()
    ptbl["date"] = pd.to_datetime(ptbl["date"])
    ptail = ptbl[ptbl["date"] >= pd.Timestamp(tail_start)]
    pred_health = {}
    for c in pred_cols:
        s = ptail[c]
        nonnull_idx = ptail.loc[s.notna(), "date"]
        pred_health[c] = {
            "tail_nonnull": round(float(s.notna().mean()), 4),
            "last_nonnull": str(nonnull_idx.max().date()) if len(nonnull_idx) else None,
        }
    del ptbl, ptail

    return {
        "board": board,
        "bundle": {
            "name": sorted(prob_head.bundle_dir().glob(f"{board}_prob_*.joblib"))[
                -1
            ].name,
            "trained_through": str(b["trained_through"]),
            "feat_cols_n": len(feat_cols),
            "age_trading_days": age,
        },
        "checkpoint": {
            "max_date": str(pd.Timestamp(uniq[-1]).date()),
            "rows": int(dates.size),
            "cols_schema": len(schema),
        },
        "schema_missing": schema_missing,
        "flags": flagged,
        "pred_cols_tail": pred_health,
        "_colrows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=21)
    args = ap.parse_args()

    out_dir = data_others_path("diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows: list[dict] = []
    summary = {}
    for board in ("main", "dual"):
        print(f"{'=' * 72}\n[{board}] 审计中...", flush=True)
        res = audit_board(board, args.tail)
        all_rows.extend(res.pop("_colrows", []))
        summary[board] = res
        print(
            f"  bundle: {res['bundle']['name']} trained_through="
            f"{res['bundle']['trained_through']} 年龄={res['bundle']['age_trading_days']} 交易日"
        )
        print(
            f"  检查点: max={res['checkpoint']['max_date']} "
            f"rows={res['checkpoint']['rows']:,} schema={res['checkpoint']['cols_schema']}列"
        )
        if res.get("fatal"):
            print(f"  FATAL: {res['fatal']}")
            continue
        if res["schema_missing"]:
            print(
                f"  FATAL schema_missing {len(res['schema_missing'])} 列 (闸已死): "
                f"{res['schema_missing'][:10]}"
            )
        for f, cols in res["flags"].items():
            print(f"  [{f}] {len(cols)} 列: {cols[:15]}")
        bad_pred = {
            c: h for c, h in res["pred_cols_tail"].items() if h["tail_nonnull"] < 0.99
        }
        if bad_pred:
            print(f"  [pred_尾段] 非空<99%: {json.dumps(bad_pred, ensure_ascii=False)}")
        else:
            print("  [pred_尾段] 全部健康")

    pq_path = out_dir / f"parallel_parity_audit_{ts}.parquet"
    pd.DataFrame(all_rows).to_parquet(pq_path, index=False)
    js_path = out_dir / f"parallel_parity_audit_{ts}.json"
    js_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"{'=' * 72}\nWORM: {pq_path}\n      {js_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
