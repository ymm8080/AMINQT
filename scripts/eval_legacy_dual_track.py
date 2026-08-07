"""legacy 双轨影子评估: 生产 prob_up 排名 vs 影子 pred_ret_3d 幅度排名, 真实结局对比.

背景 (2026-08-07): 并行模块 250d OOS 证 pred_mag 排名赢过特征排名, 用户问是否
能应用到 legacy. legacy 两信号 (prob_up=把握度 / pred_ret=幅度) 同出一个 V35 模型,
主板 Spearman +0.74 近重合, GEM/STAR 反相关 (换=选相反名单). 无干净 250d 历史
底稿 → 落地为双轨影子: 每日同一份候选按两键各出一份落 prediction DB, 1~2 月后
真实结局对比 (零偷看零风险).

本脚本读 prediction DB:
  生产排名 = prediction_stocks.rank (d3 混合/prob_up 口径)
  影子排名 = prediction_shadow.pred_ret_3d 降序 (幅度口径)
按 (date, board) 各取 TOP-5, 对比已实现 actual_ret_3d/5d (均值/上涨率/重合度).
仅统计 T+3 已成熟 (actual_ret_3d 非空) 的日期.

用法: python scripts/eval_legacy_dual_track.py
输出 (WORM) → BACKTEST_RESULT_DIR/legacy_dual_track_<ts>/
  daily_3d.csv   逐日逐板块明细
  dual_track_3d.csv / dual_track_5d.csv  按板块汇总
  summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.prediction_db import PredictionDB
from config.settings import BACKTEST_RESULT_DIR

TOP_N = 5  # 每板块对比档位 (镜像并行短名单 TOP-5)
HORIZONS = ("3d", "5d")


def _top_by(df: list[dict], key: str, ascending: bool, board: str, n: int = TOP_N):
    sub = [r for r in df if r.get("board") == board and pd.notna(r.get(key))]
    sub = sorted(sub, key=lambda r: (r[key] if r[key] is not None else float("-inf")), reverse=not ascending)
    return sub[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="legacy 双轨影子评估")
    ap.add_argument("--db", default=None, help="prediction DB 路径 (默认生产 DB)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"legacy_dual_track_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    db = PredictionDB(path=args.db) if args.db else PredictionDB()
    dates = [sd["date"] for sd in db.list_shadow_dates(200)]
    if not dates:
        print("[fatal] 无影子排名入库 (需每日 legacy 预测跑一段时间)")
        return 1

    daily_rows = []
    summary: dict = {
        "ts": ts,
        "type": "legacy_dual_track",
        "top_n": TOP_N,
        "note": "生产=prediction_stocks.rank (prob_up/d3混合) vs 影子=pred_ret_3d 幅度",
        "boards": {},
    }
    for date_str in dates:
        run = db.get_run(date_str)
        shadow = db.get_shadow(date_str)
        if not run or not run["stocks"] or not shadow:
            continue
        prod = run["stocks"]
        for board in ("main", "GEM", "STAR"):
            p_top = _top_by(prod, "rank", True, board)
            s_top = _top_by(shadow, "pred_ret_3d", False, board)
            # 只统计已实现 T+3 成熟的股票
            p_mat = [r for r in p_top if pd.notna(r.get("actual_ret_3d"))]
            s_mat = [r for r in s_top if pd.notna(r.get("actual_ret_3d"))]
            if not p_mat or not s_mat:
                continue
            p3 = pd.Series([r["actual_ret_3d"] for r in p_mat])
            s3 = pd.Series([r["actual_ret_3d"] for r in s_mat])
            p5 = pd.Series(
                [r["actual_ret_5d"] for r in p_mat if pd.notna(r.get("actual_ret_5d"))]
            )
            s5 = pd.Series(
                [r["actual_ret_5d"] for r in s_mat if pd.notna(r.get("actual_ret_5d"))]
            )
            overlap = len(
                set(r["symbol"] for r in p_mat) & set(r["symbol"] for r in s_mat)
            )
            row = {
                "date": date_str,
                "board": board,
                "n_prod": len(p_mat),
                "n_shadow": len(s_mat),
                "overlap_top5": overlap,
                "prod_mean_3d": round(float(p3.mean()), 5),
                "shadow_mean_3d": round(float(s3.mean()), 5),
                "prod_win_3d": round(float((p3 > 0).mean()), 4),
                "shadow_win_3d": round(float((s3 > 0).mean()), 4),
            }
            if len(p5) and len(s5):
                row["prod_mean_5d"] = round(float(p5.mean()), 5)
                row["shadow_mean_5d"] = round(float(s5.mean()), 5)
                row["prod_win_5d"] = round(float((p5 > 0).mean()), 4)
                row["shadow_win_5d"] = round(float((s5 > 0).mean()), 4)
            daily_rows.append(row)

    if not daily_rows:
        print("[warn] 无同时具备成熟结局的生产+影子配对日 (需等 T+3 成熟)")
        return 0

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(out_dir / "daily_3d.csv", index=False)

    agg_rows = []
    print("========== legacy 双轨影子: 生产(把握度) vs 影子(幅度) TOP-5 ==========", flush=True)
    print(f"{'板块':<6}{'日':>4}{'重合':>6}{'生产均3d':>10}{'影子均3d':>10}"
          f"{'Δ3d':>9}{'生产涨率':>9}{'影子涨率':>9}", flush=True)
    for board in ("main", "GEM", "STAR"):
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        r = sub.mean(numeric_only=True)
        d3 = r["shadow_mean_3d"] - r["prod_mean_3d"]
        print(
            f"{board:<6}{int(len(sub)):>4}{int(r['overlap_top5']):>6}"
            f"{r['prod_mean_3d']:>+10.4f}{r['shadow_mean_3d']:>+10.4f}"
            f"{d3:>+9.4f}{r['prod_win_3d']:>9.1%}{r['shadow_win_3d']:>9.1%}",
            flush=True,
        )
        if "shadow_mean_5d" in r and pd.notna(r["shadow_mean_5d"]):
            d5 = r["shadow_mean_5d"] - r["prod_mean_5d"]
            print(
                f"       5d 口径: Δ {d5:+.4f}  (生产 {r['prod_mean_5d']:+.4f} / "
                f"影子 {r['shadow_mean_5d']:+.4f})",
                flush=True,
            )
        agg_row = {
            "board": board,
            "n_days": int(len(sub)),
            "overlap_top5_avg": round(float(r["overlap_top5"]), 3),
            "prod_mean_3d": round(float(r["prod_mean_3d"]), 5),
            "shadow_mean_3d": round(float(r["shadow_mean_3d"]), 5),
            "delta_3d": round(float(d3), 5),
            "prod_win_3d": round(float(r["prod_win_3d"]), 4),
            "shadow_win_3d": round(float(r["shadow_win_3d"]), 4),
        }
        if "shadow_mean_5d" in r and pd.notna(r["shadow_mean_5d"]):
            agg_row.update(
                {
                    "prod_mean_5d": round(float(r["prod_mean_5d"]), 5),
                    "shadow_mean_5d": round(float(r["shadow_mean_5d"]), 5),
                    "delta_5d": round(float(d5), 5),
                }
            )
        agg_rows.append(agg_row)
        summary["boards"][board] = agg_row

    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(out_dir / "dual_track_3d.csv", index=False)

    summary["n_dates_total"] = len(dates)
    summary["n_dates_paired"] = int(daily["date"].nunique())
    summary["threshold"] = (
        "主板 Δ≈0 → 换排名键白换; 20cm (GEM/STAR) 反相关, 影子方向被并行 dual 250d OOS 支持; "
        "需 ≥20 配对日再定论"
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nWORM: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
