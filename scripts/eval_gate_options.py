# -*- coding: utf-8 -*-
"""E7 准入闸门方案对比评估 (数据先行, 供用户裁决)
=====================================================
在最后 N 个 OOS 交易日 (测试段, 模型未参与拟合) 逐日回放:
  截面预测 → 三种闸门 → Top15 等权 → 次日净收益 (label_pm_1d_net, 验收口径).

方案:
  A 现状:        prob≥0.60 且 pred_1d(净) ≥ 2×COST
  B 毛收益过闸:  prob≥0.60 且 pred_1d(净)+cost_total ≥ 2×COST
                 (成本加回 ≈ 用毛口径过闸, 消除双算)
  C 降低倍数:    prob≥0.60 且 pred_1d(净) ≥ 1×COST
  D 计算闸:      pred_1d(净) > 0 且 prob_up > 当日截面均值
                 (无常数: 净预期为正 = 期望盈利; prob 超过当日基准率 [B4 概念])

简化 (与生产清单的差异, 影响四种方案一致, 可横向对比):
  等权 Top15, 无行业上限/簇阻断/分布权重; 只用 1d 次日净收益评估.

用法: python scripts/eval_gate_options.py [--days 84]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.pipeline1.cleaning_pipeline import CleaningPipeline  # noqa: E402
from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.label_engine import COST, LabelEngine, slippage_tier  # noqa: E402
from app.pipeline1.predictor import V35Predictor  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("eval_gate_options")

ENTRY_PROB = 0.60
TOP_N = 15
LABEL_MATURITY_DAYS = 6  # label_pm_5d 需 T+6, 1d 需 T+2; 末端不足的天数剔除


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=84, help="回放交易日数")
    parser.add_argument("--panel", default=None, help="面板 parquet (默认取最新)")
    args = parser.parse_args()

    panel_path = (
        args.panel or sorted(Path("data/processed").glob("panel_*_3y_*.parquet"))[-1]
    )
    logger.info("面板: %s", panel_path)
    panel = pd.read_parquet(panel_path)

    # 清洗 + 特征 + 标签 (各一次; 特征 rolling 计算 PIT 安全, 标签仅用于事后评分)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    main_df, dual_df = cleaner.run_train(panel)
    frames = []
    predictor = V35Predictor(
        {
            "main": "models/pipeline1/main_2026W30.pkl",
            "dual": "models/pipeline1/dual_2026W30.pkl",
        }
    )
    for board, board_df in (("main", main_df), ("dual", dual_df)):
        if len(board_df) == 0:
            continue
        feat = features.build(board_df)
        # 逐日回放需全历史推理 (predict() 只取最新截面) → 直接对全特征矩阵推理
        bundle = predictor.bundles[board]
        cols = bundle["feature_cols"]
        X = np.nan_to_num(
            feat[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        out = (
            feat[["symbol", "date", "adv20"]].copy()
            if "adv20" in feat
            else feat[["symbol", "date"]].copy()
        )
        out["pred_ret_1d"] = bundle["models"]["1d_reg"][0].predict(X)
        raw_prob = bundle["models"]["1d_cls"][0].predict_proba(X)[:, 1]
        out["prob_up"] = bundle["calibrator"].predict_proba(raw_prob)
        frames.append(out)
        logger.info("[%s] 全历史推理完成: %d 行", board, len(out))
    preds = pd.concat(frames, ignore_index=True)

    labeled = LabelEngine.build_labels(panel)
    actuals = labeled[["symbol", "date", "label_pm_1d_net"]]
    df = preds.merge(actuals, on=["symbol", "date"], how="left")
    if "adv20" not in df.columns:
        df["adv20"] = np.nan
    df["cost_total"] = COST + 2 * df["adv20"].map(
        lambda v: slippage_tier(v) if pd.notna(v) else 0.0015
    )

    dates = sorted(df["date"].unique())[-args.days : -2]  # 末端 label_pm_1d 需 T+2
    logger.info(
        "回放 %d 个交易日 (%s ~ %s)", len(dates), dates[0].date(), dates[-1].date()
    )

    scenarios = {
        "A_现状_net>=2xCOST": lambda g: g["pred_ret_1d"] >= 2 * COST,
        "B_毛口径_net+cost>=2xCOST": lambda g: (
            (g["pred_ret_1d"] + g["cost_total"]) >= 2 * COST
        ),
        "C_降倍_net>=1xCOST": lambda g: g["pred_ret_1d"] >= 1 * COST,
        "D_计算闸_net>0&prob>日均": lambda g: (
            (g["pred_ret_1d"] > 0) & (g["prob_up"] > g["prob_up"].mean())
        ),
    }
    rows = []
    for name, gate in scenarios.items():
        daily = []
        cum = 1.0
        for d in dates:
            cross = df[df["date"] == d]
            picked = cross[(cross["prob_up"] >= ENTRY_PROB) & gate(cross)]
            picked = picked.nlargest(TOP_N, "pred_ret_1d")
            realized = picked["label_pm_1d_net"].dropna()
            ret = float(realized.mean()) if len(realized) else 0.0
            if len(picked):
                cum *= 1 + ret
            daily.append(
                {
                    "date": str(d)[:10],
                    "n": int(len(picked)),
                    "ret": ret if len(picked) else None,
                    "cum": round(cum - 1, 6),
                }
            )
        invested = [x for x in daily if x["n"] > 0]
        stats = {
            "name": name,
            "days": len(daily),
            "empty_days": sum(1 for x in daily if x["n"] == 0),
            "empty_rate": sum(1 for x in daily if x["n"] == 0) / len(daily),
            "avg_holdings": float(np.mean([x["n"] for x in invested]))
            if invested
            else None,
            "avg_daily_net": float(np.mean([x["ret"] for x in invested]))
            if invested
            else None,
            "compound": cum - 1,
            "win_rate": float(np.mean([x["ret"] > 0 for x in invested]))
            if invested
            else None,
            "daily": daily,
        }
        rows.append(stats)

    # 打印对比表
    table = pd.DataFrame(
        [
            {
                "方案": r["name"],
                "回放天数": r["days"],
                "空仓天数": r["empty_days"],
                "空仓率": f"{r['empty_rate']:.1%}",
                "平均持股数": f"{r['avg_holdings']:.1f}" if r["avg_holdings"] else "—",
                "日均净收益(持仓日)": f"{r['avg_daily_net']:.4%}"
                if r["avg_daily_net"] is not None
                else "—",
                "累计净收益": f"{r['compound']:.2%}",
                "胜率(持仓日)": f"{r['win_rate']:.1%}"
                if r["win_rate"] is not None
                else "—",
            }
            for r in rows
        ]
    )
    print("\n" + table.to_string(index=False))

    # 持久化 (WORM, 回测看板 /backtest/gate-eval 读取)
    import json
    from datetime import datetime

    out_dir = Path("data/backtest_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": [str(dates[0])[:10], str(dates[-1])[:10]],
        "top_n": TOP_N,
        "entry_prob": ENTRY_PROB,
        "scenarios": rows,
    }
    out_path = out_dir / f"gate_eval_{datetime.now():%Y%m%d}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("报告落盘: %s", out_path)
    return table


if __name__ == "__main__":
    main()
