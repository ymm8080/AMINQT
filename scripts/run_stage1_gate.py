# -*- coding: utf-8 -*-
"""
P19.0 阶段一: Alpha 点火验证 (端到端验收门禁)
=====================================================
用法: python scripts/run_stage1_gate.py --panel <面板parquet> [--tag TAG] [--small]

流程 (对齐 P19.0 W1-W4):
  W1 数据质量日检 (dq_report): OHLCV 铁律校验, 不通过显式剔除并告警
  W2 未来函数审计 (leakage_audit): 特征引擎源码静态扫描 + IC 上限哨兵
  W3 双板块训练 (CleaningPipeline -> LabelEngine -> FeatureEngineV35 ->
     DualTrackTrainer.train_window/validate_oos), 汇总 OOS Rank IC/ICIR
  W4 IC 衰减曲线 (ic_decay) + 波动桶 IC (DynamicEngine.bucket_ic)
  裁决: metrics.ignition_gate — Rank IC>=0.03 且 ICIR>=0.3 且
        高波动桶 IC>=0.02 且 训练 IC<=0.15 (无泄漏)

输出: data/stage1_gate_{tag}.json + BacktestJournal WORM 账本追加;
      exit code: 全部通过=0, 任一不通过=1.
--small: 快速 smoke (LightGBM n_estimators=20, ES_PATIENCE=5).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import app.pipeline1.dual_track_trainer as dtt
from app.pipeline1 import dq_report, leakage_audit, metrics
from app.pipeline1.backtest_adjudication import BacktestJournal
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.dynamic_engine import DynamicEngine
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.ic_decay import ic_decay_curve
from app.pipeline1.label_engine import LabelEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("run_stage1_gate")

REPO_ROOT = Path(__file__).resolve().parent.parent
# W2 静态扫描对象: 特征计算模块 (未来函数红线, 特征模块绝不豁免)
FEATURE_MODULES = [str(REPO_ROOT / "app" / "pipeline1" / "feature_engine_v35.py")]
MASK_RECENT_DAYS = 6  # V3.8: label_5d 需 T+6 收盘价, 最近 6 天标签未生成
SEED = 42  # 量化铁律: 随机种子固定


def _apply_small_mode() -> None:
    """--small: 缩减训练规模 (smoke 用), 不改变门禁口径."""
    dtt.LGB_PARAMS_REG["n_estimators"] = 20
    dtt.LGB_PARAMS_CLS["n_estimators"] = 20
    dtt.ES_PATIENCE = 5
    logger.warning(
        "small 模式: n_estimators=20, ES_PATIENCE=5 (仅 smoke, 不作验收依据)"
    )


def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """标签链 (P19.0): PM 执行口径 + 路径标签 + 停牌遮蔽 + 缩尾 + 近 6 日遮蔽."""
    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.winsorize_cross_section(df)
    return LabelEngine.mask_recent_days(df, MASK_RECENT_DAYS)


def _evaluate_board(
    board: str, df: pd.DataFrame, trainer: dtt.DualTrackTrainer
) -> dict:
    """单板块: 特征 -> 训练 -> OOS -> Rank IC/ICIR/波动桶/哨兵/IC衰减 -> 点火门禁."""
    df = FeatureEngineV35().build(df)
    cols = FeatureEngineV35.feature_columns(df)
    logger.info("[%s] 特征列 %d 个, 样本 %d 行", board, len(cols), len(df))

    trained = trainer.train_window(df, board, cols)
    oos = trainer.validate_oos(trained)
    model, label = trained["models"]["1d_reg"]
    segs = trained["segs"]

    def _scored(seg: pd.DataFrame) -> pd.DataFrame:
        sub = seg.dropna(subset=[label]).copy()
        sub["score"] = model.predict(np.nan_to_num(sub[cols].values, nan=0.0))
        return sub

    test = _scored(segs["test"])
    train = _scored(segs["train"])

    # W3: OOS Rank IC / ICIR (1d_reg 模型, 验收口径标签)
    rank_ic = metrics.rank_ic(test, "score", label)
    icir = metrics.icir(test, "score", label)
    # 训练段 IC (泄漏哨兵口径: A股日频横截面 alpha 不可能稳定 > 0.15)
    train_ic = metrics.rank_ic(train, "score", label)
    # W2 防线 2: 特征级 IC 上限哨兵
    sentinel = leakage_audit.ic_sentinel(train, cols, label)
    # E.4: ATR 五桶 IC, Q5=高波动桶 (上线前必跑)
    if "ATR_pct" in test.columns:
        bucket = DynamicEngine.bucket_ic(test, "score", label)
        high_vol_ic = bucket["high_vol_ic"]
    else:
        logger.error("[%s] 缺 ATR_pct 列, 高波动桶 IC 按 0 计 (门禁不通过)", board)
        bucket = {"buckets": {}, "high_vol_ic": 0.0, "action": "missing_atr"}
        high_vol_ic = 0.0
    # W4: IC 衰减曲线 (t+1/2/3)
    decay = ic_decay_curve(test, "score")

    gate = metrics.ignition_gate(test, "score", label, high_vol_ic, train_ic)
    return {
        "label": label,
        "n_features": len(cols),
        "oos_ics": oos["ics"],
        "rank_ic": round(rank_ic, 4),
        "icir": round(icir, 4),
        "train_ic": round(train_ic, 4),
        "bucket_ic": bucket,
        "ic_sentinel": sentinel,
        "ic_decay": decay,
        "gate": gate,
    }


def _aggregate_checks(boards: dict[str, dict]) -> dict:
    """跨板块汇总四项门禁 (保守口径: 任一板块不通过即不通过)."""
    keys = ("rank_ic", "icir", "high_vol_ic", "train_ic_no_leak")
    checks = {}
    for k in keys:
        per_board = {b: r["gate"]["checks"][k] for b, r in boards.items()}
        checks[k] = {
            "value": {b: c["value"] for b, c in per_board.items()},
            "pass": all(c["pass"] for c in per_board.values()),
        }
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P19.0 阶段一 Alpha 点火验证门禁")
    parser.add_argument("--panel", required=True, help="全市场日线面板 parquet 路径")
    parser.add_argument("--tag", default=None, help="报告标签 (默认当日 YYYYMMDD)")
    parser.add_argument(
        "--small", action="store_true", help="快速 smoke (缩减训练规模)"
    )
    parser.add_argument("--out-dir", default="data", help="报告输出目录")
    parser.add_argument("--journal-dir", default="data/journal", help="WORM 账本目录")
    parser.add_argument("--model-dir", default="models/pipeline1", help="模型目录")
    args = parser.parse_args(argv)

    random.seed(SEED)
    np.random.seed(SEED)
    if args.small:
        _apply_small_mode()
    tag = args.tag or time.strftime("%Y%m%d")

    # ---- W1: 加载 + 数据质量日检 ----
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    logger.info(
        "面板加载: %d 行, %d 只, %s ~ %s",
        len(panel),
        panel["symbol"].nunique(),
        str(panel["date"].min())[:10],
        str(panel["date"].max())[:10],
    )
    dq = dq_report.daily_report(panel)
    if not dq["pass"]:
        logger.error("W1 数据质量日检不通过, 显式剔除违规行并告警 (不静默丢弃)")
        panel = dq_report.drop_violations(panel)
    dq_summary = {k: v for k, v in dq.items() if k != "violations"}
    dq_summary["violations_head"] = (
        dq["violations"][["symbol", "date", "violation"]].head(10).to_dict("records")
        if len(dq["violations"])
        else []
    )

    # ---- W2 防线 1: 特征引擎源码未来函数静态扫描 ----
    audit = leakage_audit.audit_feature_modules(FEATURE_MODULES)
    if not audit["pass"]:
        logger.error(
            "W2 未来函数静态扫描命中 %d 处, 点火验收中止", len(audit["violations"])
        )

    # ---- W3: 清洗 -> 标签 -> 特征 -> 双板块训练 ----
    main_df, dual_df = CleaningPipeline().run_train(panel)
    logger.info("清洗完成: 主板 %d 行 / 双创 %d 行", len(main_df), len(dual_df))
    trainer = dtt.DualTrackTrainer(model_dir=args.model_dir)
    boards = {}
    for board, bdf in (("main", main_df), ("dual", dual_df)):
        if len(bdf) == 0:
            logger.error("[%s] 清洗后为空, 该板块跳过 (门禁不通过)", board)
            boards[board] = {"gate": {"pass": False, "checks": {}}, "skipped": True}
            continue
        boards[board] = _evaluate_board(board, _build_labels(bdf), trainer)

    # ---- 裁决: 点火门禁 (四 checks 全过 + 审计过 + 双板块过) ----
    valid = {b: r for b, r in boards.items() if not r.get("skipped")}
    checks = _aggregate_checks(valid) if valid else {}
    gate_pass = (
        bool(valid) and len(valid) == 2 and all(c["pass"] for c in checks.values())
    )
    sentinel_pass = all(r["ic_sentinel"]["pass"] for r in valid.values())
    final_pass = bool(audit["pass"]) and gate_pass and sentinel_pass

    report = {
        "tag": tag,
        "panel": str(args.panel),
        "small_mode": bool(args.small),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dq_report": dq_summary,
        "leakage_audit": audit,
        "boards": boards,
        "checks": checks,
        "source_audit_pass": bool(audit["pass"]),
        "ic_sentinel_pass": sentinel_pass,
        "pass": final_pass,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"stage1_gate_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    logger.info("报告已写入: %s", out_path)

    # ---- WORM 账本 (反数据窥探, 失败区间一并保留) ----
    journal_path = BacktestJournal(args.journal_dir).log(
        tag=f"stage1_gate_{tag}",
        params={
            "panel": str(args.panel),
            "small": bool(args.small),
            "mask_recent_days": MASK_RECENT_DAYS,
            "seed": SEED,
        },
        metrics={"checks": checks, "pass": final_pass},
    )
    logger.info("WORM 账本已追加: %s", journal_path)

    # ---- 中文摘要 ----
    print("\n========== P19.0 阶段一 Alpha 点火验收 ==========")
    print(f"标签: {tag} | 面板: {args.panel} | small: {bool(args.small)}")
    dq_verdict = "通过" if dq["pass"] else "不通过(已显式剔除违规行)"
    print(
        f"W1 数据质量日检: {dq_verdict} "
        f"(OHLCV违规 {dq['n_ohlcv_violations']}, 重复键 {dq['n_duplicate_keys']}, "
        f"缺失 {dq['n_missing_key_cols']})"
    )
    audit_verdict = "通过" if audit["pass"] else "不通过"
    print(f"W2 源码未来函数扫描: {audit_verdict} ({len(audit['violations'])} 处命中)")
    for board, r in boards.items():
        if r.get("skipped"):
            print(f"[{board}] 跳过 (清洗后为空)")
            continue
        g = r["gate"]["checks"]
        marks = "/".join(f"{k}={'OK' if v['pass'] else 'X'}" for k, v in g.items())
        decay = r["ic_decay"]
        print(
            f"[{board}] OOS Rank IC={r['rank_ic']:.4f} ICIR={r['icir']:.4f} "
            f"高波动桶IC={r['bucket_ic']['high_vol_ic']:.4f} "
            f"训练IC={r['train_ic']:.4f} | IC衰减 t+1/2/3="
            f"{decay.get('ic_t+1')}/{decay.get('ic_t+2')}/{decay.get('ic_t+3')} "
            f"| 门禁: {marks}"
        )
    if final_pass:
        verdict = "通过 — 允许进入阶段二"
    else:
        verdict = "不通过 — 按序排查 未来函数->复权->幸存者偏差, 禁止加特征硬堆 IC"
    print(f"最终裁决: {verdict}")
    print(f"报告: {out_path}")
    return 0 if final_pass else 1


if __name__ == "__main__":
    sys.exit(main())
