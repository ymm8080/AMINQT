# -*- coding: utf-8 -*-
"""V3 面板 3 年窗口训练 + 今日预测 → MAIN/DUAL 名单.

面板源: config PANEL_V3_PATH (已含当日, 由 _daily_fetch 追加, 含 4 个 bt_ 原始列).
训练: 最近 3 年切片 (用户裁决 "窗口是3年数据") → run_training → {board}_{tag}.pkl.
预测: 全量面板 (保留最长 EWMA/滚动窗口记忆) → run_prediction → 名单按 board 拆 MAIN/DUAL.
名单: data/lists/list_{date}.parquet (run_prediction 落盘) + list_{date}_{board}.parquet 分板块.

用法: python scripts/train_predict_bt3y.py [YYYYMMDD] [TAG]   (默认取面板最新日; 可选 tag 覆盖)
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from app.pipeline1.train_runner import run_training
from app.pipeline1.predict_runner import run_prediction
from config.settings import PANEL_V3_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_predict_bt3y")

YEARS = 3
LIST_DIR = "data/lists"


def main() -> None:
    trade_date = sys.argv[1] if len(sys.argv) > 1 else None
    panel = pd.read_parquet(PANEL_V3_PATH)
    latest = panel["date"].max()
    trade_date = trade_date or latest.strftime("%Y%m%d")
    logger.info(
        "Panel: %d rows, %d stocks, %d cols; latest=%s",
        len(panel),
        panel["symbol"].nunique(),
        len(panel.columns),
        latest.date(),
    )
    if pd.Timestamp(trade_date) > latest:
        logger.error(
            "目标日 %s 不在面板 (latest=%s), 先跑 _daily_fetch",
            trade_date,
            latest.date(),
        )
        return

    # ── 1. 3 年训练窗口 ──
    cutoff = latest - pd.DateOffset(years=YEARS)
    train_panel = panel[panel["date"] >= cutoff].copy()
    logger.info(
        "训练窗口: %s .. %s (%d rows, %d stocks)",
        train_panel["date"].min().date(),
        train_panel["date"].max().date(),
        len(train_panel),
        train_panel["symbol"].nunique(),
    )

    # ── 2. 训练 (WORM 命名: {board}_{tag}.pkl) ──
    tag = sys.argv[2] if len(sys.argv) > 2 else f"{trade_date}_3y"
    results = run_training(train_panel, tag=tag, use_ic_screen=True)
    if not results:
        logger.error("训练未产出任何板块模型, 终止")
        return
    bundles = {b: res["path"] for b, res in results.items()}
    for b, res in results.items():
        logger.info(
            "[%s] %s | OOS weighted_IC=%.4f | feats=%d | switched=%s",
            b,
            os.path.basename(res["path"]),
            res["oos"].get("weighted_ic", 0.0),
            res["n_features"],
            res.get("switched"),
        )

    # ── 3. 预测 + 名单 (全量面板, 含当日 bt_ EWMA 记忆) ──
    os.makedirs(LIST_DIR, exist_ok=True)
    result = run_prediction(
        panel=panel,
        trade_date=trade_date,
        bundle_paths=bundles,
        list_dir=LIST_DIR,
    )
    lst = result.get("list")
    if result.get("empty") or lst is None or len(lst) == 0:
        logger.warning(
            "名单为空 (mode=%s, valve=%s)",
            result.get("mode"),
            result.get("valve_state"),
        )
        return

    # ── 4. 按板块拆 MAIN/DUAL (board 值: main / GEM / STAR) ──
    for name, mask in (
        ("main", lst["board"] == "main"),
        ("dual", lst["board"].isin(["GEM", "STAR"])),
    ):
        sub = lst[mask] if "board" in lst.columns else pd.DataFrame()
        if len(sub):
            path = os.path.join(LIST_DIR, f"list_{trade_date}_{name}.parquet")
            sub.to_parquet(path, index=False)
            logger.info("[%s] %d 只 -> %s", name, len(sub), path)
            print(f"\n=== {name.upper()} ({len(sub)}) ===")
            print(sub["symbol"].to_string(index=False))
        else:
            print(f"\n=== {name.upper()}: 0 只 ===")
    if "board" in lst.columns:
        print(f"\n合计: {lst['board'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
