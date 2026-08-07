"""V3 面板 3 年窗口训练 + 今日预测 → MAIN/DUAL 名单.

面板源: config PANEL_V3_PATH (已含当日, 由 _daily_fetch 追加, 含 4 个 bt_ 原始列).
训练: 最近 3 年切片 (用户裁决 "窗口是3年数据") → run_training → {board}_{tag}.pkl.
预测: 全量面板 (保留最长 EWMA/滚动窗口记忆) → run_prediction → 名单按 board 拆 MAIN/DUAL.
名单: data/lists/list_{date}.parquet (run_prediction 落盘) + list_{date}_{board}.parquet 分板块.

用法: python scripts/train_predict_bt3y.py [YYYYMMDD] [TAG]   (默认取面板最新日; 可选 tag 覆盖)
"""

import logging
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from app.pipeline1.predict_runner import run_prediction
from app.pipeline1.train_runner import run_training
from config.settings import PANEL_V3_PATH, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_predict_bt3y")

YEARS = 3
LIST_DIR = "data/lists"
MODEL_DIR = "models/pipeline1"


def _backup_keepers(trade_date: str) -> None:
    """WORM 备份关键文件到仓库外 (防 automation git 误删). 失败仅告警."""
    try:
        from app.core.config_loader import load_config
        from app.pipeline1.backup import backup_keepers

        bk = (load_config("data_pipeline_config") or {}).get("backup") or {}
        if not bk.get("enabled"):
            return
        res = backup_keepers(
            root=PROJECT_ROOT,
            backup_dir=bk.get("dir"),
            keepers=bk.get("keepers") or [],
            trade_date=trade_date,
            retention=int(bk.get("retention") or 2),
        )
        for pat, msg in res.items():
            logger.info("backup %s: %s", pat, msg)
    except Exception as exc:
        logger.warning("backup_keepers 失败 (非阻塞): %s", exc)


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
    del panel  # 训练期释放全量面板, 降低峰值内存 (本机 commit 上限紧张)
    import gc

    gc.collect()
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

    # ── 2.5 提升当前指针 + 更新元数据 (OOS 合格才切换, 保留旧模型) ──
    # 使后续独立预测 (predict_only / 看板) 用刚训练好的模型, 而非旧 tag.
    from app.pipeline1.model_meta import load_modules, save_modules

    mods = load_modules()
    for b, res in results.items():
        if not res.get("switched"):
            logger.warning(
                "[%s] OOS 未过 (weighted_IC=%.4f), 保留旧当前模型",
                b,
                res["oos"].get("weighted_ic", 0.0),
            )
            continue
        cur = os.path.join(MODEL_DIR, f"{b}_current.pkl")
        backup_cur = os.path.join(MODEL_DIR, f"{b}_current_{trade_date}_backup.pkl")
        if os.path.exists(cur) and not os.path.exists(backup_cur):
            shutil.copy(cur, backup_cur)
            logger.info("[%s] 旧当前模型备份 → %s", b, os.path.basename(backup_cur))
        shutil.copy(res["path"], cur)
        mods[b] = {
            "tag": tag,
            "file": os.path.basename(res["path"]),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        logger.info(
            "[%s] 当前模型 → %s (tag=%s)", b, os.path.basename(res["path"]), tag
        )
    if mods:
        save_modules(mods)
        logger.info("current_meta.json 更新: %s", mods)

    # ── 3. 预测 + 名单 (全量面板, 含当日 bt_ EWMA 记忆) ──
    del train_panel  # 训练结束释放切片, 再重读全量面板做预测 (峰值内存解耦)
    gc.collect()
    panel = pd.read_parquet(PANEL_V3_PATH)
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
        _backup_keepers(trade_date)  # 模型/元数据已更新, 仍备份
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

    _backup_keepers(trade_date)


if __name__ == "__main__":
    main()
