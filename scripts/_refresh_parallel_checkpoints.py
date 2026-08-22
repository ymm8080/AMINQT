"""_refresh_parallel_checkpoints.py — 重建 parallel pipeline 的 main/dual 3y 检查点 (2026-08-04).

parallel pipeline (app/pipeline_parallel) 的 load_panel 读取两个 3y 检查点:
  data/_diag_stage_main_3y.parquet, data/_diag_stage_dual_3y.parquet
每天日更后 V3 面板前移, 检查点若仍指向旧日期 → 短名单/回测会缺最新交易日.
本脚本从 V3 面板重建两个检查点 (复用生产 build_board_slice → 与生产行集完全一致),
旧检查点改名为 <name>.stale_<ts> 而非删除 (可回溯).

用法: python scripts/_refresh_parallel_checkpoints.py [--force]
输出: 两个新检查点 + 控制台日志 (最新日期 / 行数 / 列数).

跳过判定 (指纹): 检查点内容由下述源文件 + 面板数据决定. 指纹=这些源文件
内容 hash; 存入 data/_diag_stage_3y.fingerprint.json (含构建时的最新交易日).
仅当 指纹未变 且 面板无新增交易日 且 两检查点都存在 → 跳过全量重建.
任何 特征/标签/清洗代码或参数 变更 → 指纹变化 → 必全量重建 (绝不静默跳过).
--force → 无条件全量重建 (参数改了想重新跑一遍时用).
"""

import gc
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from config.settings import PANEL_V3_PATH
from scripts._reclassify_all_features import (
    DUAL_CHECKPOINT,
    MAIN_CHECKPOINT,
    build_board_slice,
)

# 决定检查点内容的源文件 → 任一变化指纹必变 → 全量重建 (绝不静默跳过).
# 检查点内容 = 特征 + 标签 (fe.build + LabelEngine 计算). 指纹覆盖:
#   - 特征/清洗/标签/构建代码 (FeatureEngineV35 / CleaningPipeline / LabelEngine / build_board_slice)
#   - 特征配置: fe.build 从 config/settings.py 读 LHB_V2_SPEC → settings 变化会改特征 → 必须算.
# 排名/评分配置 (SHORTLIST_SCORE / parallel HORIZONS / select_gate) 在预测期运行时应用,
# 不进入检查点 → 不触发重建 (改了排名旧检查点依然有效, 新评分自动生效).
_FINGERPRINT_FILES = [
    "app/pipeline1/feature_engine_v35.py",
    "app/pipeline1/cleaning_pipeline.py",  # load_panel_v3 预过滤口径 (2026-08-10)
    "app/pipeline1/label_engine.py",
    "scripts/_reclassify_all_features.py",
    "scripts/_diag_column_feed.py",  # MASK_RECENT_DAYS 等构造常量
    "config/settings.py",  # fe.build 读 LHB_V2_SPEC (特征参数, 影响检查点内容)
    "scripts/_refresh_parallel_checkpoints.py",  # load_panel 预过滤改变检查点行集 (2026-08-10)
]
_FINGERPRINT_META = os.path.join("data", "_diag_stage_3y.fingerprint.json")


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_fingerprint() -> str:
    h = hashlib.sha256()
    root = _repo_root()
    for rel in _FINGERPRINT_FILES:
        with open(os.path.join(root, rel), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _read_fingerprint_meta() -> dict | None:
    if not os.path.exists(_FINGERPRINT_META):
        return None
    try:
        with open(_FINGERPRINT_META, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_fingerprint_meta(latest_date: str) -> None:
    with open(_FINGERPRINT_META, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "fingerprint": compute_fingerprint(),
                "latest_date": latest_date,
                "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


def _skip_if_unchanged(force: bool) -> bool:
    """数据无新增交易日 且 指纹未变 且 检查点都在 → 跳过重建 (True)."""
    if force:
        return False
    meta = _read_fingerprint_meta()
    if meta is None or meta.get("fingerprint") != compute_fingerprint():
        return False
    if not (os.path.exists(MAIN_CHECKPOINT) and os.path.exists(DUAL_CHECKPOINT)):
        return False
    panel_latest = pd.read_parquet(str(PANEL_V3_PATH), columns=["date"])["date"].max()
    if meta.get("latest_date") == str(pd.Timestamp(panel_latest).date()):
        print(
            f"[skip] 数据无新增交易日 ({meta['latest_date']}) 且 特征/标签代码未变, "
            "无需重建 (--force 强制重建)",
            flush=True,
        )
        return True
    return False


def main(force: bool = False) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if _skip_if_unchanged(force):
        return 0

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # 1. 旧检查点改名为 .stale_<ts> (保留可回溯)
    for ck in (MAIN_CHECKPOINT, DUAL_CHECKPOINT):
        if os.path.exists(ck):
            bak = f"{ck}.stale_{ts}"
            os.rename(ck, bak)
            print(f"[stale] {ck} -> {bak}", flush=True)

    # 2. 读 V3 面板 → 逐板块 run_train(board=...) → 重建两检查点 (内存分期)
    print("读取 V3 面板 ...", flush=True)
    fe = FeatureEngineV35()
    cleaner = CleaningPipeline()
    latest_date = None
    # 分板构建: run_train(board=board) 让另一板块返回空帧 → 峰值只含当前板块
    # 清洗帧+特征构建. 旧实现一次 run_train 双板同时驻留 (dual 首建时 main
    # 清洗帧仍占 ~1.5GB), 2026-08-13 dual 特征构建下 2.26MiB 分配失败 OOM →
    # 改逐板重建 (本机 15.8GB 物理, 峰值 ~8GB 落在 main 单板).
    for board, ckpt in (("dual", DUAL_CHECKPOINT), ("main", MAIN_CHECKPOINT)):
        t_panel = time.time()
        panel = load_panel_v3()
        t_clean = time.time()
        main_df, dual_df = cleaner.run_train(panel, board=board)
        del panel
        gc.collect()
        print(
            f"[timing][{board}] panel load: {t_clean - t_panel:.1f}s | "
            f"run_train: {time.time() - t_clean:.1f}s",
            flush=True,
        )
        bdf = main_df if board == "main" else dual_df
        del main_df, dual_df
        gc.collect()
        if bdf is None or len(bdf) == 0:
            print(f"[{board}] 空, 跳过", flush=True)
            del bdf
            continue
        print(f"run_train[{board}]: rows={len(bdf):,}", flush=True)
        t_build = time.time()
        d3 = build_board_slice(cleaner, fe, bdf, board, ckpt)
        print(
            f"[timing][{board}] build_board_slice: {time.time() - t_build:.1f}s",
            flush=True,
        )
        latest_date = d3["date"].max()
        print(
            f"[{board}] 检查点已写 {ckpt} | latest={latest_date:%Y-%m-%d} "
            f"rows={len(d3):,} cols={d3.shape[1]:,}",
            flush=True,
        )
        del bdf, d3
        gc.collect()
    del fe, cleaner
    gc.collect()
    if latest_date is not None:
        _write_fingerprint_meta(str(pd.Timestamp(latest_date).date()))
    print("完成", flush=True)
    return 0


if __name__ == "__main__":
    force = "--force" in sys.argv
    raise SystemExit(main(force=force))
