"""全量周频重训 (V3 面板直读, 含当日) — 2026-08-05 用户确认跑全量.

与 run_weekly_retrain.py 的区别: 训练面板直接读 PANEL_V3_PATH (已含当日, 免 akshare
网络装配), 对齐 3 年周频窗口. OOS IC 过闸才把 bundle 发布为 current (镜像 weekly 语义),
current_meta.json 同步更新.

用法: python scripts/_retrain_legacy_full.py [tag]
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from app.pipeline1.cleaning_pipeline import load_panel_v3
from app.pipeline1.train_runner import run_training

MODEL_DIR = "models/pipeline1"


def main() -> int:
    # 子模块全部用 logging.getLogger(__name__) 传播到 root, 无 handler 时 info 被丢弃
    # (Python last-resort handler 只放 WARNING+), 重训会"看似卡住". 这里挂一个 handler.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    tag = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
    # LEGACY_FORCE_FALLBACK=1 → 只对 main 跳过 FeatureSelector (bruteforce_dedup 选择
    # 过大必 OOM, 直用 FeatureEngine 全量 316 特征), dual 仍走 gate_d (38). 两段式发布:
    # 先落 cls 修复, 再单独验证 cap 选择过 OOS 门.
    fallback_boards = (
        {"main"} if os.environ.get("LEGACY_FORCE_FALLBACK", "0") == "1" else None
    )
    t0 = time.time()
    # 读取时行级预过滤 (amount>=5000万 且 非停牌): 与 run_train 内部过滤同口径,
    # 少读 ~20% 行, 输出与整表读取后 run_train 完全一致 (2026-08-10).
    panel = load_panel_v3()
    print(
        f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    # 对齐周频 3y 训练窗口 (assemble_panel years=3 语义: 截止日前 3 个日历年)
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(
        f"[slice] {cut.date()}.. {panel['date'].max():%Y-%m-%d} "
        f"-> {len(panel):,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )

    results = run_training(
        panel,
        tag,
        model_dir=MODEL_DIR,
        use_ic_screen=True,
        fallback_boards=fallback_boards,
    )
    del panel
    gc.collect()

    from app.pipeline1.model_meta import load_modules, save_modules

    # LEGACY_FORCE_FALLBACK=1 (20260811c 安全发布): 双创 gate_d 本轮抽到 208 特征
    # 过拟合抽签 (r3=38, 本轮=208, 输入完全一致 → LGBM n_jobs 线程非确定性), 不让
    # 该抽签发布, 保留 proven 的 20260811b (38 特征). 主板 cls 修复照常发布.
    skip_dual_switch = os.environ.get("LEGACY_FORCE_FALLBACK", "0") == "1"
    mods = load_modules()
    for board, res in results.items():
        if skip_dual_switch and board == "dual" and res["switched"]:
            print(
                "[dual] gate_d 非确定性漂移 (38→208 特征), 保留 20260811b, 跳过切换",
                flush=True,
            )
            continue
        if res["switched"]:
            cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
            bak = os.path.join(MODEL_DIR, f"{board}_current_retrain_backup.pkl")
            if os.path.exists(cur) and not os.path.exists(bak):
                shutil.copy(cur, bak)
                print(f"[{board}] 旧 current 备份 -> {bak}", flush=True)
            shutil.copy(res["path"], cur)
            mods[board] = {
                "tag": tag,
                "file": os.path.basename(res["path"]),
                "updated": time.strftime("%Y-%m-%d %H:%M"),
            }
            print(f"[{board}] switched -> current = {res['path']}", flush=True)
        else:
            print(
                f"[{board}] OOS weighted_IC={res['oos'].get('weighted_ic'):.4f} "
                f"< {res['oos'].get('threshold', '?')}, 保留旧模型",
                flush=True,
            )
    save_modules(mods)
    print(f"[meta] current_meta.json = {mods}", flush=True)
    print(f"[done] 全部完成 ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
