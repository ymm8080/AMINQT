"""_diag_gate_d_phase2_train.py — Phase 2 候选特征集 dual 训练 (2026-08-18).

gate_d 季度重选协议 阶段2 (候选头对头): 每候选 = 一个 registry pin 文件,
只重训 dual 板 (同配方同面板, 仅特征集不同), bundle 落
models/pipeline1/dual_phase2_<cand>.pkl. 不发布 current — OOS 过闸与否由
replay 阶段 (phase2_replay) 统一判定, 这里只产出候选模型.

用法:
  python scripts/_diag_gate_d_phase2_train.py --pin selected_dual_candA_20260818.json --cand candA
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline1.feature_selector import FeatureSelector
from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from app.pipeline1.train_runner import run_training
from config.settings import (
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

MODEL_DIR = "models/pipeline1"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    ap = argparse.ArgumentParser(description="Phase 2 候选特征集 dual 训练 (不发布)")
    ap.add_argument("--pin", required=True, help="registry 下 pin 文件名 (候选特征集)")
    ap.add_argument(
        "--cand", required=True, help="候选标识 (bundle 名): candA/candB/pin50/neg200"
    )
    args = ap.parse_args()

    registry = str(data_others_path("data/factor_registry"))
    pin_path = os.path.join(registry, args.pin)
    if not os.path.exists(pin_path):
        raise SystemExit(f"pin 文件不存在: {pin_path}")
    with open(pin_path, encoding="utf-8") as fh:
        pin_feats = json.load(fh).get("features", [])
    if not pin_feats:
        raise SystemExit(f"pin 文件无 features: {pin_path}")
    print(
        f"[phase2] cand={args.cand} pin={args.pin} n_features={len(pin_feats)}",
        flush=True,
    )

    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    selector_config = copy.deepcopy(FeatureSelector.DEFAULT_CONFIG)
    selector_config["dual"]["gate_d"]["pinned"] = args.pin

    tag = f"phase2_{args.cand}"
    t0 = time.time()
    results = run_training(
        panel=None,
        panel_path=PANEL_V3_PATH,
        tag=tag,
        model_dir=MODEL_DIR,
        use_ic_screen=True,
        boards=("dual",),
        selector_config=selector_config,
    )
    elapsed = (time.time() - t0) / 60
    print(f"\n[phase2] cand={args.cand} 总耗时 {elapsed:.0f}min", flush=True)
    for board, res in results.items():
        print(
            f"[phase2] {board}: n_features={res.get('n_features')} path={res.get('path')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
