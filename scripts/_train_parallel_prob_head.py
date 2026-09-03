"""_train_parallel_prob_head.py — 并行概率头训练 (每日自动化, 自判断新鲜度).

定案 (memory parallel-gbm-wf-verdict): 每 refit_every_days 交易日扩窗重训全局
LGBM 概率头 (mfe_3d >= abs_target 二分类), WORM bundle 落盘 data/prob_head/.
训练读 _diag_stage_{board}_3y.parquet (parallel 步骤当日检查点产物, 与短名单同源);
全史扩窗 = 面板全部行 (mfe_3d 尾段 NaN 行自动排除), 预测目标是次日及以后的决策日
→ 无前瞻. trailing 242d 训练=数据饥饿退化, 勿用.

半衰期集成 (2026-09-03 用户定案 ensB3): PROB_GATE["half_lives"] 逐档训练
(<board>_prob_hl<hl>_<ts>.joblib), 新鲜度逐档判断, serving 侧概率取均值.

用法: python scripts/_train_parallel_prob_head.py [--force]
自判断: 某档最新 bundle 距今 < refit_every_days 交易日 → skip 该档 (--force 强制重训).
WORM: <board>_prob_hl<hl>_<ts>.joblib, 旧 bundle 不覆盖不删除.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel import prob_head
from config.settings import DATA_DIR, PROB_GATE


def _load_board(board: str) -> pd.DataFrame | None:
    """同回测载入: 全部特征 + mfe_3d + label_pain (行序任意, 训练按行)."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    schema = pq.read_schema(str(fp)).names
    need = [
        c
        for c in schema
        if not c.startswith("label_")
        and not c.startswith("pred_")
        and c not in prob_head.META
    ]
    need += ["symbol", "date", "label_pain"]
    t = pq.read_table(str(fp), columns=list(dict.fromkeys(need))).to_pandas()
    if t.empty:
        return None
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    t = prob_head._add_mfe_3d(t)
    return t


def main() -> int:
    force = "--force" in sys.argv[1:]
    ok = True
    for board in ("main", "dual"):
        t = _load_board(board)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            ok = False
            continue
        dates = np.unique(t["date"].values)
        latest = pd.Timestamp(dates[-1])
        for hl in PROB_GATE["half_lives"]:
            b = prob_head.load_latest_tier(board, hl)
            age = (
                None
                if b is None
                else prob_head.bundle_age_trading_days(dates, str(b["trained_through"]))
            )
            if (
                not force
                and b is not None
                and age is not None
                and age < PROB_GATE["refit_every_days"]
            ):
                print(
                    f"[{board}/hl{hl}] skip: bundle 距今 {age} 交易日 < "
                    f"{PROB_GATE['refit_every_days']} (面板最新 {latest:%Y-%m-%d})",
                    flush=True,
                )
                continue
            path = prob_head.train_bundle(board, t, str(latest.date()), hl)
            n_pos = int((t["mfe_3d"] >= PROB_GATE["abs_target"]).sum())
            print(
                f"[{board}/hl{hl}] 训练 {len(t):,} 行 (正样本 {n_pos:,}, "
                f"特征 {len(prob_head.feature_cols(t.drop(columns=['mfe_3d'])))} 列) "
                f"-> {path.name}",
                flush=True,
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
