# -*- coding: utf-8 -*-
"""688228 个案 + 生产 10d 头 TreeSHAP 归因 (dual 板, 用户原始问题: HOW 排进 TOP10).

用 dual staging 帧 (09-09 数据) 的 688228 最新行 + 生产 dual_current 包自己的
特征列, LightGBM 原生 pred_contrib 输出 top-12 特征贡献。超额口径加回
mkt_expected_10d 复原绝对承诺 (与 09-09 交付清单 #8 的 +5.27% 同口径对账)。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from config.settings import data_others_path

STAGE = "data/_diag_stage_dual_3y.parquet"
MODEL_DIR = "models/pipeline1"
SYM_PREFIX = "688228"


def main() -> int:
    t0 = time.time()
    prod = DualTrackTrainer.load(os.path.join(MODEL_DIR, "dual_current.pkl"))
    cols = list(prod["feature_cols"])

    sym_col = pd.read_parquet(STAGE, columns=["symbol"])
    matches = sym_col[sym_col["symbol"].astype(str).str.startswith(SYM_PREFIX)]
    if not len(matches):
        print(f"[case] {SYM_PREFIX} 不在 dual 帧", flush=True)
        return 1
    sym_exact = str(matches["symbol"].iloc[0])
    del sym_col, matches
    print(f"[case] symbol={sym_exact} ({time.time() - t0:.0f}s)", flush=True)

    crow = pd.read_parquet(STAGE, filters=[("symbol", "=", sym_exact)])
    crow = crow.sort_values("date").tail(1)
    print(f"[case] 行取回 date={crow['date'].iloc[0]} ({time.time() - t0:.0f}s)", flush=True)

    close = crow["close_hfq"].iloc[0] if "close_hfq" in crow else None
    close_raw = crow["close"].iloc[0] if "close" in crow else None
    r5 = None
    if {"symbol", "date", "close_hfq"} <= set(crow.columns):
        hist = pd.read_parquet(STAGE, columns=["symbol", "date", "close_hfq"])
        hist = hist[hist["symbol"].astype(str).str.startswith(SYM_PREFIX)].sort_values("date")
        r5_series = hist.groupby("symbol")["close_hfq"].pct_change(5)
        r5 = float(r5_series.iloc[-1]) if len(r5_series) else None
        del hist
    X = np.nan_to_num(
        crow[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    m = prod["models"]["10d_reg"][0]
    raw = float(m.predict(X)[0])
    mkt = float(prod["mkt_expected_10d"]) if prod.get("label_excess") and prod.get("mkt_expected_10d") is not None else 0.0
    contrib = m.predict(X, pred_contrib=True)[0]
    vals = contrib[:-1]
    order = np.argsort(-np.abs(vals))[:12]
    rep = {
        "symbol": sym_exact,
        "date": str(pd.Timestamp(crow["date"].iloc[0]).date()),
        "pred10_raw_demeaned": raw,
        "mkt_expected_10d": mkt,
        "pred10_abs": raw + mkt,
        "delivered_0909_list_value": 0.0527,
        "close_hfq": None if close is None else float(close),
        "close_raw": None if close_raw is None else float(close_raw),
        "r5_at_day": r5,
        "bias_demeaned": float(contrib[-1]),
        "top_contrib": [
            {"feature": cols[i], "contrib": float(vals[i])} for i in order
        ],
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"shap_688228_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[worm] {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
