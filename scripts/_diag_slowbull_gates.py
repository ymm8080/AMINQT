"""慢牛硬门槛逐门槛通过率诊断 (2026-08-05).

验证 load_panel 后四道门槛各自生效而非某门槛意外全 False/全 True.
结果 WORM 落盘 data/_diag_slowbull_gates_<ts>.json.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from app.pipeline_parallel.backtest import load_panel
from app.pipeline_parallel.config import ADX_SPEC
from app.pipeline_parallel.screener import _yesterday


def _gate_counts(df: pd.DataFrame) -> dict:
    s = ADX_SPEC
    g1 = (
        (df["close_cont"] > df["ma5"])
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
        & (df["ma_slope5"] > 0)
        & (df["ma_slope10"] > 0)
        & (df["ma_slope20"] > 0)
        & ((df["ma5"] - df["ma10"]).abs() / df["ma10"] < s["ma_bias_max"])
    )
    g2 = (df["adx"] > s["adx_min"]) & (df["pdi"] > df["mdi"]) & (df["adx_rise5"] > 0)
    g3 = (
        (df["amplitude_20"] < s["amplitude_20_max"])
        & (df["max_drop_20"] > -s["max_drop_20_max"])
        & (df["limit_down_20"] == 0)
    )
    g4 = (
        (df["ma_vol_5"] > df["ma_vol_10"])
        & (df["ma_vol_10"] > df["ma_vol_20"])
        & (_yesterday(df, "vol_ratio") < s["vol_ratio_max"])
        & df["turnover_rate"].between(s["turnover_min"], s["turnover_max"])
    )
    n = len(df)
    out = {"rows": int(n), "stocks": int(df["symbol"].nunique())}
    for name, g in (
        ("g1_ma", g1),
        ("g2_adx", g2),
        ("g3_lowvol", g3),
        ("g4_volume", g4),
    ):
        out[name] = {
            "rows": int(g.sum()),
            "pct": round(float(g.mean()), 5),
            "stocks": int(df.loc[g, "symbol"].nunique()),
        }
    out["all_4"] = {
        "rows": int((g1 & g2 & g3 & g4).sum()),
        "stocks": int(df.loc[g1 & g2 & g3 & g4, "symbol"].nunique()),
    }
    # g4 子条件分解 (定位瓶颈: 单调放量 / 昨日量比 / 换手区间)
    g4a = (df["ma_vol_5"] > df["ma_vol_10"]) & (df["ma_vol_10"] > df["ma_vol_20"])
    g4b = _yesterday(df, "vol_ratio") < 3.0
    g4c = df["turnover_rate"].between(0.03, 0.15)
    for name, g in (
        ("g4a_vol_mono", g4a),
        ("g4b_volratio_yest", g4b),
        ("g4c_turnover_3_15", g4c),
    ):
        out[name] = {
            "rows": int(g.sum()),
            "pct": round(float(g.mean()), 5),
            "stocks": int(df.loc[g, "symbol"].nunique()),
        }
    out["g4a_g4b_g4c"] = {
        "rows": int((g4a & g4b & g4c).sum()),
        "stocks": int(df.loc[g4a & g4b & g4c, "symbol"].nunique()),
    }
    # NaN 中毒检查: 量比/换手 缺失率
    for col in ("vol_ratio", "turnover_rate"):
        out[f"nan_{col}"] = {
            "rows": int(df[col].isna().sum()),
            "pct": round(float(df[col].isna().mean()), 5),
        }
    vr = df["vol_ratio"].dropna()
    out["vol_ratio_desc"] = {
        "min": float(vr.min()),
        "max": float(vr.max()),
        "mean": round(float(vr.mean()), 3),
        "zero_frac": round(float((vr == 0).mean()), 5),
    }
    return out


def main() -> int:
    work = load_panel()
    counts = _gate_counts(work)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_gates_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
