# -*- coding: utf-8 -*-
# 每层当月实绩 (滚动重算): GENIOUS 交付「当月样本口径」列的检索源。
# 产出: .cache/kongduo_layer_monthly_YYYYMM.json (每月 WORM 一份)
# 用法: python perf/kongduo_layer_monthly.py [YYYYMM]
import json, sys
from datetime import date
import numpy as np, pandas as pd

sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
from app.pipeline1 import kongduo_triggers as ktl  # noqa: E402
from config.settings import PANEL_V3_PATH  # noqa: E402


def build(yearmonth: str, panel_path=PANEL_V3_PATH) -> str:
    ym_start = pd.Timestamp(yearmonth + "01").strftime("%Y%m%d")
    ym_end = (pd.Timestamp(yearmonth + "01") + pd.offsets.MonthEnd(0)).strftime(
        "%Y%m%d"
    )
    df = ktl.load_panel(panel_path, ym_end, lookback_days=0)
    if df.empty:
        raise SystemExit(f"panel empty for {yearmonth}")
    # 全历史算完层再截月窗 — 月内单独算缺滚动热身 (r120/SIG/MA10斜率), 分层全为空
    df = ktl.compute_features(df)
    df = ktl.compute_triggers(df)
    df["_layer"] = ktl.assign_layers(df).fillna("")
    df = df[(df["date"] >= ym_start) & (df["date"] <= ym_end)]

    g = df.groupby("symbol", sort=False)
    c = df["close"].to_numpy(float)
    fut = np.column_stack([g["close"].shift(-k).to_numpy(float) for k in range(1, 6)])
    STOP = 0.05

    def outcome(i):
        fs = np.where(np.isnan(fut[i]), np.inf, fut[i])
        c0 = c[i]
        if (fs < c0 * (1 - STOP)).any():
            return -STOP
        ok = fut[i][np.isfinite(fut[i])]
        ok = ok[ok < np.inf]
        return ok[-1] / c0 - 1 if len(ok) else np.nan

    nd = df["date"].nunique()
    lay = df["_layer"].to_numpy()
    out = {}
    for name in ktl.ALL_LAYERS:
        idx = np.where(lay == name)[0]
        if not len(idx):
            continue
        r = np.array([outcome(i) for i in idx])
        fin = r[np.isfinite(r)]
        if not len(fin):
            continue
        # 月末 5 日内未来不足的票自然 NaN 剔除; 下月重算时补回
        out[name] = dict(
            n=len(fin),
            daily=round(len(fin) / nd, 1),
            win=round(float((fin > 0).mean()), 4),
            exp=round(float(fin.mean()), 4),
        )
    assert out, f"{yearmonth} 无任何层命中"
    path = f".cache/kongduo_layer_monthly_{yearmonth}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"month": yearmonth, "built": str(date.today()), "data_by_layer": out},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


if __name__ == "__main__":
    ym = sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y%m")
    print(build(ym))
