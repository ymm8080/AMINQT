# -*- coding: utf-8 -*-
"""
dim20 注入验证 (2026-09-09) — 数值对拍部分 (registry 登记已单独完成)
对拍: engine dim20 的 close_vs_low_ma5 vs 实验脚本口径 mean5(close/low-1)*100
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.feature_engine_v35 import FeatureEngineV35

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
p = pq.read_table(
    PANEL, columns=["symbol", "date", "open", "high", "low", "close", "volume"]
).to_pandas()
p["symbol"] = p["symbol"].astype(str)
p = p.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

warm_dates = sorted(p["date"].unique())[-75:]  # 60 评测 + 15 预热
sam = p[p["date"].isin(warm_dates)].copy()
print(f"[sample] rows={len(sam):,} symbols={sam['symbol'].nunique()}")

fe = FeatureEngineV35()
out = fe.dim20_short_horizon(sam.copy())
col = "close_vs_low_ma5"
print(f"[engine] {col} in columns: {col in out.columns}")

cl = p["close"] / p["low"] - 1.0
base = cl.groupby(p["symbol"]).transform(lambda s: s.rolling(5).mean()) * 100
base = base.loc[sam.index]

mrg = pd.DataFrame({"engine": out[col], "base": base}).dropna()
diff = (mrg["engine"] - mrg["base"]).abs()
print(f"[PASS-check] n={len(mrg):,}  max|diff|={diff.max():.2e}  identical={(diff < 1e-9).all()}")

nan_rate = out[col].isna().mean()
inf_rate = np.isinf(out[col].fillna(0)).mean()
sub = out[["close_vs_low", col]].dropna()
corr = sub.corr().iloc[0, 1]
print(f"[health] NaN={nan_rate:.2%} inf={inf_rate:.2%} corr(vs raw)={corr:.3f}")
print(f"[health] median={out[col].median():.2f} p5={out[col].quantile(0.05):.2f} p95={out[col].quantile(0.95):.2f}")
assert (diff < 1e-9).all() and inf_rate == 0
print("PASS")
