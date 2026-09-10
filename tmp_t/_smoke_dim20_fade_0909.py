# -*- coding: utf-8 -*-
"""dim20 fade_score 真实面板 smoke (09-09): 今晚 20:15 重训是注入代码首跑, 先小切片验证.
40 股 × ~500 交易日, 直接跑 FeatureEngineV35.dim20_short_horizon, 断言 fade_score
非空率/值域/与手算公式一致。只读面板, 不写任何文件。
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"

t0 = pd.Timestamp.now()
tbl = pq.read_table(
    PANEL,
    columns=["symbol", "date", "open", "high", "low", "close", "volume",
             "turnover_rate"],
    filters=[("date", ">=", pd.Timestamp("2024-06-01"))],
)
df = tbl.to_pandas()
del tbl
df["symbol"] = df["symbol"].astype(str)
recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=10)]
syms = recent["symbol"].drop_duplicates().head(40)
df = df[df["symbol"].isin(syms)].sort_values(["symbol", "date"]).reset_index(drop=True)
print(f"[load] {df['symbol'].nunique()} symbols x {df.groupby('symbol').size().max()} rows, {len(df)} total, {pd.Timestamp.now()-t0}")

from app.pipeline1.feature_engine_v35 import FeatureEngineV35

t1 = pd.Timestamp.now()
out = FeatureEngineV35().dim20_short_horizon(df)
print(f"[dim20] ok in {pd.Timestamp.now()-t1}")

fs = out["fade_score"]
n_ok = fs.notna().sum()
per_sym = out.groupby("symbol")["fade_score"].agg(["count", "min", "max"])
# 每股最后 30 日应全部有值 (250d min_periods=60 预热已过)
tail30 = out.groupby("symbol").tail(30)
tail_cov = tail30["fade_score"].notna().mean()
print(f"[fade_score] notna={n_ok}/{len(fs)} ({n_ok/len(fs):.1%}); tail30 覆盖率={tail_cov:.1%}")
print(f"[fade_score] 值域 [{fs.min():.4f}, {fs.max():.4f}] (期望 [0,1])")

# 手算复验 (同一公式独立实现)
g = out.groupby("symbol", sort=False)
ret1 = out["close"] / g["close"].shift(1) - 1.0
raw = pd.DataFrame({
    "r20": g["close"].pct_change(20),
    "vol20": ret1.groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(20).std()),
    "turn20": out["turnover_rate"].groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(20).mean()),
    "pos250": out["close"] / g["high"].transform(lambda s: s.rolling(250, min_periods=60).max()) - 1.0,
})
m = raw.notna().all(axis=1)
exp = raw[m].groupby(out.loc[m, "date"]).rank(pct=True).mean(axis=1)
join = pd.DataFrame({"got": fs, "exp": exp}).dropna()
diff = (join["got"] - join["exp"]).abs().max()
print(f"[verify] join={len(join)} max|got-exp|={diff:.2e}")
assert tail_cov > 0.95, "tail30 覆盖率异常"
assert fs.dropna().between(0, 1).all(), "值域越界"
assert diff < 1e-9, "与手算公式不一致"
print("SMOKE PASS")
