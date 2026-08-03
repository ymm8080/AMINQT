"""Daily fetch script — append one day's data to V3 panel from Tushare.
NEVER loads full panel into memory. Uses pyarrow streaming read/write.

Usage: python _daily_fetch.py [YYYYMMDD]  (default: today)

Sources fetched:
  1. OHLCV (daily) — all stocks, filtered to panel universe
  2. adj_factor → hfq prices
  3. daily_basic — turnover, PE, PB, float, MV
  4. stk_limit — up/down limit prices
  5. suspend_d — suspension markers
  6. cyq_perf — chip distribution (batch, one call)
  7. margin_detail — margin balance (per-stock)
  8. top_list — LHB (dragon-tiger board)
  9. stock_basic — name/list_date (ingest gate: ST/*ST 或 上市 <150 交易日不入库)

Derived (computed from panel history, reads only needed columns):
  - bias_5..250, bias_cross, ma_vol_ratio_5_20, amplitude_5d
  - vol_surge, amt_surge
  - pct_90_con (cyq formula: (hi-lo)/(hi+lo), [0,1] range)
  - cost_bias, conc_trend_20d, conc_90_industry_rank (CYQ 派生列, dim21 公式)
  - Forward-fill: financials (quarterly), margin(T+1 gap), announce_date,
    sw_l1_name/sw_l2_name/sw_l3_name (Shenwan classification, static per symbol)

NOT fetched here (separate pipelines):
  - fina_indicator → run_announcement_pipeline.py (Pipeline 2, quarterly)
  - holdertrade/holdernumber → removed from panel (no longer tracked)
"""
import os
import sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import time
import tushare as ts
from dotenv import load_dotenv
from config.settings import INGEST_MIN_LIST_DAYS
from app.pipeline1.ingest_scan import apply_ingest_scan
load_dotenv()

from app.pipeline1 import cyq_ext
from app.pipeline1.cleaning_pipeline import board_of

TRADE_DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

_token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
if not _token:
    print("FATAL: No Tushare token found. Set TUSHARE_TOKEN or run `tushare.set_token('your_token')`")
    sys.exit(1)
pro = ts.pro_api(_token)

# ── 0. Get valid stock universe from panel ──
print("[0] Getting stock universe...")
yesterday = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
max_date = yesterday["date"].max()
valid_universe = set(yesterday[yesterday["date"] == max_date]["symbol"].unique())
print(f"    Universe: {len(valid_universe)} stocks, max date: {max_date.date()}")

if pd.Timestamp(TRADE_DATE) <= max_date:
    # 替换历史日: 合并当日已存在的 symbol, 防止刚停牌股票 (不在最新日 universe,
    # 如 07-30 停牌的 300333) 在替换该日行时被整行丢弃.
    existing_today = set(yesterday[yesterday["date"] == pd.Timestamp(TRADE_DATE)]["symbol"].unique())
    kept = existing_today - valid_universe
    valid_universe = valid_universe | existing_today
    print(f"    {TRADE_DATE} already in panel (max={max_date.date()}), will replace today's rows. "
          f"Kept {len(kept)} symbols present on {TRADE_DATE} but not on max date.")

# ── 1. Fetch all Tushare sources ──
print(f"\n[1] Fetching Tushare data for {TRADE_DATE}...")

def to_symbol(df):
    if "ts_code" in df.columns:
        df["symbol"] = df["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
    return df

def safe_fetch(fn, name, max_retries=3, **kwargs):
    """Fetch with retry. Tushare 限流/超时是瞬时的, 重试避免当日行整列留空."""
    for attempt in range(1, max_retries + 1):
        try:
            df = fn(**kwargs)
            if len(df):
                print(f"    {name}: {len(df)} rows")
                return to_symbol(df)
            print(f"    {name}: {len(df)} rows (empty, attempt {attempt}/{max_retries})")
        except Exception as e:
            print(f"    {name}: attempt {attempt}/{max_retries} FAILED ({e})")
        time.sleep(2 * attempt)
    print(f"    {name}: giving up after {max_retries} attempts")
    return pd.DataFrame()

_stock_basic_cache: pd.DataFrame | None = None

def fetch_stock_basic_cached() -> pd.DataFrame:
    """Tushare stock_basic (name/list_date), 模块级缓存 — 静态数据每日不变."""
    global _stock_basic_cache
    if _stock_basic_cache is None:
        _stock_basic_cache = safe_fetch(
            pro.stock_basic, "stock_basic", exchange="", list_status="L",
            fields="ts_code,name,list_date",
        )
        if len(_stock_basic_cache):
            _stock_basic_cache = _stock_basic_cache.set_index("symbol")
    return _stock_basic_cache

ohlcv = safe_fetch(pro.daily, "OHLCV", trade_date=TRADE_DATE)
adj   = safe_fetch(pro.adj_factor, "adj_factor", trade_date=TRADE_DATE)
basic = safe_fetch(pro.daily_basic, "daily_basic", trade_date=TRADE_DATE)
limit = safe_fetch(pro.stk_limit, "stk_limit", trade_date=TRADE_DATE)
susp  = safe_fetch(pro.suspend_d, "suspend", trade_date=TRADE_DATE)
cyq   = safe_fetch(pro.cyq_perf, "cyq_perf", trade_date=TRADE_DATE)
margin= safe_fetch(pro.margin_detail, "margin_detail", trade_date=TRADE_DATE)
lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)
bt    = safe_fetch(pro.block_trade, "block_trade", trade_date=TRADE_DATE)

if not len(ohlcv):
    print("FATAL: No OHLCV data")
    sys.exit(1)

# ── block_trade: 当日大宗明细 → raw 缓存 (FINAL STOCK SCAN 读它, 需每日刷新) ──
# 只刷新 raw 缓存, 不落面板 bt_* 列 (特征工程已 REVERT; SCAN 出名单时直接读缓存剔除).
if len(bt):
    try:
        from app.core.config_loader import load_config
        _bt_cache = (
            load_config("data_pipeline_config")
            .get("final_stock_scan", {})
            .get("block_trade_cache", "data/supply_cache/alt_data/block_trade/block_trade_full.parquet")
        )
        _bt_path = (
            _bt_cache if os.path.isabs(_bt_cache)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), _bt_cache)
        )
        old = pd.read_parquet(_bt_path) if os.path.exists(_bt_path) else pd.DataFrame()
        bt_raw = bt.copy()
        bt_raw["date"] = pd.Timestamp(TRADE_DATE)
        _keep = ["symbol", "date", "ts_code", "trade_date", "price", "vol", "amount", "buyer", "seller"]
        bt_raw = bt_raw[[c for c in _keep if c in bt_raw.columns]]
        # 去重: 同 symbol+trade_date 保留最新 (历史日重跑安全)
        merged = pd.concat([old, bt_raw], ignore_index=True)
        merged = merged.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(_bt_path), exist_ok=True)
        _tmp = _bt_path + ".tmp"
        merged.to_parquet(_tmp, index=False)
        os.replace(_tmp, _bt_path)  # 原子替换, 避免崩溃写坏缓存导致 SCAN fail-open
        print(f"    block_trade raw cache: +{len(bt)} rows -> {len(merged)} total")
    except Exception as e:
        print(f"    block_trade raw cache: FAILED ({e})")

# ── 2. Build today's DataFrame ──
print("\n[2] Building today's frame...")
# Start from OHLCV
df = ohlcv.copy()
df["date"] = pd.Timestamp(TRADE_DATE)
# Filter to valid universe
df = df[df["symbol"].isin(valid_universe)].copy()
print(f"    After universe filter: {len(df)} rows")

# Read panel schema
schema = pq.read_schema(PANEL)
panel_cols = schema.names

# ── 3. Merge all sources ──
print("\n[3] Merging sources...")
merges = {
    "daily_basic": (basic, ["volume_ratio",
        "pe_ttm", "pb", "ps_ttm", "total_share", "float_share",
        "free_share", "total_mv", "circ_mv", "dv_ratio", "dv_ttm",
        "turnover_rate"]),
    # V3 (2026-08-02): 只直连 KEEP 基础列 (同名字段).
    # winner_rate→winner_ratio / avg_cost=cost_50pct 由显式块处理;
    # pct_90_high/pct_90_con 由下方从 cost_5pct/cost_95pct 推导; 删除列不落盘.
    "cyq_perf": (cyq, ["cost_50pct", "cost_95pct", "weight_avg"]),
    "margin_detail": (margin, ["rzye", "rqye", "rzmre", "rqyl"]),
}
# Aggregate LHB by symbol
if len(lhb):
    lhb = lhb.groupby("symbol").agg({"l_buy": "sum", "l_sell": "sum", "net_amount": "sum"}).reset_index()

for src_name, (src_df, cols) in merges.items():
    if not len(src_df):
        continue
    available = [c for c in cols if c in src_df.columns]
    if not available:
        continue
    smap = src_df.set_index("symbol")[available]
    for c in available:
        tgt = c
        if tgt in panel_cols and tgt not in df.columns:
            df[tgt] = df["symbol"].map(smap[c])

# Direct mapping: Tushare source cols -> panel target cols (names differ)
if len(cyq) and "winner_rate" in cyq.columns and "winner_rate" not in df.columns:
    wmap = cyq.set_index("symbol")["winner_rate"]
    df["winner_rate"] = df["symbol"].map(wmap)

if len(basic):
    bmap = basic.set_index("symbol")
    if "turnover_rate_f" in bmap.columns and "free_float_turnover_rate" in panel_cols:
        df["free_float_turnover_rate"] = df["symbol"].map(bmap["turnover_rate_f"])

# stk_limit direct mapping
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols and tgt_col not in df.columns:
            df[tgt_col] = df["symbol"].map(lmap[src_col])

# Rename mappings
rename_map = {
    "rzye": "margin_balance", "rqye": "short_balance",
    "rzmre": "margin_buy_amt", "rqyl": "short_sell_vol",
}
for old, new in rename_map.items():
    if old in df.columns:
        df[new] = df[old]

# LHB merge
if len(lhb):
    for src, tgt in [("l_buy", "lhb_buy_amt"), ("l_sell", "lhb_sell_amt"), ("net_amount", "lhb_net_buy")]:
        if tgt in panel_cols:
            smap = lhb.set_index("symbol")[src]
            df[tgt] = df["symbol"].map(smap)

# GLM 龙虎榜 spec: 席位明细 (散户大本营"拉萨"/机构专用) per 股-日.
# 列由回填脚本首次加入面板 schema, 此处仅填充已存在的列 (panel_cols 守卫).
if len(lhb):
    try:
        from app.pipeline1.data_supply import DataSupplyChain
        _seat_supply = DataSupplyChain()
    except Exception:
        _seat_supply = None
    if _seat_supply is not None:
        seat_cols = ["lhb_retail_buy", "lhb_retail_sell", "lhb_inst_buy", "lhb_inst_sell"]
        for col in seat_cols:
            if col in panel_cols and col not in df.columns:
                df[col] = np.nan
        for sym in lhb["symbol"].unique().tolist():
            agg = _seat_supply.fetch_lhb_seat_aggregate(sym, TRADE_DATE)
            if not agg:
                continue
            m = df["symbol"] == sym
            for k, v in agg.items():
                if k in panel_cols and k in df.columns:
                    df.loc[m, k] = v

# ── 4. Compute derived columns ──
print("\n[4] Computing derived columns...")

# --- hfq prices ---
if len(adj) and "adj_factor" in adj.columns:
    fmap = adj.set_index("symbol")["adj_factor"]
    factor = df["symbol"].map(fmap)
    for col in ["open", "high", "low", "close"]:
        tgt = f"{col}_hfq"
        if tgt in panel_cols:
            df[tgt] = df[col] * factor

# --- winner_ratio = winner_rate / 100 ([0,1] ratio) ---
if "winner_rate" in df.columns and "winner_ratio" in panel_cols:
    df["winner_ratio"] = df["winner_rate"] / 100.0

# --- avg_cost = cost_50pct ---
if "cost_50pct" in df.columns and "avg_cost" in panel_cols:
    df["avg_cost"] = df["cost_50pct"]

# --- pct_90/70 CI ---
# Formula from cyq_calculator.py: concentration = (hi-lo)/(hi+lo)
# Result in [0, 1]: smaller = more concentrated (bullish)
def safe_div(a, b):
    return np.where((b.notna() & (b != 0)), a / b, np.nan)

# V3 删列 (2026-08-02): pct_90_low / pct_70_* / cost_5pct/15pct/85pct 不再落盘.
# pct_90_high / pct_90_con 仍 KEEP → 直接从原始 cyq (Tushare cyq_perf) 推导.
if len(cyq) and {"cost_5pct", "cost_95pct"} <= set(cyq.columns):
    cmap = cyq.set_index("symbol")[["cost_5pct", "cost_95pct"]]
    c5 = df["symbol"].map(cmap["cost_5pct"])
    c95 = df["symbol"].map(cmap["cost_95pct"])
    if "pct_90_high" in panel_cols and "pct_90_high" not in df.columns:
        df["pct_90_high"] = c95
    if "pct_90_con" in panel_cols and "pct_90_con" not in df.columns:
        df["pct_90_con"] = safe_div(c95 - c5, c95 + c5)

# --- is_suspended ---
if len(susp):
    suspended_set = set(susp["symbol"].unique())
    df["is_suspended"] = df["symbol"].isin(suspended_set).astype(int)

# --- 元数据列: 与基建面板同语义 (board/industry) ---
# 基建 (panel_builder): board=board_of (main/GEM/STAR); industry=东财行业板块, 缺省 UNKNOWN.
# 日更必须延续这些语义 — 不能再引入 stock_basic 的交易所代码 / 109 行业, 否则尾部特征跳变.
# is_st/list_days 已从 V3 移除: ST/次新由下方 ingest scan 按 stock_basic name/list_date 过滤.
if "board" in panel_cols:
    df["board"] = df["symbol"].map(board_of)
meta_carry_cols = ["industry"]
meta_hist = pq.read_table(PANEL, columns=["symbol", "date"] + meta_carry_cols).to_pandas()
meta_hist = meta_hist[meta_hist["date"] < pd.Timestamp(TRADE_DATE)]
last_meta = meta_hist.sort_values("date").groupby("symbol").last()
if "industry" in panel_cols and "industry" in last_meta.columns:
    smap = last_meta["industry"].dropna()
    df["industry"] = df["symbol"].map(smap).fillna("UNKNOWN")
print(f"    meta carry: board={df['board'].nunique() if 'board' in df.columns else '-'} | "
      f"industry nunique={df['industry'].nunique() if 'industry' in df.columns else '-'}")

# --- 入库扫描: ST/*ST 股 或 上市 < INGEST_MIN_LIST_DAYS 交易日 不进入 V3 ---
_stock_info = fetch_stock_basic_cached()
# 交易日历 = 面板唯一 date 列 (与面板历史一致的交易日口径).
_trade_cal = pd.DatetimeIndex(sorted(yesterday["date"].unique()))
df, _dropped = apply_ingest_scan(df, _stock_info, TRADE_DATE, INGEST_MIN_LIST_DAYS, _trade_cal)
if len(_stock_info):
    print(f"    Ingest scan: dropped {_dropped} rows (ST or trading_days < {INGEST_MIN_LIST_DAYS})")
else:
    print("    WARN: Ingest scan SKIPPED — stock_basic 不可用, ST/次新股未被过滤!")

# --- Simple derived features (no history needed) ---
if all(c in df.columns for c in ["high", "low", "pre_close"]):
    df["intraday_range"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, np.nan)

if all(c in df.columns for c in ["close", "pre_close"]):
    df["pctChg"] = (df["close"] / df["pre_close"] - 1) * 100

if all(c in df.columns for c in ["amount", "close"]):
    valid = df["amount"].notna() & df["close"].notna() & (df["close"] > 0)
    if "volume" in panel_cols and "volume" not in df.columns:
        df["volume"] = np.nan
    df.loc[valid, "volume"] = df.loc[valid, "amount"] / df.loc[valid, "close"]

# ── 5. Read history for rolling features ──
print("\n[5] Computing rolling features from history...")
symbols = sorted(df["symbol"].unique())

# Read only needed columns from panel
hist_cols = ["symbol", "date", "close_hfq", "volume", "high", "low", "pre_close", "open", "amount", "pct_90_con"]
hist_cols = [c for c in hist_cols if c in panel_cols]
hist = pq.read_table(PANEL, columns=hist_cols).to_pandas()
hist = hist[hist["symbol"].isin(symbols)]
hist = hist[hist["date"] <= pd.Timestamp(TRADE_DATE)]
hist = hist.sort_values(["symbol", "date"])
print(f"    History: {len(hist):,} rows for {hist.symbol.nunique()} symbols")

close_today = df.set_index("symbol")["close_hfq"]

# bias features
bias_windows = {"bias_5": 5, "bias_10": 10, "bias_20": 20, "bias_60": 60, "bias_120": 120, "bias_250": 250}
for col_name, window in bias_windows.items():
    if col_name not in panel_cols:
        continue
    ma = hist.groupby("symbol")["close_hfq"].rolling(window, min_periods=window).mean()
    ma.index = ma.index.droplevel(0)
    last_ma = ma.groupby(hist["symbol"]).last()
    vals = {}
    for sym, m in last_ma.dropna().items():
        if sym in close_today.index and pd.notna(close_today[sym]) and m > 0:
            vals[sym] = close_today[sym] / m - 1
    df[col_name] = df["symbol"].map(vals)
    n_filled = df[col_name].notna().sum()
    if n_filled < len(df):
        print(f"    {col_name}: {n_filled}/{len(df)}")

# bias cross (set to 0 — no diff history for single day)
for cross_col in ["bias_5_20_cross", "bias_20_60_cross"]:
    if cross_col in panel_cols:
        df[cross_col] = 0.0

# ma_vol_ratio_5_20
if "ma_vol_ratio_5_20" in panel_cols and "volume" in hist.columns:
    g = hist.groupby("symbol")["volume"]
    ma5 = g.rolling(5, min_periods=3).mean()
    ma5.index = ma5.index.droplevel(0)
    ma20 = g.rolling(20, min_periods=10).mean()
    ma20.index = ma20.index.droplevel(0)
    l5, l20 = ma5.groupby(hist["symbol"]).last(), ma20.groupby(hist["symbol"]).last()
    vals = {}
    for sym in l5.index.intersection(l20.index):
        if pd.notna(l20[sym]) and l20[sym] > 0:
            vals[sym] = l5[sym] / l20[sym]
    df["ma_vol_ratio_5_20"] = df["symbol"].map(vals)

# amplitude_5d
if "amplitude_5d" in panel_cols and all(c in hist.columns for c in ["high", "low", "pre_close"]):
    denom = hist["pre_close"].replace(0, np.nan)
    hist["_amp"] = (hist["high"] - hist["low"]) / denom
    ma5 = hist.groupby("symbol")["_amp"].rolling(5, min_periods=3).mean()
    ma5.index = ma5.index.droplevel(0)
    df["amplitude_5d"] = df["symbol"].map(ma5.groupby(hist["symbol"]).last())

# vol_surge, amt_surge
for col, surge_col in [("volume", "vol_surge"), ("amount", "amt_surge")]:
    if surge_col in panel_cols and col in hist.columns:
        g = hist.groupby("symbol")[col]
        ma20 = g.rolling(20, min_periods=10).mean()
        ma20.index = ma20.index.droplevel(0)
        std20 = g.rolling(20, min_periods=10).std()
        std20.index = std20.index.droplevel(0)
        last_m = ma20.groupby(hist["symbol"]).last()
        last_s = std20.groupby(hist["symbol"]).last()
        today_vals = df.set_index("symbol")[col]
        vals = {}
        for sym in last_m.index.intersection(last_s.index):
            if (sym in today_vals.index and pd.notna(today_vals[sym])
                and pd.notna(last_s[sym]) and last_s[sym] > 0):
                vals[sym] = (today_vals[sym] - last_m[sym]) / last_s[sym]
        df[surge_col] = df["symbol"].map(vals)

# --- CYQ 筹码形态扩展列 (chip_morphology): compute_cyq_today 补今日 8 列 ---
cyq_ext_cols = [c for c in cyq_ext.TARGET_COLS if c in panel_cols]
cyq_need = ["open", "high", "low", "close", "turnover_rate"]
if cyq_ext_cols and all(c in df.columns for c in cyq_need):
    cyq_hist_cols = ["symbol", "date"] + cyq_need
    if "peak_price" in panel_cols:
        cyq_hist_cols.append("peak_price")
    cyq_hist = pq.read_table(PANEL, columns=cyq_hist_cols).to_pandas()
    cyq_hist = cyq_hist[cyq_hist["symbol"].isin(symbols)]
    cyq_hist = cyq_hist[cyq_hist["date"] < pd.Timestamp(TRADE_DATE)]
    # NaN turnover_rate → 0, 否则 min(1.0, nan/100) 污染整个筹码分布
    cyq_hist["turnover_rate"] = cyq_hist["turnover_rate"].fillna(0.0)
    today_slim = df[["symbol", "date"] + cyq_need].copy()
    today_slim["turnover_rate"] = today_slim["turnover_rate"].fillna(0.0)
    try:
        cyq_today = cyq_ext.compute_cyq_today(cyq_hist, today_slim)
        for c in cyq_ext_cols:
            if c in cyq_today.columns:
                df[c] = df["symbol"].map(cyq_today.set_index("symbol")[c])
        print(f"    cyq_ext today: {len(cyq_ext_cols)} 列已计算 "
              f"({', '.join(cyq_ext_cols)})")
    except Exception as e:
        print(f"    cyq_ext today: FAILED ({e})")

# --- CYQ 派生列 (cost_bias / conc_trend_20d / conc_90_industry_rank) ---
# 公式与 dim21_chip_tushare (feature_engine_v35.py) / scripts/_add_cyq_derived_cols.py 一致.
# cost_bias 与 conc_90_industry_rank 仅用今日行; conc_trend_20d 用 hist 的 t-20 pct_90_con.
derived_cyq = ["cost_bias", "conc_trend_20d", "conc_90_industry_rank"]
if any(c in panel_cols for c in derived_cyq):
    if "cost_bias" in panel_cols and {"close_hfq", "cost_50pct"} <= set(df.columns):
        df["cost_bias"] = (df["close_hfq"] - df["cost_50pct"]) / df["cost_50pct"].replace(0, np.nan)
    if "conc_trend_20d" in panel_cols and "pct_90_con" in df.columns and "pct_90_con" in hist.columns:
        h90 = hist[hist["date"] < pd.Timestamp(TRADE_DATE)]
        prev20 = h90.groupby("symbol")["pct_90_con"].shift(19).groupby(h90["symbol"]).last()
        df["conc_trend_20d"] = df["pct_90_con"] / df["symbol"].map(prev20).replace(0, np.nan)
    if "conc_90_industry_rank" in panel_cols and "pct_90_con" in df.columns and "industry" in df.columns:
        df["conc_90_industry_rank"] = (
            df.groupby("industry", observed=True)["pct_90_con"].rank(pct=True).fillna(0.5)
        )
    print(f"    cyq derived today: {[c for c in derived_cyq if c in panel_cols]}")

# ── 6. Forward-fill quarterly & slow data ──
print("\n[6] Forward-filling quarterly/slow data...")
ffill_cols = [
    "roe", "roa", "gross_margin", "net_margin", "eps_yoy", "rev_yoy", "profit_yoy",
    "debt_ratio", "current_ratio", "asset_turnover", "inventory_turnover",
    "ocf_to_or", "eps", "bps", "ocfps", "revenue_ps",
    "roe_deducted", "roe_yoy", "q_roe",
    "ar_turnover",
    "sh_change_vol", "sh_change_amt_total",
    "sh_net_change_sign", "sh_net_sign",
    "sh_evt_start_date", "sh_evt_end_date",
    "sw_l1_name", "sw_l2_name", "sw_l3_name",
    "margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol",
    "dt_eps", "q_ocf_to_sales",
    "announce_date",
]
ffill_cols = [c for c in ffill_cols if c in panel_cols]
# 慢列始终 ffill: 旧逻辑 `df[c].isna().all()` 在今日行部分拉取成功时会跳过整列,
# 导致其余股票该列 NaN (2026-07-29/30/31 尾行缺失的根因之一)。
needed_ffill = ffill_cols

if needed_ffill:
    ffill_hist = pq.read_table(PANEL, columns=["symbol", "date"] + needed_ffill).to_pandas()
    ffill_hist = ffill_hist[ffill_hist["symbol"].isin(symbols)]
    ffill_hist = ffill_hist[ffill_hist["date"] < pd.Timestamp(TRADE_DATE)]
    ffill_hist = ffill_hist.sort_values(["symbol", "date"])
    for col in needed_ffill:
        if col in ffill_hist.columns:
            ffill_hist[col] = ffill_hist.groupby("symbol")[col].ffill()
    last_per_stock = ffill_hist.groupby("symbol").last()
    filled_count = 0
    for col in needed_ffill:
        if col in last_per_stock.columns:
            smap = last_per_stock[col].dropna()
            if not len(smap):
                continue
            mapped = df["symbol"].map(smap)
            # fillna 而非覆盖: 保留今日直接拉取的实时值, 只补缺失行
            if col in df.columns:
                df[col] = df[col].fillna(mapped)
            else:
                df[col] = mapped
            if df[col].notna().sum() > 0:
                filled_count += 1
    print(f"    Forward-filled: {filled_count} columns")

# ── 6.5 大宗交易当日聚合 → 4 个 bt_ 原始列 (dim33 EWMA 上游, 非特征) ──
# 算法忠实还原自 block_trade_agg v2.0 (pyc): L1 噪音清洗 (不删除, 仅排除出聚合) +
# 有效单聚合. 单位: vol=万股, amount=万元, daily_amt=万元 (面板 amount=千元 → /10),
# circ_mv=万元, close=元.
if len(bt):
    try:
        _bt = bt.copy()
        _bt["date"] = pd.Timestamp(TRADE_DATE)
        snap = df[["symbol", "close", "amount", "circ_mv"]].drop_duplicates("symbol")
        snap["daily_amt"] = snap["amount"] / 10.0  # 千元 → 万元
        _bt = _bt.merge(
            snap[["symbol", "close", "daily_amt", "circ_mv"]],
            on=["symbol"], how="left",
        )
        _bt = _bt[_bt["close"].notna()].copy()
        if len(_bt):
            # L1 噪音标记 (规则与历史聚合一致)
            def _broker(seat):
                s = str(seat)
                idx = s.find("证券")
                return s[idx:idx + 2] if idx >= 0 else ""

            same_broker = (_bt["buyer"].map(_broker) == _bt["seller"].map(_broker)) & (
                _bt["buyer"] != _bt["seller"]
            )
            # 折价 = (price − close) / close; 噪音判定依赖折价, 须先算
            _bt["discount"] = (_bt["price"] - _bt["close"]) / _bt["close"].replace(0, np.nan)
            _bt["is_noise"] = (
                (_bt["buyer"] == _bt["seller"])
                | (same_broker & (_bt["discount"] < -0.1))
                | ((_bt["vol"] < 10) & (_bt["discount"].abs() < 0.01))
            )
            _inst_kw = ("机构专用", "QFII", "合格境外", "社保", "养老", "资产管理", "资管", "保险", "信托")
            _bt["is_inst_buyer"] = _bt["buyer"].map(lambda s: any(k in str(s) for k in _inst_kw))
            v = _bt[~_bt["is_noise"]].copy()
            if len(v):
                grp = v.groupby("symbol")
                total_amt = grp["amount"].sum()
                wavg = (v["price"] * v["vol"]).groupby(v["symbol"]).sum() / grp["vol"].sum().replace(0, np.nan)
                close = grp["close"].first()
                daily_amt = grp["daily_amt"].first()
                circ_mv = grp["circ_mv"].first()
                disc = (wavg - close) / close.replace(0, np.nan)
                any_inst = v.groupby("symbol")["is_inst_buyer"].max()
                bt_today = pd.DataFrame({
                    "bt_count": grp.size(),
                    "bt_disc_raw": (-disc).clip(lower=0),
                    "bt_inst_absorb": any_inst * total_amt / daily_amt.replace(0, np.nan),
                    "bt_amt_ratio_float_mv": total_amt / circ_mv.replace(0, np.nan),
                })
                for c in bt_today.columns:
                    df[c] = df["symbol"].map(bt_today[c])
                print(f"    block_trade agg: {len(bt_today)} symbols -> "
                      f"{[c for c in bt_today.columns if c in df.columns]}")
    except Exception as e:
        print(f"    block_trade agg: FAILED ({e})")

# ── 7. Align to panel schema ──
print("\n[7] Aligning to panel schema...")
for c in panel_cols:
    if c not in df.columns:
        df[c] = pd.NA
df = df[list(panel_cols)]

# Cast types
for field in schema:
    if field.name in df.columns:
        try:
            df[field.name] = df[field.name].astype(field.type.to_pandas_dtype())
        except Exception:
            pass

# ── 8. Append to panel (streaming) ──
print("\n[8] Appending to panel...")
today_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

pf = pq.ParquetFile(PANEL)
tmp_path = PANEL + ".tmp"
writer = pq.ParquetWriter(tmp_path, schema=schema)

for rg_idx in range(pf.metadata.num_row_groups):
    rg = pf.read_row_group(rg_idx)
    rg_df = rg.to_pandas()
    # Remove today's old rows (idempotent: re-run after crash rebuilds cleanly)
    rg_df = rg_df[rg_df["date"] != pd.Timestamp(TRADE_DATE)]
    if len(rg_df):
        writer.write_table(pa.Table.from_pandas(rg_df, schema=schema, preserve_index=False))

writer.write_table(today_table)
writer.close()
pf.close()

# Atomic replace
if os.path.exists(PANEL):
    os.remove(PANEL)
os.rename(tmp_path, PANEL)

# ── 9. Audit ──
print(f"\n{'='*55}")
print(f"DONE: {TRADE_DATE}")
print(f"{'='*55}")
pf2 = pq.ParquetFile(PANEL)
print(f"Panel: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
pf2.close()

final = pq.read_table(PANEL, filters=[("date", "=", pd.Timestamp(TRADE_DATE))]).to_pandas()
n_cols = len(final.columns)
filled = sum(final.notna().sum() > 0)
empty = n_cols - filled
print(f"Filled: {filled}/{n_cols} ({filled/n_cols:.1%})")
empty_cols = sorted([c for c in final.columns if final[c].notna().sum() == 0])
if empty_cols:
    print(f"Empty ({len(empty_cols)}): {', '.join(empty_cols)}")
else:
    print("All columns have data!")
