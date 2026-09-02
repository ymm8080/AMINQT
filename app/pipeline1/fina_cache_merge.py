"""公告管线 fina_indicator 缓存 → V3 面板合并 — 纯逻辑抽离自 _daily_fetch.py.

2026-09-02 事故: 面板第 6 节只做 ffill (面板历史停在 Q1 值), 而公告管线
(run_announcement_pipeline.py → DataSupplyChain.fetch_fina_indicator) 每日把
新鲜财报写进 data/supply_cache/alt_data/fina_indicator/ 却没人接进面板 —
Q2 财报季 (2026-08-14→09-01) 4950 只股票只有 15 只 roe 值变化.

缓存文件布局 (DataSupplyChain._alt_cache_path 生成, 口径以 _fina_convert 为准):
  all__{start}_{end}.parquet  — 每文件 = 公告管线一次运行抓到的"当日已公告"快照
  列: symbol (6 位代码 str), announce_date (公告日, datetime64),
      report_period (报告期, datetime64 — Tushare end_date 的面板口径),
      _ts_code (多数为 NaN, 忽略), 15 个财务数值列.
  同一股票可跨文件出现多期/重复记录 → 选择时按 (report_period, announce_date)
  取每股最新, 加载时全行去重.

另有 new_symbols_*.parquet 前缀文件 (建仓快照), glob "all__*" 天然排除.
"""

import glob
import os

import numpy as np
import pandas as pd

# 面板 fina 列中允许从财务缓存覆盖的列 (ffill_cols 的 fina 子集).
# 绝不含 announce_date / sh_* / margin / sw_l*_name — 这些各有语义
# (sh_* 刚修过, margin 有 T+1 语义, dim31 用户暂缓; announce_date 是
# datetime64, 走 overwrite_today_announce_date 专用车道不混入本 float64 清单),
# 实际取用 = 本清单 ∩ ffill_cols ∩ 缓存实际有的列.
FINA_CACHE_COLS = [
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "eps_yoy",
    "rev_yoy",
    "profit_yoy",
    "debt_ratio",
    "current_ratio",
    "asset_turnover",
    "inventory_turnover",
    "ocf_to_or",
    "eps",
    "bps",
    "ocfps",
    "revenue_ps",
    "roe_deducted",
    "roe_yoy",
    "q_roe",
    "ar_turnover",
    "dt_eps",
    "q_ocf_to_sales",
]

# 缓存目录默认值 — 权威定义在 DataSupplyChain (cache_dir="data/supply_cache"
# + ALT_CACHE_SUBDIR="alt_data" + "fina_indicator"); 此处按模块位置反推仓库根,
# 避免第二份硬编码. 生产脚本也可显式传参覆盖.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE_DIR = os.path.join(
    REPO_ROOT, "data", "supply_cache", "alt_data", "fina_indicator"
)


def load_fina_cache(cache_dir=DEFAULT_CACHE_DIR):
    """读全部 all__*.parquet 日快照并合并 (全行去重 — 跨文件快照重叠).

    单文件损坏只跳过并告警, 不拖垮当日 fetch (缺失部分由 ffill 兜底).
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(cache_dir, "all__*.parquet"))):
        try:
            frames.append(pd.read_parquet(path))
        except Exception as e:
            print(f"    fina cache: skip unreadable {os.path.basename(path)} ({e})")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates().reset_index(drop=True)


def _allowed_fina_cols(cache_df, fina_cols=None):
    """实际取用列 = (fina_cols or FINA_CACHE_COLS) ∩ FINA_CACHE_COLS 白名单
    ∩ 缓存实际有的列 — 白名单是硬闸, 即使调用方显式传参也挡住 announce_date /
    sh_* / margin 等保护条目."""
    requested = FINA_CACHE_COLS if fina_cols is None else fina_cols
    return [c for c in requested if c in FINA_CACHE_COLS and c in cache_df.columns]


def select_latest_fina(cache_df, fina_cols=None):
    """每股最新一期选择: report_period 最新, 其次 announce_date 最新 (向量化).

    Returns:
        DataFrame [symbol, announce_date, report_period] + fina 列 (取
        fina_cols ∩ 缓存实际有的列), 每股一行.
    """
    cols = _allowed_fina_cols(cache_df, fina_cols)
    if not len(cache_df):
        # 空缓存没有"缓存实际有的列"可交 — 保留白名单结构列, 行为空
        req = FINA_CACHE_COLS if fina_cols is None else fina_cols
        return pd.DataFrame(
            columns=["symbol", "announce_date", "report_period"]
            + [c for c in req if c in FINA_CACHE_COLS]
        )
    if not len(cols):
        return pd.DataFrame(columns=["symbol", "announce_date", "report_period"] + cols)
    df = cache_df[cache_df["report_period"].notna()]
    if not len(df):
        return pd.DataFrame(columns=["symbol", "announce_date", "report_period"] + cols)
    latest = df.sort_values(
        ["symbol", "report_period", "announce_date"], kind="stable"
    ).drop_duplicates("symbol", keep="last")
    return latest[["symbol", "announce_date", "report_period"] + cols].reset_index(
        drop=True
    )


def overwrite_today_from_cache(df, cache_df, fina_cols=None):
    """今日行 fina 列用缓存值覆盖 (in-place), 返回实际覆盖的列名列表.

    覆盖而非 fillna: 缓存是更新的真相 (2026-09-02 事故根因就是只 ffill 停在 Q1).
    缓存没有的股票/该列缺值的格 → NaN, 由调用方其后保留的 ffill 逻辑兜底
    (面板历史值), 因此本函数必须在 ffill 之前调用.
    """
    if not len(cache_df) or not len(df):
        return []
    cols = _allowed_fina_cols(cache_df, fina_cols)
    if not cols:
        return []
    latest = select_latest_fina(cache_df, fina_cols=cols)
    smap = latest.set_index("symbol")
    for col in cols:
        df[col] = df["symbol"].map(smap[col])
    return cols


def replay_fina_asof(panel_rows, cache_df, fina_cols=None):
    """PIT 逐公告日回放: 对每股取 announce_date <= 行日 的最新缓存记录值.

    面板存量修复 (tmp_t/repair_fina_panel_20260902.py) 用. 修复窗内若一律
    覆盖每股最新值会前视 — Q2 于 08-28 公告的股票, 其 08-10 行就拿到 Q2 值
    (违反 look-ahead 铁律); merge_asof backward 保证每股只有自身公告日之后
    的行拿到新值, 之前的行为 NaN (调用方保留面板原值 — 修复窗起点前 Q1 值
    本就是 PIT 正确的, 无需动).

    Args:
        panel_rows: DataFrame [symbol, date] (任意行序, 按"位置"返回对齐结果).
        cache_df:   load_fina_cache() 的输出.

    Returns:
        DataFrame 与 panel_rows 同行序 (RangeIndex), 列 = fina_cols ∩ 缓存列;
        无公告记录的格为 NaN.
    """
    cols = _allowed_fina_cols(cache_df, fina_cols)
    n = len(panel_rows)
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols, dtype="float64")
    if n == 0 or not cols:
        return out
    right = cache_df[
        cache_df["announce_date"].notna() & cache_df["report_period"].notna()
    ]
    if not len(right):
        return out
    # 同股同公告日多条 (跨文件快照重叠) → 保留 report_period 最新一条 (更正公告为准)
    right = right.sort_values(
        ["symbol", "announce_date", "report_period"], kind="stable"
    ).drop_duplicates(["symbol", "announce_date"], keep="last")
    right = right[["symbol", "announce_date"] + cols].sort_values(
        "announce_date", kind="stable"
    )
    # merge_asof 要求两侧 on 键全局有序; 行序还原用稳定排序的 perm (向量化)
    dates = panel_rows["date"].to_numpy()
    perm = np.argsort(dates, kind="stable")
    left = panel_rows.iloc[perm][["symbol", "date"]].reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        right,
        by="symbol",
        left_on="date",
        right_on="announce_date",
        direction="backward",
    )
    out.iloc[perm] = merged[cols].to_numpy(dtype="float64")
    return out


def overwrite_today_announce_date(df, cache_df):
    """今日行 announce_date 用缓存每股最新记录的公告日覆盖 (in-place).

    独立车道不走 FINA_CACHE_COLS (float64 财报数值白名单): announce_date 是
    datetime64 语义列, 但与 fina 车道同 PIT 语义 — 缓存快照只含已公告记录,
    今日行取"已知最新公告日"无前瞻. report_period 缺失的脏行不入选; 缓存外
    股票 → NaT, 由调用方其后保留的 ffill 兜底, 因此必须在 ffill 之前调用.
    返回覆盖后今日行非空的行数 (缓存空/无 announce_date 列 → 0).
    """
    if not len(cache_df) or not len(df) or "announce_date" not in cache_df.columns:
        return 0
    right = cache_df[cache_df["announce_date"].notna()]
    if not len(right):
        return 0
    if "report_period" in right.columns:
        right = right[right["report_period"].notna()]
        if not len(right):
            return 0
        sort_keys = ["symbol", "report_period", "announce_date"]
    else:
        sort_keys = ["symbol", "announce_date"]
    latest = right.sort_values(sort_keys, kind="stable").drop_duplicates(
        "symbol", keep="last"
    )
    smap = latest.set_index("symbol")["announce_date"]
    df["announce_date"] = df["symbol"].map(smap)
    return int(df["announce_date"].notna().sum())


def replay_announce_date(panel_rows, cache_df):
    """PIT 回放 announce_date: 每行 = 截至该日已知最新一期报告的公告日.

    面板 announce_date 存量修复用, 同 replay_fina_asof 的 merge_asof backward
    模式: 每股只有自身公告日之后的行才拿到新公告日, 公告前 → NaT (调用方保留
    面板原值). 同股多期记录值=键 (公告日本身), 公告日更晚者自然胜出.

    Returns:
        Series datetime64[ns], 与 panel_rows 同行序同 index; 无记录 → NaT.
    """
    n = len(panel_rows)
    out = pd.Series(pd.NaT, index=panel_rows.index, dtype="datetime64[ns]")
    if n == 0 or not len(cache_df) or "announce_date" not in cache_df.columns:
        return out
    right = cache_df.loc[cache_df["announce_date"].notna(), ["symbol", "announce_date"]]
    if not len(right):
        return out
    right = right.copy()
    right["symbol"] = right["symbol"].astype(str)
    right = right.drop_duplicates().sort_values("announce_date", kind="stable")
    dates = pd.to_datetime(panel_rows["date"])
    perm = np.argsort(dates.to_numpy(), kind="stable")
    left = pd.DataFrame(
        {
            "symbol": panel_rows["symbol"].astype(str).to_numpy()[perm],
            "date": dates.to_numpy()[perm],
        }
    )
    merged = pd.merge_asof(
        left,
        right,
        by="symbol",
        left_on="date",
        right_on="announce_date",
        direction="backward",
    )
    out.iloc[perm] = merged["announce_date"].to_numpy()
    return out
