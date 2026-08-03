# -*- coding: utf-8 -*-
"""Global configuration: paths, stock pool, date ranges, execution mode.

Secrets (iFinD credentials) are loaded from environment / .env — never
hardcoded. Date handling uses datetime objects, not string comparison.
"""
import os
from datetime import date
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTRADAY_DIR = DATA_DIR / "intraday"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "app" / "models" / "trained"

# Non-parquet data outputs (logs, reports, JSON state, CSV exports, etc.)
# are kept outside the repo data/ directory so that data/ contains only
# parquet-format analytical datasets.  Override via AMINQT_DATA_OTHERS env.
DATA_OTHERS_DIR = Path(
    os.getenv("AMINQT_DATA_OTHERS", str(PROJECT_ROOT.parent / "DATA OTHERS"))
)

for _d in (RAW_DIR, INTRADAY_DIR, PROCESSED_DIR, MODEL_DIR, DATA_OTHERS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── V3 Panel (single source of truth) ────────────────────────
# Override via PANEL_PATH env var; defaults to project data/ directory.
# _daily_fetch.py writes here → all read paths must resolve to the same file.
PANEL_V3_PATH = Path(
    os.getenv("PANEL_PATH", str(DATA_DIR / "panel_full_enriched_v3.parquet"))
)
PANEL_V3_FALLBACK = DATA_DIR / "panel_full_enriched_v2.parquet"

# ── V3 CYQ 基础列删减 (2026-08-02 A/B/C 决策) ────────────────
# benefit_part 已并入 winner_ratio (别名); cost_50pct 因 cost_bias 依赖而保留.
CYQ_BASE_KEEP = [
    "winner_ratio",
    "avg_cost",
    "pct_90_high",
    "pct_90_con",
    "cost_50pct",
    "cost_95pct",
    "weight_avg",
]
CYQ_BASE_DELETE = [
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
    "pct_90_low",
    "cost_5pct",
    "cost_15pct",
    "cost_85pct",
]


def data_others_path(path: str | Path) -> Path:
    """Return the DATA_OTHERS location for a non-parquet path.

    Paths that start with ``data/`` are mapped to ``DATA_OTHERS_DIR`` so that
    ``data/`` contains only parquet-format analytical datasets. Parquet paths
    are rejected because they must remain in ``DATA_DIR``.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        raise ValueError(f"parquet files must stay in data/: {path}")
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    return DATA_OTHERS_DIR.joinpath(*parts)


def data_path(path: str | Path) -> Path:
    """Return the location under DATA_DIR for a parquet dataset path."""
    p = Path(path)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == "data":
        parts = parts[1:]
    return DATA_DIR.joinpath(*parts)

# ── Data source: "ifind" | "akshare" (akshare = fallback/dev) ──
DATA_SOURCE = os.getenv("AMINQT_DATA_SOURCE", "akshare")

# iFinD credentials (env only)
IFIND_USER = os.getenv("IFIND_USER", "")
IFIND_PASSWORD = os.getenv("IFIND_PASSWORD", "")

# Tushare token (env only)
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ── Stock pool (5-symbol test set; expand later) ─────────────
STOCK_LIST = ["000001", "000002", "600519", "000858", "600036"]

# ── Date ranges (Phase 3 split: train 18-20, val 21, test 22-24) ──
DATA_START = date(2018, 1, 1)
DATA_END = date(2024, 12, 31)
TRAIN_END = date(2020, 12, 31)
VAL_END = date(2021, 12, 31)
TEST_START = date(2022, 1, 1)

# Anti-crawl delay between symbol fetches
DOWNLOAD_SLEEP_SEC = 0.5


class ExecutionMode(str, Enum):
    """M3 execution modes."""
    AUTO = "auto"      # granted: orders sent to broker directly
    MANUAL = "manual"  # pop-up recommendation only, user confirms


# ── Execution ─────────────────────────────────────────────────
EXECUTION_MODE = ExecutionMode(os.getenv("AMINQT_EXEC_MODE", "manual"))
EXECUTION_BROKER = os.getenv("AMINQT_BROKER", "sim")  # "sim" | "xt"

# ── Risk filter hard constraints (Phase 4) ────────────────────
MIN_AMOUNT = 50_000_000         # 成交额 >= 5000万
PRICE_LIMIT_PCT = 9.5           # |涨跌幅| <= 9.5%
MAX_ACCOUNT_DRAWDOWN_PCT = 3.0  # 账户回撤 > 3% → 返回空列表
