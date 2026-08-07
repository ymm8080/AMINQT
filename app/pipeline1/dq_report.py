"""
数据质量日检 (P19.0 W1, 量化铁律 OHLCV 校验)
==================================================
铁律: 加载后必须校验 high >= low, high >= open/close, low <= open/close,
volume >= 0; 异常数据不得静默丢弃 — 本模块产出违规明细 + 日检报告,
校验不通过的数据必须显式处置 (剔除并留痕, 或告警中止).

每日 16:00 前自动生成日检报告 (P19.0 W1 检查点).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KEY_COLS = ("open", "high", "low", "close", "volume", "amount")


def ohlcv_violations(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 铁律校验: 返回违规行明细 (附违规原因), 空 = 全部通过.

    校验项 (量化铁律):
      high >= low / high >= open / high >= close / low <= open / low <= close
      volume >= 0 / amount >= 0
    异常数据不得静默丢弃 — 违规行全部列出让调用方显式处置.
    """
    checks = {
        "high<low": df["high"] < df["low"],
        "high<open": df["high"] < df["open"],
        "high<close": df["high"] < df["close"],
        "low>open": df["low"] > df["open"],
        "low>close": df["low"] > df["close"],
        "volume<0": df["volume"] < 0,
        "amount<0": df["amount"] < 0,
    }
    mask = np.zeros(len(df), dtype=bool)
    reasons = [""] * len(df)
    for name, m in checks.items():
        m = m.fillna(False).values
        mask |= m
        reasons = [
            f"{r};{n}" if mm else r
            for r, n, mm in zip(reasons, [name] * len(df), m, strict=False)
        ]
    out = df[mask].copy()
    out["violation"] = [r.lstrip(";") for r in np.array(reasons)[mask]]
    return out


def daily_report(df: pd.DataFrame, trade_date: str | None = None) -> dict:
    """日检报告: OHLCV 违规 + 重复键 + 关键字段缺失 + 停牌占比.

    Returns:
        {'trade_date', 'n_rows', 'n_symbols', 'pass',
         'n_ohlcv_violations', 'n_duplicate_keys', 'n_missing_key_cols',
         'suspended_ratio', 'violations': DataFrame (明细, 不静默丢弃)}
    """
    vio = ohlcv_violations(df)
    dup = int(df.duplicated(subset=["symbol", "date"]).sum())
    missing = int(df[list(KEY_COLS)].isna().any(axis=1).sum())
    susp = (
        float(df["is_suspended"].astype(bool).mean())
        if "is_suspended" in df.columns
        else 0.0
    )
    passed = len(vio) == 0 and dup == 0 and missing == 0
    if not passed:
        logger.error(
            "数据质量日检不通过: OHLCV违规%d / 重复键%d / 缺失%d",
            len(vio),
            dup,
            missing,
        )
    return {
        "trade_date": trade_date or str(df["date"].max())[:10],
        "n_rows": len(df),
        "n_symbols": int(df["symbol"].nunique()),
        "pass": passed,
        "n_ohlcv_violations": len(vio),
        "n_duplicate_keys": dup,
        "n_missing_key_cols": missing,
        "suspended_ratio": round(susp, 4),
        "violations": vio,  # 违规明细 (显式处置依据, 不静默丢弃)
    }


def drop_violations(df: pd.DataFrame) -> pd.DataFrame:
    """显式剔除违规行 (留痕日志, 非静默). 返回干净面板."""
    vio = ohlcv_violations(df)
    if len(vio):
        logger.error(
            "剔除 OHLCV 违规行 %d 条 (留痕): %s",
            len(vio),
            vio[["symbol", "date", "violation"]].head(10).to_dict("records"),
        )
        df = df.drop(index=vio.index)
    return df
