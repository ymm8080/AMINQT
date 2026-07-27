# -*- coding: utf-8 -*-
"""预测准确度评估 (每次预测运行时对成熟往期预测打分)
=====================================================
口径 (与训练标签一致, B9 验收权威):
  actual_k = label_pm_kd_net (T 日清单 → T+1 14:55≈收盘执行 → T+1+k 收盘, 净收益)
  pred_k   = 清单 pred_ret_kd (模型训练目标同口径)

指标 (按 horizon k=1/3/5 分别计算):
  WMAPE_k = Σ|actual_k - pred_k| / Σ|actual_k|   (分母为 0 → NaN)
  bias_k  = mean(pred_k - actual_k)              (正 = 系统性高估)

每次 run_prediction 生成当日清单后, 对 list_dir 中 5d 视野已成熟的往期清单
打分并持久化 (WORM): data/forecast_accuracy/accuracy_<date>.json (汇总) +
detail_<date>.parquet (逐股误差). 已打分的跳过 (幂等).
"""

from __future__ import annotations

import json
import logging
import os
import re

import pandas as pd

from .label_engine import LabelEngine

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 5)
ACCURACY_DIR = os.path.join("data", "forecast_accuracy")


def wmape(actual: pd.Series, pred: pd.Series) -> float:
    """加权平均绝对百分比误差; 分母 Σ|actual| 为 0 时返回 NaN (除零防护)."""
    denom = float(actual.abs().sum())
    if denom < 1e-12:
        return float("nan")
    return float((actual - pred).abs().sum() / denom)


def bias(actual: pd.Series, pred: pd.Series) -> float:
    """预测系统性偏差: mean(pred - actual), 正 = 高估."""
    return float((pred - actual).mean())


def score_forecast(
    forecast_df: pd.DataFrame,
    labeled_panel: pd.DataFrame,
    forecast_date: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    """对一期预测打分.

    Args:
        forecast_df: 往期清单 (含 symbol/pred_ret_1d/3d/5d)
        labeled_panel: 含 label_pm_{k}d_net 列的面板 (LabelEngine.build_labels 输出),
                       需覆盖 forecast_date 及其后 k+2 个交易日
        forecast_date: 'YYYYMMDD'

    Returns:
        {'forecast_date', 'horizons': {k: {'wmape','bias','n'}},
         'mature': bool (5d 成熟才为 True), 'detail': DataFrame(逐股)}
    """
    fdate = pd.to_datetime(forecast_date)
    rows = labeled_panel[labeled_panel["date"] == fdate]
    detail = forecast_df[["symbol"]].copy()
    for k in horizons:
        actual_col = f"label_pm_{k}d_net"
        if actual_col not in rows.columns:
            actual_col = f"label_{k}d_net"  # 兼容无 PM 口径的面板
        actuals = (
            rows.set_index("symbol")[actual_col]
            if len(rows)
            else pd.Series(dtype=float)
        )
        detail[f"actual_{k}d"] = detail["symbol"].map(actuals)
        detail[f"pred_{k}d"] = forecast_df[f"pred_ret_{k}d"].values

    out = {"forecast_date": forecast_date, "horizons": {}, "detail": detail}
    mature = True
    for k in horizons:
        sub = detail.dropna(subset=[f"actual_{k}d", f"pred_{k}d"])
        if k == max(horizons) and len(sub) == 0:
            mature = False
        out["horizons"][k] = {
            "wmape": wmape(sub[f"actual_{k}d"], sub[f"pred_{k}d"])
            if len(sub)
            else float("nan"),
            "bias": bias(sub[f"actual_{k}d"], sub[f"pred_{k}d"])
            if len(sub)
            else float("nan"),
            "n": int(len(sub)),
        }
    out["mature"] = mature
    return out


def _labeled(panel: pd.DataFrame, symbols: set) -> pd.DataFrame:
    """对面板子集计算验收标签 (PM 净口径)."""
    sub = panel[panel["symbol"].isin(symbols)]
    return LabelEngine.build_labels(sub)


def score_matured_forecasts(
    list_dir: str,
    panel: pd.DataFrame,
    out_dir: str = ACCURACY_DIR,
) -> list[dict]:
    """扫描 list_dir 往期清单, 对 5d 视野成熟且未打分的打分 (幂等).

    Returns:
        本次新打分的汇总列表 (已打分/未成熟的跳过)
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(list_dir):
        return []
    scored = []
    for fname in sorted(os.listdir(list_dir)):
        m = re.fullmatch(r"list_(\d{8})\.parquet", fname)
        if not m:
            continue
        fdate = m.group(1)
        summary_path = os.path.join(out_dir, f"accuracy_{fdate}.json")
        if os.path.exists(summary_path):
            continue  # 幂等: 已打分
        forecast_df = pd.read_parquet(os.path.join(list_dir, fname))
        labeled = _labeled(panel, set(forecast_df["symbol"]))
        result = score_forecast(forecast_df, labeled, fdate)
        if not result["mature"]:
            logger.info("预测 %s 5d 视野未成熟, 下次再评", fdate)
            continue
        detail = result.pop("detail")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        detail.to_parquet(os.path.join(out_dir, f"detail_{fdate}.parquet"), index=False)
        h1 = result["horizons"][1]
        logger.info(
            "预测 %s 准确度: WMAPE(1d)=%.4f bias(1d)=%+.4f n=%d",
            fdate,
            h1["wmape"],
            h1["bias"],
            h1["n"],
        )
        scored.append(result)
    return scored
