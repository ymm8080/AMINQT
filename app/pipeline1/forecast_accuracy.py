# -*- coding: utf-8 -*-
"""预测准确度评估 (IMPLEMENTATION_PLAN_v3.2 P25.1-P25.2)
========================================================
v3.2 变更:
  - 废弃 WMAPE (分母爆炸), 改用 MAE + BIAS + direction_accuracy [D-23]
  - 新增 BIAS 分桶计算 [D-24]: 大涨/小涨/小跌/大跌
  - 保留旧 wmape() 标记 deprecated, 禁止新代码引用
  - 纪律: MAE/BIAS/方向准确率仅用于监控和校准执行参数,
    不得用于优化 LightGBM 目标函数 (会导致模型保守化, Rank IC 下降)

口径 (与训练标签一致, B9 验收权威):
  actual_k = label_pm_kd_net
  pred_k   = 清单 pred_ret_kd
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd

from .label_engine import LabelEngine

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 5)
ACCURACY_DIR = os.path.join("data", "forecast_accuracy")


# ---- deprecated ----
def wmape(actual, pred):
    """[DEPRECATED v3.2] 加权平均绝对百分比误差; 分母爆炸, 改用 MAE.
    保留仅向后兼容, 禁止新代码引用."""
    warnings.warn(
        "wmape is deprecated since v3.2, use compute_quality_metrics instead",
        DeprecationWarning,
        stacklevel=2,
    )
    denom = float(actual.abs().sum())
    if denom < 1e-12:
        return float("nan")
    return float((actual - pred).abs().sum() / denom)


def bias(actual, pred):
    """预测系统性偏差: mean(pred - actual), 正 = 高估."""
    return float((pred - actual).mean())


# ---- P25.1 核心指标 (D-23) ----
def compute_quality_metrics(actual, predicted):
    """D-23: 预测质量三维指标 (替代 WMAPE).

    Args:
        actual: np.ndarray 实际值
        predicted: np.ndarray 预测值

    Returns:
        {'mae_1d', 'bias_1d', 'direction_accuracy', 'n_samples'}
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    errors = predicted - actual
    n = len(actual)
    if n == 0:
        return {
            "mae_1d": float("nan"),
            "bias_1d": float("nan"),
            "direction_accuracy": float("nan"),
            "n_samples": 0,
        }
    dir_acc = np.mean(np.sign(predicted) == np.sign(actual))
    return {
        "mae_1d": float(np.mean(np.abs(errors))),
        "bias_1d": float(np.mean(errors)),
        "direction_accuracy": float(dir_acc),
        "n_samples": n,
    }


# ---- P25.2 BIAS 分桶 (D-24) ----
def compute_bias_buckets(actual, predicted):
    """D-24: BIAS分桶 — 大涨/小涨/小跌/大跌.
    整体BIAS可能为0, 但分桶BIAS揭示致命偏差.

    Returns:
        {'bias_big_up', 'bias_small_up', 'bias_small_down', 'bias_big_down'}
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    buckets = {
        "big_up": actual > 0.03,
        "small_up": (actual > 0) & (actual <= 0.03),
        "small_down": (actual < 0) & (actual >= -0.03),
        "big_down": actual < -0.03,
    }
    result = {}
    for name, mask in buckets.items():
        if mask.sum() > 10:
            result[f"bias_{name}"] = float(np.mean(predicted[mask] - actual[mask]))
        else:
            result[f"bias_{name}"] = float("nan")
    return result


# ---- 向后兼容的 score_forecast ----
def score_forecast(forecast_df, labeled_panel, forecast_date, horizons=HORIZONS):
    """对一期预测打分 (v3.2 扩展: 含 MAE/方向准确率).

    Args:
        forecast_df: 往期清单 (含 symbol/pred_ret_1d/3d/5d)
        labeled_panel: 含 label_pm_{k}d_net 列的面板
        forecast_date: 'YYYYMMDD'

    Returns:
        {'forecast_date', 'horizons': {k: {'mae','bias','direction_accuracy','n'}},
         'mature': bool, 'detail': DataFrame}
    """
    fdate = pd.to_datetime(forecast_date)
    rows = labeled_panel[labeled_panel["date"] == fdate]
    detail = forecast_df[["symbol"]].copy()
    for k in horizons:
        actual_col = f"label_pm_{k}d_net"
        if actual_col not in rows.columns:
            actual_col = f"label_{k}d_net"
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
        if len(sub):
            act, pred = sub[f"actual_{k}d"].values, sub[f"pred_{k}d"].values
            q = compute_quality_metrics(act, pred)
            buckets = compute_bias_buckets(act, pred)
        else:
            q = {
                "mae_1d": float("nan"),
                "bias_1d": float("nan"),
                "direction_accuracy": float("nan"),
                "n_samples": 0,
            }
            buckets = {
                "bias_big_up": float("nan"),
                "bias_small_up": float("nan"),
                "bias_small_down": float("nan"),
                "bias_big_down": float("nan"),
            }
        out["horizons"][k] = {**q, **buckets}
    out["mature"] = mature
    return out


def _labeled(panel, symbols):
    sub = panel[panel["symbol"].isin(symbols)]
    return LabelEngine.build_labels(sub)


def score_matured_forecasts(list_dir, panel, out_dir=ACCURACY_DIR):
    """扫描 list_dir 往期清单, 对 5d 视野成熟且未打分的打分 (幂等)."""
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
            continue
        forecast_df = pd.read_parquet(os.path.join(list_dir, fname))
        labeled = _labeled(panel, set(forecast_df["symbol"]))
        result = score_forecast(forecast_df, labeled, fdate)
        if not result["mature"]:
            logger.info("预测 %s 5d 视野未成熟, 下次再评", fdate)
            continue
        detail = result.pop("detail")
        try:
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
            detail.to_parquet(
                os.path.join(out_dir, f"detail_{fdate}.parquet"), index=False
            )
        except Exception:
            logger.warning("准确度报告写入失败: %s (非阻塞)", fdate, exc_info=True)
            continue
        h1 = result["horizons"][1]
        logger.info(
            "预测 %s 准确度: MAE(1d)=%.4f bias(1d)=%+.4f dir_acc=%.2f n=%d",
            fdate,
            h1["mae_1d"],
            h1["bias_1d"],
            h1.get("direction_accuracy", float("nan")),
            h1["n_samples"],
        )
        scored.append(result)
    return scored
