# -*- coding: utf-8 -*-
"""数据集划分与防过拟合 (P10.7, ARCH §5.12).

Train/Val/Test/OOS 四段严格按时间切分 + Purged K-Fold (gap=5 日)。
严禁数据泄露: 任何划分不得让训练集包含验证/测试期之后的信息。
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 默认切分边界 (ARCH §5.12.1: train<2023 / val=2023 / test=2024 / oos>=2025)
_DEFAULT_SPLIT = {
    "train_start": "2018-01-01",
    "train_end": "2022-12-31",
    "val_start": "2023-01-01",
    "val_end": "2023-12-31",
    "test_start": "2024-01-01",
    "test_end": "2024-12-31",
    "oos_start": "2025-01-01",
    "purge_gap_days": 5,
}


class DatasetSplitter:
    """时序数据集划分器."""

    def __init__(self, config: dict = None) -> None:
        """加载切分配置 (training_config.yaml: data_split 段).

        含 train/val/test/oos 时间段, purge_gap_days: 5。
        缺省使用 ARCH §5.12.1 默认边界。

        Args:
            config: data_split 配置字典 (可为 None)。
        """
        merged = dict(_DEFAULT_SPLIT)
        if config:
            merged.update({k: v for k, v in config.items() if v is not None})
        self.config = merged
        self._bounds = {
            key: pd.Timestamp(self.config[key])
            for key in (
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "test_start",
                "test_end",
                "oos_start",
            )
        }

    def split_by_time(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Dict[str, pd.DataFrame]:
        """四段时间切分.

        Args:
            df: 含 date 列的特征数据 (已按时间排序)。
            date_col: 日期列名 (默认 "date")。

        Returns:
            {"train": ..., "val": ..., "test": ..., "oos": ...}。

        Raises:
            KeyError: 缺少日期列。
        """
        if date_col not in df.columns:
            raise KeyError(f"缺少日期列: {date_col}")
        dates = pd.to_datetime(df[date_col])
        b = self._bounds
        segments = {
            "train": df[(dates >= b["train_start"]) & (dates <= b["train_end"])],
            "val": df[(dates >= b["val_start"]) & (dates <= b["val_end"])],
            "test": df[(dates >= b["test_start"]) & (dates <= b["test_end"])],
            "oos": df[dates >= b["oos_start"]],
        }
        result = {name: seg.reset_index(drop=True) for name, seg in segments.items()}
        logger.info(
            "时间切分: train=%d, val=%d, test=%d, oos=%d",
            *(len(result[k]) for k in ("train", "val", "test", "oos")),
        )
        return result

    def purged_kfold(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        gap_days: int = 5,
        date_col: str = "date",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Purged K-Fold: 每折训练/验证索引, 验证集前后各剔除 gap_days.

        数据按时间排序后均分为 n_splits 个连续段, 第 i 折以第 i 段为
        验证集; 训练集 = 其余样本中剔除验证窗口前后各 gap_days 个样本
        (防标签重叠泄露, ARCH §5.12.3.A.2)。

        Args:
            df: 时序数据 (含 date 列则先按日期排序, 否则按行序)。
            n_splits: 折数。
            gap_days: 隔离天数 (默认 5, 防标签重叠泄露)。
            date_col: 日期列名 (默认 "date")。

        Returns:
            [(train_idx, val_idx), ...], 索引为排序后数据的位置索引
            (升序排列的整数数组, 对应排序后 df 的 iloc 位置)。
        """
        if date_col in df.columns:
            ordered = (
                df.assign(**{date_col: pd.to_datetime(df[date_col])})
                .sort_values(date_col)
                .reset_index(drop=True)
            )
        else:
            ordered = df.reset_index(drop=True)
        n = len(ordered)
        if n == 0 or n_splits < 1:
            return []
        n_splits = min(n_splits, n)
        # 均分为 n_splits 个连续段 (前余数段各多 1 个样本)
        fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
        fold_sizes[: n % n_splits] += 1
        bounds = np.concatenate([[0], np.cumsum(fold_sizes)])

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_splits):
            val_start, val_end = int(bounds[i]), int(bounds[i + 1])
            val_idx = np.arange(val_start, val_end, dtype=int)
            # 剔除验证窗口前后各 gap_days 个样本 (purge)
            purge_lo = max(0, val_start - gap_days)
            purge_hi = min(n, val_end + gap_days)
            train_idx = np.concatenate(
                [
                    np.arange(0, purge_lo, dtype=int),
                    np.arange(purge_hi, n, dtype=int),
                ]
            )
            folds.append((train_idx, val_idx))
        logger.info(
            "Purged K-Fold: n=%d, n_splits=%d, gap_days=%d", n, n_splits, gap_days
        )
        return folds
