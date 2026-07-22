# -*- coding: utf-8 -*-
"""DatasetSplitter 测试 (P10.7, ARCH §5.12)."""

import numpy as np
import pandas as pd
import pytest

from app.core.dataset_splitter import DatasetSplitter


def _daily_df(start, periods, value=1.0):
    return pd.DataFrame({
        "date": pd.date_range(start, periods=periods, freq="D"),
        "feat": value,
    })


@pytest.fixture
def full_df():
    """2018-01-01 ~ 2025-12-31 日频数据."""
    return _daily_df("2018-01-01", 2922)  # 8 年


class TestSplitByTime:
    def test_segment_boundaries(self, full_df):
        splitter = DatasetSplitter()
        parts = splitter.split_by_time(full_df)
        assert parts["train"]["date"].max() == pd.Timestamp("2022-12-31")
        assert parts["val"]["date"].min() == pd.Timestamp("2023-01-01")
        assert parts["val"]["date"].max() == pd.Timestamp("2023-12-31")
        assert parts["test"]["date"].min() == pd.Timestamp("2024-01-01")
        assert parts["test"]["date"].max() == pd.Timestamp("2024-12-31")
        assert parts["oos"]["date"].min() == pd.Timestamp("2025-01-01")

    def test_no_date_overlap_between_segments(self, full_df):
        splitter = DatasetSplitter()
        parts = splitter.split_by_time(full_df)
        date_sets = {name: set(p["date"]) for name, p in parts.items()}
        names = ["train", "val", "test", "oos"]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                assert not (date_sets[names[i]] & date_sets[names[j]]), \
                    f"{names[i]} 与 {names[j]} 日期重叠"

    def test_total_rows_preserved(self, full_df):
        splitter = DatasetSplitter()
        parts = splitter.split_by_time(full_df)
        total = sum(len(p) for p in parts.values())
        assert total == len(full_df)

    def test_configurable_boundaries(self):
        df = _daily_df("2020-01-01", 730)  # 2 年
        splitter = DatasetSplitter({
            "train_start": "2020-01-01", "train_end": "2020-12-31",
            "val_start": "2021-01-01", "val_end": "2021-06-30",
            "test_start": "2021-07-01", "test_end": "2021-09-30",
            "oos_start": "2021-10-01",
        })
        parts = splitter.split_by_time(df)
        assert parts["train"]["date"].max() == pd.Timestamp("2020-12-31")
        assert len(parts["val"]) == 181
        assert len(parts["oos"]) == 91  # 2021-10-01 ~ 2021-12-30

    def test_missing_date_col_raises(self):
        splitter = DatasetSplitter()
        with pytest.raises(KeyError):
            splitter.split_by_time(pd.DataFrame({"feat": [1, 2, 3]}))

    def test_string_dates_accepted(self):
        df = pd.DataFrame({
            "date": ["2022-12-31", "2023-06-30", "2024-06-30", "2025-06-30"],
            "feat": [1, 2, 3, 4],
        })
        parts = DatasetSplitter().split_by_time(df)
        assert [len(parts[k]) for k in ("train", "val", "test", "oos")] \
            == [1, 1, 1, 1]


class TestPurgedKFold:
    @pytest.fixture
    def df(self):
        return _daily_df("2024-01-01", 100)

    def test_fold_count_and_val_coverage(self, df):
        splitter = DatasetSplitter()
        folds = splitter.purged_kfold(df, n_splits=5, gap_days=5)
        assert len(folds) == 5
        all_val = np.concatenate([v for _, v in folds])
        assert sorted(all_val.tolist()) == list(range(100))  # 验证集全覆盖无重叠

    def test_no_train_val_overlap(self, df):
        splitter = DatasetSplitter()
        for train_idx, val_idx in splitter.purged_kfold(df, n_splits=5,
                                                        gap_days=5):
            assert not set(train_idx.tolist()) & set(val_idx.tolist())

    def test_purge_gap_excludes_boundary_samples(self, df):
        splitter = DatasetSplitter()
        gap = 5
        folds = splitter.purged_kfold(df, n_splits=5, gap_days=gap)
        # 第 2 折 (i=1): 验证集 [20, 40), 训练集不得含 [15, 45)
        train_idx, val_idx = folds[1]
        assert val_idx.tolist() == list(range(20, 40))
        train_set = set(train_idx.tolist())
        for pos in range(15, 45):
            assert pos not in train_set, f"purge 边界样本 {pos} 泄漏进训练集"
        # purge 区之外的样本保留
        assert 14 in train_set and 45 in train_set

    def test_first_fold_purges_only_after(self, df):
        splitter = DatasetSplitter()
        train_idx, val_idx = splitter.purged_kfold(df, n_splits=5,
                                                   gap_days=5)[0]
        assert val_idx.tolist() == list(range(0, 20))
        assert train_idx.tolist() == list(range(25, 100))

    def test_last_fold_purges_only_before(self, df):
        splitter = DatasetSplitter()
        train_idx, val_idx = splitter.purged_kfold(df, n_splits=5,
                                                   gap_days=5)[-1]
        assert val_idx.tolist() == list(range(80, 100))
        assert train_idx.tolist() == list(range(0, 75))

    def test_uses_time_order_not_row_order(self):
        # 打乱行序: 索引仍按时间排序后的位置
        df = _daily_df("2024-01-01", 20).sample(frac=1.0, random_state=42)
        splitter = DatasetSplitter()
        folds = splitter.purged_kfold(df, n_splits=2, gap_days=2)
        train_idx, val_idx = folds[1]
        assert val_idx.tolist() == list(range(10, 20))
        assert train_idx.tolist() == list(range(0, 8))

    def test_gap_days_configurable(self, df):
        splitter = DatasetSplitter()
        _, val_idx = splitter.purged_kfold(df, n_splits=5, gap_days=0)[1]
        train_idx, _ = splitter.purged_kfold(df, n_splits=5, gap_days=0)[1]
        assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(100))

    def test_empty_df(self):
        splitter = DatasetSplitter()
        assert splitter.purged_kfold(pd.DataFrame({"date": []}), 5, 5) == []
