# -*- coding: utf-8 -*-
"""Tests for app/core/data_loader — Parquet load + canonicalize + OHLCV validation."""

import pandas as pd
import pytest

from app.core import data_loader
from config import settings


@pytest.fixture
def sample_parquet(tmp_path, monkeypatch):
    """Create a valid sample Parquet in a temporary raw dir."""
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.5, 10.5],
            "close": [10.5, 11.5],
            "volume": [1000, 2000],
            "amount": [10500, 23000],
        }
    )
    df.to_parquet(tmp_path / "000001.parquet", index=False)
    return tmp_path


def test_load_parquet_canonicalizes_and_validates(sample_parquet):
    df = data_loader.load_parquet("000001")
    assert list(df.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert len(df) == 2
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-01")


def test_load_all_skips_missing_symbols(sample_parquet, caplog):
    out = data_loader.load_all(["000001", "999999"])
    assert "000001" in out
    assert "999999" not in out
    assert "Missing Parquet" in caplog.text


def test_validate_ohlcv_rejects_invalid_high_low(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "open": [10.0],
            "high": [9.0],  # high < low
            "low": [10.5],
            "close": [10.5],
            "volume": [1000],
        }
    )
    df.to_parquet(tmp_path / "000002.parquet", index=False)
    with pytest.raises(ValueError, match="OHLCV validation failed"):
        data_loader.load_parquet("000002")


def test_validate_ohlcv_rejects_negative_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [-1],
        }
    )
    df.to_parquet(tmp_path / "000003.parquet", index=False)
    with pytest.raises(ValueError, match="OHLCV validation failed"):
        data_loader.load_parquet("000003")
