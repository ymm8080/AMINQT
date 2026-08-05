# -*- coding: utf-8 -*-
"""bt_ rolling snapshot (dim33 upstream 4 columns) refresh tests.

The rolling snapshot is the restore source for the 4 V3 panel block-trade
columns after a panel rebuild. It accumulates from each daily fetch's live
aggregation (not the deduped raw cache), so it survives rebuilds.
"""

import pandas as pd

from app.pipeline1.bt_snapshot import BT_COLS, refresh_rolling_snapshot


def _event(symbol, date, count):
    return {
        "symbol": symbol,
        "date": pd.Timestamp(date),
        "bt_count": count,
        "bt_disc_raw": 0.05,
        "bt_inst_absorb": 0.3,
        "bt_amt_ratio_float_mv": 0.001,
    }


def test_first_run_seeds_history_then_appends_today(tmp_path):
    path = tmp_path / "bt_v3_snapshot_rolling.parquet"
    seed = pd.DataFrame(
        [
            _event("600519", "2026-08-01", 1),
            _event("300750", "2026-08-01", 3),
            _event("000001", "2026-08-02", 5),
        ]
    )
    today = pd.DataFrame(
        [_event("600519", "2026-08-03", 2), _event("300750", "2026-08-03", 1)]
    )
    res = refresh_rolling_snapshot(today, path, seed=seed)
    out = pd.read_parquet(path)

    assert res == {"appended": 2, "total": 5}
    assert len(out) == 5
    assert list(out.columns) == ["symbol", "date"] + BT_COLS
    assert (
        out[(out.symbol == "600519") & (out.date == pd.Timestamp("2026-08-03"))][
            "bt_count"
        ].iloc[0]
        == 2
    )


def test_rerun_is_idempotent_and_overlap_keeps_last(tmp_path):
    path = tmp_path / "bt_v3_snapshot_rolling.parquet"
    seed = pd.DataFrame([_event("600519", "2026-08-01", 1)])
    today = pd.DataFrame(
        [_event("600519", "2026-08-01", 9)]
    )  # same (symbol, date), new value

    refresh_rolling_snapshot(today, path, seed=seed)
    out1 = pd.read_parquet(path)
    # crash-safe re-run with the same input must not grow the file
    refresh_rolling_snapshot(today, path, seed=seed)
    out2 = pd.read_parquet(path)

    assert len(out1) == 1
    assert len(out2) == 1
    assert out2["bt_count"].iloc[0] == 9  # keep=last
    assert len(out2.drop_duplicates(["symbol", "date"])) == 1


def test_non_event_rows_are_ignored(tmp_path):
    path = tmp_path / "bt_v3_snapshot_rolling.parquet"
    today = pd.DataFrame(
        [
            _event("600519", "2026-08-03", 2),
            {
                "symbol": "300750",
                "date": pd.Timestamp("2026-08-03"),
                "bt_count": float("nan"),
                "bt_disc_raw": float("nan"),
                "bt_inst_absorb": float("nan"),
                "bt_amt_ratio_float_mv": float("nan"),
            },
        ]
    )
    res = refresh_rolling_snapshot(today, path)
    out = pd.read_parquet(path)
    assert res["appended"] == 1
    assert len(out) == 1
    assert out["symbol"].iloc[0] == "600519"


def test_first_run_with_no_events_creates_nothing(tmp_path):
    path = tmp_path / "bt_v3_snapshot_rolling.parquet"
    no_event = pd.DataFrame(
        [
            {
                "symbol": "600519",
                "date": pd.Timestamp("2026-08-03"),
                "bt_count": float("nan"),
                "bt_disc_raw": float("nan"),
                "bt_inst_absorb": float("nan"),
                "bt_amt_ratio_float_mv": float("nan"),
            }
        ]
    )
    res = refresh_rolling_snapshot(no_event, path)
    assert res == {"appended": 0, "total": 0}
    assert not path.exists()


def test_restore_after_panel_rebuild(tmp_path):
    path = tmp_path / "bt_v3_snapshot_rolling.parquet"
    history = pd.DataFrame(
        [
            _event("600519", "2026-08-01", 1),
            _event("600519", "2026-08-02", 4),
            _event("600519", "2026-08-03", 2),
        ]
    )
    refresh_rolling_snapshot(history, path)
    snap = pd.read_parquet(path)

    # rebuilt panel: same rows, bt_* columns dropped
    rebuilt = pd.DataFrame(
        {
            "symbol": ["600519"] * 4,
            "date": [
                pd.Timestamp("2026-08-01"),
                pd.Timestamp("2026-08-02"),
                pd.Timestamp("2026-08-03"),
                pd.Timestamp("2026-08-04"),
            ],
            "close": [10.0] * 4,
        }
    )
    restored = rebuilt.merge(snap, on=["symbol", "date"], how="left")
    assert restored["bt_count"].tolist()[:3] == [1, 4, 2]
    assert pd.isna(restored.loc[3, "bt_count"])  # non-event day stays NaN
