"""面板 fina 列接公告管线缓存回归测试 (2026-09-02 事故).

背景: 面板第 6 节只做 ffill (面板历史停在 Q1 值), 公告管线每日把新鲜财报写进
data/supply_cache/alt_data/fina_indicator/ 却没人接进面板 — Q2 财报季
(2026-08-14→09-01) 4950 只股票只有 15 只 roe 值变化. 纯注入测试: 假缓存帧 /
临时目录假 parquet, 不碰真数据不触网.
"""

import pandas as pd
import pytest

from app.pipeline1.fina_cache_merge import (
    FINA_CACHE_COLS,
    load_fina_cache,
    overwrite_today_announce_date,
    overwrite_today_from_cache,
    replay_announce_date,
    replay_fina_asof,
    select_latest_fina,
)


def _cache_row(symbol, ann, period, **fina):
    row = {
        "symbol": symbol,
        "announce_date": pd.Timestamp(ann),
        "report_period": pd.Timestamp(period),
    }
    row.update(fina)
    return row


def _cache_frame(rows):
    return pd.DataFrame(rows)


# ── select_latest_fina: 每股最新一期选择 ────────────────────────────────────


def test_select_latest_prefers_new_period_then_new_announce():
    """回归: 多期记录必须取 report_period 最新者; 同股同期的更正公告
    (announce_date 更晚) 必须胜出 — 否则今日行永远停在 Q1 (事故根因)."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-04-30", "2026-03-31", roe=1.0),
            _cache_row("000001", "2026-08-28", "2026-06-30", roe=2.0),
            _cache_row("000002", "2026-08-20", "2026-06-30", roe=3.0),
            _cache_row("000002", "2026-08-25", "2026-06-30", roe=3.5),  # 更正公告
            _cache_row("000003", "2026-08-21", "2026-06-30", roe=4.0),
        ]
    )
    out = select_latest_fina(cache, fina_cols=["roe"])
    assert len(out) == 3
    by_sym = out.set_index("symbol")
    assert by_sym.loc["000001", "roe"] == 2.0
    assert by_sym.loc["000001", "report_period"] == pd.Timestamp("2026-06-30")
    assert by_sym.loc["000002", "roe"] == 3.5  # announce 更晚的更正
    assert by_sym.loc["000003", "roe"] == 4.0


def test_select_latest_drops_rows_without_period_and_filters_cols():
    """report_period 缺失的脏行不得入选; 输出列 = 请求列 ∩ 缓存实际列
    (缓存没有 net_margin → 不产出该列)."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-08-28", "2026-06-30", roe=2.0),
            {
                "symbol": "000009",
                "announce_date": pd.Timestamp("2026-08-28"),
                "report_period": pd.NaT,
                "roe": 9.0,
            },
        ]
    )
    out = select_latest_fina(cache, fina_cols=["roe", "net_margin"])
    assert list(out.columns) == ["symbol", "announce_date", "report_period", "roe"]
    assert "000009" not in set(out["symbol"])


def test_select_latest_empty_cache_returns_empty_frame():
    out = select_latest_fina(pd.DataFrame(), fina_cols=["roe"])
    assert len(out) == 0
    assert "roe" in out.columns  # 列结构保留, 行为空


# ── overwrite_today_from_cache: 今日行覆盖语义 ──────────────────────────────


def test_overwrite_replaces_stale_values():
    """回归: 今日行陈旧值必须被缓存值覆盖 (缓存是更新的真相, 不是 fillna 补缺)
    — 事故里 roe 停在 Q1 就是因为只有 fillna 语义的 ffill."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-08-28", "2026-06-30", roe=2.0, bps=10.0),
        ]
    )
    df = pd.DataFrame({"symbol": ["000001", "000002"], "roe": [1.0, 1.5]})
    applied = overwrite_today_from_cache(df, cache, fina_cols=["roe", "bps"])
    assert applied == ["roe", "bps"]
    assert df.loc[0, "roe"] == 2.0
    assert df.loc[0, "bps"] == 10.0
    # 缓存没有的股票 → NaN, 留给调用方其后保留的 ffill 兜底
    assert pd.isna(df.loc[1, "roe"])


def test_overwrite_skips_non_fina_columns():
    """缓存里的 announce_date/_ts_code 等列绝不参与覆盖 (sh_*/margin/announce_date
    各有语义, 属 ffill_cols 保护条目)."""
    cache = _cache_frame([_cache_row("000001", "2026-08-28", "2026-06-30", roe=2.0)])
    cache["_ts_code"] = "000001.SZ"
    df = pd.DataFrame({"symbol": ["000001"]})
    applied = overwrite_today_from_cache(df, cache, fina_cols=["roe", "_ts_code"])
    assert applied == ["roe"]
    assert "_ts_code" not in df.columns


def test_overwrite_noop_on_empty_cache():
    df = pd.DataFrame({"symbol": ["000001"]})
    assert overwrite_today_from_cache(df, pd.DataFrame()) == []
    assert "roe" not in df.columns


# ── load_fina_cache: 目录加载 + 去重 + 容错 ─────────────────────────────────


def test_load_fina_cache_reads_daily_snaps_dedups_and_ignores_other_prefixes(tmp_path):
    """回归: 跨文件快照重叠须全行去重; new_symbols_* 前缀文件 (建仓快照)
    天然被 all__* glob 排除; 损坏文件跳过不拖垮加载."""
    good1 = _cache_frame([_cache_row("000001", "2026-08-27", "2026-06-30", roe=2.0)])
    good2 = _cache_frame(
        [
            _cache_row("000001", "2026-08-27", "2026-06-30", roe=2.0),  # 跨文件重复
            _cache_row("000002", "2026-08-28", "2026-06-30", roe=3.0),
        ]
    )
    good1.to_parquet(tmp_path / "all__20260827_20260828.parquet", index=False)
    good2.to_parquet(tmp_path / "all__20260828_20260829.parquet", index=False)
    (tmp_path / "new_symbols_20260816_001140.parquet").touch()  # 非 all__* 前缀
    (tmp_path / "all__20260829_20260830.parquet").write_bytes(b"not parquet")

    out = load_fina_cache(str(tmp_path))
    assert len(out) == 2  # 重复行已去重, 损坏文件已跳过
    assert set(out["symbol"]) == {"000001", "000002"}


# ── replay_fina_asof: 面板存量修复的 PIT 回放 ───────────────────────────────


def test_replay_respects_each_stock_announce_date():
    """回归核心: 修复窗内每股只有自身公告日之后的行才拿到新值 — 一律覆盖
    每股最新值会让 08-28 公告的股票 08-10 行就有 Q2 值 (look-ahead)."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-08-20", "2026-06-30", roe=2.0),
            _cache_row("000002", "2026-08-28", "2026-06-30", roe=3.0),
        ]
    )
    rows = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000002", "000099"],
            "date": pd.to_datetime(
                [
                    "2026-08-15",
                    "2026-08-21",
                    "2026-08-25",
                    "2026-08-29",
                    "2026-08-21",
                ]
            ),
        }
    )
    out = replay_fina_asof(rows, cache, fina_cols=["roe"])
    assert pd.isna(out.loc[0, "roe"])  # 公告前 → NaN (调用方保留面板原值)
    assert out.loc[1, "roe"] == 2.0
    assert pd.isna(out.loc[2, "roe"])  # 000002 尚未公告
    assert out.loc[3, "roe"] == 3.0
    assert pd.isna(out.loc[4, "roe"])  # 缓存外股票


def test_replay_keeps_original_row_order_and_handles_correction():
    """行序任意 (面板行序本就不按日期) → 按"位置"还原; 更正公告 (同股二次公告)
    从第二次公告日起生效."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-08-20", "2026-06-30", roe=2.0),
            _cache_row("000001", "2026-08-25", "2026-06-30", roe=2.5),  # 更正
        ]
    )
    rows = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001"],
            "date": pd.to_datetime(["2026-08-26", "2026-08-15", "2026-08-21"]),
        }
    )
    out = replay_fina_asof(rows, cache, fina_cols=["roe"])
    assert out.loc[0, "roe"] == 2.5  # 更正后
    assert pd.isna(out.loc[1, "roe"])
    assert out.loc[2, "roe"] == 2.0  # 首公告后、更正前


# ── 保护条目: FINA_CACHE_COLS 永不触碰 ffill_cols 的非 fina 条目 ────────────


def test_fina_cache_cols_exclude_protected_ffill_entries():
    """回归: sh_* (刚修过) / margin (T+1 语义) / announce_date / sw_l*_name
    绝不从财务缓存覆盖 — dim31 用户暂缓, 动了会破坏既有语义."""
    banned = {
        "announce_date",
        "sh_change_vol",
        "sh_change_amt_total",
        "sh_net_change_sign",
        "sh_net_sign",
        "sw_l1_name",
        "sw_l2_name",
        "sw_l3_name",
        "margin_balance",
        "short_balance",
        "margin_buy_amt",
        "short_sell_vol",
    }
    assert not (set(FINA_CACHE_COLS) & banned)
    assert not any(c.startswith(("sh_", "margin")) for c in FINA_CACHE_COLS)


def test_replay_empty_inputs():
    """空缓存 → 行数与输入对齐, 无可回放列 (调用方按列存在性跳过替换)."""
    rows = pd.DataFrame({"symbol": ["000001"], "date": pd.to_datetime(["2026-08-20"])})
    out = replay_fina_asof(rows, pd.DataFrame(), fina_cols=["roe"])
    assert len(out) == 1
    assert list(out.columns) == []


@pytest.mark.parametrize(
    "sym,ann,period",
    [
        ("000001", "2026-08-20", "2026-06-30"),
    ],
)
def test_replay_single_row_smoke(sym, ann, period):
    cache = _cache_frame([_cache_row(sym, ann, period, roe=7.0)])
    rows = pd.DataFrame(
        {
            "symbol": [sym],
            "date": pd.to_datetime(["2026-08-21"]),
        }
    )
    out = replay_fina_asof(rows, cache, fina_cols=["roe"])
    assert out.loc[0, "roe"] == 7.0


# ── announce_date 车道: PIT 回放 + 今日覆盖 (2026-09-02 冻结修复) ─────────────


def test_replay_announce_date_pit_by_each_stock():
    """每行 = 截至该日已知最新一期报告的公告日 (merge_asof backward); 公告前
    → NaT (调用方保留面板原值). 一律覆盖每股最新公告日会让 08-25 公告的股票
    08-01 行就带新日期 (look-ahead)."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-04-30", "2026-03-31"),
            _cache_row("000002", "2026-07-10", "2026-03-31"),
            _cache_row("000002", "2026-08-25", "2026-06-30"),
        ]
    )
    rows = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000002", "000002", "000099"],
            "date": pd.to_datetime(
                ["2026-04-01", "2026-05-01", "2026-08-01", "2026-08-26", "2026-08-26"]
            ),
        }
    )
    out = replay_announce_date(rows, cache)
    assert out.dtype == "datetime64[ns]"
    assert pd.isna(out.loc[0])  # 000001 公告前
    assert out.loc[1] == pd.Timestamp("2026-04-30")
    assert out.loc[2] == pd.Timestamp("2026-07-10")  # 当时已知 Q1 公告日
    assert out.loc[3] == pd.Timestamp("2026-08-25")  # Q2 公告后推进
    assert pd.isna(out.loc[4])  # 缓存外股票


def test_replay_announce_date_preserves_row_order_and_index():
    cache = _cache_frame([_cache_row("000001", "2026-08-20", "2026-06-30")])
    rows = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001"],
            "date": pd.to_datetime(["2026-08-26", "2026-08-15", "2026-08-21"]),
        }
    )
    out = replay_announce_date(rows, cache)
    assert list(out.index) == list(rows.index)
    assert out.loc[0] == pd.Timestamp("2026-08-20")
    assert pd.isna(out.loc[1])
    assert out.loc[2] == pd.Timestamp("2026-08-20")


def test_replay_announce_date_empty_cache_all_nat():
    rows = pd.DataFrame(
        {"symbol": ["000001"], "date": pd.to_datetime(["2026-08-20"])}
    )
    out = replay_announce_date(rows, pd.DataFrame())
    assert len(out) == 1 and out.isna().all()


def test_overwrite_today_announce_date_uses_latest_known_record():
    """今日行 = 每股缓存最新记录的公告日 (PIT 安全: 缓存只含已公告记录);
    report_period 缺失的脏行不得入选; 缓存外股票 → NaT 留给 ffill 兜底."""
    cache = _cache_frame(
        [
            _cache_row("000001", "2026-04-30", "2026-03-31"),
            _cache_row("000001", "2026-08-28", "2026-06-30"),
            _cache_row("000002", "2026-08-20", "2026-06-30"),
            _cache_row("000002", "2026-08-27", pd.NaT),  # 脏行: 无报告期
        ]
    )
    df = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003"],
            "announce_date": pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]),
        }
    )
    n = overwrite_today_announce_date(df, cache)
    assert n == 2
    assert df.loc[0, "announce_date"] == pd.Timestamp("2026-08-28")
    assert df.loc[1, "announce_date"] == pd.Timestamp("2026-08-20")
    assert pd.isna(df.loc[2, "announce_date"])


def test_overwrite_today_announce_date_noop_on_empty_cache():
    df = pd.DataFrame({"symbol": ["000001"], "announce_date": [pd.NaT]})
    assert overwrite_today_announce_date(df, pd.DataFrame()) == 0
    assert pd.isna(df.loc[0, "announce_date"])
