"""合并清单单文件 (scripts/_stocklist_combined.py) 单测: 四线源页序 +
多模块重叠页 + 缺源跳页 + WORM 写出。
"""

import pandas as pd
import pytest

from scripts import _stocklist_combined as sc

DATE = "20260105"


def _legacy(tmp_path):
    pd.DataFrame(
        {
            "symbol": ["600000", "600001"],
            "pred_ret_10d": ["10.0%", "8.0%"],
            "prob_up_10d": ["70%", "60%"],
        }
    ).to_csv(tmp_path / f"legacy_stocklist_{DATE}__M1.csv", index=False)


def _parallel(tmp_path):
    pd.DataFrame(
        {
            "symbol": ["600000", "601000"],
            "pred_mag_10d": ["7.0%", "5.0%"],
            "pred_prob_10d": ["51%", "52%"],
        }
    ).to_csv(tmp_path / f"parallel_shortlist_{DATE}__M1.csv", index=False)


def test_build_overlap_first_and_missing_sources_skipped(tmp_path):
    _legacy(tmp_path)
    _parallel(tmp_path)
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path / "nope")
    assert [n for n, _ in sheets] == ["多模块重叠", "LEGACY", "PARALLEL"]
    overlap = sheets[0][1]
    assert overlap["symbol"].tolist() == ["600000"]  # 唯一双模块票
    assert overlap["模块"].tolist() == ["LEGACY+PARALLEL"]
    row = overlap.iloc[0]
    assert row["LEGACY_预测10d"] == "10.0%"
    assert row["LEGACY_概率10d"] == "70%"
    assert row["PARALLEL_预测10d"] == "7.0%"
    assert row["PARALLEL_概率10d"] == "51%"
    # 单模块票不上重叠页
    assert set(overlap["symbol"]) == {"600000"}


def test_build_slowbull_from_shadow_dashdate_counts_overlap(tmp_path):
    _legacy(tmp_path)
    _parallel(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    pd.DataFrame({"symbol": ["600000", "300001"], "wr5": ["0.1", "0.2"]}).to_csv(
        shadow / "slowbull_list_2026-01-05__v4.csv", index=False
    )
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=shadow)
    assert [n for n, _ in sheets] == ["多模块重叠", "LEGACY", "PARALLEL", "SLOW_BULL"]
    assert sheets[0][1].iloc[0]["模块数"] == 3  # 600000 三模块居首


def test_build_no_core_sources_raises(tmp_path):
    with pytest.raises(SystemExit):
        sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path)


def test_market_fade_sheet_formats():
    fc = {
        "next_date": "2026-09-10",
        "state_date": "20260909",
        "close": 3951.51,
        "ret": 0.0028,
        "regime": "线上(0~+2%)",
        "above_ma20": 0.0038,
        "r5": 0.0026,
        "vratio": 1.01,
        "p_surge": 0.539,
        "p_fade_given_surge": 0.337,
        "p_fade_day": 0.182,
        "base_fade": 0.219,
        "verdict": "≈常态",
        "n_regime": 1409,
    }
    name, df = sc.market_fade_sheet(fc_fn=lambda: fc)
    assert name == "市场FADE预测"
    assert len(df) == 11
    got = dict(zip(df["指标"], df["值"]))
    assert got["P(回落日) 预测"] == "18%"
    assert got["预测交易日"] == "2026-09-10"
    assert "线上(0~+2%)" in got["行情带 (距MA20)"]


def test_market_fade_sheet_failopen():
    def boom():
        raise RuntimeError("refetch fail")

    assert sc.market_fade_sheet(fc_fn=boom) is None


def test_write_xlsx_roundtrip(tmp_path):
    _legacy(tmp_path)
    _parallel(tmp_path)
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path)
    fp = sc.write(sheets, DATE, list_dir=tmp_path)
    assert fp.name == f"stocklist_combined_{DATE}.xlsx"
    xl = pd.ExcelFile(fp)
    assert xl.sheet_names == ["多模块重叠", "LEGACY", "PARALLEL"]
    leg = xl.parse("LEGACY", dtype=str)
    assert leg["symbol"].tolist() == ["600000", "600001"]
    assert leg["pred_ret_10d"].tolist() == ["10.0%", "8.0%"]  # 显示层原样


def test_write_same_day_rerun_goes_to_variants_not_skip(tmp_path):
    """同日重跑必须**真落文件** (用户 2026-09-15 "SHOULD NOT SKIP")。

    旧行为 = 已存在就 print 跳过 + rc=0: 上游改了清单后重跑, 下游合成表仍是旧行
    却报成功 —— 无声失真。现行为 = 退到 __v2/__v3, 规范名首份不动 (WORM)。
    """
    _legacy(tmp_path)
    _parallel(tmp_path)
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path)
    got = [sc.write(sheets, DATE, list_dir=tmp_path).name for _ in range(3)]
    assert got == [
        f"stocklist_combined_{DATE}.xlsx",
        f"stocklist_combined_{DATE}__v2.xlsx",
        f"stocklist_combined_{DATE}__v3.xlsx",
    ]
    assert all((tmp_path / n).exists() for n in got)  # 三份都在, 无一被覆盖


def test_next_path_prefers_canonical_then_fills_first_gap(tmp_path):
    canonical = tmp_path / f"stocklist_combined_{DATE}.xlsx"
    assert sc._next_path(DATE, tmp_path) == canonical
    canonical.write_text("x", encoding="utf-8")
    assert sc._next_path(DATE, tmp_path).name == f"stocklist_combined_{DATE}__v2.xlsx"
    # 跨过已存在的号取下一个空位 (不重用 __v2)
    (tmp_path / f"stocklist_combined_{DATE}__v2.xlsx").write_text("x", encoding="utf-8")
    assert sc._next_path(DATE, tmp_path).name == f"stocklist_combined_{DATE}__v3.xlsx"
    # 别日互不影响
    assert (
        sc._next_path("20260106", tmp_path).name == "stocklist_combined_20260106.xlsx"
    )


# ── bigdrop_sheet (2026-09-16 用户令: combined 就绪后 bigdrop 必跑) ──


def test_bigdrop_sheet_failopen_on_missing_bundle(tmp_path, monkeypatch, capsys):
    """包缺失 → 扫描 None → 跳页返回 None, 不抛 (旁路页契约)."""
    import scripts.bigdrop_check as bc

    fake_dir = tmp_path / "bigdrop"
    fake_dir.mkdir()
    monkeypatch.setattr(bc, "BUNDLE_DIR", fake_dir)
    out = sc.bigdrop_sheet(["600000"])
    assert out is None
    assert out is None  # 跳页契约: 返回 None 不抛, 上游 log 已大声 (caplog 可断)


def test_bigdrop_sheet_dedupes_and_marks_directional(monkeypatch):
    """全池去重 + (大跌风险, 波动风险, 空) 三态语义 + 排序: 方向支在前、倍数大在前."""
    fake_scan = {
        "600000": "大跌风险 3.2x",
        "601000": "大跌风险 1.5x",
        "002001": "波动风险 2.0x",
        "002002": "",
    }
    monkeypatch.setattr("scripts._genious_excel._bigdrop_module_run", lambda: True)
    monkeypatch.setattr(
        "scripts._genious_excel._bigdrop_scan", lambda syms: dict(fake_scan)
    )

    import scripts.bigdrop_check as bc

    monkeypatch.setattr(
        bc,
        "load_bundle",
        lambda: {"oos_base": 0.03},
    )
    out = sc.bigdrop_sheet(["600000", "600000", "002001", "002002", "999999"])
    assert out is not None
    name, df = out
    assert name == "BIGDROP"
    # 送 5 去重 4 (600000 重复); 999999 无标注 → 空, 排最末 (合法: 全池只有 4 行? 不 —
    # 空也在列里, 垫底), 方向支先于波动支先于空
    assert df["symbol"].duplicated().sum() == 0
    assert df["symbol"].iloc[0] == "600000"  # 大跌风险 3.2x (倍数大在前)
    assert df["symbol"].iloc[1] == "002001"  # 波动风险
    assert df["symbol"].iloc[-1] == "999999"  # 空垫底
    cats = df["BIGDROP SCAN"].str[:4]
    assert cats.iloc[0] == "大跌风险"


def test_bigdrop_sheet_all_unmarked_skips_page(monkeypatch):
    """当日全池无任何标注 → 跳页 (合法空不是故障)."""
    monkeypatch.setattr("scripts._genious_excel._bigdrop_module_run", lambda: True)
    monkeypatch.setattr(
        "scripts._genious_excel._bigdrop_scan", lambda syms: {s: "" for s in syms}
    )
    import scripts.bigdrop_check as bc

    monkeypatch.setattr(bc, "load_bundle", lambda: {"oos_base": 0.03})
    assert sc.bigdrop_sheet(["600000"]) is None


def test_bigdrop_page_integration(tmp_path, monkeypatch):
    """端到端: build 出的池送 bigdrop → 页序 (重叠, FADE?, BIGDROP, ...)."""
    sc._SOURCES_DEFAULT = None  # 哨兵: 防误改全局 (测试全局常量污染陷阱)
    # 走 monkeypatch 的 bigdrop, 不动源文件
    _legacy(tmp_path)
    monkeypatch.setattr(
        "scripts._stocklist_combined.STOCK_LIST_DIR",
        tmp_path,
    )
    monkeypatch.setattr("scripts._genious_excel._bigdrop_module_run", lambda: True)
    monkeypatch.setattr(
        "scripts._genious_excel._bigdrop_scan",
        lambda syms: {s: "大跌风险 2.0x" for s in syms},
    )
    import scripts.bigdrop_check as bc

    monkeypatch.setattr(bc, "load_bundle", lambda: {"oos_base": 0.03})
    sheets = sc.build(DATE, list_dir=tmp_path)
    pool = [
        s
        for n, d in sheets
        if n in ("LEGACY", "PARALLEL", "密度")
        for s in d.get("symbol", [])
    ]
    bd = sc.bigdrop_sheet(pool)
    assert bd is not None and bd[0] == "BIGDROP"
