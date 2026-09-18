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


# ── insert_bigdrop_column (2026-09-16 用户令: 不另开页, 记进既有表的列) ──


def _patch_bigdrop(monkeypatch, scan):
    """模块运行恒成功 + 扫描结果注入 (不动源文件, 不起真模型)。"""
    monkeypatch.setattr("scripts._genious_excel._bigdrop_module_run", lambda: True)
    monkeypatch.setattr("scripts._genious_excel._bigdrop_scan", scan)


def test_insert_bigdrop_column_is_first_column_and_zfills(monkeypatch):
    """列必须**最前** (用户 0915 令: 追加到末尾会被列宽/横向滚动吞掉);

    送扫集 = 去重 + zfill(6) + 排序; 行序原样不动 (标注列不重排交付表)。
    """
    df = pd.DataFrame({"symbol": ["600000", "1"], "x": ["a", "b"]})
    seen = {}

    def fake_scan(syms):
        seen["syms"] = syms
        return {"600000": "大跌风险 3.2x", "000001": "波动风险"}

    _patch_bigdrop(monkeypatch, fake_scan)

    assert sc.insert_bigdrop_column([("LEGACY", df)]) == 1
    assert seen["syms"] == ["000001", "600000"]
    assert df.columns[0] == "BIGDROP SCAN"
    assert df["BIGDROP SCAN"].tolist() == ["大跌风险 3.2x", "波动风险"]
    assert df["symbol"].tolist() == ["600000", "1"]  # 行序原样


def test_insert_bigdrop_column_matches_suffixed_symbol(monkeypatch):
    """清单侧带 `.BJ` 的票必须**查得到**自己的读数 (去后缀后再查表)。

    不归一时它查空 → 显示成"两边没举手", 而真值是有标注的 —— 静默丢分, 且丢得
    看不出来。实测 legacy/parallel 的历史清单里确有此形态。
    """
    df = pd.DataFrame({"symbol": ["920367.BJ", "600000"], "x": ["a", "b"]})
    seen = {}

    def fake_scan(syms):
        seen["syms"] = syms
        return {"920367": "波动风险", "600000": "无风险"}

    _patch_bigdrop(monkeypatch, fake_scan)

    assert sc.insert_bigdrop_column([("LEGACY", df)]) == 1
    assert seen["syms"] == ["600000", "920367"]  # 送扫也是归一后的键
    assert df["BIGDROP SCAN"].tolist() == ["波动风险", "无风险"]


def test_insert_bigdrop_column_covers_every_symbol_sheet(monkeypatch):
    """送扫 = **所有**表的并集 (含 SLOW_BULL); 没 symbol 的表不加列也不进送扫。

    只送一部分的话, 没送到的票显示成空 —— 与"两边没举手"长得一模一样。
    """
    sheets = [
        ("LEGACY", pd.DataFrame({"symbol": ["600000"]})),
        ("SLOW_BULL", pd.DataFrame({"symbol": ["300001"]})),
        ("市场FADE预测", pd.DataFrame({"指标": ["P(冲高)"], "值": ["50%"]})),
    ]
    seen = {}

    def fake_scan(syms):
        seen["syms"] = syms
        return dict.fromkeys(syms, "无风险")

    _patch_bigdrop(monkeypatch, fake_scan)

    assert sc.insert_bigdrop_column(sheets) == 2  # FADE 无 symbol, 不计数
    assert seen["syms"] == ["300001", "600000"]  # SLOW_BULL 也在送扫范围
    assert "BIGDROP SCAN" in sheets[0][1].columns
    assert "BIGDROP SCAN" in sheets[1][1].columns
    assert "BIGDROP SCAN" not in sheets[2][1].columns


def test_insert_bigdrop_column_failopen_omits_column(monkeypatch):
    """扫描拿不到 (缺包/失败) → 整列不加, 返回 0, 不抛 (旁路标注契约)。"""
    df = pd.DataFrame({"symbol": ["600000"]})
    _patch_bigdrop(monkeypatch, lambda syms: None)

    assert sc.insert_bigdrop_column([("LEGACY", df)]) == 0
    assert "BIGDROP SCAN" not in df.columns


def test_insert_bigdrop_column_leaves_no_blank_for_scored_stocks(monkeypatch):
    """送扫的票**不留空** (用户 0916 令: 无风险要有标志, 所有 STOCK 都该有标志)。

    scan 故意漏掉 300002 (模拟不在面板): 它标 未评分, **不是空** —— 空格与"查过且
    无风险"在表上同形, 会把没测的票静默读成安全。
    """
    df = pd.DataFrame({"symbol": ["600000", "600001", "300002"]})
    _patch_bigdrop(
        monkeypatch, lambda syms: {"600000": "大跌风险 3.0x", "600001": "无风险"}
    )

    assert sc.insert_bigdrop_column([("LEGACY", df)]) == 1
    assert df["BIGDROP SCAN"].tolist() == ["大跌风险 3.0x", "无风险", "未评分"]


def test_insert_bigdrop_column_no_targets(monkeypatch):
    """一张带 symbol 的表都没有 → 不扫不抛。"""
    _patch_bigdrop(monkeypatch, lambda syms: pytest.fail("不该被调用"))

    sheets = [("市场FADE预测", pd.DataFrame({"指标": ["x"]}))]
    assert sc.insert_bigdrop_column(sheets) == 0


def test_write_xlsx_bigdrop_column_first_every_sheet(tmp_path, monkeypatch):
    """端到端: 落盘的 xlsx 里每张有 symbol 的表第一列都是 BIGDROP SCAN。"""
    _legacy(tmp_path)
    _parallel(tmp_path)
    _patch_bigdrop(
        monkeypatch,
        lambda syms: {s: "大跌风险 2.0x" if s == "600000" else "无风险" for s in syms},
    )

    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path / "nope")
    sc.insert_bigdrop_column(sheets)
    xl = pd.ExcelFile(sc.write(sheets, DATE, list_dir=tmp_path))

    for name in ("多模块重叠", "LEGACY", "PARALLEL"):
        assert xl.parse(name, dtype=str).columns[0] == "BIGDROP SCAN", name
    leg = xl.parse("LEGACY", dtype=str)
    assert leg["symbol"].tolist() == ["600000", "600001"]  # 原列原样还在
    assert leg["BIGDROP SCAN"].fillna("").tolist() == ["大跌风险 2.0x", "无风险"]
