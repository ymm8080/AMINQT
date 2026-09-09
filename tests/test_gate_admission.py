"""闸准入协议单测 (2026-09-09): 合成票池验证判词 — 稳定正效果/半窗翻转/
样本不足/赢家误杀富集/全窗差不足, 以及 killed 规则与清单装载."""

import numpy as np
import pandas as pd

from scripts import _gate_admission as ga

CRIT = {
    "min_days": 40,
    "min_killed": 20,
    "min_edge_pp": 0.5,
    "half_tol_pp": 0.0,
    "winner_ret5": 0.10,
    "max_leak_enrich": 1.5,
}


def _mk(n_days=40, killed_ret5=-0.02, kept_ret5=0.01, back_killed_ret5=None,
        back_kept_ret5=None, leak_winner=False):
    """n_days 清单日 × 10 票/日; 前2票 killed。前后半可分别设收益."""
    days = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []
    half = n_days // 2
    for i, d in enumerate(days):
        back = i >= half and back_killed_ret5 is not None
        kr = back_killed_ret5 if back else killed_ret5
        pr = back_kept_ret5 if back else kept_ret5
        for j in range(10):
            killed = j < 2
            ret5 = kr if killed else pr
            # leak_winner: 每天 1 个大赢家且落在删除区
            if leak_winner and killed and j == 0:
                ret5 = 0.12
            rows.append({"list_date": d, "symbol": f"{i:03d}{j:02d}",
                         "killed": killed, "ret5": ret5, "o2c": ret5 / 2})
    return pd.DataFrame(rows)


def test_pass_stable_positive():
    rep = ga.evaluate_gate(_mk(), CRIT)
    assert rep["verdict"] == "PASS", rep["reasons"]
    assert rep["reasons"] == []
    assert rep["edge_full_pp"] > 2.0
    assert rep["edge_front_pp"] > 2.0 and rep["edge_back_pp"] > 2.0


def test_fail_half_flip():
    # fade 删线翻车模式: 前半有效 (删除区差), 后半翻转 (删除区反而更好)
    rep = ga.evaluate_gate(
        _mk(back_killed_ret5=0.03, back_kept_ret5=0.01), CRIT
    )
    assert rep["verdict"] == "FAIL"
    assert any("后半" in r and "半窗不稳" in r for r in rep["reasons"])
    assert rep["edge_front_pp"] > 0 > rep["edge_back_pp"]


def test_insufficient_thin_sample():
    rep = ga.evaluate_gate(_mk(n_days=10), CRIT)
    assert rep["verdict"] == "INSUFFICIENT"
    assert any("清单日 10 < 40" in r for r in rep["reasons"])
    # 诊断指标照给 (供人工判断)
    assert rep["edge_full_pp"] > 2.0


def test_fail_winner_leak_enrichment():
    # 大赢家全部落删除区: P(删除|赢家)=1.0, 删除率=0.2 → 富集 5.0x
    rep = ga.evaluate_gate(_mk(leak_winner=True), CRIT)
    assert rep["verdict"] == "FAIL"
    assert any("误杀富集" in r for r in rep["reasons"])
    assert rep["winner_leak"]["10%"]["enrich"] > 2.0


def test_fail_min_edge():
    # 双半窗同向但差太薄 (0.2pp < 0.5pp)
    rep = ga.evaluate_gate(_mk(killed_ret5=-0.001, kept_ret5=0.001), CRIT)
    assert rep["verdict"] == "FAIL"
    assert any("全窗留存−删除差" in r for r in rep["reasons"])


def test_apply_kill_rule_fade_threshold():
    recs = pd.DataFrame({
        "line": "legacy", "list_date": pd.Timestamp("2026-09-09"),
        "symbol": [f"{i:06d}" for i in range(4)],
        "factor": [0.9, 0.75, 0.749, np.nan],
    })
    out = ga.apply_kill_rule(recs, "fade")
    assert list(out["killed"]) == [True, True, False, False]  # ≥0.75 且 NaN 不删


def test_apply_kill_rule_amt_top_frac():
    recs = pd.DataFrame({
        "line": ["legacy"] * 10,
        "list_date": pd.Timestamp("2026-09-09"),
        "symbol": [f"{i:06d}" for i in range(10)],
        "factor": [0.01 * i for i in range(10)],  # top 20% = 最高2只
    })
    out = ga.apply_kill_rule(recs, "amt_agree")
    assert sorted(out.loc[out["killed"], "symbol"]) == ["000008", "000009"]


def test_load_lists_keep_last_and_dedupe(tmp_path):
    d = tmp_path
    (d / "legacy_stocklist_20260908__m__v1.csv").write_text("symbol\n000001\n000002\n")
    (d / "legacy_stocklist_20260908__m__v2.csv").write_text("symbol\n000001\n")  # keep-last
    (d / "parallel_shortlist_20260908__p.csv").write_text("symbol\n600000\n")
    (d / "legacy_stocklist_20260907__m.csv").write_text("symbol\n000001\n")
    (d / "unrelated.csv").write_text("symbol\n300000\n")
    L = ga.load_lists(list_dir=d)
    assert len(L) == 3
    v2 = L[(L["line"] == "legacy") & (L["list_date"] == pd.Timestamp("2026-09-08"))]
    assert list(v2["symbol"]) == ["000001"]  # v1 的 000002 被新版覆盖
    assert set(L["line"]) == {"legacy", "parallel"}
