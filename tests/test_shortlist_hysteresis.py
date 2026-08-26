"""短名单迟滞滞留 (2026-08-26 用户定案 "清单加迟滞降换手") 单元测试.

规则 (scripts/_shortlist_t5_t10.hysteresis_keep + config SHORTLIST_HYSTERESIS):
- 昨日上榜股今日跌出 TOP-10 但仍在板内前 band_factor×10 名 → keep_flag="滞留"
- 每板块最多 max_keep 只; 超带/无历史文件/开关关 → 不滞留
- 不改新选股: 原 TOP-10 行全部保留且 keep_flag 为空
"""

import pandas as pd
import pytest

from scripts import _shortlist_t5_t10 as mod


def _cands(n=25, board="main", seed_mag=0.10):
    """n 只候选: pred_mag_10d 从 seed_mag 递减 → 000001 恒第一, 第 n 只恒第 n."""
    return pd.DataFrame(
        {
            "symbol": [f"{i:06d}" for i in range(1, n + 1)],
            "board": board,
            "pred_mag_10d": [round(seed_mag - 0.002 * i, 6) for i in range(n)],
            "pred_prob": [0.5] * n,
        }
    )


def _yesterday_file(tmp_path, syms, board="main", stamp="20260825"):
    df = pd.DataFrame({"symbol": syms, "board": board})
    df.to_csv(tmp_path / f"parallel_shortlist_{stamp}__{stamp}.csv", index=False)


def _run(tmp_path, full, trade_date="20260826"):
    truncated = mod.rank_and_truncate(full)
    return mod.hysteresis_keep(truncated, full, trade_date, hist_dir=str(tmp_path))


def test_keep_yesterday_pick_within_band(tmp_path):
    """昨日上榜股今日第 12 名 (≤ 2×10 带) → 滞留, 原 TOP-10 不动."""
    _yesterday_file(tmp_path, ["000012"])
    out = _run(tmp_path, _cands())
    kept = out[out["keep_flag"] == "滞留"]
    assert list(kept["symbol"]) == ["000012"]
    assert len(out) == 11
    assert (out[out["keep_flag"] == ""]["keep_flag"] == "").all()
    assert set(out[out["keep_flag"] == ""]["symbol"]) == {f"{i:06d}" for i in range(1, 11)}


def test_drop_yesterday_pick_beyond_band(tmp_path):
    """昨日上榜股今日第 22 名 (> 2×10 带) → 不滞留, 清单仍 10 只."""
    _yesterday_file(tmp_path, ["000022"])
    out = _run(tmp_path, _cands())
    assert len(out) == 10
    assert (out["keep_flag"] == "").all()


def test_max_keep_cap(tmp_path):
    """5 只昨日股在带内但 max_keep=3 → 只滞留排名最靠前的 3 只."""
    _yesterday_file(tmp_path, ["000011", "000012", "000013", "000014", "000015"])
    out = _run(tmp_path, _cands())
    kept = sorted(out.loc[out["keep_flag"] == "滞留", "symbol"])
    assert kept == ["000011", "000012", "000013"]
    assert len(out) == 13


def test_no_history_file_noop(tmp_path):
    """无昨日文件 → 原样返回, keep_flag 全空."""
    out = _run(tmp_path, _cands())
    assert len(out) == 10
    assert (out["keep_flag"] == "").all()


def test_disabled_noop(tmp_path, monkeypatch):
    """开关关 → 不滞留 (即使带内有昨日股)."""
    _yesterday_file(tmp_path, ["000012"])
    monkeypatch.setattr(mod, "SHORTLIST_HYSTERESIS", {"enable": False, "band_factor": 2.0, "max_keep": 3})
    out = _run(tmp_path, _cands())
    assert len(out) == 10
    assert (out["keep_flag"] == "").all()


def test_today_pick_not_duplicated(tmp_path):
    """昨日股今日仍在 TOP-10 → 正常入选, 无重复行无滞留标."""
    _yesterday_file(tmp_path, ["000001"])
    out = _run(tmp_path, _cands())
    assert (out["symbol"] == "000001").sum() == 1
    assert (out["keep_flag"] == "").all()


def test_kept_row_rank_blend_normalized(tmp_path):
    """滞留行 rank_blend = mag×prob (有 prob 时), 与入选行输出约定一致."""
    _yesterday_file(tmp_path, ["000012"])
    out = _run(tmp_path, _cands())
    row = out[out["symbol"] == "000012"].iloc[0]
    full = _cands().set_index("symbol")
    assert row["rank_blend"] == pytest.approx(full.loc["000012", "pred_mag_10d"] * 0.5)
