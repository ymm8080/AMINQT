# -*- coding: utf-8 -*-
"""app.pipeline1.backup 单元测试.

验证目标 (意图):
- keeper 文件按 <stem>__<trade_date>.parquet 复制到备份目录
- glob keeper 取 mtime 最新的源文件
- 缺失的 keeper 跳过且不报错
- WORM: 同日重跑不覆盖已有备份
- retention: 只保留最新 N 份, 清理最旧
"""

import os

from app.pipeline1.backup import backup_keepers


def _mk(path, content=b"x", mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_static_keeper_copied_with_date_suffix(tmp_path):
    _mk(tmp_path / "data/cyq_panel.parquet")
    out = backup_keepers(
        tmp_path, tmp_path / "bk", ["data/cyq_panel.parquet"], "20260731"
    )
    assert (tmp_path / "bk/cyq_panel__20260731.parquet").exists()
    assert out["data/cyq_panel.parquet"].startswith("ok")


def test_wildcard_keeper_picks_newest_source(tmp_path):
    _mk(
        tmp_path / "data/factor_registry/features_main_20260730T010000.parquet",
        mtime=1000,
    )
    _mk(
        tmp_path / "data/factor_registry/features_main_20260731T075252.parquet",
        mtime=2000,
    )
    backup_keepers(
        tmp_path,
        tmp_path / "bk",
        ["data/factor_registry/features_main_*.parquet"],
        "20260731",
    )
    assert (tmp_path / "bk/features_main_20260731T075252__20260731.parquet").exists()
    assert not (
        tmp_path / "bk/features_main_20260730T010000__20260731.parquet"
    ).exists()


def test_absolute_keeper_outside_root(tmp_path):
    # 仓库外主数据 (如 D:/AMINQT/PARQUET/) 用绝对路径 keeper 备份
    ext = tmp_path / "external" / "panel_full_enriched_v3.parquet"
    _mk(ext)
    out = backup_keepers(tmp_path, tmp_path / "bk", [str(ext)], "20260803")
    assert (tmp_path / "bk/panel_full_enriched_v3__20260803.parquet").exists()
    assert out[str(ext)].startswith("ok")


def test_missing_keeper_skipped_without_error(tmp_path):
    out = backup_keepers(tmp_path, tmp_path / "bk", ["data/nope.parquet"], "20260731")
    assert "skip" in out["data/nope.parquet"]


def test_worm_same_day_rerun_does_not_overwrite(tmp_path):
    _mk(tmp_path / "data/cyq_panel.parquet", content=b"v1")
    backup_keepers(tmp_path, tmp_path / "bk", ["data/cyq_panel.parquet"], "20260731")
    (tmp_path / "data/cyq_panel.parquet").write_bytes(b"v2")
    out = backup_keepers(
        tmp_path, tmp_path / "bk", ["data/cyq_panel.parquet"], "20260731"
    )
    assert (tmp_path / "bk/cyq_panel__20260731.parquet").read_bytes() == b"v1"
    assert "exists" in out["data/cyq_panel.parquet"]


def test_retention_keeps_newest_n(tmp_path):
    bk = tmp_path / "bk"
    for i, d in enumerate(["20260728", "20260729", "20260730"]):
        _mk(bk / f"cyq_panel__{d}.parquet", mtime=1000 + i)
    _mk(tmp_path / "data/cyq_panel.parquet", mtime=2000)
    backup_keepers(tmp_path, bk, ["data/cyq_panel.parquet"], "20260731", retention=2)
    remaining = sorted(p.name for p in bk.glob("cyq_panel__*.parquet"))
    assert remaining == ["cyq_panel__20260730.parquet", "cyq_panel__20260731.parquet"]


def test_empty_keepers_noop(tmp_path):
    assert backup_keepers(tmp_path, tmp_path / "bk", [], "20260731") == {}
