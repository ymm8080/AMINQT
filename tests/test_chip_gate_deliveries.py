"""筹码派发闸三线接线单测 (2026-09-05 用户拍板 "所有模型。只要派发都删吧").

口径: 获利盘5日回落 wr5<0 → 剔除, 不补齐; cyq 数据缺 → fail-open 不拦。
LEGACY 交付 (_deliver_legacy_list) 与 PARALLEL 短名单 (_shortlist_t5_t10)
共用 scripts/_prob10_density_shadow.apply_chip_gate; 密度影子单走
density_picks 内联 apply_wr5_gate (另见 test_prob10_density_shadow.py)。
"""

import importlib

import pandas as pd

import scripts._prob10_density_shadow as mod
from scripts._prob10_density_shadow import apply_chip_gate

DAY = pd.Timestamp("2026-09-04")


def _df(syms):
    return pd.DataFrame({"symbol": list(syms)})


def _chip(mapping):
    return pd.DataFrame({"symbol": list(mapping), "wr5": list(mapping.values())})


def test_apply_chip_gate_cuts_and_logs(capsys, monkeypatch):
    monkeypatch.setattr(mod, "load_chip_features",
                        lambda ts: _chip({"000001": -0.23, "000002": 0.05,
                                          "000003": float("nan")}))
    df = _df(["000001", "000002", "000003"])
    out = apply_chip_gate(df, DAY)
    assert list(out["symbol"]) == ["000002", "000003"]  # wr5<0 剔; NaN 保留
    log = capsys.readouterr().out
    assert "剔除 1 只" in log and "000001" in log


def test_apply_chip_gate_failopen_and_clean(capsys, monkeypatch):
    # cyq 数据缺 → 原样返回
    monkeypatch.setattr(mod, "load_chip_features", lambda ts: None)
    df = _df(["000001"])
    out = apply_chip_gate(df, DAY)
    assert len(out) == 1
    assert "fail-open" in capsys.readouterr().out
    # 全员健康 → 无剔除无日志
    monkeypatch.setattr(mod, "load_chip_features",
                        lambda ts: _chip({"000001": 0.10}))
    out2 = apply_chip_gate(df, DAY)
    assert len(out2) == 1
    assert capsys.readouterr().out == ""


def test_delivery_scripts_wire_shared_gate():
    """两交付端 import 干净且接线到共享 apply_chip_gate (wiring 锁死)."""
    legacy = importlib.import_module("scripts._deliver_legacy_list")
    parallel = importlib.import_module("scripts._shortlist_t5_t10")
    assert legacy.apply_chip_gate is apply_chip_gate
    assert parallel.apply_chip_gate is apply_chip_gate
    import inspect

    src = inspect.getsource(parallel.main)
    assert "apply_chip_gate(res, sel_date" in src  # 滞留行之后、锚定之前
    src_l = inspect.getsource(legacy.main)
    assert "apply_chip_gate(df" in src_l           # symbol 规整后、滞涨标记之前
