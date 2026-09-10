"""筹码派发标注三线接线单测 (2026-09-09 用户拍板 "派发不删, 清单标注").

口径: 获利盘5日回落 wr5<0 → chip_flag=派发 列标注 + chip_wr5 数值列, 不删票;
cyq 数据缺 → fail-open 不标。LEGACY 交付 (_deliver_legacy_list) 与 PARALLEL
短名单 (_shortlist_t5_t10) 共用 scripts/_prob10_density_shadow.apply_chip_gate;
密度影子单走 density_picks 内联 apply_wr5_gate (另见 test_prob10_density_shadow.py)。
沿革: 09-05 曾为删除闸 (wr5<0 真删), 09-09 用户推翻改标注。
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


def test_apply_chip_gate_marks_and_logs(capsys, monkeypatch):
    monkeypatch.setattr(
        mod,
        "load_chip_features",
        lambda ts: _chip({"000001": -0.23, "000002": 0.05, "000003": float("nan")}),
    )
    df = _df(["000001", "000002", "000003"])
    out = apply_chip_gate(df, DAY)
    # 不删票: 三行全在; wr5<0 标派发, NaN 不标
    assert list(out["symbol"]) == ["000001", "000002", "000003"]
    assert out.loc[out.symbol == "000001", "chip_flag"].iloc[0] == "派发"
    assert out.loc[out.symbol == "000002", "chip_flag"].iloc[0] == ""
    assert out.loc[out.symbol == "000003", "chip_flag"].iloc[0] == ""
    assert abs(out.loc[out.symbol == "000001", "chip_wr5"].iloc[0] + 0.23) < 1e-12
    log = capsys.readouterr().out
    assert "派发标注 1 只" in log and "000001" in log


def test_apply_chip_gate_failopen_and_clean(capsys, monkeypatch):
    # cyq 数据缺 → 原样返回, 无标注列
    monkeypatch.setattr(mod, "load_chip_features", lambda ts: None)
    df = _df(["000001"])
    out = apply_chip_gate(df, DAY)
    assert len(out) == 1
    assert "chip_flag" not in out.columns
    assert "fail-open" in capsys.readouterr().out
    # 全员健康 → 标注列在但全空, 无日志
    monkeypatch.setattr(mod, "load_chip_features", lambda ts: _chip({"000001": 0.10}))
    out2 = apply_chip_gate(df, DAY)
    assert len(out2) == 1
    assert (out2["chip_flag"] == "").all()
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
    assert "apply_chip_gate(df" in src_l  # symbol 规整后、滞涨标记之前


def test_parallel_chip_gate_enabled_by_default():
    """2026-09-09 三线统一标注: 默认 enable=True (标注接线), 调用点必须被开关
    守卫 (回退改 False = parallel 线不标注; 密度/legacy 线不受此开关影响)."""
    from config.settings import PARALLEL_CHIP_GATE

    assert PARALLEL_CHIP_GATE["enable"] is True
    import inspect

    parallel = importlib.import_module("scripts._shortlist_t5_t10")
    src = inspect.getsource(parallel.main)
    assert 'if PARALLEL_CHIP_GATE.get("enable"' in src  # 守卫锁死, 防裸调用回归


def test_legacy_marks_wired_into_md_and_docx():
    """legacy 交付端人读层带派发提示块 (md ⚠ 筹码派发 / docx 同款段落)."""
    import inspect

    legacy = importlib.import_module("scripts._deliver_legacy_list")
    src_md = inspect.getsource(legacy.write_md)
    assert "筹码派发" in src_md and "chip_flag" in src_md
    src_main = inspect.getsource(legacy.main)
    assert "筹码派发" in src_main and "chip_flag" in src_main  # docx 段落
