"""筹码水位标注四线接线单测 (2026-09-22 用户令 "SET VALUE TO 低获利，涨 & 高获利，跌").

口径: 获利盘水位 chip_wr<0.5 → chip_flag="低获利，涨" / ≥0.5 → "高获利，跌" /
缺 (NaN) → 空。列标注 + chip_wr 水位列 (wr5 数值列 0922 午后随用户令退役),
不删票; cyq 数据缺 → fail-open 不标。沿革: 09-05 删除闸 (wr5<0) → 09-09 用户
推翻改标注 → 0922 更名 "派发 & 后续回归" → 同日改水位轴 (wr5 判死, 水位是唯一
有信息的筹码族)。
LEGACY 交付 (_deliver_legacy_list)、PARALLEL 短名单 (_shortlist_t5_t10) 与
GENIOUS 冠军表 (_genious_excel) 共用 scripts._prob10_density_shadow.apply_chip_gate
(四线同源读 cyq_panel); 密度影子单走 density_picks 内联 apply_wr5_gate
(另见 test_prob10_density_shadow.py / test_genious_bigdrop_scan.py)。
"""

import importlib

import pandas as pd

import scripts._prob10_density_shadow as mod
from scripts._prob10_density_shadow import (
    CHIP_FLAG_HIGH,
    CHIP_FLAG_LOW,
    apply_chip_gate,
)

DAY = pd.Timestamp("2026-09-04")


def _df(syms):
    return pd.DataFrame({"symbol": list(syms)})


def _chip(mapping, lvl=None):
    out = {"symbol": list(mapping)}
    if lvl is not None:
        out["wr"] = [lvl[s] for s in mapping]
    return pd.DataFrame(out)


def test_apply_chip_gate_marks_and_logs(capsys, monkeypatch):
    monkeypatch.setattr(
        mod,
        "load_chip_features",
        lambda ts: _chip(
            {"000001": -0.23, "000002": 0.05, "000003": float("nan")},
            lvl={"000001": 0.91, "000002": 0.23, "000003": float("nan")},
        ),
    )
    df = _df(["000001", "000002", "000003"])
    out = apply_chip_gate(df, DAY)
    # 不删票: 三行全在; 水位 0.91→高获利，跌 / 0.23→低获利，涨 / NaN→空
    assert list(out["symbol"]) == ["000001", "000002", "000003"]
    assert out.loc[out.symbol == "000001", "chip_flag"].iloc[0] == CHIP_FLAG_HIGH
    assert out.loc[out.symbol == "000002", "chip_flag"].iloc[0] == CHIP_FLAG_LOW
    assert out.loc[out.symbol == "000003", "chip_flag"].iloc[0] == ""
    assert abs(out.loc[out.symbol == "000001", "chip_wr"].iloc[0] - 0.91) < 1e-12
    assert abs(out.loc[out.symbol == "000002", "chip_wr"].iloc[0] - 0.23) < 1e-12
    log = capsys.readouterr().out
    assert "筹码水位标注" in log and "低获利，涨 1 只" in log
    assert "高获利，跌 1 只" in log


def test_apply_chip_gate_failopen_and_clean(capsys, monkeypatch):
    # cyq 数据缺 → 原样返回, 无标注列
    monkeypatch.setattr(mod, "load_chip_features", lambda ts: None)
    df = _df(["000001"])
    out = apply_chip_gate(df, DAY)
    assert len(out) == 1
    assert "chip_flag" not in out.columns
    assert "fail-open" in capsys.readouterr().out
    # 旧 fixture 无 wr 列 (cyq 水位缺) → 标注列在但全空, 无日志
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


def test_genious_wires_shared_level_split():
    """GENIOUS 冠军表 (第四线) 走共享 apply_chip_gate — 勿另写切分: 四线口径必须
    一致, 表内切分漂移 = 同一只票在不同清单标相反方向。表内「获利盘」列 (0923 加回)
    只是数值展示, 切分不读它; 标注列只插 筹码标注 一列文本 (旁路契约同 BIGDROP SCAN)。"""
    import inspect

    gx = importlib.import_module("scripts._genious_excel")
    assert gx.apply_chip_gate is apply_chip_gate
    src = inspect.getsource(gx.main)
    assert "apply_chip_gate(s1" in src  # 同源切分锁死
    assert '"筹码标注"' in src  # 附加列紧随 BIGDROP SCAN 放最前 (用户 0915 令)


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
    """legacy 交付端人读层带筹码标注读法 (md/docx ℹ 筹码标注: 低获利，涨/高获利，跌)."""
    import inspect

    legacy = importlib.import_module("scripts._deliver_legacy_list")
    src_md = inspect.getsource(legacy.write_md)
    assert "筹码标注" in src_md and "chip_flag" in src_md
    assert CHIP_FLAG_LOW in src_md and CHIP_FLAG_HIGH in src_md
    src_main = inspect.getsource(legacy.main)
    assert "筹码标注" in src_main and "chip_flag" in src_main  # docx 段落
    assert CHIP_FLAG_LOW in src_main and CHIP_FLAG_HIGH in src_main
