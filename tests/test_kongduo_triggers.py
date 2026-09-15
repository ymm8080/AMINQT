"""链路狙击引擎测试 (2026-09-14).

覆盖: no-look-ahead 铁律 / 三触发器阈值与边界 / 类型优先级 / 分层互斥 / 观察带四臂 /
真实面板案例回归。阈值全部从 settings.GENIOUS 读, 改旋钮测试自动跟随。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1 import kongduo_triggers as kt
from config.settings import GENIOUS, PANEL_V3_PATH

# ── no-look-ahead (项目铁律) ──────────────────────────────────────────────────


def _synth_panel(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """单票确定性随机游走, 制造足够长的 history 让 r120/SIG 预热完成。"""
    np.random.seed(seed)
    close = 10.0 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.02))
    high = close * (1 + np.abs(np.random.rand(n_rows)) * 0.02)
    low = close * (1 - np.abs(np.random.rand(n_rows)) * 0.02)
    dates = pd.bdate_range("2024-01-01", periods=n_rows).strftime("%Y%m%d")
    return pd.DataFrame(
        {
            "symbol": "000001",
            "date": dates,
            "board": "main",
            "high": high,
            "low": low,
            "close": close,
            "volume": 1e6 * (1 + np.random.rand(n_rows)),
            "winner_ratio": np.clip(np.random.rand(n_rows), 0.05, 0.95),
        }
    )


def test_no_lookahead_tail_mutation_does_not_change_history():
    """改 t 之后的行, t 及之前的全部特征/触发器必须逐位不变。"""
    base = _synth_panel()
    cut = 300
    dirty = base.copy()
    dirty.loc[cut + 1 :, ["high", "low", "close", "volume", "winner_ratio"]] *= 3.0

    fb = kt.compute_triggers(kt.compute_features(base))
    fd = kt.compute_triggers(kt.compute_features(dirty))

    cols = [
        c
        for c in (
            "duo",
            "kong",
            "gap",
            "pct",
            "r20",
            "r60",
            "r120",
            "wr_rise60",
            "pb5",
            "band20",
            "kong10",
            "d_since_kong",
            "ma10up",
            "ext10",
            "ext10p",
            "vr",
            "sl_flip_age",
            "sl_wash_days",
            "T1",
            "T2",
            "T3",
        )
        if c in fb.columns
    ]
    pd.testing.assert_frame_equal(
        fb.loc[:cut, cols], fd.loc[:cut, cols], check_dtype=True
    )


# ── 触发器阈值 / 边界 ─────────────────────────────────────────────────────────


def _trigger_frame(rows: list[dict]) -> pd.DataFrame:
    """只给 compute_triggers 需要的特征列 (它是纯函数, 可直接吃特征帧)。"""
    base = {
        "kong": False,
        "kong10": False,
        "duo": False,
        "r20": np.nan,
        "winner_ratio": np.nan,
        "wr_rise60": np.nan,
        "ma10up": False,
        "ext10": 1.0,
        "ext10p": 1.0,
        "pct": 0.0,
        "gap": 0.0,
        "pb5": 0.0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


_T1_OK = {
    "kong": True,
    "r20": 0.10,
    "winner_ratio": 0.70,
    "wr_rise60": 0.10,
    "ma10up": True,
    "ext10": 1.0,
}


def test_t1_thresholds_and_edges():
    df = kt.compute_triggers(
        _trigger_frame(
            [
                _T1_OK,  # 0 fire
                {
                    **_T1_OK,
                    "winner_ratio": GENIOUS["wr_min"],
                },  # 1 == wr_min → 不触发 (strict >)
                {**_T1_OK, "winner_ratio": 0.61},  # 2 fire
                {**_T1_OK, "wr_rise60": GENIOUS["wr_rise_min"]},  # 3 == 阈值 → 不触发
                {
                    **_T1_OK,
                    "ext10": GENIOUS["t1_ma10_dev_max"],
                },  # 4 == 1.05 → fire (<=)
                {**_T1_OK, "ext10": 1.06},  # 5 超标 → 不触发
                {**_T1_OK, "ma10up": False},  # 6 MA10 未上行 → 不触发
                {**_T1_OK, "r20": -0.01},  # 7 r20 未为正 → 不触发
                {**_T1_OK, "kong": False, "duo": True},  # 8 非洗盘日 → 不触发
            ]
        )
    )
    assert df["T1"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_t2_thresholds_and_edges():
    df = kt.compute_triggers(
        _trigger_frame(
            [
                {"kong10": True, "duo": True, "pct": 0.03},  # 0 fire
                {
                    "kong10": True,
                    "duo": True,
                    "pct": GENIOUS["t2_gain_min"],
                },  # 1 == 2% → 不触发
                {"kong10": True, "duo": True, "pct": 0.021},  # 2 fire
                {
                    "kong10": False,
                    "duo": True,
                    "pct": 0.03,
                },  # 3 前10日无空图标 → 不触发
                {"kong10": True, "duo": False, "pct": 0.03},  # 4 今日未上穿 → 不触发
            ]
        )
    )
    assert df["T2"].tolist() == [True, False, True, False, False]


def test_t3_thresholds_and_edges():
    df = kt.compute_triggers(
        _trigger_frame(
            [
                {"gap": 1.0, "pb5": -0.05, "pct": 0.06, "ext10p": 1.0},  # 0 fire
                {
                    "gap": 1.0,
                    "pb5": GENIOUS["t3_pb5_max"],
                    "pct": 0.06,
                },  # 1 == -3% → fire (<=)
                {"gap": 1.0, "pb5": -0.029, "pct": 0.06},  # 2 回撤不足 → 不触发
                {"gap": -1.0, "pb5": -0.05, "pct": 0.06},  # 3 空头态 → 不触发
                {
                    "gap": 1.0,
                    "pb5": -0.05,
                    "pct": GENIOUS["t3_gain_min"],
                },  # 4 == 5% → 不触发
                {
                    "gap": 1.0,
                    "pb5": -0.05,
                    "pct": 0.06,
                    "ext10p": 1.06,
                },  # 5 昨日乖离超标 → 不触发
            ]
        )
    )
    assert df["T3"].tolist() == [True, True, False, False, False, False]


def test_trigger_name_is_mutually_exclusive_for_t1():
    """T1 与 T2/T3 天然互斥 (kong ⇒ gap<0, 而 T3 要 gap>0) → 只有 T2+T3 会并。"""
    df = kt.compute_triggers(
        _trigger_frame(
            [
                {**_T1_OK, "gap": -1.0},  # T1
                {"kong10": True, "duo": True, "pct": 0.03, "gap": -1.0},  # T2
                {"gap": 1.0, "pb5": -0.05, "pct": 0.06},  # T3
                {
                    "kong10": True,
                    "duo": True,
                    "pct": 0.06,
                    "gap": 1.0,
                    "pb5": -0.05,
                },  # T2+T3
            ]
        )
    )
    assert kt.trigger_name(df).tolist() == ["T1", "T2", "T3", "T2+T3"]


# ── 类型优先级 ────────────────────────────────────────────────────────────────


def test_classify_type_precedence():
    df = pd.DataFrame(
        {
            "r60": [-0.40, -0.10, -0.10, 0.00, 0.00, 0.15, 0.30, np.nan],
            "r120": [0.50, 0.50, 0.00, -0.10, 0.10, 0.00, 0.00, 0.00],
        }
    )
    assert kt.classify_type(df).tolist() == [
        kt.TYPE_A,  # r60<=-30 压 C (r120>+40)
        kt.TYPE_C,  # 强牛深回调
        kt.TYPE_B1,  # 中跌带
        kt.TYPE_B2,  # 长平底 = B3 ∩ r120<=0
        kt.TYPE_B3,  # 浅跌平 (r120>0 故非 B2)
        kt.TYPE_D1,
        kt.TYPE_D2,
        kt.TYPE_UNKNOWN,
    ]


# ── 分层互斥 / 段位名 ─────────────────────────────────────────────────────────


_LAYER_COLS = {
    "T1": False,
    "T2": False,
    "T3": False,
    "r60": 0.0,
    "r120": -0.5,
    "r20": 0.0,
    "vr": 1.0,
    "pct": 0.0,
    "ext10p": 1.0,
    "ext10": 1.0,
    "pb5": 0.0,
    "winner_ratio": 0.5,
    "band20": 0.10,
    "board": "main",
    "close": 10.0,
    "sl_flip_age": 5.0,
    "sl_wash_days": 8.0,
    "date": "20260911",
    "symbol": "000000",
    # 主力筹码比例 (逐股递归) 与 换手率 需要这几列
    "open": 10.0,
    "high": 10.2,
    "low": 9.8,
    "turnover_rate": 2.0,
    # 大涨闸三条件的默认值 = **过闸** (r10<=0 / r5>0 / vr<=1), 于是不用管闸的单测保持原语义
    "r10": -0.10,
    "r5": 0.05,
}


def _layer_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**_LAYER_COLS, **r} for r in rows])


@pytest.fixture
def chip_stub(monkeypatch):
    """筹码替身: build_delivery 里的 compute_main_chip_ratio / compute_chip_trend 都是逐股
    递归真算, 单测要可控。

    被替身读 `_chip` / `_slope` 列 (缺省 95.0 / 1.0), 于是每行能单独指定红柱与控盘MA10斜率;
    `集中度Δ10` 只是展示列, 替身给 0.0。
    """

    def _apply(default: float = 95.0, slope: float = 1.0):
        def _ratio(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d["主力筹码比例"] = d["_chip"] if "_chip" in d.columns else default
            return d

        def _trend(d: pd.DataFrame) -> pd.DataFrame:
            d = d.copy()
            d["控盘MA10斜率"] = d["_slope"] if "_slope" in d.columns else slope
            d["集中度Δ10"] = d["_d10"] if "_d10" in d.columns else 0.0
            return d

        monkeypatch.setattr(kt, "compute_main_chip_ratio", _ratio)
        monkeypatch.setattr(kt, "compute_chip_trend", _trend)

    return _apply


def test_assign_layers_each_segment():
    df = _layer_frame(
        [
            {"T3": True, "r60": -0.40, "vr": 1.0, "pct": 0.08},  # CH3
            {
                "T3": True,
                "r60": -0.40,
                "vr": GENIOUS["vr_quiet"] + 0.1,
            },  # vr 超缩量闸 → 非 CH3
            {"T2": True, "r60": -0.35, "pct": 0.03},  # CH2
            {"T1": True, "r120": -0.20, "r20": 0.10},  # CH1
            {"T2": True, "r60": -0.10, "ext10p": 0.90, "pct": 0.03},  # CH2B
            {"T1": True, "r120": 0.10, "r20": 0.00},  # T1余 (T1宽内)
            {"T2": True, "r60": 0.00, "pct": 0.03, "vr": 1.0},  # 带T2温火
            {"T2": True, "r60": 0.00, "pct": 0.10, "vr": 1.0},  # 带T2涨停
            {"T3": True, "r60": -0.05, "pct": 0.06, "vr": 1.0},  # 带T3温火
            {"T3": True, "r60": -0.05, "pct": 0.10, "vr": 1.0},  # 带T3涨停
        ]
    )
    assert kt.assign_layers(df).tolist() == [
        kt.CH3_T3_DEEP_QUIET,
        "",  # vr 超 1.5 且 r60<=-30 未进 CH2(T2 非真) → 无层
        kt.CH2_T2_DEEP,
        kt.CH1_T1_LONGBASE,
        kt.CH2B_T2_STEADY,
        kt.T1_REST,
        kt.BAND_T2_WARM,
        kt.BAND_T2_LIMIT,
        kt.BAND_T3_WARM,
        kt.BAND_T3_LIMIT,
    ]


def test_t1_rest_uses_wide_gate():
    """T1余 走 T1宽: 剔 r120>+40% 与 r60<-5%, 否则票量从 8.4 涨到 10.4/日。"""
    df = _layer_frame(
        [
            {
                "T1": True,
                "r120": GENIOUS["t1_wide_r120_max"] + 0.01,
                "r20": 0.0,
            },  # r120 超宽闸 → 弃
            {
                "T1": True,
                "r120": 0.0,
                "r60": GENIOUS["t1_wide_r60_min"] - 0.01,
            },  # r60 超宽闸 → 弃
            {
                "T1": True,
                "r120": 0.0,
                "r60": GENIOUS["t1_wide_r60_min"],
            },  # 边界含 → T1余
        ]
    )
    assert kt.assign_layers(df).tolist() == ["", "", kt.T1_REST]


def test_assign_layers_mutual_exclusivity_and_precedence():
    """T2∩T3 同火归最高层 CH3; 任一行最多一个段位。"""
    df = _layer_frame(
        [
            {
                "T2": True,
                "T3": True,
                "r60": -0.40,
                "vr": 1.0,
                "pct": 0.03,
            },  # 双触发 → CH3
            {
                "T2": True,
                "T3": True,
                "r60": 0.00,
                "pct": 0.10,
                "vr": 1.0,
            },  # 双触发非深跌 → T2 臂先占
            {
                "T2": True,
                "T3": True,
                "r60": -0.10,
                "ext10p": 0.90,
                "pct": 0.03,
            },  # → CH2B
        ]
    )
    layer = kt.assign_layers(df)
    assert layer.tolist() == [kt.CH3_T3_DEEP_QUIET, kt.BAND_T2_LIMIT, kt.CH2B_T2_STEADY]
    assert (layer != "").all()


def test_band_requires_trigger_and_drops_untuned_gap():
    """观察带底必须是 T2余/T3余: 无触发器的当日上涨票不得进带 (否则量级失控)。"""
    df = _layer_frame(
        [
            {"pct": 0.03, "vr": 1.0, "r120": -0.5},  # 无触发器 → 无层
            {
                "T2": True,
                "pct": 0.06,
                "vr": 3.0,
                "r120": -0.5,
            },  # 涨幅 5-9% 且放量 → 无层
            {"T2": True, "pct": 0.03, "vr": 3.0, "r120": -0.5},  # 温火但放量 → 无层
            {"T2": True, "pct": 0.03, "vr": 1.0, "r120": 0.5},  # 温火但 r120>0 → 无层
        ]
    )
    assert kt.assign_layers(df).tolist() == ["", "", "", ""]


def test_build_delivery_splits_sheets_and_ranks(chip_stub):
    """Sheet1=冠军四段; Sheet2=其余, 排名从 1 起; 只取当日。"""
    chip_stub()  # 本测测分表/排名, 不测红柱闸 → 全体过闸
    df = _layer_frame(
        [
            {
                "T1": True,
                "r120": -0.20,
                "r20": 0.10,
                "date": "20260911",
                "symbol": "000001",
            },
            {
                "T3": True,
                "r60": -0.40,
                "vr": 1.0,
                "pct": 0.08,
                "date": "20260911",
                "symbol": "000002",
            },
            {
                "T1": True,
                "r120": 0.10,
                "r20": 0.00,
                "date": "20260911",
                "symbol": "000003",
            },
            {
                "T3": True,
                "r60": -0.05,
                "pct": 0.10,
                "vr": 1.0,
                "date": "20260911",
                "symbol": "000004",
            },
            {
                "T1": True,
                "r120": -0.20,
                "r20": 0.10,
                "date": "20260910",
                "symbol": "000005",
            },
        ]
    )
    s1, s2, s2_full = kt.build_delivery(df, "20260911")
    assert list(s1["层"]) == [kt.CH3_T3_DEEP_QUIET, kt.CH1_T1_LONGBASE]
    # 000004 的 r60/r120 都更低 (更没涨) → 观察分更高, 排在 000003 前
    assert list(s2["symbol"]) == ["000004", "000003"]
    assert list(s2["层"]) == [kt.BAND_T3_LIMIT, kt.T1_REST]
    assert list(s1["排名"]) == [1, 2] and list(s2["排名"]) == [1, 2]
    # 全量表 = 冠军段 + 观察池**全部** (含 20260910 的隔日票不算), 一票不丢
    assert len(s2_full) == 4
    assert set(s2_full["symbol"]) == {"000001", "000002", "000003", "000004"}
    assert list(s2_full["排名"]) == [1, 2, 3, 4]
    assert set(s1["层"]) <= set(kt.SHEET1_LAYERS)
    assert set(s2["层"]) <= set(kt.SHEET2_LAYERS)
    assert list(s1.columns)[:4] == ["排名", "symbol", "层", "触发器"]
    # 每个段位都要有执行档与研究口径, 不能出现 NaN
    assert s1["执行档"].notna().all() and s1["全样本口径"].notna().all()
    # S-L 形态标注只上 Sheet2, 且插在数值块末尾 (5日回撤 之后), 不打散前面几列
    for extra in kt.SHEET2_EXTRA_COLUMNS:
        assert extra not in s1.columns
        assert s2.columns.get_loc(extra) > s2.columns.get_loc("5日回撤")
        assert s2.columns.get_loc(extra) < s2.columns.get_loc("全样本口径")
        assert s2_full[extra].notna().all()


def test_sheet2_ranks_unrun_first_and_truncates(chip_stub):
    """观察分把"还没涨透"的顶到前面, 已涨透的沉底; Sheet2 截断, 全量表不截断。"""
    chip_stub()
    top_n = int(GENIOUS["sheet2_top_n"])
    rows = [
        # 已涨透: 带宽大 / 乖离高 / 获利盘高 / r60 正
        {
            "T3": True,
            "r60": 0.15,
            "r120": 0.0,
            "pct": 0.10,
            "band20": 0.90,
            "ext10": 1.20,
            "winner_ratio": 0.95,
            "date": "20260911",
            "symbol": "000001",
        },
        # 未启动: 带宽窄 / 未偏离 / 获利盘低 / 跌得比填充行深 (但不到 CH3 的 r60<=-30)
        {
            "T3": True,
            "r60": -0.20,
            "r120": -0.20,
            "pct": 0.10,
            "band20": 0.05,
            "ext10": 0.95,
            "winner_ratio": 0.10,
            "date": "20260911",
            "symbol": "000002",
        },
    ]
    # 填充到超过 top_n (T1余: r120<=0.40 且 r60>=-0.05)
    for i in range(top_n + 3 - len(rows)):
        rows.append(
            {
                "T1": True,
                "r60": 0.0,
                "r120": 0.0,
                "r20": 0.0,
                "pct": 0.0,
                "band20": 0.10 + 0.001 * i,
                "ext10": 1.0,
                "winner_ratio": 0.5,
                "date": "20260911",
                "symbol": f"9{i:05d}",
            }
        )
    df = _layer_frame(rows)
    _, s2, s2_full = kt.build_delivery(df, "20260911")

    assert len(s2_full) == len(rows)  # 全量表不截断
    assert len(s2) == top_n  # 可读表截断
    assert list(s2["排名"]) == list(range(1, top_n + 1))
    assert s2_full["排名"].iloc[0] == 1
    # 未涨的 000002 第一, 已涨透的 000001 沉到全量表最后
    assert s2["symbol"].iloc[0] == "000002"
    assert s2["乖离MA10"].iloc[0] == pytest.approx(0.95)
    assert s2_full["symbol"].iloc[-1] == "000001"
    assert set(s2["层"]) <= set(kt.SHEET2_LAYERS)


def test_sheet2_sort_key_switch_orders_by_sl_flip(monkeypatch, chip_stub):
    """ "SL翻正" 备用模式: 刚翻正 + 洗得久 → 前; 从未翻正的沉底。"""
    chip_stub()
    monkeypatch.setitem(GENIOUS, "sheet2_sort_key", "SL翻正")
    df = _layer_frame(
        [
            {
                "T1": True,
                "r120": 0.0,
                "r20": 0.0,
                "sl_flip_age": 0.0,
                "sl_wash_days": 30.0,
                "date": "20260911",
                "symbol": "000001",
            },
            {
                "T1": True,
                "r120": 0.0,
                "r20": 0.0,
                "sl_flip_age": 9.0,
                "sl_wash_days": 5.0,
                "date": "20260911",
                "symbol": "000002",
            },
            {
                "T1": True,
                "r120": 0.0,
                "r20": 0.0,
                "sl_flip_age": np.nan,
                "sl_wash_days": np.nan,
                "date": "20260911",
                "symbol": "000003",
            },
        ]
    )
    _, s2, _ = kt.build_delivery(df, "20260911")
    assert list(s2["symbol"]) == ["000001", "000002", "000003"]
    ages = s2["SL翻正年龄"].to_numpy()
    assert ages[0] == 0.0 and ages[1] == 9.0 and np.isnan(ages[2])


def test_sheet2_sort_key_default_is_observation_score():
    """默认排序键必须是观察分 (全 897 日实测最优); 形态模式只是备用。"""
    assert GENIOUS["sheet2_sort_key"] == "观察分"


# ── 大涨闸: 低动量 + 右侧拐头 + 缩量 (2026-09-14 用户令, 用 B 摘除见 config 注释) ──


def test_dir_gate_marks_without_dropping_rows(chip_stub):
    """三条件闸**只标注不删行** (0914 用户令): 三张表一票不丢, 只多一列「大涨闸」。"""
    chip_stub()
    ok = {"r10": -0.10, "r5": 0.05, "vr": 1.0}
    df = _layer_frame(
        [
            {"T3": True, "r60": -0.40, "pct": 0.08, "symbol": "000001", **ok},
            # 近 5 日还在跌 = 左侧 → 拦
            {
                "T3": True,
                "r60": -0.40,
                "pct": 0.08,
                "symbol": "000002",
                **{**ok, "r5": -0.01},
            },
            {"T1": True, "r120": 0.0, "r20": 0.0, "symbol": "000003", **ok},
            # 十日已涨 = 非低动量 → 拦
            {
                "T1": True,
                "r120": 0.0,
                "r20": 0.0,
                "symbol": "000004",
                **{**ok, "r10": 0.05},
            },
        ]
    )
    s1, s2, s3 = kt.build_delivery(df, "20260911")

    # 冠军表两只都在 (被拦的 000002 不再被剔掉), 只按层序 + 段内 r60 排
    assert list(s1["symbol"]) == ["000001", "000002"]
    assert list(s1["大涨闸"]) == ["过闸", "被拦"]
    assert set(s2["symbol"]) == {"000003", "000004"}
    assert dict(zip(s2["symbol"], s2["大涨闸"])) == {"000003": "过闸", "000004": "被拦"}
    assert set(s3["symbol"]) == {"000001", "000002", "000003", "000004"}
    assert dict(zip(s3["symbol"], s3["大涨闸"])) == {
        "000001": "过闸",
        "000002": "被拦",
        "000003": "过闸",
        "000004": "被拦",
    }
    # 全量表: 被拦者排最前 (回看优先), 过闸者殿后
    assert list(s3["大涨闸"]) == ["被拦", "被拦", "过闸", "过闸"]
    assert "大涨闸" in s1.columns and "大涨闸" in s2.columns


def test_dir_gate_ignores_chip_ma_slope(chip_stub):
    """筹码MA10斜率已摘出闸: 它下行/为 NaN 都不影响过闸 (只当展示列)。"""
    chip_stub()
    ok = {"r10": -0.10, "r5": 0.05, "vr": 1.0}
    df = _layer_frame(
        [
            {
                "T3": True,
                "r60": -0.40,
                "pct": 0.08,
                "symbol": "000001",
                **{**ok, "_slope": -1.0},
            },
            {
                "T3": True,
                "r60": -0.40,
                "pct": 0.08,
                "symbol": "000002",
                **{**ok, "_slope": np.nan},
            },
        ]
    )
    s1, _, s3 = kt.build_delivery(df, "20260911")

    assert list(s1["symbol"]) == ["000001", "000002"]
    assert set(s1["大涨闸"]) == {"过闸"}
    assert set(s3["大涨闸"]) == {"过闸"}


def test_dir_gate_boundary_is_strict_and_fails_closed(chip_stub):
    """边界: r5 严格 >0, r10 / 量比 取等号算过闸; NaN (历史不足) 一律被拦。"""
    chip_stub()
    vr_max = float(GENIOUS["dir_gate_vr_max"])
    edge = {"r10": 0.0, "r5": 1e-9, "vr": vr_max}
    rows = [
        ("000001", {}),  # 恰在阈值上, 三条全过
        ("000002", {"r5": 0.0}),  # 严格 > 才过
        ("000003", {"r10": 1e-9}),  # 低动量上限取等号算过, 超一点就拦
        ("000004", {"vr": vr_max + 1e-9}),
        ("000005", {"r5": np.nan}),  # 算不出 → 不静默放行
    ]
    df = _layer_frame(
        [
            {"T3": True, "r60": -0.40, "pct": 0.08, "symbol": sym, **{**edge, **over}}
            for sym, over in rows
        ]
    )
    s1, _, s3 = kt.build_delivery(df, "20260911")

    assert len(s1) == 5  # 全在, 一只没删
    assert dict(zip(s1["symbol"], s1["大涨闸"])) == {
        "000001": "过闸",
        "000002": "被拦",
        "000003": "被拦",
        "000004": "被拦",
        "000005": "被拦",
    }
    assert dict(zip(s3["symbol"], s3["大涨闸"])) == {
        "000001": "过闸",
        "000002": "被拦",
        "000003": "被拦",
        "000004": "被拦",
        "000005": "被拦",
    }


def test_dir_gate_off_admits_everything(chip_stub, monkeypatch):
    """回退旋钮 dir_gate=False: 闸不生效, 全部标「过闸」。"""
    chip_stub()
    monkeypatch.setitem(GENIOUS, "dir_gate", False)
    df = _layer_frame(
        [
            # 三条件全不满足 (十日已涨 / 五日还在跌 / 量比>1 但仍在 CH3 的 vr 闸内)
            {
                "T3": True,
                "r60": -0.40,
                "pct": 0.08,
                "symbol": "000001",
                "r10": 0.50,
                "r5": -0.20,
                "vr": 1.2,
            },
            # T1余 属观察池 (不筛), 用来验证闸关时它也标「过闸」
            {"T1": True, "r120": 0.0, "r20": 0.0, "symbol": "000002", "r10": 0.50},
        ]
    )
    s1, _, s3 = kt.build_delivery(df, "20260911")
    assert list(s1["symbol"]) == ["000001"]
    assert set(s3["大涨闸"]) == {"过闸"}


def test_dir_gate_does_not_reorder_sheet2(chip_stub, monkeypatch):
    """Sheet2 不过闸 → 闸开关对观察池逐位不变, 只在「大涨闸」列上体现。"""
    chip_stub()
    df = _layer_frame(
        [
            {"T1": True, "r120": -0.30, "r20": 0.0, "symbol": "000001"},
            {
                "T1": True,
                "r120": -0.20,
                "r20": 0.0,
                "symbol": "000002",
                "r5": -0.50,
            },  # 唯一被拦
            {"T1": True, "r120": 0.30, "r20": 0.0, "symbol": "000003"},
        ]
    )
    monkeypatch.setitem(GENIOUS, "dir_gate", False)
    _, s2_off, _ = kt.build_delivery(df, "20260911")
    monkeypatch.setitem(GENIOUS, "dir_gate", True)
    _, s2_on, _ = kt.build_delivery(df, "20260911")

    assert list(s2_off["symbol"]) == ["000001", "000002", "000003"]
    assert list(s2_on["symbol"]) == ["000001", "000002", "000003"]  # 逐位不变
    assert dict(zip(s2_on["symbol"], s2_on["大涨闸"]))["000002"] == "被拦"
    assert set(s2_off["大涨闸"]) == {"过闸"}


# ── 真实面板案例回归 ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_day():
    import os

    if not os.path.exists(PANEL_V3_PATH):
        pytest.skip(f"面板缺失: {PANEL_V3_PATH}")
    df = kt.load_panel(PANEL_V3_PATH, "20260911", lookback_days=400)
    df = kt.compute_features(df)
    df = kt.compute_triggers(df)
    df["层"] = kt.assign_layers(df)
    return df


@pytest.mark.parametrize(
    "symbol,date,want_layer,want_trig",
    [
        ("000978", "20260903", kt.CH1_T1_LONGBASE, "T1"),  # 洗盘日 → 长基冠军段
        # 601869 r60≈-14% 不满足 CH2 的 r60<=-30 闸 → 记忆里它是 "T2 命中" 而非冠军段
        ("601869", "20260907", None, "T2"),
        ("002815", "20260907", kt.BAND_T3_LIMIT, "T3"),  # 状态点火 + 超跌涨停
        (
            "603421",
            "20260908",
            kt.BAND_T2_WARM,
            "T2",
        ),  # 翻转 + 温火 (r60 仅 -9% 不进冠军段)
    ],
)
def test_real_cases_hit_expected_layer(real_day, symbol, date, want_layer, want_trig):
    row = real_day[(real_day["symbol"] == symbol) & (real_day["date"] == date)]
    assert len(row) == 1, f"{symbol}@{date} 应恰好 1 行"
    trig = kt.trigger_name(row).iloc[0]
    assert want_trig in trig, f"{symbol}@{date} 触发器 {trig} 不含 {want_trig}"
    if want_layer is not None:
        assert row["层"].iloc[0] == want_layer


def test_real_panel_no_ohlcv_corruption(real_day):
    """铁律: OHLCV 校验 (异常不静默)。"""
    d = real_day
    bad = d[
        (d["high"] < d["low"])
        | (d["high"] < d["close"])
        | (d["low"] > d["close"])
        | (d["volume"] < 0)
    ]
    assert len(bad) == 0, f"{len(bad)} 行 OHLCV 异常"


def test_sl_flip_state_machine_matches_independent_derivation(real_day):
    """SL翻正年龄 / SL洗盘天数 必须与独立重算逐行一致。

    这里刻意**不**调 kt._rsv / kt._roll, 用裸 pandas 重走一遍 正/负 段:
    验证的是意图 (翻正=负段结束那天; 洗盘天数=翻正前那段负段的长度), 不是复述实现。
    """
    d = real_day.sort_values(["symbol", "date"])
    checked = 0
    for sym in d["symbol"].unique()[:12]:
        s = d[d["symbol"] == sym]
        hh14 = s["high"].rolling(14, min_periods=14).max()
        ll14 = s["low"].rolling(14, min_periods=14).min()
        rsv14 = (100 * (s["close"] - hh14) / (hh14 - ll14).replace(0, np.nan)).ffill()
        hh34 = s["high"].rolling(34, min_periods=34).max()
        ll34 = s["low"].rolling(34, min_periods=34).min()
        rsv34 = (100 * (s["close"] - hh34) / (hh34 - ll34).replace(0, np.nan)).ffill()
        ma34 = rsv34.rolling(19, min_periods=19).mean()
        pos = (rsv14 > ma34).fillna(False).to_numpy()

        want_age, want_wash = [], []
        last_up, neg_len, cur = None, None, 0
        for i, p in enumerate(pos):
            if not p:
                cur += 1
                neg_len = cur
                want_wash.append(float(neg_len))
            else:
                cur = 0
                want_wash.append(float(neg_len) if neg_len is not None else np.nan)
                if i == 0 or not pos[i - 1]:
                    last_up = i
            want_age.append(float(i - last_up) if last_up is not None else np.nan)

        assert s["sl_flip_age"].to_numpy() == pytest.approx(
            np.array(want_age), nan_ok=True
        )
        assert s["sl_wash_days"].to_numpy() == pytest.approx(
            np.array(want_wash), nan_ok=True
        )
        checked += 1
    assert checked >= 8, "真实面板应能取到 >=8 只做核对"

    # 洗盘天数只在"刚翻正"时表达完整含义: 翻正日的洗盘天数应 >= 1
    flips = real_day[real_day["sl_flip_age"] == 0]
    assert len(flips) > 0, "真实面板应有翻正日"
    assert (flips["sl_wash_days"] >= 1).all(), (
        "翻正当日洗盘天数必须 >=1 (至少洗过 1 天)"
    )
