"""概率头密度版影子单纯函数单测 (2026-09-05 建, 2026-09-06 口径替换).

口径锁死 (2026-09-06 用户拍板 "把密度王替换成 L3 TOP20这条线", 勿静默改):
  名单 = 每板 prob 降序前20带 + 回撤闸 (-10%) + 带内密度 occ5≥3
  + 筹码水位标注 (0922 用户令 "SET VALUE TO 低获利，涨 & 高获利，跌":
    获利盘水位 chip_wr<0.5 → chip_flag="低获利，涨" / ≥0.5 → "高获利，跌",
    不删票; 数据缺 fail-open; 原 wr5<0 派发标注同日判死改水位轴);
  免额 (额不作闸, amt 仅展示列) — 09-06 拍板 "去额";
  belief_down = prob − 3个上榜日前 prob (标签列, 非闸)。
"""

import sys

import numpy as np
import pandas as pd

from scripts._prob10_density_shadow import (
    _COLS,
    CHIP_FLAG_HIGH,
    CHIP_FLAG_LOW,
    CHIP_WR_LEVEL_SPLIT,
    OCC_MIN,
    OCC_WIN,
    PULL_FLAG_MAX,
    PULL_FLOOR,
    TOP_N,
    TREND_MA10_GATE,
    apply_wr5_gate,
    chip_level_label,
    density_picks,
    load_chip_features,
    prob10_membership,
    trend_rising,
)

DAY = pd.Timestamp("2026-09-04")


def _cand():
    # main 板 4 只 prob 降序: A>B>C>D; dual 板 2 只 E>F (夹具避开 000xxx 撞码段)
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002", "600003", "600004", "300005", "688006"],
            "board": ["main", "main", "main", "main", "GEM", "STAR"],
            "prob_up_10d": [0.9, 0.8, 0.7, 0.6, 0.95, 0.85],
            "pred_ret_10d": [0.10, 0.09, 0.08, 0.07, 0.12, 0.11],
        }
    )


def _panel(symbols, close_last, amount_last, n=12, pull_ok=None):
    """构造 close/amount 透视表: close 平台后到 close_last (pull_ok[s]=False 则先冲高)."""
    idx = pd.bdate_range("2026-08-15", periods=n)
    close, amount = {}, {}
    for s, last in zip(symbols, close_last):
        seq = np.full(n, 10.0)
        if not (pull_ok or {}).get(s, True):
            seq[-3] = last / 0.85  # 3 日前高点 → 当日回撤 >10%
        seq[-1] = last
        close[s] = seq
        amount[s] = np.full(n, 2e8)
    return (pd.DataFrame(close, index=idx), pd.DataFrame(amount, index=idx))


def _hist():
    # main/600001 在最近 5 个历史日全勤 (occ5=5); main/600002 仅 1 天;
    # main/600003 全勤; dual/300005 全勤; prob 恒 0.88 → belief_down=0
    dates = pd.bdate_range("2026-08-28", periods=5)
    rows = []
    for d in dates:
        rows.append((d, "main", "600001", 0.88))
        rows.append((d, "main", "600003", 0.68))
        rows.append((d, "dual", "300005", 0.93))
    rows.append((dates[-1], "main", "600002", 0.78))
    return pd.DataFrame(rows, columns=["date", "board", "symbol", "prob"])


def test_prob10_topn_per_board_and_mapping():
    m = prob10_membership(_cand(), DAY)
    assert set(m["board"]) == {"main", "dual"}
    assert list(m[m.board == "main"]["symbol"]) == [
        "600001",
        "600002",
        "600003",
        "600004",
    ]
    assert list(m[m.board == "dual"]["symbol"]) == ["300005", "688006"]
    assert len(m) == 6


def test_band_cap_at_twenty_and_deterministic():
    big = pd.DataFrame(
        {
            "symbol": [f"{600100 + i:06d}" for i in range(25)],
            "board": ["main"] * 25,
            "prob_up_10d": [0.90 - 0.01 * i for i in range(25)],
            "pred_ret_10d": [0.10] * 25,
        }
    )
    m1 = prob10_membership(big, DAY)
    m2 = prob10_membership(big, DAY)
    pd.testing.assert_frame_equal(m1, m2)
    assert len(m1) == TOP_N == 20  # 25 只 main → cap 20
    assert "600120" not in set(m1["symbol"])  # prob 最低 5 只被 cap 剔
    assert "600100" in set(m1["symbol"])


def test_density_gates_pull_occ_no_amount():
    """免额 (09-06 拍板 "去额"): 额低不再剔 — amt 仅展示列."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    amount["600003"] = 5e7  # 旧额闸会剔; 现口径保留
    out = density_picks(_cand(), _hist(), close, amount, DAY)
    # 600001 (occ5=5) 与 600003 (occ5=5, hist 全勤) 过; 600002 occ5=1 剔; 300005 dual 过
    assert list(out["symbol"]) == ["600001", "600003", "300005"]
    assert list(out["board"]) == ["main", "main", "dual"]
    assert out["occ5"].min() >= OCC_MIN
    assert (out["pull"] >= PULL_FLOOR).all()
    assert "amt" in out.columns  # 展示列保留
    assert abs(float(out[out.symbol == "600003"]["amt"].iloc[0]) - 5e7) < 1e-6


def test_parallel_pred_columns_merged_and_nan_fill():
    """双模型列 (2026-09-05 用户): legacy_* = 选股口径两列; parallel_* = raw
    pred_prob_10d/pred_mag_10d 按 symbol 合并, 缺股/缺文件 → NaN, 列名锁死."""
    from scripts._prob10_density_shadow import _COLS

    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    par = pd.DataFrame(
        {
            "symbol": ["600001", "999999", "600001"],
            "pred_prob_10d": [0.55, 0.40, 0.56],
            "pred_mag_10d": [0.08, 0.05, 0.081],
        }
    )
    out = density_picks(_cand(), _hist(), close, amount, DAY, par=par)
    r1 = out[out.symbol == "600001"].iloc[0]
    assert abs(r1["parallel_prob"] - 0.56) < 1e-12  # 同码多行取末行
    assert abs(r1["parallel_pred10"] - 0.081) < 1e-12
    assert out[out.symbol == "600003"]["parallel_prob"].isna().all()
    # 缺文件 → parallel 两列全 NaN
    out2 = density_picks(_cand(), _hist(), close, amount, DAY)
    assert out2["parallel_prob"].isna().all()
    assert out2["parallel_pred10"].isna().all()
    # 列名 = 模型出处锁死
    assert list(out.columns) == _COLS


def test_legacy_columns_carry_selection_model_name():
    """选股口径 = legacy prob10 → 展示列名 legacy_prob/legacy_pred10 (出处可查)."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    out = density_picks(_cand(), _hist(), close, amount, DAY)
    r1 = out[out.symbol == "600001"].iloc[0]
    assert abs(r1["legacy_prob"] - 0.90) < 1e-12  # 今日候选 prob (原 prob 列)
    assert abs(r1["legacy_pred10"] - 0.10) < 1e-12


def test_csv_percent_display_layer():
    """交付 CSV 百分比显示层 (2026-09-05 用户 "输出的EXCEL是百分比"): 0..1 列
    ×100 加 %; pctChg 本就是百分数值只加 %; NaN → 空; 纯显示层不改入参."""
    from scripts._prob10_density_shadow import fmt_pct_display

    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    picks = density_picks(_cand(), _hist(), close, amount, DAY)
    picks["pctChg"] = [1.5, -3.1, 0.0]
    disp = fmt_pct_display(picks)
    mask = disp.symbol == "600001"
    r1 = disp[mask].iloc[0]
    assert r1["legacy_prob"] == "90.00%"
    assert r1["legacy_pred10"] == "10.00%"
    assert r1["parallel_prob"] == ""  # 缺 raw 文件 → NaN → 空
    assert r1["pull"] == "0.00%"
    assert r1["pctChg"] == "1.50%"  # 不再 ×100
    # 入参保持数值 (机器读/单测口径不变)
    assert abs(float(picks[mask]["legacy_prob"].iloc[0]) - 0.90) < 1e-12


def test_density_pull_flag_marks_deep_pull():
    """[0913 撤删改标] 深回撤不再删票: 留在清单标 pull_flag=回撤 (原闸语义降级为标注)."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4, pull_ok={"600001": False})
    out = density_picks(_cand(), _hist(), close, amount, DAY)
    assert "600001" in list(out["symbol"])  # 闸撤, 不删
    assert out.loc[out.symbol == "600001", "pull_flag"].iloc[0] == "回撤"
    assert (out.loc[out.symbol != "600001", "pull_flag"] == "").all()


def test_belief_down_tag_three_days_back():
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    hist = _hist()
    out = density_picks(_cand(), hist, close, amount, DAY)
    r1 = out[out.symbol == "600001"].iloc[0]
    # 信念降 = 今日候选 prob (0.90) − 3 个**交易日**前 hist prob (0.88)
    # [0915 修] 基准日按交易日历 (panel 索引) 取, 不按 hist 里出现的日期 —— 后者
    # 正是被修掉的那个假设 (hist 缺夜/含当日都会让"第 3 个"落到别的日子)
    tdays = [d for d in close.index if d < DAY]
    d3 = tdays[-3]
    p_3ago = float(hist[(hist.symbol == "600001") & (hist.date == d3)]["prob"].iloc[0])
    assert p_3ago == 0.88
    assert abs(r1["belief_down"] - (0.90 - p_3ago)) < 1e-9


def test_rerun_and_backfill_are_idempotent():
    """[0915 修] occ5 窗口必须按交易日历取, 与 hist 文件内容无关.

    事故: 09-14 同日跑两遍拿到**互斥**结果 (首跑 3 只 dual, 重跑 1 只 main)。
    机理 = 窗口取 "hist 里最后 4 个日期", 而 save_history 每次运行都追加当日 →
    重跑时窗口整体前移一格且当日被重复计数。同一 bug 还带 look-ahead: 先补较晚
    日期再补较早的, 较早那次窗口里含未来日期 (实测 09-13 补跑窗含 09-14)。

    三种 hist 必须给出**逐位相同**的清单: ① 首次跑 (hist 止于昨) ② 重跑
    (hist 已含当日) ③ 乱序补跑 (hist 含未来日期)。
    """
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    # 600001 在 5 个交易日全勤 (win4 满窗) → 窗口一挪 occ5 立刻变
    base_dates = pd.to_datetime(
        ["2026-08-26", "2026-08-27", "2026-08-28", "2026-08-31", "2026-09-01"]
    )
    rows = [(d, "main", "600001", 0.80 + 0.01 * i) for i, d in enumerate(base_dates)]
    hist_first = pd.DataFrame(rows, columns=["date", "board", "symbol", "prob"])
    # ② 重跑: save_history 已把当日 (DAY=09-04) 追加进去了
    hist_rerun = pd.concat(
        [
            hist_first,
            pd.DataFrame(
                [(DAY, "main", "600001", 0.99)],
                columns=["date", "board", "symbol", "prob"],
            ),
        ],
        ignore_index=True,
    )
    # ③ 乱序补跑: hist 里混进未来日期 (09-08 > DAY)
    hist_future = pd.concat(
        [
            hist_first,
            pd.DataFrame(
                [(pd.Timestamp("2026-09-08"), "main", "600001", 0.99)],
                columns=["date", "board", "symbol", "prob"],
            ),
        ],
        ignore_index=True,
    )
    out_first = density_picks(_cand(), hist_first, close, amount, DAY)
    out_rerun = density_picks(_cand(), hist_rerun, close, amount, DAY)
    out_future = density_picks(_cand(), hist_future, close, amount, DAY)
    assert list(out_first["symbol"]) == ["600001"]  # 只有它有带史, 其余 occ5=1
    pd.testing.assert_frame_equal(out_first, out_rerun)
    pd.testing.assert_frame_equal(out_first, out_future)
    r = out_first.iloc[0]
    assert r["occ5"] == 5  # 1 (今日在带) + 4 (日历窗前 4 个交易日全勤)
    # belief_down 基准日 = 日历第 3 个交易日 (08-28, prob 0.82), 不是 09-01/09-08
    assert abs(r["belief_down"] - (0.90 - 0.82)) < 1e-9


def test_constants_locked():
    # 0913 用户令撤回撤闸: PULL_FLOOR −0.10→−1.0 等效关闭, 原闸档降为标注线
    assert (TOP_N, PULL_FLOOR, OCC_WIN, OCC_MIN) == (20, -1.0, 5, 3)
    assert PULL_FLAG_MAX == -0.10  # 0913 撤删改标: 标注线=原闸档
    # 0922 水位切分: 获利盘过半=深获利 (清单票分布中位 0.534, 切分天然均衡)
    assert CHIP_WR_LEVEL_SPLIT == 0.5
    assert CHIP_FLAG_LOW == "低获利，涨" and CHIP_FLAG_HIGH == "高获利，跌"
    assert TREND_MA10_GATE is True  # 0914 用户拍板闸位 B: occ5 后终选滤当日 MA10↑
    assert "amt" in _COLS  # 免额后 amt 保留为展示列 (不作闸)
    assert {"chip_wr", "chip_flag", "pull_flag"} <= set(
        _COLS
    )  # 0922 水位/标注 (wr5 列同日退役) + 0913 回撤列交付


def test_trend_rising_pure():
    """MA10↑ 纯判定: 斜坡升 True / 末日暴跌拐头 False / 恒平 False / 行不足 None."""
    idx = pd.bdate_range("2026-08-01", periods=12)
    ramp = pd.DataFrame({"A": 10.0 + 0.5 * np.arange(12)}, index=idx)
    assert bool(trend_rising(ramp).iloc[0]) is True
    crash = ramp.copy()
    crash.iloc[-1, 0] = 5.0  # 末日 5.0: MA10 末行 < 前行
    assert bool(trend_rising(crash).iloc[0]) is False
    flat = pd.DataFrame({"A": [10.0] * 12}, index=idx)
    assert bool(trend_rising(flat).iloc[0]) is False  # 恒平 = 不升
    assert trend_rising(ramp.iloc[:1]) is None  # 行不足 fail-open


def _chip(wr=None):
    """水位标注夹具: 默认水位 NaN (cyq 缺); wr 按 symbol 覆盖."""
    syms = ["600001", "600002", "600003", "300005"]
    wl = wr or {}
    return pd.DataFrame({"symbol": syms, "wr": [wl.get(s, np.nan) for s in syms]})


def test_chip_level_label_split_direction_and_nan():
    """chip_level_label 纯判定 (四线同源切分): <0.5 低获利，涨 / ≥0.5 高获利，跌 /
    NaN 空。GENIOUS 也走此函数, 边界锁死防两表口径漂移。"""
    out = chip_level_label(pd.Series([0.23, 0.49, 0.50, 0.91, np.nan]))
    assert list(out) == [
        CHIP_FLAG_LOW,
        CHIP_FLAG_LOW,
        CHIP_FLAG_HIGH,
        CHIP_FLAG_HIGH,
        "",
    ]


def test_chip_gate_marks_distribution_direction():
    """水位标注 (0922 用户令): chip_wr≥0.5 → 高获利，跌, 不删票. 0.91 = 000001
    冒烟实测水位."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    chip = _chip(wr={"600001": 0.91})
    out = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert list(out["symbol"]) == [
        "600001",
        "600003",
        "300005",
    ]  # 600002 occ5=1 被密度剔
    assert out.loc[out.symbol == "600001", "chip_flag"].iloc[0] == CHIP_FLAG_HIGH
    assert out.loc[out.symbol == "600003", "chip_flag"].iloc[0] == ""  # 水位缺不标


def test_chip_gate_marks_low_level_as_rise():
    """chip_wr<0.5 → 低获利，涨 (浅获利, 0922 回测前向更强)."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    chip = _chip(wr={"600001": 0.23})
    out = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert "600001" in list(out["symbol"])  # 不删
    assert out.loc[out.symbol == "600001", "chip_flag"].iloc[0] == CHIP_FLAG_LOW


def test_chip_gate_failopen_none_and_nan():
    """fail-open: chip=None (cyq 数据缺) → 无标注列; 个股水位 NaN → 保留不标."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    out0 = density_picks(_cand(), _hist(), close, amount, DAY)
    out1 = density_picks(_cand(), _hist(), close, amount, DAY, chip=None)
    assert list(out0["symbol"]) == list(out1["symbol"])
    chip = _chip(wr={"600003": np.nan})
    out2 = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert "600003" in list(out2["symbol"])
    assert out2.loc[out2.symbol == "600003", "chip_flag"].iloc[0] == ""


def test_chip_gate_missing_wr_column_kept_all_blank():
    """旧 fixture 形态 (chip 无 wr 列, cyq 水位缺) / chip 空 DataFrame →
    列在但全空, 票全保留 (水位缺不许影响名单)."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    chip = _chip(wr={})
    out = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert list(out["symbol"]) == ["600001", "600003", "300005"]  # 600002 occ5=1 非闸剔
    assert (out["chip_flag"] == "").all()
    out2 = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip.iloc[:0])
    assert list(out2["symbol"]) == ["600001", "600003", "300005"]
    assert (out2["chip_flag"] == "").all()  # cyq 空 → 列在但全空


def test_apply_wr5_gate_marks_and_keeps_all_rows():
    """通用标注 (三线共享): 返回 (标注后 df, 有标注清单); 行数不变."""
    df = pd.DataFrame({"symbol": ["1", "2", "3", "4"]})
    chip = pd.DataFrame({"symbol": ["000001", "000003"], "wr": [0.91, 0.49]})
    out, flagged = apply_wr5_gate(df, chip)
    assert flagged == ["000001", "000003"]
    assert list(out["symbol"]) == ["1", "2", "3", "4"]  # 不删: 4 行还是 4 行
    assert out.loc[out.symbol == "1", "chip_flag"].iloc[0] == CHIP_FLAG_HIGH
    assert out.loc[out.symbol == "3", "chip_flag"].iloc[0] == CHIP_FLAG_LOW
    assert out.loc[out.symbol == "4", "chip_flag"].iloc[0] == ""  # 缺水位不标
    out2, flagged2 = apply_wr5_gate(df, None)
    assert flagged2 == [] and len(out2) == 4 and "chip_flag" not in out2.columns


def test_apply_wr5_gate_level_column():
    """0922 获利盘水位列: chip 带 wr → chip_wr 落值 + 按水位标 flag; 不带 →
    NaN 不炸 (旧 fixture 兼容)。"""
    df = pd.DataFrame({"symbol": ["1", "2"]})
    chip = pd.DataFrame({"symbol": ["000001", "000002"], "wr": [0.91, 0.23]})
    out, flagged = apply_wr5_gate(df, chip)
    assert abs(out.loc[out.symbol == "1", "chip_wr"].iloc[0] - 0.91) < 1e-12
    assert abs(out.loc[out.symbol == "2", "chip_wr"].iloc[0] - 0.23) < 1e-12
    assert out.loc[out.symbol == "1", "chip_flag"].iloc[0] == CHIP_FLAG_HIGH
    assert out.loc[out.symbol == "2", "chip_flag"].iloc[0] == CHIP_FLAG_LOW
    assert flagged == ["000001", "000002"]
    chip_old = pd.DataFrame({"symbol": ["000001"]})  # 无 wr 列
    out2, flagged2 = apply_wr5_gate(df, chip_old)
    assert out2["chip_wr"].isna().all() and flagged2 == []


def test_load_chip_features_values_nan_and_failopen(tmp_path, monkeypatch):
    """装载器: wr 取最新日; 个股缺行 → NaN; 文件缺失 → None; <6 行 → None."""
    import scripts._prob10_density_shadow as mod

    dates = pd.bdate_range("2026-08-24", periods=8)
    rows = []
    for i, d in enumerate(dates):
        rows.append((d, "600001", 0.60 + 0.02 * i))
        rows.append((d, "600002", 0.50))
    for d in dates[:3]:
        rows.append((d, "600003", 0.55))  # 近期缺行 → NaN
    fp = tmp_path / "cyq_panel.parquet"
    pd.DataFrame(rows, columns=["date", "symbol", "winner_ratio"]).to_parquet(fp)
    monkeypatch.setattr(mod, "CYQ_PATH", str(fp))
    out = load_chip_features(pd.Timestamp("2026-09-04"))
    r1 = out[out.symbol == "600001"].iloc[0]
    assert abs(r1["wr"] - 0.74) < 1e-12  # 0922 水位列 = T 日获利盘 (最新行)
    assert "wr5" not in out.columns  # 0922 午后用户令: wr5 列退役
    r3 = out[out.symbol == "600003"].iloc[0]
    assert pd.isna(r3["wr"])
    monkeypatch.setattr(mod, "CYQ_PATH", str(tmp_path / "nope.parquet"))
    assert load_chip_features(pd.Timestamp("2026-09-04")) is None


def test_empty_history_occ_never_passes():
    syms = ["600001", "300005"]
    close, amount = _panel(syms, [10.0] * 2, [2e8] * 2)
    out = density_picks(
        _cand(),
        pd.DataFrame(columns=["date", "board", "symbol", "prob"]),
        close,
        amount,
        DAY,
    )
    assert out.empty  # 无历史 → occ5<3 全剔 (影子冷启动由 bootstrap 兜底)


def test_empty_night_still_records_band_membership(tmp_path, monkeypatch, capsys):
    """[0913] 空夜也落带上榜史: 换模后 occ5 窗若冻结在旧模型带 (空夜不记史),
    新带成员 occ5 恒 0 → 永久空清单死锁. 榜=prob 带成员, 与闸过否无关."""
    import scripts._prob10_density_shadow as mod

    day = pd.Timestamp("2026-09-13")
    cand = pd.DataFrame(
        {
            "symbol": ["600011", "600012", "600013"],
            "board": ["main"] * 3,
            "prob_up_10d": [0.9, 0.8, 0.7],
            "pred_ret_10d": [0.1, 0.09, 0.08],
        }
    )
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    cand.to_parquet(lists_dir / "candidates_20260913.parquet")
    # 旧史全是无关旧码 → 今日新带 occ5=0 → 闸后必空
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2026-09-07", periods=5, freq="B").tolist() * 1,
            "board": ["main"] * 5,
            "symbol": ["600999"] * 5,
            "prob": [0.5] * 5,
        }
    )
    hist_fp = tmp_path / "hist.parquet"
    hist.to_parquet(hist_fp)
    idx = pd.date_range("2026-08-28", periods=12, freq="B")  # 末行 09-11 (≤ day)
    ramp = 10.0 + 0.5 * np.arange(12)  # 斜坡夹具沿用 (B 终选闸不动带史)
    panel = pd.DataFrame(
        {
            "symbol": ["600011"] * 12 + ["600012"] * 12 + ["600013"] * 12,
            "date": list(idx) * 3,
            "close_hfq": list(ramp) * 3,
            "amount": [2e8] * 36,
            "pctChg": [0.5] * 36,
        }
    )
    panel_fp = tmp_path / "panel.parquet"
    panel.to_parquet(panel_fp)

    monkeypatch.setattr(mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HIST_PATH", str(hist_fp))
    monkeypatch.setattr(mod, "STOCK_LIST_DIR", tmp_path)
    monkeypatch.setattr(mod, "PANEL_V3_PATH", str(panel_fp))
    monkeypatch.setattr(mod, "CYQ_PATH", str(tmp_path / "no_cyq.parquet"))

    monkeypatch.setattr(
        sys, "argv", ["_prob10_density_shadow.py", "20260913", "--gen-only"]
    )
    rc = mod.main()
    assert rc == 0
    assert "无票" in capsys.readouterr().out  # 空夜 fail-safe 照跳
    saved = pd.read_parquet(hist_fp)
    today_rows = saved[saved["date"] == day]
    assert set(today_rows["symbol"]) == {"600011", "600012", "600013"}  # 带成员已记
    assert (saved["symbol"] == "600999").any()  # 旧史保留 (dedup 不误删)


def test_trend_gate_final_stage_b_wired_in_main(tmp_path, monkeypatch, capsys):
    """[0914 用户拍板闸位 B] 趋势闸在终选: 带史/occ5 用原始带 (跌票也记史攒
    occ5), occ5≥3 后终选滤当日 MA10↑ — 刚拐头票当天即可出 (B 独有 301220)."""
    import scripts._prob10_density_shadow as mod

    day = pd.Timestamp("2026-09-13")
    cand = pd.DataFrame(
        {
            "symbol": ["600011", "600012"],
            "board": ["main"] * 2,
            "prob_up_10d": [0.9, 0.95],  # 跌票 prob 更高 (cls 头偏超卖画像)
            "pred_ret_10d": [0.1, 0.12],
        }
    )
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    cand.to_parquet(lists_dir / "candidates_20260913.parquet")
    # 两票近 4 个上榜日全勤 → occ5=5 双双过密度闸, 终选闸是唯一分拣者
    hist = pd.DataFrame(
        {
            "date": pd.date_range("2026-09-07", periods=4, freq="B").tolist() * 2,
            "board": ["main"] * 8,
            "symbol": ["600011"] * 4 + ["600012"] * 4,
            "prob": [0.88] * 4 + [0.93] * 4,
        }
    )
    hist_fp = tmp_path / "hist.parquet"
    hist.to_parquet(hist_fp)
    idx = pd.date_range("2026-08-28", periods=12, freq="B")  # 末行 09-11 (≤ day)
    ramp = 10.0 + 0.5 * np.arange(12)  # MA10↑
    crash = np.full(12, 10.0)  # 恒平后末日崩 5.0 → MA10 拐头向下
    crash[-1] = 5.0
    panel = pd.DataFrame(
        {
            "symbol": ["600011"] * 12 + ["600012"] * 12,
            "date": list(idx) * 2,
            "close_hfq": list(ramp) + list(crash),
            "amount": [2e8] * 24,
            "pctChg": [0.5] * 24,
        }
    )
    panel_fp = tmp_path / "panel.parquet"
    panel.to_parquet(panel_fp)

    monkeypatch.setattr(mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HIST_PATH", str(hist_fp))
    monkeypatch.setattr(mod, "STOCK_LIST_DIR", tmp_path)
    monkeypatch.setattr(mod, "PANEL_V3_PATH", str(panel_fp))
    monkeypatch.setattr(mod, "CYQ_PATH", str(tmp_path / "no_cyq.parquet"))
    monkeypatch.setattr(
        mod._deadzone_guard, "is_alarm", lambda line, date: (False, "test")
    )
    monkeypatch.setattr(
        sys, "argv", ["_prob10_density_shadow.py", "20260913", "--gen-only"]
    )
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "MA10↑ 趋势闸 (终选): 2 → 1" in out  # 闸动作日志: 跌票终选被滤
    saved = pd.read_parquet(hist_fp)
    today_rows = saved[saved["date"] == day]
    assert set(today_rows["symbol"]) == {"600011", "600012"}  # 带史=原始带, 跌票照记
    picks = pd.read_csv(
        tmp_path / "prob10dens_20260913__prob10dens.csv", dtype={"symbol": str}
    )
    assert list(picks["symbol"]) == ["600011"]  # 清单只剩升势票


def test_session_taken_from_data_not_filename(tmp_path, monkeypatch, capsys):
    """[0915 回归] 会话日取 candidates 内 date 列, 不取文件名/argv.

    病根: daily_pipeline 补跑时拿**墙钟日**命名 candidates_{D}.parquet, 数据却是最近
    一个交易日 —— 实见 20260823→08-21, 20260830→08-28, 20260913→09-11 (清一色
    "周日名/周五数据")。密度史上榜按文件名记日期 → 真 09-11 的带被记到周日 09-13
    名下; 而 occ5 的 win4 取自面板交易日历 (不含周日) → 该行**永远取不到**, 同时
    09-11 那一格永远贡献 0, occ5 恒够不到 3 → 清单自 09-10 起永久空。
    函数级冒烟即可复现, 不必跑全链 (生产实录见 tmp_t/_density_occ_diag_0915.py)。
    """
    import scripts._prob10_density_shadow as mod

    cand = pd.DataFrame(
        {
            "symbol": ["600021", "600022"],
            "board": ["main"] * 2,
            "prob_up_10d": [0.9, 0.8],
            "pred_ret_10d": [0.10, 0.09],
            "date": [pd.Timestamp("2026-09-11")] * 2,  # 数据日 = 周五
        }
    )
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    cand.to_parquet(lists_dir / "candidates_20260913.parquet")  # 文件名 = 周日

    hist_fp = tmp_path / "hist.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-09-10")] * 2,
            "board": ["main"] * 2,
            "symbol": ["600021", "600022"],
            "prob": [0.9, 0.8],
        }
    ).to_parquet(hist_fp)

    idx = pd.date_range("2026-08-28", periods=12, freq="B")  # 末交易日 09-11
    panel = pd.DataFrame(
        {
            "symbol": ["600021"] * 12 + ["600022"] * 12,
            "date": list(idx) * 2,
            "close_hfq": list(10.0 + 0.5 * np.arange(12)) * 2,
            "amount": [2e8] * 24,
            "pctChg": [0.5] * 24,
        }
    )
    panel_fp = tmp_path / "panel.parquet"
    panel.to_parquet(panel_fp)

    monkeypatch.setattr(mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "HIST_PATH", str(hist_fp))
    monkeypatch.setattr(mod, "STOCK_LIST_DIR", tmp_path)
    monkeypatch.setattr(mod, "PANEL_V3_PATH", str(panel_fp))
    monkeypatch.setattr(mod, "CYQ_PATH", str(tmp_path / "no_cyq.parquet"))
    monkeypatch.setattr(
        sys, "argv", ["_prob10_density_shadow.py", "20260913", "--gen-only"]
    )

    assert mod.main() == 0
    assert "会话日对齐" in capsys.readouterr().out

    saved = pd.read_parquet(hist_fp)
    # 非交易日 09-13 一行都不许落 (落了就是永不可读的死行)
    assert pd.Timestamp("2026-09-13") not in set(pd.to_datetime(saved["date"]))
    # 真会话 09-11 必须落盘, 否则 occ5 窗永远缺一格
    got = set(pd.to_datetime(saved[saved["symbol"] == "600021"]["date"]))
    assert pd.Timestamp("2026-09-11") in got
