"""概率头密度版影子单纯函数单测 (2026-09-05 建, 2026-09-06 口径替换).

口径锁死 (2026-09-06 用户拍板 "把密度王替换成 L3 TOP20这条线", 勿静默改):
  名单 = 每板 prob 降序前20带 + 回撤闸 (-10%) + 带内密度 occ5≥3
  + 派发标注 (09-09 用户拍板 "派发不删, 清单标注": 获利盘5日回落 wr5<0 →
    chip_flag=派发 列, 不删票; 数据缺 fail-open);
  免额 (额不作闸, amt 仅展示列) — 09-06 拍板 "去额";
  belief_down = prob − 3个上榜日前 prob (标签列, 非闸)。
"""

import sys

import numpy as np
import pandas as pd

from scripts._prob10_density_shadow import (
    _COLS,
    CHIP_WR5_MAX,
    OCC_MIN,
    OCC_WIN,
    PULL_FLAG_MAX,
    PULL_FLOOR,
    TOP_N,
    TREND_MA10_GATE,
    apply_wr5_gate,
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
    # 信念降 = 今日候选 prob (0.90) − 3 个上榜日前 hist prob (0.88)
    d3 = sorted(hist["date"].unique())[-3]
    p_3ago = float(hist[(hist.symbol == "600001") & (hist.date == d3)]["prob"].iloc[0])
    assert p_3ago == 0.88
    assert abs(r1["belief_down"] - (0.90 - p_3ago)) < 1e-9


def test_constants_locked():
    # 0913 用户令撤回撤闸: PULL_FLOOR −0.10→−1.0 等效关闭, 原闸档降为标注线
    assert (TOP_N, PULL_FLOOR, OCC_WIN, OCC_MIN) == (20, -1.0, 5, 3)
    assert PULL_FLAG_MAX == -0.10  # 0913 撤删改标: 标注线=原闸档
    assert CHIP_WR5_MAX == 0.0  # wr5<0 即标派发 (09-09 标注口径; 原 cost5 组合条件废除)
    assert TREND_MA10_GATE is True  # 0914 用户拍板闸位 B: occ5 后终选滤当日 MA10↑
    assert "amt" in _COLS  # 免额后 amt 保留为展示列 (不作闸)
    assert {"chip_wr5", "chip_flag", "pull_flag"} <= set(
        _COLS
    )  # 09-09 派发 + 0913 回撤标注列交付


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


def _chip(**over):
    """派发标注夹具: 默认全股健康 (获利盘5日升); over 按 symbol 覆盖 wr5."""
    syms = ["600001", "600002", "600003", "300005"]
    rows = [(s, over.get(s, 0.05)) for s in syms]
    return pd.DataFrame(rows, columns=["symbol", "wr5"])


def test_chip_gate_marks_distribution_direction():
    """派发标注 (09-09 用户拍板 "派发不删, 清单标注"): 获利盘5日回落 wr5<0 →
    chip_flag=派发, 不删票. 特征值 = 002098 0903 实测."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    chip = _chip(**{"600001": -0.058})
    out = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert list(out["symbol"]) == [
        "600001",
        "600003",
        "300005",
    ]  # 600002 occ5=1 被密度剔
    assert out.loc[out.symbol == "600001", "chip_flag"].iloc[0] == "派发"
    assert out.loc[out.symbol == "600003", "chip_flag"].iloc[0] == ""


def test_chip_gate_marks_wr5_dip_without_cost_rise():
    """000980形 (获利盘暴降而成本没上移, 0904 wr5=-0.230): 标注口径下单条件即标
    — 本测试锁死该行为 (不因改标注丢信号)."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    chip = _chip(**{"600001": -0.230})
    out = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert "600001" in list(out["symbol"])  # 不删
    assert out.loc[out.symbol == "600001", "chip_flag"].iloc[0] == "派发"
    assert abs(out.loc[out.symbol == "600001", "chip_wr5"].iloc[0] + 0.230) < 1e-12


def test_chip_gate_failopen_none_and_nan():
    """fail-open: chip=None (cyq 数据缺) → 无标注列; 个股特征 NaN → 保留不标."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    out0 = density_picks(_cand(), _hist(), close, amount, DAY)
    out1 = density_picks(_cand(), _hist(), close, amount, DAY, chip=None)
    assert list(out0["symbol"]) == list(out1["symbol"])
    chip = _chip(**{"600003": np.nan})
    out2 = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert "600003" in list(out2["symbol"])
    assert out2.loc[out2.symbol == "600003", "chip_flag"].iloc[0] == ""


def test_chip_gate_single_sided_direction_kept():
    """获利盘5日在升 (wr5>0, 高位接盘回放反向) / 边界恰好取等 wr5=0
    (严格不等号, 不标) / chip 空 DataFrame (cyq 空) → 全保留无标."""
    syms = ["600001", "600002", "600003", "300005"]
    close, amount = _panel(syms, [10.0] * 4, [2e8] * 4)
    chip = _chip(**{"600003": 0.05, "300005": 0.0})
    out = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip)
    assert list(out["symbol"]) == ["600001", "600003", "300005"]  # 600002 occ5=1 非闸剔
    assert (out["chip_flag"] == "").all()  # wr5=0 严格不等号 → 不标
    out2 = density_picks(_cand(), _hist(), close, amount, DAY, chip=chip.iloc[:0])
    assert list(out2["symbol"]) == ["600001", "600003", "300005"]
    assert (out2["chip_flag"] == "").all()  # cyq 空 → 列在但全空


def test_apply_wr5_gate_marks_and_keeps_all_rows():
    """通用标注 (三线共享): 返回 (标注后 df, 被标清单); 行数不变."""
    df = pd.DataFrame({"symbol": ["1", "2", "3", "4"]})
    chip = pd.DataFrame({"symbol": ["000001", "000003"], "wr5": [-0.10, -0.02]})
    out, flagged = apply_wr5_gate(df, chip)
    assert flagged == ["000001", "000003"]
    assert list(out["symbol"]) == ["1", "2", "3", "4"]  # 不删: 4 行还是 4 行
    assert out.loc[out.symbol == "1", "chip_flag"].iloc[0] == "派发"
    assert out.loc[out.symbol == "4", "chip_flag"].iloc[0] == ""  # 缺特征不标
    out2, flagged2 = apply_wr5_gate(df, None)
    assert flagged2 == [] and len(out2) == 4 and "chip_flag" not in out2.columns


def test_load_chip_features_values_nan_and_failopen(tmp_path, monkeypatch):
    """装载器: wr5 计算正确; 个股缺行 → NaN; 文件缺失 → None; <6 行 → None."""
    import scripts._prob10_density_shadow as mod

    dates = pd.bdate_range("2026-08-24", periods=8)
    rows = []
    for i, d in enumerate(dates):
        rows.append((d, "600001", 0.60 + 0.02 * i))
        rows.append((d, "600002", 0.50))
    for d in dates[:3]:
        rows.append((d, "600003", 0.55))  # 近 5 日缺行 → NaN
    fp = tmp_path / "cyq_panel.parquet"
    pd.DataFrame(rows, columns=["date", "symbol", "winner_ratio"]).to_parquet(fp)
    monkeypatch.setattr(mod, "CYQ_PATH", str(fp))
    out = load_chip_features(pd.Timestamp("2026-09-04"))
    r1 = out[out.symbol == "600001"].iloc[0]
    assert abs(r1["wr5"] - 0.10) < 1e-12  # 0.74 − 0.64
    assert "cost5" not in out.columns  # 09-05 升级: 仅 wr5
    r3 = out[out.symbol == "600003"].iloc[0]
    assert pd.isna(r3["wr5"])
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
