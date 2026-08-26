"""generate_columns 族级预过滤 ≡ 旧全族白算路径 (2026-08-15 提速修复).

根因: train_runner 后注入对 BRUTE_FAMILIES 7 族全循环, generate_columns 对每个
symbol 先算全族再挑 need 交集 — need 与该族零交集时 (空族) 白算全族,
20260812 main 注入 85 特征被空族吃掉 ~4h50m (日志 09:33→13:48 静默).

修复: per-symbol 循环前用静态候选列名 (raw × windows, 与 _family_for_symbol
命名规则逐字节一致, 含 rolling_max 产 max+min 双列特例) 做 need ∩ 族 == ∅
短路. 静态集是任何 symbol 产出的精确上界 (per-symbol 只做减法), 短路判定
永不误杀.

等价性铁律: 非短路族的输出必须与旧路径逐字节一致 (列名+值), 否则注入漂移
→ 训练特征改变 → 模型质量回归.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator

_RAW = ["close", "open", "high", "low", "volume"]


def _synth_df(n_dates=40, n_symbols=4, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_symbols):
        sym = f"{600000 + s}"
        price = 10 + s + rng.randn(n_dates).cumsum() * 0.5
        vol = rng.rand(n_dates) * 1e5
        for i in range(n_dates):
            d = pd.Timestamp("2025-01-02") + pd.Timedelta(days=i)
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "close": price[i],
                    "open": price[i] - 0.1,
                    "high": price[i] + 0.2,
                    "low": price[i] - 0.2,
                    "volume": vol[i],
                }
            )
    df = pd.DataFrame(rows)
    df["close"].iloc[5] = np.inf
    df["volume"].iloc[12] = -np.inf
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def _count_wrapper(gen):
    """包一层 _family_columns_vec 计数, 验证短路族不进向量化内核 (零计算).

    2026-08-25 起族计算走 _family_columns_vec (向量化, 免逐 symbol 循环),
    计数锚点从 _family_for_symbol (每 symbol 1 次) 换到内核 (每族至多 1 次).
    """
    calls = {"n": 0}
    orig = gen._family_columns_vec

    def counted(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    gen._family_columns_vec = counted
    return calls


class TestGenerateColumnsPrefilter:
    def test_static_candidates_cover_all_actual_columns(self):
        """不变量: 每族静态候选集 ⊇ 该族实际产出列 (短路判定的安全前提)."""
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        for fam in BRUTE_FAMILIES:
            candidates = gen._family_candidate_names(fam, raw_cols)
            assert candidates is not None, f"{fam}: 候选集缺失"
            actual = set(gen.generate_family(df, fam, raw_cols=raw_cols).columns)
            assert actual <= candidates, (
                f"{fam}: 静态候选缺列 {sorted(actual - candidates)}"
            )

    def test_zero_intersection_family_skips_per_symbol_work(self):
        """need 只含 pct 族列 → 其余 6 族返回 None 且零内核计算."""
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        need = {"close_brute_pct1", "close_brute_pct5", "volume_brute_pct10"}
        calls = _count_wrapper(gen)
        for fam in BRUTE_FAMILIES:
            new = gen.generate_columns(df, fam, need, raw_cols=raw_cols)
            if fam == "pct_change":
                assert new is not None, "pct_change 与 need 有交集, 不应短路"
                assert set(new.columns) == need, "只应驻留 need 交集列"
            else:
                assert new is None, f"{fam}: 零交集应短路返回 None"
        # 白算路径: 7 族各进 1 次内核; 短路后只 pct_change 进 1 次
        assert calls["n"] == 1, f"内核计算应只发生 1 次, 实际 {calls['n']}"

    def test_overlap_family_output_byte_identical_to_materialized(self):
        """有交集族: 输出 = generate_family 物化宽帧的 need 子集 (逐字节一致)."""
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        for fam in BRUTE_FAMILIES:
            full = gen.generate_family(df, fam, raw_cols=raw_cols, dtype="float32")
            need = set(list(full.columns)[:5])
            new = gen.generate_columns(df, fam, need, raw_cols=raw_cols)
            assert new is not None, f"{fam}: 有交集族被误短路"
            assert set(new.columns) == need, f"{fam}: 驻留列漂移"
            for c in need:
                np.testing.assert_array_equal(
                    new[c].to_numpy(dtype=np.float32),
                    full[c].to_numpy(dtype=np.float32),
                    err_msg=f"{fam}/{c}: 数值漂移 (含 inf→nan 处理)",
                )

    def test_empty_need_short_circuits_all_families(self):
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        calls = _count_wrapper(gen)
        for fam in BRUTE_FAMILIES:
            assert gen.generate_columns(df, fam, set(), raw_cols=raw_cols) is None
        assert calls["n"] == 0

    def test_rolling_max_need_hits_both_max_and_min_names(self):
        """rolling_max 族产出 max+min 双列: need 只含 min 列也必须跑该族."""
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        need = {"close_brute_min10"}
        new = gen.generate_columns(df, "rolling_max", need, raw_cols=raw_cols)
        assert new is not None
        assert set(new.columns) == need
