"""P19.0 阶段一质量门: 泄漏审计 / metrics / IC衰减 / 数据质量日检."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.dq_report import daily_report, drop_violations, ohlcv_violations
from app.pipeline1.ic_decay import ic_decay_curve
from app.pipeline1.leakage_audit import audit_source, ic_sentinel
from app.pipeline1.metrics import bucket_ic_high_vol, icir, ignition_gate, rank_ic


# ============================================================
# 泄漏审计
# ============================================================
class TestLeakageAudit:
    def test_source_scan_clean(self):
        src = 'df["ma5"] = df["close"].rolling(5).mean()\ndf["lag"] = df["close"].shift(1)'
        assert audit_source(src) == []

    def test_source_scan_forbidden(self):
        hits = audit_source('x = ZIG(close, 5)\ny = df["close"].shift(-1)')
        assert len(hits) == 2
        assert {h["pattern"] for h in hits} == {r"\bZIG\b", r"\.shift\(\s*-\d"}

    def test_source_scan_ignores_comments(self):
        assert audit_source("# 禁用 ZIG 函数") == []

    def test_ic_sentinel(self):
        dates = pd.bdate_range("2025-01-01", periods=60)
        rng = np.random.default_rng(3)
        rows = []
        for d in dates:
            f = rng.normal(size=30)
            for i in range(30):
                rows.append(
                    {
                        "date": d,
                        "leak": f[i],
                        "noise": rng.normal(),
                        "label_1d": f[i] + rng.normal(0, 0.01),
                    }
                )
        df = pd.DataFrame(rows)
        r = ic_sentinel(df, ["leak", "noise"])
        assert not r["pass"]
        assert "leak" in r["suspects"] and "noise" not in r["suspects"]

    def test_feature_engine_source_clean(self):
        """红线回归: 特征引擎源码不得含未来函数模式."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        hits = audit_source(
            (root / "app" / "pipeline1" / "feature_engine_v35.py").read_text(
                encoding="utf-8"
            ),
            "feature_engine_v35.py",
        )
        assert hits == []


# ============================================================
# metrics: Rank IC / ICIR / 点火门禁
# ============================================================
class TestMetrics:
    def _df(self, corr_noise=0.01, days=60, n=30, seed=7):
        dates = pd.bdate_range("2025-01-01", periods=days)
        rng = np.random.default_rng(seed)
        rows = []
        for d in dates:
            s = rng.normal(size=n)
            for i in range(n):
                rows.append(
                    {
                        "date": d,
                        "score": s[i],
                        "label": s[i] + rng.normal(0, corr_noise),
                    }
                )
        return pd.DataFrame(rows)

    def test_rank_ic_and_icir(self):
        df = self._df()
        assert rank_ic(df, "score", "label") > 0.9
        assert icir(df, "score", "label") > 5
        noise = self._df(corr_noise=100, seed=9)
        assert abs(rank_ic(noise, "score", "label")) < 0.1

    def test_ignition_gate(self):
        df = self._df()
        ok = ignition_gate(df, "score", "label", high_vol_ic=0.03, train_ic=0.10)
        assert ok["pass"]
        bad = ignition_gate(df, "score", "label", high_vol_ic=0.01, train_ic=0.20)
        assert not bad["pass"]
        assert not bad["checks"]["high_vol_ic"]["pass"]
        assert not bad["checks"]["train_ic_no_leak"]["pass"]

    def test_bucket_ic_high_vol_and_auto_gate(self):
        """E.4: bucket_ic_high_vol 直接取 Q5; ignition_gate 缺省自动分桶."""
        rng = np.random.default_rng(11)
        n = 500
        atr = rng.uniform(0.01, 0.10, n)
        score = atr + rng.normal(0, 0.01, n)  # score∝ATR → 高波动桶 IC 高
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-25"] * n),
                "ATR_pct": atr,
                "score": score,
                "label": score + rng.normal(0, 0.01, n),
            }
        )
        assert bucket_ic_high_vol(df, "score", "label") > 0.5
        # 单日期截面 → rank_ic/icir 为 0, 仅验证 high_vol_ic 自动注入检查键
        r = ignition_gate(df, "score", "label", train_ic=0.0)
        assert r["checks"]["high_vol_ic"]["value"] > 0.5
        assert r["checks"]["high_vol_ic"]["pass"]


# ============================================================
# IC 衰减曲线
# ============================================================
class TestICDecay:
    def test_decay_curve(self):
        # 价格由 f 驱动: t 日 f 影响 t+1 收益最强, 随后衰减
        days, n = 80, 20
        dates = pd.bdate_range("2025-01-01", periods=days)
        rng = np.random.default_rng(5)
        frames = []
        for s in range(n):
            f = rng.normal(size=days)
            ret = 0.01 * np.roll(f, 1) + rng.normal(0, 0.01, days)  # t+1 收益∝f
            close = 100 * np.cumprod(1 + ret)
            frames.append(
                pd.DataFrame(
                    {"symbol": f"S{s}", "date": dates, "close_hfq": close, "score": f}
                )
            )
        df = pd.concat(frames, ignore_index=True)
        r = ic_decay_curve(df, "score")
        assert r["ic_t+1"] > 0.3  # t+1 强相关
        assert r["ic_t+1"] > r["ic_t+3"]  # 单调衰减
        assert "fast_decay" in r


# ============================================================
# 数据质量日检 (OHLCV 铁律)
# ============================================================
class TestDQReport:
    def _panel(self, bad=False):
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "date": pd.to_datetime(["2026-07-25"] * 3),
                "open": [10.0, 20.0, 30.0],
                "high": [10.5, 20.5, 30.5],
                "low": [9.5, 19.5, 29.5],
                "close": [10.2, 20.2, 30.2],
                "volume": [1e6, 2e6, 3e6],
                "amount": [1e7, 2e7, 3e7],
            }
        )
        if bad:
            df.loc[1, "high"] = 19.0  # high < open/close (B 违规)
            df.loc[2, "volume"] = -1  # volume < 0 (C 违规)
        return df

    def test_clean_panel_passes(self):
        assert len(ohlcv_violations(self._panel())) == 0
        assert daily_report(self._panel())["pass"]

    def test_violations_not_silent(self):
        vio = ohlcv_violations(self._panel(bad=True))
        assert set(vio["symbol"]) == {"B", "C"}
        assert any("high<open" in v for v in vio["violation"])
        r = daily_report(self._panel(bad=True))
        assert not r["pass"] and r["n_ohlcv_violations"] == 2

    def test_drop_violations_explicit(self):
        df = drop_violations(self._panel(bad=True))
        assert list(df["symbol"]) == ["A"]  # 违规行被显式剔除

    def test_duplicate_and_missing_detected(self):
        df = self._panel()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # 重复键
        df.loc[0, "close"] = np.nan  # 关键字段缺失
        r = daily_report(df)
        assert not r["pass"]
        assert r["n_duplicate_keys"] == 1 and r["n_missing_key_cols"] == 1
