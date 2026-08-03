# -*- coding: utf-8 -*-
"""预测准确率 A/B 回测 — 当前特征集 vs +Kimi holdertrade 特征.

目标 (用户: "个股上涨幅度和上涨概率 正确是第一位的"):
  - 当前可推理模型包特征集 = 基线
  - 加 Kimi 5 特征 (net_ratio / g / p / c / ratio_30d) = 对照
  - 时间切分留出段 (test = 最近 ~12 个月, 与训练无重叠 → 诚实 OOS)
  - 度量: 方向准确率 / MAE / bias / AUC / 概率校准 / TOP10 命中率

口径与生产一致: 清洗 run_train → FeatureEngineV35.build(inference_cols) →
LabelEngine.build_labels(label_pm_*_net) + mask_suspension + mask_recent_days.
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.pipeline1.cleaning_pipeline import CleaningPipeline  # noqa: E402
from app.pipeline1.dual_track_trainer import (  # noqa: E402
    LGB_PARAMS_CLS,
    LGB_PARAMS_REG,
    risk_filter,
)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.label_engine import LabelEngine  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
RAW = os.path.join(ROOT, "data", "_holder_cmp_raw.parquet")
BUNDLE = os.path.join(ROOT, "models", "pipeline1", "main_2026W31_real_tr.pkl")
# 聚焦实验 (用户: 短测试窗口 + 只在有目标特征[LHB]的股票上训练/测试):
#   股票池 = 主板上最近 TEST_WINDOW_DAYS 交易日内有 LHB (lhb_buy_amt) 的股票
#   训练   = 这些股票全历史 < TEST_START
#   测试   = 最后 TEST_WINDOW_DAYS 交易日 (短窗口, 带 252 交易日暖机)
TEST_WINDOW_DAYS = 120
WARMUP_DAYS = 252  # 引擎最长回看窗口 (年线特征), 测试段暖机必须覆盖
HORIZONS = (1, 3, 5)
KIMI_COLS = [
    "kimi_net_ratio",
    "kimi_g_ratio",
    "kimi_p_ratio",
    "kimi_c_ratio",
    "kimi_ratio_30d",
]


def _is_main_board(sym: str) -> bool:
    code = str(sym).split(".")[0]
    return code.startswith(("60", "000", "001", "002", "003"))


def _load_current_cols() -> list[str]:
    import pickle

    with open(BUNDLE, "rb") as fh:
        bundle = pickle.load(fh)
    return list(bundle["feature_cols"])


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 → float32 (16GB 机器上 260 列宽表 consolidate 会 OOM)."""
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


def _mem(label: str) -> None:
    if os.name == "posix":
        import resource

        print(
            f"  [mem] {label}: RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f}GB",
            flush=True,
        )
    else:
        import psutil

        print(
            f"  [mem] {label}: RSS={psutil.Process().memory_info().rss / 1e9:.2f}GB",
            flush=True,
        )


def _kimi_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """在已 build 的面板上加 Kimi 特征 (合并日聚合 + 每股 trailing 30d 滚动和)."""
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    for t in ("G", "P", "C"):
        raw["sr_" + t.lower()] = np.where(
            raw["holder_type"] == t, raw["signed_ratio"], 0.0
        )
    agg = (
        raw.groupby(["symbol", "date"], as_index=False)
        .agg(
            net_ratio=("signed_ratio", "sum"),
            g_ratio=("sr_g", "sum"),
            p_ratio=("sr_p", "sum"),
            c_ratio=("sr_c", "sum"),
        )
        .rename(
            columns={
                "net_ratio": "kimi_net_ratio",
                "g_ratio": "kimi_g_ratio",
                "p_ratio": "kimi_p_ratio",
                "c_ratio": "kimi_c_ratio",
            }
        )
    )
    df = df.merge(agg, on=["symbol", "date"], how="left")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    net = df["kimi_net_ratio"].fillna(0.0)
    df["kimi_ratio_30d"] = (
        net.groupby(df["symbol"])
        .rolling(30, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df


def _build_pass(
    main_df: pd.DataFrame,
    current: list[str],
    floor: pd.Timestamp,
    date_hi: pd.Timestamp | None = None,
    min_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """单段特征构建: 内存安全的关键是宽表行数受限于 [floor, date_hi) 切片.

    语义与整表 build 一致: 每个日期截面完整, 滚动特征在 SPLIT 边界已由
    floor(=SPLIT-252交易日) 提供完整暖机. 构建后立即裁剪到所需列 + float32.
    """
    mask = main_df["date"] >= floor
    if date_hi is not None:
        mask &= main_df["date"] < date_hi
    sub = main_df[mask].copy()
    print(
        f"  [pass] range={floor.date()}..{date_hi.date() if date_hi is not None else 'max'} "
        f"rows={len(sub)}",
        flush=True,
    )
    feats = FeatureEngineV35().build(sub, None, inference_cols=current)
    del sub
    gc.collect()
    missing = [c for c in current if c not in feats.columns]
    if missing:
        print(
            f"  [warn] 当前特征集 {len(missing)} 列引擎未复现 (补0): {missing[:8]}",
            flush=True,
        )
        for c in missing:
            feats[c] = 0.0
    for need in ("close_hfq", "amount"):
        if need not in feats.columns and need in main_df.columns:
            feats = feats.merge(
                main_df[["symbol", "date", need]], on=["symbol", "date"], how="left"
            )
    keep = ["symbol", "date"] + current + ["close_hfq", "amount", "is_suspended"]
    keep = [c for c in keep if c in feats.columns]
    for need in ("close_hfq", "amount", "is_suspended"):
        if need not in feats.columns and need in main_df.columns:
            feats = feats.merge(
                main_df[["symbol", "date", need]], on=["symbol", "date"], how="left"
            )
            keep.append(need)
    feats = feats[keep].copy()
    _downcast(feats)
    if min_date is not None:
        feats = feats[feats["date"] >= min_date]
    print(f"  [pass] kept rows={len(feats)} cols={len(keep)}", flush=True)
    _mem("pass")
    return feats


def _prepare():
    print("[1/4] 载入面板 + 清洗 run_train (main) ...", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    _dates = np.array(sorted(panel["date"].unique()))
    TEST_START = pd.Timestamp(_dates[-TEST_WINDOW_DAYS])
    WARMUP_START = pd.Timestamp(_dates[-TEST_WINDOW_DAYS - WARMUP_DAYS])
    print(
        f"  测试窗口: 最后 {TEST_WINDOW_DAYS} 交易日 = [{TEST_START.date()}, {_dates[-1].date()}]  "
        f"暖机起点 {WARMUP_START.date()}",
        flush=True,
    )
    # LHB 目标特征股票池: 主板上测试窗口内有 lhb_buy_amt 的股票
    test_dates = set(pd.Timestamp(d) for d in _dates[-TEST_WINDOW_DAYS:])
    has_lhb = panel["lhb_buy_amt"].notna() & panel["date"].isin(test_dates)
    universe = {s for s in panel.loc[has_lhb, "symbol"].unique() if _is_main_board(s)}
    print(
        f"  目标特征股票池 (主板 LHB {TEST_WINDOW_DAYS} 日): {len(universe)} 只",
        flush=True,
    )
    _mem("panel")
    cleaner = CleaningPipeline()
    main_df, _ = cleaner.run_train(panel, board="main")
    del panel
    gc.collect()
    main_df = main_df[main_df["symbol"].isin(universe)]
    print(
        f"  清洗+LHB池后 rows={len(main_df)} stocks={main_df['symbol'].nunique()}",
        flush=True,
    )
    _mem("cleaned")

    current = _load_current_cols()
    print(
        f"[2/4] 构建特征 inference_cols={len(current)} (训练<{TEST_START.date()}) ...",
        flush=True,
    )
    t0 = time.time()
    fa = _build_pass(
        main_df, current, pd.Timestamp(_dates[0]), date_hi=TEST_START
    )  # 训练全历史
    fb = _build_pass(
        main_df, current, WARMUP_START, min_date=TEST_START
    )  # 测试段 (暖机后)
    feats = pd.concat([fa, fb], ignore_index=True)
    del fa, fb, main_df
    gc.collect()
    print(f"  特征构建 {time.time() - t0:.1f}s, 合并后 rows={len(feats)}", flush=True)
    _mem("features")

    print("[3/4] 合并 Kimi 事件特征 + 标签 ...", flush=True)
    raw = pd.read_parquet(RAW)
    feats = _kimi_features(feats, raw)
    del raw
    _downcast(feats)
    for c in KIMI_COLS:
        nn = feats[c].notna().sum() if c in feats.columns else -1
        nz = (feats[c] != 0).sum() if c in feats.columns else -1
        print(f"  [diag] {c}: nonnull={nn} nonzero={nz}", flush=True)
    feats = LabelEngine.build_labels(feats)
    feats = LabelEngine.mask_suspension(feats)
    feats = LabelEngine.mask_recent_days(feats, days=6)
    _mem("labels")
    return feats, current, TEST_START


def _fit_model(rows, Xcols, kind, k, es_days: int = 40):
    y_col = f"label_pm_{k}d_net"
    data = rows.dropna(subset=[y_col])
    if len(data) < 5000:
        return None
    data = data.sort_values("date")
    # 尾部 es_days 作早停验证集 (与生产 es 段一致)
    es_cut = sorted(data["date"].unique())[-es_days]
    fit = data[data["date"] < es_cut]
    es = data[data["date"] >= es_cut]
    X = np.nan_to_num(fit[Xcols].to_numpy(dtype=float), nan=0.0)
    y = fit[y_col].to_numpy(dtype=float)
    X_es = np.nan_to_num(es[Xcols].to_numpy(dtype=float), nan=0.0)
    y_es = es[y_col].to_numpy(dtype=float)
    if kind == "cls":
        y = (y > 0).astype(int)
        y_es = (y_es > 0).astype(int)
        model = lgb.LGBMClassifier(**LGB_PARAMS_CLS)
    else:
        model = lgb.LGBMRegressor(**LGB_PARAMS_REG)
    model.fit(
        X,
        y,
        eval_set=[(X_es, y_es)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _eval(actual, pred, prob):
    m = {
        "n": len(actual),
        "dir_acc": np.nan,
        "mae": np.nan,
        "bias": np.nan,
        "auc": np.nan,
        "calib_err": np.nan,
        "hit50": np.nan,
        "hit55": np.nan,
        "hit60": np.nan,
        "mean_actual": float(np.mean(actual)),
        "std_actual": float(np.std(actual)),
    }
    if len(actual) < 50:
        return m
    m["dir_acc"] = float(np.mean(np.sign(pred) == np.sign(actual)))
    m["mae"] = float(np.mean(np.abs(pred - actual)))
    m["bias"] = float(np.mean(pred - actual))
    up = actual > 0
    if up.sum() >= 10 and (~up).sum() >= 10:
        try:
            m["auc"] = float(roc_auc_score(up, prob))
        except ValueError:
            pass
    dfp = pd.DataFrame({"p": prob, "y": up}).dropna()
    if len(dfp) >= 200:
        try:
            dfp["b"] = pd.qcut(dfp["p"], 10, duplicates="drop")
            g = dfp.groupby("b", observed=True).agg(
                mean_p=("p", "mean"), rate=("y", "mean")
            )
            m["calib_err"] = float(np.mean((g["rate"] - g["mean_p"]).abs()))
        except Exception:
            pass
    for t, key in ((0.5, "hit50"), (0.55, "hit55"), (0.60, "hit60")):
        sub = actual[prob >= t]
        if len(sub) >= 30:
            m[key] = float(np.mean(sub > 0))
    return m


def _topk(te, tr, Xcols, cache, tag):
    k = 5
    reg = cache[(tag, k, "reg")]
    cls = cache[(tag, k, "cls")]
    data = te.dropna(subset=["label_pm_5d_net"]).copy()
    if len(data) < 1000:
        return {}
    X = np.nan_to_num(data[Xcols].to_numpy(dtype=float), nan=0.0)
    data["p_ret5"] = reg.predict(X)
    data["p_prob5"] = cls.predict_proba(X)[:, 1] if cls is not None else np.nan
    res = {}
    all_hit = float(np.mean(data["label_pm_5d_net"] > 0))
    res["all"] = {"hit": all_hit, "mean": float(np.mean(data["label_pm_5d_net"]))}
    for topk in (5, 10, 20):
        hits, means, dhs = [], [], []
        for d, g in data.groupby("date"):
            top = g.nlargest(topk, "p_ret5")
            hits.append(float((top["label_pm_5d_net"] > 0).mean()))
            means.append(float(top["label_pm_5d_net"].mean()))
            g2 = g[g["p_prob5"] >= g["p_prob5"].quantile(0.7)]
            if len(g2) >= topk:
                dhs.append(float(g2.nlargest(topk, "p_ret5")["label_pm_5d_net"].mean()))
        res[f"top{topk}_ret"] = {
            "hit": float(np.mean(hits)),
            "mean": float(np.mean(means)),
        }
        res[f"top{topk}_bothhigh"] = {
            "hit": np.nan,
            "mean": float(np.mean(dhs)) if dhs else np.nan,
        }
    return res


def main() -> None:
    feats, current, TEST_START = _prepare()
    feats = feats.sort_values(["symbol", "date"]).reset_index(drop=True)
    feat_cols = [c for c in current if c in feats.columns]
    kimi_cols = [c for c in KIMI_COLS if c in feats.columns]
    print(
        f"基线特征 {len(feat_cols)}, +Kimi {len(kimi_cols)} 特征: {kimi_cols}",
        flush=True,
    )

    tr = risk_filter(feats[feats["date"] < TEST_START].copy())
    te = risk_filter(feats[feats["date"] >= TEST_START].copy())
    del feats
    print(
        f"[4/4] 切分 train={len(tr)} ({tr['date'].nunique()}日) test={len(te)} "
        f"({te['date'].nunique()}日) TEST_START={TEST_START.date()}",
        flush=True,
    )

    sets = {"BASE": feat_cols, "KIMI": feat_cols + kimi_cols}
    cache = {}
    for tag, Xcols in sets.items():
        for k in HORIZONS:
            for kind in ("reg", "cls"):
                t0 = time.time()
                cache[(tag, k, kind)] = _fit_model(tr, Xcols, kind, k)
                print(f"  fit {tag} {k}d_{kind} {time.time() - t0:.1f}s", flush=True)
    for k in HORIZONS:
        d = te.dropna(subset=[f"label_pm_{k}d_net"])
        xB = np.nan_to_num(d[sets["BASE"]].to_numpy(dtype=float), nan=0.0)
        xK = np.nan_to_num(d[sets["KIMI"]].to_numpy(dtype=float), nan=0.0)
        for kind in ("reg", "cls"):
            mb = cache[("BASE", k, kind)]
            mk = cache[("KIMI", k, kind)]
            if kind == "reg":
                pb, pk = mb.predict(xB), mk.predict(xK)
            else:
                pb, pk = mb.predict_proba(xB)[:, 1], mk.predict_proba(xK)[:, 1]
            print(
                f"  [diag] {k}d_{kind} max|BASE-KIMI pred|={np.abs(pb - pk).max():.3e} "
                f"corr={np.corrcoef(pb, pk)[0, 1]:.6f}",
                flush=True,
            )
    print("\n" + "=" * 102)
    print(
        f"聚焦 OOS 预测准确率 (测试窗口 {TEST_START.date()} 起, 主板 LHB 股票池, "
        f"仅训练有 LHB 特征的股票) — 基线(当前76特征) vs +Kimi holdertrade"
    )
    print("=" * 102)
    for k in HORIZONS:
        rows = []
        for tag in ("BASE", "KIMI"):
            m = cache[(tag, k, "reg")]
            c = cache[(tag, k, "cls")]
            data = te.dropna(subset=[f"label_pm_{k}d_net"])
            X = np.nan_to_num(data[sets[tag]].to_numpy(dtype=float), nan=0.0)
            ev = _eval(
                data[f"label_pm_{k}d_net"].to_numpy(dtype=float),
                m.predict(X),
                c.predict_proba(X)[:, 1],
            )
            rows.append(ev)
        b, m2 = rows[0], rows[1]
        print(
            f"\n--- {k}d   n={b['n']}  平均实际={b['mean_actual'] * 100:+.2f}%  std={b['std_actual'] * 100:.2f}% ---"
        )
        print(f"{'metric':<12s} {'BASE':>12s} {'+KIMI':>12s} {'Δ':>9s}")
        for metric, lab, is_pct in (
            ("dir_acc", "方向准确率", True),
            ("mae", "MAE", False),
            ("bias", "bias", False),
            ("auc", "AUC", True),
            ("calib_err", "校准误差", False),
            ("hit50", "命中@P≥0.5", True),
            ("hit55", "命中@P≥0.55", True),
            ("hit60", "命中@P≥0.6", True),
        ):
            bv, mv = b.get(metric, np.nan), m2.get(metric, np.nan)
            if np.isnan(bv):
                continue
            if is_pct:
                print(
                    f"{lab:<12s} {bv * 100:>11.2f}% {mv * 100:>11.2f}% {(mv - bv) * 100:>+8.2f}pp"
                )
            else:
                print(f"{lab:<12s} {bv:>12.4f} {mv:>12.4f} {(mv - bv):>+9.4f}")

    print("\n" + "=" * 102)
    print("TOP-K 买入清单视角 (5d): 每测试日按 pred_ret_5d 取 K 只")
    print("=" * 102)
    print(
        f"{'variant':<18s} {'BASE hit':>10s} {'+KIMI hit':>10s} {'Δhit':>8s} | {'BASE mean':>10s} {'+KIMI mean':>10s} {'Δmean':>8s}"
    )
    tb = _topk(te, tr, sets["BASE"], cache, "BASE")
    tk = _topk(te, tr, sets["KIMI"], cache, "KIMI")
    for key in (
        "all",
        "top5_ret",
        "top10_ret",
        "top20_ret",
        "top5_bothhigh",
        "top10_bothhigh",
        "top20_bothhigh",
    ):
        if key not in tb or key not in tk:
            continue
        bh, mh = tb[key].get("hit", np.nan), tk[key].get("hit", np.nan)
        bm, mm = tb[key].get("mean", np.nan), tk[key].get("mean", np.nan)
        print(
            f"{key:<18s} {bh * 100:>9.2f}% {mh * 100:>9.2f}% {(mh - bh) * 100:>+7.2f}pp | "
            f"{bm * 100:>9.3f}% {mm * 100:>9.3f}% {(mm - bm) * 100:>+7.3f}pp"
        )


if __name__ == "__main__":
    main()
