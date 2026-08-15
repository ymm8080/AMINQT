"""app/pipeline_parallel/prob_head.py — 并行真模型概率头 + 边际闸 (2026-08-15 定案).

背景 (memory parallel-gbm-wf-verdict): 并行闸空间穷尽后唯一成立的真模型闸.
全局 LGBM 概率头 (mfe_3d >= abs_target 二分类) 在并行特征空间训练, 短名单侧
边际闸 保留 ⇔ pred_prob > base_rate + margin (legacy 配方, 扩窗训练).
回测 250d OOS: dual 命中 68→70% / 实得 +8.06→+8.82%; main 60→61% / +3.63→+4.08%,
双板 4/4 子窗实得赢. trailing 242d 训练=数据饥饿退化, 勿用.

bundle (WORM, data/prob_head/<board>_prob_<ts>.joblib):
  {board, trained_through ("YYYY-MM-DD"), feat_cols, params, model}
训练 = 全史扩窗 (行 <= 面板最新日), 每 refit_every_days 交易日重训一次 (训练脚本自判断);
预测 = bundle.model 在当日截面特征上 predict_proba[:, 1];
base_rate = 最近 base_rate_days 个可观测日 mfe 达标率均值 (无前瞻, 当日可观测 mfe
只到 latest-4 — mfe_3d 窗口需 +4 交易日未来价, 见 _add_mfe_3d).
失败模式: bundle 缺失/过旧/特征缺列/当日截面为空 → 大声告警 + 闸失效 (fail-open,
不杀每日清单); 特征缺列是面板 schema 漂移, 属严重错误, predict 直接 raise.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier

from app.pipeline1.label_engine import COST, slippage_tier
from config.settings import DATA_DIR, PROB_GATE

META = {
    "symbol", "date", "board", "is_suspended", "name", "code", "exec_px",
}
RAW_COLS = {
    "open", "high", "low", "close", "open_hfq", "high_hfq", "low_hfq", "close_hfq",
    "volume", "amount", "pre_close", "turnover_rate", "total_mv", "adv20",
}
# 与回测 _diag_parallel_gbm_signal.py 完全一致 (阶段1/2 验证配方)
LGB_PARAMS = dict(
    objective="binary",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def bundle_dir() -> Path:
    return Path(PROB_GATE["model_dir"])


def _add_mfe_3d(df: pd.DataFrame) -> pd.DataFrame:
    """生产口径 mfe_3d = 窗口(T+2..T+4)最高价 / T+1 买入价 - 1 - cost (同回测)."""
    g = df.groupby("symbol", sort=False)
    exec_px = g["close_hfq"].shift(-1)
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, 5)], axis=1, keys=range(2, 5)
    )
    slip = df["adv20"].map(slippage_tier)
    cost_total = COST + 2 * slip
    df["mfe_3d"] = shifts.max(axis=1, skipna=False) / exec_px - 1 - cost_total
    return df


def feature_cols(t: pd.DataFrame) -> list[str]:
    """并行特征空间: 数值列, 剔 raw 价格量额/meta/label/pred/派生列 (同回测口径)."""
    excluded = {
        "symbol", "date", "board", "score", "mfe_3d", "label_pain",
        "label_pm_3d_net", "label_pm_10d_net",
    }
    return [
        c
        for c in t.columns
        if c not in excluded
        and c not in RAW_COLS
        and not c.startswith("label_")
        and not c.startswith("pred_")
        and pd.api.types.is_numeric_dtype(t[c].dtype)
    ]


def train_bundle(board: str, t: pd.DataFrame, trained_through: str) -> Path:
    """全史扩窗训练概率头 → WORM bundle. t 需含全部特征 + mfe_3d + label_pain.

    trained_through = 训练数据覆盖到的最后交易日 ("YYYY-MM-DD", 面板最新日);
    行过滤与回测同口径: mfe_3d 非 NaN 且 label_pain 非 NaN (mfe 尾段 NaN 不可训练).
    """
    cols = feature_cols(t)
    y = (t["mfe_3d"] >= PROB_GATE["abs_target"]).astype(float)
    ok = y.notna() & t["label_pain"].notna()
    x = t.loc[ok, cols].to_numpy(dtype="float32")
    if len(x) < 5000:
        raise ValueError(f"[{board}] 训练样本不足 ({len(x)})")
    model = LGBMClassifier(**LGB_PARAMS)
    model.fit(x, y.loc[ok].to_numpy())
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = bundle_dir() / f"{board}_prob_{ts}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "board": board,
            "trained_through": trained_through,
            "feat_cols": cols,
            "params": dict(LGB_PARAMS),
            "model": model,
        },
        path,
    )
    return path


def load_latest(board: str) -> dict | None:
    """该板块最新 WORM bundle (joblib dict); 无 → None."""
    cands = sorted(bundle_dir().glob(f"{board}_prob_*.joblib"))
    if not cands:
        return None
    return joblib.load(cands[-1])


def bundle_age_trading_days(panel_dates: np.ndarray, trained_through: str) -> int | None:
    """bundle 年龄 = 面板最新日与 trained_through 之间的交易日数; 未对齐 → None."""
    tt = pd.Timestamp(trained_through)
    if tt.to_datetime64() not in panel_dates:
        return None
    pos = int(np.searchsorted(panel_dates, tt.to_datetime64()))
    return int(len(panel_dates) - 1 - pos)


def predict(bundle: dict, df: pd.DataFrame) -> pd.Series:
    """当日截面特征 → pred_prob Series (index 对齐 df). 特征缺列 → raise (schema 漂移)."""
    cols = bundle["feat_cols"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"面板缺 {len(missing)} 个概率头特征列 (schema 漂移): {missing[:10]}"
        )
    x = df[cols].to_numpy(dtype="float32")
    return pd.Series(bundle["model"].predict_proba(x)[:, 1], index=df.index)


def _base_rate(tail: pd.DataFrame) -> float | None:
    """最近 base_rate_days 个可观测日 mfe 达标率均值 (无前瞻).

    当日可观测 mfe_3d 只到 latest-4 (窗口需 +4 交易日未来价) → 取可观测尾部
    的最近 base_rate_days 个非 NaN 日达标率求均值; 不足 → None (闸失效).
    注意: NaN >= 阈值 在 pandas 里是 False, 必须先剔 NaN 行再比较, 否则尾 4 行
    不可观测日被当 0% 达标拉低 base_rate.
    """
    tail = _add_mfe_3d(tail)
    tail = tail[tail["mfe_3d"].notna()]
    hit = (tail["mfe_3d"] >= PROB_GATE["abs_target"]).groupby(tail["date"]).mean()
    if len(hit) < PROB_GATE["base_rate_days"]:
        return None
    return float(hit.tail(PROB_GATE["base_rate_days"]).mean())


def gate_probabilities(board: str) -> tuple[pd.Series, float] | None:
    """当日截面每股 pred_prob + base_rate 最新值; 不可用 → None (已大声告警).

    读 _diag_stage_{board}_3y.parquet (parallel 检查点, 与短名单同源):
    - base_rate 用窄尾读 (symbol/date/close/high/adv20, 近 ~base_rate_days+14 日)
    - pred_prob 用当日截面读 (symbol + bundle feat_cols)
    """
    b = load_latest(board)
    if b is None:
        print(f"[prob_head] {board} 无概率头 bundle -> 闸不可用", flush=True)
        return None
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    dates = pd.to_datetime(
        pq.read_table(str(fp), columns=["date"]).to_pandas()["date"]
    )
    uniq = np.unique(dates.values)
    if len(uniq) < PROB_GATE["base_rate_days"] + 20:
        print(f"[prob_head] {board} 面板日期不足 -> 闸不可用", flush=True)
        return None
    age = bundle_age_trading_days(uniq, str(b["trained_through"]))
    if age is None or age > PROB_GATE["max_stale_days"]:
        print(
            f"[prob_head] {board} bundle 年龄 {age} 交易日 > "
            f"{PROB_GATE['max_stale_days']} -> 闸不可用 (请重训概率头)",
            flush=True,
        )
        return None
    latest = pd.Timestamp(uniq[-1])
    cutoff = pd.Timestamp(uniq[-PROB_GATE["base_rate_days"] - 14])
    tail = pq.read_table(
        str(fp),
        columns=["symbol", "date", "close_hfq", "high_hfq", "adv20"],
        filters=[("date", ">=", cutoff)],
    ).to_pandas()
    tail["symbol"] = tail["symbol"].astype(str)
    tail["date"] = pd.to_datetime(tail["date"])
    base = _base_rate(tail)
    if base is None:
        print(f"[prob_head] {board} base_rate 可观测样本不足 -> 闸不可用", flush=True)
        return None
    cs = pq.read_table(
        str(fp),
        columns=["symbol"] + list(b["feat_cols"]),
        filters=[("date", "==", latest)],
    ).to_pandas()
    if cs.empty:
        print(f"[prob_head] {board} 当日截面为空 -> 闸不可用", flush=True)
        return None
    cs["symbol"] = cs["symbol"].astype(str)
    pred = predict(b, cs)
    return pd.Series(pred.to_numpy(), index=cs["symbol"]), base


def apply_prob_gate(res: pd.DataFrame) -> pd.DataFrame:
    """真模型概率闸 (2026-08-15 定案): t3 门后、pred_mag_10d TOP-5 排名前.

    保留 ⇔ pred_prob > base_rate + margin. bundle 缺失/过旧/当日不可用 →
    fail-open (保留) + 大声告警 (不杀清单); 个股 pred_prob 缺失 → fail-open 保留.
    """
    cfg = PROB_GATE
    if not cfg.get("enable", True):
        return res
    out = res.copy()
    for board in ("main", "dual"):
        mask = out["board"] == board
        if not mask.any():
            continue
        got = gate_probabilities(board)
        if got is None:
            print(f"[prob_gate] {board} 概率头不可用 -> 闸失效 (fail-open)", flush=True)
            continue
        prob, base = got
        thr = base + cfg["margin"]
        p = out.loc[mask, "symbol"].astype(str).map(prob)
        keep = (p > thr) | p.isna()
        n_drop = int((~keep).sum())
        if n_drop:
            dropped = out.loc[mask & ~keep, "symbol"].astype(str).tolist()
            print(
                f"[prob_gate] {board} 剔除 {n_drop} 只 (pred_prob≤{thr:.1%}, "
                f"base_rate {base:.1%}): {', '.join(dropped)}",
                flush=True,
            )
        else:
            print(
                f"[prob_gate] {board} 全过 (pred_prob>{thr:.1%}, base_rate {base:.1%})",
                flush=True,
            )
        out = out.loc[~mask | keep].reset_index(drop=True)
    return out
