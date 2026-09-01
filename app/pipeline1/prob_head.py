"""app/pipeline1/prob_head.py — legacy 并行式概率头 + 边际闸 (2026-08-15 代码先行).

状态: 已落地 (2026-08-17) — bundle 已训练 (data/prob_head_legacy/, 首训 2026-08-16),
list_generator.emit + daily_pipeline._prob_gate_inputs 已接线, 自动化步骤
"legacy_prob_head" (run_daily_automation) 每 refit_every_days 交易日自判断重训;
端到端验证见 scripts/_diag_legacy_prob_gate_verify.py.
背景: legacy cls 概率头太粗 (闸内 22 唯一值 → blend 排名键 A/B 证伪, memory
legacy-blend-rank-verdict); 用户定案 legacy 建并行式全局 LGBM 概率头.
配方镜像 app/pipeline_parallel/prob_head.py (250d OOS 定案: 扩窗训练, mfe_3d >=
abs_target 二分类, 保留 ⇔ pred_prob > base_rate + margin). 与并行版唯一差异=数据流:
legacy 无 stage 特征检查点 → 特征由调用方构建 (FeatureEngineV35.build, 训练脚本
scripts/_train_legacy_prob_head.py 现场构建, 与 _diag_legacy_hitrate_topn 同构).
排名键保持纯 pred_ret_10d (legacy blend 证伪, pred_prob 列仅诊断用).

bundle (WORM, data/prob_head_legacy/<board>_prob_<ts>.joblib):
  {board, trained_through ("YYYY-MM-DD"), feat_cols, params, model}
训练 = 全史扩窗 (行 <= 面板最新日), 每 refit_every_days 交易日重训一次 (训练脚本自判断);
预测 = bundle.model 在当日截面特征上 predict_proba[:, 1];
base_rate = 最近 base_rate_days 个可观测日 mfe 达标率均值 (无前瞻, 当日可观测 mfe
只到 latest-4 — mfe_3d 窗口需 +4 交易日未来价, 见 _add_mfe_3d).
失败模式: bundle 缺失/过旧/特征缺列/当日截面为空 → 大声告警 + 闸失效 (fail-open,
不杀每日清单); 特征缺列是面板 schema 漂移, 属严重错误, predict 直接 raise.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, callback

from app.pipeline1.label_engine import COST, slippage_tier
from config.settings import LEGACY_PROB_GATE

META = {
    "symbol",
    "date",
    "board",
    "is_suspended",
    "name",
    "code",
    "exec_px",
}
RAW_COLS = {
    "open",
    "high",
    "low",
    "close",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "volume",
    "amount",
    "pre_close",
    "turnover_rate",
    "total_mv",
    "adv20",
}
# 生产 board 命名兼容: list_generator 双创=GEM/STAR, 内部=dual (同 model_meta.BOARD_TO_TRACK)
_BOARD_GROUP = {"main": "main", "dual": "dual", "GEM": "dual", "STAR": "dual"}

# ③+④ (2026-08-22 定案, WORM 153820): lr 0.03 + n800 + early stop + 地板 50
# (与并行概率头完全一致; ES 见 _fit_with_es, 地板防 dual 短验证窗塌缩)
LGB_PARAMS = dict(
    objective="binary",
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=800,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def bundle_dir() -> Path:
    return Path(LEGACY_PROB_GATE["model_dir"])


def _add_mfe_3d(df: pd.DataFrame) -> pd.DataFrame:
    """生产口径 mfe_3d = 窗口(T+2..T+4)最高价 / T+1 买入价 - 1 - cost (同并行)."""
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
    """legacy V35 特征空间: 数值列, 剔 raw 价格量额/meta/label/pred/派生列 (同并行)."""
    excluded = {
        "symbol",
        "date",
        "board",
        "score",
        "mfe_3d",
        "label_pain",
        "label_pm_3d_net",
        "label_pm_10d_net",
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


def _fit_with_es(
    board: str,
    model: LGBMClassifier,
    x: np.ndarray,
    y: np.ndarray,
    row_dates: np.ndarray,
) -> LGBMClassifier:
    """③+④ early stop + 地板 (08-22 定案, WORM 153820, 镜像 lab _es_fit).

    验证集 = 训练行尾部 val_days 个交易日 (无前瞻); val 样本 < 50 → 无早停普通拟合.
    早停树数 < es_floor → 固定 es_floor 树重训 (无早停), 防短验证窗早停塌缩成常数.
    """
    cfg = LEGACY_PROB_GATE
    if not cfg.get("es", False):
        model.fit(x, y)
        return model
    dates = np.unique(row_dates)
    val_days = min(int(cfg["val_days"]), len(dates) - 1)
    val_mask = np.isin(row_dates, dates[-val_days:])
    if val_mask.sum() < 50:
        model.fit(x, y)
        return model
    tr_mask = ~val_mask
    model.fit(
        x[tr_mask],
        y[tr_mask],
        eval_X=x[val_mask],
        eval_y=y[val_mask],
        callbacks=[
            callback.early_stopping(int(cfg["es_patience"])),
            callback.log_evaluation(0),
        ],
    )
    floor = int(cfg.get("es_floor", 0))
    if floor > 0:
        bi = getattr(model, "best_iteration_", None)
        if bi is not None and bi < floor:
            print(
                f"[prob_head] {board} 早停 {bi} 树 < 地板 {floor} → 固定 {floor} 树重训",
                flush=True,
            )
            fresh = model.__class__(**model.get_params())
            fresh.set_params(n_estimators=floor)
            fresh.fit(x, y)
            return fresh
    return model


def train_bundle(board: str, t: pd.DataFrame, trained_through: str) -> Path:
    """全史扩窗训练概率头 → WORM bundle. t 需含全部特征 + mfe_3d + label_pain.

    trained_through = 训练数据覆盖到的最后交易日 ("YYYY-MM-DD", 面板最新日);
    行过滤与并行同口径: mfe_3d 非 NaN 且 label_pain 非 NaN (mfe 尾段 NaN 不可训练).
    """
    cols = feature_cols(t)
    y = (t["mfe_3d"] >= LEGACY_PROB_GATE["abs_target"]).astype(float)
    ok = y.notna() & t["label_pain"].notna()
    x = t.loc[ok, cols].to_numpy(dtype="float32")
    if len(x) < 5000:
        raise ValueError(f"[{board}] 训练样本不足 ({len(x)})")
    model = LGBMClassifier(**LGB_PARAMS)
    model = _fit_with_es(
        board, model, x, y.loc[ok].to_numpy(), t.loc[ok, "date"].to_numpy()
    )
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


def bundle_age_trading_days(
    panel_dates: np.ndarray, trained_through: str
) -> int | None:
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
    hit = (
        (tail["mfe_3d"] >= LEGACY_PROB_GATE["abs_target"]).groupby(tail["date"]).mean()
    )
    if len(hit) < LEGACY_PROB_GATE["base_rate_days"]:
        return None
    return float(hit.tail(LEGACY_PROB_GATE["base_rate_days"]).mean())


def gate_probabilities(
    board: str,
    feat_day: pd.DataFrame,
    tail: pd.DataFrame,
    panel_dates: np.ndarray,
) -> tuple[pd.Series, float] | None:
    """当日截面每股 pred_prob + base_rate 最新值; 不可用 → None (已大声告警).

    legacy 数据流 (与并行版差异): 无 stage 特征检查点 → 全部由调用方传入 —
    - feat_day = 当日截面 V35 特征帧 (symbol + bundle feat_cols), 调用方构建
    - tail = 面板尾 (symbol/date/close_hfq/high_hfq/adv20, 近 ~base_rate_days+14 交易日)
    - panel_dates = 面板全局交易日 (bundle staleness 判定)
    """
    b = load_latest(board)
    if b is None:
        print(f"[prob_head] {board} 无概率头 bundle -> 闸不可用", flush=True)
        return None
    if len(panel_dates) < LEGACY_PROB_GATE["base_rate_days"] + 20:
        print(f"[prob_head] {board} 面板日期不足 -> 闸不可用", flush=True)
        return None
    age = bundle_age_trading_days(panel_dates, str(b["trained_through"]))
    if age is None or age > LEGACY_PROB_GATE["max_stale_days"]:
        print(
            f"[prob_head] {board} bundle 年龄 {age} 交易日 > "
            f"{LEGACY_PROB_GATE['max_stale_days']} -> 闸不可用 (请重训概率头)",
            flush=True,
        )
        return None
    base = _base_rate(tail)
    if base is None:
        print(f"[prob_head] {board} base_rate 可观测样本不足 -> 闸不可用", flush=True)
        return None
    if feat_day.empty:
        print(f"[prob_head] {board} 当日截面为空 -> 闸不可用", flush=True)
        return None
    cs = feat_day.copy()
    cs["symbol"] = cs["symbol"].astype(str)
    pred = predict(b, cs)
    return pd.Series(pred.to_numpy(), index=cs["symbol"]), base


def _gate_margin_dir() -> Path:
    return Path(LEGACY_PROB_GATE["gate_margin_dir"])


def _history_dates(board: str, today: str, limit: int) -> list[str]:
    """spread 历史日期 (严格 < today, 降序, 最多 limit 个) — no-lookahead 边界."""
    d = _gate_margin_dir()
    dates = []
    for fp in d.glob(f"spreads_{board}_*.csv"):
        date = fp.stem.removeprefix(f"spreads_{board}_")
        if date.isdigit() and date < today:
            dates.append(date)
    return sorted(dates, reverse=True)[:limit]


def _breaker_tripped(board: str, today: str) -> bool:
    """连续 3 个决策日 (n_total>0) 全部 0 保留 → True (闸退化成板禁时强制放开)."""
    d = _gate_margin_dir()
    dates = []
    for fp in d.glob(f"margin_{board}_*.json"):
        date = fp.stem.removeprefix(f"margin_{board}_")
        if date.isdigit() and date < today:
            dates.append(date)
    n_checked = 0
    for date in sorted(dates, reverse=True):
        try:
            with open(d / f"margin_{board}_{date}.json", encoding="utf-8") as fh:
                dec = json.load(fh)
        except Exception:
            continue
        if int(dec.get("n_total", 0)) <= 0:
            continue  # 无参与者的日子不算决策日
        n_checked += 1
        if int(dec.get("n_kept", 0)) > 0:
            return False
        if n_checked >= 3:
            return True
    return False


def compute_adaptive_margin(
    board: str, today_spreads: pd.Series | None, today: str
) -> tuple[float, str]:
    """自适应 margin: 滚动分位数 / 熔断 / bootstrap / 静态回退.

    Returns:
        (margin, mode): mode ∈ fixed | breaker | rolling_q | bootstrap | fixed_fallback
    """
    cfg = LEGACY_PROB_GATE
    if cfg.get("margin_mode", "fixed") == "fixed":
        return float(cfg["margin"]), "fixed"
    q = float(cfg["margin_q"])
    lo = float(cfg["margin_min"])
    hi = float(cfg["margin_max"])
    if _breaker_tripped(board, today):
        print(
            f"[prob_gate] {board} 熔断: 连续 3 决策日 100% 剔除 → margin 放开至地板 "
            f"{lo} (检查概率头/分布漂移!)",
            flush=True,
        )
        return lo, "breaker"
    vals: list[np.ndarray] = []
    for date in _history_dates(board, today, int(cfg["spread_lookback_days"])):
        try:
            df = pd.read_csv(_gate_margin_dir() / f"spreads_{board}_{date}.csv")
            v = pd.to_numeric(df["spread"], errors="coerce").dropna().to_numpy(float)
        except Exception:
            continue
        if len(v):
            vals.append(v[np.isfinite(v)])
    if vals:
        pooled = np.concatenate(vals)
        return float(np.clip(np.quantile(pooled, q), lo, hi)), "rolling_q"
    if today_spreads is not None:
        s = pd.Series(today_spreads).dropna()
        s = s[np.isfinite(s.astype(float))].astype(float)
        if len(s):
            return float(np.clip(np.quantile(s.to_numpy(), q), lo, hi)), "bootstrap"
    print(
        f"[prob_gate] {board} 无 spread 历史/当日截面 → 回退静态 margin "
        f"{cfg['margin']} (fail-open 保守)",
        flush=True,
    )
    return float(cfg["margin"]), "fixed_fallback"


def _persist_gate_state(
    board: str,
    today: str,
    symbols: pd.Index,
    p: pd.Series,
    base: float,
    margin: float,
    mode: str,
    thr: float,
    keep: pd.Series,
) -> None:
    """当日 spread 逐股 + margin 决策落盘 (WORM 逐日文件; 当日重跑覆盖当日文件)."""
    d = _gate_margin_dir()
    d.mkdir(parents=True, exist_ok=True)
    hist_dates = _history_dates(
        board, today, int(LEGACY_PROB_GATE["spread_lookback_days"])
    )
    try:
        sp = pd.DataFrame(
            {
                "symbol": p.index.astype(str).tolist(),
                "pred_prob": p.to_numpy(float),
                "spread": (p - base).to_numpy(float),
            }
        )
        sp.to_csv(d / f"spreads_{board}_{today}.csv", index=False)
        n_total = int(len(symbols))
        n_kept = int(keep.sum())
        with open(d / f"margin_{board}_{today}.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "date": today,
                    "board": board,
                    "mode": mode,
                    "margin": margin,
                    "thr": thr,
                    "base_rate": base,
                    "q": LEGACY_PROB_GATE["margin_q"],
                    "n_total": n_total,
                    "n_kept": n_kept,
                    "n_history_days": len(hist_dates),
                },
                fh,
                ensure_ascii=False,
                indent=1,
            )
    except Exception as exc:  # 状态落盘失败不影响闸 (非阻塞)
        print(f"[prob_gate] {board} 状态落盘失败 (非阻塞): {exc}", flush=True)


def apply_prob_gate(
    res: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    tail: pd.DataFrame,
    panel_dates: np.ndarray,
    audit_sink: list | None = None,
) -> pd.DataFrame:
    """legacy 并行式概率闸 (接线后启用): t3 门后、pred_ret_10d 排名前.

    保留 ⇔ pred_prob > base_rate + margin. margin 自适应 (margin_mode=rolling_q):
    滚动分位数重锚到近 N 日实际可达 spread 分布 (静态 0.22 漂移后不可达 → main 板
    每日全灭的根因修复); "fixed" → 静态档 (一键回退). bundle 缺失/过旧/当日不可用 →
    fail-open (保留) + 大声告警 (不杀清单); 个股 pred_prob 缺失 → fail-open 保留.
    feats = {board: 当日截面 V35 特征帧}; 缺板块 → fail-open.
    res board 兼容生产命名 (main/GEM/STAR) 与内部 (main/dual) — GEM/STAR 并入 dual 组.
    附带: pred_prob 列写入输出 (仅诊断用 — legacy 排名键保持纯 pred_ret_10d,
    blend 已证伪, memory legacy-blend-rank-verdict, 勿再提).
    audit_sink: 传入 list 时逐股追加被剔记录 (symbol/board/pred_prob/spread/thr) —
    emit 汇总写入 gate_audit → 交付"仅供参考"节 (2026-08-31).
    当日 spreads/decision WORM 落盘 {gate_margin_dir}/ (次日自适应输入).
    """
    cfg = LEGACY_PROB_GATE
    if not cfg.get("enable", True):
        return res
    out = res.copy()
    if "date" in out.columns and out["date"].notna().any():
        today = pd.Timestamp(out["date"].max()).strftime("%Y%m%d")
    else:
        today = datetime.date.today().strftime("%Y%m%d")
    for board in cfg.get("gated_boards", ("main", "dual")):
        mask = out["board"].astype(str).map(_BOARD_GROUP).eq(board)
        if not mask.any():
            continue
        feat_day = feats.get(board)
        if feat_day is None:
            print(
                f"[prob_gate] {board} 无当日特征截面 -> 闸失效 (fail-open)", flush=True
            )
            continue
        got = gate_probabilities(board, feat_day, tail, panel_dates)
        if got is None:
            print(f"[prob_gate] {board} 概率头不可用 -> 闸失效 (fail-open)", flush=True)
            continue
        prob, base = got
        symbols = out.loc[mask, "symbol"].astype(str)
        p = symbols.map(prob)
        spreads = (p - base).dropna()
        margin, mode = compute_adaptive_margin(board, spreads, today)
        thr = base + margin
        out.loc[mask, "pred_prob"] = p.to_numpy()
        keep = (p > thr) | p.isna()
        _persist_gate_state(
            board, today, symbols.index, p, base, margin, mode, thr, keep
        )
        n_drop = int((~keep).sum())
        if audit_sink is not None and n_drop:
            for idx in out.index[mask & ~keep]:
                pv = p.get(idx)
                audit_sink.append(
                    {
                        "symbol": str(out.at[idx, "symbol"]),
                        "board": out.at[idx, "board"],
                        "stage": "prob_gate",
                        "reason": "prob_gate",
                        "pred_prob": float(pv) if pd.notna(pv) else np.nan,
                        "spread": float(pv - base) if pd.notna(pv) else np.nan,
                        "thr": float(thr),
                        "pain_prob": np.nan,
                        "pred_ret_10d": out.at[idx, "pred_ret_10d"]
                        if "pred_ret_10d" in out.columns
                        else np.nan,
                    }
                )
        if n_drop:
            dropped = out.loc[mask & ~keep, "symbol"].astype(str).tolist()
            print(
                f"[prob_gate] {board} 剔除 {n_drop} 只 (margin={margin:.3f}/{mode}, "
                f"pred_prob≤{thr:.1%}, base_rate {base:.1%}): "
                f"{', '.join(dropped[:20])}{' ...' if len(dropped) > 20 else ''}",
                flush=True,
            )
        else:
            print(
                f"[prob_gate] {board} 全过 (pred_prob>{thr:.1%}, base_rate {base:.1%})",
                flush=True,
            )
        out = out.loc[~mask | keep].reset_index(drop=True)
    return out
