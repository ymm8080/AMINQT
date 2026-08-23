"""_diag_legacy_prob_quality_lab.py — legacy prob 质量三视界矩阵 (2026-08-22).

问题 (用户 08-22): mag×prob 排名键无效, 怀疑 prob 本身质量差 — 生产概率头用
mfe_3d touch-ceiling 标签 (窗口最高价 ≥ 3%), 而 touch 不等于可兑现收益. 本脚本
在 legacy 同一特征 / 同一 walk-forward 框架下三视界 mfe 头 + 3d c2c 头:
  - model_mfe_h:  y = (mfe_{h}d          >= 0.03)  (h∈{3,5,10}, 窗口最高价触达)
  - model_c2c:     y = (label_pm_3d_net   >= 0.03)  (候选修复: T+1 收盘买入 → T+4
                                                    收盘卖出扣成本后的可兑现净收益)

统一评估 (OOS, 默认末 125 交易日):
  1. AUC: mfe_h 头对同视界 {mfe 达标, c2c 达标} 两套真实标签 — 决策键是
     「mfe_h 头对 c2c 可兑现达标的 AUC」跨视界对比.
  2. 分散度: IQR / 唯一值 / 众数占比 (prob 过平是历史排名键死因).
  3. 排名键矩阵 (2026-08-22 用户定案: 只按 TOP-10 质量评估, 同并行 rank_ab 口径):
     mag{3,5,10}(pred_ret_3d/5d/10d) × prob{mfe3,mfe5,mfe10} 全矩阵 + 纯 mag +
     纯 prob + mag10×pred_c2c 对照, 已实现 T+10 c2c 净收益 (成本 0.2%), 4 子窗.

候选池/实得口径与 _diag_legacy_prob_head_replay 完全一致 (基准闸行, pain 排除标记),
walk-forward 配方同生产 (prob_head.LGB_PARAMS, 每 refit_every_days 扩窗重训).

防前瞻: 训练掩码止于 pos-(h+1) — mfe_h 标签窗口需 +h+1 交易日未来价; 特征列显式
过滤全部 mfe_ (生产 feature_cols 只排 mfe_3d, 三视界帧里 mfe_5d/10d 会泄漏).

检查点: data/_diag_legacy_prob_lab_<board>_e<eval>.parquet (崩溃重跑免重训).
WORM:  DATA_OTHERS/diag/legacy_prob_quality_lab_<ts>.csv/.json

用法:
  python scripts/_diag_legacy_prob_quality_lab.py                        # 全量 125d
  python scripts/_diag_legacy_prob_quality_lab.py --eval 60              # 60d OOS
  python scripts/_diag_legacy_prob_quality_lab.py --slice 150 --eval 30  # 冒烟
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, callback
from sklearn.metrics import roc_auc_score

from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import COST as LABEL_COST
from app.pipeline1.label_engine import LabelEngine, _ensure_sorted, slippage_tier
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from app.pipeline1.prob_calibrator import ProbCalibrator
from config.settings import DATA_DIR, LEGACY_PROB_GATE, PANEL_V3_PATH, data_others_path

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
REALIZED_COST = 0.0020  # 实得口径往返成本 (同 replay, 佣金+印花税+滑点 ≈ 0.2%)
REFIT_EVERY = LEGACY_PROB_GATE["refit_every_days"]  # 21 交易日
ABS_TARGET = LEGACY_PROB_GATE["abs_target"]  # 0.03
CALIB_WINDOW_DAYS = 40  # 隔离校准窗 (生产将落 LEGACY_PROB_GATE['calib_window_days'])
REALIZED_BUY_LAG = 1
REALIZED_SELL_LAG = 11
KEYS = ("mag", "mag_x_mfe", "mag_x_c2c")
DEPTHS = (10,)  # 2026-08-22 用户定案: 所有测试结果只按 TOP-10 质量评估
N_SUB = 4
HORIZONS = (3, 5, 10)  # 2026-08-22 矩阵: 训练 mfe3/5/10 三视界概率头 (同并行)


def _build_realized_pivot(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """symbol×date → close_hfq (每股 ffill 处理停牌), 返回 (宽表 pivot, 全局交易日历)."""
    cal = np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy())
    cal = np.sort(cal)
    pivot = (
        panel.assign(dt=pd.to_datetime(panel["date"]).dt.normalize())
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reindex(columns=pd.to_datetime(cal)).ffill(axis=1)
    return pivot, cal


def _realized_net(pivot: pd.DataFrame, cal: np.ndarray, i: int, symbol: str) -> float:
    """决策日 cal[i] 的 T+10 净实得: buy=cal[i+buy_lag] close, sell=cal[i+sell_lag] close."""
    buy_dt = pd.Timestamp(cal[i + REALIZED_BUY_LAG])
    sell_dt = pd.Timestamp(cal[i + REALIZED_SELL_LAG])
    try:
        pb = float(pivot.at[symbol, buy_dt])
        ps = float(pivot.at[symbol, sell_dt])
    except KeyError:
        return float("nan")
    if not (np.isfinite(pb) and np.isfinite(ps)) or pb <= 0:
        return float("nan")
    return ps / pb - 1.0 - REALIZED_COST


def _gate_mask(
    scored: pd.DataFrame,
    prob_margin: float = 0.0,
    ret_thresh: float = 0.0,
    pain_thresh: float = 0.5,
):
    """生产 entry_filter 非 bear 口径 (同 _diag_legacy_prob_head_replay)."""
    ok = pd.Series(True, index=scored.index)
    cp = (
        scored["compound_prob"]
        if "compound_prob" in scored.columns
        else scored["prob_up"]
    )
    cr = (
        scored["compound_ret"]
        if "compound_ret" in scored.columns
        else scored["pred_ret_10d"]
    )
    ok &= cp > scored["base_rate"] + prob_margin
    ok &= cr > ret_thresh
    if all(c in scored.columns for c in ("pred_q50_3d", "pred_q50_5d")):
        ok &= (scored["pred_q50_3d"].fillna(cr) > 0) & (
            scored["pred_q50_5d"].fillna(cr) > 0
        )
    if "pain_prob" in scored.columns and pain_thresh is not None:
        ok &= scored["pain_prob"].fillna(0) <= pain_thresh
    return ok


def _add_mfe_h(df: pd.DataFrame, h: int) -> None:
    """就地补 mfe_{h}d: 窗口(T+2..T+h+1)最高 / T+1 买入价 - 1 - (COST+2×分层滑点).

    h=3 与生产 prob_head._add_mfe_3d 字节一致 (shift 偏移 2..4, 同并行 _diag_parallel
    复验脚本口径). 先建 adv20 (滚动20日均量) 供滑点分层.
    """
    if "adv20" not in df.columns:
        if "amount" not in df.columns:
            raise ValueError("缺 amount 无法现算 adv20 打标签")
        df = _ensure_sorted(df)
        df["adv20"] = (
            df.groupby("symbol")["amount"]
            .rolling(20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )
    g = df.groupby("symbol", sort=False)
    exec_px = g["close_hfq"].shift(-1)
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, h + 2)],
        axis=1,
        keys=range(2, h + 2),
    )
    slip = df["adv20"].map(slippage_tier)
    cost_total = LABEL_COST + 2 * slip
    df[f"mfe_{h}d"] = shifts.max(axis=1, skipna=False) / exec_px - 1 - cost_total


def _build_raw_labels(dfb: pd.DataFrame) -> pd.DataFrame:
    """清洗帧 → 小 raw 帧: symbol/date/close_hfq/high_hfq/low_hfq/adv20 +
    mfe_{3,5,10}d + label_pm_{3,5,10}d_net + label_pain.

    mfe_h 用本地 _add_mfe_h (h=3 同生产 prob_head._add_mfe_3d); c2c 直接镜像
    LabelEngine.build_labels PM 会话日K近似口径: exec_px=close(T+1),
    label_pm_{h}d_net = close(T+h+1)/close(T+1) - (COST+2×分层滑点)
    (不整调 build_labels 免 ~40 列瞬时帧的内存 churn).
    """
    raw = dfb[["symbol", "date", "close_hfq", "high_hfq", "low_hfq", "amount"]].copy()
    raw["symbol"] = raw["symbol"].astype(str)
    for h in HORIZONS:
        _add_mfe_h(raw, h)
        g = raw.groupby("symbol", sort=False)
        exec_px = g["close_hfq"].shift(-1)
        future_close = g["close_hfq"].shift(-(h + 1))
        slip = raw["adv20"].map(slippage_tier)
        raw[f"label_pm_{h}d_net"] = future_close / exec_px - 1 - (LABEL_COST + 2 * slip)
    pain = LabelEngine.build_path_labels(raw)["label_pain"]
    if "is_suspended" in dfb.columns:
        rs = (
            dfb.groupby("symbol")["is_suspended"]
            .rolling(5)
            .sum()
            .reset_index(level=0, drop=True)
        )
        vals = rs.values
        susp = np.zeros(len(vals), dtype=bool)
        if len(vals) > 4:
            susp[: len(vals) - 4] = vals[4:] > 0
        pain = pain.where(~pd.Series(susp, index=rs.index), np.nan)
    raw["label_pain"] = pain
    return raw


def _auc(pred: pd.Series, lbl: pd.Series) -> float:
    m = lbl.notna() & pred.notna()
    if int(m.sum()) < 50:
        return float("nan")
    return float(roc_auc_score(lbl[m].astype(int), pred[m]))


def _dispersion(p: pd.Series) -> dict:
    p = p.round(4).dropna()
    if p.empty:
        return {"iqr": float("nan"), "nuniq": 0, "mode_share": float("nan")}
    return {
        "iqr": float(p.quantile(0.75) - p.quantile(0.25)),
        "nuniq": int(p.nunique()),
        "mode_share": float(p.value_counts().iloc[0] / len(p)),
    }


def _brier(pred: pd.Series, lbl: pd.Series) -> float:
    m = lbl.notna() & pred.notna()
    if int(m.sum()) < 50:
        return float("nan")
    return float(np.mean((pred[m].to_numpy() - lbl[m].to_numpy()) ** 2))


def _ece(pred: pd.Series, lbl: pd.Series, n_bins: int = 10) -> float:
    """样本加权期望校准误差 (ECE): mean_bucket (桶权重 × |桶内 pred 均值 - 桶内实际率|)."""
    m = lbl.notna() & pred.notna()
    if int(m.sum()) < 50:
        return float("nan")
    p = pred[m].to_numpy()
    t = lbl[m].to_numpy().astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        bm = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if bm.sum() == 0:
            continue
        ece += (bm.sum() / n) * abs(p[bm].mean() - t[bm].mean())
    return float(ece)


def _fit_early_stop(
    model: LGBMClassifier,
    x_all: np.ndarray,
    y: pd.Series,
    idx: np.ndarray,
    board_dates_arr: np.ndarray,
    pos: int,
    tr_mask: np.ndarray,
    val_days: int,
    floor: int = 0,
) -> None:
    """训练窗尾部隔离 val_days 交易日做验证集, callbacks.early_stopping(50) (无前瞻).

    floor>0 时仿生产 CLS_MIN_TREES 地板: 早停树数 < floor → 以固定 floor 树重训
    (无早停), 防止短验证窗早停塌缩成常数 prob.
    """
    val_dates = board_dates_arr[max(0, pos - 4 - val_days) : pos - 4]
    val_mask = tr_mask & np.isin(board_dates_arr[idx], val_dates)
    if val_mask.sum() < 50:
        model.fit(x_all[tr_mask], y.loc[tr_mask].to_numpy())
        return model
    train_mask = tr_mask & ~val_mask
    model.fit(
        x_all[train_mask],
        y.loc[train_mask].to_numpy(),
        eval_X=x_all[val_mask],
        eval_y=y.loc[val_mask].to_numpy(),
        callbacks=[callback.early_stopping(50), callback.log_evaluation(0)],
    )
    if floor > 0:
        bi = getattr(model, "best_iteration_", None)
        if bi is not None and bi < floor:
            print(
                f"[es-floor] 早停 {bi} 树 < 地板 {floor} → 固定 {floor} 树重训",
                flush=True,
            )
            fresh = model.__class__(**model.get_params())
            fresh.set_params(n_estimators=floor)
            fresh.fit(x_all[tr_mask], y.loc[tr_mask].to_numpy())
            return fresh
    return model


def _rank_ab(sub: pd.DataFrame, keys: dict[str, str]) -> list[dict]:
    """每个排名键 × 深度 → 已实现 T+10 c2c 净收益统计 (4 子窗)."""
    out = []
    for name, rc in keys.items():
        for n in DEPTHS:
            top = (
                sub.sort_values(["date", rc], ascending=[True, False])
                .groupby("date", sort=False)
                .head(n)
            )
            r = top["realized_net"].dropna()
            days = sorted(sub["date"].unique())
            step = len(days) // N_SUB
            subs = []
            for i in range(N_SUB):
                s0, s1 = i * step, len(days) if i == N_SUB - 1 else (i + 1) * step
                seg = r[top["date"].isin(days[s0:s1])]
                subs.append(
                    {
                        "win": f"{i + 1}/{N_SUB}",
                        "hit": float((seg > 0).mean()) if len(seg) else float("nan"),
                        "mean": float(seg.mean()) if len(seg) else float("nan"),
                    }
                )
            out.append(
                {
                    "key": name,
                    "depth": n,
                    "hits": float((r > 0).mean()) if len(r) else float("nan"),
                    "mean": float(r.mean()) if len(r) else float("nan"),
                    "sub_windows": subs,
                }
            )
    return out


def _parse_lgb_override(s: str | None) -> dict:
    """'k:v,k:v' → dict (int 优先, 失败转 float). 空/None → {}."""
    if not s:
        return {}
    out: dict = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition(":")
        if not k or v == "":
            raise ValueError(f"无法解析 --lgb 项: {tok!r}")
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="面板切片交易日数")
    ap.add_argument("--eval", type=int, default=125, help="评估的已实现决策日数")
    ap.add_argument(
        "--tag",
        type=str,
        default="c2c_v1",
        help="本次 prob 配置标签 (区分 walk-forward 检查点)",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="特征缓存目录 (默认 <DATA_DIR>/_diag_prob_lab_cache)",
    )
    ap.add_argument("--rebuild", action="store_true", help="忽略特征缓存强制重建")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="跳过候选池 detail 构建与排名 A/B, 只算 walk-forward AUC+分散度 "
        "(prob 质量快速判词, 同 125d 口径 ~5x 快)",
    )
    ap.add_argument(
        "--calib",
        action="store_true",
        help="wf 每次重训后, 在训练窗尾部隔离 CALIB_WINDOW_DAYS 拟合 ProbCalibrator "
        "(无前瞻) 再应用到评估日 pred; 并加算 Brier/ECE 校准指标. "
        "AUC 对单调变换不变, 校准杠杆只看 Brier/ECE.",
    )
    ap.add_argument(
        "--calib-method",
        choices=("isotonic", "platt"),
        default="isotonic",
        help="校准器方法 (默认 isotonic; 生产镜像 dual_track 惯例)",
    )
    ap.add_argument(
        "--calib-days",
        type=int,
        default=CALIB_WINDOW_DAYS,
        help="隔离校准窗交易日数 (默认 %(default)s)",
    )
    ap.add_argument(
        "--lgb",
        type=str,
        default=None,
        help="LGBM 参数覆盖 k:v,k:v 逗号分隔 (无空格/引号, 防 shell 转义), "
        "如 learning_rate:0.03,n_estimators:800 (默认 prob_head.LGB_PARAMS; 仅改列出的键)",
    )
    ap.add_argument(
        "--early-stop",
        action="store_true",
        help="训练窗尾部隔离 val_days 交易日做验证集, callbacks.early_stopping(50) "
        "(无前瞻; 与 --lgb 组合为杠杆 ③ 调参配方)",
    )
    ap.add_argument(
        "--val-days",
        type=int,
        default=30,
        help="early-stop 验证集交易日数 (默认 %(default)s)",
    )
    ap.add_argument(
        "--es-floor",
        type=int,
        default=0,
        help="early-stop 最小树地板 (0=关; >0 仿生产 CLS_MIN_TREES: 早停树数<地板 → "
        "以固定地板树重训无早停, 防短验证窗塌缩成常数 prob)",
    )
    args = ap.parse_args()
    wf_params = {**prob_head.LGB_PARAMS, **_parse_lgb_override(args.lgb)}
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else DATA_DIR / "_diag_prob_lab_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    predictor = V35Predictor(BUNDLES)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    lister = ListGenerator()

    print(f"[load] panel {PANEL_V3_PATH}", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(
        f"[load] {len(panel):,}r max={panel['date'].max()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    dates = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    print(
        f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )

    pivot, cal = _build_realized_pivot(panel)
    print(
        f"[pivot] symbols={len(pivot)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    main_df, dual_df, state = cleaner.run_inference(panel)
    print(
        f"[clean] valve={state} main={len(main_df):,} dual={len(dual_df):,} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    del panel
    gc.collect()

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}

    detail: list[dict] = []
    board_auc: dict[str, list[dict]] = {}
    board_disp: dict[str, list[dict]] = {}
    board_calib: dict[str, list[dict]] = {}
    board_rank: dict[str, list[dict]] = {}

    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        feat_cache = cache_dir / f"feat_{board}_s{args.slice}.feather"
        use_cache = feat_cache.exists() and not args.rebuild
        if use_cache:
            feat = pd.read_feather(str(feat_cache))
            print(
                f"[{board}] 特征缓存命中 {len(feat):,}r ({time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            cols = predictor.bundles[board]["feature_cols"]
            feat = features.build(
                dfb, None, inference_cols=cols, cross_sectional_rank=csr
            )
            print(
                f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c ({time.time() - t0:.0f}s)",
                flush=True,
            )

        day_dates = sorted(pd.unique(pd.to_datetime(feat["date"])))
        eval_days = [
            d
            for d in day_dates
            if d in i_of and i_of[d] + REALIZED_SELL_LAG < len(all_cal)
        ][-args.eval :]
        print(
            f"[{board}] eval days {len(eval_days)} "
            f"({pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()})",
            flush=True,
        )

        if not args.quick:
            # ---- 1) 候选池: 基准闸行 + 只被 pain 拦下的行 (同 rescan/replay 口径) ----
            warm_days = [d for d in day_dates if d < eval_days[0]]
            for d in warm_days:
                day_feat = feat[pd.to_datetime(feat["date"]) == d]
                if day_feat.empty:
                    continue
                try:
                    pred = predictor.predict(day_feat, board)
                    if not pred.empty:
                        lister.compute_scores(pred)
                except Exception:
                    pass
            print(f"[{board}] base_rate 预热 {len(warm_days)} 天", flush=True)

            for k, d in enumerate(eval_days):
                di = i_of[d]
                day_feat = feat[pd.to_datetime(feat["date"]) == d]
                if day_feat.empty:
                    continue
                try:
                    pred = predictor.predict(day_feat, board)
                except Exception as exc:
                    print(
                        f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}",
                        flush=True,
                    )
                    continue
                if pred.empty:
                    continue
                scored = lister.compute_scores(pred)
                scored["date"] = d
                scored["board"] = board
                base_mask = _gate_mask(scored)
                pain_mask = _gate_mask(scored, pain_thresh=None)
                record = scored[base_mask | pain_mask].copy()
                record["pain_excluded"] = ~base_mask[record.index]
                for _, row in record.iterrows():
                    detail.append(
                        {
                            "date": str(pd.Timestamp(d).date()),
                            "board": board,
                            "symbol": str(row["symbol"]),
                            "pred_ret_3d": float(row.get("pred_ret_3d", np.nan)),
                            "pred_ret_5d": float(row.get("pred_ret_5d", np.nan)),
                            "pred_ret_10d": float(
                                row.get("compound_ret", row.get("pred_ret_10d", np.nan))
                            ),
                            "pain_excluded": bool(row["pain_excluded"]),
                            "realized_net": _realized_net(
                                pivot, cal, di, str(row["symbol"])
                            ),
                        }
                    )
                if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                    print(
                        f"[{board}] detail {k + 1}/{len(eval_days)} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
        else:
            print(
                f"[{board}] quick 模式: 跳过候选池/排名 A/B (只算 prob 质量判词)",
                flush=True,
            )

        # ---- 2) 双标签: 仅特征新建路径 (缓存帧已带标签, 跨杠杆复用) ----
        if not use_cache:
            raw = _build_raw_labels(dfb)
            feat["symbol"] = feat["symbol"].astype(str)
            feat["date"] = pd.to_datetime(feat["date"])
            merge_cols = (
                ["symbol", "date", "label_pain"]
                + [f"mfe_{h}d" for h in HORIZONS]
                + [f"label_pm_{h}d_net" for h in HORIZONS]
            )
            feat = feat.merge(raw[merge_cols], on=["symbol", "date"], how="left")
            print(f"[{board}] 三视界标签已合并 ({time.time() - t0:.0f}s)", flush=True)
            feat.to_feather(str(feat_cache))
            print(f"[{board}] 特征缓存已落盘: {feat_cache.name}", flush=True)
        del dfb
        gc.collect()

        board_dates_arr = np.array(pd.to_datetime(day_dates))
        meta = feat[["symbol", "date"]].reset_index(drop=True)

        # 防 look-ahead: 生产 feature_cols 只排 mfe_3d, 三视界标签帧里 mfe_5d/10d
        # 会泄漏进特征 (同并行 h5 修复) — 显式过滤所有 mfe_.
        feat_cols = [
            c for c in prob_head.feature_cols(feat) if not c.startswith("mfe_")
        ]
        # NaN 标签行 (停牌/未成熟) 必须置 NaN, 不能落成 NaN>=阈值=False 假负样本
        y_mfe = {
            h: ((feat[f"mfe_{h}d"] >= ABS_TARGET).astype(float)).where(
                feat[f"mfe_{h}d"].notna()
            )
            for h in HORIZONS
        }
        y_c2c = ((feat["label_pm_3d_net"] >= ABS_TARGET).astype(float)).where(
            feat["label_pm_3d_net"].notna()
        )
        ok_mfe_arr = {
            h: (y_mfe[h].notna() & feat["label_pain"].notna()).to_numpy()
            for h in HORIZONS
        }
        ok_c2c_arr = (y_c2c.notna() & feat["label_pain"].notna()).to_numpy()
        x_all = feat[feat_cols].to_numpy(dtype="float32")
        idx = np.searchsorted(board_dates_arr, feat["date"].values)
        auc_labels = feat[
            ["symbol", "date"]
            + [f"mfe_{h}d" for h in HORIZONS]
            + [f"label_pm_{h}d_net" for h in HORIZONS]
        ].copy()
        del feat
        gc.collect()

        ckpt_tag = (
            args.tag if not args.calib else f"{args.tag}_calib{args.calib_method}"
        )
        ckpt = (
            DATA_DIR / f"_diag_legacy_prob_lab_{board}_{ckpt_tag}_e{args.eval}.parquet"
        )
        if ckpt.exists():
            wf = pd.read_parquet(str(ckpt))
            print(f"[{board}] walk-forward 从检查点恢复 ({len(wf):,} 行)", flush=True)
        else:
            models_mfe: dict[int, LGBMClassifier] = {}
            model_c2c: LGBMClassifier | None = None
            calib_mfe: dict[int, ProbCalibrator | None] = {}
            calib_c2c: ProbCalibrator | None = None
            wf_rows: list[pd.DataFrame] = []
            n_refits = 0
            for k, d in enumerate(eval_days):
                pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
                if not models_mfe or k % REFIT_EVERY == 0:
                    # 训练掩码止于 pos-(h+1) — mfe_h 标签窗口需 +h+1 交易日未来价
                    # (mfe: T+2..T+h+1 最高; c2c: T+h+1 收盘), pos-(h+1)..pos-1 行
                    # 标签用到评估日及之后价格, 生产在 cutoff 处为 NaN 被排除,
                    # 否则虚增质量.
                    for h in HORIZONS:
                        tr_m = (idx < pos - (h + 1)) & ok_mfe_arr[h]
                        model = LGBMClassifier(**wf_params)
                        if args.early_stop:
                            model = _fit_early_stop(
                                model,
                                x_all,
                                y_mfe[h],
                                idx,
                                board_dates_arr,
                                pos,
                                tr_m,
                                args.val_days,
                                args.es_floor,
                            )
                        else:
                            model.fit(x_all[tr_m], y_mfe[h].loc[tr_m].to_numpy())
                        models_mfe[h] = model
                    tr_c = (idx < pos - 4) & ok_c2c_arr
                    model_c2c = LGBMClassifier(**wf_params)
                    if args.early_stop:
                        model_c2c = _fit_early_stop(
                            model_c2c,
                            x_all,
                            y_c2c,
                            idx,
                            board_dates_arr,
                            pos,
                            tr_c,
                            args.val_days,
                            args.es_floor,
                        )
                    else:
                        model_c2c.fit(x_all[tr_c], y_c2c.loc[tr_c].to_numpy())
                    n_refits += 1
                    if args.calib:
                        # 隔离校准窗 = 训练窗尾部 args.calib_days 个交易日 (全部 < pos-4, 无前瞻).
                        calib_dates = board_dates_arr[
                            max(0, pos - 4 - args.calib_days) : pos - 4
                        ]
                        calib_mask = np.isin(board_dates_arr[idx], calib_dates)
                        for h in HORIZONS:
                            cm = calib_mask & ok_mfe_arr[h]
                            calib_mfe[h] = (
                                ProbCalibrator(method=args.calib_method).fit(
                                    models_mfe[h].predict_proba(x_all[cm])[:, 1],
                                    y_mfe[h].loc[cm].to_numpy(),
                                )
                                if cm.sum() >= 100
                                else None
                            )
                        cc = calib_mask & ok_c2c_arr
                        calib_c2c = (
                            ProbCalibrator(method=args.calib_method).fit(
                                model_c2c.predict_proba(x_all[cc])[:, 1],
                                y_c2c.loc[cc].to_numpy(),
                            )
                            if cc.sum() >= 100
                            else None
                        )
                te = idx == pos
                if not te.any():
                    continue
                p_mfe = {
                    h: models_mfe[h].predict_proba(x_all[te])[:, 1] for h in HORIZONS
                }
                p_c2c = model_c2c.predict_proba(x_all[te])[:, 1]
                if args.calib:
                    for h in HORIZONS:
                        cal = calib_mfe.get(h)
                        if cal is not None:
                            p_mfe[h] = cal.predict_proba(p_mfe[h])
                    if calib_c2c is not None:
                        p_c2c = calib_c2c.predict_proba(p_c2c)
                row = {"pred_c2c": p_c2c}
                for h in HORIZONS:
                    row[f"pred_mfe_{h}d"] = p_mfe[h]
                wf_rows.append(meta.loc[te].assign(**row).reset_index(drop=True))
                if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                    print(
                        f"[{board}] wf {k + 1}/{len(eval_days)} "
                        f"(refits={n_refits}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
            wf = pd.concat(wf_rows, ignore_index=True)
            wf.to_parquet(ckpt)
            print(
                f"[{board}] walk-forward 完成: {n_refits} 次重训 → {ckpt.name}",
                flush=True,
            )

        # ---- 3) AUC + 分散度 (评估日全截面行, 非仅池内) ----
        wf["date"] = pd.to_datetime(wf["date"])
        wl = wf.merge(auc_labels, on=["symbol", "date"], how="left")
        # 缺标签行 (停牌/未成熟) 必须置 NaN, 不能落到 NaN>=阈值=False → 假负样本
        y_c2c_out = {
            h: ((wl[f"label_pm_{h}d_net"] >= ABS_TARGET).astype(float)).where(
                wl[f"label_pm_{h}d_net"].notna()
            )
            for h in HORIZONS
        }
        y_mfe_out = {
            h: ((wl[f"mfe_{h}d"] >= ABS_TARGET).astype(float)).where(
                wl[f"mfe_{h}d"].notna()
            )
            for h in HORIZONS
        }
        auc_rows: list[dict] = []
        for h in HORIZONS:
            # 决策键: mfe_h 头对同视界 c2c 可兑现达标的 AUC (才是排名键要的)
            auc_rows += [
                {
                    "model": f"mfe{h}",
                    "target": f"c2c{h}",
                    "auc": _auc(wl[f"pred_mfe_{h}d"], y_c2c_out[h]),
                },
                {
                    "model": f"mfe{h}",
                    "target": f"mfe{h}",
                    "auc": _auc(wl[f"pred_mfe_{h}d"], y_mfe_out[h]),
                },
            ]
        auc_rows += [
            {
                "model": "c2c",
                "target": "c2c3",
                "auc": _auc(wl["pred_c2c"], y_c2c_out[3]),
            },
            {
                "model": "c2c",
                "target": "mfe3",
                "auc": _auc(wl["pred_c2c"], y_mfe_out[3]),
            },
        ]
        board_auc[board] = auc_rows
        print(f"\n===== {board} | AUC (pred vs 真实标签) =====", flush=True)
        print(
            f"  {'model':<7}{'target':<8}{'AUC':>8}",
            flush=True,
        )
        for a in auc_rows:
            print(f"  {a['model']:<7}{a['target']:<8}{a['auc']:>8.4f}", flush=True)

        disp_rows = [
            {"model": f"mfe{h}", **_dispersion(wl[f"pred_mfe_{h}d"])} for h in HORIZONS
        ] + [{"model": "c2c", **_dispersion(wl["pred_c2c"])}]
        board_disp[board] = disp_rows
        print(f"===== {board} | 分散度 =====", flush=True)
        print(f"  {'model':<7}{'IQR':>8}{'唯一值':>7}{'众数占比':>8}", flush=True)
        for d_ in disp_rows:
            print(
                f"  {d_['model']:<7}{d_['iqr']:>8.4f}{d_['nuniq']:>7}{d_['mode_share']:>8.2%}",
                flush=True,
            )

        # ---- 3b) 校准 (Brier/ECE) — AUC 对单调变换不变, 校准杠杆只看这两个 ----
        calib_rows: list[dict] = []
        for h in HORIZONS:
            calib_rows += [
                {
                    "model": f"mfe{h}",
                    "target": f"c2c{h}",
                    "brier": _brier(wl[f"pred_mfe_{h}d"], y_c2c_out[h]),
                    "ece": _ece(wl[f"pred_mfe_{h}d"], y_c2c_out[h]),
                },
                {
                    "model": f"mfe{h}",
                    "target": f"mfe{h}",
                    "brier": _brier(wl[f"pred_mfe_{h}d"], y_mfe_out[h]),
                    "ece": _ece(wl[f"pred_mfe_{h}d"], y_mfe_out[h]),
                },
            ]
        calib_rows += [
            {
                "model": "c2c",
                "target": "c2c3",
                "brier": _brier(wl["pred_c2c"], y_c2c_out[3]),
                "ece": _ece(wl["pred_c2c"], y_c2c_out[3]),
            },
            {
                "model": "c2c",
                "target": "mfe3",
                "brier": _brier(wl["pred_c2c"], y_mfe_out[3]),
                "ece": _ece(wl["pred_c2c"], y_mfe_out[3]),
            },
        ]
        board_calib[board] = calib_rows
        print(f"===== {board} | 校准 (Brier / ECE) =====", flush=True)
        print(f"  {'model':<7}{'target':<8}{'Brier':>9}{'ECE':>9}", flush=True)
        for c_ in calib_rows:
            print(
                f"  {c_['model']:<7}{c_['target']:<8}{c_['brier']:>9.4f}{c_['ece']:>9.4f}",
                flush=True,
            )

        if not args.quick:
            # ---- 4) 排名键头对头 (交付池 = 非 pain 排除行) ----
            sub = pd.DataFrame(
                [r for r in detail if r["board"] == board and not r["pain_excluded"]]
            )
            sub["date"] = pd.to_datetime(sub["date"])
            sub = sub.merge(wf, on=["symbol", "date"], how="left")
            # 矩阵: mag{3,5,10} × prob{mfe3,mfe5,mfe10} (同并行 rank_ab 口径)
            mag_cols = {
                "mag": "pred_ret_10d",
                "mag5": "pred_ret_5d",
                "mag3": "pred_ret_3d",
            }
            prob_cols = {f"mfe{h}": f"pred_mfe_{h}d" for h in HORIZONS}
            for mname, mcol in mag_cols.items():
                sub[mname] = sub[mcol]
            for pname, pcol in prob_cols.items():
                sub[pname] = sub[pcol]
                for mname, _ in mag_cols.items():
                    sub[f"rank_{mname}x_{pname}"] = sub[mname] * sub[pname]
            # c2c 3d 头对照 (历史基线)
            sub["rank_mag_x_c2c"] = sub["mag"] * sub["pred_c2c"]
            rank_keys = {}
            for mname, _ in mag_cols.items():
                rank_keys[mname] = mname
            for pname, _ in prob_cols.items():
                rank_keys[pname] = pname
                for mname, _ in mag_cols.items():
                    rank_keys[f"{mname}x_{pname}"] = f"rank_{mname}x_{pname}"
            rank_keys["mag_x_c2c"] = "rank_mag_x_c2c"
            rank_rows = _rank_ab(sub, rank_keys)
            board_rank[board] = rank_rows
            print(
                f"===== {board} | 排名键 TOP-{DEPTHS[0]} 已实现 T+10 c2c 净收益 =====",
                flush=True,
            )
            print(
                f"  {'key':<14}{'depth':>5}{'命中':>7} {'实得':>8}  子窗实得",
                flush=True,
            )
            for r_ in rank_rows:
                subs = "  ".join(
                    f"{w['win']}:{w['mean']:+.2%}" for w in r_["sub_windows"]
                )
                print(
                    f"  {r_['key']:<14}{r_['depth']:>5}{r_['hits']:>7.1%} "
                    f"{r_['mean']:>+8.2%}  {subs}",
                    flush=True,
                )
            del sub
        del wf, wl, x_all, meta
        gc.collect()

    if not args.quick and not detail:
        print("无任何过闸候选", flush=True)
        return 1

    # ---- WORM 输出 ----
    df = pd.DataFrame(detail)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    df.to_csv(out_dir / f"legacy_prob_quality_lab_{ts}.csv", index=False)
    (out_dir / f"legacy_prob_quality_lab_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "slice": args.slice,
                "eval": args.eval,
                "quick": args.quick,
                "cost": REALIZED_COST,
                "abs_target": ABS_TARGET,
                "refit_every": REFIT_EVERY,
                "calib": args.calib,
                "calib_method": args.calib_method if args.calib else None,
                "calib_window_days": args.calib_days if args.calib else None,
                "lgb_params": wf_params,
                "early_stop": args.early_stop,
                "val_days": args.val_days if args.early_stop else None,
                "es_floor": args.es_floor if args.early_stop else None,
                "note": "三视界 walk-forward: model_mfe_h(y=mfe_{h}d>=3%, h∈{3,5,10}) "
                "+ model_c2c(y=label_pm_3d_net>=3%, T+1 收盘买→T+4 收盘卖扣成本). "
                "决策键=各头对同视界 c2c 可兑现达标 AUC; 排名键矩阵在交付池(非 pain "
                "排除)上按 T+10 c2c 净实得(成本 0.2%)评估, TOP-10 only "
                "(2026-08-22 用户定案, 同并行 rank_ab 口径): mag{3,5,10}×prob{mfe3,5,10} "
                "全矩阵 + 纯 mag + 纯 prob + mag10×pred_c2c 对照. quick 模式跳过候选池/"
                "排名 A/B. calib=True 时 wf 每次重训在训练窗尾部隔离窗拟合 "
                "ProbCalibrator (无前瞻) 应用到评估日, 并加算 Brier/ECE. "
                "防前瞻: 特征列显式过滤全部 mfe_, 训练掩码止于 pos-(h+1).",
                "auc": board_auc,
                "dispersion": board_disp,
                "calib_metrics": board_calib,
                "rank_ab": board_rank,
                "n_detail": len(df),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"\n[saved] {out_dir}/legacy_prob_quality_lab_{ts}.csv/.json "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
