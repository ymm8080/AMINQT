"""概率闸视界改造回测: 3d 摸 → 10d 摸 X% — 250d OOS replay (2026-08-20).

动机: 300911 8/17-19 被 legacy 3d 概率闸 (mfe_3d>=3%, pred_prob > base_rate+margin)
连续剔除, 8/19 直接涨停 +20% — 3d 摸窗口在突破前刚好结束, 视界错配 (memory 无此条目,
用户 2026-08-20 口头要求). 本脚本在 250d OOS 上回放概率闸 4 种视界变体 +
无闸基线 + v3 OR v10_8 组合, 评估各变体闸后 top-N 实得 10d + 子窗稳定性.

口径 (与生产逐位一致, 已核实 app/pipeline1/prob_head.py):
- 池: CleaningPipeline N=800 / E6=10% (08-20 定案) dual 板块 (main 无排名 bundle, 不扫)
- 概率头: LGB_PARAMS (prob_head.py 配方, 无早停), 特征 = prob_head.feature_cols
- 标签: mfe_3d = max(high_hfq[T+2..T+4])/close_hfq[T+1]-1-cost (生产 _add_mfe_3d)
       mfe_10d = max(high_hfq[T+2..T+11])/close_hfq[T+1]-1-cost (同式推广)
- 闸: pred_prob > base_rate + margin; base_rate = 当日可观测尾部 (mfe 非 NaN)
      最近 base_rate_days 个达标日均值, 先剔 NaN 再算 (prob_head._base_rate 语义,
      无前瞻 — 当日 mfe_3d 只可观测到 d-4, mfe_10d 到 d-11)
- 排名: pred_ret_10d (bundle 10d_reg) 降序 top5/10/20; 指标 = label_pm_10d_net
      mean + hit (>0), 全窗 250d + 4 子窗 (3d/5d 实得仅 top10 作参考)
- 闸作用于每日全截面 (N=800 池), 未叠加 t3/t5 门 — 与 _sweep_e6_for_N800 骨架一致,
  各闸变体共享同一切片, 相对比较公平

训练切点 (冻结单次训练, 无前瞻):
- eval 窗口 = 面板尾部 250 交易日; 训练 = eval_start 之前 (270d 特征暖机切片内),
  且训练行日期 <= eval_start - 12 交易日 (mfe_10d 最大未来偏移 +11 → 训练标签在
  eval 开始前完全成熟, 模型不带任何 eval 窗口价格信息)
- 注: 生产为 21 日滚动重训 + 全史 3y; 冻结训练是视界对比的可接受近似 (4 变体
  共享同一数据流与切点, 相对差异可比; 绝对水平勿与生产闸对齐)

用法:
  python scripts/_replay_prob_gate_horizon_20260820.py
      [--eval-days=250] [--targets=0.05,0.08,0.10] [--case-dates=2026-08-18,2026-08-19]

注意: 严禁与重活 (E6 sweep / 重训 / refresh) 并发 — 本机 15.8GB 物理内存,
背景任务 + 本脚本双特征帧会 _ArrayMemoryError (今晨已两次死掉).
输出: BACKTEST_RESULT_DIR/prob_gate_horizon_20260820_<ts>/result.json (WORM)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from app.pipeline1.cleaning_pipeline import (
    CleaningConfig,
    CleaningPipeline,
)
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import (
    COST,
    MASK_RECENT_DAYS,
    LabelEngine,
    slippage_tier,
)
from app.pipeline1.prob_head import LGB_PARAMS
from app.pipeline1.prob_head import feature_cols as prob_feature_cols
from config.settings import BACKTEST_RESULT_DIR, LEGACY_PROB_GATE, PANEL_V3_PATH

MODEL_DIR = "models/pipeline1"
BUNDLE = "dual_20260819.pkl"  # 生产 bundle (208 特征, neg200 pin)
N_POOL = 800  # 08-20 定案 dual serving 池
E6_PCT = 0.00  # 08-20 E6 重扫定案: N=800 下 0% 最优 (top5 +32.6%/top10 +26.1%)
FEATURE_WARMUP_DAYS = 270  # 特征滚动窗口暖机 (历史统计量积累)
EVAL_DAYS = 250
N_SUB = 4
TOPN = (5, 10, 20)
TRAIN_GAP_DAYS = 12  # mfe_10d 需 +11 交易日未来价 → 训练行须 <= eval_start-12
CASE_DATES = ("2026-08-18", "2026-08-19")  # 300911 被 3d 闸剔除的两日
CASE_SYMBOL = "300911"
DEFAULT_T10_TARGETS = (0.05, 0.08, 0.10)

MARGIN = float(LEGACY_PROB_GATE["margin"])
BASE_RATE_DAYS = int(LEGACY_PROB_GATE["base_rate_days"])
BASE_RATE_WINDOW = BASE_RATE_DAYS + 14  # 生产 _prob_gate_inputs: 尾部 base_rate_days+14 交易日


def _add_mfe(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """生产口径 mfe_<window>d = max(high_hfq[T+2..T+1+window]) / close_hfq[T+1] - 1 - cost.

    window=3 与 prob_head._add_mfe_3d 逐位一致 (groupby symbol, skipna=False →
    停牌缺口 NaN 传播 → 标签不可用行自动剔除, 不参与训练/达标率).
    """
    g = df.groupby("symbol", sort=False)
    exec_px = g["close_hfq"].shift(-1)
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, 2 + window)],
        axis=1,
        keys=range(2, 2 + window),
    )
    slip = df["adv20"].map(slippage_tier)
    cost_total = COST + 2 * slip
    df[f"mfe_{window}d"] = shifts.max(axis=1, skipna=False) / exec_px - 1 - cost_total
    return df


def _daily_hit(df: pd.DataFrame, mfe_col: str, target: float) -> pd.Series:
    """每日 mfe 达标率 (先剔 NaN 行 — 同 prob_head._base_rate: NaN>=x 是 False 陷阱).

    只含可观测日 (mfe 非 NaN): mfe_3d 只到 d-4, mfe_10d 只到 d-11 → 序列本身
    就是无前瞻的达标率记录.
    """
    sub = df[df[mfe_col].notna()]
    hit = (sub[mfe_col] >= target).groupby(sub["date"]).mean()
    return hit.sort_index()


def _base_rate_at(dates_arr: np.ndarray, hit: pd.Series, d, days: int, window: int):
    """day d 的 base_rate: 最近 window 个交易日 (<=d) 内的可观测日达标率,
    取最近 days 个求均值 (prob_head._base_rate 语义); 不足 → None (闸不可用 → fail-open).
    """
    pos = int(np.searchsorted(dates_arr, d, side="right"))
    lo = max(0, pos - window)
    win = hit.reindex(dates_arr[lo:pos]).dropna()
    if len(win) < days:
        return None
    return float(win.iloc[-days:].mean())


def _train_prob_variants(
    df: pd.DataFrame, prob_cols: list[str], train_end, variants: dict
) -> dict:
    """冻结单次训练: 每变体一个 LGBM 概率头 (生产 train_bundle 同过滤: mfe 非 NaN
    且 label_pain 非 NaN; label_pain 遮蔽停牌/未成熟行 → 与 mfe NaN 行重合剔除)."""
    tr = df[df["date"] <= train_end]
    models: dict = {}
    for name, cfg in variants.items():
        mfe_col, target = cfg["mfe"], cfg["target"]
        y = (tr[mfe_col] >= target).astype(float)
        ok = y.notna() & tr["label_pain"].notna()
        x = tr.loc[ok, prob_cols].to_numpy(dtype="float32")
        if len(x) < 5000:
            raise RuntimeError(f"[train {name}] 训练样本不足 ({len(x)})")
        print(
            f"[train {name}] rows={len(x):,} mfe={mfe_col} target={target:.2%}",
            flush=True,
        )
        m = LGBMClassifier(**LGB_PARAMS)
        m.fit(x, y.loc[ok].to_numpy())
        models[name] = m
        del x
        gc.collect()
    del tr
    gc.collect()
    return models


def _f(v) -> float | None:
    return None if pd.isna(v) else float(v)


def _topn_stats(rows: list[dict], eval_dates: list, n_sub: int, with_ref: bool) -> dict:
    """top-N 池 10d 实得 mean/hit + 4 子窗; with_ref → 附加 3d/5d 参考均值."""
    out = {"n_selected": len(rows)}
    sub = [r for r in rows if r["label_pm_10d_net"] is not None]
    out["n_matured"] = len(sub)
    if sub:
        vals = np.array([r["label_pm_10d_net"] for r in sub])
        dates = np.array([r["date"] for r in sub], dtype="datetime64[ns]")
        out["mean_10d"] = float(vals.mean())
        out["hit_10d"] = float((vals > 0).mean())
        ev = np.array(eval_dates, dtype="datetime64[ns]")
        step = len(ev) // n_sub
        subs: list[float | None] = []
        for i in range(n_sub):
            d0 = ev[i * step]
            d1 = ev[-1] if i == n_sub - 1 else ev[(i + 1) * step - 1]
            m = (dates >= d0) & (dates <= d1)
            subs.append(float(vals[m].mean()) if m.any() else None)
        out["sub10d_mean"] = subs
        got = [s for s in subs if s is not None]
        out["sub10d_std"] = float(np.std(got)) if got else None
        if with_ref:
            for lab, key in (
                ("label_pm_3d_net", "mean_3d"),
                ("label_pm_5d_net", "mean_5d"),
            ):
                v3 = np.array([r[lab] for r in sub if r[lab] is not None])
                out[key] = float(v3.mean()) if len(v3) else None
    return out


def _gate_stats(day_records: list[tuple[int, int, bool]]) -> dict:
    """每日 (n_pass, n_total, unavailable) → 汇总 (fail-open 日计 unavailable)."""
    n_pass = np.array([r[0] for r in day_records], dtype=float)
    n_tot = np.array([r[1] for r in day_records], dtype=float)
    unavail = sum(1 for r in day_records if r[2])
    return {
        "days": len(day_records),
        "days_unavailable": unavail,
        "mean_n_pass": float(n_pass.mean()),
        "mean_n_total": float(n_tot.mean()),
        "mean_pass_rate": float((n_pass / n_tot).mean()),
    }


def main() -> int:
    import sys as _sys

    _eval_days = EVAL_DAYS
    _t10_targets = DEFAULT_T10_TARGETS
    _case_dates = CASE_DATES
    for a in _sys.argv[1:]:
        if a.startswith("--eval-days="):
            _eval_days = int(a.split("=", 1)[1])
        elif a.startswith("--targets="):
            _t10_targets = tuple(float(x) for x in a.split("=", 1)[1].split(","))
        elif a.startswith("--case-dates="):
            _case_dates = tuple(a.split("=", 1)[1].split(","))
    n_sub = max(2, _eval_days // 60)  # 250d→4 子窗 / 125d→2
    warmup_days = FEATURE_WARMUP_DAYS + _eval_days
    case_ts = {pd.Timestamp(x) for x in _case_dates}

    t0 = time.time()
    print(
        f"[cfg] eval_days={_eval_days} t10_targets={_t10_targets} "
        f"margin={MARGIN} base_rate_days={BASE_RATE_DAYS} window={BASE_RATE_WINDOW}",
        flush=True,
    )

    # ---- 1. bundle feature_cols (特征构建 inference_cols) ----
    bundle_path = os.path.join(MODEL_DIR, BUNDLE)
    b = DualTrackTrainer.load(bundle_path)
    reg_cols = list(b["feature_cols"])
    print(f"[bundle] {BUNDLE} n_feats={len(reg_cols)}", flush=True)
    del b
    gc.collect()

    # ---- 2. 面板: 日期预过滤切片 (对齐 _bt_sweep_topn_20260820, 省内存) ----
    import pyarrow.parquet as pq

    _all_dates = (
        pq.read_table(str(PANEL_V3_PATH), columns=["date"])["date"]
        .to_pandas()
        .dt.date.unique()
    )
    _all_dates = pd.Series(sorted(_all_dates))
    cutoff_date = pd.Timestamp(_all_dates.iloc[-warmup_days])
    del _all_dates
    gc.collect()
    print(f"[panel] cutoff {cutoff_date.date()} (last {warmup_days} trading days)", flush=True)
    panel = pq.read_table(
        str(PANEL_V3_PATH),
        filters=[
            ("amount", ">=", CleaningConfig().min_amount),
            ("date", ">=", cutoff_date),
        ],
    ).to_pandas()
    panel = panel[panel["is_suspended"] == 0].reset_index(drop=True)
    print(f"[panel] date-filtered load -> {len(panel):,}r", flush=True)

    # ---- 3. 清洗 (N=800, E6=10% 显式钉住) + 特征构建 (dual) ----
    cleaner = CleaningPipeline(
        CleaningConfig(
            liquidity_top_n=N_POOL,
            bottom_amount_pct=E6_PCT,
            bottom_amount_pct_main=0.0,
        )
    )
    main_b, dual_b, state = cleaner.run_inference(panel)
    del main_b, panel
    gc.collect()
    if state == "empty":
        print("[FATAL] valve empty -> abort", flush=True)
        return 3
    fe = FeatureEngineV35()
    df = fe.build(dual_b, None, inference_cols=reg_cols, cross_sectional_rank=True)
    del dual_b
    gc.collect()
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    for col in ("close_hfq", "high_hfq", "adv20"):
        if col not in df.columns:
            print(f"[FATAL] 特征帧缺 {col} (schema 漂移) -> abort", flush=True)
            return 3
    df = _add_mfe(df, 3)
    df = _add_mfe(df, 10)
    # mfe_10d 不在 prob_head.feature_cols 排除集里 (只排了 mfe_3d) — 手动剔, 防泄漏
    prob_cols = [c for c in prob_feature_cols(df) if not c.startswith("mfe_")]
    missing = [c for c in reg_cols if c not in df.columns]
    if missing:
        print(f"[FATAL] 缺失 reg 特征 {missing[:5]} -> abort", flush=True)
        return 3
    ddates = sorted(df["date"].unique())
    dates_arr = np.asarray(ddates)
    eval_start = ddates[-_eval_days]
    eval_dates = ddates[-_eval_days:]
    print(
        f"[features] frame {len(df):,}r, prob_cols={len(prob_cols)}, "
        f"eval {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} ({_eval_days}d)",
        flush=True,
    )

    # ---- 4. 概率头变体: 冻结单次训练 (eval_start - 12 交易日以前) ----
    pos_es = int(np.searchsorted(dates_arr, eval_start))
    if pos_es < TRAIN_GAP_DAYS:
        print("[FATAL] eval 窗口过大 (训练切点越界) -> abort", flush=True)
        return 3
    train_end = dates_arr[pos_es - TRAIN_GAP_DAYS]
    variants: dict = {}
    variants["v3"] = {"mfe": "mfe_3d", "target": float(LEGACY_PROB_GATE["abs_target"])}
    for t in _t10_targets:
        variants[f"v10_{int(round(t * 100))}"] = {"mfe": "mfe_10d", "target": t}
    print(
        f"[train] 冻结窗口 ..{pd.Timestamp(train_end):%Y-%m-%d} "
        f"(eval_start-{TRAIN_GAP_DAYS}d, 无 eval 窗口价格信息)",
        flush=True,
    )
    models = _train_prob_variants(df, prob_cols, train_end, variants)
    model_names = list(models)
    # 组合闸: v3 OR v10_8 (任务钉死; targets 覆盖且无 0.08 时退化为首个 10d 档)
    combo_10 = "v10_8" if "v10_8" in model_names else model_names[-1]
    if combo_10 != "v10_8":
        print(f"[warn] --targets 无 0.08 -> 组合闸用 {combo_10} 代替 v10_8", flush=True)
    gates = ["none"] + model_names + [f"combo_v3_{combo_10}"]

    # ---- 5. 每日达标率序列 (全帧, 无前瞻 — 只取 <=d 的窗口) ----
    hits = {n: _daily_hit(df, c["mfe"], c["target"]) for n, c in variants.items()}

    # ---- 6. 250d 回放 ----
    b = DualTrackTrainer.load(bundle_path)
    reg10 = b["models"]["10d_reg"][0]
    del b
    gc.collect()
    ev = df[df["date"] >= eval_start]
    del df  # 回放期只保留 eval 切片, 省 ~2GB (本机 15.8GB, 与背景任务共存)
    gc.collect()
    picked: dict = {g: {n: [] for n in TOPN} for g in gates}
    gate_days: dict = {g: [] for g in gates if g != "none"}
    case: dict = {}
    n_days = 0
    for d, day in ev.groupby("date", sort=False):
        n_days += 1
        sym = day["symbol"].astype(str).to_numpy()
        pred_10d = reg10.predict(np.nan_to_num(day[reg_cols].values, nan=0.0))
        X = day[prob_cols].to_numpy(dtype="float32")
        n_day = len(day)
        probs: dict = {}
        thrs: dict = {}
        for vn in model_names:
            probs[vn] = models[vn].predict_proba(X)[:, 1]
            br = _base_rate_at(dates_arr, hits[vn], d, BASE_RATE_DAYS, BASE_RATE_WINDOW)
            thrs[vn] = None if br is None else br + MARGIN
        keep: dict = {}
        keep["none"] = np.ones(n_day, dtype=bool)
        for vn in model_names:
            thr = thrs[vn]
            if thr is None:
                keep[vn] = np.ones(n_day, dtype=bool)  # fail-open (生产: 闸不可用→保留)
                gate_days[vn].append((n_day, n_day, True))
            else:
                keep[vn] = probs[vn] > thr
                gate_days[vn].append((int(keep[vn].sum()), n_day, False))
        t3, t8 = thrs["v3"], thrs[combo_10]
        if t3 is None or t8 is None:
            keep[f"combo_v3_{combo_10}"] = np.ones(n_day, dtype=bool)
            gate_days[f"combo_v3_{combo_10}"].append((n_day, n_day, True))
        else:
            keep[f"combo_v3_{combo_10}"] = (probs["v3"] > t3) | (probs[combo_10] > t8)
            gate_days[f"combo_v3_{combo_10}"].append(
                (int(keep[f"combo_v3_{combo_10}"].sum()), n_day, False)
            )
        # 排名 (pred_ret_10d 降序) → top-N 采集
        order_all = np.argsort(-pred_10d, kind="stable")
        rank_all = np.empty(n_day, dtype=int)
        rank_all[order_all] = np.arange(n_day)
        for gname, mask in keep.items():
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            order = idx[np.argsort(-pred_10d[idx], kind="stable")]
            for n in TOPN:
                for pos in range(min(n, len(order))):
                    i = order[pos]
                    picked[gname][n].append(
                        {
                            "date": d,
                            "symbol": sym[i],
                            "label_pm_3d_net": _f(day["label_pm_3d_net"].iat[i]),
                            "label_pm_5d_net": _f(day["label_pm_5d_net"].iat[i]),
                            "label_pm_10d_net": _f(day["label_pm_10d_net"].iat[i]),
                        }
                    )
        # 300911 个案
        if d in case_ts:
            pos = np.flatnonzero(sym == CASE_SYMBOL)
            if len(pos):
                i = int(pos[0])
                rec: dict = {
                    "n_pool": n_day,
                    "pred_ret_10d": _f(pred_10d[i]),
                    "rank_before_gate": int(rank_all[i]) + 1,
                    "in_top10_before_gate": bool(rank_all[i] < 10),
                    "realized": {
                        "1d_net": _f(day["label_pm_1d_net"].iat[i]),
                        "3d_net": _f(day["label_pm_3d_net"].iat[i]),
                        "10d_net": _f(day["label_pm_10d_net"].iat[i]),
                    },
                    "gates": {},
                }
                for vn in model_names:
                    rec["gates"][vn] = {
                        "pred_prob": _f(probs[vn][i]),
                        "base_rate": _f(
                            thrs[vn] - MARGIN if thrs[vn] is not None else None
                        ),
                        "threshold": _f(thrs[vn]),
                        "passed": bool(keep[vn][i]),
                        "in_top10_after": bool(
                            keep[vn][i] and rank_all[i] < 10
                        ),
                    }
                cn = f"combo_v3_{combo_10}"
                rec["gates"][cn] = {
                    "pred_prob_v3": _f(probs["v3"][i]),
                    "pred_prob_10": _f(probs[combo_10][i]),
                    "base_rate_v3": _f(t3 - MARGIN if t3 is not None else None),
                    "base_rate_10": _f(t8 - MARGIN if t8 is not None else None),
                    "threshold_v3": _f(t3),
                    "threshold_10": _f(t8),
                    "passed": bool(keep[cn][i]),
                    "in_top10_after": bool(keep[cn][i] and rank_all[i] < 10),
                }
                case[str(d.date())] = rec
            else:
                print(
                    f"[warn] {CASE_SYMBOL} 不在 {d.date()} 池 (n={n_day})", flush=True
                )
        if n_days % 50 == 0:
            print(f"[replay] {n_days}/{_eval_days} days done ({time.time()-t0:.0f}s)", flush=True)
    del ev, models, reg10
    gc.collect()
    print(f"[replay] done {n_days} days ({time.time()-t0:.0f}s)", flush=True)

    # ---- 7. 汇总指标 ----
    metrics: dict = {}
    for gname in gates:
        gm = {
            "gate_stats": _gate_stats(gate_days[gname]) if gname != "none" else None,
        }
        for n in TOPN:
            gm[f"top{n}"] = _topn_stats(picked[gname][n], eval_dates, n_sub, with_ref=(n == 10))
        metrics[gname] = gm
        t10 = gm["top10"]
        print(
            f"[{gname}] top10 10d={t10.get('mean_10d'):+.3f} hit={t10.get('hit_10d'):.3f} "
            f"sub={[round(s, 3) if s is not None else None for s in t10.get('sub10d_mean', [])]}",
            flush=True,
        )
    # 判定: top10 10d 实得最高 + 子窗赢无闸基线次数
    verdict: dict = {}
    for n in TOPN:
        best = max(gates, key=lambda g: metrics[g][f"top{n}"].get("mean_10d", -9.0))
        verdict[f"top{n}_best"] = best
        verdict[f"top{n}_mean_10d"] = {
            g: metrics[g][f"top{n}"].get("mean_10d") for g in gates
        }
    base_sub = metrics["none"]["top10"].get("sub10d_mean") or []
    verdict["top10_sub_win_vs_none"] = {}
    for g in gates:
        subs = metrics[g]["top10"].get("sub10d_mean") or []
        verdict["top10_sub_win_vs_none"][g] = sum(
            1 for a, b in zip(subs, base_sub) if a is not None and b is not None and a > b
        )

    # ---- 8. WORM JSON ----
    results = {
        "_meta": {
            "script": "scripts/_replay_prob_gate_horizon_20260820.py",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bundle": BUNDLE,
            "pool": {"N": N_POOL, "E6": E6_PCT, "board": "dual"},
            "gate_cfg": {
                "margin": MARGIN,
                "base_rate_days": BASE_RATE_DAYS,
                "tail_window_trading_days": BASE_RATE_WINDOW,
            },
            "variants": {
                name: {"mfe": cfg["mfe"], "target": cfg["target"]}
                for name, cfg in variants.items()
            },
            "combo_gate": f"v3 OR {combo_10}",
            "eval_days": _eval_days,
            "eval_window": [str(eval_dates[0]), str(eval_dates[-1])],
            "train_window": [str(pd.Timestamp(dates_arr[0])), str(pd.Timestamp(train_end))],
            "note_frozen_train": (
                "生产为 21 日滚动重训 + 全史 3y; 本回放为冻结单次训练 (eval_start-12d 以前), "
                "4 变体共享同一数据流, 相对比较可比, 绝对水平勿与生产闸对齐"
            ),
            "note_gate_pool": (
                "闸作用于每日全截面 (N=800 池), 未叠加 t3/t5 门; 各闸变体共享同一切片"
            ),
        },
        "variants": metrics,
        "case_300911": case,
        "_verdict": verdict,
    }
    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"prob_gate_horizon_20260820_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] WORM -> {out_dir} ({time.time()-t0:.0f}s)", flush=True)
    for g in gates:
        t10 = metrics[g]["top10"]
        print(f"[summary {g}] top10 10d={t10.get('mean_10d'):+.3f} hit={t10.get('hit_10d'):.3f}", flush=True)
    if case:
        for dt, rec in case.items():
            v3 = rec["gates"]["v3"]
            print(
                f"[case {CASE_SYMBOL} {dt}] pred_ret_10d={rec['pred_ret_10d']:.3f} "
                f"rank={rec['rank_before_gate']} | v3 passed={v3['passed']} "
                f"(p={v3['pred_prob']:.3f} thr={v3['threshold']:.3f}) | "
                f"combo passed={rec['gates'][f'combo_v3_{combo_10}']['passed']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
