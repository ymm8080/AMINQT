"""_diag_rank_head_replay.py — 对齐分类头重训+125d 全池回放 A/B (干净栈重判, 2026-09-02).

背景: 08-30 旧实验 (rank_head_exp_20260830_225128.json, 脚本未留档) 判对齐头
Δ-1.49pp/日, 但当时 main bundle 85 个 _brute_ 列推理缺失 (predictor 补 0),
训练/推理特征错位 → 结论存疑. brute 修复 (6f29f74c, PR#125) + q50 闸撤 (09-02)
+ main 20260903 包晋升后按用户指令重判.

口径 (复刻旧稿骨架, 特征升级为全量; 偏差点显式声明):
- 标签: 训练窗每日 "当日 net3 TOP10" = 1, 其余 0. 候选宇宙 = amount>=3e7 且
  net3 有效 (旧稿"全候选无闸"脚本已失传; 此处对齐评估端真赢家定义,
  amount 地板保证标签可交易). 真赢家 TOPN=10.
- 特征: bundle feature_cols **全量** (brute 修复后推理端已齐) — 旧稿仅 268 交集列.
- 模型: LGBM 二分类 trees=400 lr=0.05, 早停=时间序后 20% 验证窗, seed=42,
  subsample=1.0 (确定性), scale_pos_weight=neg/pos.
- 窗口: 训练窗=评估窗起点前 280 交易日; 评估窗=末 125 日 (与 q90 回放同窗可比);
  slice 420 同 _diag_q90_slot_replay (训练窗头部有特征 warmup 偏差, 旧稿同担).
- 回放: 与 q90 回放同机械 (warmup predict → 逐日 predict+compute_scores),
  额外记录 head_prob = 模型对当日截面的 predict_proba.
- 评估 (离线, 闸重建同 _q90_slot_eval 非 bear 口径, q50 闸已撤):
  base = pred_ret_10d TOP10 / head = head_prob TOP10 /
  blend = 0.5*(ret10 分位 + head 分位). 配对 Δ + 两半窗 + win rate + cov.
- 口径警告: 影子回放含训练数据, 绝对水平上偏; 两臂同担, Δ 可比.

WORM: DATA OTHERS/diag/rank_head_replay_<ts>.parquet + .json

用法:
  python scripts/_diag_rank_head_replay.py                 # 全量 420/280/125
  python scripts/_diag_rank_head_replay.py --slice 120 --train 60 --eval 8   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gc

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from config.settings import (
    LEGACY_ENTRY_GATE,
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)
from scripts._run_guard import find_conflicts

BUNDLE_MAIN = "models/pipeline1/main_current.pkl"
BUNDLE_DUAL = "models/pipeline1/dual_current.pkl"
COST = 0.0020  # 佣金+印花税+滑点 ≈ 0.2% 往返, 与 hitrate/q90 回放一致
KEEP_PRED = [
    "symbol",
    "pred_ret_10d",
    "pred_ret_3d",
    "prob_up",
    "prob_up_10d",
    "base_rate",
    "pain_prob",
    "pred_q50_3d",
    "pred_q50_5d",
    "pred_q75_3d",
    "pred_q90_3d",
]
TOPN = 10
WINNER_TOPN = 10
AMOUNT_FLOOR = 3e7
SEED = 42


def _pivots(panel: pd.DataFrame):
    """symbol×date 宽表: close_hfq (ffill) + amount, 返回 (px, amt, cal)."""
    cal = np.sort(
        np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy())
    )
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    px = (
        panel.assign(dt=dt)
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.to_datetime(cal))
        .ffill(axis=1)
    )
    amt = (
        panel.assign(dt=dt)
        .pivot_table(index="symbol", columns="dt", values="amount", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.to_datetime(cal))
    )
    return px, amt, cal


def _net_vec(px: pd.DataFrame, symbols: pd.Series, buy_dt, sell_dt) -> np.ndarray:
    pb = px[buy_dt].reindex(symbols).to_numpy(dtype=float)
    ps = px[sell_dt].reindex(symbols).to_numpy(dtype=float)
    out = ps / pb - 1.0 - COST
    out[~(pb > 0)] = np.nan
    return out


def train_head(
    feat: pd.DataFrame,
    feat_cols: list[str],
    train_days: list,
    vld_frac: float = 0.2,
    trees: int = 400,
) -> tuple:
    """训练窗每日 net3 TOP10 二分类头. 返回 (model, meta)."""
    import lightgbm as lgb

    d = feat[pd.to_datetime(feat["date"]).isin(train_days)]
    y = d["label"].to_numpy()
    X = d[feat_cols]
    meta = {"train_rows": int(len(d)), "pos": int(y.sum())}
    if meta["pos"] < 100:
        raise RuntimeError(f"训练正样本过少 pos={meta['pos']}")

    day_arr = pd.to_datetime(d["date"]).to_numpy()
    uniq = np.sort(pd.unique(day_arr))
    n_vld = max(int(len(uniq) * vld_frac), 1)
    vld_days = set(uniq[-n_vld:])
    is_vld = np.isin(day_arr, np.array(sorted(vld_days)))
    X_tr, y_tr = X[~is_vld], y[~is_vld]
    X_va, y_va = X[is_vld], y[is_vld]
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    meta.update(
        {
            "train_days": int(len(uniq)),
            "vld_days": int(n_vld),
            "vld_rows": int(is_vld.sum()),
            "scale_pos_weight": spw,
        }
    )
    model = lgb.LGBMClassifier(
        n_estimators=trees,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        colsample_bytree=0.8,
        subsample=1.0,
        scale_pos_weight=spw,
        random_state=SEED,
        n_jobs=8,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    meta["best_iter"] = int(model.best_iteration_ or trees)
    meta["vld_ap"] = float(
        model.best_score_["valid_0"]["average_precision"]
        if "valid_0" in model.best_score_
        else np.nan
    )
    return model, meta


def gate_mask(df: pd.DataFrame, board: str) -> pd.Series:
    """当期 LEGACY_ENTRY_GATE 非 bear (q50 闸已撤, 同 _q90_slot_eval)."""
    margin = LEGACY_ENTRY_GATE["prob_margin"][board]
    pain_max = LEGACY_ENTRY_GATE["pain_max"][board]
    prob_ok = df["prob_up"] > (df["base_rate"] + margin)
    ret_ok = df["pred_ret_10d"] > 0
    pain_ok = df["pain_prob"].fillna(0) <= pain_max
    return prob_ok & ret_ok & pain_ok


def eval_arms(df: pd.DataFrame, board: str) -> dict:
    """base/head/blend 三臂配对 A/B (录得行上离线重建, 同 _q90_slot_eval 语义)."""
    b = df[df["board"] == board].copy()
    b["symbol"] = b["symbol"].astype(str).str.zfill(6)
    b["date"] = b["date"].astype(str)
    ok = gate_mask(b, board)
    gated = b[ok].copy()
    b["ret_pct"] = b.groupby("date")["pred_ret_10d"].rank(pct=True)
    b["head_pct"] = b.groupby("date")["head_prob"].rank(pct=True)
    b["q90_pct"] = b.groupby("date")["pred_q90_3d"].rank(pct=True)
    gated = gated.merge(
        b[["ret_pct", "head_pct", "q90_pct"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    gated["blend_key"] = 0.5 * (
        gated["ret_pct"].fillna(0) + gated["head_pct"].fillna(0)
    )
    gated["blend_rq_key"] = 0.5 * (
        gated["ret_pct"].fillna(0) + gated["q90_pct"].fillna(0)
    )

    # 真赢家: 当日全池 amount 地板 + net3 TOP (同 _q90_slot_eval)
    full_ok = (df["amount"] >= AMOUNT_FLOOR) & df["net_3d"].notna()
    u = df[full_ok].copy()
    u["symbol"] = u["symbol"].astype(str).str.zfill(6)
    u["date"] = u["date"].astype(str)
    u = u.drop_duplicates(["date", "symbol"])
    winners: dict[str, set[str]] = {}
    for d, g in u.groupby("date"):
        winners[d] = set(g.nlargest(WINNER_TOPN, "net_3d")["symbol"])

    day_net: dict[str, dict[str, float]] = {}
    for d, g in b.groupby("date"):
        day_net[d] = dict(zip(g["symbol"], g["net_3d"]))

    daily: dict[str, dict[str, list[str]]] = {}
    for d, g in gated.groupby("date"):
        daily.setdefault(d, {})["base"] = list(
            g.sort_values(["pred_ret_10d", "symbol"], ascending=[False, True]).head(
                TOPN
            )["symbol"]
        )
        daily[d]["head"] = list(
            g.sort_values(["head_prob", "symbol"], ascending=[False, True]).head(TOPN)[
                "symbol"
            ]
        )
        daily[d]["blend"] = list(
            g.sort_values(["blend_key", "symbol"], ascending=[False, True]).head(TOPN)[
                "symbol"
            ]
        )
        daily[d]["q90"] = list(
            g.sort_values(["pred_q90_3d", "symbol"], ascending=[False, True]).head(
                TOPN
            )["symbol"]
        )
        daily[d]["blend_rq"] = list(
            g.sort_values(
                ["blend_rq_key", "symbol"], ascending=[False, True]
            ).head(TOPN)["symbol"]
        )

    def day_net3(picks: list[str]) -> float | None:
        vals = [day_net[d][s] for s in picks if s in day_net[d] and pd.notna(day_net[d][s])]
        return float(np.mean(vals)) if vals else None

    base_daily = {d: day_net3(v["base"]) for d, v in daily.items()}
    out: dict = {
        "pool_days": int(gated.groupby("date").ngroups),
        "pool_mean_per_day": float(
            len(gated) / max(gated.groupby("date").ngroups, 1)
        ),
        "arms": {},
    }
    for arm in ("base", "head", "blend", "q90", "blend_rq"):
        picks_sets = {d: set(v[arm]) for d, v in daily.items()}
        net3s, hits, covs = [], [], []
        for d, ps in picks_sets.items():
            vals = [day_net[d][s] for s in ps if s in day_net[d] and pd.notna(day_net[d][s])]
            if vals:
                net3s.append(np.mean(vals))
                hits.append(np.mean([v > 0 for v in vals]))
            covs.append(len(ps & winners.get(d, set())))
        st = {
            "days": len(picks_sets),
            "net3": float(np.mean(net3s)) if net3s else None,
            "hit3": float(np.mean(hits)) if hits else None,
            "cov": float(np.mean(covs)) if covs else None,
            "cov_total": int(sum(covs)),
        }
        if arm != "base":
            common = sorted(set(daily) & set(base_daily))
            dif = np.array(
                [
                    a_v - b_v
                    for d in common
                    if (a_v := day_net3(daily[d][arm])) is not None
                    and (b_v := base_daily[d]) is not None
                ]
            )
            h = len(dif) // 2
            st.update(
                {
                    "d3_full": float(dif.mean()) if len(dif) else None,
                    "d3_h1": float(dif[:h].mean()) if h else None,
                    "d3_h2": float(dif[h:].mean()) if len(dif) - h else None,
                    "win": float((dif > 0).mean()) if len(dif) else None,
                    "win_h1": float((dif[:h] > 0).mean()) if h else None,
                    "win_h2": float((dif[h:] > 0).mean()) if len(dif) - h else None,
                }
            )
        out["arms"][arm] = st
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420)
    ap.add_argument("--train", type=int, default=280)
    ap.add_argument("--eval", type=int, default=125)
    ap.add_argument("--trees", type=int, default=400)
    args = ap.parse_args()

    t0 = time.time()
    conflicts = find_conflicts()
    if conflicts:
        for c in conflicts:
            print(f"[guard] 冲突: {c['sentinel']} (PID {c['pid']})", flush=True)
        return 2
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB)
    _mon = start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB, RETRAIN_RAM_GUARD_POLL_S)

    predictor = V35Predictor({"main": BUNDLE_MAIN, "dual": BUNDLE_DUAL})
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

    px, amt, cal = _pivots(panel)
    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    max_lag = 11
    print(f"[pivot] symbols={len(px)} days={len(cal)} ({time.time() - t0:.0f}s)", flush=True)

    main_df, dual_df, state = CleaningPipeline(CleaningConfig()).run_inference(panel)
    print(
        f"[clean] valve={state} main={len(main_df):,} dual={len(dual_df):,} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    del panel
    gc.collect()

    boards_df = {"main": main_df, "dual": dual_df}
    del main_df, dual_df
    model = None
    meta: dict = {}
    rows: list[dict] = []
    for board in ("main", "dual"):
        dfb = boards_df.pop(board)
        cols = predictor.bundles[board]["feature_cols"]
        csr = board == "dual"  # 与训练/生产一致: 仅双创开截面排名
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        print(f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c ({time.time() - t0:.0f}s)", flush=True)
        del dfb
        gc.collect()

        # 全框 net3 + amount (逐日向量化, 按 index 对齐写入)
        sym = feat["symbol"].astype(str).str.zfill(6)
        day_arr = pd.to_datetime(feat["date"]).dt.normalize()
        net3 = pd.Series(np.nan, index=feat.index, dtype=float)
        amt_arr = pd.Series(np.nan, index=feat.index, dtype=float)
        for d, idx in feat.groupby(day_arr).groups.items():
            if d not in i_of or i_of[d] + 4 >= len(all_cal):
                continue
            s = sym.loc[idx]
            net3.loc[idx] = _net_vec(px, s, all_cal[i_of[d] + 1], all_cal[i_of[d] + 4])
            if d in amt.columns:
                amt_arr.loc[idx] = amt[d].reindex(s).to_numpy(dtype=float)
        feat["net_3d"] = net3
        feat["amount_px"] = amt_arr

        day_dates = sorted(pd.unique(day_arr))
        eval_days = [
            d for d in day_dates if d in i_of and i_of[d] + max_lag < len(all_cal)
        ][-args.eval :]

        if board == "main":
            pre = [d for d in day_dates if d < eval_days[0]]
            train_days = pre[-args.train :]
            print(
                f"[split] train {pd.Timestamp(train_days[0]).date()}..{pd.Timestamp(train_days[-1]).date()} "
                f"({len(train_days)}d) | eval {pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()} "
                f"({len(eval_days)}d)",
                flush=True,
            )
            # 训练标签: 当日 amount>=3e7 且 net3 有效 中 net3 TOP10
            lbl = pd.Series(False, index=feat.index)
            grp = feat[pd.to_datetime(feat["date"]).isin(train_days)]
            for d, g in grp.groupby(pd.to_datetime(grp["date"]).dt.normalize()):
                ok = (g["amount_px"] >= AMOUNT_FLOOR) & g["net_3d"].notna()
                cand = g[ok]
                if len(cand) < TOPN:
                    continue
                top = cand.nlargest(TOPN, "net_3d").index
                lbl.loc[top] = True
            feat["label"] = lbl.astype(np.int8)
            model, meta = train_head(feat, cols, train_days, trees=args.trees)
            print(f"[train] {meta} ({time.time() - t0:.0f}s)", flush=True)
            feat = feat.drop(columns=["label"])

        # warmup predict (EMA/锚定状态), 同 q90 回放
        warm = [d for d in day_dates if d < eval_days[0]]
        for d in warm:
            day_feat = feat[day_arr == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if not pred.empty:
                    lister.compute_scores(pred)
            except Exception:
                pass
        print(f"[warmup:{board}] {len(warm)}d ({time.time() - t0:.0f}s)", flush=True)

        for k, d in enumerate(eval_days):
            day_feat = feat[day_arr == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
            except Exception as exc:
                print(f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}", flush=True)
                continue
            if pred.empty:
                continue
            scored = lister.compute_scores(pred)
            if "compound_ret" in scored.columns:
                scored["pred_ret_10d"] = scored["compound_ret"]
            if "compound_prob" in scored.columns:
                scored["prob_up"] = scored["compound_prob"]
            have = [c for c in KEEP_PRED if c in scored.columns]
            sub = scored[have].copy()
            sub["symbol"] = sub["symbol"].astype(str).str.zfill(6)
            if board == "main" and model is not None:
                Xd = day_feat[cols]
                proba = pd.DataFrame(
                    {
                        "symbol": sym.loc[Xd.index].to_numpy(),
                        "head_prob": model.predict_proba(Xd)[:, 1].astype(float),
                    }
                )
                sub = sub.merge(proba, on="symbol", how="left")
            else:
                sub["head_prob"] = np.nan  # 头仅 main 训练, dual 不外推
            di = i_of[d]
            sub["net_3d"] = _net_vec(px, sub["symbol"], all_cal[di + 1], all_cal[di + 4])
            sub["net_10d"] = _net_vec(px, sub["symbol"], all_cal[di + 1], all_cal[di + 11])
            if d in amt.columns:
                sub["amount"] = amt[d].reindex(sub["symbol"]).to_numpy(dtype=float)
            else:
                sub["amount"] = np.nan
            sub["date"] = str(pd.Timestamp(d).date())
            sub["board"] = board
            rows.extend(sub.to_dict("records"))
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                print(
                    f"[{board}] {k + 1}/{len(eval_days)} rows={len(rows):,} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        del feat
        gc.collect()

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"rank_head_replay_{ts}.parquet"
    df.to_parquet(pq_path, index=False)

    result = {
        "ts": ts,
        "slice": args.slice,
        "train": args.train,
        "eval": args.eval,
        "trees": args.trees,
        "seed": SEED,
        "cost": COST,
        "bundle": BUNDLE_MAIN,
        "bundle_mtime": pd.Timestamp(
            os.path.getmtime(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", BUNDLE_MAIN)
            ),
            unit="s",
        ).isoformat(),
        "train_meta": meta,
        "rows": int(len(df)),
        "days": int(df["date"].nunique()),
        "range": [str(df["date"].min()), str(df["date"].max())],
        "amount_floor": AMOUNT_FLOOR,
    }
    try:
        result["main"] = eval_arms(df, "main")
    except Exception as exc:
        result["main_eval_error"] = repr(exc)
        print(f"[eval-error] {exc!r} (parquet 已落盘可离线补评)", flush=True)
    (out_dir / f"rank_head_replay_{ts}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[saved] {pq_path} rows={len(df):,} ({time.time() - t0:.0f}s)", flush=True)
    arms = result.get("main", {}).get("arms", {})
    for name, a in arms.items():
        fmt = lambda v: f"{v:+.5f}" if v is not None else "  n/a  "
        extra = (
            f" Δ={fmt(a.get('d3_full'))} (h1 {fmt(a.get('d3_h1'))} / h2 {fmt(a.get('d3_h2'))}) "
            f"win={a.get('win'):.2f}" if a.get("d3_full") is not None else ""
        )
        print(
            f"  {name:9s} days={a['days']:3d} net3={a['net3']:+.4f} hit3={a['hit3']:.3f} "
            f"cov={a['cov']:.3f} ({a['cov_total']}只){extra}",
            flush=True,
        )
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
