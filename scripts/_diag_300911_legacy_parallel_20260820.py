"""_diag_300911_legacy_parallel_20260820.py — 300911 8/18 双系统逐层归因 (零重建).

用户问题: 300911 8/18 (缩量洗盘日, 次日 8/19 涨停 +20%) 为何不在 LEGACY/PARALLEL
短名单? 检查筹码/价格/试盘上下影线/洗盘/下探不破等维度模型"看到了什么".

已知根因 1 (池层): legacy run_inference N=200 截断 (score_rank 307) → N=800 已定案待生效.
本脚本回答根因 2 (模型层): 若在池内, 模型会给什么分? 差多少?

全部读既有产物 (低内存, 不重建管线):
  - parallel 检查点 data/_diag_stage_dual_3y.parquet (run_train 全谱池: 无 top-N 截断,
    仅 5000万底线 + E6 10% 尾切; 含 300911 8/18 行)
  - parallel 概率头 data/prob_head/dual_prob_*.joblib (取 8/18 当日有效版)
  - legacy 概率头 data/prob_head_legacy/dual_prob_*.joblib (同上)
  - legacy 生产 bundle models/pipeline1/dual_20260818.pkl (8/18 清单所用版)

口径注意: 检查点 _xrank 截面排名 = 全谱 dual 池 (~1500 只); 生产 legacy 推理 = N=200
池内排名 → legacy 侧打分为近似 (N=800 生效后需正式 replay 复核); parallel 侧 = 生产精确口径.

输出 (WORM): BACKTEST_RESULT_DIR/_diag_300911_20260820_<ts>/report.json + stdout.

用法: python scripts/_diag_300911_legacy_parallel_20260820.py
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
import pyarrow.dataset as ds

from app.pipeline_parallel import prob_head
from app.pipeline_parallel.scoring import pool_score
from app.pipeline_parallel.config import FUSION, SNIPER
from config.settings import BACKTEST_RESULT_DIR, DATA_DIR, PROB_GATE

TARGET = "300911"
T_DATE = pd.Timestamp("2026-08-18")
# 8/20 自动化把当前检查点改名 stale (重建死在面板读取); 最新 stale 含 8/18+8/19 数据
CKPT_DUAL = DATA_DIR / "_diag_stage_dual_3y.parquet.stale_20260820_085347"
LEGACY_BUNDLE = "models/pipeline1/dual_20260818.pkl"
TRAJ_START = pd.Timestamp("2026-08-06")  # 试盘 (8/13) 前后窗口
TAIL_START = pd.Timestamp("2026-06-01")  # base_rate 观测尾窗

report: dict = {"target": TARGET, "trade_date": str(T_DATE.date())}


def ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0


def log(msg: str) -> None:
    print(msg, flush=True)


def load_prob_bundle_live_on(model_dir: Path, board: str, as_of: pd.Timestamp):
    """该目录中 trained_through <= as_of 的最新 WORM bundle (8/18 当日生产生效版)."""
    cands = sorted(model_dir.glob(f"{board}_prob_*.joblib"))
    picked = None
    for p in cands:
        import joblib

        try:
            b = joblib.load(p)
        except Exception:
            continue
        tt = b.get("trained_through")
        if tt is None:
            continue
        if pd.Timestamp(tt) <= as_of:
            if picked is None or pd.Timestamp(tt) > pd.Timestamp(picked[1]["trained_through"]):
                picked = (p, b)
        del b
    return picked


def main() -> int:
    t0 = time.time()
    log(f"[ram] available {ram_gb():.1f} GB")
    d = ds.dataset(str(CKPT_DUAL))

    # ---------- A. 300911 轨迹 + 是否在检查点 8/18 截面 ----------
    pool_feats = list(dict.fromkeys(list(SNIPER.pool) + list(FUSION.pool)))
    traj_cols = ["date", "symbol", "amount", "turnover_rate", "close_hfq"] + pool_feats
    traj = d.to_table(
        filter=(ds.field("symbol") == TARGET) & (ds.field("date") >= TRAJ_START),
        columns=traj_cols,
    ).to_pandas()
    traj["date"] = pd.to_datetime(traj["date"])
    log(f"[A] 300911 轨迹行 {len(traj)} ({traj['date'].min():%m-%d}..{traj['date'].max():%m-%d})")
    if len(traj) == None:  # noqa: E711  (防御: 完全无行)
        log("[A][FATAL] 检查点无 300911 行!")
        return 3
    has_818 = (traj["date"] == T_DATE).any()
    log(f"[A] 8/18 当日行存在: {has_818}")
    report["in_checkpoint_818"] = bool(has_818)
    report["traj_amount"] = {
        str(x.date()): float(y) for x, y in zip(traj["date"], traj["amount"])
    }

    # ---------- B. 每日截面: sniper/fusion 打分轨迹 + 当日 cutoff ----------
    log("[B] 读 8/06.. 截面 (池特征) 计算 parallel 打分轨迹 ...")
    xs = d.to_table(
        filter=ds.field("date") >= TRAJ_START,
        columns=["date", "symbol"] + pool_feats,
    ).to_pandas()
    xs["date"] = pd.to_datetime(xs["date"])
    rows = []
    cutoffs = {}
    for dt, sub in xs.groupby("date"):
        s_sn = pool_score(sub, SNIPER.pool)
        s_fu = pool_score(sub, FUSION.pool)
        sub = sub.assign(_sn=s_sn.values, _fu=s_fu.values)
        top10 = float(sub["_fu"].nlargest(10).min())  # T-10 cutoff (fusion 口径)
        top5 = float(sub["_sn"].nlargest(5).min())
        cutoffs[str(dt.date())] = {"sniper_top5_cut": top5, "fusion_top10_cut": top10}
        r911 = sub[sub["symbol"] == TARGET]
        if len(r911):
            x = r911.iloc[0]
            rows.append(
                {
                    "date": str(dt.date()),
                    "n_pool": int(len(sub)),
                    "sniper_score": float(x["_sn"]),
                    "sniper_rank": int((sub["_sn"] > x["_sn"]).sum() + 1),
                    "fusion_score": float(x["_fu"]),
                    "fusion_rank": int((sub["_fu"] > x["_fu"]).sum() + 1),
                }
            )
        del s_sn, s_fu, sub
    traj_df = pd.DataFrame(rows)
    log("[B] 300911 parallel 打分轨迹:")
    log("\n" + traj_df.to_string(index=False))
    log("[B] 当日 cutoffs:")
    for k, v in cutoffs.items():
        log(f"    {k}: sniper_top5={v['sniper_top5_cut']:.4f} fusion_top10={v['fusion_top10_cut']:.4f}")
    report["parallel_score_traj"] = traj_df.to_dict("records")
    report["parallel_cutoffs"] = cutoffs
    del xs
    gc.collect()

    # ---------- C. base_rate (as of 8/18, PIT) ----------
    log("[C] 读尾窗算 dual base_rate (as of 8/18) ...")
    tail = d.to_table(
        filter=ds.field("date") >= TAIL_START,
        columns=["date", "symbol", "close_hfq", "high_hfq", "adv20"],
    ).to_pandas()
    tail["date"] = pd.to_datetime(tail["date"])
    tail = tail[tail["date"] <= T_DATE].sort_values(["symbol", "date"]).reset_index(drop=True)
    base_rate = prob_head._base_rate(tail)
    log(f"[C] base_rate (mfe_3d>=3%, 近20可观测日, 截至8/18) = {base_rate}")
    report["base_rate_818"] = base_rate
    del tail
    gc.collect()

    # ---------- D. 8/18 全截面 (宽列): 概率头 + legacy bundle ----------
    log("[D] 装配 8/18 全截面特征 ...")
    pb_par = load_prob_bundle_live_on(Path(PROB_GATE["model_dir"]), "dual", T_DATE)
    pb_leg = load_prob_bundle_live_on(
        Path(DATA_DIR / "prob_head_legacy"), "dual", T_DATE
    )
    if pb_par is None or pb_leg is None:
        log("[D][FATAL] 找不到 8/18 当日有效的概率头 bundle (parallel/legacy)")
        return 3
    log(
        f"[D] parallel 概率头: {pb_par[0].name} (trained_through={pb_par[1]['trained_through']}, "
        f"{len(pb_par[1]['feat_cols'])} 特征)"
    )
    log(
        f"[D] legacy 概率头: {pb_leg[0].name} (trained_through={pb_leg[1]['trained_through']}, "
        f"{len(pb_leg[1]['feat_cols'])} 特征)"
    )
    report["prob_bundles"] = {
        "parallel": {"file": pb_par[0].name, "trained_through": pb_par[1]["trained_through"]},
        "legacy": {"file": pb_leg[0].name, "trained_through": pb_leg[1]["trained_through"]},
    }

    import joblib

    leg_bundle = None
    if ram_gb() > 2.5:
        leg_bundle = joblib.load(LEGACY_BUNDLE)
        log(f"[D] legacy bundle {LEGACY_BUNDLE}: {len(leg_bundle['feature_cols'])} 特征")
    else:
        log("[D][WARN] RAM < 2.5GB → 跳过 legacy bundle (只跑概率头)")

    need = list(
        dict.fromkeys(
            ["date", "symbol", "amount", "close_hfq", "high_hfq", "adv20"]
            + pb_par[1]["feat_cols"]
            + pb_leg[1]["feat_cols"]
            + (leg_bundle["feature_cols"] if leg_bundle else [])
            + pool_feats
        )
    )
    have = set(d.schema.names)
    missing = [c for c in need if c not in have]
    if missing:
        log(f"[D][WARN] 检查点缺 {len(missing)} 列 (前10): {missing[:10]}")
    need = [c for c in need if c in have]
    x818 = d.to_table(filter=ds.field("date") == T_DATE, columns=need).to_pandas()
    x818["date"] = pd.to_datetime(x818["date"])
    log(f"[D] 8/18 截面 {len(x818)} 只 × {len(x818.columns)} 列")
    report["n_cross_818"] = int(len(x818))
    r911 = x818[x818["symbol"] == TARGET]
    if len(r911) == 0:
        log("[D][FATAL] 8/18 截面无 300911 (E6/底线剔除?) → 池层根因成立, 模型层无法评")
        report["in_pool_full_spectrum"] = False
        return 0
    report["in_pool_full_spectrum"] = True

    # ---------- E. 概率头: 300911 vs 闸 ----------
    for tag, pb in (("parallel", pb_par), ("legacy", pb_leg)):
        b = pb[1]
        X = x818[b["feat_cols"]].to_numpy(dtype="float32")
        prob = b["model"].predict_proba(X)[:, 1]
        xprob = x818.assign(_p=prob)
        p911 = float(xprob.loc[xprob["symbol"] == TARGET, "_p"].iloc[0])
        gate = (base_rate or 0) + PROB_GATE["margin"]
        pct = float((xprob["_p"] < p911).mean())
        log(
            f"[E.{tag}] 300911 prob={p911:.3f} | base={base_rate:.3f} 闸=base+0.08={gate:.3f} "
            f"{'PASS' if p911 > gate else 'FAIL'} | 截面分位 {pct:.2f}"
        )
        report[f"prob_{tag}"] = {
            "prob": p911,
            "gate": gate,
            "pass_": bool(p911 > gate),
            "pct": pct,
        }
        if tag == "legacy":
            # SHAP: legacy 概率头对 300911 的 top 贡献 (8/18 行)
            X911 = r911[b["feat_cols"]].to_numpy(dtype="float32")
            try:
                contrib = b["model"].predict(X911, pred_contrib=True)[0]
                imp = pd.Series(contrib[:-1], index=b["feat_cols"])
                top = pd.concat([imp.nlargest(10), imp.nsmallest(10)])
                log("[E.legacy SHAP] 300911 概率头 top±10 贡献:")
                for k, v in top.items():
                    log(f"    {k}: {v:+.4f}")
                report["legacy_prob_shap"] = {k: float(v) for k, v in top.items()}
            except Exception as e:
                log(f"[E.legacy SHAP] 失败: {e}")
        del X, prob, xprob
        gc.collect()

    # ---------- F. legacy bundle 全模型打分 (近似口径: 全谱 xrank) ----------
    if leg_bundle is not None:
        from app.pipeline1.predictor import V35Predictor

        pred = V35Predictor({"dual": LEGACY_BUNDLE}).predict(x818, "dual")
        p = pred[pred["symbol"] == TARGET]
        if len(p) == 0:
            log("[F][FATAL] legacy bundle 对 300911 无预测行")
        else:
            x = p.iloc[0]
            med = pred.select_dtypes("number").median(numeric_only=True)
            log("[F] legacy dual_20260818 对 300911 (8/18, 全谱池近似口径):")
            key_cols = ["pred_ret_3d", "pred_ret_5d", "pred_ret_10d", "prob_up", "pain_prob"]
            for c in key_cols:
                if c in pred.columns:
                    pct = float((pred[c] < x[c]).mean())
                    log(
                        f"    {c}: {x[c]:+.4f} (截面中位 {med.get(c, float('nan')):+.4f}, "
                        f"分位 {pct:.2f})"
                    )
            report["legacy_pred_911"] = {
                c: (float(x[c]) if c in pred.columns else None) for c in key_cols
            }
            report["legacy_pred_pct"] = {
                c: float((pred[c] < x[c]).mean())
                for c in key_cols
                if c in pred.columns
            }
            # 8/18 已交付清单对照 (neg200 版): pred_ret_10d 最低 ~0.078 / pain<0.4
            # ---------- G. SHAP: 10d_reg (排名键) ----------
            models = leg_bundle["models"]
            cols = leg_bundle["feature_cols"]
            X911 = np.nan_to_num(
                r911[cols].to_numpy(dtype="float64"), nan=0.0
            )
            for mkey in ("10d_reg", "3d_reg"):
                if mkey not in models:
                    continue
                try:
                    contrib = models[mkey][0].predict(X911, pred_contrib=True)[0]
                    imp = pd.Series(contrib[:-1], index=cols)
                    top = pd.concat([imp.nlargest(12), imp.nsmallest(12)])
                    log(f"[G] 300911 {mkey} SHAP top±12:")
                    for k, v in top.items():
                        log(f"    {k}: {v:+.5f}")
                    report[f"legacy_{mkey}_shap"] = {k: float(v) for k, v in top.items()}
                except Exception as e:
                    log(f"[G] {mkey} SHAP 失败: {e}")
        del pred
        gc.collect()

    # ---------- H. 用户维度特征盘点: 模型看到了什么 (8/18) ----------
    log("[H] 300911 8/18 筹码/影线/洗盘/K线/价格位置特征值 + 截面分位:")
    all_cols = list(x818.columns)
    groups = {
        "筹码": [c for c in all_cols if "chip" in c.lower() or "获利" in c or c.startswith("A04")],
        "影线": [c for c in all_cols if "shadow" in c.lower()],
        "洗盘/吸筹/拉高/出货": [
            c
            for c in all_cols
            if any(k in c for k in ("洗盘", "吸筹", "拉高", "出货", "见顶", "底部区域"))
        ],
        "K线形态": [
            c
            for c in all_cols
            if any(k in c for k in ("hammer", "engulfing", "star", "intensity", "big_white", "big_black"))
        ],
        "价格位置": [
            c
            for c in all_cols
            if any(k in c for k in ("close_position", "MA250", "close_vs_low", "position", "drawdown"))
        ],
    }
    see = {}
    for gname, gcols in groups.items():
        for c in gcols:
            if c not in x818.columns:
                continue
            v = r911[c].iloc[0]
            if pd.isna(v):
                continue
            pct = float((x818[c] < v).mean())
            see[f"{gname}|{c}"] = {"value": float(v), "pct": pct}
            log(f"    [{gname}] {c}: {float(v):+.4f} (分位 {pct:.2f})")
    report["feature_seen"] = see

    # ---------- 落盘 ----------
    out_dir = Path(BACKTEST_RESULT_DIR) / (
        "_diag_300911_" + time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    log(f"\n[done] WORM -> {out_dir / 'report.json'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
