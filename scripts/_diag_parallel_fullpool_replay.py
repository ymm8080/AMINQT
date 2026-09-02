"""_diag_parallel_fullpool_replay.py — 并行模块全池逐日回放 (当前最优包, 2026-09-02).

背景: 影子排名 (xmodule_shadow) 首版判死用的是历史交付小池; 用户明令重判必须
「全池 backtest + 双臂=当前最优包重放 TOP10」(先例: q90_slot_replay 的 legacy
125d 当前 predictor 重放). 本脚本产 parallel 臂的全池明细, 与 legacy 重放
(q90_slot_replay_<ts>.parquet) 在共同日期上做 blend 回测.

重放口径 (对齐生产 build_merged_shortlist + 交付 rank 键):
  score      = max(pool_score(SNIPER.pool), pool_score(FUSION.pool)) 截面分位分
  pred_mag   = calibrate_mag10d(score→label_pm_10d_net) walk-forward (生产排名键,
               内部只吃已实现标签, 无前瞻)
  pred_prob  = Platt P(label_mfe_10d_net >= 0.06) — 与生产 ABS_TARGET["10d"] 同靶.
               偏差 1: 生产用 OOS 清单 records 拟合 (回测期数据), 全池重放改用
               **池内已实现行** 每 21 交易日 walk-forward 重拟合 (诚实无前瞻).
  rank_blend = pred_mag × pred_prob (生产 CAND_RANK_KEY 语义)
  实得       = buy D+1 close / sell D+(1+h) close − COST 0.0020 (与 q90 重放同口径)

偏差声明 (相对生产交付链, 均为防前瞻/防路径依赖):
  - LGBM prob_head 闸省略 (现网 bundle trained_through=今日, 套历史=泄漏)
  - EMA 平滑 / 迟滞滞留 / 制度门 / 解禁过滤省略 (路径依赖, 非排名键)
  - pv_corr_5 检查点缺列自动跳过 (与生产 pool_score 行为一致)

口径警告: 当前包含训练期数据的检查点重放历史, 绝对水平上偏; 三臂 (legacy 重放 /
parallel 重放 / blend) 共用同一预测, 相对 delta 可比. 结论须与 08-06 起诚实生产
清单窗互证.

WORM: DATA_OTHERS/diag/parallel_fullpool_replay_<ts>.parquet + .json (meta)

用法:
  python scripts/_diag_parallel_fullpool_replay.py               # --slice 420 --eval 125
  python scripts/_diag_parallel_fullpool_replay.py --eval 10     # 冒烟
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

from app.pipeline_parallel.backtest import add_mfe_labels, tradability_gate
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, PANEL, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import PANEL_V3_PATH, data_others_path
from scripts._reclassify_all_features import _finalize_slice

COST = 0.0020  # 与 q90 重放一致: 佣金+印花税+滑点 ≈ 0.2% 往返
PROB_TARGET = 0.06  # 生产 ABS_TARGET["10d"]: P(mfe_10d 净 >= 6%)
PROB_REFIT_EVERY = 21  # 交易日 (对齐 prob_head 生产重训节奏)
PROB_REFIT_FROM = 40  # 首次拟合前最少历史交易日
PROB_MIN_ROWS = 500  # 拟合最小行数, 不足该 epoch 无 prob (rank 退化纯 mag)


def _pivots(panel: pd.DataFrame):
    """symbol×date 宽表: close_hfq (ffill) — 与 _diag_q90_slot_replay 同构."""
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
    px.index = px.index.astype(str).str.zfill(6)
    return px, cal


def _panel_pivots(cutoff):
    """实得价基座 = 原始面板窄读 (symbol/date/close_hfq) — 与 legacy 重放逐字同源.

    检查点行是清洗后行集 (低流动性行被剔), 其 ffill 价基会在断行日给出陈旧买价
    (dry-run 实测与 legacy 重放 net3 最大差 11pp) → 必须用原始面板, 三臂实得才同价.
    """
    panel = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq"],
        filters=[("date", ">=", cutoff)],
    )
    return _pivots(panel)


def _net_vec(px: pd.DataFrame, symbols: pd.Series, buy_dt, sell_dt) -> np.ndarray:
    pb = px[buy_dt].reindex(symbols).to_numpy(dtype=float)
    ps = px[sell_dt].reindex(symbols).to_numpy(dtype=float)
    out = ps / pb - 1.0 - COST
    out[~(pb > 0)] = np.nan
    return out


def _platt_probs(work: pd.DataFrame) -> pd.Series:
    """每 21 交易日 walk-forward 拟合 Platt: score → P(mfe_10d_net >= 0.06).

    拟合只用 refit 日已实现行 (label_mfe_10d_net 非 NaN = 卖价已打印, 无前瞻);
    拟合结果应用到 [refit 日, 下个 refit 日) 的全部行. 返回与 work 对齐的 Series.
    """
    from sklearn.linear_model import LogisticRegression

    dates = np.sort(work["date"].unique())
    out = pd.Series(np.nan, index=work.index)
    {d: i for i, d in enumerate(dates)}
    dv = work["date"].to_numpy()
    is_fit_col = work["label_mfe_10d_net"].notna().to_numpy()
    x_all = work["score"].to_numpy(dtype=float).reshape(-1, 1)
    y_all = (work["label_mfe_10d_net"].to_numpy(dtype=float) >= PROB_TARGET).astype(int)
    lr = None
    for i, d in enumerate(dates):
        if i >= PROB_REFIT_FROM and i % PROB_REFIT_EVERY == 0:
            m = (dv <= d) & is_fit_col & np.isfinite(x_all[:, 0])
            if int(m.sum()) >= PROB_MIN_ROWS:
                lr = LogisticRegression()
                lr.fit(x_all[m], y_all[m])
        if lr is None:
            continue
        rows = np.nonzero(dv == d)[0]
        out.iloc[rows] = lr.predict_proba(x_all[rows])[:, 1]
    return out


def process_board(board: str, cutoff, eval_n: int, t0: float, px, cal) -> pd.DataFrame:
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    print(f"[{board}] read {ckpt} cutoff>={cutoff}", flush=True)
    df = pd.read_parquet(ckpt, filters=[("date", ">=", cutoff)])
    print(f"[{board}] raw {len(df):,}r ({time.time() - t0:.0f}s)", flush=True)
    df = _finalize_slice(df)
    df = add_mfe_labels(df, horizons=(10,), already_sorted=True)
    df, gate = tradability_gate(df)
    print(
        f"[{board}] gate -{gate['removed_rows']:,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )
    df["board"] = board

    # 全池 score = max(狙击, 融合) 截面分位分 (生产 build_merged_shortlist 同式)
    score_s = pool_score(df, SNIPER.pool)
    score_f = pool_score(df, FUSION.pool)
    scored = df[
        [
            "symbol",
            "date",
            "board",
            "close_hfq",
            "label_pm_10d_net",
            "label_mfe_10d_net",
        ]
    ].copy()
    scored["score"] = np.maximum(score_s.values, score_f.values)
    del score_s, score_f
    gc.collect()
    scored = scored.dropna(subset=["score"]).reset_index(drop=True)
    print(f"[{board}] scored {len(scored):,}r ({time.time() - t0:.0f}s)", flush=True)

    mag = calibrate_mag10d(scored, score_col="score", target_col="label_pm_10d_net")
    print(f"[{board}] mag {len(mag):,}r ({time.time() - t0:.0f}s)", flush=True)
    scored = scored.merge(
        mag[["symbol", "date", "mag"]], on=["symbol", "date"], how="inner"
    )
    del mag
    gc.collect()

    scored["prob"] = _platt_probs(scored).to_numpy()
    scored["rank_blend"] = scored["mag"] * scored["prob"]

    # 实得: 全池 (含未过闸行, 供 blend 回测统一取 net)
    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    max_lag = 11
    day_dates = sorted(pd.unique(pd.to_datetime(scored["date"])))
    eval_days = [
        d for d in day_dates if d in i_of and i_of[d] + max_lag < len(all_cal)
    ][-eval_n:]
    print(
        f"[{board}] eval {len(eval_days)}d | "
        f"{pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()}",
        flush=True,
    )

    sym_key = scored["symbol"].astype(str).str.zfill(6)
    out_frames = []
    for k, d in enumerate(eval_days):
        di = i_of[d]
        sub = scored[pd.to_datetime(scored["date"]) == d]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["symbol"] = sym_key[sub.index]
        b1, s3, s10 = all_cal[di + 1], all_cal[di + 4], all_cal[di + 11]
        sub["net_3d"] = _net_vec(px, sub["symbol"], b1, s3)
        sub["net_10d"] = _net_vec(px, sub["symbol"], b1, s10)
        sub["date"] = str(pd.Timestamp(d).date())
        out_frames.append(
            sub[
                [
                    "symbol",
                    "date",
                    "board",
                    "score",
                    "mag",
                    "prob",
                    "rank_blend",
                    "net_3d",
                    "net_10d",
                ]
            ]
        )
        if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
            n = sum(len(f) for f in out_frames)
            print(
                f"[{board}] {k + 1}/{len(eval_days)} rows={n:,} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    res = pd.concat(out_frames, ignore_index=True)
    del scored, df
    gc.collect()
    return res


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="读检查点末 N 交易日")
    ap.add_argument("--eval", type=int, default=125, help="评估窗交易日数")
    args = ap.parse_args()
    t0 = time.time()

    dts = pd.read_parquet(PANEL.main_checkpoint, columns=["date"])["date"].unique()
    cutoff = np.sort(pd.to_datetime(pd.Series(dts)))[-args.slice]
    print(f"[cutoff] {pd.Timestamp(cutoff).date()} (slice={args.slice})", flush=True)

    px, cal = _panel_pivots(cutoff)
    print(
        f"[panel-pivot] symbols={len(px)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    all_cal = pd.to_datetime(cal)

    frames = []
    for board in ("main", "dual"):
        frames.append(process_board(board, cutoff, args.eval, t0, px, all_cal))
    res = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    pq_path = out_dir / f"parallel_fullpool_replay_{ts}.parquet"
    res.to_parquet(pq_path, index=False)
    (out_dir / f"parallel_fullpool_replay_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "slice": args.slice,
                "eval": args.eval,
                "cost": COST,
                "prob_target": PROB_TARGET,
                "prob_refit_every": PROB_REFIT_EVERY,
                "checkpoints": {
                    "main": PANEL.main_checkpoint,
                    "dual": PANEL.dual_checkpoint,
                },
                "rows": int(len(res)),
                "days": int(res["date"].nunique()),
                "range": [str(res["date"].min()), str(res["date"].max())],
                "deviations": [
                    "prob=池内已实现行 Platt 每21交易日 walk-forward (生产=OOS清单Platt)",
                    "LGBM prob_head 闸省略 (现网 bundle 泄漏)",
                    "EMA/迟滞/制度门/解禁过滤省略 (路径依赖)",
                    "pv_corr_5 缺列自动跳过 (生产同)",
                    "实得价基=原始面板窄读 (与 q90 重放同源; 检查点清洗行断行日买价陈旧)",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {pq_path} rows={len(res):,} ({time.time() - t0:.0f}s)", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
