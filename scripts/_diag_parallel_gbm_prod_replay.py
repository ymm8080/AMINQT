"""_diag_parallel_gbm_prod_replay.py — 生产口径概率闸 250d 复验 (2026-08-15).

背景: 阶段2 wf 回测 (_diag_parallel_gbm_wf.py, 定案 0.08 边际) 的 base_rate 有两处
与生产不符, 本脚本用生产模块 app/pipeline_parallel/prob_head 的真实函数重跑同窗口:
  1. wf 旧口径: base = rolling(20, min_periods=20).mean().shift(1) 全面板日达标率 —
     窗口 [D-20, D-1] 含 D-3..D-1 三日, 其 mfe 需 D+1..D+3 价格 → 3 天 look-ahead
     (且尾 4 行 NaN>=0.03 在 pandas 为 False, 被计为 0% 达标)。
  2. 生产口径 (本脚本): 每日 D 取尾 35 日切片 [D-34, D] → prob_head._add_mfe_3d
     (切片尾部自然 NaN, 与生产当日面板同构) → prob_head._base_rate 剔 NaN 后取
     最近 20 个可观测日达标率均值 → 无前瞻 (只用 ≤ D 的价格)。
双基线同跑:
  - _wfb_* = 旧口径 + wf 语义 (pred 只算 prob_ok 行, NaN → 不过) → 逐字复现定案数字
    → 管线自检 (证明本脚本与阶段2 同口径, 差异只来自基线算法)
  - _prb_* = 生产口径 + 生产 fail-open 语义 (全截面预测, NaN → 保留) → 诚实结论
训练/评估与阶段2 完全一致: 扩窗 (每 21 交易日重训, 全史), 特征=prob_head.feature_cols,
闸加在 t3 门后 pred_mag_10d TOP-5 前, 指标=label_pm_10d_net 实得/命中/≥5%/≥10% + 4 子窗。

用法: python scripts/_diag_parallel_gbm_prod_replay.py
注意: 与 daily automation 错峰运行 (双任务并发必 OOM)。
检查点: data/_diag_replay_wf_pred_<board>.parquet (walk-forward 预测落盘, 崩溃后重跑免重训).
WORM: data/_diag_parallel_gbm_prod_replay_<ts>.csv/.json
"""

from __future__ import annotations

import gc
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier

from app.pipeline_parallel import prob_head
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

EVAL_DAYS = 250
TOPN = 5
T3_LANDED = {"main": 0.0, "dual": 0.005}
ABS_TARGET = 0.03  # 与 prob_head.PROB_GATE["abs_target"] 一致
BASE_RATE_DAYS = 20
REFIT_EVERY = 21
MARGINS = (0.04, 0.06, 0.08, 0.10)


def _load_board(board: str) -> pd.DataFrame | None:
    """同阶段2 wf: 全特征 + score + mfe_3d + 标签 (行按日期排序)."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    schema = pq.read_schema(str(fp)).names
    need = [
        c
        for c in schema
        if not c.startswith("label_")
        and c not in prob_head.META
        and not c.startswith("pred_")
    ]
    need += ["symbol", "date", "label_pain", "label_pm_3d_net", "label_pm_10d_net"]
    t = pq.read_table(str(fp), columns=list(dict.fromkeys(need))).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    t = prob_head._add_mfe_3d(t)
    return t


def _prod_base_series(t: pd.DataFrame, dates: np.ndarray) -> pd.Series:
    """生产口径 base_rate (无前瞻): 每日 D 用尾 35 日切片, 剔 NaN 后取近 20 可观测日.

    t 需含 symbol/date/close_hfq/high_hfq/adv20; _add_mfe_3d 在切片尾部自然 NaN
    (与生产当日面板同构 — mfe 需 +4 交易日未来价), _base_rate 先剔 NaN 再求均值.
    """
    base_map: dict[pd.Timestamp, float] = {}
    pos_all = np.searchsorted(dates, t["date"].values)
    for k, d in enumerate(dates):
        if k < BASE_RATE_DAYS + 14:
            continue
        rows = np.where((pos_all >= k - BASE_RATE_DAYS - 14) & (pos_all <= k))[0]
        tail = t.iloc[rows][["symbol", "date", "close_hfq", "high_hfq", "adv20"]].copy()
        b = prob_head._base_rate(tail)
        if b is not None:
            base_map[pd.Timestamp(d)] = b
    return pd.Series(base_map, name="base_prod")


def _eval_top5(
    mm: pd.DataFrame, gate_col: str | None, t3: float, days: list, nan_keep: bool
) -> tuple[dict, pd.DataFrame]:
    """生产同款 TOP-5 (t3 门 + pred_mag_10d 排名) + 可选真模型闸.

    nan_keep=True → 生产 fail-open (pred_prob NaN → 保留);
    nan_keep=False → 阶段2 wf 语义 (NaN 比较 → False → 剔除, 复现定案数字).
    """
    g = mm[mm["pred_ret_3d"] > t3]
    if gate_col:
        keep = g[gate_col]
        if nan_keep:
            keep = keep | g["pred_prob"].isna()
        g = g[keep]
    top = (
        g.sort_values(["date", "pred_mag_10d"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(TOPN)
    )
    n = len(top)
    row = {
        "rows": n,
        "days_with_picks": int(top["date"].nunique()),
        "picks_per_day": n / len(days),
        "realized_10d": float(top["label_pm_10d_net"].mean()) if n else float("nan"),
        "hit_10d": float((top["label_pm_10d_net"] > 0).mean()) if n else float("nan"),
        "pct_ge5pct": float((top["label_pm_10d_net"] >= 0.05).mean())
        if n
        else float("nan"),
        "pct_ge10pct": (
            float((top["label_pm_10d_net"] >= 0.10).mean()) if n else float("nan")
        ),
    }
    return row, top


def _sub_windows(top: pd.DataFrame, days: list, n_sub: int) -> list[dict]:
    """4 子窗命中/实得 (稳定性检验)."""
    step = len(days) // n_sub
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(days) if i == n_sub - 1 else (i + 1) * step
        seg = top[top["date"].isin(days[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "rows": int(len(seg)),
                "hit10": float((seg["label_pm_10d_net"] > 0).mean())
                if len(seg)
                else float("nan"),
                "mean10": float(seg["label_pm_10d_net"].mean())
                if len(seg)
                else float("nan"),
            }
        )
    return subs


def main() -> int:
    rows_out: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            continue
        dates = np.unique(t["date"].values)
        cal_test = dates[-EVAL_DAYS:]
        feat_cols = prob_head.feature_cols(t)
        y_prob = (t["mfe_3d"] >= ABS_TARGET).astype(float)
        prob_ok = y_prob.notna() & t["label_pain"].notna()
        idx = np.searchsorted(dates, t["date"].values)

        # ---- 内存裁剪 (2026-08-16): 面板扩建后 main 1.63M 行宽表在末次扩窗重训时
        # 因 int64 块拷贝 201MiB 分配失败 (_ArrayMemoryError) 崩溃.
        # 只留训练/评估所需列; 特征列统一 float32 (与循环内 to_numpy(dtype="float32")
        # 位级一致, 不改变任何数值); 价格列供 _prod_base_series/_base_rate 使用.
        keep = [
            "symbol",
            "date",
            "board",
            "score",
            "close_hfq",
            "high_hfq",
            "adv20",
            "mfe_3d",
            "label_pain",
            "label_pm_3d_net",
            "label_pm_10d_net",
        ] + feat_cols
        # 不 copy: t[keep].copy() 触发 pandas block consolidation, 257 列 x 161万行
        # float64 需 3.08GiB 连续块 (2026-08-16 第三次 _ArrayMemoryError 在此);
        # 列选择共享原块引用, 后续 astype 只替换 feat_cols 列块.
        t = t[keep]
        t[feat_cols] = t[feat_cols].astype("float32")
        # 特征矩阵一次性转 numpy (t 全程不变): 循环内 t.loc[tr, feat_cols] 每次
        # 分配 int64 块拷贝 (22列×160万×8B≈283MiB), 碎片化时 OOM (2026-08-16 两次
        # _ArrayMemoryError 均在此). numpy 布尔索引无 int64 taker, 峰值减半.
        feat_arr = t[feat_cols].to_numpy(dtype="float32")
        gc.collect()

        # ---- walk-forward 扩窗重训 (生产配方, 每 21 交易日) ----
        # pred   = 全截面预测 (生产 gate_probabilities 语义, 任何行都给概率)
        # pred_wf = 仅 prob_ok 行 (阶段2 wf 语义, 其余 NaN → 旧闸剔除)
        ckpt = DATA_DIR / f"_diag_replay_wf_pred_{board}.parquet"
        if ckpt.exists():
            cp = pq.read_table(str(ckpt)).to_pandas()
            pred = pd.Series(cp["pred"].to_numpy(), index=t.index, dtype="float64")
            pred_wf = pd.Series(
                cp["pred_wf"].to_numpy(), index=t.index, dtype="float64"
            )
            print(
                f"[{board}] walk-forward 从检查点恢复: 测试 {len(cal_test)} 日, "
                f"特征 {len(feat_cols)}",
                flush=True,
            )
        else:
            model = None
            pred = pd.Series(np.nan, index=t.index, dtype="float64")
            pred_wf = pd.Series(np.nan, index=t.index, dtype="float64")
            n_refits = 0
            for k, d in enumerate(cal_test):
                pos = len(dates) - EVAL_DAYS + k
                if model is None or pos % REFIT_EVERY == 0:
                    tr = ((idx < pos) & prob_ok).to_numpy()
                    x = feat_arr[tr]
                    y = y_prob.loc[tr].to_numpy()
                    model = LGBMClassifier(**prob_head.LGB_PARAMS)
                    model.fit(x, y)
                    n_refits += 1
                te = t["date"].values == d
                if not te.any():
                    continue
                xd = feat_arr[te]
                if len(xd) == 0:
                    continue
                p = model.predict_proba(xd)[:, 1]
                pred.loc[te] = p
                pred_wf.loc[te & prob_ok] = p[prob_ok[te].to_numpy()]
            pd.DataFrame(
                {"pred": pred.to_numpy(), "pred_wf": pred_wf.to_numpy()}
            ).to_parquet(ckpt)
            print(
                f"\n[{board}] walk-forward 完成: {n_refits} 次扩窗重训 / "
                f"测试 {len(cal_test)} 日, 特征 {len(feat_cols)}",
                flush=True,
            )

        # ---- 评估 (同阶段2: t3 门 + pred_mag_10d TOP-5, label_pm_10d_net 实得) ----
        work = t[
            ["symbol", "date", "board", "score", "label_pm_3d_net", "label_pm_10d_net"]
        ].copy()
        p3 = calibrate_mag10d(work, target_col="label_pm_3d_net", label_horizon=3)
        p10 = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        mm = work.merge(
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        ).merge(
            p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=["label_pm_10d_net"])
        days = sorted(rr["date"].unique())[-EVAL_DAYS:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)

        sub = t.loc[np.isin(t["date"].values, cal_test), ["symbol", "date"]].copy()
        sub["pred_prob"] = pred[sub.index].to_numpy()
        sub["pred_prob_wf"] = pred_wf[sub.index].to_numpy()
        rr = rr.merge(sub, on=["symbol", "date"], how="left")

        # 旧口径 (wf): 全面板日达标率 rolling20 shift1 (带 3 天 look-ahead, 复现定案数字)
        daily_rate = (
            t.assign(_hit=(t["mfe_3d"] >= ABS_TARGET).astype(float))
            .groupby("date")["_hit"]
            .mean()
        )
        base_wf = (
            daily_rate.rolling(BASE_RATE_DAYS, min_periods=BASE_RATE_DAYS)
            .mean()
            .shift(1)
            .rename("base_wf")
        )
        rr = rr.merge(base_wf, left_on="date", right_index=True, how="left")
        # 生产口径 (诚实): 每决策日尾 35 日切片 _base_rate (无前瞻)
        rr = rr.merge(
            _prod_base_series(t, dates), left_on="date", right_index=True, how="left"
        )

        for m in MARGINS:
            rr[f"_wfb_{m}"] = rr["pred_prob_wf"] > rr["base_wf"] + m
            rr[f"_prb_{m}"] = rr["pred_prob"] > rr["base_prod"] + m
        gates = (
            [("基线(关)", None, False)]
            + [(f"wfb+{m:.2f}", f"_wfb_{m}", False) for m in MARGINS]
            + [(f"prb+{m:.2f}", f"_prb_{m}", True) for m in MARGINS]
        )
        print(
            f"\n===== {board}  末 250 已实现交易日 (t3 门 {T3_LANDED[board]:.2%}) =====",
            flush=True,
        )
        print(
            f"{'闸':>14} {'票/日':>6} {'实得10d':>8} {'命中10d':>7} "
            f"{'≥+5%':>6} {'≥+10%':>6}",
            flush=True,
        )
        n_sub = max(2, EVAL_DAYS // 60)
        for gname, gcol, nan_keep in gates:
            r, top = _eval_top5(rr, gcol, T3_LANDED[board], days, nan_keep)
            r["board"] = board
            r["gate"] = gname
            r["sub_windows"] = _sub_windows(top, days, n_sub)
            rows_out.append(r)
            sub_s = "  ".join(
                f"{s['win']}:{s['hit10']:.0%}/{s['mean10']:+.2%}"
                for s in r["sub_windows"]
            )
            print(
                f"{gname:>14} {r['picks_per_day']:>6.2f} {r['realized_10d']:>+8.2%} "
                f"{r['hit_10d']:>7.0%} {r['pct_ge5pct']:>6.0%} "
                f"{r['pct_ge10pct']:>6.0%}",
                flush=True,
            )
            print(f"    sub: {sub_s}", flush=True)

    if not rows_out:
        print("[error] 无任何板块可评估", flush=True)
        return 1
    df = pd.DataFrame(rows_out)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_parallel_gbm_prod_replay_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_parallel_gbm_prod_replay_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": EVAL_DAYS,
                "topn": TOPN,
                "refit_every": REFIT_EVERY,
                "margins": list(MARGINS),
                "note": "wfb=阶段2旧口径(rolling20 shift1, 3天look-ahead, NaN剔除); "
                "prb=生产诚实口径(_base_rate, 只观测≤D-4, NaN保留)",
                "rows": df.to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
