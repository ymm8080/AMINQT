"""_diag_gap_pick_eval.py — 停牌缺口污染 vs 干净票的 TOP-5 命中率对比 (2026-08-14).

背景: 面板停牌行在构建时被删 (is_suspended 全 False, cleaning_pipeline 读面板
时按 is_suspended=False 过滤), 所有滚动类特征 (ret60/rps_60/ADX 链/sharpe_20/
ma 族 + sniper/fusion 池特征) 按行序跨缺口桥接 — 停牌复牌后的前 60 行特征
混入停牌前的陈数据。暴露量化 (_diag_st_gap_exposure, 08-14): dual 53.6% /
main 35.0% 行位于缺口后 60 行内; 交付短名单 ST 命中 = 0。

本脚本回答"钱的问题": 生产同款 250d OOS TOP-5 里, 特征被缺口污染的票
vs 干净票, 谁 10d 命中高/实得多。污染票占比 = 修复 (gap-aware rolling)
的潜力上限; 若污染票不输干净票, 则桥接无害, 不修。

生产口径复现 (同 _diag_t3min_sweep / _diag_parallel_pain_gate):
- 3y 诊断面板截 n_tail, pool score = max(sniper, fusion)
- calibrate_mag10d (21d 纯横截面 OLS walk-forward) → pred_ret_3d / pred_mag_10d
- 入选门 pred_ret_3d > t3_min {main:0, dual:0.5%} (08-14 定案, 生产已落地)
- 排名键 pred_mag_10d, 每板块日 TOP-5, 末 250 已实现交易日, 4 子窗稳定性
- 每行附 gap60 旗标 (缺口后 60 行内, 交易日跳日>1) + label_gap10 旗标
  ([T,T+10] 标签窗口内跨缺口 — 标签按行序 shift, 停牌行被删后窗口漂移;
  生产 mask_suspension 依赖 is_suspended 列=死掩码, 见 label_engine.py:278
  与 _reclassify_all_features.add_label_pm_10d_net) + st 旗标
  (Tushare namechange ST/*ST/退 期间)
- 分组: 基线(全部) / 干净(~gap&~st) / 缺口(gap&~st) / ST(st) / 标签跨缺口 —
  干净 vs 缺口 的命中率/实得差 = 特征修复价值; 标签跨缺口组 = 标签修复价值

WORM 输出 data/_diag_gap_pick_eval_<ts>.csv/.json。

用法: python scripts/_diag_gap_pick_eval.py [--eval-days=250]
注意: 与 daily automation (23:30 北京) 错峰运行 — 双任务并发必 OOM。
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

POOL_COLS = sorted({c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"})
EVAL_DAYS = 250
TOPN = 5  # 2026-08-14 定案: 每板块 TOP-5
T3_LANDED = {"main": 0.0, "dual": 0.005}  # 08-14 t3_min 定案 (生产已落地)
GAP_WINDOW = 60  # ret60/rps_60 等 60d 类特征污染带 (5/20d 类为其子集)
HORIZONS = ("3d", "5d", "10d")
LABEL = {h: f"label_pm_{h}_net" for h in HORIZONS}
NAME_CACHE = DATA_DIR / "supply_cache" / "namechange"


def _st_periods() -> pd.DataFrame:
    """namechange 缓存 → ST/*ST/退 期间表 [symbol, start, end, name]."""
    files = sorted(glob.glob(str(NAME_CACHE / "namechange_full_*.parquet")))
    if not files:
        return pd.DataFrame(columns=["symbol", "start", "end", "name"])
    nc = pd.read_parquet(files[-1])
    nc["symbol"] = nc["ts_code"].str.split(".").str[0]
    bad = nc[nc["name"].str.contains("ST|退", na=False)].copy()
    bad["start"] = pd.to_datetime(bad["start_date"])
    bad["end"] = pd.to_datetime(bad["end_date"]).fillna(pd.Timestamp("2027-01-01"))
    return bad[["symbol", "start", "end", "name"]].reset_index(drop=True)


def _hit_mask(symbols: pd.Series, dates: pd.Series, periods: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=symbols.index)
    for _, r in periods.iterrows():
        m |= (symbols == r["symbol"]) & (dates >= r["start"]) & (dates <= r["end"])
    return m


def _gap_mask(t: pd.DataFrame) -> pd.Series:
    """行级: 是否位于停牌缺口后 GAP_WINDOW 行内 (滚动特征跨缺口桥接)."""
    cal = np.unique(pd.to_datetime(t["date"]).values)
    idx = np.searchsorted(cal, pd.to_datetime(t["date"]).values)
    skip = pd.Series(idx, index=t.index).groupby(t["symbol"]).diff().fillna(1.0)
    roll = skip.groupby(t["symbol"]).transform(
        lambda s: s.rolling(GAP_WINDOW, min_periods=1).max()
    )
    return roll > 1


def _label_gap_mask(t: pd.DataFrame, k: int = 10) -> pd.Series:
    """行级: [T, T+k] 标签窗口内是否跨停牌缺口 (交易日跳日>1).

    标签按行序 shift 计算 (close[T+1+k] = 第 k+1 行后的收盘), 停牌行被删后
    窗口跨越停牌段 → 标签衡量的是"含停牌的持有", 非原定 k 个交易日;
    且生产 mask_suspension 依赖 is_suspended 列 (面板全 False, 死掩码)。
    """
    cal = np.unique(pd.to_datetime(t["date"]).values)
    idx = np.searchsorted(cal, pd.to_datetime(t["date"]).values)
    skip = pd.Series(idx, index=t.index).groupby(t["symbol"]).diff().fillna(1.0)
    cands = pd.concat(
        [skip.groupby(t["symbol"]).shift(-j) for j in range(1, k + 1)], axis=1
    )
    return cands.max(axis=1, skipna=True) > 1


def _load_board(board: str, n_tail: int, periods: pd.DataFrame) -> pd.DataFrame | None:
    """同 _diag_t3min_sweep 载入 + gap/st 旗标."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
    uniq = np.unique(dates.values)
    if len(uniq) < n_tail + 20:
        return None
    cutoff = uniq[-(n_tail + 20)]
    need = ["symbol", "date"] + POOL_COLS + list(LABEL.values())
    t = pq.read_table(
        str(fp), columns=need, filters=[("date", ">=", cutoff)]
    ).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    t["date"] = pd.to_datetime(t["date"])
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    t["gap60"] = _gap_mask(t[["symbol", "date"]])
    t["label_gap10"] = _label_gap_mask(t)
    t["st"] = _hit_mask(t["symbol"], t["date"], periods)
    return t.copy()


def _eval_group(top: pd.DataFrame, name: str, days: list, n_sub: int) -> dict:
    n = int(len(top))
    n_days = int(top["date"].nunique())
    row = {
        "board": str(top["board"].iloc[0]) if n else "",
        "group": name,
        "rows": n,
        "days_with_picks": n_days,
        "picks_per_day": n / len(days),
    }
    for h in HORIZONS:
        col = LABEL[h]
        row[f"realized_{h}"] = float(top[col].mean()) if n else float("nan")
        row[f"hit_{h}"] = float((top[col] > 0).mean()) if n else float("nan")
    row["pct_ge5pct"] = float((top[LABEL["10d"]] >= 0.05).mean()) if n else float("nan")
    row["pct_ge10pct"] = (
        float((top[LABEL["10d"]] >= 0.10).mean()) if n else float("nan")
    )
    # 4 子窗命中 (稳定性)
    step = len(days) // n_sub
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(days) if i == n_sub - 1 else (i + 1) * step
        seg = top[top["date"].isin(days[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "rows": int(len(seg)),
                "hit10": float((seg[LABEL["10d"]] > 0).mean()) if len(seg) else float("nan"),
                "mean10": float(seg[LABEL["10d"]].mean()) if len(seg) else float("nan"),
            }
        )
    row["sub_windows"] = subs
    return row


def main() -> int:
    _eval_days = EVAL_DAYS
    _args = [a for a in sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    n_tail = _eval_days + 40
    n_sub = max(2, _eval_days // 60)
    periods = _st_periods()
    all_rows: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board, n_tail, periods)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            continue
        work = t[["symbol", "date", "board", "score"] + list(LABEL.values())].copy()
        p3 = calibrate_mag10d(work, target_col=LABEL["3d"], label_horizon=3)
        p10 = calibrate_mag10d(work, target_col=LABEL["10d"], label_horizon=10)
        mm = work.merge(
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm = mm.merge(
            p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        mm = mm.merge(
            t[["symbol", "date", "gap60", "label_gap10", "st"]],
            on=["symbol", "date"],
        )
        rr = mm.dropna(subset=[LABEL["10d"]])
        days = sorted(rr["date"].unique())[-_eval_days:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)

        # 生产同款 TOP-5
        g = rr[rr["pred_ret_3d"] > T3_LANDED[board]]
        top = (
            g.sort_values(["date", "pred_mag_10d"], ascending=[True, False])
            .groupby("date", sort=True)
            .head(TOPN)
        )
        n_gap = int(top["gap60"].sum())
        n_st = int(top["st"].sum())
        n_lgap = int(top["label_gap10"].sum())
        print(
            f"\n===== {board}  TOP-5 共 {len(top)} 票: "
            f"特征缺口污染 {n_gap} ({n_gap / max(len(top), 1):.0%}), "
            f"标签窗口跨缺口 {n_lgap} ({n_lgap / max(len(top), 1):.0%}), "
            f"ST 期间 {n_st} ({n_st / max(len(top), 1):.0%}) =====",
            flush=True,
        )
        groups = [
            ("基线(全部)", top),
            ("干净(~gap&~st)", top[~(top["gap60"]) & ~(top["st"])]),
            ("缺口(gap&~st)", top[top["gap60"] & ~(top["st"])]),
            ("ST(st)", top[top["st"]]),
            ("标签跨缺口(lgap10)", top[top["label_gap10"]]),
        ]
        for name, seg in groups:
            r = _eval_group(seg, name, days, n_sub)
            all_rows.append(r)
            subs = "  ".join(
                f"{s['win']}:{s['hit10']:.0%}/{s['mean10']:+.2%}" for s in r["sub_windows"]
            )
            print(
                f"{name:>14} {r['picks_per_day']:>7.2f} 票/日  "
                f"实得10d {r['realized_10d']:>+8.2%}  命中10d {r['hit_10d']:>7.0%}  "
                f"≥+5% {r['pct_ge5pct']:>6.0%}  ≥+10% {r['pct_ge10pct']:>6.0%}",
                flush=True,
            )
            print(f"    sub: {subs}", flush=True)

    if not all_rows:
        print("[error] 无任何板块可评估", flush=True)
        return 1
    df = pd.DataFrame(all_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_gap_pick_eval_{_eval_days}d_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_gap_pick_eval_{_eval_days}d_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": _eval_days,
                "topn": TOPN,
                "gap_window": GAP_WINDOW,
                "t3_landed": T3_LANDED,
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
