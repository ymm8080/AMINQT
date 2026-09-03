"""_diag_pool2stage_check.py — 三段式选股核算 (2026-09-03): 排名池 top5% → 剔下跌信号 → 上涨信号共选 TOP10.

动机 (用户 09-03): 生产 TOP10 抓不全真赢家 — 赢家集中在排名键 98 分位一带, head(10)
接不住. 三段式设计 (用户定稿): ① rank_blend 先取 top5% 候选池 (~46 席);
② 池内剔除有下跌信号的股票 (⑥ F3 / F1-F4 并集); ③ 幸存者里用涨前预热指标
(④ A1/A3/A4) 共同选出 TOP10.
与已判死的 keep 模式区别: keep=全池只留 A top10% (会捡到排名 50~200 名的股票, 等于
放弃排名键); 本设计边界锚定排名 top5% (最差捡到第 46 名) — 只在近赢家区内重排.
单段法 (a*, veto_*) 保留作 stage 消融, s3_* = 完整三段式主方法.

数据: time_decay_ab_20260903_035003.parquet (rank_blend_base + net_3d, 250d 双板全池)
    + 面板 550d 秩矩阵 (与 preflush/flushfilter 同口径).
指标: 各方法 TOP10 净收益/日 vs 生产, D3/D5/D10 三视界 (2026-09-03 用户质疑
    A1-A4 有效期可能只在 3-5 日, 补 D5/D10 对拍; 首跑只算 D3), 半窗拆分, 双板;
    赢家 (net_3d>=5%) 捕获数/日; 赢家排名分位分布与 top5% 池当日覆盖率.

WORM: DATA OTHERS/diag/pool2stage_check_<ts>.parquet + .json
用法: python scripts/_diag_pool2stage_check.py
    python scripts/_diag_pool2stage_check.py --src q90_slot_replay_20260902_100826.parquet --rank pred_ret_10d --tag _legacy
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

TOP_N = 10
POOL_PCT = 0.05
WIN_THR = 0.05
COST = 0.0020
SRC_PQ = "time_decay_ab_20260903_035003.parquet"

MOM_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
HOTVOL_COLS = ("amount", "turnover_rate", "volume_ratio", "ma_vol_ratio_5_20")
VOL_COLS_A = (*HOTVOL_COLS, "free_float_turnover_rate")  # ④ A2 口径 5 列
F4_COLS = (*MOM_COLS, "peak_roc_20d")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default=SRC_PQ)
    ap.add_argument("--rank", type=str, default="rank_blend_base")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    src = pd.read_parquet(str(data_others_path("diag") / args.src))
    src["date"] = src["date"].astype(str)
    print(
        f"[src] {len(src):,} rows, {src['date'].nunique()} days ({time.time()-t0:.0f}s)",
        flush=True,
    )

    # 面板秩矩阵 (全历史 550d, 按日期列对齐)
    dts = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].unique()
    cutoff = np.sort(pd.to_datetime(pd.Series(dts)).unique())[-550]
    read_cols = ["symbol", "date", "close_hfq", *MOM_COLS, *VOL_COLS_A, "peak_roc_20d"]
    panel = pd.read_parquet(
        str(PANEL_V3_PATH), columns=read_cols, filters=[("date", ">=", cutoff)]
    )
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    dt = pd.to_datetime(panel["date"]).dt.normalize()
    cal = np.sort(dt.unique())
    symbols = np.sort(panel["symbol"].unique())
    rank_mats: dict[str, np.ndarray] = {}
    for col in [*MOM_COLS, *VOL_COLS_A, "peak_roc_20d"]:
        df = pd.DataFrame({"s": panel["symbol"], "d": dt, "v": panel[col]})
        df["rk"] = df.groupby("d")["v"].rank(pct=True)
        rank_mats[col] = (
            df.pivot(index="s", columns="d", values="rk").sort_index().to_numpy("float32")
        )
        del df
    px = (
        panel.assign(d=dt)
        .pivot_table(index="symbol", columns="d", values="close_hfq", aggfunc="last")
        .sort_index()
        .reindex(columns=pd.DatetimeIndex(cal))
        .ffill(axis=1)
    )
    del panel
    gc.collect()

    def nanmean_mats(cols):
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.stack([rank_mats[c] for c in cols]), axis=0).astype("float32")

    F1 = nanmean_mats(MOM_COLS)
    F2 = nanmean_mats(HOTVOL_COLS)
    F3 = np.nanmean(np.stack([F1, F2]), axis=0).astype("float32")
    F4 = nanmean_mats(F4_COLS)
    A2 = nanmean_mats(VOL_COLS_A)
    A2_5 = pd.DataFrame(A2).T.rolling(5, min_periods=3).mean().T.to_numpy("float32")
    A3 = np.nanmean(np.stack([F1, A2]), axis=0).astype("float32")
    A4 = np.nanmean(np.stack([F1, A2_5]), axis=0).astype("float32")
    del A2, A2_5
    gc.collect()

    d_idx = {str(pd.Timestamp(d).date()): j for j, d in enumerate(cal)}
    s_idx = {s: i for i, s in enumerate(symbols)}

    def decile_mask(W):
        m = np.zeros_like(W, dtype=bool)
        for j in range(W.shape[1]):
            w = W[:, j]
            ok = np.isfinite(w)
            if ok.sum() < 100:
                continue
            m[ok, j] = w[ok] >= np.quantile(w[ok], 0.90)
        return m

    f3m = decile_mask(F3)
    union_m = decile_mask(F1) | decile_mask(F2) | f3m | decile_mask(F4)
    print(f"[mats] done ({time.time()-t0:.0f}s)", flush=True)

    METHODS = (
        "prod", "a1", "a3", "a4", "veto_f3", "veto_union",
        "s3_f3_a1", "s3_f3_a3", "s3_f3_a4", "s3_union_a3", "s3_union_a4",
    )
    acc: dict[str, dict[str, list]] = {m: {"main": [], "dual": []} for m in METHODS}
    win_pos: dict[str, list] = {"main": [], "dual": []}
    pool_cov: dict[str, list] = {"main": [], "dual": []}

    for (d, board), g in src.groupby(["date", "board"]):
        j = d_idx.get(d)
        if j is None:
            continue
        g = g.dropna(subset=[args.rank]).sort_values(args.rank, ascending=False)
        n = len(g)
        if n < TOP_N + 5:
            continue
        n_pool = max(TOP_N, int(round(POOL_PCT * n)))
        pool = g.head(n_pool)

        def vals(W):
            return np.array(
                [W[s_idx[s], j] if s in s_idx else np.nan for s in pool["symbol"]],
                dtype=float,
            )

        def flags(M):
            return np.array(
                [bool(M[s_idx[s], j]) if s in s_idx else False for s in pool["symbol"]]
            )

        a1v, a3v, a4v = vals(F1), vals(A3), vals(A4)
        f3f, unf = flags(f3m), flags(union_m)
        rank_order = np.arange(len(pool))

        sels: dict[str, pd.DataFrame] = {"prod": pool.head(TOP_N)}
        # stage 消融: 只③ (a*) / 只② (veto_*)
        for nm, v in (("a1", a1v), ("a3", a3v), ("a4", a4v)):
            order = np.lexsort((rank_order, -np.nan_to_num(v, nan=-np.inf)))
            sels[nm] = pool.iloc[order[:TOP_N]]
        keep3, keepu = ~f3f, ~unf
        sels["veto_f3"] = pool[keep3].head(TOP_N)
        sels["veto_union"] = pool[keepu].head(TOP_N)
        # 完整三段式: ①池 → ②剔下跌 → ③上涨信号共选
        for nm, fl, v in (
            ("s3_f3_a1", f3f, a1v),
            ("s3_f3_a3", f3f, a3v),
            ("s3_f3_a4", f3f, a4v),
            ("s3_union_a3", unf, a3v),
            ("s3_union_a4", unf, a4v),
        ):
            keep = ~fl
            order = np.lexsort(
                (rank_order[keep], -np.nan_to_num(v[keep], nan=-np.inf))
            )
            sels[nm] = pool[keep].iloc[order[:TOP_N]]

        def net_px(sel, sell_off):
            k = j + sell_off
            if j + 1 >= len(cal) or k >= len(cal):
                return np.full(len(sel), np.nan)
            syms = sel["symbol"].astype(str)
            pb = px.iloc[:, j + 1].reindex(syms).to_numpy(float)
            ps = px.iloc[:, k].reindex(syms).to_numpy(float)
            out = ps / pb - 1.0 - COST
            out[~(pb > 0)] = np.nan
            return out

        for nm, sel in sels.items():
            n3 = sel["net_3d"].to_numpy(float)
            acc[nm][board].append(
                (
                    float(np.nanmean(n3)) if len(n3) else np.nan,
                    int((n3 >= WIN_THR).sum()),
                    float(np.nanmean(net_px(sel, 6))),
                    float(np.nanmean(net_px(sel, 11))),
                )
            )

        n3_all = g["net_3d"].to_numpy(float)
        wpos = np.nonzero(n3_all >= WIN_THR)[0]
        if len(wpos):
            win_pos[board].extend((wpos / n).tolist())
            pool_cov[board].append(float((wpos < n_pool).mean()))

    rows = []
    for m in METHODS:
        for board in ("main", "dual"):
            arr = acc[m][board]
            if not arr:
                continue
            n3 = np.array([x[0] for x in arr], dtype=float)
            w = np.array([x[1] for x in arr], dtype=float)
            n5 = np.array([x[2] for x in arr], dtype=float)
            n10 = np.array([x[3] for x in arr], dtype=float)
            b3 = np.array([x[0] for x in acc["prod"][board]], dtype=float)
            b5 = np.array([x[2] for x in acc["prod"][board]], dtype=float)
            b10 = np.array([x[3] for x in acc["prod"][board]], dtype=float)
            half = len(n3) // 2
            row = {
                "method": m,
                "board": board,
                "days": int(len(n3)),
                "winners3_per_day": round(float(w.mean()), 3),
            }
            for tag, v, b in (("d3", n3, b3), ("d5", n5, b5), ("d10", n10, b10)):
                dv = v - b
                row[f"net{tag}_per_day"] = round(float(np.nanmean(v)), 5)
                row[f"{tag}_vs_prod"] = round(float(np.nanmean(dv)), 5)
                row[f"{tag}_h1"] = round(float(np.nanmean(dv[:half])), 5)
                row[f"{tag}_h2"] = round(float(np.nanmean(dv[half:])), 5)
            rows.append(row)
    res = pd.DataFrame(rows)

    for board in ("main", "dual"):
        wp = np.array(win_pos[board])
        if len(wp) == 0:
            continue
        print(
            f"[winner-dist {board}] n={len(wp)} "
            f"<1%:{(wp < 0.01).mean():.1%} 1-2%:{((wp >= 0.01) & (wp < 0.02)).mean():.1%} "
            f"2-5%:{((wp >= 0.02) & (wp < 0.05)).mean():.1%} >5%:{(wp >= 0.05).mean():.1%} | "
            f"top5%池覆盖 {np.mean(pool_cov[board]):.1%}",
            flush=True,
        )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    res.to_parquet(out_dir / f"pool2stage_check{args.tag}_{ts}.parquet", index=False)
    (out_dir / f"pool2stage_check{args.tag}_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "src": args.src,
                "rank_col": args.rank,
                "pool_pct": POOL_PCT,
                "win_thr": WIN_THR,
                "note": "三段式: ①排名top5%池 ②剔下跌信号(veto) ③涨前指标共选(s3_*); "
                "D3/D5/D10 三视界 (买 T+1 收盘, 卖 T+4/T+6/T+11 收盘, 扣 20bp); "
                "d*_vs_prod>0=优于生产TOP10; 半窗需同号才算稳; winners3=net_3d>=5% 席位数/日",
                "summary": res.to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[saved] pool2stage_check_{ts} ({time.time()-t0:.0f}s)", flush=True)
    print(res.to_string(index=False), flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
