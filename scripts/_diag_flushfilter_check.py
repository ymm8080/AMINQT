"""_diag_flushfilter_check.py — 放量下跌避雷否决权 × 生产TOP10 代价核算 (2026-09-03).

动机 (用户 09-03): 「F1F2F3F4 都别碰, TOP10 filter out?」— ⑥证 F3 top10% 5日内
放量下跌率 4.5x 基率. 本核算: 若把 F1-F4 top10% 并集当否决权, 从生产式 TOP10
(rank_blend base 臂 = time_decay_ab 无权重臂) 里踢掉踩雷票并顺位补位, 历史代价/收益如何.

数据: time_decay_ab_20260903_035003.parquet (462,937 行, base 臂 rank_blend + net_3d)
    + 现算 F1-F4 秩矩阵 (面板 550d, 与 preflush_detector 同口径).
指标: 踩雷率 (TOP10 里处于避雷区比例), 被踢 vs 留下 net3 差, 踢除+补位后 TOP10
    net3 变化 (Δ/日), 半窗拆分, 双板. 按单一臂否决与并集否决分别核算.

WORM: DATA OTHERS/diag/flushfilter_check_<ts>.parquet + .json
用法: python scripts/_diag_flushfilter_check.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

COST = 0.0020
TOP_Q = 0.10
TOP_N = 10
SRC_PQ = "time_decay_ab_20260903_035003.parquet"

MOM_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
HOTVOL_COLS = ("amount", "turnover_rate", "volume_ratio", "ma_vol_ratio_5_20")
VOL_COLS_A = (*HOTVOL_COLS, "free_float_turnover_rate")  # ④ A2 口径 5 列


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()

    src = pd.read_parquet(str(data_others_path("diag") / SRC_PQ))
    src["date"] = src["date"].astype(str)
    print(f"[src] {len(src):,} rows, {src['date'].nunique()} days ({time.time()-t0:.0f}s)", flush=True)

    # 面板秩矩阵 (全历史, 再按 src 日期对齐)
    dts = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].unique()
    cutoff = np.sort(pd.to_datetime(pd.Series(dts)).unique())[-550]
    read_cols = ["symbol", "date", *MOM_COLS, *VOL_COLS_A, "peak_roc_20d"]
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
    del panel
    gc.collect()

    def nanmean_mats(cols):
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.stack([rank_mats[c] for c in cols]), axis=0).astype("float32")

    F1 = nanmean_mats(MOM_COLS)
    F2 = nanmean_mats(HOTVOL_COLS)
    F3 = np.nanmean(np.stack([F1, F2]), axis=0).astype("float32")
    F4 = nanmean_mats([*MOM_COLS, "peak_roc_20d"])
    arms = {"F1": F1, "F2": F2, "F3": F3, "F4": F4}

    # ---- 涨前预热臂 (④口径): A1 动量 / A2 量能预热(5日滚均) / A3 / A4 ----
    A1 = F1
    A2 = nanmean_mats(VOL_COLS_A)
    A2_5 = (
        pd.DataFrame(A2).T.rolling(5, min_periods=3).mean().T.to_numpy("float32")
    )
    A3 = np.nanmean(np.stack([A1, A2]), axis=0).astype("float32")
    A4 = np.nanmean(np.stack([A1, A2_5]), axis=0).astype("float32")
    keep_arms = {"A1": A1, "A3": A3, "A4": A4}
    gc.collect()

    d_idx = {str(pd.Timestamp(d).date()): j for j, d in enumerate(cal)}
    s_idx = {s: i for i, s in enumerate(symbols)}
    top_masks = {}
    for name, W in {**arms, **keep_arms}.items():
        m = np.zeros_like(W, dtype=bool)
        for j in range(W.shape[1]):
            w = W[:, j]
            ok = np.isfinite(w)
            if ok.sum() < 100:
                continue
            m[ok, j] = w[ok] >= np.quantile(w[ok], 1 - TOP_Q)
        top_masks[name] = m
        del W
    gc.collect()
    print(f"[arms] masks done ({time.time()-t0:.0f}s)", flush=True)

    def flagged_matrix(names):
        out = np.zeros_like(next(iter(top_masks.values())))
        for n in names:
            out |= top_masks[n]
        return out

    veto_sets = {
        **{f"{k}_solo": (k, "kick") for k in arms},
        "union": ("F1", "F2", "F3", "F4", "kick"),
        "F3_solo": ("F3", "kick"),
        # 涨前预热做第二步: 只保留预热分 top10% 的席位 (kick=剔雷, keep=只留预热)
        **{f"{k}_keep": (k, "keep") for k in keep_arms},
    }

    rows = []
    for veto_name, memb in veto_sets.items():
        *names, mode = memb if isinstance(memb[-1], str) and memb[-1] in ("kick", "keep") else (*memb, "kick")
        FM = flagged_matrix(names)
        per_day = []
        for (d, board), g in src.groupby(["date", "board"]):
            j = d_idx.get(d)
            if j is None:
                continue
            g = g.dropna(subset=["rank_blend_base"]).sort_values("rank_blend_base", ascending=False)
            if len(g) < TOP_N + 5:
                continue
            top = g.head(TOP_N).copy()
            fl = np.array(
                [
                    bool(FM[s_idx.get(s, -1), j]) if s in s_idx else False
                    for s in top["symbol"]
                ]
            )
            # 补位: kick 模式=剔 flagged 后前 TOP_N; keep 模式=只取 flagged 前 TOP_N
            fl_all = np.array(
                [
                    bool(FM[s_idx.get(s, -1), j]) if s in s_idx else False
                    for s in g["symbol"]
                ]
            )
            repl = g[fl_all if mode == "keep" else ~fl_all].head(TOP_N)
            n3 = top["net_3d"].to_numpy(float)
            n3r = repl["net_3d"].to_numpy(float)
            per_day.append(
                {
                    "date": d,
                    "board": board,
                    "n_flag": int(fl.sum()),
                    "net3_kept": float(np.nanmean(n3[~fl])) if (~fl).any() else np.nan,
                    "net3_kicked": float(np.nanmean(n3[fl])) if fl.any() else np.nan,
                    "net3_before": float(np.nanmean(n3)),
                    "net3_after": float(np.nanmean(n3r)) if len(n3r) else np.nan,
                }
            )
        pdf = pd.DataFrame(per_day)
        half = len(pdf) // 2
        h1, h2 = pdf.iloc[:half], pdf.iloc[half:]
        d_all = (pdf["net3_after"] - pdf["net3_before"]).mean()
        rows.append(
            {
                "veto": veto_name,
                "days": int(len(pdf)),
                "flag_rate": round(float(pdf["n_flag"].mean() / TOP_N), 4),
                "n3_kicked": round(float(pdf["net3_kicked"].mean()), 5),
                "n3_kept": round(float(pdf["net3_kept"].mean()), 5),
                "d_net3_per_day": round(float(d_all), 5),
                "d_net3_h1": round(float((h1["net3_after"] - h1["net3_before"]).mean()), 5),
                "d_net3_h2": round(float((h2["net3_after"] - h2["net3_before"]).mean()), 5),
            }
        )
        print(f"[{veto_name}] {rows[-1]}", flush=True)

    res = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    res.to_parquet(out_dir / f"flushfilter_check_{ts}.parquet", index=False)
    (out_dir / f"flushfilter_check_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "src": SRC_PQ,
                "top_q": TOP_Q,
                "note": "d_net3>0=踢雷+补位改善TOP10; 负=过滤有代价",
                "summary": res.to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[saved] flushfilter_check_{ts} ({time.time()-t0:.0f}s)", flush=True)
    print(res.to_string(index=False), flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
