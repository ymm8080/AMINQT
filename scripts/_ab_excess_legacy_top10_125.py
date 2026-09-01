"""超额标签 A/B Tier-2b (legacy main, 2026-08-29): 125 交易日窗 TOP10 复核.

用户裁决: Tier-2 (60 日 OOS) main 翻案候选成立但单窗口单行情, 加验 125 交易日.
窗构成: 末 60 日 = split_window OOS 测试段 (与 Tier-2 同段, 应复现 +4.36pp),
前 ~65 日 = train/es 段 — 两臂同帧同段头对头仍可比, 但前段含在样本, 判定
以全窗前后半稳定性为准 (与 Tier-2 预登记规则同构)。

判定: delta10 (excess 臂 − baseline 臂, 板内 pred 降序 top10, 10d 净) 全窗 ≥ 0
且前半 > 0 且后半 > 0 → 通过, main 超额标签建入 legacy 管线; 否则留档不动。
"""

import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

BOARD = sys.argv[1] if len(sys.argv) > 1 else "main"
PKLS = {
    "baseline": f"models/pipeline1/{BOARD}_20260828.pkl",
    "excess": f"models/pipeline1/{BOARD}_abexcess0829.pkl",
}
EVAL_DAYS = 125
OOS_DAYS = 60
TOP_N = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ab_top10_125")


def brute_inject(df: pd.DataFrame, need: set) -> pd.DataFrame:
    gen = BruteForceGenerator()
    raw_cols = gen._eligible(df)
    picks = []
    for fam in BRUTE_FAMILIES:
        new = gen.generate_columns(df, fam, need, raw_cols=raw_cols, dtype="float32")
        if new is None or not len(new.columns):
            continue
        picks.append(new)
    if picks:
        block = pd.concat(picks, axis=1)
        for c in block.columns:
            df[c] = block[c].to_numpy()
    return df


def daily_top10(t: pd.DataFrame) -> pd.DataFrame:
    t = t.sort_values(["date", "pred"], ascending=[True, False])
    t["rk"] = t.groupby("date").cumcount() + 1
    p = t[t["rk"] <= TOP_N].dropna(subset=["label_pm_10d_net"])
    return p.groupby("date").agg(
        net10=("label_pm_10d_net", "mean"),
        net3=("label_pm_3d_net", "mean"),
        n=("symbol", "size"),
    )


def main() -> int:
    t0 = time.time()
    panel = load_panel_v3(path=PANEL_V3_PATH)
    panel = panel[panel["date"] >= panel["date"].max() - pd.DateOffset(years=3)]
    log.info("[125d] 面板 %d 行", len(panel))
    cleaner = CleaningPipeline()
    board_dfs = dict(zip(("main", "dual"), cleaner.run_train(panel, board=None)))
    del panel
    gc.collect()
    registry = FeatureRegistry(
        path=os.path.join(
            str(data_others_path("data/factor_registry")), "feature_registry.json"
        )
    )
    board_df = board_dfs[BOARD]
    del board_dfs
    gc.collect()
    df = prepare_board_frame(
        board_df,
        FeatureEngineV35(),
        None,
        cross_sectional_rank=(BOARD != "main"),
        registry=registry,
    )
    del board_df
    gc.collect()
    log.info("[125d] 特征帧 %d 行 (%.0fs)", len(df), time.time() - t0)

    bundles = {a: DualTrackTrainer.load(p) for a, p in PKLS.items()}
    need = set()
    for b in bundles.values():
        need |= set(b["feature_cols"])
    missing = need - set(df.columns)
    if missing:
        df = brute_inject(df, missing)
        log.info("[125d] BruteForce 后注入 %d 列", len(missing))
    still = {c for b in bundles.values() for c in b["feature_cols"]} - set(df.columns)
    if still:
        log.error("[125d] FAIL 缺列 %d: %s", len(still), list(still)[:5])
        return 2

    dates = np.array(sorted(df["date"].unique()))
    test = df[df["date"].isin(dates[-EVAL_DAYS:])].copy()
    oos_cut = dates[-OOS_DAYS]
    log.info(
        "[125d] 评估段 %d 日 (OOS 段自 %s)",
        test["date"].nunique(),
        pd.Timestamp(oos_cut).date(),
    )

    arms = {}
    keep_cols = ["symbol", "date", "label_pm_3d_net", "label_pm_10d_net"]
    if "label_pm_1d_net" in test.columns:  # 市场日收益代理 (regime 分桶用)
        keep_cols.insert(2, "label_pm_1d_net")
    for arm, b in bundles.items():
        model, _label = b["models"]["10d_reg"]
        t = test[keep_cols].copy()
        t["pred"] = model.predict(
            np.nan_to_num(test[b["feature_cols"]].values, nan=0.0)
        )
        arms[arm] = t
    predrows = arms["baseline"].rename(columns={"pred": "pred_base"})
    predrows["pred_exc"] = arms["excess"]["pred"].to_numpy()
    del df, test, bundles
    gc.collect()

    dl = {a: daily_top10(t) for a, t in arms.items()}
    common = dl["baseline"].index.intersection(dl["excess"].index)
    b, e = dl["baseline"].loc[common], dl["excess"].loc[common]
    d10 = (e["net10"] - b["net10"]).dropna()
    d3 = (e["net3"] - b["net3"]).dropna()
    half = len(d10) // 2
    oos_mask = d10.index >= oos_cut
    rep = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "board": BOARD,
        "eval_days": int(len(common)),
        "baseline_net10": float(b["net10"].mean()),
        "excess_net10": float(e["net10"].mean()),
        "win10_base": float((b["net10"] > 0).mean()),
        "win10_exc": float((e["net10"] > 0).mean()),
        "baseline_net3": float(b["net3"].mean()),
        "excess_net3": float(e["net3"].mean()),
        "delta10_full": float(d10.mean()),
        "delta10_h1": float(d10.iloc[:half].mean()),
        "delta10_h2": float(d10.iloc[half:].mean()),
        "delta10_oos60": float(d10[oos_mask].mean()),
        "delta10_pre": float(d10[~oos_mask].mean()),
        "delta3_full": float(d3.mean()),
    }
    rep["verdict_pass"] = bool(
        rep["delta10_full"] >= 0 and rep["delta10_h1"] > 0 and rep["delta10_h2"] > 0
    )
    print(
        f"\n===== {BOARD} TOP10 @125d (同日 {rep['eval_days']} 天, 10d 净) =====\n"
        f"baseline: {rep['baseline_net10']:+.2%}/日 胜率 {rep['win10_base']:.0%} | "
        f"3d {rep['baseline_net3']:+.2%}\n"
        f"excess  : {rep['excess_net10']:+.2%}/日 胜率 {rep['win10_exc']:.0%} | "
        f"3d {rep['excess_net3']:+.2%}\n"
        f"10d 差: 全窗 {rep['delta10_full']:+.2%}pp | 前半 {rep['delta10_h1']:+.2%} | "
        f"后半 {rep['delta10_h2']:+.2%}\n"
        f"分段: OOS60 {rep['delta10_oos60']:+.2%}pp | 前段(train/es) "
        f"{rep['delta10_pre']:+.2%}pp | 3d 差 {rep['delta3_full']:+.2%}pp\n"
        f"125d 复核判定: {'通过' if rep['verdict_pass'] else '不通过'}",
        flush=True,
    )

    out = (
        Path("data/others")
        / f"_ab_excess_top10_125d_{BOARD}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    # 逐行预测 + 逐日净: regime 分桶/名次分析离线重算用 (WORM)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pr = Path("data/others") / f"_ab_excess_125d_predrows_{BOARD}_{stamp}.parquet"
    predrows.to_parquet(pr, index=False)
    daily = pd.DataFrame(
        {"base_net10": b["net10"], "exc_net10": e["net10"], "delta10": d10}
    ).dropna(how="all")
    dc = Path("data/others") / f"_ab_excess_125d_daily_{BOARD}_{stamp}.csv"
    daily.to_csv(dc, index=True, index_label="date")
    print(f"[125d] 明细落盘 {pr.name} / {dc.name}", flush=True)
    print(f"\n[125d] 结果落盘 {out} ({time.time() - t0:.0f}s)", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
