"""超额标签 A/B Tier-2 (legacy, 2026-08-29): 按用户裁决以 TOP10 实测质量重判.

背景: Tier-1 (IC 口径) 判 REJECT (main +14% / dual -15%, 双板不一致). 用户定夺:
所有决定按 TOP10 质量判断 — IC 只是代理. 本脚本在同一 OOS 测试段 (split_window
test = 末 60 交易日, 与 IC 同窗) 上, 用两套模型包对同一特征帧逐日预测,
板内按 pred(10d_reg) 降序取 top10, 用**绝对**已实现标签对比.

- 特征帧: 与训练完全同配方 (load_panel_v3 3y + run_train 清洗 + prepare_board_frame,
  registry 同路径, float_shares_map=None — 两臂训练时同口径, 已核).
- BruteForce 后注入: 复刻 select_features 的 missing-col 生成 (两臂特征列并集).
- 标签: 本脚本不做 demean (预测不依赖标签), 评估用绝对 label_pm_{3,10}d_net.
- 两臂模型: baseline = 20260828 生产批次, excess = abexcess0829 影子批次.

判定 (预登记, TOP10 口径): 同日头对头, excess 臂 top10 日均净收益 (10d) 前后两半
均 > 0 且全窗 ≥ 0 → 翻案采纳候选 (还需用户批); 否则维持 REJECT.
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

PKLS = {
    "baseline": "models/pipeline1/{board}_20260828.pkl",
    "excess": "models/pipeline1/{board}_abexcess0829.pkl",
}
TOP_N = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ab_top10")


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
    return (
        p.groupby("date")
        .agg(net10=("label_pm_10d_net", "mean"),
             net3=("label_pm_3d_net", "mean"),
             n=("symbol", "size"))
    )


def main() -> int:
    t0 = time.time()
    panel = load_panel_v3(path=PANEL_V3_PATH)
    panel = panel[panel["date"] >= panel["date"].max() - pd.DateOffset(years=3)]
    log.info("[tier2] 面板 %d 行", len(panel))
    cleaner = CleaningPipeline()
    board_dfs = dict(zip(("main", "dual"), cleaner.run_train(panel, board=None)))
    del panel
    gc.collect()
    registry = FeatureRegistry(
        path=os.path.join(str(data_others_path("data/factor_registry")),
                          "feature_registry.json")
    )
    features = FeatureEngineV35()
    trainer = DualTrackTrainer()
    report = {"ts": datetime.now().isoformat(timespec="seconds"), "boards": {}}

    for board in ("main", "dual"):
        board_df = board_dfs.pop(board)
        if not len(board_df):
            log.warning("[%s] 清洗后空, 跳过", board)
            continue
        df = prepare_board_frame(
            board_df, features, None,
            cross_sectional_rank=(board != "main"), registry=registry,
        )
        del board_df
        gc.collect()
        log.info("[%s] 特征帧 %d 行 (%.0fs)", board, len(df), time.time() - t0)

        bundles = {
            arm: DualTrackTrainer.load(p.format(board=board))
            for arm, p in PKLS.items()
        }
        need = set()
        for b in bundles.values():
            need |= set(b["feature_cols"])
        missing = need - set(df.columns)
        if missing:
            df = brute_inject(df, missing)
            log.info("[%s] BruteForce 后注入 %d 列", board, len(missing))
        still = {c for b in bundles.values() for c in b["feature_cols"]} - set(
            df.columns
        )
        if still:
            log.error("[%s] FAIL 缺列 %d: %s", board, len(still), list(still)[:5])
            return 2

        test = trainer.split_window(df)["test"]
        log.info("[%s] 测试段 %d 日", board, test["date"].nunique())
        arms = {}
        for arm, b in bundles.items():
            model, _label = b["models"]["10d_reg"]
            t = test[["symbol", "date", "label_pm_3d_net",
                      "label_pm_10d_net"]].copy()
            t["pred"] = model.predict(
                np.nan_to_num(test[b["feature_cols"]].values, nan=0.0)
            )
            arms[arm] = t
        del df, test, bundles
        gc.collect()

        dl = {a: daily_top10(t) for a, t in arms.items()}
        common = dl["baseline"].index.intersection(dl["excess"].index)
        b, e = dl["baseline"].loc[common], dl["excess"].loc[common]
        half = len(common) // 2
        d10 = (e["net10"] - b["net10"]).dropna()
        d3 = (e["net3"] - b["net3"]).dropna()
        bd = report["boards"][board] = {
            "days": int(len(common)),
            "baseline_net10": float(b["net10"].mean()),
            "excess_net10": float(e["net10"].mean()),
            "baseline_net3": float(b["net3"].mean()),
            "excess_net3": float(e["net3"].mean()),
            "win10_base": float((b["net10"] > 0).mean()),
            "win10_exc": float((e["net10"] > 0).mean()),
            "delta10_full": float(d10.mean()),
            "delta10_h1": float(d10.iloc[:half].mean()) if half else float("nan"),
            "delta10_h2": float(d10.iloc[half:].mean()) if half else float("nan"),
            "delta3_full": float(d3.mean()),
        }
        print(
            f"\n===== {board} TOP10 (同日 {len(common)} 天, 10d 净) =====\n"
            f"baseline: {bd['baseline_net10']:+.2%}/日 胜率 {bd['win10_base']:.0%} | "
            f"3d {bd['baseline_net3']:+.2%}\n"
            f"excess  : {bd['excess_net10']:+.2%}/日 胜率 {bd['win10_exc']:.0%} | "
            f"3d {bd['excess_net3']:+.2%}\n"
            f"10d 差: 全窗 {bd['delta10_full']:+.2%}pp | 前半 {bd['delta10_h1']:+.2%} | "
            f"后半 {bd['delta10_h2']:+.2%} | 3d 差 {bd['delta3_full']:+.2%}pp\n"
            f"TOP10 口径判定: "
            f"{'翻案候选' if (bd['delta10_h1'] > 0 and bd['delta10_h2'] > 0 and bd['delta10_full'] >= 0) else '维持 REJECT'}",
            flush=True,
        )

    out = Path("data/others") / f"_ab_excess_top10_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[tier2] 结果落盘 {out} ({time.time()-t0:.0f}s)", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
