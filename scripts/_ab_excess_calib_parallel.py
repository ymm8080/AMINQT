"""超额标签 A/B (parallel, 2026-08-29): 校准目标 label_pm_10d_net vs 板内当日超额.

复刻生产 build_merged_shortlist 选股链 (pool_score max(sniper,fusion) →
calibrate_mag10d walk-forward → 板内 mag 降序 top10), 双臂唯一差异 = 校准目标列:
- baseline: label_pm_10d_net (绝对净 c2c, 生产现行)
- excess:   label_pm_10d_net − 板内×当日全池等权均值 (市场中性)

超额基准取 板×日 (与交付 pred_excess 列同口径): 消除板块结构性差异与日级牛熊,
让校准只学个股特异性. 无前瞻: 去均值只用与被去均值行同日已实现的标签行,
calibrate_mag10d 内部已按已实现边界过滤.

判定 (预登记): OOS 评估窗前后两半段, excess 臂的日均超额收益差均 > 0,
且全窗绝对净收益差不恶化 → 过闸建议采纳.
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.pipeline_parallel.backtest import load_panel
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import BACKTEST_RESULT_DIR

TOP_N = 10
TARGETS = {"baseline": "label_pm_10d_net", "excess": "label_excess_10d_net"}


def top10_daily(picks: pd.DataFrame) -> pd.DataFrame:
    """每日每板 top10 → (net, exc) 日均值, 只保留标签已实现的行."""
    p = picks.dropna(subset=["label_pm_10d_net"])
    return p.groupby(["board", "date"]).agg(
        net=("label_pm_10d_net", "mean"),
        exc=("label_excess_10d_net", "mean"),
        n=("symbol", "size"),
    )


def arm_stats(daily: pd.DataFrame) -> dict:
    return {
        "days": int(len(daily)),
        "net_mean": float(daily["net"].mean()),
        "exc_mean": float(daily["exc"].mean()),
        "exc_win": float((daily["exc"] > 0).mean()),
        "net_win": float((daily["net"] > 0).mean()),
    }


def main() -> int:
    t0 = time.time()
    work = load_panel()
    print(f"[ab] 面板加载完成 {len(work):,} 行 ({time.time() - t0:.0f}s)", flush=True)

    # 超额标签: 板×日 全池已实现等权均值去均值 (与交付 pred_excess 同口径)
    mkt = work.groupby(["board", "date"])["label_pm_10d_net"].transform("mean")
    work["label_excess_10d_net"] = work["label_pm_10d_net"] - mkt
    del mkt
    gc.collect()

    sub = work[
        ["symbol", "date", "board", "label_pm_10d_net", "label_excess_10d_net"]
    ].copy()
    score_s = pool_score(work, SNIPER.pool)
    score_f = pool_score(work, FUSION.pool)
    sub["score"] = np.maximum(score_s.values, score_f.values)
    del work, score_s, score_f
    gc.collect()
    sub = sub.dropna(subset=["score"]).reset_index(drop=True)
    print(f"[ab] 打分池 {len(sub):,} 行, {sub['symbol'].nunique():,} 只", flush=True)

    picks = {}
    for arm, tgt in TARGETS.items():
        mag = calibrate_mag10d(
            sub[["symbol", "date", "board", "score", tgt]],
            score_col="score",
            target_col=tgt,
        )
        if mag.empty:
            print(f"[ab] FAIL {arm} 臂校准为空", flush=True)
            return 2
        res = sub.merge(mag, on=["symbol", "date", "board"], how="inner")
        res = res.sort_values(["board", "date", "mag"], ascending=[True, True, False])
        res["rk"] = res.groupby(["board", "date"]).cumcount() + 1
        picks[arm] = res[res["rk"] <= TOP_N].reset_index(drop=True)
        print(
            f"[ab] {arm} 臂: {picks[arm]['date'].nunique()} 决策日 "
            f"× {len(picks[arm]):,} 票 ({time.time() - t0:.0f}s)",
            flush=True,
        )
        del mag, res
        gc.collect()

    report: dict = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "top_n": TOP_N,
        "boards": {},
    }
    for board in ("main", "dual"):
        pk = {a: p[p["board"] == board] for a, p in picks.items()}
        dl = {a: top10_daily(p) for a, p in pk.items()}
        # 同日头对头 (两臂都有出票的日子)
        common = dl["baseline"].index.intersection(dl["excess"].index)
        b, e = dl["baseline"].loc[common], dl["excess"].loc[common]
        half = len(common) // 2
        h1, h2 = slice(0, half), slice(half, None)
        d_full = float((e["exc"] - b["exc"]).mean())
        d_h1 = float((e["exc"] - b["exc"]).iloc[h1].mean())
        d_h2 = float((e["exc"] - b["exc"]).iloc[h2].mean())
        d_net = float((e["net"] - b["net"]).mean())
        # 换手重合度
        ov = []
        for _d in pk["baseline"]["date"].unique():
            sb = set(pk["baseline"].query("date == @_d")["symbol"])
            se = set(pk["excess"].query("date == @_d")["symbol"])
            if se:
                ov.append(len(sb & se) / max(len(sb), len(se)))
        bd = report["boards"][board] = {
            "common_days": int(len(common)),
            "baseline": arm_stats(dl["baseline"]),
            "excess": arm_stats(dl["excess"]),
            "delta_exc_full": d_full,
            "delta_exc_h1": d_h1,
            "delta_exc_h2": d_h2,
            "delta_net_full": d_net,
            "pick_overlap": float(np.mean(ov)) if ov else float("nan"),
        }
        bd["verdict_pass"] = bool(d_h1 > 0 and d_h2 > 0 and d_net >= -0.002)
        print(
            f"\n===== {board} (同日 {len(common)} 天) =====\n"
            f"baseline: 净 {bd['baseline']['net_mean']:+.2%}/日  超额 {bd['baseline']['exc_mean']:+.2%}  "
            f"胜率 {bd['baseline']['exc_win']:.0%}\n"
            f"excess  : 净 {bd['excess']['net_mean']:+.2%}/日  超额 {bd['excess']['exc_mean']:+.2%}  "
            f"胜率 {bd['excess']['exc_win']:.0%}\n"
            f"超额差: 全窗 {d_full:+.2%}pp/日 | 前半 {d_h1:+.2%} | 后半 {d_h2:+.2%} | "
            f"净差 {d_net:+.2%}pp\n换手重合 {bd['pick_overlap']:.0%} | "
            f"过闸 {'YES' if bd['verdict_pass'] else 'NO'}",
            flush=True,
        )

    out_dir = Path(BACKTEST_RESULT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"_ab_excess_calib_{datetime.now():%Y%m%d_%H%M%S}.json"
    fp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ab] 结果 WORM 落盘 {fp} ({time.time() - t0:.0f}s)", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
