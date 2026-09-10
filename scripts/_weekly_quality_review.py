"""周度质量复盘 (2026-09-08 首版落地, 同日被误删未 commit; 2026-09-09 按记忆
规格重建 — 口径以本文件为准, 标 [重建] 的参数为首版丢失后的重建值).

三线 (legacy / parallel / density) 近 DZ_WINDOW=10 个可结算出票日实绩 vs
全市场同口径基线 (常量与 _deadzone_guard 同源 import, 不另抄):
  出票日 t → T+1 收盘买 → T+4 收盘卖, net4 = C[t+4]/C[t+1] − 1 − DZ_COST;
  α_日 = 当日该线出票 net4 均值 − 当日全市场 net4 均值 (市场有效股票数
  ≥ DZ_MKT_MIN_SYMBOLS 才算, 否则当日无 α); 赢 = net4 ≥ DZ_WIN。

旗标 (大声报告, 恒 exit 0 — 由 run_weekly_selfevolve 读入决定是否进化):
  wr_flag    线窗内赢率 < DZ_ENTER (25%) 且样本 ≥ DZ_MIN_SAMPLES
  trend_flag 窗内后半均值 α − 前半 ≤ TREND_DROP_PP 且后半天数 ≥ 2 [重建]
  wow_flag   线 mean_alpha 较上一份 review 恶化 > WOW_DROP_PP [重建]

WORM: DATA_OTHERS_DIR/weekly_review_{ts}.json (只增) +
weekly_review_latest.json (keep-latest, 供自进化链读取)。
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DATA_OTHERS_DIR, PANEL_V3_PATH, STOCK_LIST_DIR  # noqa: E402
from scripts._deadzone_guard import (  # noqa: E402
    DZ_COST,
    DZ_ENTER,
    DZ_MIN_SAMPLES,
    DZ_MKT_MIN_SYMBOLS,
    DZ_SETTLE,
    DZ_WIN,
    DZ_WINDOW,
)

_LINE_PATTERNS = {
    "legacy": r"^legacy_stocklist_(\d{8})__.*\.csv$",
    "parallel": r"^parallel_shortlist_(\d{8})__.*\.csv$",
    "density": r"^prob10dens_(\d{8})__.*\.csv$",
}
# [重建] 首版阈值丢失, 以下为重建值: 趋势恶化 1pp + 后半转负; 周环比恶化 1pp
TREND_DROP_PP = -1.0
WOW_DROP_PP = 1.0


def load_line_picks(list_dir=None) -> dict[str, pd.DataFrame]:
    root = Path(list_dir if list_dir is not None else STOCK_LIST_DIR)
    out: dict[str, pd.DataFrame] = {}
    for line, pat in _LINE_PATTERNS.items():
        best: dict[str, str] = {}
        for f in sorted(os.listdir(root)):
            m = re.match(pat, f)
            if m:
                best[m.group(1)] = f
        frames = []
        for d, f in sorted(best.items()):
            try:
                df = pd.read_csv(root / f, dtype={"symbol": str})
            except Exception:
                continue
            if not len(df) or "symbol" not in df.columns:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "list_date": pd.to_datetime(d),
                        "symbol": df["symbol"]
                        .astype(str)
                        .str.split(".")
                        .str[0]
                        .str.zfill(6),
                    }
                )
            )
        if frames:
            out[line] = pd.concat(frames, ignore_index=True).drop_duplicates(
                ["list_date", "symbol"]
            )
    return out


def review(list_dir=None, panel_path=None) -> dict:
    picks_by_line = load_line_picks(list_dir)
    all_picks = (
        pd.concat(picks_by_line.values(), ignore_index=True)
        if picks_by_line
        else pd.DataFrame(columns=["list_date", "symbol"])
    )
    d0 = pd.Timestamp(all_picks["list_date"].min()) - pd.Timedelta(days=10)
    df = pd.read_parquet(
        panel_path or PANEL_V3_PATH,
        columns=["symbol", "date", "close"],
        filters=[("date", ">=", d0)],
    )
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df = df.drop_duplicates(["symbol", "date"])
    c = df.pivot(index="date", columns="symbol", values="close").sort_index()
    dates = c.index
    pos = {s: i for i, s in enumerate(c.columns)}
    net4 = c.shift(-DZ_SETTLE) / c.shift(-1) - 1 - DZ_COST
    mkt_n = c.notna().sum(axis=1)
    mkt_alpha_base = net4.mean(axis=1).where(mkt_n >= DZ_MKT_MIN_SYMBOLS)

    lines_rep = {}
    for line, L in picks_by_line.items():
        per_day: dict = {}
        for t, g in L.groupby("list_date"):
            if t not in dates:
                continue
            ti = dates.get_loc(t)
            if ti + DZ_SETTLE >= len(dates):  # 未成熟
                continue
            vals = [
                float(net4.iloc[ti, pos[s]])
                for s in g["symbol"]
                if s in pos and pd.notna(net4.iloc[ti, pos[s]])
            ]
            if not vals:
                continue
            base = mkt_alpha_base.iloc[ti]
            per_day[t] = {
                "n": len(vals),
                "net4_mean": float(np.mean(vals)),
                "win_rate": float(np.mean([v >= DZ_WIN for v in vals])),
                "alpha": float(np.mean(vals) - base) if pd.notna(base) else None,
            }
        if not per_day:
            lines_rep[line] = {"flag": [], "reason": "无可结算出票日"}
            continue
        days = sorted(per_day)[-DZ_WINDOW:]
        win = [per_day[d] for d in days]
        alphas = [d_["alpha"] for d_ in win if d_["alpha"] is not None]
        n_tot = sum(d_["n"] for d_ in win)
        wr = float(np.mean([d_["win_rate"] for d_ in win]))
        mean_alpha = float(np.mean(alphas)) if alphas else None
        flags, reasons = [], []
        if n_tot >= DZ_MIN_SAMPLES and wr < DZ_ENTER:
            flags.append("wr_flag")
            reasons.append(f"窗内赢率 {wr:.1%} < {DZ_ENTER:.0%} (n={n_tot})")
        if len(alphas) >= 6:
            h = len(alphas) // 2
            h1, h2 = float(np.mean(alphas[:h])), float(np.mean(alphas[h:]))
            if (h2 - h1) * 100 <= TREND_DROP_PP and h2 < 0:
                flags.append("trend_flag")
                reasons.append(f"α 前半 {h1:+.2f}pp → 后半 {h2:+.2f}pp 恶化且转负")
        lines_rep[line] = {
            "days": len(days),
            "picks": n_tot,
            "win_rate": wr,
            "mean_alpha": mean_alpha,
            "alpha_by_day": {d.strftime("%Y-%m-%d"): per_day[d]["alpha"] for d in days},
            "flag": flags,
            "reason": reasons,
        }

    prev = _read_latest()
    if prev:
        for line, rep in lines_rep.items():
            p = prev.get("lines", {}).get(line, {})
            pa = p.get("mean_alpha")
            if (
                pa is not None
                and rep.get("mean_alpha") is not None
                and (pa - rep["mean_alpha"]) * 100 > WOW_DROP_PP
            ):
                rep["flag"].append("wow_flag")
                rep["reason"].append(
                    f"mean_alpha 周环比 {pa:+.2f}pp → {rep['mean_alpha']:+.2f}pp 恶化"
                )

    return {
        "ts": time.strftime("%Y%m%d_%H%M%S"),
        "any_flag": any(r.get("flag") for r in lines_rep.values()),
        "lines": lines_rep,
    }


def _latest_path() -> Path:
    return DATA_OTHERS_DIR / "weekly_review_latest.json"


def _read_latest() -> dict | None:
    p = _latest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    rep = review()
    doc = DATA_OTHERS_DIR / f"weekly_review_{rep['ts']}.json"
    with open(doc, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    with open(_latest_path(), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(f"[weeklyreview] WORM {doc.name} + keep-latest", flush=True)
    for line, r in rep["lines"].items():
        ma = r.get("mean_alpha")
        ma_s = f"{ma * 100:+.2f}pp" if ma is not None else "NA"
        print(
            f"  {line}: {r.get('days', 0)}结算日 {r.get('picks', 0)}票 "
            f"赢率{r.get('win_rate', 0):.1%} α{ma_s} 旗标={r.get('flag') or '无'}",
            flush=True,
        )
        for rs in r.get("reason", []):
            print(f"    [flag] {rs}", flush=True)
    verdict = "有旗标 → 触发本周进化" if rep["any_flag"] else "零旗标 → 本周不进化"
    print(f"[weeklyreview] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
