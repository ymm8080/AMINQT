"""_diag_moneyflow_hold.py — 主力持仓比例代理 诊断 (Tushare moneyflow).

背景: 用户问"同花顺主力持仓比例取得到嘛". 结论 (2026-08-05):
  同花顺真值 = 资金流派生专有口径, 无公开 API, 历史日频序列取不到.
  改用 Tushare moneyflow (官方接口, 单日全市场截面一次调用, 日频全历史, 积分够):
    主力净流入(日) = (buy_lg + buy_elg) - (sell_lg + sell_elg)   # 千元
    持仓比例代理     = 累计主力净流入 / 流通市值 (circ_mv)          # 对齐同花顺"持仓比例"量纲
  东财 capital_feed 只够每日子集快照 (逐股拉会被封 IP, 无历史截面接口), 不做历史特征.

评估 (用户 2026-08-04 方法论):
  - per-stock time-series IC (T+2/3/5) — 个股时序, 不得用横截面
  - 单端 TOP-N (特征降序取 TOP-10) 绝对上涨幅度 + 胜率 (5/10) vs 无条件基准
HS300 × 最近 N 交易日 (快速验证), 通过再全量回填.

WORM → data/_diag_moneyflow_hold_<ts>.json + .log

用法: python scripts/_diag_moneyflow_hold.py [--days 250] [--refresh]
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from app.pipeline1.label_engine import LabelEngine
from config.settings import PANEL_V3_PATH
from scripts._diag_chip_weekly import per_stock_ts_ic
from scripts._diag_column_feed import LABELS, MASK_RECENT_DAYS, weighted_ic
from scripts._measure_topn import HORIZONS, MIN_WINRATE, measure_topn
from scripts._reclassify_all_features import add_label_pm_10d_net

logging.disable(logging.CRITICAL)

HS300_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "universe_csi300.txt"
)
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "supply_cache",
    "alt_data",
    "moneyflow",
)
THROTTLE = 0.35  # moneyflow 免费 token 限流保护

MIN_OBS = 20
WINDOW_DAYS = 250  # 默认最近 ~1 年


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_hs300() -> list[str]:
    with open(HS300_FILE, encoding="utf-8") as f:
        syms = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            syms.append(line.split(".")[0].zfill(6))
    return sorted(set(syms))


def load_panel_window(days: int, hs300: set[str] | None) -> pd.DataFrame:
    """面板最近 days 交易日; hs300=None → 全市场. 返回 (df, trade_dates)."""
    read_cols = [
        "date",
        "symbol",
        "is_suspended",
        "close_hfq",
        "circ_mv",
        "volume",
        "amount",
    ]
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols, engine="pyarrow")
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    if hs300 is not None:
        df = df[df["symbol"].isin(hs300)].reset_index(drop=True)
    trade_dates = sorted(df["date"].unique())
    cutoff = trade_dates[-days]
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    trade_dates = [d for d in trade_dates if d >= cutoff]
    return df, trade_dates


def backfill_moneyflow(trade_dates: list, refresh: bool = False) -> pd.DataFrame:
    """单日全市场截面逐日回填 (Tushare moneyflow), 缓存全市场截面 (可复用)."""
    start = trade_dates[0].strftime("%Y%m%d")
    end = trade_dates[-1].strftime("%Y%m%d")
    cache_path = os.path.join(CACHE_DIR, f"moneyflow_{start}_{end}.parquet")
    if not refresh and os.path.exists(cache_path):
        print(f"[moneyflow] 命中缓存: {cache_path}")
        return pd.read_parquet(cache_path)

    from dotenv import load_dotenv

    load_dotenv()
    import tushare as ts

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未配置 (.env)")
    pro = ts.pro_api(token)

    frames = []
    for i, d in enumerate(trade_dates):
        dstr = d.strftime("%Y%m%d")
        try:
            raw = pro.moneyflow(trade_date=dstr)
            if raw is not None and len(raw):
                frames.append(raw)
        except Exception as exc:
            print(f"[moneyflow] {dstr} 失败: {exc}")
        if i < len(trade_dates) - 1:
            time.sleep(THROTTLE)
        if (i + 1) % 50 == 0:
            print(f"[moneyflow] 进度 {i + 1}/{len(trade_dates)}")
    if not frames:
        raise RuntimeError("moneyflow 回填: 全部失败")
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"[moneyflow] 回填完成: {len(out)} 行 → {cache_path}")
    return out


def build_features(work: pd.DataFrame, mf: pd.DataFrame) -> pd.DataFrame:
    """主力持仓比例代理: 累计主力净流入(千元) / 流通市值(万元).

    单位对齐到元: 千元×1000, 万元×10000 → ratio = cumsum(千元) / (circ_mv万×10).
    0.1 常数不影响排序/IC, 但绝对量纲需标注.
    """
    mf = mf.copy()
    mf["symbol"] = mf["ts_code"].str.split(".").str[0].str.zfill(6)
    mf["date"] = pd.to_datetime(mf["trade_date"].astype(str), format="%Y%m%d")
    mf["main_net_kqy"] = (
        mf["buy_lg_amount"]
        + mf["buy_elg_amount"]
        - mf["sell_lg_amount"]
        - mf["sell_elg_amount"]
    )
    mf = (
        mf[["symbol", "date", "main_net_kqy"]]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    work = work.sort_values(["symbol", "date"]).reset_index(drop=True)
    work = work.merge(mf, on=["symbol", "date"], how="left")
    g = work.groupby("symbol")["main_net_kqy"]
    gc = work.groupby("symbol")["close_hfq"]
    # 全窗累计 (千元) / (流通市值万元×10) → 无量纲比例 (对齐同花顺持仓比例量纲)
    work["main_hold_ratio"] = g.cumsum() / (work["circ_mv"] * 10.0)
    # 60 日滚动累计 (规避远古资金流锚定漂移)
    work["main_hold_ratio_60d"] = g.transform(
        lambda x: x.rolling(60, min_periods=20).sum()
    ) / (work["circ_mv"] * 10.0)
    # ── 周/月频组合变体 (用户方法论: 同频率特征×均线组合, 5=周 20=月) ──
    work["bias5"] = (
        work["close_hfq"] / gc.transform(lambda x: x.rolling(5, min_periods=5).mean())
        - 1.0
    )
    work["bias20"] = (
        work["close_hfq"] / gc.transform(lambda x: x.rolling(20, min_periods=20).mean())
        - 1.0
    )
    work["hold_chg_5d"] = work["main_hold_ratio_60d"] - work.groupby("symbol")[
        "main_hold_ratio_60d"
    ].shift(5)  # 周频: 主力持仓代理 5 日变化 (加仓速度)
    work["hold_chg_20d"] = work["main_hold_ratio_60d"] - work.groupby("symbol")[
        "main_hold_ratio_60d"
    ].shift(20)  # 月频: 20 日变化
    work["hold_chg_5d_x_sgnbias5"] = work["hold_chg_5d"] * np.sign(work["bias5"])
    work["hold_chg_20d_x_sgnbias20"] = work["hold_chg_20d"] * np.sign(work["bias20"])
    work["hold_60d_x_sgnbias20"] = work["main_hold_ratio_60d"] * np.sign(work["bias20"])
    return work


def _baseline(work: pd.DataFrame) -> dict:
    base = {}
    for k in HORIZONS:
        lab = f"label_pm_{k}d_net"
        v = work[lab].dropna() if lab in work.columns else pd.Series(dtype=float)
        base[k] = {
            "mag": float(v.mean()) if len(v) else None,
            "winrate": float((v > 0).mean()) if len(v) else None,
            "n": int(len(v)),
        }
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--universe", choices=["hs300", "all"], default="hs300")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ts = _ts()
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        f"_diag_moneyflow_hold_{ts}.log",
    )
    json_path = log_path.replace(".log", ".json")
    out = []
    print(f"[start] universe={args.universe} × 最近 {args.days} 交易日, ts={ts}")

    hs300 = set(load_hs300()) if args.universe == "hs300" else None
    print(f"[universe] {'HS300 ' + str(len(hs300)) + ' 只' if hs300 else '全市场'}")
    out.append(
        f"[universe] {'HS300 ' + str(len(hs300)) + ' 只' if hs300 else '全市场'}"
    )

    work, trade_dates = load_panel_window(args.days, hs300)
    print(
        f"[panel] {len(work)} 行 / {work['symbol'].nunique()} 只 / "
        f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}"
    )
    out.append(
        f"[panel] {len(work)} 行 / {work['symbol'].nunique()} 只 / "
        f"{trade_dates[0].date()} ~ {trade_dates[-1].date()}"
    )

    mf = backfill_moneyflow(trade_dates, refresh=args.refresh)
    work = build_features(work, mf)
    del mf
    gc.collect()

    # 标签 (生产口径): build_labels → mask_suspension → mask_recent_days → 补 10d
    work = LabelEngine.build_labels(work, session="PM")
    work = LabelEngine.mask_suspension(work)
    work = LabelEngine.mask_recent_days(work, days=MASK_RECENT_DAYS)
    work = add_label_pm_10d_net(work)
    print(f"[labels] 生产行集 {len(work)} 行")
    out.append(f"[labels] 生产行集 {len(work)} 行")

    feats = {
        "main_hold_ratio": "主力持仓代理(全窗累计/流通市值)",
        "main_hold_ratio_60d": "主力持仓代理(60日滚动/流通市值)",
        # 周/月频组合变体 (同频率特征×均线, 用户方法论)
        "hold_chg_5d": "主力持仓周变化(60d代理5日差)",
        "hold_chg_20d": "主力持仓月变化(60d代理20日差)",
        "hold_chg_5d_x_sgnbias5": "持仓周变化×sgn(bias5)",
        "hold_chg_20d_x_sgnbias20": "持仓月变化×sgn(bias20)",
        "hold_60d_x_sgnbias20": "持仓60d×sgn(bias20)",
    }
    res = {
        "ts": ts,
        "universe": args.universe,
        "days": args.days,
        "n_rows": int(len(work)),
        "n_symbols": int(work["symbol"].nunique()),
    }

    # 1) per-stock TSIC
    out.append("")
    out.append("=== per-stock TSIC (T+2/3/5) ===")
    out.append(f"{'feature':<28}{'wTSIC':>9}{'TS2d':>9}{'TS3d':>9}{'TS5d':>9}")
    out.append("-" * 64)
    tsic_res = {}
    for name, label in feats.items():
        per = per_stock_ts_ic(work, {name: work[name]}, LABELS, min_obs=MIN_OBS)
        row = {lab: per[lab][name] for lab in LABELS}
        tsic_res[name] = {
            lab: (round(float(v), 4) if v == v else None) for lab, v in row.items()
        }
        out.append(
            f"{name:<28}{_f(weighted_ic(row)):>9}"
            + "".join(_f(row[lab]) for lab in LABELS)
        )
    res["tsic"] = tsic_res

    # 2) 单端 TOP-N (5/10) vs 无条件基准
    out.append("")
    out.append("=== 单端 TOP-10 (特征降序, 每日截面) vs 无条件基准 ===")
    base = _baseline(work)
    out.append(
        "基准: "
        + " | ".join(
            f"T+{k}d mag={base[k]['mag']:+.4f} win={base[k]['winrate']:.3f} n={base[k]['n']}"
            for k in HORIZONS
        )
    )
    res["baseline"] = base

    for name, label in feats.items():
        m = measure_topn(
            work,
            name,
            top_n=10,
            per="date",
            ascending=False,
            winrate_threshold=MIN_WINRATE,
        )
        out.append(f"\n[{name}] ({label})")
        res[name] = {"label": label}
        for k in HORIZONS:
            r = m.get(k) if isinstance(m, dict) else {}
            ok = r.get("ok") if r else None
            out.append(
                f"  T+{k}d  mag={_f(r.get('mag'))}  win={_fmt_pct(r.get('winrate'))}  "
                f"n={r.get('n') if r else 0}  {'PASS' if ok else ('--' if ok is None else 'FAIL')}"
            )
            res[name][k] = r
        res[name]["passed"] = any((m.get(k) or {}).get("ok") for k in HORIZONS)

    # 3) WORM 落盘
    print("\n".join(out))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[WORM] {log_path}")
    print(f"[WORM] {json_path}")


def _f(v):
    return f"{v:+.4f}" if v is not None and v == v else "    nan"


def _fmt_pct(v):
    return f"{v:.3f}" if v is not None and v == v else "  nan"


if __name__ == "__main__":
    main()
