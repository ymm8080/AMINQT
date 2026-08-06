# -*- coding: utf-8 -*-
"""PIPELINE 并行多系统 编排入口 (2026-08-04).

用法:
    python -m app.pipeline_parallel.runner
    python -m app.pipeline_parallel.runner --system sniper   # 只跑指定系统
    python -m app.pipeline_parallel.runner --oos-days 504    # 2y 样本外
    python -m app.pipeline_parallel.runner --skip-backtest   # 只加载面板并 dump 行集元数据

回测默认全量跑 (狙击+融合+慢牛), 同时报告全窗 + 末段 OOS.
输出 WORM 到 <DATA OTHERS>/BACKTESTING RESULT/_parallel_backtest_<ts>.json.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import pandas as pd

from app.pipeline_parallel import signals
from app.pipeline_parallel.backtest import (
    export_stock_lists,
    load_panel,
    run_all,
    write_daily_shortlist,
    write_last_days_csv,
    write_worm,
)
from app.pipeline_parallel.config import SLOW_BULL, SLOW_BULL_REGIME, SLOW_BULL_VERSION
from config.settings import STOCK_LIST_DIR


def write_slowbull_pool(work: pd.DataFrame, board: str, date=None) -> str:
    """慢牛系统每日 Top-20 观察池 → STOCK_LIST_DIR (含买卖信号列).

    date: 显式指定选股日 (默认该板块最新交易日). 空池 → 不落盘 (返回 "").
    文件名 = slowbull_pool_<board>_<YYYYMMDD>__slow_bull_<ver>.csv
    (交易日期 + 模块名 + 版本, 对齐 parallel_shortlist_<date>__<module> 约定).
    """
    if date is None:
        latest = work.loc[work["board"] == board, "date"].max()
    else:
        latest = pd.Timestamp(date)
    pool = signals.daily_slowbull_pool(work, latest, board, SLOW_BULL, SLOW_BULL.top_n)
    if pool.empty:
        return ""
    stamp = str(pd.Timestamp(latest).date()).replace("-", "")
    os.makedirs(str(STOCK_LIST_DIR), exist_ok=True)
    fp = STOCK_LIST_DIR / (
        f"slowbull_pool_{board}_{stamp}__slow_bull_{SLOW_BULL_VERSION}.csv"
    )
    # 市场状态条件退出 (2026-08-06): 上升段 trail8 正常出候选; 下行段 + no_open → 不开仓
    up = bool(pool["slow_bull_regime"].iloc[0]) \
        if "slow_bull_regime" in pool.columns else True
    if not up and SLOW_BULL_REGIME["down_mode"] == "no_open":
        note = pd.DataFrame(
            {"board": [board], "date": [str(pd.Timestamp(latest).date())],
             "slow_bull_regime": [False],
             "note": ["下行阶段不开仓 (slow_bull down_mode=no_open, def="
                      f"{SLOW_BULL_REGIME['def']}), 今日无候选"]})
        note.to_csv(fp, index=False)
        return fp.name
    pool.to_csv(fp, index=False)
    return fp.name


def main() -> int:
    ap = argparse.ArgumentParser(description="PIPELINE 并行多系统回测")
    ap.add_argument(
        "--system",
        default=None,
        help="只跑指定系统 (sniper/fusion/slow_bull), 默认全量",
    )
    ap.add_argument(
        "--skip-backtest", action="store_true", help="只加载面板 dump 元数据, 不跑回测"
    )
    ap.add_argument(
        "--oos-days",
        type=int,
        default=None,
        help="样本外窗口交易日数 (覆盖默认 6m/3m/10d 三窗, 只跑单窗)",
    )
    ap.add_argument(
        "--shortlist-date",
        default=None,
        help="短名单选股日 YYYY-MM-DD (默认每板块最新交易日)",
    )
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("加载行集 (快速路径: 复用 3y 检查点 + MFE 标签)...", flush=True)
    work = load_panel()
    print(
        f"行集 rows={len(work):,} stocks={work['symbol'].nunique():,} "
        f"latest={work['date'].max():%Y-%m-%d}",
        flush=True,
    )

    if args.skip_backtest:
        return 0

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = run_all(work, ts, oos_days=args.oos_days)
    p, log, run_dir = write_worm(out, ts)
    # 默认 6m/3m/10d 三窗; 导出清单以最长 OOS 窗为全 OOS 名单 (兼容 --oos-days 单窗).
    longest = max(out["window"]["oos"].values(), key=lambda ow: ow["trading_days"])
    files = export_stock_lists(work, longest["start"], run_dir)
    for b, bd in out["boards"].items():
        files.append(write_last_days_csv(bd["last_days"], run_dir, board=b))
    for b in out["boards"]:
        fn = write_daily_shortlist(
            work, out, run_dir, board=b, date=args.shortlist_date
        )
        if fn:
            files.append(fn)
        fn = write_slowbull_pool(work, board=b, date=args.shortlist_date)
        if fn:
            files.append(fn)
    del work
    gc.collect()

    if args.system:
        for bd in out["boards"].values():
            keep = {args.system: bd["systems"].pop(args.system)}
            bd["systems"] = keep

    print(f"\n落盘目录: {run_dir}")
    print(f"  {p.name}\n  {log.name}")
    for fn in files:
        print(f"  {fn}")
    w = out["window"]
    for lab, ow in w["oos"].items():
        print(f"OOS[{lab}]: {ow['start']} → {ow['end']} ({ow['trading_days']} 交易日)")
    for b, bd in out["boards"].items():
        print(
            f"[板块 {b}] {bd['label']} | 行 {bd['rows']:,} "
            f"股票 {bd['stocks']:,} | 阈值: 胜率>={bd['criteria']['min_winrate']} "
            f"幅度>{bd['criteria']['min_mag']}"
        )
        for name, s in bd["systems"].items():
            if not s.get("enabled"):
                print(f"  [{name}] 未启用 (占位)")
                continue
            for lab, oos in s["oos"].items():
                pr = oos["primary"]
                print(
                    f"  [{name}|OOS {lab}] TOP-{s['top_n']['primary']}: "
                    f"通过 {pr['passed'] or '无'} | 保留={oos['kept']}"
                )
        lt = bd["last_days"].get("last_testable") or {}
        if lt:
            print(
                "  各视界可测日期 (末 15 交易日, 同一选股日): "
                + " ".join(
                    f"{h}=至{lt[h]['last_date']}({lt[h]['n']}日)"
                    for h in ("2d", "3d", "5d", "10d")
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
