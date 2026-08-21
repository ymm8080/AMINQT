"""诊断: 用本地面板 (PANEL_V3_PATH, 含当日) 直接生成 legacy 清单, 绕开生产 _assemble_panel 的网络依赖.

用法: python scripts/_gen_legacy_list.py [YYYYMMDD]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.data_supply import DataSupplyChain
from config.settings import PANEL_V3_PATH

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}


def _timed(name, fn):
    """按阶段计时打印 (只影响本诊断脚本, 不改生产代码)."""

    def wrapper(*args, **kwargs):
        ts = time.time()
        out = fn(*args, **kwargs)
        detail = ""
        df = args[0] if args else kwargs.get("df")
        if isinstance(df, pd.DataFrame) and len(df) and "board" in df:
            detail = f" board={df['board'].iloc[0]}"
        print(f"[{name}{detail}] {time.time() - ts:.0f}s", flush=True)
        return out

    return wrapper


def _instrument(pipe):
    """清洗/特征(按板)/预测/闸+落盘 各段计时 — 为板并行构建决策提供实测拆分."""
    pipe.cleaner.run_inference = _timed("clean", pipe.cleaner.run_inference)
    pipe.features.build = _timed("feat", pipe.features.build)
    pipe.predictor.predict = _timed("predict", pipe.predictor.predict)
    pipe.lister.emit = _timed("emit", pipe.lister.emit)


def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else "20260804"
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(
        f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    # 内存护栏: 只保留末尾 300 交易日 (清洗 min_list_days=180 + 特征最大窗口 252
    # 全在 300 内) → 最新日期的滚动特征与全面板逐值一致, 面板体积缩 ~3×
    dates = sorted(panel["date"].unique())
    cut = dates[-300]
    panel = panel[panel["date"] >= cut]
    print(
        f"[slice] {cut.date()}..{dates[-1].date()} -> {len(panel):,}r "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=BUNDLES)
    _instrument(pipe)
    res = pipe.run(trade_date, panel=panel)
    lst = res.get("list")
    print(
        f"[done] mode={res.get('mode')} empty={res.get('empty')} "
        f"n={0 if lst is None else len(lst)} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if lst is not None and len(lst):
        path = f"data/lists/list_{trade_date}.parquet"
        lst.to_parquet(path, index=False)
        print(f"[saved] {path} ({len(lst)} 只)", flush=True)
        print(lst.to_string(index=False), flush=True)
    else:
        print("[warn] 空清单/降级清单", flush=True)


if __name__ == "__main__":
    main()
