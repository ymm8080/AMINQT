"""合并清单单文件 (2026-09-08 用户: "i need combined sheet daily. modify pipeline").

09-07 手工版 stocklist_combined_20260907__v3.xlsx (tmp_t/_merge_fourmodules_0907.py)
的每日产线化。五页 xlsx:

- 多模块重叠: 被 ≥2 模块同时选中的票, 各模块自有 10d 口径
  (LEGACY=pred_ret_10d/prob_up_10d, PARALLEL=pred_mag_10d/pred_prob_10d)
- LEGACY / PARALLEL / 密度: 当日清单 CSV 全列原样 (dtype=str, 百分比显示层保留)
- SLOW_BULL: shadow 目录长持清单 (只入表不推送)

缺源跳页 (密度/SLOW_BULL 常缺, 影子单当日未跑); LEGACY+PARALLEL 双缺才退出。
输出: STOCK_LIST_DIR/stocklist_combined_{date}.xlsx — WORM: 已存在打印跳过 rc=0。
链路: run_daily_automation "stocklist_combined" 步骤, 置 final_stocklist 后 (非关键)。
"""

import argparse
import datetime
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DATA_OTHERS_DIR, STOCK_LIST_DIR  # noqa: E402

SHADOW_DIR = DATA_OTHERS_DIR / "shadow"

# (页名, glob, 目录) — 顺序 = 页序; SLOW_BULL 文件名用连字符日期
_SOURCES = [
    ("LEGACY", "legacy_stocklist_{date}__*.csv", "list"),
    ("PARALLEL", "parallel_shortlist_{date}__*.csv", "list"),
    ("密度", "prob10dens_{date}__*.csv", "list"),
    ("SLOW_BULL", "slowbull_list_{dashdate}__*.csv", "shadow"),
]

OVERLAP_COLS = [
    "symbol",
    "模块数",
    "模块",
    "LEGACY_预测10d",
    "LEGACY_概率10d",
    "PARALLEL_预测10d",
    "PARALLEL_概率10d",
]


def _newest(pattern: str, d: Path) -> Path | None:
    hits = glob.glob(str(d / pattern))
    if not hits:
        return None
    return Path(max(hits, key=os.path.getmtime))


def build(date: str, list_dir=STOCK_LIST_DIR, shadow_dir=SHADOW_DIR):
    """四线源 CSV → [(页名, DataFrame)]; 多模块重叠页置首 (纯读盘, 可单测)."""
    list_dir, shadow_dir = Path(list_dir), Path(shadow_dir)
    dashdate = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    sheets: list[tuple[str, pd.DataFrame]] = []
    for name, pat, which in _SOURCES:
        pat = pat.format(date=date, dashdate=dashdate)
        d = list_dir if which == "list" else shadow_dir
        fp = _newest(pat, d)
        if fp is None:
            print(f"[combined] {name}: 无当日清单 ({d.name}/{pat}), 跳页")
            continue
        df = pd.read_csv(fp, dtype=str).fillna("")
        if df.empty or "symbol" not in df.columns:
            print(f"[combined] {name}: {fp.name} 空表/无 symbol 列, 跳页")
            continue
        sheets.append((name, df))

    core = [n for n, _ in sheets if n in ("LEGACY", "PARALLEL")]
    if not core:
        raise SystemExit(f"[combined] {date} LEGACY/PARALLEL 清单均无, 不产出")

    # 多模块重叠页: 全部已到模块计数 (含 密度/SLOW_BULL); 无重叠 → 空表仍出页
    overlap: dict[str, dict] = {}
    for name, df in sheets:
        for _, r in df.iterrows():
            sym = str(r["symbol"]).zfill(6)
            info = overlap.setdefault(sym, {"modules": [], "pred": {}})
            info["modules"].append(name)
            if name == "LEGACY":
                info["pred"]["LEGACY_预测10d"] = r.get("pred_ret_10d", "")
                info["pred"]["LEGACY_概率10d"] = r.get("prob_up_10d", "")
            elif name == "PARALLEL":
                info["pred"]["PARALLEL_预测10d"] = r.get("pred_mag_10d", "")
                info["pred"]["PARALLEL_概率10d"] = r.get("pred_prob_10d", "")
    multi = sorted(
        (
            {
                "symbol": sym,
                "模块数": len(info["modules"]),
                "模块": "+".join(info["modules"]),
                **{k: info["pred"].get(k, "") for k in OVERLAP_COLS[3:]},
            }
            for sym, info in overlap.items()
            if len(info["modules"]) >= 2
        ),
        key=lambda x: (-x["模块数"], x["symbol"]),
    )
    return [("多模块重叠", pd.DataFrame(multi, columns=OVERLAP_COLS))] + sheets


def write(sheets, date: str, list_dir=STOCK_LIST_DIR) -> Path:
    fp = Path(list_dir) / f"stocklist_combined_{date}.xlsx"
    with pd.ExcelWriter(fp, engine="openpyxl") as xw:
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = max(len(str(col)), *(len(str(v)) for v in df[col]))
                ws.column_dimensions[
                    ws.cell(row=1, column=i).column_letter
                ].width = min(width + 4, 40)
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "date", nargs="?", default=None, help="YYYYMMDD (缺省取最新 legacy 清单日)"
    )
    args = ap.parse_args()
    date = args.date
    if date is None:
        fp = _newest("legacy_stocklist_????????__*.csv", Path(STOCK_LIST_DIR))
        if fp is None:
            raise SystemExit("STOCK LIST 目录无任何 legacy 清单")
        date = re.search(r"legacy_stocklist_(\d{8})__", fp.name).group(1)
    out = Path(STOCK_LIST_DIR) / f"stocklist_combined_{date}.xlsx"
    if out.exists():
        print(f"[combined] WORM: 已存在 {out.name}, 跳过")
        return 0
    sheets = build(date)
    write(sheets, date)
    for name, df in sheets:
        print(f"[combined] sheet {name}: {len(df)} 行")
    print(f"[combined] {datetime.datetime.now():%H:%M} → {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
