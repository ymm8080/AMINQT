"""合并清单单文件 (2026-09-08 用户: "i need combined sheet daily. modify pipeline").

09-07 手工版 stocklist_combined_20260907__v3.xlsx (tmp_t/_merge_fourmodules_0907.py)
的每日产线化。六页 xlsx:

- 多模块重叠: 被 ≥2 模块同时选中的票, 各模块自有 10d 口径
  (LEGACY=pred_ret_10d/prob_up_10d, PARALLEL=pred_mag_10d/pred_prob_10d)
- 市场FADE预测 (2026-09-09 用户指令): 明日上证冲高回落概率
  (_fade_market_forecast.forecast, 失败跳页 fail-open 不拦产出)
- LEGACY / PARALLEL / 密度: 当日清单 CSV 全列原样 (dtype=str, 百分比显示层保留)
- SLOW_BULL: shadow 目录长持清单 (只入表不推送)

每张带 symbol 的表**最前列**insert 一列 BIGDROP SCAN (2026-09-16 用户令: 大跌
扫描结果记在既有表里, 不另开页; 见 insert_bigdrop_column)。

缺源跳页 (密度/SLOW_BULL 常缺, 影子单当日未跑); LEGACY+PARALLEL 双缺才退出。
输出: STOCK_LIST_DIR/stocklist_combined_{date}.xlsx — WORM: 同名已存在则退到
__v2/__v3/... 变体 (绝不覆盖, 也不再跳过)。下游 `run_daily_automation._combined_delivered`
本就按 `stocklist_combined_{tag}*.xlsx` glob 认变体, 故同日重跑必须真的落文件。
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


def _next_path(date: str, list_dir=STOCK_LIST_DIR) -> Path:
    """WORM 取号: 首选规范名; 已存在则依次 __v2/__v3/... 找第一个空位 (绝不覆盖)。

    为什么要取号而不是跳过 (2026-09-15 用户 "SHOULD NOT SKIP"): 同日重跑若静默
    skip, 拿到的是**旧内容**且 rc=0 —— 上游(如改了闸/校正了清单)重跑后下游合成表
    仍是旧行, 无声失真。变体号在本仓已是既有约定 (09-07 手工版 __v3,
    tests/test_daily_automation.py 的 __v2 用例, _combined_delivered 的 glob)。
    """
    d = Path(list_dir)
    if not (d / f"stocklist_combined_{date}.xlsx").exists():
        return d / f"stocklist_combined_{date}.xlsx"
    n = 2
    while (d / f"stocklist_combined_{date}__v{n}.xlsx").exists():
        n += 1
    return d / f"stocklist_combined_{date}__v{n}.xlsx"


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


def insert_bigdrop_column(sheets) -> int:
    """BIGDROP SCAN 标注列 (2026-09-16 用户令 2: **不要独立页**, 记进既有表)。

    当日合并清单全 workbook 的 symbol 并集送 bigdrop 次日大跌模块 → 每张带 symbol
    的表 insert(0) 一列【BIGDROP SCAN】。三态语义与 _genious_excel 同名列逐字一致:
      大跌风险 N.Nx = 模型支 (唯一带看跌方向), 波动风险 N.Nx = 仅规则支 (零方向),
      空            = 两边没举手。

    送扫范围 = **所有**表的 symbol 并集 (含 SLOW_BULL)。只送一部分的话, 没送到的票
    在列里显示成空 —— 与"两边没举手"长得一模一样, 而那是两个完全不同的意思。

    列序: insert(0) 放最前 (用户 0915 令 —— 旁路列追加到末尾会被列宽/横向滚动吞掉)。

    模块运行 (建包新鲜度闸) 复用 _genious_excel._bigdrop_module_run (同源跳过 /
    非同源重建 / 失败回退旧包); 扫描拿到 None (缺包/失败) 就整列不加, 不拦其余产出
    (旁路标注, 同 genious 的契约)。返回实际加了列的表数。
    """
    try:
        from scripts._genious_excel import (
            _bigdrop_module_run,
            _bigdrop_scan,
            _norm_sym,
        )

        targets = [df for _, df in sheets if "symbol" in df.columns]
        if not targets:
            return 0
        syms = sorted({_norm_sym(s) for df in targets for s in df["symbol"]})
        _bigdrop_module_run()
        scan = _bigdrop_scan(syms)
        if scan is None:
            return 0
        for df in targets:
            df.insert(
                0,
                "BIGDROP SCAN",
                df["symbol"].map(lambda s: scan.get(_norm_sym(s), "")),
            )
        log.info(
            "[combined] BIGDROP SCAN 列: %d 张表, %d 只送扫",
            len(targets),
            len(syms),
        )
        return len(targets)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — 旁路标注, 失败不加列不拦产出
        log.error("[combined] BIGDROP SCAN 列失败, 不加列: %s", exc)
        return 0


def market_fade_sheet(fc_fn=None) -> tuple[str, pd.DataFrame] | None:
    """市场FADE预测页 (2026-09-09 用户: "市场冲高回落概率预测值写进COMBINED")."""
    if fc_fn is None:
        try:
            from _fade_market_forecast import forecast as fc_fn
        except Exception as e:
            print(f"[combined] 市场FADE预测不可用, 跳页: {e}")
            return None
    try:
        fc = fc_fn()
    except Exception as e:
        print(f"[combined] 市场FADE预测失败, 跳页: {e}")
        return None
    rows = [
        ("预测交易日", fc["next_date"]),
        (
            "状态基准 (上证收盘)",
            f"{fc['state_date']}  {fc['close']:.2f} ({fc['ret'] * 100:+.2f}%)",
        ),
        ("行情带 (距MA20)", f"{fc['regime']} ({fc['above_ma20'] * 100:+.2f}%)"),
        ("近5日涨幅 r5", f"{fc['r5'] * 100:+.2f}%"),
        ("量比 (vs 20日均量)", f"{fc['vratio']:.2f}"),
        ("P(冲高)", f"{fc['p_surge'] * 100:.0f}%"),
        ("P(回落|冲高)", f"{fc['p_fade_given_surge'] * 100:.0f}%"),
        ("P(回落日) 预测", f"{fc['p_fade_day'] * 100:.0f}%"),
        ("P(回落日) 无条件基准", f"{fc['base_fade'] * 100:.1f}%"),
        ("判读", f"{fc['verdict']}; 回落日次日不偏空 (+0.05% vs +0.03%)"),
        (
            "口径",
            f"上证2005-今条件频率 状态=前收盘 n={fc['n_regime']}日; 回落日=g≥0.4%且吐回≥60%",
        ),
    ]
    return ("市场FADE预测", pd.DataFrame(rows, columns=["指标", "值"]))


def write(sheets, date: str, list_dir=STOCK_LIST_DIR) -> Path:
    fp = _next_path(date, list_dir)
    with pd.ExcelWriter(fp, engine="openpyxl") as xw:
        for name, df in sheets:
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col]])
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
    sheets = build(date)
    mk = market_fade_sheet()
    if mk:
        sheets.insert(1, mk)
    # BIGDROP SCAN 列 (2026-09-16 用户令): combined 就绪后 bigdrop 必跑, 全 workbook
    # 的票标大跌/波动风险 —— 记进既有表的列, 不另开页。
    insert_bigdrop_column(sheets)
    out = write(sheets, date)
    for name, df in sheets:
        print(f"[combined] sheet {name}: {len(df)} 行")
    print(f"[combined] {datetime.datetime.now():%H:%M} → {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
