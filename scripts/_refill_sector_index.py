"""回填申万行业指数列到 V3 面板 (sector_index 数据源).

背景: 2026-08-02 审计把 sw_ret_1d/sw_index_close/sw_index_vol 当失源孤儿列删了,
但用户要求面板显示正确的申万行业数据 → 重新用生产逻辑 (东财 industry 模糊匹配
申万一级指数名) merge 回填.

数据流:
  fetch_sector_index(20230103, 20260807) 缓存 sw_20230103_20260807.parquet
  → 按 panel.industry ↔ 申万 index_name 匹配 → sw_ret_1d / sw_index_close / sw_index_vol
  → WORM 备份后写回 V3_PATH.

注: 旧申万接口 801020 采掘 2021-12 停更, 27/28 行业有 2023-2026 数据,
    未匹配行 (UNKNOWN / 无法模糊匹配) 保持 NaN, 覆盖率如实上报.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging

import pandas as pd

from app.pipeline1.data_supply import DataSupplyChain
from scripts.data_fetch_pipeline import load_v3, save_v3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refill_sector_index")

START, END = "20230103", "20260807"
TARGETS = {"sw_ret_1d", "sw_index_close", "sw_index_vol"}


def _build_sw_data() -> pd.DataFrame:
    sup = DataSupplyChain()
    sw = sup.fetch_sector_index(start_date=START, end_date=END)
    if sw is None or sw.empty:
        raise SystemExit("申万行业指数拉取为空, 中止回填")
    logger.info("SW 指数: %d 行业, %d 行", sw["index_code"].nunique(), len(sw))
    return sw


def main() -> None:
    panel = load_v3()
    for c in TARGETS:
        if c in panel.columns:
            logger.info("面板已有 %s, 先删除再回填", c)
            panel = panel.drop(columns=[c])

    sw = _build_sw_data()
    name_to_code = dict(zip(sw["index_name"], sw["index_code"], strict=False))

    # 东财 industry → 申万一级指数名 (与 panel_builder.enrich_alt_data 同款逻辑)
    ind_map: dict[str, str] = {}
    for ind_name in panel["industry"].dropna().unique():
        if ind_name in name_to_code:
            ind_map[ind_name] = ind_name
        else:
            for sw_name in name_to_code:
                if ind_name in sw_name or sw_name in ind_name:
                    ind_map[ind_name] = sw_name
                    break
    logger.info("行业映射: %d/%d", len(ind_map), panel["industry"].nunique())

    panel["_sw_name"] = panel["industry"].map(ind_map)
    sw_data = sw.rename(
        columns={
            "ret_pct": "sw_ret_1d",
            "close": "sw_index_close",
            "volume": "sw_index_vol",
        }
    )
    panel = panel.merge(
        sw_data[["index_name", "date"] + sorted(TARGETS)],
        left_on=["_sw_name", "date"],
        right_on=["index_name", "date"],
        how="left",
    )
    panel = panel.loc[:, ~panel.columns.duplicated()]
    panel = panel.drop(columns=["_sw_name", "index_name"], errors="ignore")

    for c in TARGETS:
        non_na = panel[c].notna().sum()
        logger.info("  %s: 非空 %d (%.1f%%)", c, non_na, non_na / len(panel) * 100)

    save_v3(panel)


if __name__ == "__main__":
    main()
