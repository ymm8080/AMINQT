# -*- coding: utf-8 -*-
"""从 Tushare 拉取北向资金, 尽最大可能填充 V3 面板.

V3 = data/panel_full_enriched_v3.parquet (2.7M rows, 105 cols, 3244 stocks).
北向资金 6 列:
  north_net_buy_sh / north_net_buy_sz  — moneyflow_hsgt hgt / sgt (全量净流向)
  north_buy_amt_sh/sz, north_sell_amt_sh/sz — hsgt_top10 前10大 buy/sell 汇总
    (仅 2024-08-16 前有数据, 交易所 2024-08-19 起停止公布明细)

数据源:
  1. moneyflow_hsgt: 市场级日频净流向 (hgt=沪股通, sgt=深股通), 全量覆盖
  2. hsgt_top10: 每日前10大成交股 buy/sell (2024-08-19 前有值), 按 date+market 汇总

北向资金是市场级日频数据, 按 date 广播到所有个股.

内存优化: 只读 symbol+date+6 北向列 (8 cols 而非 105), 避免全量加载 OOM.
输出: 用 pyarrow 合并回完整面板后保存.

用法:
  python fill_northbound_v3.py
"""
import os
import sys
import logging
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PANEL_PATH = "data/panel_full_enriched_v3.parquet"
NORTHBOUND_COLS = [
    "north_net_buy_sh",
    "north_net_buy_sz",
    "north_buy_amt_sh",
    "north_buy_amt_sz",
    "north_sell_amt_sh",
    "north_sell_amt_sz",
]

# hsgt_top10 的 buy/sell 在 2024-08-19 后停止公布 (交易所政策变更)
HSGT_DETAIL_CUTOFF = "20240819"


def _fetch_hsgt_top10_all(pro, start_date, end_date):
    """逐日拉取 hsgt_top10, 按 date+market 汇总为市场级日频 buy/sell/amount.

    hsgt_top10 每日返回 20 行 (沪市 top10 + 深市 top10),
    汇总后得到 (date, market_type) → buy/sell/amount 的市场级日频数据.

    2024-08-19 后 buy/sell/net_amount 为 None, 只保留 amount.

    Returns:
        DataFrame [date, north_buy_amt_sh, north_sell_amt_sh,
                   north_buy_amt_sz, north_sell_amt_sz]
    """
    # 生成交易日列表 (跳过周末, 非交易日 Tushare 返回空)
    all_dates = pd.date_range(start_date, end_date, freq="B").strftime("%Y%m%d").tolist()

    frames = []
    n_ok = 0
    n_empty = 0
    n_err = 0

    for i, dt in enumerate(all_dates):
        if dt >= HSGT_DETAIL_CUTOFF:
            break  # 2024-08-19 后 buy/sell 为 None, 无需拉取
        try:
            df = pro.hsgt_top10(trade_date=dt)
            if df is None or len(df) == 0:
                n_empty += 1
                continue

            # 按 market_type 汇总
            # market_type 1 = 沪市 (沪股通 → sh)
            # market_type 3 = 深市 (深股通 → sz)
            for mt, prefix in [("1", "sh"), ("3", "sz")]:
                sub = df[df["market_type"].astype(str) == mt]
                if len(sub) == 0:
                    continue
                buy = pd.to_numeric(sub.get("buy"), errors="coerce").sum()
                sell = pd.to_numeric(sub.get("sell"), errors="coerce").sum()
                # buy/sell 全为 NaN (2024-08-19 后) → 跳过
                if pd.isna(buy) and pd.isna(sell):
                    continue
                frames.append({
                    "date": pd.to_datetime(dt, format="%Y%m%d"),
                    f"north_buy_amt_{prefix}": buy if not pd.isna(buy) else None,
                    f"north_sell_amt_{prefix}": sell if not pd.isna(sell) else None,
                })
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                logger.warning("hsgt_top10 %s 失败: %s", dt, e)

        # 进度日志 + 限速 (Tushare: 5000积分 500次/分钟)
        if (i + 1) % 50 == 0:
            logger.info(
                "hsgt_top10 进度: %d/%d (%.0f%%), ok=%d, empty=%d, err=%d",
                i + 1, len(all_dates), (i + 1) / len(all_dates) * 100,
                n_ok, n_empty, n_err,
            )
        time.sleep(0.15)  # ~400 calls/min, 安全在 500/min 限制内

    logger.info(
        "hsgt_top10 完成: %d 交易日有数据, %d 空, %d 错误 (共 %d 请求)",
        n_ok, n_empty, n_err, len(all_dates),
    )

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    # 同一 date 可能有多行 (sh + sz), 按 date 聚合
    result = result.groupby("date", as_index=False).first()
    return result.sort_values("date").reset_index(drop=True)


def main():
    # ── 1. 加载 V3 面板 (仅读需要的列, 避免 OOM) ──
    if not os.path.exists(PANEL_PATH):
        logger.error("面板文件不存在: %s", PANEL_PATH)
        sys.exit(1)

    read_cols = ["symbol", "date"] + NORTHBOUND_COLS
    panel = pd.read_parquet(PANEL_PATH, columns=read_cols)
    logger.info(
        "加载 V3 面板 (8 cols): %d rows, %d stocks, %s ~ %s",
        len(panel),
        panel["symbol"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )

    # 检查当前北向列填充率
    logger.info("\n=== 填充前北向资金列状态 ===")
    for col in NORTHBOUND_COLS:
        if col in panel.columns:
            nn = panel[col].notna().sum()
            logger.info("  %-22s  %8d non-null  (%5.2f%%)", col, nn, nn / len(panel) * 100)
        else:
            logger.info("  %-22s  不在面板中", col)

    # ── 2. 确定日期范围 ──
    start_date = panel["date"].min().strftime("%Y%m%d")
    end_date = panel["date"].max().strftime("%Y%m%d")
    logger.info("\n拉取日期范围: %s ~ %s", start_date, end_date)

    # ── 3. 初始化 Tushare ──
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    pro = supply._tushare_pro()
    if pro is None:
        logger.error("TUSHARE_TOKEN 未配置, 无法拉取北向资金")
        sys.exit(1)
    logger.info("Tushare pro_api 初始化成功")

    # ══════════════════════════════════════════════════════════
    # 数据源 1: moneyflow_hsgt — 市场级净流向 (全量)
    # ══════════════════════════════════════════════════════════
    logger.info("\n=== 数据源 1: moneyflow_hsgt (净流向) ===")

    # moneyflow_hsgt 每次最多返回 300 条, 需分段拉取
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    segments = []
    cur = start_dt
    while cur <= end_dt:
        seg_end = min(cur + relativedelta(months=6) - relativedelta(days=1), end_dt)
        segments.append((cur.strftime("%Y%m%d"), seg_end.strftime("%Y%m%d")))
        cur = seg_end + relativedelta(days=1)

    nb_frames = []
    for seg_start, seg_end in segments:
        try:
            raw = pro.moneyflow_hsgt(start_date=seg_start, end_date=seg_end)
            if raw is not None and len(raw) > 0:
                nb_frames.append(raw)
                logger.info("  %s~%s: %d 行", seg_start, seg_end, len(raw))
            time.sleep(0.2)
        except Exception as e:
            logger.warning("  %s~%s 失败: %s", seg_start, seg_end, e)

    if not nb_frames:
        logger.error("moneyflow_hsgt 全部失败")
        sys.exit(1)

    raw_nb = pd.concat(nb_frames, ignore_index=True)
    raw_nb = raw_nb.drop_duplicates(subset=["trade_date"]).sort_values("trade_date")

    nb_df = pd.DataFrame({
        "date": pd.to_datetime(raw_nb["trade_date"], format="%Y%m%d"),
        "north_net_buy_sh": pd.to_numeric(raw_nb["hgt"], errors="coerce"),
        "north_net_buy_sz": pd.to_numeric(raw_nb["sgt"], errors="coerce"),
    })
    logger.info(
        "moneyflow_hsgt: %d 交易日 (%s ~ %s)",
        len(nb_df), nb_df["date"].min().date(), nb_df["date"].max().date(),
    )

    # ══════════════════════════════════════════════════════════
    # 数据源 2: hsgt_top10 — 前10大成交 buy/sell (2024-08-19 前)
    # ══════════════════════════════════════════════════════════
    logger.info("\n=== 数据源 2: hsgt_top10 (买卖明细, 截止 %s) ===", HSGT_DETAIL_CUTOFF)

    top10_df = _fetch_hsgt_top10_all(pro, start_date, HSGT_DETAIL_CUTOFF)

    if len(top10_df) > 0:
        logger.info(
            "hsgt_top10 汇总: %d 交易日 (%s ~ %s)",
            len(top10_df), top10_df["date"].min().date(), top10_df["date"].max().date(),
        )
        nb_df = nb_df.merge(top10_df, on="date", how="outer")
    else:
        logger.warning("hsgt_top10 无有效数据")
        for c in ["north_buy_amt_sh", "north_buy_amt_sz", "north_sell_amt_sh", "north_sell_amt_sz"]:
            nb_df[c] = None

    # ── 4. 按日期广播更新面板北向列 ──
    logger.info("\n=== 更新面板北向列 ===")

    nb_by_date = nb_df.drop_duplicates(subset=["date"]).set_index("date")

    panel_dates = set(panel["date"].unique())
    nb_dates = set(nb_by_date.index.unique())
    overlap = panel_dates & nb_dates
    logger.info(
        "面板交易日: %d, 北向交易日: %d, 重叠: %d (%.1f%%)",
        len(panel_dates), len(nb_dates), len(overlap),
        len(overlap) / max(len(panel_dates), 1) * 100,
    )

    for col in NORTHBOUND_COLS:
        if col not in nb_by_date.columns:
            logger.info("  %-22s  数据源不提供, 跳过", col)
            continue
        if col not in panel.columns:
            panel[col] = pd.NA

        val_map = nb_by_date[col].dropna()
        if len(val_map) == 0:
            logger.info("  %-22s  数据全 NaN, 跳过", col)
            continue

        # 按日期映射: 有数据用新值, 无数据保留原值
        new_vals = panel["date"].map(val_map)
        panel[col] = new_vals.combine_first(panel[col])

        nn = panel[col].notna().sum()
        logger.info(
            "  %-22s  更新后 %8d non-null  (%5.2f%%)",
            col, nn, nn / len(panel) * 100,
        )

    # ── 5. 合并回完整面板并保存 (WORM 原则) ──
    logger.info("\n=== 合并回完整面板 ===")

    # 读取完整面板为 pyarrow Table (内存高效, 不转 pandas)
    full_table = pq.read_table(PANEL_PATH)

    # 逐列替换: 用更新后的 panel 列覆盖原表中的北向列
    for col in NORTHBOUND_COLS:
        if col in full_table.column_names:
            # 创建新的 column 数组
            new_arr = pa.array(panel[col])
            # 替换 full_table 中的列
            idx = full_table.column_names.index(col)
            full_table = full_table.set_column(idx, col, new_arr)

    today = datetime.now().strftime("%Y%m%d")
    out_path = f"data/panel_full_enriched_v3_northbound_{today}.parquet"
    pq.write_table(full_table, out_path)
    logger.info("已保存: %s (%d rows, %d cols)", out_path, full_table.num_rows, len(full_table.column_names))

    # 释放内存
    del full_table

    # ── 6. 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("=== 北向资金 Tushare 全量填充汇总 ===")
    logger.info("=" * 60)
    logger.info("面板: %s ~ %s (%d 交易日, %d 股票)",
                panel["date"].min().date(), panel["date"].max().date(),
                len(panel_dates), panel["symbol"].nunique())

    logger.info("\n数据源:")
    logger.info("  1. moneyflow_hsgt: %d 交易日净流向 (hgt/sgt, 全量)", len(nb_dates))
    if len(top10_df) > 0:
        t10_dates = top10_df["date"].dt.date.unique()
        logger.info("  2. hsgt_top10: %d 交易日买卖明细 (前10大汇总, 截止 %s)",
                    len(t10_dates), HSGT_DETAIL_CUTOFF)

    logger.info("\n填充结果:")
    for col in NORTHBOUND_COLS:
        if col in panel.columns:
            nn = panel[col].notna().sum()
            pct = nn / len(panel) * 100
            logger.info("  %-22s  %8d non-null  (%5.2f%%)", col, nn, pct)

    logger.info("\n注意:")
    logger.info("  - north_net_buy_sh/sz: moneyflow_hsgt hgt/sgt (百万元)")
    logger.info("  - north_buy/sell_amt: hsgt_top10 前10大汇总 (元), 仅 2024-08-16 前有值")
    logger.info("  - 2024-08-19 后交易所停止公布北向明细, buy/sell 不可得")
    logger.info("  - 如需覆盖 V3: Copy-Item %s %s -Force", out_path, PANEL_PATH)


if __name__ == "__main__":
    main()
