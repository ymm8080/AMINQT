"""GLM 龙虎榜 spec — 席位明细回填 (散户大本营"拉萨"/机构专用).

把面板内 **有 LHB 记录的股-日** (lhb_buy_amt 非空, 2023-01 起全时域) 的席位
买卖金额聚合回填进面板:
  - 散户席位 (营业部名含 "拉萨")  → lhb_retail_buy / lhb_retail_sell
  - 机构专用席位 (名 == "机构专用") → lhb_inst_buy   / lhb_inst_sell

数据源: AKShare stock_lhb_stock_detail_em, per 股-日 (东财接口, 受限流约束),
因此必须逐股-日抓取。本脚本:
  1. 读面板 LHB 股-日清单 → 过滤已 done (断点续跑)
  2. 逐股-日抓取 (买入+卖出 flag), 失败进 retry 列表, 同一 run 内再补一轮
  3. 每 CHUNK_SIZE 个写一块 WORM parquet (data/_lhb_seats_run/chunk_*.parquet) + done.tsv
  4. 全部完成后 --merge: WORM 备份面板 → 并 4 席位列 → 写回 → 验证覆盖率

用法:
  python scripts/_backfill_lhb_seats.py                  # 抓取 (全量, 断点续跑)
  python scripts/_backfill_lhb_seats.py --limit 50       # 验证性小批量
  python scripts/_backfill_lhb_seats.py --sleep 0.6      # 提速 (东财封IP风险自担)
  python scripts/_backfill_lhb_seats.py --merge          # 抓取完成后合并进面板
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.data_supply import DataSupplyChain

PANEL_PATH = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
RUN_DIR = os.path.join(os.path.dirname(PANEL_PATH), "_lhb_seats_run")
DONE_TSV = os.path.join(RUN_DIR, "done.tsv")
RETRY_TSV = os.path.join(RUN_DIR, "retry.tsv")

SEAT_COLS = ["lhb_retail_buy", "lhb_retail_sell", "lhb_inst_buy", "lhb_inst_sell"]
CHUNK_SIZE = 200
DEFAULT_SLEEP = 1.0
CONSECUTIVE_FAIL_PAUSE = 8  # 连续失败 N 个 → 暂停更久 (东财防封)
BACKOFF_SLEEP = 30.0


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_done() -> set[str]:
    if os.path.exists(DONE_TSV):
        with open(DONE_TSV, encoding="utf-8") as fh:
            return {ln.strip() for ln in fh if ln.strip()}
    return set()


def _lhb_stock_days() -> list[tuple[str, str]]:
    """面板中所有有 LHB 记录的 (symbol, date_str), 按日期升序."""
    sub = pd.read_parquet(PANEL_PATH, columns=["symbol", "date", "lhb_buy_amt"])
    sub = sub[sub["lhb_buy_amt"].notna()]
    rows = list(zip(sub["symbol"], sub["date"].dt.strftime("%Y%m%d")))
    return sorted(set(rows), key=lambda x: x[1])


def _flush_chunk(idx: int, df: pd.DataFrame) -> None:
    path = os.path.join(RUN_DIR, f"chunk_{idx:06d}.parquet")
    df.to_parquet(path, index=False)
    _log(f"  WORM 分块落盘: {path} ({len(df)} 行)")


def fetch_seats(limit: int | None, sleep: float) -> None:
    os.makedirs(RUN_DIR, exist_ok=True)
    done = _load_done()
    all_keys = _lhb_stock_days()
    pending = [k for k in all_keys if f"{k[0]}\t{k[1]}" not in done]
    _log(f"LHB 股-日总数 {len(all_keys)}, 已完成 {len(done)}, 待抓 {len(pending)}")
    if limit:
        pending = pending[:limit]
        _log(f"--limit {limit}: 本次只抓 {len(pending)}")

    supply = DataSupplyChain()
    results: list[dict] = []
    chunk_idx = len(glob.glob(os.path.join(RUN_DIR, "chunk_*.parquet")))
    consec_fail = 0
    done_keys: set[str] = done  # 本次 run 内所有完成 key (含已落盘), retry 过滤用

    def _flush():
        nonlocal chunk_idx
        if not results:
            return
        _flush_chunk(chunk_idx, pd.DataFrame(results))
        chunk_idx += 1
        # chunk 落盘成功后才写 done.tsv — "done 标记"与"数据"同生共死,
        # 崩溃时最多丢未落盘的缓冲行, 且它们未标记 done → 续跑会重新抓取
        with open(DONE_TSV, "a", encoding="utf-8") as fh:
            for r in results:
                fh.write(f"{r['symbol']}\t{r['date'].strftime('%Y%m%d')}\n")
        results.clear()

    def _record(key: str, agg: dict) -> None:
        sym, dstr = key
        results.append(
            {
                "symbol": sym,
                "date": pd.Timestamp(dstr),
                **{c: agg.get(c, 0.0) for c in SEAT_COLS},
            }
        )
        done_keys.add(f"{sym}\t{dstr}")

    for i, key in enumerate(pending, 1):
        sym, dstr = key
        try:
            agg = supply.fetch_lhb_seat_aggregate(sym, dstr)
        except Exception:  # noqa: BLE001 — 网络层异常统一兜底
            agg = None
        if agg is None:
            consec_fail += 1
            with open(RETRY_TSV, "a", encoding="utf-8") as fh:
                fh.write(f"{sym}\t{dstr}\n")
            if consec_fail >= CONSECUTIVE_FAIL_PAUSE:
                _log(f"连续 {consec_fail} 个失败, 暂停 {BACKOFF_SLEEP}s (防东财封IP)")
                time.sleep(BACKOFF_SLEEP)
                consec_fail = 0
            continue
        _record(key, agg)
        consec_fail = 0
        if i % 500 == 0:
            _log(f"  进度 {i}/{len(pending)}")
        if i % CHUNK_SIZE == 0:
            _flush()
        time.sleep(sleep)

    _flush()
    _log(f"抓取完成: 处理 {len(pending)} 个, 成功写 done, 失败在 retry.tsv")

    # retry 补一轮 (只补尚未 done 的, 避免重复抓取在 merge 时被 groupby.sum 翻倍)
    if os.path.exists(RETRY_TSV):
        with open(RETRY_TSV, encoding="utf-8") as fh:
            retry_keys = sorted(
                {tuple(ln.strip().split("\t")) for ln in fh if ln.strip()}
            )
        # done_keys 含已落盘 + 本次 run 在途成功的, 二者都不重试, 防重复抓取被合并翻倍
        retry_keys = [k for k in retry_keys if f"{k[0]}\t{k[1]}" not in done_keys]
        still_fail: list[tuple[str, str]] = []
        if retry_keys:
            _log(f"重试 {len(retry_keys)} 个失败项...")
            for sym, dstr in retry_keys:
                try:
                    agg = supply.fetch_lhb_seat_aggregate(sym, dstr)
                except Exception:  # noqa: BLE001
                    agg = None
                if agg is None:
                    still_fail.append((sym, dstr))
                else:
                    _record((sym, dstr), agg)
                    time.sleep(sleep)
            _flush()
        # 重写 retry.tsv 只保留仍失败的 (下次 run 再试)
        with open(RETRY_TSV, "w", encoding="utf-8") as fh:
            for sym, dstr in still_fail:
                fh.write(f"{sym}\t{dstr}\n")
        if still_fail:
            _log(f"仍有 {len(still_fail)} 个失败 (记录在 retry.tsv, 下次 run 再试)")
    _log("回填脚本抓取阶段结束 (未 --merge 则面板未改动)")


def merge_into_panel() -> None:
    chunks = sorted(glob.glob(os.path.join(RUN_DIR, "chunk_*.parquet")))
    if not chunks:
        _log("没有 chunk 文件, 无数据可合并")
        return
    seat = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    # 正常情况每个 (symbol,date) 只出现一次; drop_duplicates 兜底 chunk写盘/done写
    # 之间的崩溃窗口 (重抓会多一行, 同键取最后一次)
    seat = seat.drop_duplicates(subset=["symbol", "date"], keep="last")
    seat = seat.groupby(["symbol", "date"], as_index=False)[SEAT_COLS].sum()
    _log(f"席位表: {len(seat)} 行 (来自 {len(chunks)} 个分块)")

    panel = pd.read_parquet(PANEL_PATH)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.splitext(PANEL_PATH)[0] + f"_prelhb_seats_{ts}.parquet"
    _log(f"WORM 备份: {backup}")
    panel.to_parquet(backup, index=False)

    existing = [c for c in SEAT_COLS if c in panel.columns]
    if existing:
        _log(f"面板已有席位列, 先移除旧列再覆盖合并: {existing}")
        panel = panel.drop(columns=existing)
    before = set(panel.columns)
    panel = panel.merge(seat, on=["symbol", "date"], how="left")
    _log(
        f"合并后列数 {len(before)} → {len(panel.columns)} (新增: {set(panel.columns) - before})"
    )
    panel.to_parquet(PANEL_PATH, index=False)
    _log(f"写回: {PANEL_PATH}")

    lhb_rows = panel["lhb_buy_amt"].notna().sum()
    for c in SEAT_COLS:
        cov = panel[c].notna().sum()
        nz = (panel[c].fillna(0) != 0).sum()
        _log(f"  {c}: 非空 {cov} ({100 * cov / lhb_rows:.1f}% of LHB 行), 非零 {nz}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只抓前 N 个股-日 (验证用)")
    ap.add_argument(
        "--sleep", type=float, default=DEFAULT_SLEEP, help="每股-日间隔秒数"
    )
    ap.add_argument("--merge", action="store_true", help="抓取后合并进面板")
    args = ap.parse_args()

    fetch_seats(args.limit, args.sleep)
    if args.merge:
        merge_into_panel()


if __name__ == "__main__":
    main()
