"""全市场宇宙修复 Step 5c: 补拉 income 失败/缺失股票的 fina 3 列 (2026-08-16).

背景: _fix_new_panel_alt.py 逐股拉 income 时偶发 Tushare 超时 (Read timed out),
失败股被跳过 → 面板中这些股票 net_margin/eps_yoy/profit_yoy 全 NaN.

本脚本 (08-16 用户新要求, 独立实现不复用 pull_income_yoy):
  1. 读最新 base_new_full_*.parquet (fix 脚本输出)
  2. retry 集合 = 日志 FAIL 的 symbol ∪ 面板中 net_margin 全 NaN 的 symbol
     (后者覆盖 income 返回空/无 update_flag=1 行的静默跳过股)
  3. 每只 1.5s sleep (限流严重, 不用 0.25s); 每只最多 3 次退避重试 (2/5/10s)
  4. 3 次仍失败的记录日志并跳过, 最后报告剩余失败数
  5. merge_asof (left_on=date, right_on=announce_date, by=symbol, backward,
     两边 on 列全局排序: pandas 2.3.3 要求全局单调)
  6. schema 归一化: fix 脚本 merge 时面板已有 announce_date 会产生 x/y 后缀列,
     本脚本按生产 build_full_panel 同模式归一为单一 announce_date
     (非 retry 股: y 优先, NaT 回退 x; retry 股: merge_asof 重新引入)
  7. WORM: 输出新 base_new_full_<new_ts>.parquet, 旧文件不动
"""

from __future__ import annotations

import glob
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tushare as ts  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402

OUT_PANEL_DIR = "data/new_symbols_panel"
LOG = "logs/b_chain.log"

FINA_COLS = ["net_margin", "eps_yoy", "profit_yoy"]

# 08-16 用户要求: 限流严重, 不用 pull_income_yoy 的 0.25s
CALL_SLEEP = 1.5
RETRIES = 3
BACKOFF = (2, 5, 10)
FLUSH_EVERY = 25


def _income_rows_one(raw: pd.DataFrame, sym: str) -> list[dict]:
    """单只 income 原始表 → PIT 行 (公式与 _fix_new_panel_alt.pull_income_yoy 逐字一致)."""
    raw = raw[raw["update_flag"] == "1"].copy()
    if raw.empty:
        return []
    raw["end_date"] = raw["end_date"].astype(str)
    raw = raw.drop_duplicates(subset="end_date", keep="last")
    raw = raw.sort_values("end_date")
    end_to_row = {r["end_date"]: r for _, r in raw.iterrows()}
    rows = []
    for _, r in raw.iterrows():
        end_date = r["end_date"]
        prev_end = f"{int(end_date[:4]) - 1}{end_date[4:]}"
        prev = end_to_row.get(prev_end)
        nm = (
            r["n_income"] / r["total_revenue"] * 100.0
            if r["total_revenue"] and r["total_revenue"] != 0
            else np.nan
        )
        eps_yoy = (
            (r["basic_eps"] / prev["basic_eps"] - 1.0) * 100.0
            if prev is not None and prev["basic_eps"] not in (None, 0)
            else np.nan
        )
        py_yoy = (
            (r["n_income_attr_p"] / prev["n_income_attr_p"] - 1.0) * 100.0
            if prev is not None and prev["n_income_attr_p"] not in (None, 0)
            else np.nan
        )
        rows.append(
            {
                "symbol": sym,
                "announce_date": r["ann_date"],
                "net_margin": nm,
                "eps_yoy": eps_yoy,
                "profit_yoy": py_yoy,
            }
        )
    return rows


def pull_income_retry(
    pro, symbols: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """逐股拉 income, 每只 3 次退避重试 (2/5/10s), 股间 sleep 1.5s.

    返回 (fina_df, fail_syms, empty_syms):
      fina_df    [symbol, announce_date, net_margin, eps_yoy, profit_yoy]
      fail_syms  3 次全异常的 symbol (已记日志)
      empty_syms 拉取成功但无可用数据的 symbol
    """
    parts = []
    fail_syms: list[str] = []
    empty_syms: list[str] = []
    t0 = time.time()
    for i, sym in enumerate(symbols):
        code = sym + (".SH" if sym.startswith(("6", "5")) else ".SZ")
        last_exc: Exception | None = None
        raw = None
        for attempt in range(1, RETRIES + 1):
            try:
                raw = pro.income(ts_code=code)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < RETRIES:
                    time.sleep(BACKOFF[attempt - 1])
        if last_exc is not None and raw is None:
            fail_syms.append(sym)
            print(f"    [retry] {sym}: FAIL x{RETRIES} ({last_exc})", flush=True)
            time.sleep(CALL_SLEEP)
            continue
        rows = _income_rows_one(raw, sym)
        if rows:
            parts.append(pd.DataFrame(rows))
        else:
            empty_syms.append(sym)
        time.sleep(CALL_SLEEP)
        if (i + 1) % FLUSH_EVERY == 0:
            rate = (i + 1) / (time.time() - t0) * 3600
            print(f"[retry] {i + 1}/{len(symbols)} ({rate:.0f}/hr)", flush=True)
    if not parts:
        return pd.DataFrame(), fail_syms, empty_syms
    out = pd.concat(parts, ignore_index=True)
    out["announce_date"] = pd.to_datetime(
        out["announce_date"], format="%Y%m%d", errors="coerce"
    )
    return out.dropna(subset=["announce_date"]), fail_syms, empty_syms


def _fail_symbols_from_log() -> set[str]:
    if not os.path.exists(LOG):
        return set()
    pat = re.compile(r"\[income\] (\d{6}): FAIL")
    out: set[str] = set()
    with open(LOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out.add(m.group(1))
    return out


def main() -> None:
    f = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_full_*.parquet")))[-1]
    df = pd.read_parquet(f)
    print(
        f"[retry] input={os.path.basename(f)} rows={len(df):,} "
        f"syms={df['symbol'].nunique()} cols={len(df.columns)}",
        flush=True,
    )
    if not set(FINA_COLS) <= set(df.columns):
        raise SystemExit(f"FATAL: 面板缺 {FINA_COLS}, 先跑 _fix_new_panel_alt.py")

    fail_log = _fail_symbols_from_log()
    cnt = df.groupby("symbol")["net_margin"].count()
    nan_syms = {s for s in cnt.index[cnt == 0].astype(str)}
    retry = sorted(fail_log | nan_syms)
    print(
        f"[retry] FAIL(log)={len(fail_log)} NaN(panel)={len(nan_syms)} "
        f"retry={len(retry)}: {retry[:40]}",
        flush=True,
    )
    if not retry:
        print("[retry] 无需补拉, 面板已是最新, 不做任何改动", flush=True)
        return

    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    fina, fail_syms, empty_syms = pull_income_retry(pro, retry)
    print(
        f"[retry] pull done: fina rows={len(fina):,} 成功 {fina['symbol'].nunique() if len(fina) else 0}/{len(retry)} 只 | "
        f"FAIL x3={len(fail_syms)} | 空数据={len(empty_syms)}",
        flush=True,
    )
    if len(fail_syms):
        print(f"[retry] FAIL x3 list: {fail_syms}", flush=True)
    if len(empty_syms):
        print(f"[retry] 空数据 list: {empty_syms}", flush=True)

    retry_set = set(retry)
    mask = df["symbol"].isin(retry_set)

    if fina.empty:
        # 全失败: 不写 WORM 副本 (面板无任何改动), 只报剩余失败数后退出
        print(
            f"[retry] FATAL: 补拉仍全部失败, REMAIN_FAIL={len(retry)}, "
            "面板不重写 (QC 请直接用 fix 脚本输出)",
            flush=True,
        )
        raise SystemExit(1)
    else:
        sub = df[mask]
        rest = df[~mask].copy()
        # ── sub: 去掉旧 fina 相关列后重 merge (生产 build_full_panel 同模式) ──
        drop = [
            c
            for c in FINA_COLS + ["announce_date", "announce_date_x", "announce_date_y"]
            if c in sub.columns
        ]
        sub = sub.drop(columns=drop).sort_values("date").reset_index(drop=True)
        fina_s = (
            fina[["symbol", "announce_date"] + FINA_COLS]
            .sort_values("announce_date")
            .reset_index(drop=True)
        )
        sub = pd.merge_asof(
            sub,
            fina_s,
            left_on="date",
            right_on="announce_date",
            by="symbol",
            direction="backward",
        )

        # ── rest: 归一 announce_date_x/y → 单一 announce_date (y 优先, NaT 回退 x) ──
        if "announce_date" not in rest.columns:
            if "announce_date_y" in rest.columns and "announce_date_x" in rest.columns:
                rest["announce_date"] = rest["announce_date_y"].fillna(
                    rest["announce_date_x"]
                )
                print("[retry] announce_date 归一: y 优先, NaT 回退 x", flush=True)
            elif "announce_date_x" in rest.columns:
                rest = rest.rename(columns={"announce_date_x": "announce_date"})
            elif "announce_date_y" in rest.columns:
                rest = rest.rename(columns={"announce_date_y": "announce_date"})
        rest = rest.drop(
            columns=["announce_date_x", "announce_date_y"], errors="ignore"
        )

        df_new = pd.concat([rest, sub], ignore_index=True)
        df_new = df_new.sort_values(["symbol", "date"]).reset_index(drop=True)
        del df, rest, sub

    # ── 剩余失败报告 (08-16 用户要求: 最后报告剩余失败数) ──
    retry_mask = df_new["symbol"].isin(retry_set)
    still_nan = set(
        df_new.loc[retry_mask]
        .groupby("symbol")["net_margin"]
        .count()
        .pipe(lambda s: s.index[s == 0])
        .astype(str)
    )
    print(
        f"[retry] REMAIN_FAIL={len(still_nan)} "
        f"(pull FAIL x3={len(fail_syms)}, 空数据={len(empty_syms)})",
        flush=True,
    )
    if still_nan:
        print(f"[retry] REMAIN_FAIL list: {sorted(still_nan)}", flush=True)

    cov_retry = df_new.loc[retry_mask, "net_margin"].notna().mean()
    print(
        f"[retry] retry 股 net_margin 覆盖 0% → {cov_retry:.1%}",
        flush=True,
    )

    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_full_{ts_}.parquet")
    df_new.to_parquet(out, index=False)
    print(f"[retry] save {out}", flush=True)
    cov = df_new[FINA_COLS + ["announce_date"]].notna().mean().round(3)
    print("[retry coverage]")
    print(cov.to_string(), flush=True)
    print(f"[retry] final cols={len(df_new.columns)}", flush=True)
    print("INCOME RETRY DONE", flush=True)


if __name__ == "__main__":
    main()
