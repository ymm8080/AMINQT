"""_repair_panel_amount_units.py — 修复 V3 面板尾部 volume/amount 单位 bug (2026-08-04).

背景: _daily_fetch 从 Tushare pro.daily 取数, 其 amount 单位=千元, vol 单位=手.
历史面板由 panel_builder 正确换算 (gu 惯例: volume=手×100=股, amount=千元×1000=元;
少数 ~300 只 hand 惯例: volume=手, amount=元). 但 _daily_fetch 的
`volume = amount/close` 未做千元→元换算, 导致尾部 (2026-07-27 起) 每行:
  amount 缩 1000× (仍为千元), volume 缩 1000× (gu) / 10× (hand).
后果: cleaning.step2 的 amount>=5e7 流动性过滤把最近交易日整日清空, 面板尾部不可用.

本脚本在面板原文件上原位修复 (先备份):
  - 动态检测断裂日期: 首个 ratio=amount/(volume×close)>2 的 symbol 占比跌到 0% 的日期.
  - 每 symbol 惯例由断裂前的历史中位 ratio 判定 (>2 → hand).
  - 断裂日起: amount ×=1000; volume ×=1000 (gu) / ×=10 (hand).
  - 修复后抽样验证 ratio 与自身断裂前一致.

用法: python scripts/_repair_panel_amount_units.py
输出: 面板原位更新 + <path>.pre_amountfix_<ts> 备份 + 控制台验证摘要.
"""

import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
RECENT_DAYS = 60  # 惯例判定用断裂前最近 N 交易日 (部分 symbol 历史惯例会漂移,
# 全史中位会误判, 如 001298: 3.1→2.1→1.45→1.0, 近期实为 gu)


def detect_break_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """断裂日 = 首个 ratio>2 占比 <1% 的交易日 (此后不再回升)."""
    r = df.assign(_r=df["amount"] / (df["volume"] * df["close"]))
    r = r.dropna(subset=["_r"])
    frac = r.groupby("date")["_r"].apply(lambda s: (s > 2).mean())
    frac = frac.sort_index()
    for d in frac.index:
        tail = frac.loc[d:]
        if len(tail) and (tail < 0.01).all():
            return pd.Timestamp(d)
    return None


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(PANEL):
        print(f"FATAL: panel not found {PANEL}")
        return 1

    # 保留原压缩格式 (读后立即 close, 否则 Windows 上 os.replace 报 PermissionError)
    pf = pq.ParquetFile(PANEL)
    comp = pf.metadata.row_group(0).column(0).compression
    print(f"panel compression: {comp} | rows={pf.metadata.num_rows:,}")
    pf.close()

    # 备份
    bak = f"{PANEL}.pre_amountfix_{ts}"
    print(f"备份 -> {bak} ...", flush=True)
    shutil.copy2(PANEL, bak)
    print(f"备份完成 ({os.path.getsize(bak) / 1e9:.2f} GB)", flush=True)

    print("读取面板 ...", flush=True)
    df = pd.read_parquet(PANEL)

    # 1) 断裂日
    brk = detect_break_date(df)
    if brk is None:
        print("未检测到单位断裂 (ratio>2 占比从未跌到 0), 无需修复")
        return 0
    print(f"断裂日: {brk.date()}", flush=True)
    mask = df["date"] >= brk

    # 2) 每 symbol 惯例 (断裂前最近 RECENT_DAYS 交易日的中位 ratio; >2 → hand)
    pre = df[df["date"] < brk]
    recent_days = sorted(pre["date"].unique())[-RECENT_DAYS:]
    recent_pre = pre[pre["date"].isin(recent_days)]
    rr = recent_pre.assign(
        _r=recent_pre["amount"] / (recent_pre["volume"] * recent_pre["close"])
    ).dropna(subset=["_r"])
    med = rr.groupby("symbol")["_r"].median()
    hand = set(med[med > 2].index)
    n_hand, n_gu = len(hand), len(med) - len(hand)
    print(
        f"惯例: hand(手)={n_hand} / gu(股)={n_gu} "
        f"(断裂前最近 {RECENT_DAYS} 交易日判定)",
        flush=True,
    )

    # 3) 修复
    sub = df[mask].copy()
    a = sub["amount"]
    v = sub["volume"]
    c = sub["close"]
    good = a.notna() & v.notna() & (v > 0) & c.notna() & (c > 0)
    is_hand = sub["symbol"].isin(hand) & good
    df.loc[mask, "amount"] = np.where(good, a * 1000.0, a)
    df.loc[mask, "volume"] = np.where(
        good & ~is_hand, v * 1000.0, np.where(good & is_hand, v * 10.0, v)
    )
    print(
        f"修复: {mask.sum():,} 行 (amount×1000; volume gu×1000 / hand×10)", flush=True
    )

    # 4) 验证. 注: 断裂行 volume 由 _daily_fetch 按 amount/close 重构, 故修复后
    #    ratio=amount/(volume×close) 必然是 1.0 (gu) / 100 (hand), 不能据此判一致.
    #    真实正确性判据:
    #    (a) 惯例无错换: gu→ratio<10, hand→ratio>10 (换错因子会跨到对侧量级).
    #    (b) 量级回位: 修复后断裂段 amount / volume 中位与自身断裂前中位同量级.
    after = df[mask].assign(
        _r=df.loc[mask, "amount"] / (df.loc[mask, "volume"] * df.loc[mask, "close"])
    )
    after = after.dropna(subset=["_r"])
    post_ratio = after.groupby("symbol")["_r"].median()
    pre_amt = recent_pre.groupby("symbol")["amount"].median()
    pre_vol = recent_pre.groupby("symbol")["volume"].median()
    post_amt = df.loc[mask].groupby("symbol")["amount"].median()
    post_vol = df.loc[mask].groupby("symbol")["volume"].median()

    rows = pd.DataFrame({"sym": med.index, "pre_ratio": med.values})
    rows["hand"] = rows["pre_ratio"] > 2
    rows["post_ratio"] = [post_ratio.get(s, np.nan) for s in rows["sym"]]
    rows["amt_ratio"] = [
        post_amt.get(s, np.nan) / pa if (pa := pre_amt.get(s)) else np.nan
        for s in rows["sym"]
    ]
    rows["vol_ratio"] = [
        post_vol.get(s, np.nan) / pv if (pv := pre_vol.get(s)) else np.nan
        for s in rows["sym"]
    ]

    # (a) 惯例错换 = gu 被 ×10 (post_ratio≈100) 或 hand 被 ×1000 (post_ratio≈1)
    swap = rows[
        rows["post_ratio"].notna()
        & (
            (rows["hand"] & (rows["post_ratio"] < 10))
            | (~rows["hand"] & (rows["post_ratio"] > 10))
        )
    ]
    print(f"验证(a) 惯例错换: {len(swap)} (应为 0)")
    if len(swap):
        print(f"  WARN: {swap.head(10).to_dict('records')}")

    # (b) 量级回位: amt 应回到历史量级 (±50%), vol 同理. 允许个别因停牌/涨跌停.
    amt_bad = rows[
        (rows["amt_ratio"].notna())
        & (rows["amt_ratio"] < 0.5)
        & (rows["amt_ratio"] > 0)
    ]
    vol_bad = rows[
        (rows["vol_ratio"].notna())
        & (rows["vol_ratio"] < 0.5)
        & (rows["vol_ratio"] > 0)
    ]
    amt_ok = (
        rows[rows["amt_ratio"].notna()]
        .assign(_ok=(rows["amt_ratio"] >= 0.5))
        ._ok.mean()
    )
    vol_ok = (
        rows[rows["vol_ratio"].notna()]
        .assign(_ok=(rows["vol_ratio"] >= 0.5))
        ._ok.mean()
    )
    print(
        f"验证(b) 量级回位: amount {amt_ok:.1%} / volume {vol_ok:.1%} "
        f"symbol 修复后中位 ≥ 断裂前中位 50%"
    )
    if len(amt_bad):
        print(f"  amount 异常偏低 {len(amt_bad)}: {amt_bad.head(8).to_dict('records')}")
    if len(vol_bad):
        print(f"  volume 异常偏低 {len(vol_bad)}: {vol_bad.head(8).to_dict('records')}")
    print(
        f"  修复后 ratio 分布: gu 中位={post_ratio[~rows.set_index('sym')['hand']].median():.3f} "
        f"hand 中位={post_ratio[rows.set_index('sym')['hand']].median():.1f}"
    )

    # 5) 写回 (原子)
    tmp = PANEL + ".tmp"
    df.to_parquet(tmp, index=False, compression=comp)
    os.replace(tmp, PANEL)
    del df
    print(f"写回完成 {PANEL} ({os.path.getsize(PANEL) / 1e9:.2f} GB)", flush=True)
    print("完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
