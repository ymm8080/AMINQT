"""量价删查线 (2026-09-08 用户拍板 "GO WITH 1"): 交付清单内删 amt_agree10 最高档.

证据 (tmp_t/_envrecheck_vpkill_0908.py 回放, 池=2026-08-05..09-07 两线交付短名单
13日396票): amt_agree10 (10日量价配合度) 在交付池内稳定反向 (日IC −0.081) —
删当日清单 top 20% (k=max(1,ceil(n可算×0.2))) 净 +0.62pp/10日, 被删组 FWD −2.13%,
零误杀大赢 (FWD≥15%), 留存 P90 10.6%→11.9% (legacy +0.87pp / parallel +0.39pp)。
样本警示: 12/13 日在 trend20>0 牛市区, 弱市缺读数。

接线: legacy (_deliver_legacy_list) + parallel (_shortlist_t5_t10) 清单落盘前
真删不补齐 — CSV 落盘后 THS 推送/终版 Excel 自动继承; 密度线无回放证据不接。
因子因果 (与回放同口径): 只用 ≤清单日 t 的 close_hfq/amount
  R = close_hfq.pct_change(); DA = amount.diff()
  agree = ((R>0)&(DA>0)) | ((R<0)&(DA<0)); amt_agree10 = agree.rolling(10).mean()
面板缺失/个股算不出 → 不删 (fail-open)。
留痕: STOCK_LIST_DIR/amtagree_removed_{date}__{module}__{line}.csv (WORM, 只在有删时写;
line=legacy/parallel — 两线模块 tag 相同, 无线标会同名互覆)。
"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import AMT_AGREE_GATE, PANEL_V3_PATH, STOCK_LIST_DIR  # noqa: E402

_LOAD_CAL_DAYS = 45  # 覆盖 window=10 交易日 + 节假日安全余量


def compute_amt_agree10(
    symbols, day_ts, window=10, panel_path=None
) -> pd.Series | None:
    """≤day_ts 各 symbol 的 amt_agree10 (面板读失败/全缺 → None, fail-open)."""
    syms = {str(s).split(".")[0].zfill(6) for s in symbols}
    d0 = pd.Timestamp(day_ts).normalize() - pd.Timedelta(days=_LOAD_CAL_DAYS)
    try:
        df = pd.read_parquet(
            panel_path or PANEL_V3_PATH,
            columns=["symbol", "date", "close_hfq", "amount"],
            filters=[("date", ">=", d0), ("date", "<=", pd.Timestamp(day_ts))],
        )
    except Exception:
        return None
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df = (
        df[df["symbol"].isin(syms)]
        .drop_duplicates(["symbol", "date"])
        .sort_values(["symbol", "date"])
    )
    if not len(df):
        return None
    r = df.groupby("symbol")["close_hfq"].pct_change()
    da = df.groupby("symbol")["amount"].diff()
    agree = (((r > 0) & (da > 0)) | ((r < 0) & (da < 0))).where(r.notna() & da.notna())
    df["ag"] = agree.astype(float)
    df["ag10"] = df.groupby("symbol")["ag"].transform(
        lambda s: s.rolling(window).mean()
    )
    ser = df.groupby("symbol")["ag10"].last().dropna()
    return ser.rename("amt_agree10") if len(ser) else None


def apply_amt_agree_kill(
    df: pd.DataFrame,
    day_ts,
    module: str,
    *,
    ag: pd.Series | None = None,
    cfg: dict | None = None,
    list_dir=None,
    line: str = "",
) -> pd.DataFrame:
    """清单内删 amt_agree10 最高档 (真删不补齐, fail-open, 留痕 WORM).

    module 进留痕文件名 (amtagree_removed_{date}__{module}.csv); line 为线标
    ("legacy"/"parallel") — 两线共享同一模块 tag 时区分留痕, 防同名互覆 (09-08
    首跑实发: parallel 覆盖 legacy 删票记录)。ag 供测试注入, 缺省从面板现算。
    """
    conf = cfg if cfg is not None else AMT_AGREE_GATE
    if not conf.get("enable", False) or df is None or not len(df):
        return df
    if ag is None:
        ag = compute_amt_agree10(df["symbol"], day_ts, window=conf.get("window", 10))
    if ag is None or not len(ag):
        print("[amtagree] 因子不可算, 删查线未启用 (fail-open)", flush=True)
        return df
    d = df.copy()
    d["_sym"] = d["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    m = d.drop_duplicates("_sym").merge(
        ag.rename("ag"), left_on="_sym", right_index=True, how="inner"
    )
    if not len(m):
        print("[amtagree] 清单内无因子可算股, 未删 (fail-open)", flush=True)
        return df
    k = max(1, math.ceil(len(m) * float(conf.get("del_top", 0.2))))
    m = m.sort_values("ag", ascending=False, kind="mergesort")
    kill = m["_sym"].head(k).tolist()
    out = d[~d["_sym"].isin(kill)].drop(columns=["_sym"]).reset_index(drop=True)
    cut = m.head(k)[["_sym", "ag"]].rename(
        columns={"_sym": "symbol", "ag": "amt_agree10"}
    )
    date8 = pd.Timestamp(day_ts).strftime("%Y%m%d")
    doc = Path(list_dir if list_dir is not None else STOCK_LIST_DIR) / (
        f"amtagree_removed_{date8}__{module}{f'__{line}' if line else ''}.csv"
    )
    cut.to_csv(doc, index=False)
    print(
        f"[amtagree] 删查线剔除 {len(kill)} 只 (amt_agree10 顶 "
        f"{float(conf.get('del_top', 0.2)):.0%}): {', '.join(kill)} → {doc.name}",
        flush=True,
    )
    return out
