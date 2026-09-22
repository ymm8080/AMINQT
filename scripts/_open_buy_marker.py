"""明日开盘勿买标注 (legacy+parallel+genious 三交付共享).

2026-09-22 用户令: "只要告诉我这股第二天开盘不要买就好了"。
入选 = 今日交付清单股; 判定 (全部 T 日收盘 PIT, 见 config OPEN_BUY_RISK):
  1) T 日涨幅 ≥ pct_rise (含涨停);
  2) amplitude_5d 当日全市场截面分位 ≥ amp_rank 且 winner_ratio ≥ winner_min。
命中 → 明日开盘 = "勿买·防高开低走"; 否则空串。

证据 (三臂研究 0922, tmp_t/_0922_gkdzA/B/C_*, 校准 _0922_gkdzD_*):
- trap (T+1 gap≥1% 且 open→close≤−1%) 基率 3.55%; 规则并集全期 trap 10.4%
  (2.9x), 2026H2 11.3% (3.2x); P(低走|高开) 46~48% vs 基线 39.7%;
- 清单域: trap|flag 17~44% vs 未标 1~9% (2026-09 回放);
- 删票闸 (0922 精确性回测 _0922_openbuy_btD 定案): legacy/genious 删 flagged
  (除非过涨闸), parallel 删了更差只标注 — 开关 config OPEN_BUY_RISK["kill"]。
面板缺失/读失败 → 整列空串 (fail-open), 不拦产出。删票闸失败同理 fail-open 不删。
"""

import os

import pandas as pd

from config.settings import OPEN_BUY_RISK, PANEL_V3_PATH


def open_buy_marker(df: pd.DataFrame, trade_date: str, panel_path=None) -> pd.DataFrame:
    """返回加 明日开盘 列的副本 (命中 = 勿买·防高开低走, 其余空串).

    trade_date = YYYYMMDD (清单 T 日); 阈值/列名/文案全在 config OPEN_BUY_RISK.
    """
    cfg = OPEN_BUY_RISK
    out = df.copy()
    out[cfg["col"]] = ""
    p_path = PANEL_V3_PATH if panel_path is None else panel_path
    if p_path is None or not os.path.exists(str(p_path)):
        return out
    p = pd.read_parquet(
        str(p_path),
        columns=["symbol", "date", "pctChg", "amplitude_5d", "winner_ratio"],
    )
    p["symbol"] = p["symbol"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    day = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    p = p[p["date"] == day]
    if p.empty:
        return out
    p = p.drop_duplicates(subset="symbol", keep="last")
    p["amp_rank"] = p["amplitude_5d"].rank(pct=True)
    flag = (p["pctChg"] >= cfg["pct_rise"]) | (
        (p["amp_rank"] >= cfg["amp_rank"]) & (p["winner_ratio"] >= cfg["winner_min"])
    )
    hits = set(p.loc[flag, "symbol"])
    out.loc[out["symbol"].astype(str).isin(hits), cfg["col"]] = cfg["flag_text"]
    return out


def _gate_exempt(symbols, trade_date: str, panel_path=None) -> set[str]:
    """涨闸复算 (清单无涨闸列时的兜底): 返回其中过闸的 symbol 集。

    与回测 tmp_t/_0922_openbuy_btD.py 同口径: kt.load_panel(30d) → 只留候选股
    (compute_features 逐股自身历史, 子集安全) → 当日行 → kt._gate_passed。
    """
    from app.pipeline1 import kongduo_triggers as kt

    p_path = PANEL_V3_PATH if panel_path is None else panel_path
    df = kt.load_panel(p_path, trade_date, lookback_days=30)
    want = {str(s).zfill(6) for s in symbols}
    df = df[df["symbol"].astype(str).str.zfill(6).isin(want)]
    if df.empty:
        return set()
    df = kt.compute_features(df)
    day = df[df["date"] == trade_date]
    if day.empty:
        return set()
    return set(day.loc[kt._gate_passed(day), "symbol"].astype(str).str.zfill(6))


def apply_open_buy_kill(
    df: pd.DataFrame, trade_date: str, line: str, panel_path=None
) -> pd.DataFrame:
    """删票闸 (0922 A/B 裁决): flagged 且未过涨闸 → 删除; 分线开关 config kill。

    genious 清单自带涨闸列 (过闸/没过闸) 直接用; 无该列 (legacy) 走 _gate_exempt 复算。
    涨闸复算异常 → fail-open 本日不删 (标注列仍在, 勿开盘追语义不丢)。
    """
    cfg = OPEN_BUY_RISK
    col = cfg["col"]
    if not cfg.get("kill", {}).get(line, False):
        return df
    if col not in df.columns or df.empty:
        return df
    flagged = df[col].eq(cfg["flag_text"])
    if not flagged.any():
        return df
    sym = df["symbol"].astype(str).str.zfill(6)
    if cfg.get("gate_col") in df.columns:
        exempt = flagged & df[cfg["gate_col"]].eq("过闸")
    else:
        try:
            ok = _gate_exempt(df.loc[flagged, "symbol"], trade_date, panel_path)
        except Exception as exc:  # noqa: BLE001 — 删票闸 fail-open: 标注在, 不删
            print(
                f"[openbuy-kill] {line} 涨闸复算失败 → 本日不删 (fail-open): {exc}",
                flush=True,
            )
            return df
        exempt = flagged & sym.isin(ok)
    victims = flagged & ~exempt
    if victims.any():
        names = ", ".join(sym[victims])
        print(
            f"[openbuy-kill] {line} 剔除 {int(victims.sum())} 只 "
            f"({cfg['flag_text']}, 未过涨闸): {names}; "
            f"豁免过涨闸 {int(exempt.sum())} 只",
            flush=True,
        )
    return df.loc[~victims].reset_index(drop=True)
