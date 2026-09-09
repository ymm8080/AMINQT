# -*- coding: utf-8 -*-
"""市场(上证)冲高回落预测器 — 条件频率口径 (2026-09-09 用户: "给出明天市场FADING预测").

分解: P(回落日_{t+1}) = P(冲高_{t+1}) × P(回落 | 冲高, 状态_t)
  回落日 = g>=0.4% 且 giveback>=60% (与 _fade_research_index.py 同口径)
  状态_t = 今日收盘可知: above_ma20 / r5 / vratio (全部 shift(1) 对齐, 无未来信息)

历史规律 (研究判词 09-09): 吐回过半是常态非异常; 无正簇集; 前置状态
(距MA20/r5/量比/行情切片)有区分度; 回落日次日不偏空 (+0.05% vs +0.03%).
"""
import os

import numpy as np
import pandas as pd

from _fade_research_index import OUT, build_metrics

_REGIME_BINS = [-np.inf, -0.02, 0.0, 0.02, np.inf]
_REGIME_LABELS = ["深跌破MA20(<-2%)", "线下浅(-2~0)", "线上(0~+2%)", "线上强(>+2%)"]
_R5_BINS = [-np.inf, -0.02, 0.02, np.inf]
_R5_LABELS = ["r5<-2%", "r5中(-2~+2%)", "r5>+2%"]


def load_state(fresh_days: int = 10) -> pd.DataFrame:
    """缓存 parquet + 尾部 Tushare 重取 (收盘后校准当日 bar), 只合并不落盘."""
    raw = pd.read_parquet(OUT)
    try:
        import tushare as ts

        token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
        ts.set_token(token)
        pro = ts.pro_api()
        start = raw["trade_date"].max()
        start8 = (pd.to_datetime(start) - pd.Timedelta(days=fresh_days)).strftime("%Y%m%d")
        fr = pro.index_daily(
            ts_code="000001.SH",
            start_date=start8,
            end_date=start,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
        )
        raw = (
            pd.concat([raw, fr])
            .drop_duplicates("trade_date", keep="last")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        print(f"[refetch] 尾部 {len(fr)} 行重取合并, 最新 {raw['trade_date'].iloc[-1]}")
    except Exception as e:  # 网络/token 失败 → 用缓存 (fail-open)
        print(f"[refetch] 跳过 ({e}), 用缓存至 {raw['trade_date'].iloc[-1]}")
    return build_metrics(raw)


def main() -> None:
    d = load_state()
    d["surge"] = d["g"] >= 0.004
    d["fade"] = d["surge"] & (d["giveback"] >= 0.60)
    # 状态 t 预测 t+1 → 全部 shift(1)
    d["st_above"] = d["above_ma20"].shift(1)
    d["st_r5"] = d["r5"].shift(1)
    d["st_vr"] = d["vratio"].shift(1)
    d["regime"] = pd.cut(d["st_above"], _REGIME_BINS, labels=_REGIME_LABELS)
    d["r5b"] = pd.cut(d["st_r5"], _R5_BINS, labels=_R5_LABELS)
    d["vb"] = pd.cut(d["st_vr"], [-np.inf, 0.9, 1.1, np.inf],
                     labels=["缩量<0.9", "常量0.9-1.1", "放量>1.1"])

    t = d.iloc[-1]
    nxt = t["date"] + pd.tseries.offsets.BDay(1)

    print("=" * 78)
    print(f"[今日快照 {t['trade_date']}]  收 {t['close']:.2f} ({t['ret'] * 100:+.2f}%)  "
          f"冲高 g={t['g'] * 100:.2f}%  吐回 {t['giveback'] * 100:.0f}%"
          f"{'' if pd.isna(t['giveback']) else ('  ←今日回落' if t['fade'] else '  ←今日守住' if t['surge'] else '')}")
    print(f"  状态(预测明日用): 距MA20 {t['above_ma20'] * 100:+.2f}%  r5 {t['r5'] * 100:+.2f}%  "
          f"量比 {t['vratio']:.2f}  波动年化 {t['vol20'] * 100:.0f}%")

    print("=" * 78)
    print("[条件频率表 2005-今] 状态=前一日收盘, 事件=当日 (n=交易日数)")
    g = d.groupby("regime", observed=True).agg(n=("surge", "size"), p_surge=("surge", "mean"))
    cond = d[d["surge"]].groupby("regime", observed=True)["fade"].agg(
        n_surge="size", p_fade="mean"
    )
    tab = g.join(cond)
    tab["p_fade_day"] = tab["p_surge"] * tab["p_fade"]
    print((tab.rename(columns={
        "n": "n日", "p_surge": "P(冲高)%", "n_surge": "n冲高",
        "p_fade": "P(回落|冲高)%", "p_fade_day": "P(回落日)%",
    }) * [1, 100, 1, 100, 100]).round(1).to_string())
    base_surge, base_fade = d["surge"].mean(), d["fade"].mean()
    print(f"  无条件: P(冲高) {base_surge * 100:.1f}%  P(回落日) {base_fade * 100:.1f}%")

    # 当前 regime 内按 r5 / 量比细分
    cur_reg = pd.cut([t["above_ma20"]], _REGIME_BINS, labels=_REGIME_LABELS)[0]
    cur_r5 = pd.cut([t["r5"]], _R5_BINS, labels=_R5_LABELS)[0]
    cur_v = pd.cut([t["vratio"]], [-np.inf, 0.9, 1.1, np.inf],
                   labels=["缩量<0.9", "常量0.9-1.1", "放量>1.1"])[0]
    print("-" * 78)
    print(f"[当前 regime={cur_reg}] 内细分 P(回落|冲高)%:")
    sub = d[(d["regime"] == cur_reg) & d["surge"]]
    for col, cur, name in [("r5b", cur_r5, "r5"), ("vb", cur_v, "量比")]:
        s = sub.groupby(col, observed=True)["fade"].agg(["mean", "size"])
        for lab, row in s.iterrows():
            mark = " ←当前" if lab == cur else ""
            print(f"  {name}={lab}: {row['mean'] * 100:.1f}% (n={int(row['size'])}){mark}")

    # 最终预报 = 当前 regime 的 P(回落日), 参考 r5 细分修正方向
    row = tab.loc[cur_reg]
    p_day = row["p_surge"] * row["p_fade"]
    print("=" * 78)
    print(f"[明日 {nxt:%Y-%m-%d} 市场冲高回落预测]")
    print(f"  P(冲高) ≈ {row['p_surge'] * 100:.0f}%  (regime={cur_reg}, n={int(row['n'])})")
    print(f"  P(回落|冲高) ≈ {row['p_fade'] * 100:.0f}%  (其中 r5={cur_r5} 子档, 见上表)")
    print(f"  → P(明日回落日) ≈ {p_day * 100:.0f}%  vs 无条件 {base_fade * 100:.1f}%")
    sig = "高于" if p_day > base_fade * 1.2 else ("低于" if p_day < base_fade * 0.8 else "≈")
    print(f"  判读: {sig}常态; 即便回落, 次日不偏空 (回落日次日 +0.05% vs 全部日 +0.03%)")
    print(f"  样本警示: 单格 n={int(row['n'])} 日, 条件频率非因果; 明日开盘方式未知 (跳空无差异).")


if __name__ == "__main__":
    main()
