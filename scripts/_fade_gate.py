"""冲高回落闸 (2026-09-09 用户: "TOP10里如果有冲高回落票需要给提示或者删除").

证据 (_fade_top10_replay.py 回放, 21 个清单日 280 票, 2026-08-05..09-08 legacy 交付清单):
- 状态型 fade_score = vol20/turn20/r20/pos250 的日截面 pct-rank 复合 (全市场口径,
  ≤清单日 t 收盘可知, 无 look-ahead)。≥0.75 的票 (占 8.9%): 执行日 open→close
  −1.03% vs 留守 +0.30~+0.56%, 上涨率 32% vs 55%, 开盘买入日内最深 −3.09% vs
  −2.0%, 5 日 −1.40% vs +0.45% — 五指标一致差 → 真删不补齐。
- 事件型 (清单日当天 g≥3% 且吐回≥70% 且未触板): 收益不差 (c2c +0.86% vs −0.07%)
  但执行日再回落率 14.3% vs 7.4% 翻倍 → 不删, fade_flag 提示列 (md/docx 文字段)。

接线: legacy (_deliver_legacy_list) + parallel (_shortlist_t5_t10) 清单落盘前 —
CSV 落盘后 THS 推送/终版 Excel/合并清单自动继承。回退 enable=False。
[09-09 当日撤删线] 复核发现删线优势半窗翻转 (前半有效/后半消失) + 09-09 实盘误杀
涨停股 → kill_enable=False, 删线代码保留待 ≥40 清单日复验后重议。
[09-09 用户澄清 "要预测是否会冲高回落, 非记录昨日"] fade_score 本就是 t-1 预测
体质分 (vol20/turn20/r20/pos250 → 当日吐回倾向, 研究 IC vol20 +0.19), 加 fade_risk
人读档位列 (高≥risk_hi / 中≥risk_mid / 低; 高档日内冲高回落概率≈1/3 vs 池基线
~23%); fade_flag (昨日事件) 降为辅助。
复验: 累积 ≥40 个清单日后重跑 _fade_top10_replay.py (样本警示: 现证据仅 21 日,
其中 25 只落在删除区)。
因子因果: 只用 ≤清单日 t 的 close/pre_close/high/turnover_rate;
面板缺失/因子算不出 → 不删不标 (fail-open)。
留痕: STOCK_LIST_DIR/faderemoved_{date}__{module}__{line}.csv (WORM, 只在有删时写)。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import FADE_GATE, PANEL_V3_PATH, STOCK_LIST_DIR  # noqa: E402

_LOAD_CAL_DAYS = 430  # 覆盖 pos250 窗 (250 交易日) + 节假日安全余量
_FLAG_TEXT = "昨日冲高回落·易再回落·勿追高"


def compute_fade_profile(day_ts, panel_path=None) -> dict | None:
    """全市场 ≤day_ts 的 fade_score + 当日回落标记.

    返回 {"score": Series(symbol->float), "fade_today": Series(symbol->bool)}；
    面板读失败/无数据 → None (fail-open)。fade_today 仅在面板末行 == day_ts
    当日时给出 (否则缺当日行情, 不标)。
    """
    d_hi = pd.Timestamp(day_ts).normalize()
    d0 = d_hi - pd.Timedelta(days=_LOAD_CAL_DAYS)
    try:
        df = pd.read_parquet(
            panel_path or PANEL_V3_PATH,
            columns=["symbol", "date", "high", "close", "pre_close", "turnover_rate"],
            filters=[("date", ">=", d0), ("date", "<=", d_hi)],
        )
    except Exception:
        return None
    if not len(df):
        return None
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df = df.drop_duplicates(["symbol", "date"])
    c = df.pivot(index="date", columns="symbol", values="close").sort_index()
    if c.empty or len(c) < 25:
        return None
    h = df.pivot(index="date", columns="symbol", values="high").reindex_like(c)
    pc = df.pivot(index="date", columns="symbol", values="pre_close").reindex_like(c)
    turn = df.pivot(
        index="date", columns="symbol", values="turnover_rate"
    ).reindex_like(c)
    ret = c / pc - 1

    r20 = c.pct_change(20, fill_method=None).iloc[-1]
    vol20 = ret.rolling(20).std().iloc[-1]
    turn20 = turn.rolling(20).mean().iloc[-1]
    pos250 = (c / h.rolling(250, min_periods=60).max() - 1).iloc[-1]
    feats = pd.concat([r20, vol20, turn20, pos250], axis=1).dropna()
    if not len(feats):
        return None
    score = feats.rank(pct=True).mean(axis=1).rename("fade_score")

    fade_today = pd.Series(False, index=score.index)
    if c.index[-1] == d_hi:
        g = h.iloc[-1] / pc.iloc[-1] - 1
        giveback = (h.iloc[-1] - c.iloc[-1]) / (h.iloc[-1] - pc.iloc[-1])
        syms = score.index
        ratio = pd.Series(
            [0.2 if s[:3] in ("300", "301", "688", "689") else 0.1 for s in syms],
            index=syms,
        )
        touch = g.reindex(syms).sub(ratio) >= -0.004
        fade_today = (
            (g.reindex(syms) >= 0.03) & (giveback.reindex(syms) >= 0.7) & (~touch)
        ).fillna(False)
    return {"score": score, "fade_today": fade_today}


def apply_fade_gate(
    df: pd.DataFrame,
    day_ts,
    module: str,
    *,
    profile: dict | None = None,
    cfg: dict | None = None,
    list_dir=None,
    line: str = "",
) -> pd.DataFrame:
    """清单内删 fade_score ≥ kill_score 的票 (真删不补齐) + fade_flag 提示列.

    profile 供测试注入, 缺省从面板现算 (全市场口径)。留痕 WORM
    faderemoved_{date}__{module}__{line}.csv。fail-open: 因子不可算 → 不删不标。
    """
    conf = cfg if cfg is not None else FADE_GATE
    if not conf.get("enable", False) or df is None or not len(df):
        return df
    if profile is None:
        profile = compute_fade_profile(day_ts)
    if not profile or not len(profile.get("score", pd.Series(dtype=float))):
        print("[fadegate] 因子不可算, 冲高回落闸未启用 (fail-open)", flush=True)
        return df
    d = df.copy()
    d["_sym"] = d["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    score = profile["score"]
    fade_today = profile.get("fade_today", pd.Series(dtype=bool))
    d["fade_score"] = d["_sym"].map(score).round(3)
    # fade_risk = 预测档位 (09-09 用户: 要预测是否冲高回落, 非记录昨日事件)
    hi = float(conf.get("risk_hi", 0.75))
    mid = float(conf.get("risk_mid", 0.5))
    s = d["fade_score"]
    d["fade_risk"] = pd.Series(
        np.where(s >= hi, "高", np.where(s >= mid, "中", "低")), index=d.index
    ).where(s.notna(), "")
    n_hi = int((d["fade_risk"] == "高").sum())
    if n_hi:
        print(
            f"[fadegate] 冲高回落预测: 高风险 {n_hi} 只 (fade_risk=高, 全市场分位≥{hi}; "
            "该档日内回落概率≈1/3, 慎追高)",
            flush=True,
        )
    thr = float(conf.get("kill_score", 0.75))
    kill = []
    if conf.get("kill_enable", True):
        kill = sorted(d.loc[d["fade_score"].notna() & (d["fade_score"] >= thr), "_sym"])
    if kill:
        out = d[~d["_sym"].isin(kill)].drop(columns=["_sym"]).reset_index(drop=True)
        date8 = pd.Timestamp(day_ts).strftime("%Y%m%d")
        doc = Path(list_dir if list_dir is not None else STOCK_LIST_DIR) / (
            f"faderemoved_{date8}__{module}{f'__{line}' if line else ''}.csv"
        )
        ccols = ["_sym", "fade_score"] + (["board"] if "board" in d.columns else [])
        cut = d[d["_sym"].isin(kill)][ccols].copy().rename(columns={"_sym": "symbol"})
        cut.to_csv(doc, index=False)
        print(
            f"[fadegate] 冲高回落体质闸剔除 {len(kill)} 只 (fade_score≥{thr}): "
            f"{', '.join(kill)} → {doc.name}",
            flush=True,
        )
    else:
        out = d.drop(columns=["_sym"]).reset_index(drop=True)
    if conf.get("flag_enable", True) and len(fade_today):
        out = out.copy()
        flagged = (
            out["symbol"]
            .astype(str)
            .str.split(".")
            .str[0]
            .str.zfill(6)
            .isin(set(fade_today[fade_today].index))
        )
        out["fade_flag"] = np.where(flagged, _FLAG_TEXT, "")
        n_flag = int((out["fade_flag"] != "").sum())
        if n_flag:
            print(
                f"[fadegate] 昨日冲高回落提示 {n_flag} 只 (fade_flag 列, 勿追高)",
                flush=True,
            )
    return out
