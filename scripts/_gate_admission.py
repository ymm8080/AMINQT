"""闸准入协议 (2026-09-09 用户: "先判断是否应该用闸, 闸是否有正效果再行动,
写进PIPELINE并验证")。

起因: FADE_GATE 删线凭 21 清单日回放均值直接接线, 当日复核发现三重问题 —
① 半窗翻转 (删除区优势全集中前半: 前半差 +2.2pp / 后半 −0.3pp);
② 大赢家误杀富集 (ret5≥10% 落删除区 18.2% vs 基础删除率 8.9% ≈ 2.0x);
③ 09-09 实盘误杀涨停股 603186(+10%)/601869(+6.0%) → 同日撤线 (528c50bd)。
教训: 均值级回放证据不够, 必须查子窗稳定 + 赢家误杀 + 样本量。

协议 (判据常量 GATE_ADMISSION, config/settings.py):
  接线前: python scripts/_gate_admission.py --gate <name> 判 PASS 才可接线;
  上线后: run_weekly_selfevolve 每周 --gate all 复审, FAIL 大声报告 (告警式)。

口径: 两线 (legacy/parallel) 交付清单回放。执行 = 清单日 t 的 T+1 开盘;
  o2c = T+1 开→收 (执行日体验); 主判据 ret5 = T+1收 → T+5收。
  删除区 (killed) 与生产闸同规则: fade = fade_score ≥ FADE_GATE.kill_score;
  amt_agree = 清单内 (line,date) 组 top AMT_AGREE_GATE.del_top 档。
  因子只用 ≤清单日数据 (滚动窗/pct_change 在 pivot 行 ti 处取值, 无未来泄漏;
  逐清单日调生产 compute_* 因果等价, 一次性向量化仅为提速)。
判词: PASS / FAIL / INSUFFICIENT (样本不足仍给诊断指标, reasons 枚举全部不过线项)。
WORM: DATA_OTHERS_DIR/gate_admission/gate_admission_{gate}_{ts}.json。
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (  # noqa: E402
    AMT_AGREE_GATE,
    DATA_OTHERS_DIR,
    FADE_GATE,
    GATE_ADMISSION,
    PANEL_V3_PATH,
    STOCK_LIST_DIR,
)

_LINE_PATTERNS = {
    "legacy": r"^legacy_stocklist_(\d{8})__.*\.csv$",
    "parallel": r"^parallel_shortlist_(\d{8})__.*\.csv$",
}
_CAL_LOOKBACK = 450  # 覆盖 pos250 (250 交易日) + 节假日余量


# ── 通用评估器 (纯函数, 判据可注入供测试) ──────────────────────
def evaluate_gate(recs: pd.DataFrame, criteria: dict | None = None) -> dict:
    """闸准入判词。

    recs 需含列: list_date, killed(bool), ret5 (o2c 可选, 仅报告)。
    返回 dict: verdict ∈ {PASS, FAIL, INSUFFICIENT} + 全部诊断指标 + reasons。
    """
    cr = criteria if criteria is not None else GATE_ADMISSION
    r = recs[recs["ret5"].notna()].copy()
    insufficient: list[str] = []
    checks: list[str] = []
    n_days = int(r["list_date"].nunique())
    n_killed = int(r["killed"].sum())
    kill_rate = n_killed / len(r) if len(r) else float("nan")

    if n_days < int(cr["min_days"]):
        insufficient.append(f"清单日 {n_days} < {cr['min_days']}")
    if n_killed < int(cr["min_killed"]):
        insufficient.append(f"删除区成熟票 {n_killed} < {cr['min_killed']}")

    killed = r[r["killed"]]
    kept = r[~r["killed"]]

    def _edge(k: pd.DataFrame, p: pd.DataFrame) -> float:
        if not len(k) or not len(p):
            return float("nan")
        return float((p["ret5"].mean() - k["ret5"].mean()) * 100)

    edge_full = _edge(killed, kept)
    days = sorted(pd.to_datetime(r["list_date"]).unique())
    half = len(days) // 2
    halves = {}
    for name, dset in (("front", set(days[:half])), ("back", set(days[half:]))):
        sub = r[pd.to_datetime(r["list_date"]).isin(dset)]
        halves[name] = _edge(sub[sub["killed"]], sub[~sub["killed"]])

    leaks = {}
    for wt in (float(cr["winner_ret5"]), 0.05):
        w = r[r["ret5"] >= wt]
        if len(w) and kill_rate and kill_rate > 0:
            leak = float(w["killed"].mean())
            leaks[f"{wt:.0%}"] = {"leak": leak, "enrich": leak / kill_rate}
        else:
            leaks[f"{wt:.0%}"] = {"leak": None, "enrich": None}  # 无赢家样本=无从误杀

    if not (edge_full >= float(cr["min_edge_pp"])):
        checks.append(f"全窗留存−删除差 {edge_full:+.2f}pp < {cr['min_edge_pp']}pp")
    for name, label in (("front", "前半"), ("back", "后半")):
        if not (halves[name] > float(cr["half_tol_pp"])):
            checks.append(f"{label}半窗差 {halves[name]:+.2f}pp ≤ {cr['half_tol_pp']}pp (半窗不稳)")
    crit = leaks[f"{float(cr['winner_ret5']):.0%}"]
    if crit["enrich"] is not None and crit["enrich"] > float(cr["max_leak_enrich"]):
        checks.append(
            f"大赢家(ret5≥{cr['winner_ret5']:.0%})误杀富集 {crit['enrich']:.2f}x > {cr['max_leak_enrich']}x"
        )

    reasons = insufficient + checks
    verdict = "INSUFFICIENT" if insufficient else ("FAIL" if checks else "PASS")
    rep: dict = {
        "verdict": verdict,
        "reasons": reasons,
        "n_days": n_days,
        "n_picks": int(len(r)),
        "n_killed": n_killed,
        "kill_rate": kill_rate,
        "edge_full_pp": edge_full,
        "edge_front_pp": halves["front"],
        "edge_back_pp": halves["back"],
        "killed_ret5_mean": float(killed["ret5"].mean()) if len(killed) else float("nan"),
        "kept_ret5_mean": float(kept["ret5"].mean()) if len(kept) else float("nan"),
        "winner_leak": leaks,
        "criteria": dict(cr),
    }
    if "o2c" in r.columns and len(killed) and len(kept):
        rep["killed_o2c_mean"] = float(killed["o2c"].mean())
        rep["kept_o2c_mean"] = float(kept["o2c"].mean())
    return rep


# ── 数据装载 ──────────────────────────────────────────────────
def load_lists(lines=("legacy", "parallel"), list_dir=None) -> pd.DataFrame:
    """各线交付清单 (每线每日期 keep-last 版本), 返回 [line, list_date, symbol]."""
    root = Path(list_dir if list_dir is not None else STOCK_LIST_DIR)
    frames = []
    for line, pat in _LINE_PATTERNS.items():
        if line not in lines:
            continue
        best: dict[str, str] = {}
        for f in sorted(os.listdir(root)):
            m = re.match(pat, f)
            if m:
                best[m.group(1)] = f
        for d, f in sorted(best.items()):
            try:
                df = pd.read_csv(root / f, dtype={"symbol": str})
            except Exception:
                continue
            if not len(df) or "symbol" not in df.columns:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "line": line,
                        "list_date": pd.to_datetime(d),
                        "symbol": df["symbol"]
                        .astype(str)
                        .str.split(".")
                        .str[0]
                        .str.zfill(6),
                    }
                )
            )
    if not frames:
        return pd.DataFrame(columns=["line", "list_date", "symbol"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["line", "list_date", "symbol"])
        .reset_index(drop=True)
    )


def _pivots(panel_path, d0) -> dict:
    df = pd.read_parquet(
        panel_path,
        columns=[
            "symbol",
            "date",
            "open",
            "high",
            "close",
            "pre_close",
            "turnover_rate",
            "close_hfq",
            "amount",
        ],
        filters=[("date", ">=", d0)],
    )
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df = df.drop_duplicates(["symbol", "date"])
    pv = {
        "c": df.pivot(index="date", columns="symbol", values="close").sort_index(),
    }
    for k, col in (
        ("o", "open"),
        ("h", "high"),
        ("pc", "pre_close"),
        ("turn", "turnover_rate"),
        ("hfq", "close_hfq"),
        ("amt", "amount"),
    ):
        pv[k] = df.pivot(index="date", columns="symbol", values=col).reindex_like(pv["c"])
    return pv


# ── 因子帧 (与生产 compute_* 因果同口径, 一次性向量化) ─────────
def fade_factor_frame(pv: dict) -> pd.DataFrame:
    """全市场逐日 fade_score (vol20/turn20/r20/pos250 日截面 pct-rank 复合)."""
    c, h, pc, turn = pv["c"], pv["h"], pv["pc"], pv["turn"]
    ret = c / pc - 1
    r20 = c.pct_change(20, fill_method=None)
    vol20 = ret.rolling(20).std()
    turn20 = turn.rolling(20).mean()
    pos250 = c / h.rolling(250, min_periods=60).max() - 1
    return (
        r20.rank(axis=1, pct=True)
        + vol20.rank(axis=1, pct=True)
        + turn20.rank(axis=1, pct=True)
        + pos250.rank(axis=1, pct=True)
    ) / 4


def amt_factor_frame(pv: dict, window: int = 10) -> pd.DataFrame:
    """全市场逐日 amt_agree10 (10日量价配合度)."""
    R = pv["hfq"].pct_change(fill_method=None)
    DA = pv["amt"].diff()
    agree = (((R > 0) & (DA > 0)) | ((R < 0) & (DA < 0))).where(R.notna() & DA.notna())
    return agree.astype(float).rolling(window).mean()


_GATE_FACTORS = {
    "fade": lambda pv: fade_factor_frame(pv),
    "amt_agree": lambda pv: amt_factor_frame(pv, int(AMT_AGREE_GATE.get("window", 10))),
}


def build_records(L: pd.DataFrame, pv: dict, factor: pd.DataFrame) -> pd.DataFrame:
    """清单票 × 因子 × 前向收益 (o2c = T+1开→收; ret5 = T+1收→T+5收).

    ti+5 越界 (ret5 未成熟) 的清单日剔除 — 只审计可结算样本。
    """
    c = pv["c"]
    o2c = c.shift(-1) / pv["o"].shift(-1) - 1
    ret5 = c.shift(-5) / c.shift(-1) - 1
    dates = c.index
    pos = {s: i for i, s in enumerate(c.columns)}
    recs = []
    for line, t, s in zip(L["line"], L["list_date"], L["symbol"]):
        if s not in pos or t not in dates:
            continue
        ti = dates.get_loc(t)
        if ti + 5 >= len(dates):
            continue
        j = pos[s]
        recs.append(
            (line, t, s, factor.iloc[ti, j], o2c.iloc[ti, j], ret5.iloc[ti, j])
        )
    return pd.DataFrame(
        recs, columns=["line", "list_date", "symbol", "factor", "o2c", "ret5"]
    )


def apply_kill_rule(recs: pd.DataFrame, gate: str) -> pd.DataFrame:
    """按生产闸规则打 killed 标 (fade=绝对阈值; amt_agree=组内 top 档)."""
    out = recs.copy()
    if gate == "fade":
        out["killed"] = out["factor"] >= float(FADE_GATE.get("kill_score", 0.75))
    elif gate == "amt_agree":
        frac = float(AMT_AGREE_GATE.get("del_top", 0.2))
        killed = pd.Series(False, index=out.index)
        for _, g in out.groupby(["line", "list_date"]):
            gg = g[g["factor"].notna()].sort_values(
                "factor", ascending=False, kind="mergesort"
            )
            k = max(1, math.ceil(len(gg) * frac))
            killed.loc[gg.head(k).index] = True
        out["killed"] = killed
    else:
        raise ValueError(f"未知闸: {gate}")
    return out


def audit_gate(
    gate: str,
    *,
    panel_path=None,
    list_dir=None,
    criteria: dict | None = None,
    pv: dict | None = None,
    L: pd.DataFrame | None = None,
) -> dict:
    """单闸审计: 装载 → 因子 → killed → 判词 → WORM 落盘."""
    if pv is None:
        if L is None:
            L = load_lists(list_dir=list_dir)
        if not len(L):
            rep = {"gate": gate, "verdict": "INSUFFICIENT", "reasons": ["无交付清单"]}
            _write_report(rep)
            return rep
        d0 = pd.Timestamp(L["list_date"].min()) - pd.Timedelta(days=_CAL_LOOKBACK)
        pv = _pivots(panel_path or PANEL_V3_PATH, d0)
    if L is None:
        L = load_lists(list_dir=list_dir)
    recs = apply_kill_rule(build_records(L, pv, _GATE_FACTORS[gate](pv)), gate)
    rep = evaluate_gate(recs, criteria)
    rep["gate"] = gate
    rep["window"] = {
        "first": str(pd.Timestamp(recs["list_date"].min()).date()) if len(recs) else None,
        "last": str(pd.Timestamp(recs["list_date"].max()).date()) if len(recs) else None,
    }
    _write_report(rep)
    return rep


def _write_report(rep: dict) -> None:
    out_dir = DATA_OTHERS_DIR / "gate_admission"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    doc = out_dir / f"gate_admission_{rep.get('gate', 'unknown')}_{ts}.json"
    with open(doc, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print(
        f"[gateadm] {rep.get('gate')}: {rep['verdict']} "
        f"(日{rep.get('n_days', '-')} 删{rep.get('n_killed', '-')} "
        f"全窗差{rep.get('edge_full_pp', float('nan')):+.2f}pp "
        f"前半{rep.get('edge_front_pp', float('nan')):+.2f}pp/"
        f"后半{rep.get('edge_back_pp', float('nan')):+.2f}pp) → {doc}",
        flush=True,
    )
    for rs in rep.get("reasons", []):
        print(f"  [reason] {rs}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="闸准入审计 (接线前预检 / 每周复审)")
    ap.add_argument("--gate", required=True, choices=["fade", "amt_agree", "all"])
    ap.add_argument("--panel", default=None, help="面板路径 (缺省 PANEL_V3_PATH)")
    ap.add_argument("--list-dir", default=None, help="清单目录 (缺省 STOCK_LIST_DIR)")
    args = ap.parse_args()
    gates = list(_GATE_FACTORS) if args.gate == "all" else [args.gate]

    L = load_lists(list_dir=args.list_dir)
    if not len(L):
        print("[gateadm] 无交付清单, 全部 INSUFFICIENT", flush=True)
        for g in gates:
            rep = {"gate": g, "verdict": "INSUFFICIENT", "reasons": ["无交付清单"]}
            _write_report(rep)
        return 0
    d0 = pd.Timestamp(L["list_date"].min()) - pd.Timedelta(days=_CAL_LOOKBACK)
    pv = _pivots(args.panel or PANEL_V3_PATH, d0)
    print(
        f"[gateadm] 清单 {L['list_date'].nunique()} 日 {len(L)} 票 "
        f"({L['list_date'].min():%Y-%m-%d}..{L['list_date'].max():%Y-%m-%d}), "
        f"面板 {pv['c'].shape}",
        flush=True,
    )
    for g in gates:
        audit_gate(g, panel_path=args.panel, list_dir=args.list_dir, pv=pv, L=L)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
