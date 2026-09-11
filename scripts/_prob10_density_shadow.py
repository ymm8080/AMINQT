"""概率头密度版影子单: prob前20带+带内密度≥3+回撤闸+派发闸 → 同花顺自选股
(2026-09-06 用户拍板把原 TOP10+额1亿 口径整线替换为 L3×TOP20免额; 线名/文件/
夜链位置/死区线名不变).

口径 (09-06 拍板; 125d 回放 tmp_t/_band_compare_0906.py 2026-02-09..08-17:
  main 5.5只/日 赢率54.7%/+7.82pp 大亏3.5%, dual 6.0只/日 47.7%/+7.62pp 大亏5.1%
  — 随机基准 = 全市场净≥5% 赢率 13.5%, 两板 ≈3.5~4 倍随机):
  ①带成员 = 每板 (main; dual=GEM+STAR) 按 legacy 概率头 prob_up_10d 降序 前20
    (原 TOP10 榜 → TOP20 带; 密度累计宇宙同步换成带, 11-20 名滞留也攒天数)
  ②回撤闸 = 收盘距 10 日高点回撤 ≥ -10% (信号夜可知)
  ③密度 = occ5≥3, occ5 对应研究带内 OCC5=rolling(5) 含当日: 今日在带 +
    近4个上榜历日在带数
  ④免额 (09-06 拍板 "去额"; 2×2 终审: 额闸在 prob 池头部近似装饰 — 撤之
    +0.1~0.2只/日流量, 赢率代价 1~1.7pp); amt 列保留仅展示, 不作闸
  ⑤撞指数码 000xxx 不剔 (09-05 用户澄清 "不是删除股票号"), 推送端隔离指数行
    — 见 _ths_watchlist_push._build_chunks
  ⑥筹码派发标注 (09-05 三线统一删 → 09-09 用户推翻改标注 "派发不删, 清单标注"):
    获利盘5日回落 (wr5<0) → chip_flag=派发 列标注, 不删票; cyq 数据缺/个股特征缺
    → 不标 (fail-open)。同标注接 LEGACY 交付 (_deliver_legacy_list) 与 PARALLEL
    短名单 (_shortlist_t5_t10)。
  标签列 信念降 belief_down = 今日 prob − 3个上榜历日前 prob (非闸; 09-06 L4
  对照 = 半流量换 +1.6pp 判不接, 列保留供影子期攒证据)。
  双模型列 (2026-09-05 用户): legacy_prob/legacy_pred10 = legacy 概率头
  prob_up_10d / 幅度头 pred_ret_10d (选股口径即 legacy 概率头); parallel_prob/
  parallel_pred10 = parallel raw pred_prob_10d / pred_mag_10d
  (parallel_preds_raw_{date}__*.csv 全池落盘; raw 未校准), 缺文件 → NaN。
  交付 CSV 百分比显示 (2026-09-05 用户): 上述预测列+回撤列写 "%" 文本,
  pctChg 只加 %; 见 fmt_pct_display, 纯显示层不影响机器读。

上榜历史: data/prob10_density_history.parquet (date/board/symbol/prob)。
  引导: data/_diag_rankkey_scored_{board}_e125.parquet (125d 连续全池打分, 与
  研究同源) 补 candidates 未覆盖日期, data/lists/candidates_*.parquet 补其后的
  日期 (旧 vintage 缺 prob_up_10d 列跳过); 每夜追当日成员, 重跑同日先删后追
  (幂等)。2026-09-06 口径替换时已按 TOP20 带整文件重建 (e125+candidates)。
  注: 打分文件止于 08-17, 08-18..08-29 无 candidates 文件为空洞 — 密度窗会伸到
  08-17, candidates 积累 ≥5 日后自然收敛到纯交易日历窗。

生成 (STOCK_LIST_DIR, WORM):
  prob10dens_{date}__prob10dens.csv      交付文档
  ths_watchlist_{date}__prob10dens.txt   ths_push 同款导入格式; _ths_flush_guard
                                         glob 成员并集自动覆盖本清单

用法: python scripts/_prob10_density_shadow.py [YYYYMMDD] [--gen-only] [--dry-run]
  缺省 date = candidates 最新日; 当日 candidates 缺失则跳过 (fail-safe, 非关键步骤)。
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import DATA_DIR, PANEL_V3_PATH, STOCK_LIST_DIR
from scripts import _deadzone_guard
from scripts._pctfmt import fmt_pct_columns

MODULE = "prob10dens"
TOP_N = 20  # 带成员: 每板 prob 前20 (09-06 拍板, 原 top10 榜)
PULL_FLOOR = -0.10  # 回撤闸: 距10日高点回撤下限
OCC_WIN = 5  # 密度窗: 近 5 个上榜日
OCC_MIN = 3  # 密度阈: 带内在榜 ≥3 天 (免额, 09-06 拍板)
HIST_PATH = os.path.join(DATA_DIR, "prob10_density_history.parquet")
CHIP_WR5_MAX = (
    0.0  # 派发闸: 获利盘5日变化须低于此值 (负=回落; 09-05 三线统一 wr5<0 即剔)
)
CYQ_PATH = os.path.join(DATA_DIR, "cyq_panel.parquet")

_COLS = [
    "rank",
    "board",
    "symbol",
    "legacy_prob",
    "legacy_pred10",
    "parallel_prob",
    "parallel_pred10",
    "occ5",
    "pull",
    "amt",
    "belief_down",
    "chip_wr5",
    "chip_flag",
]


def _board_of(b: str) -> str:
    return "main" if b == "main" else "dual"  # GEM/STAR → dual


def _membership_core(
    c: pd.DataFrame, prob_col: str, top_n: int = TOP_N
) -> pd.DataFrame:
    """按 date/board 分组取 prob 降序前 top_n → date/board/symbol/prob (纯函数).

    默认 top_n=TOP_N=20 (09-06 拍板 TOP20 带, 原 top10 榜).
    """
    c = c.copy()
    c["symbol"] = c["symbol"].astype(str).str.zfill(6)
    c["board"] = c["board"].map(_board_of)
    c["prob"] = c[prob_col].astype(float)
    c = c.sort_values(
        ["board", "date", "prob", "symbol"], ascending=[True, True, False, True]
    )
    return (
        c.groupby(["board", "date"], sort=False)
        .head(top_n)[["date", "board", "symbol", "prob"]]
        .reset_index(drop=True)
    )


def prob10_membership(cand: pd.DataFrame, day_ts: pd.Timestamp) -> pd.DataFrame:
    """candidates 当日截面 → TOP20 带成员 (date/board/symbol/prob, 纯函数).

    board 映射 main→main, GEM/STAR→dual; 每板按 prob 降序 (并列 symbol 升序)
    取前20 (09-06 拍板, 原 top10); 北交所不在 candidates 无需剔。
    """
    return _membership_core(cand.assign(date=day_ts), "prob_up_10d")


def load_chip_features(day_ts: pd.Timestamp) -> pd.DataFrame | None:
    """筹码派发特征 wr5 (入选日收盘可知, 无前视); 数据缺 → None (fail-open).

    wr5 = 获利盘 − 5 个交易日前获利盘 (负=回落=派发方向)。
    T = cyq 最新一日 ≤ day_ts (cyq 止于 T-1 时特征滞后一日, 方向不变), T-5 取其
    前第 5 行; 个股缺行 → NaN (闸内比较恒 False → 不拦)。文件缺失/空/不足 6 行
    → None, 派发闸整体不启用。
    """
    if not os.path.exists(CYQ_PATH):
        return None
    cq = pd.read_parquet(
        CYQ_PATH,
        columns=["symbol", "date", "winner_ratio"],
        filters=[
            ("date", ">=", day_ts - pd.Timedelta(days=21)),
            ("date", "<=", day_ts),
        ],
    )
    if cq.empty:
        return None
    cq["symbol"] = cq["symbol"].astype(str).str.zfill(6)
    cq = cq.drop_duplicates(["symbol", "date"], keep="last")
    wr = cq.pivot(index="date", columns="symbol", values="winner_ratio").sort_index()
    if len(wr.index) < 6:
        return None
    i = len(wr.index) - 1
    return pd.DataFrame(
        {
            "symbol": wr.columns.astype(str),
            "wr5": wr.iloc[i].values - wr.iloc[i - 5].values,
        }
    )


def apply_wr5_gate(
    df: pd.DataFrame, chip: pd.DataFrame | None
) -> tuple[pd.DataFrame, list[str]]:
    """派发标注 (2026-09-09 用户拍板 "派发不删, 清单标注"): wr5<0 → chip_flag=派发.

    09-05~09-09 曾为删除闸 (wr5<0 真删不补齐); 09-09 用户推翻 — 单日 wr5 噪声大,
    删票丢强名, 改为清单标注让人裁。加列 chip_wr5 (获利盘5日变化, 数值) +
    chip_flag ("派发"/""); 不删任何行。
    chip None/空 或 df 空 → 原样返回 (fail-open); 个股 wr5 NaN → 不标。
    返回 (标注后 df 副本, 被标 symbol 列表)。
    生产接线: 密度影子单 (density_picks) / LEGACY 交付 (_deliver_legacy_list) /
    PARALLEL 短名单 (_shortlist_t5_t10)。
    """
    if chip is None or not len(chip) or df.empty:
        return df, []
    ch = chip.drop_duplicates("symbol", keep="last").set_index("symbol")
    d = df.copy()
    sym = d["symbol"].astype(str).str.zfill(6)
    wr = sym.map(ch["wr5"])
    flagged = wr < CHIP_WR5_MAX  # NaN < x → False → 不标
    d["chip_wr5"] = wr
    d["chip_flag"] = np.where(flagged, "派发", "")
    if not flagged.any():
        return d, []
    return d, sorted(sym[flagged].unique())


def apply_chip_gate(
    df: pd.DataFrame, day_ts: pd.Timestamp, flush: bool = False
) -> pd.DataFrame:
    """派发标注接线入口 (三线共享): load_chip_features → apply_wr5_gate → 标注日志.

    cyq 数据缺 → 原样返回 (fail-open)。LEGACY 交付 (_deliver_legacy_list) 与
    PARALLEL 短名单 (_shortlist_t5_t10) 调用; 密度影子单走 density_picks 内联
    (main 里另有标注日志, 不走此处避免重复打印)。
    """
    chip = load_chip_features(day_ts)
    if chip is None:
        print("[chipgate] 筹码数据缺失, 派发标注未启用 (fail-open)", flush=flush)
        return df
    out, cut = apply_wr5_gate(df, chip)
    if cut:
        print(
            f"[chipgate] 派发标注 {len(cut)} 只 (获利盘5日回落, chip_flag=派发): "
            f"{', '.join(cut)}",
            flush=flush,
        )
    return out


def density_picks(
    cand: pd.DataFrame,
    hist: pd.DataFrame,
    close: pd.DataFrame,
    amount: pd.DataFrame,
    day_ts: pd.Timestamp,
    par: pd.DataFrame | None = None,
    chip: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """带密度≥3+回撤闸+派发方向 → 当日影子清单 (纯函数, 可单测).

    cand: 当日 candidates 截面 (symbol/board/prob_up_10d/pred_ret_10d)
    hist: 带上榜历史 (date/board/symbol/prob), 须不含当日 (当日成员由 cand 现算)
    close/amount: 透视表 (date × symbol), ≤ day_ts; amount 仅算 amt 展示列
    par: parallel 全池 raw 预测 (symbol/pred_mag_10d/pred_prob_10d);
         None/缺 → parallel 两列 NaN
    chip: 筹码派发特征 (symbol/wr5, load_chip_features 产出); None →
          不加派发标注列; 个股特征 NaN → 不标 (fail-open)
    """
    memb = prob10_membership(cand, day_ts)
    c = cand.copy()
    c["symbol"] = c["symbol"].astype(str).str.zfill(6)
    c["board"] = c["board"].map(_board_of)
    m = memb.merge(
        c[["symbol", "board", "pred_ret_10d"]].rename(
            columns={"pred_ret_10d": "pred10"}
        ),
        on=["symbol", "board"],
        how="left",
    )

    cl = close[close.index <= day_ts].sort_index()
    am = amount.reindex(cl.index)
    pull = (cl / cl.rolling(10, min_periods=2).max() - 1).iloc[-1]
    amt = am.iloc[-1]
    m["pull"] = m["symbol"].map(pull)
    m["amt"] = m["symbol"].map(amt)

    # occ5 研究口径 OCC5=rolling(5) 含当日: 1(今日在带) + 近4个上榜历日在带数;
    # belief_down 对应研究 PM3=shift(3): 今日 prob − 3个上榜历日前 prob (未在带=NaN)
    hdates = sorted(hist["date"].unique())
    win4 = set(hdates[-(OCC_WIN - 1) :]) if hdates else set()
    d3 = hdates[-3] if len(hdates) >= 3 else None
    occ, p3v = [], []
    for r in memb.itertuples():
        h = hist[(hist["board"] == r.board) & (hist["symbol"] == r.symbol)]
        ds = set(h["date"].unique())
        occ.append(1 + len(ds & win4))
        p3 = h[h["date"] == d3]["prob"] if d3 is not None else None
        p3v.append(float(p3.iloc[0]) if p3 is not None and len(p3) else np.nan)
    m["occ5"] = occ
    m["belief_down"] = [
        np.nan if np.isnan(v) else r.prob - v for v, r in zip(p3v, memb.itertuples())
    ]

    ok = m[
        (m["pull"].fillna(-1) >= PULL_FLOOR) & (m["occ5"] >= OCC_MIN)
    ].copy()  # 免额 (09-06 拍板): 额不作闸, amt 仅展示列
    ok, _ = apply_wr5_gate(ok, chip)
    if "chip_wr5" not in ok.columns:  # chip 缺 (fail-open) 也保稳定 schema
        ok["chip_wr5"] = np.nan
        ok["chip_flag"] = ""
    ok = ok.rename(columns={"prob": "legacy_prob", "pred10": "legacy_pred10"})
    if par is not None and len(par):
        p = par[["symbol", "pred_prob_10d", "pred_mag_10d"]].copy()
        p["symbol"] = p["symbol"].astype(str).str.zfill(6)
        ok = ok.merge(
            p.drop_duplicates("symbol", keep="last").rename(
                columns={
                    "pred_prob_10d": "parallel_prob",
                    "pred_mag_10d": "parallel_pred10",
                }
            ),
            on="symbol",
            how="left",
        )
    else:
        ok["parallel_prob"] = np.nan
        ok["parallel_pred10"] = np.nan
    ok["_b"] = (ok["board"] != "main").astype(int)  # 交付顺序 main 在前
    ok = ok.sort_values(["_b", "legacy_prob", "symbol"], ascending=[True, False, True])
    ok = ok.drop(columns="_b").reset_index(drop=True)
    ok.insert(0, "rank", np.arange(1, len(ok) + 1))
    return ok[_COLS]


def load_or_bootstrap_history(day_ts: pd.Timestamp) -> pd.DataFrame:
    """上榜历史: 有文件读文件; 无则引导 (打分文件补连续史 + candidates 补尾部)."""
    if os.path.exists(HIST_PATH):
        return pd.read_parquet(HIST_PATH)
    frames, cand_dates = [], set()
    for fp in sorted(
        glob.glob(os.path.join(DATA_DIR, "lists", "candidates_*.parquet"))
    ):
        d = pd.Timestamp(os.path.basename(fp)[11:19])
        if d >= day_ts:
            continue
        try:
            c = pd.read_parquet(fp, columns=["symbol", "board", "prob_up_10d"])
        except ValueError:
            print(f"[warn] 引导跳过 (缺 prob_up_10d): {os.path.basename(fp)}")
            continue
        cand_dates.add(d)
        frames.append(_membership_core(c.assign(date=d), "prob_up_10d"))
    for board in ("main", "dual"):
        fp = os.path.join(DATA_DIR, f"_diag_rankkey_scored_{board}_e125.parquet")
        if not os.path.exists(fp):
            print(f"[warn] 引导缺打分文件: {os.path.basename(fp)}")
            continue
        ck = pd.read_parquet(fp, columns=["date", "board", "symbol", "prob"])
        ck["date"] = pd.to_datetime(ck["date"])
        ck = ck[(ck["date"] < day_ts) & (~ck["date"].isin(cand_dates))]
        if len(ck):
            frames.append(_membership_core(ck, "prob"))
    h = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["date", "board", "symbol", "prob"])
    )
    return h


def save_history(h: pd.DataFrame, memb: pd.DataFrame) -> None:
    h = h[~h["date"].isin(memb["date"].unique())]
    out = pd.concat([h, memb], ignore_index=True)
    out.to_parquet(HIST_PATH, index=False)


def fmt_pct_display(df: pd.DataFrame) -> pd.DataFrame:
    """交付 CSV 百分比显示层 (2026-09-05 用户: "输出的EXCEL是百分比"): 概率/幅度/
    回撤列 ×100 加 %, pctChg 本就是百分数值只加 %; NaN → 空. 纯显示 — 入参
    DataFrame 不改, density_picks 上游保持数值供机器读. 实现共享于 _pctfmt."""
    return fmt_pct_columns(
        df,
        (
            "legacy_prob",
            "legacy_pred10",
            "parallel_prob",
            "parallel_pred10",
            "pull",
            "belief_down",
            "chip_wr5",
            "pctChg",
        ),
        already_pct_cols=("pctChg",),
    )


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen_only = "--gen-only" in sys.argv
    dry_run = "--dry-run" in sys.argv

    if args:
        day_ts = pd.Timestamp(args[0])
    else:
        fs = sorted(glob.glob(os.path.join(DATA_DIR, "lists", "candidates_*.parquet")))
        if not fs:
            print("[prob10dens] 无 candidates 文件, 跳过 (fail-safe)")
            return 0
        day_ts = pd.Timestamp(os.path.basename(fs[-1])[11:19])
    date = day_ts.strftime("%Y%m%d")

    cand_fp = os.path.join(DATA_DIR, "lists", f"candidates_{date}.parquet")
    if not os.path.exists(cand_fp):
        print(f"[prob10dens] 无当日 {os.path.basename(cand_fp)}, 跳过 (fail-safe)")
        return 0
    cand = pd.read_parquet(
        cand_fp, columns=["symbol", "board", "prob_up_10d", "pred_ret_10d"]
    )
    if cand.empty:
        print(f"[prob10dens] {date} candidates 空, 跳过")
        return 0

    close = pd.read_parquet(
        PANEL_V3_PATH,
        columns=["symbol", "date", "close_hfq", "amount"],
        filters=[
            ("date", ">=", day_ts - pd.Timedelta(days=45)),
            ("date", "<=", day_ts),
        ],
    )
    close["symbol"] = close["symbol"].astype(str).str.zfill(6)
    close = close[~close["symbol"].str.endswith(".BJ")]
    cl = close.pivot(index="date", columns="symbol", values="close_hfq")
    am = close.pivot(index="date", columns="symbol", values="amount")
    day_px = pd.read_parquet(
        PANEL_V3_PATH, columns=["symbol", "pctChg"], filters=[("date", "=", day_ts)]
    )
    day_px["symbol"] = day_px["symbol"].astype(str).str.zfill(6)

    hist = load_or_bootstrap_history(day_ts)
    par_fps = sorted(
        glob.glob(os.path.join(STOCK_LIST_DIR, f"parallel_preds_raw_{date}__*.csv"))
    )
    par = None
    if par_fps:
        par = pd.read_csv(
            par_fps[-1],
            dtype={"symbol": str},
            usecols=["symbol", "pred_mag_10d", "pred_prob_10d"],
        )
    else:
        print(
            f"[prob10dens] 无 parallel_preds_raw_{date}__*.csv, parallel 两列空 (NaN)"
        )
    chip = load_chip_features(day_ts)
    if chip is None:
        print("[prob10dens] 筹码数据缺失, 派发标注未启用 (fail-open)")
    picks = density_picks(cand, hist, cl, am, day_ts, par=par, chip=chip)
    if chip is not None and len(picks):
        flagged = picks.loc[picks["chip_flag"] == "派发", "symbol"].tolist()
        if flagged:
            print(f"[prob10dens] 派发标注 {len(flagged)} 只: {', '.join(flagged)}")
    if picks.empty:
        print(f"[prob10dens] {date} 密度/回撤闸后无票, 跳过 (fail-safe)")
        return 0
    save_history(hist, prob10_membership(cand, day_ts))

    picks = picks.merge(day_px[["symbol", "pctChg"]], on="symbol", how="left")

    csv_path = STOCK_LIST_DIR / f"prob10dens_{date}__{MODULE}.csv"
    fmt_pct_display(picks).to_csv(csv_path, index=False, encoding="utf-8-sig")
    # 死区停推闸 (2026-09-05 用户拍板 "那就一起停吧"): 报警夜清单照出不推;
    # txt 不写 (_ths_flush_guard 按 ths_watchlist_*__*.txt glob, 不写即不误动);
    # gen-only/dry-run 为人工演练不拦 (fail-open 在闸内; 密度史 09-03 起冷启动
    # 攒样本期闸自动 fail-open)
    alarm, why = _deadzone_guard.is_alarm("prob10dens", date)
    if alarm:
        print(f"[deadzone] 死区报警 (密度影子单): {why}")
        if not gen_only and not dry_run:
            _deadzone_guard.annotate_stop("prob10dens", date, why)
            from scripts._ths_watchlist_push import write_push_result

            write_push_result(
                STOCK_LIST_DIR / f"ths_watchlist_{date}__{MODULE}.txt",
                picks["symbol"].tolist(),
                [],
                note="deadzone",
            )
            print(f"[deadzone] 今晚停推: 清单照出 {csv_path.name} (不写 txt 不加自选)")
            print(
                "[deadzone] 已标注: STOPPED_DEADZONE 标记"
                " + 推送结果单 status=deadzone (与没推成功区分)"
            )
            print(picks.to_string(index=False))
            return 0
    # [2026-09-10 用户令"不需要推了"] 一次性推送总闸: 旗标存在当晚全线不推 (清单照出,
    # 不写 txt 不加自选); 次日删旗标即恢复. 一次性 override 勿 commit.
    _no_push_flag = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "tmp_t",
        "_no_push_20260910.flag",
    )
    if os.path.exists(_no_push_flag):
        print(
            f"[no-push] 推送总闸关闭 ({os.path.basename(_no_push_flag)}):"
            " 清单照出, 不写 txt 不加自选"
        )
        from scripts._ths_watchlist_push import write_push_result

        write_push_result(
            STOCK_LIST_DIR / f"ths_watchlist_{date}__{MODULE}.txt",
            picks["symbol"].tolist(),
            [],
            note="no_push_user_hold",
        )
        print(picks.to_string(index=False))
        return 0
    txt_path = STOCK_LIST_DIR / f"ths_watchlist_{date}__{MODULE}.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(picks["symbol"]) + "\n")
    print(f"[prob10dens] {csv_path}")
    print(f"[prob10dens] {txt_path} ({len(picks)} 只)")
    print(picks.to_string(index=False))

    if not gen_only:
        from scripts._ths_ui import THS_HEXIN_PATH
        from scripts._ths_watchlist_push import push_via_ths

        if not THS_HEXIN_PATH.exists():
            print(f"[warn] 同花顺客户端不存在: {THS_HEXIN_PATH}")
            return 0
        if not push_via_ths(txt_path, dry_run):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
