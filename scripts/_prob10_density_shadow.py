"""概率头密度版影子单: prob10+回撤闸+近5日上榜≥3天+额1亿 → 同花顺自选股 (2026-09-05 用户拍板).

口径 (125d ckpt 回放 2026-02-09..08-17, tmp_t/_winner_streak2_0905.py, 同窗对生产
  全面占优 — 总表见 memory/prob-head-verdict-0904.md):
  ①prob10 = 每板 (main; dual=GEM+STAR) 按 legacy 概率头 prob_up_10d 降序 top10
  ②回撤闸 = 收盘距 10 日高点回撤 ≥ -10% (信号夜可知)
  ③密度 = occ5≥3, occ5 对应研究 OCC5=rolling(5) 含当日: 今日在榜 + 近4个上榜历日
    在榜数 ≥2 (6.8只/日 56.5%/+10.09pp 大亏3.4%, 优于连续版 streak≥2 7.4只/日
    54.6%/+9.44)
  ④额 ≥1亿 (streak/密度赢家画像前置条件, 非额 streak 19.0%)
  ⑤撞指数码 000xxx 不剔 (09-05 用户澄清 "不是删除股票号"), 推送端隔离指数行
    — 见 _ths_watchlist_push._build_chunks
  ⑥筹码派发闸 (2026-09-05 用户拍板 "基本方向是派发就删除"; 09-05 晚升级三线统一
    "只要派发都删"): 获利盘5日回落 (wr5<0) → 剔除, 不补齐; 125d 1151 票次回放
    升级后保留组 6.6只/日 赢率56.1% 大亏2.8% (基线 3.6%, tmp_t/_densgate_upgrade_0905.py);
    cyq 数据缺/个股特征缺 → 不拦 (fail-open)。同闸接 LEGACY 交付
    (_deliver_legacy_list) 与 PARALLEL 短名单 (_shortlist_t5_t10)。
  标签列 信念降 belief_down = 今日 prob − 3个上榜历日前 prob < 0 (质量档闸, 2~3个月
  影子证据够了再决定是否升格; h2 大亏 4.8% vs 密度版 10.9%, memory 09-05 终表)。
  双模型列 (2026-09-05 用户: "影子名单上两模型预测幅度和概率, 列出是什么模型出"):
  legacy_prob/legacy_pred10 = legacy 概率头 prob_up_10d / 幅度头 pred_ret_10d
  (选股口径即 legacy prob10); parallel_prob/parallel_pred10 = parallel raw
  pred_prob_10d / pred_mag_10d (parallel_preds_raw_{date}__*.csv 全池落盘; raw
  未校准 — 校准版 pred_ret_10d 仅短名单内, 全池逐股校准未复刻), 缺文件 → NaN。
  交付 CSV 百分比显示 (2026-09-05 用户 "输出的EXCEL是百分比"): 上述预测列+回撤
  列写 "%" 文本, pctChg 只加 %; 见 fmt_pct_display, 纯显示层不影响机器读。

上榜历史: data/prob10_density_history.parquet (date/board/symbol/prob)。
  首跑自动引导: data/_diag_rankkey_scored_{board}_e125.parquet (125d 连续全池打分,
  与研究同源) 补 candidates 未覆盖日期, data/lists/candidates_*.parquet 补其后的
  日期 (旧 vintage 缺 prob_up_10d 列跳过); 每夜追当日成员, 重跑同日先删后追 (幂等)。
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
TOP_N = 10          # 每板 top10 (prob10 口径)
PULL_FLOOR = -0.10  # 回撤闸: 距10日高点回撤下限
AMT_MIN = 1e8       # 成交额下限 (元)
OCC_WIN = 5         # 密度窗: 近 5 个上榜日
OCC_MIN = 3         # 密度阈: 在榜 ≥3 天
HIST_PATH = os.path.join(DATA_DIR, "prob10_density_history.parquet")
CHIP_WR5_MAX = 0.0  # 派发闸: 获利盘5日变化须低于此值 (负=回落; 09-05 三线统一 wr5<0 即剔)
CYQ_PATH = os.path.join(DATA_DIR, "cyq_panel.parquet")

_COLS = ["rank", "board", "symbol", "legacy_prob", "legacy_pred10",
         "parallel_prob", "parallel_pred10", "occ5", "pull",
         "amt", "belief_down"]


def _board_of(b: str) -> str:
    return "main" if b == "main" else "dual"  # GEM/STAR → dual


def _membership_core(c: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """按 date/board 分组取 prob 降序 top10 → date/board/symbol/prob (纯函数)."""
    c = c.copy()
    c["symbol"] = c["symbol"].astype(str).str.zfill(6)
    c["board"] = c["board"].map(_board_of)
    c["prob"] = c[prob_col].astype(float)
    c = c.sort_values(["board", "date", "prob", "symbol"],
                      ascending=[True, True, False, True])
    return c.groupby(["board", "date"], sort=False).head(TOP_N)[
        ["date", "board", "symbol", "prob"]].reset_index(drop=True)


def prob10_membership(cand: pd.DataFrame, day_ts: pd.Timestamp) -> pd.DataFrame:
    """candidates 当日截面 → prob10 成员 (date/board/symbol/prob, 纯函数).

    board 映射 main→main, GEM/STAR→dual; 每板按 prob 降序 (并列 symbol 升序)
    取 top10; 北交所不在 candidates 无需剔。
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
        CYQ_PATH, columns=["symbol", "date", "winner_ratio"],
        filters=[("date", ">=", day_ts - pd.Timedelta(days=21)),
                 ("date", "<=", day_ts)])
    if cq.empty:
        return None
    cq["symbol"] = cq["symbol"].astype(str).str.zfill(6)
    cq = cq.drop_duplicates(["symbol", "date"], keep="last")
    wr = cq.pivot(index="date", columns="symbol",
                  values="winner_ratio").sort_index()
    if len(wr.index) < 6:
        return None
    i = len(wr.index) - 1
    return pd.DataFrame({"symbol": wr.columns.astype(str),
                         "wr5": wr.iloc[i].values - wr.iloc[i - 5].values})


def apply_wr5_gate(df: pd.DataFrame,
                   chip: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """派发闸通用过滤 (2026-09-05 用户拍板三线统一 "只要派发都删"): wr5<0 → 剔除.

    chip None/空 或 df 空 → 原样返回; 个股 wr5 NaN → 比较恒 False → 保留
    (fail-open)。返回 (过滤后 df, 被剔 symbol 列表); 不补齐 — 生产 TOP10 回放
    补齐被不补全面压制 (tmp_t/_top10_chipdir_replay_0905.py)。
    生产接线: 密度影子单 (density_picks) / LEGACY 交付 (_deliver_legacy_list) /
    PARALLEL 短名单 (_shortlist_t5_t10)。
    """
    if chip is None or not len(chip) or df.empty:
        return df, []
    ch = chip.drop_duplicates("symbol", keep="last").set_index("symbol")
    sym = df["symbol"].astype(str).str.zfill(6)
    mask = sym.map(ch["wr5"]) < CHIP_WR5_MAX  # NaN < x → False → 保留
    if not mask.any():
        return df, []
    cut = sorted(sym[mask].unique())
    return df[~mask.to_numpy()].copy(), cut


def apply_chip_gate(df: pd.DataFrame, day_ts: pd.Timestamp,
                    flush: bool = False) -> pd.DataFrame:
    """派发闸接线入口 (三线共享): load_chip_features → apply_wr5_gate → 剔除日志.

    cyq 数据缺 → 原样返回 (fail-open)。LEGACY 交付 (_deliver_legacy_list) 与
    PARALLEL 短名单 (_shortlist_t5_t10) 调用; 密度影子单走 density_picks 内联
    (main 里另有双跑剔除日志, 不走此处避免重复打印)。
    """
    chip = load_chip_features(day_ts)
    if chip is None:
        print("[chipgate] 筹码数据缺失, 派发闸未启用 (fail-open)", flush=flush)
        return df
    out, cut = apply_wr5_gate(df, chip)
    if cut:
        print(f"[chipgate] 派发闸剔除 {len(cut)} 只 (获利盘5日回落): "
              f"{', '.join(cut)}", flush=flush)
    return out


def density_picks(cand: pd.DataFrame, hist: pd.DataFrame,
                  close: pd.DataFrame, amount: pd.DataFrame,
                  day_ts: pd.Timestamp,
                  par: pd.DataFrame | None = None,
                  chip: pd.DataFrame | None = None) -> pd.DataFrame:
    """prob10+回撤闸+密度+额+派发方向 → 当日影子清单 (纯函数, 可单测).

    cand: 当日 candidates 截面 (symbol/board/prob_up_10d/pred_ret_10d)
    hist: 上榜历史 (date/board/symbol/prob), 须不含当日 (当日成员由 cand 现算)
    close/amount: 透视表 (date × symbol), ≤ day_ts
    par: parallel 全池 raw 预测 (symbol/pred_mag_10d/pred_prob_10d);
         None/缺 → parallel 两列 NaN
    chip: 筹码派发特征 (symbol/wr5, load_chip_features 产出); None →
          派发闸不启用; 个股特征 NaN → 不拦 (fail-open)
    """
    memb = prob10_membership(cand, day_ts)
    c = cand.copy()
    c["symbol"] = c["symbol"].astype(str).str.zfill(6)
    c["board"] = c["board"].map(_board_of)
    m = memb.merge(c[["symbol", "board", "pred_ret_10d"]].rename(
        columns={"pred_ret_10d": "pred10"}), on=["symbol", "board"], how="left")

    cl = close[close.index <= day_ts].sort_index()
    am = amount.reindex(cl.index)
    pull = (cl / cl.rolling(10, min_periods=2).max() - 1).iloc[-1]
    amt = am.iloc[-1]
    m["pull"] = m["symbol"].map(pull)
    m["amt"] = m["symbol"].map(amt)

    # occ5 研究口径 OCC5=rolling(5) 含当日: 1(今日在榜) + 近4个上榜历日在榜数;
    # belief_down 对应研究 PM3=shift(3): 今日 prob − 3个上榜历日前 prob (未在榜=NaN)
    hdates = sorted(hist["date"].unique())
    win4 = set(hdates[-(OCC_WIN - 1):]) if hdates else set()
    d3 = hdates[-3] if len(hdates) >= 3 else None
    occ, p3v = [], []
    for r in memb.itertuples():
        h = hist[(hist["board"] == r.board) & (hist["symbol"] == r.symbol)]
        ds = set(h["date"].unique())
        occ.append(1 + len(ds & win4))
        p3 = h[h["date"] == d3]["prob"] if d3 is not None else None
        p3v.append(float(p3.iloc[0]) if p3 is not None and len(p3) else np.nan)
    m["occ5"] = occ
    m["belief_down"] = [np.nan if np.isnan(v) else r.prob - v
                        for v, r in zip(p3v, memb.itertuples())]

    ok = m[(m["pull"].fillna(-1) >= PULL_FLOOR) & (m["amt"] >= AMT_MIN)
           & (m["occ5"] >= OCC_MIN)].copy()
    ok, _ = apply_wr5_gate(ok, chip)
    ok = ok.rename(columns={"prob": "legacy_prob", "pred10": "legacy_pred10"})
    if par is not None and len(par):
        p = par[["symbol", "pred_prob_10d", "pred_mag_10d"]].copy()
        p["symbol"] = p["symbol"].astype(str).str.zfill(6)
        ok = ok.merge(p.drop_duplicates("symbol", keep="last").rename(
            columns={"pred_prob_10d": "parallel_prob",
                     "pred_mag_10d": "parallel_pred10"}),
            on="symbol", how="left")
    else:
        ok["parallel_prob"] = np.nan
        ok["parallel_pred10"] = np.nan
    ok["_b"] = (ok["board"] != "main").astype(int)  # 交付顺序 main 在前
    ok = ok.sort_values(["_b", "legacy_prob", "symbol"],
                        ascending=[True, False, True])
    ok = ok.drop(columns="_b").reset_index(drop=True)
    ok.insert(0, "rank", np.arange(1, len(ok) + 1))
    return ok[_COLS]


def load_or_bootstrap_history(day_ts: pd.Timestamp) -> pd.DataFrame:
    """上榜历史: 有文件读文件; 无则引导 (打分文件补连续史 + candidates 补尾部)."""
    if os.path.exists(HIST_PATH):
        return pd.read_parquet(HIST_PATH)
    frames, cand_dates = [], set()
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "lists",
                                            "candidates_*.parquet"))):
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
    h = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "board", "symbol", "prob"])
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
        ("legacy_prob", "legacy_pred10", "parallel_prob", "parallel_pred10",
         "pull", "belief_down", "pctChg"),
        already_pct_cols=("pctChg",),
    )


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen_only = "--gen-only" in sys.argv
    dry_run = "--dry-run" in sys.argv

    if args:
        day_ts = pd.Timestamp(args[0])
    else:
        fs = sorted(glob.glob(os.path.join(DATA_DIR, "lists",
                                           "candidates_*.parquet")))
        if not fs:
            print("[prob10dens] 无 candidates 文件, 跳过 (fail-safe)")
            return 0
        day_ts = pd.Timestamp(os.path.basename(fs[-1])[11:19])
    date = day_ts.strftime("%Y%m%d")

    cand_fp = os.path.join(DATA_DIR, "lists", f"candidates_{date}.parquet")
    if not os.path.exists(cand_fp):
        print(f"[prob10dens] 无当日 {os.path.basename(cand_fp)}, 跳过 (fail-safe)")
        return 0
    cand = pd.read_parquet(cand_fp,
                           columns=["symbol", "board", "prob_up_10d",
                                    "pred_ret_10d"])
    if cand.empty:
        print(f"[prob10dens] {date} candidates 空, 跳过")
        return 0

    close = pd.read_parquet(
        PANEL_V3_PATH, columns=["symbol", "date", "close_hfq", "amount"],
        filters=[("date", ">=", day_ts - pd.Timedelta(days=45)),
                 ("date", "<=", day_ts)])
    close["symbol"] = close["symbol"].astype(str).str.zfill(6)
    close = close[~close["symbol"].str.endswith(".BJ")]
    cl = close.pivot(index="date", columns="symbol", values="close_hfq")
    am = close.pivot(index="date", columns="symbol", values="amount")
    day_px = pd.read_parquet(PANEL_V3_PATH, columns=["symbol", "pctChg"],
                             filters=[("date", "=", day_ts)])
    day_px["symbol"] = day_px["symbol"].astype(str).str.zfill(6)

    hist = load_or_bootstrap_history(day_ts)
    par_fps = sorted(glob.glob(os.path.join(
        STOCK_LIST_DIR, f"parallel_preds_raw_{date}__*.csv")))
    par = None
    if par_fps:
        par = pd.read_csv(par_fps[-1], dtype={"symbol": str},
                          usecols=["symbol", "pred_mag_10d", "pred_prob_10d"])
    else:
        print(f"[prob10dens] 无 parallel_preds_raw_{date}__*.csv, "
              "parallel 两列空 (NaN)")
    chip = load_chip_features(day_ts)
    if chip is None:
        print("[prob10dens] 筹码数据缺失, 派发闸未启用 (fail-open)")
    picks = density_picks(cand, hist, cl, am, day_ts, par=par, chip=chip)
    if chip is not None:
        base = density_picks(cand, hist, cl, am, day_ts, par=par)
        cut = sorted(set(base["symbol"]) - set(picks["symbol"]))
        if cut:
            print(f"[prob10dens] 派发闸剔除 {len(cut)} 只: {', '.join(cut)}")
    if picks.empty:
        print(f"[prob10dens] {date} 密度/派发闸后无票, 跳过 (fail-safe)")
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
                picks["symbol"].tolist(), [], note="deadzone")
            print(f"[deadzone] 今晚停推: 清单照出 {csv_path.name} (不写 txt 不加自选)")
            print("[deadzone] 已标注: STOPPED_DEADZONE 标记"
                  " + 推送结果单 status=deadzone (与没推成功区分)")
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
