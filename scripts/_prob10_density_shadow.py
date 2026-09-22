"""概率头密度版影子单: prob前20带+带内密度≥3+回撤闸+筹码水位标注 → 同花顺自选股
(2026-09-06 用户拍板把原 TOP10+额1亿 口径整线替换为 L3×TOP20免额; 线名/文件/
夜链位置/死区线名不变).

口径 (09-06 拍板; 125d 回放 tmp_t/_band_compare_0906.py 2026-02-09..08-17:
  main 5.5只/日 赢率54.7%/+7.82pp 大亏3.5%, dual 6.0只/日 47.7%/+7.62pp 大亏5.1%
  — 随机基准 = 全市场净≥5% 赢率 13.5%, 两板 ≈3.5~4 倍随机):
  ①带成员 = 每板 (main; dual=GEM+STAR) 按 legacy 概率头 prob_up_10d 降序 前20
    (原 TOP10 榜 → TOP20 带; 密度累计宇宙同步换成带, 11-20 名滞留也攒天数)
  ②回撤 [0913 撤删改标]: 原硬闸 pull ≥ -10% 已撤 (PULL_FLOOR=-1.0 等效关闭);
    pull < -0.10 (PULL_FLAG_MAX) 标 pull_flag=回撤 列标注不删票 (同 ⑥ 派发标注模式)
  ③密度 = occ5≥3, occ5 对应研究带内 OCC5=rolling(5) 含当日: 今日在带 +
    近4个**交易日**在带数 (0915 修: 原先按"历史文件里最后 4 个日期"取窗,
    重跑同一日会前移窗口+重复计数当日 → 结果不稳定; 改用交易日历)
  ④免额 (09-06 拍板 "去额"; 2×2 终审: 额闸在 prob 池头部近似装饰 — 撤之
    +0.1~0.2只/日流量, 赢率代价 1~1.7pp); amt 列保留仅展示, 不作闸
  ⑤撞指数码 000xxx 不剔 (09-05 用户澄清 "不是删除股票号"), 推送端隔离指数行
    — 见 _ths_watchlist_push._build_chunks
  ⑥筹码水位标注 (09-05 三线统一删 → 09-09 用户推翻改标注 → 0922 改获利盘水位轴):
    获利盘水位 chip_wr<0.5 → chip_flag="低获利，涨" / ≥0.5 → "高获利，跌" (0922
    用户令定值; 水位是唯一有信息的筹码轴, 原 wr5<0 派发标注判死, 数值列 chip_wr5
    同日随用户令从交付清单退役), 不删票; cyq 数据缺/个股水位缺 → 不标 (fail-open)。
    同标注接 LEGACY 交付 (_deliver_legacy_list)、PARALLEL 短名单 (_shortlist_t5_t10)
    与 GENIOUS 冠军表 (_genious_excel, 走同一 apply_chip_gate 四线同源)。
  ⑦趋势闸 [0913 用户令 "DENSITY 接闸后票" → 0914 用户拍板闸位选 B]: MA10↑ 闸
    在终选 — 带史/occ5 按原始 TOP20 带 (跌票也记史攒 occ5), occ5≥3 后终选滤
    当日 MA10↑ (刚拐头票当天即可出, B 独有 301220 案例; 数据面 A 带前闸微胜,
    diag/density_gate_pos_ab_0913_*.json); cls 头原始排序偏超卖 (跌票高概率),
    LEGACY 线已闸 (LEGACY_SELECTION.trend_gate, 同规则同扫描依据
    diag/trend_gate_sweep_0913_*.json); 面板行不足 → 闸跳过 (fail-open)。
  标签列 信念降 belief_down = 今日 prob − 3个上榜历日前 prob (非闸; 09-06 L4
  对照 = 半流量换 +1.6pp 判不接, 列保留供影子期攒证据)。
  双模型列 (2026-09-05 用户): legacy_prob/legacy_pred10 = legacy 概率头
  prob_up_10d / 幅度头 pred_ret_10d (选股口径即 legacy 概率头); parallel_prob/
  parallel_pred10 = parallel raw pred_prob_10d / pred_mag_10d
  (parallel_preds_raw_{date}__*.csv 全池落盘; raw 未校准), 缺文件 → NaN。
  交付 CSV 百分比显示 (2026-09-05 用户): 上述预测列+回撤列写 "%" 文本,
  pctChg 只加 %; 见 fmt_pct_display, 纯显示层不影响机器读。

上榜历史: data/prob10_density_history.parquet (date/board/symbol/prob)。
  **date 必须是交易日** (0915 修): 该键直接与 occ5 的 win4 (取自面板交易日历) 对表,
  记成非交易日 = 永久取不到的死行。会话日一律取 candidates 内的 date 列, 不信文件名
  —— daily_pipeline 补跑时会拿墙钟日命名 (实见 20260823/20260830/20260913 三个
  "周日名/周五数据"文件), 照文件名记会把真周五的带全部记死。见 main() 内注释。
  引导: data/_diag_rankkey_scored_{board}_e125.parquet (125d 连续全池打分, 与
  研究同源) 补 candidates 未覆盖日期, data/lists/candidates_*.parquet 补其后的
  日期 (旧 vintage 缺 prob_up_10d 列跳过); 每夜追当日成员, 重跑同日先删后追
  (幂等)。2026-09-06 口径替换时已按 TOP20 带整文件重建 (e125+candidates)。
  注: 打分文件止于 08-17, 08-18..08-29 无 candidates 文件为空洞 (0915 修后不再把
  窗口往回伸, 空洞就是空洞 — 该日贡献 0), candidates 积累 ≥5 日后自然收敛到纯
  交易日历窗。

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
PULL_FLOOR = -1.0  # [0913 用户令撤回撤闸] -1.0 等效关闭; 原档 -0.10, 恢复改回
PULL_FLAG_MAX = -0.10  # [0913 撤删改标] 原闸档降为标注线: pull 低于此值标"回撤"不删
OCC_WIN = 5  # 密度窗: 近 5 个上榜日
OCC_MIN = 3  # 密度阈: 带内在榜 ≥3 天 (免额, 09-06 拍板)
HIST_PATH = os.path.join(DATA_DIR, "prob10_density_history.parquet")
CHIP_WR_LEVEL_SPLIT = (
    0.5  # 水位切分: 获利盘过半=深获利 (0922; 清单票分布中位0.534, 切分天然均衡)
)
CHIP_FLAG_LOW = "低获利，涨"  # 水位 < 0.5: 浅获利, 0922 回测前向更强
CHIP_FLAG_HIGH = "高获利，跌"  # 水位 >= 0.5: 深获利, 更弱
CYQ_PATH = os.path.join(DATA_DIR, "cyq_panel.parquet")
TREND_MA10_GATE = True  # [0914 用户拍板闸位 B] 终选闸: occ5 后滤当日 MA10↑; False 关


def trend_rising(close: pd.DataFrame) -> pd.Series | None:
    """MA10↑ 判定 (纯函数): 末行 MA10 > 前一行 → True; 行不足 → None (fail-open).

    与 LEGACY _select_prob10_pull 趋势闸同公式 (rolling(10, min_periods=1));
    MA10 vs 益盟长线定裁见 diag/ym_vs_ma10_faceoff_0913_*.json (MA10 全指标胜)."""
    if len(close.index) < 2:
        return None
    ma10 = close.rolling(10, min_periods=1).mean()
    return ma10.iloc[-1] > ma10.iloc[-2]


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
    "pull_flag",
    "amt",
    "belief_down",
    "chip_wr",
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
    """筹码特征 wr (入选日收盘可知, 无前视); 数据缺 → None (fail-open).

    wr = 当日获利盘水位 (0922 清单回测: 唯一有信息的筹码轴 — 低=浅获利前向更强,
    高=深获利更弱; Δ族 wr5 判死, 数值列同日随用户令从交付清单退役, 见
    tmp_t/_0922_chip_levels_list_test)。
    T = cyq 最新一日 ≤ day_ts (cyq 止于 T-1 时特征滞后一日, 方向不变); 个股缺行
    → NaN (标注恒空)。文件缺失/空/不足 6 行 → None, 水位标注整体不启用 (6 行
    下限是 wr5 时代的历史口径, 保留使标注触发面零变化)。
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
    return pd.DataFrame({"symbol": wr.columns.astype(str), "wr": wr.iloc[-1].values})


def chip_level_label(wr) -> np.ndarray:
    """获利盘水位 → 方向标注 (0922 用户令 "SET VALUE TO 低获利，涨 & 高获利，跌")。

    <0.5 低获利，涨 (浅获利, 回测前向更强) / ≥0.5 高获利，跌 (深获利, 更弱) /
    NaN 空 (无筹码数据)。四线交付 (密度/LEGACY/PARALLEL/GENIOUS) 同源共用,
    勿在任一线另写切分。
    """
    w = pd.Series(wr).astype(float).reset_index(drop=True)
    return np.where(
        w.isna(),
        "",
        np.where(w < CHIP_WR_LEVEL_SPLIT, CHIP_FLAG_LOW, CHIP_FLAG_HIGH),
    )


def apply_wr5_gate(
    df: pd.DataFrame, chip: pd.DataFrame | None
) -> tuple[pd.DataFrame, list[str]]:
    """筹码水位标注 (0922 用户令): chip_wr<0.5 → "低获利，涨" / ≥0.5 → "高获利，跌".

    沿革: 09-05 三线统一删 (wr5<0 删除闸) → 09-09 用户推翻改标注 "派发不删,
    清单标注" (wr5<0 → 派发) → 0922 清单回测 wr5 判死、水位轴 (chip_wr) 是唯一
    有信息的筹码族 → 用户令改水位标注。加列 chip_wr (水位) + chip_flag
    (wr5 数值列同日随用户令从交付清单退役); 不删任何行。
    chip None/空 或 df 空 → 原样返回 (fail-open); 个股 wr 缺 (旧 fixture 无 wr
    列 / cyq 缺行) → chip_wr NaN → flag 空。
    返回 (标注后 df 副本, 有标注的 symbol 列表)。
    生产接线: 密度影子单 (density_picks) / LEGACY 交付 (_deliver_legacy_list) /
    PARALLEL 短名单 (_shortlist_t5_t10) / GENIOUS 冠军表 (_genious_excel, 经
    apply_chip_gate 同入口)。
    """
    if chip is None or not len(chip) or df.empty:
        return df, []
    ch = chip.drop_duplicates("symbol", keep="last").set_index("symbol")
    d = df.copy()
    sym = d["symbol"].astype(str).str.zfill(6)
    lvl = sym.map(ch["wr"]) if "wr" in ch.columns else np.nan
    d["chip_wr"] = lvl
    d["chip_flag"] = chip_level_label(d["chip_wr"])
    return d, sorted(sym[d["chip_flag"] != ""].unique())


def apply_chip_gate(
    df: pd.DataFrame, day_ts: pd.Timestamp, flush: bool = False
) -> pd.DataFrame:
    """筹码水位标注接线入口 (三线共享): load_chip_features → apply_wr5_gate → 标注日志.

    cyq 数据缺 → 原样返回 (fail-open)。LEGACY 交付 (_deliver_legacy_list) 与
    PARALLEL 短名单 (_shortlist_t5_t10) 调用; 密度影子单走 density_picks 内联
    (main 里另有标注日志, 不走此处避免重复打印)。
    """
    chip = load_chip_features(day_ts)
    if chip is None:
        print("[chipgate] 筹码数据缺失, 水位标注未启用 (fail-open)", flush=flush)
        return df
    out, _ = apply_wr5_gate(df, chip)
    if "chip_flag" in out.columns and (out["chip_flag"] != "").any():
        n_low = int((out["chip_flag"] == CHIP_FLAG_LOW).sum())
        n_high = int((out["chip_flag"] == CHIP_FLAG_HIGH).sum())
        print(
            f"[chipgate] 筹码水位标注: {CHIP_FLAG_LOW} {n_low} 只 / "
            f"{CHIP_FLAG_HIGH} {n_high} 只",
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
    hist: 带上榜历史 (date/board/symbol/prob); 含不含当日均可 (occ5 窗按交易日历取
          day_ts 之前 4 日, 当日在窗内被排除, 故重跑幂等 — 0915 修)
    close/amount: 透视表 (date × symbol), ≤ day_ts; amount 仅算 amt 展示列
    par: parallel 全池 raw 预测 (symbol/pred_mag_10d/pred_prob_10d);
         None/缺 → parallel 两列 NaN
    chip: 筹码特征 (symbol/wr, load_chip_features 产出); None →
          不加水位标注列; 个股水位 NaN → 不标 (fail-open)
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

    # occ5 研究口径 OCC5=rolling(5) 含当日: 1(今日在带) + 近4个**交易日**在带数;
    # belief_down 对应研究 PM3=shift(3): 今日 prob − 3个**交易日**前 prob (未在带=NaN)
    # [0915 修] 窗口按交易日历 (cl.index) 取 day_ts 之前最后 4 个交易日, **不**取
    # "历史文件里最后 4 个日期"。后者有两个坑:
    #   ① save_history 每次运行都追加当日 → 重跑同一日时 hist 已含当日, 该窗口整体
    #      前移一格且当日被算进"近4日"(与 +1 今日在带 重复计数) → 同日两次跑出不同
    #      清单 (09-14 实测: 首次 3 只 dual, 重跑 1 只 main, 两者互斥);
    #   ② 先补较早日期时窗口里会含未来日期 → look-ahead (09-13 补跑窗口含 09-14)。
    # 改按交易日历取窗后: 重跑幂等 (窗口与 hist 内容无关), 未来日期不可能入窗, 且
    # 漏跑的夜 = 空洞 (该日无史 → 贡献 0), 不再把窗口静默往回伸一个交易日。
    tdays = [d for d in cl.index if d < day_ts]
    win4 = set(tdays[-(OCC_WIN - 1) :])
    d3 = tdays[-3] if len(tdays) >= 3 else None
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
    if "chip_wr" not in ok.columns:  # chip 缺 (fail-open) 也保稳定 schema
        ok["chip_wr"] = np.nan
        ok["chip_flag"] = ""
    ok["pull_flag"] = np.where(  # [0913 撤删改标] 原回撤闸降为标注 (同 chip_flag 模式)
        ok["pull"].fillna(-1) < PULL_FLAG_MAX, "回撤", ""
    )
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
            c = pd.read_parquet(fp, columns=["symbol", "board", "prob_up_10d", "date"])
            # [0915 修] 同 main(): 会话日以数据内 date 列为准 (墙钟日命名陷阱)
            u = sorted(pd.to_datetime(c["date"].unique()))
            if len(u):
                d = pd.Timestamp(u[0])
        except ValueError:
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
            "chip_wr",
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
    try:
        cand = pd.read_parquet(
            cand_fp,
            columns=["symbol", "board", "prob_up_10d", "pred_ret_10d", "date"],
        )
    except ValueError:  # 老 vintage 无 date 列
        cand = pd.read_parquet(
            cand_fp, columns=["symbol", "board", "prob_up_10d", "pred_ret_10d"]
        )
    if cand.empty:
        print(f"[prob10dens] {date} candidates 空, 跳过")
        return 0
    # [0915 修] 会话日以**数据内 date 列**为准, 不以文件名/argv 为准。
    # 病根: 补跑时 daily_pipeline 拿墙钟日命名 candidates_{D}.parquet, 数据却是最近一个
    # 交易日 —— 实见 20260823→08-21, 20260830→08-28, 20260913→09-11 (清一色"周日名/
    # 周五数据")。密度史原先按文件名记 → 真 09-11 的带被记到周日 09-13 名下, 而 occ5
    # 的 win4 取自面板交易日历 (不含周日) → 该行**永远取不到**, 同时 09-11 那一格永远
    # 贡献 0, occ5 恒够不到 3 → 清单自 09-10 起永久空。见 tmp_t/_density_occ_diag_0915.py
    # 的逐日漏斗 (09-14 窗 09-08/09/10/11, 09-15 窗 09-09/10/11/14 各缺一格)。
    if "date" in cand.columns:
        du = sorted(pd.to_datetime(cand["date"].unique()))
        if len(du) != 1:
            print(
                f"[prob10dens] candidates 含多日 {[str(x.date()) for x in du]}, 取最早"
            )
        sess = pd.Timestamp(du[0])
        if sess != day_ts:
            print(
                f"[prob10dens] 会话日对齐: 文件标 {day_ts.date()} 但数据为 {sess.date()}"
                f" → 以数据为准 (补跑墙钟日陷阱)"
            )
            day_ts = sess
            date = sess.strftime("%Y%m%d")

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
        print("[prob10dens] 筹码数据缺失, 水位标注未启用 (fail-open)")
    picks = density_picks(cand, hist, cl, am, day_ts, par=par, chip=chip)
    if chip is not None and len(picks) and "chip_flag" in picks.columns:
        n_low = int((picks["chip_flag"] == CHIP_FLAG_LOW).sum())
        n_high = int((picks["chip_flag"] == CHIP_FLAG_HIGH).sum())
        print(
            f"[prob10dens] 筹码水位标注: {CHIP_FLAG_LOW} {n_low} 只 / {CHIP_FLAG_HIGH} {n_high} 只"
        )
    # [0914 用户拍板闸位 B] ⑦趋势闸在终选: 带史/occ5 用原始带, occ5≥3 后滤当日
    # MA10↑ — 刚拐头票当天即可出 (B 独有 301220); A 带前闸判词
    # diag/density_gate_pos_ab_0913_*.json.
    if TREND_MA10_GATE and len(picks):
        rising = trend_rising(cl)
        if rising is None:
            print("[prob10dens] 面板行不足, MA10 趋势闸跳过 (fail-open)")
        else:
            n0 = len(picks)
            picks = picks[picks["symbol"].map(rising).fillna(True)]
            print(f"[prob10dens] MA10↑ 趋势闸 (终选): {n0} → {len(picks)}")
    # [0913] 上榜史先于空判落盘: 空夜不记史 → occ5 窗冻结在旧模型带 → 换模
    # (0913 main 首切 cls) 后新带 occ5 恒 0 → 永久空清单死锁. 榜 = 原始带成员
    # (⑦终选闸口径 — 闸在终选不动带史; 闸位 A/B 判词见
    # diag/density_gate_pos_ab_0913_*.json), 空夜也记, 窗口才能滚动.
    save_history(hist, prob10_membership(cand, day_ts))
    if picks.empty:
        print(f"[prob10dens] {date} 密度/回撤闸后无票, 跳过 (fail-safe)")
        return 0

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
