"""GENIOUS — 链路狙击交付层 Excel (2026-09-14 用户令: 交易日 20:30 自动跑).

Sheet1 = 冠军表 (CH3 T3深跌缩量 / CH2 T2深跌 / CH2B T2稳健过闸)
0919 用户令: 观察池与全量表**删除** — 只出冠军单表。层序+层内 r60 深→浅
即排名。层定义与阈值见 app/pipeline1/kongduo_triggers.py 与 settings.GENIOUS。

口径: 名单 = **当日 (trade date) 起火的票**, 20:30 已收盘故"fire日收盘进"实际落到
T+1, 故每行带执行档 (温火/质量层=T+1开盘进; 涨停/深跌层=T+1仍涨确认→T+1收盘进)。
**扣 0.7% 往返费后火群整体≈0**, 钱在层头部 → 请按层序读, 勿无脑全买。

新鲜度: 面板由 AMINQT-MarketData-22h (19:15) 日更, 20:30 跑时当日行已在库; 本脚本
要求 lag==0, 不足则有限等待 (容忍抓取未完成) → 仍缺则自拉当日截面**只拼内存不写面板**。
拿不到当日数据一律 exit 2 不产文件 (交付昨天的名单比不交付更危险)。

输出: STOCK_LIST_DIR/GENIOUS_{date}.xlsx; 同日重跑 → GENIOUS_{date}__{HHMMSS}.xlsx
(WORM, 绝不覆盖)。日志 logs/genious_{tag}.log, 状态 logs/genious_{tag}.state.json。

用法:
    python scripts/_genious_excel.py                 # 当日 (非交易日跳过)
    python scripts/_genious_excel.py 20260911        # 指定日
    python scripts/_genious_excel.py --verify        # 全历史复现研究层表 (验收闸)
    python scripts/_genious_excel.py 20260911 --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.intraday.v51.safe_div import safe_divide  # noqa: E402
from app.pipeline1 import kongduo_triggers as kt  # noqa: E402
from app.pipeline1.freshness_guard import (  # noqa: E402
    expected_trading_date,
    lag_trading_days,
    load_trade_cal,
)
from config.settings import (  # noqa: E402
    GENIOUS,
    PANEL_V3_PATH,
    PROJECT_ROOT,
    STOCK_LIST_DIR,
)
from scripts._prob10_density_shadow import (  # noqa: E402
    CHIP_FLAG_HIGH,
    CHIP_FLAG_LOW,
    apply_chip_gate,
)
from scripts._stall_marker import stall_marker  # noqa: E402

LOG_DIR = Path(PROJECT_ROOT) / "logs"
DIAG_DIR = Path(PROJECT_ROOT) / "diag"
WAIT_TICK_S = 60
HEAL_TIMEOUT_S = 180  # 自拉硬超时; 超时=大声失败, 不留挂死实例 (见 _heal_rows_bounded)

BANNER1 = (
    "GENIOUS 冠军二段 — 段位全留, 【涨闸】列标出其中哪几只是 右侧拐头+缩量。"
    "OOS f10+5%止损口径: 过闸全取 胜率58%/期望+5.2%/大涨15.6% (低动量子集胜率76%); "
    "所以**涨闸只标注、不删票** — 过闸那几只是窄名单, 其余仍按层序读。"
    "【横盘提示】= 近10日未涨+冷静市 (0922 消融: 日闸真边, 横盘是条件性红利); "
    "【市场温度】<73% = 冷静市 (对模型有利), ≥73% 建议轻仓 (表尾 参与建议 列)。"
    "「当月样本口径」列 = 该层**当月实绩** (滚动重算, 月后补全); 扣0.7%往返费后火群整体≈0, 钱在层头部, "
    "请按层序自上而下读。执行档: 温火/质量层=T+1开盘进; 涨停/深跌层=T+1仍涨确认→T+1收盘进"
)

FB_BANNER = (
    "首板点名页 v2 — 当日全部首板(主板, 前10日无板)全量清单按【T+3板概率】降序 (0923 排序校准路); "
    "★冠军格 = 板前获利盘≥0.65 ∩ 一字板 (TE 5日续板率 68.2% / 次日封板 61.4%, 但 45% 次日再一字根本买不到)。"
    "⚠ 观察页勿当买入清单: 次日开盘买入口径 TE 笔均 −2.40% (Top-3), 冠军格可成交子集 −4.17% "
    "(开得出货让你买的恰是弱冠军 = 反向选择)。模型 = F18+B5 情绪生态 (promo=修正版真昨日晋级率, W22), "
    "TE: k3@1 39.8% / AUC_k3 0.572 / k2@1 45.8% / AUC次封 0.591。"
    "概率读法 (T+3/T+5 校准): 模型分为下界读数, 顶部实测率更高 — p_k3≥0.4 档 TE 实测 T+3 板率 ~54%。"
)

FB_LEGEND = (
    "列说明: 排名=按T+3板概率降序(全量清单, 超80行截断但★行豁免保留); ★=冠军格(板前获利盘≥0.65∩一字); 代码=6位裸代码",
    "T+3板概率 = P(D0+1..D0+3 内再涨停)(主排序键); T+5板概率 = P(D0+1..D0+5 内再涨停)",
    "校准档位 = 该行 T+3板概率分桶的 TE 实测板率: ≥0.4→~54% / 0.3-0.4→~28% / 0.2-0.3→~21% / <0.2→~18% (0923 rank_calib)",
    "一字 = 最低价贴涨停价(全天未开板) = 『买不到』提示非否决; 次日一字风险 = 一字∩T+3板概率≥0.4 → 次日大概率仍一字买不到 (仅提示)",
    "板前获利盘 wr1 = 昨日获利盘(D0 前信息); 板块涨停数 = 当日同板块(申万二级)涨停家数(含自身)",
    "昨日晋级率 = 昨日板中今日续板占比(≤D 信息, 修正版); 市场涨停数 = 主板当日涨停家数",
    "★ 读数=事件胜率非交易胜率(可成交子集笔均−4.2%) — 勿按胜率下单",
)

PB_BANNER = (
    "板前哨页(滚动命中日志) — 近20个交易日内命中过深睡签名(10~40日前放量脉冲∧其后无板守住90%∧获利盘升≥10pp"
    "∧横盘±6%∧5日缩量∧近10日无板)的股, 每股多行: 每行=一次命中日, 击中日期=该行日期, "
    "获利盘/换手等列=命中当天画像。显示=今日有点火旗(T1/T2/T1+T2)的股, 其余只在后台表 preboard_watch_hits_*.csv (全量)。"
    "次数10日/次数20日=过去10/20个交易日命中总数, 仅上下文勿筛选(回测: 点火前命中数不预测, 次数≥2纯度更低)。"
    "排序 = 点火旗(T1+T2>T1>T2) → 击中日期(新→旧) → 次数10日(多→少)。"
    "⚠ 深睡签名整体是反信号组 (5日首板率 2.4% vs 全池基线 5.2%) — 本页只提供可见性, 不是买入清单。"
    "回测: 点火后各天数次日进 TE 全负(−0.7~−1.6%), 点火旗=去看提示非买入依据; "
    "页内点火桶胜率为页内最强但绝对低于全主板基线。"
    "【通道优先】= 获利盘≥0.65 ∩ 20日内LHB净买+机构双旗 (TE 通道率 20.9%, 领先仅2~3天, 华瓷即此类); "
    "通道优先在深睡页≈恒空(结构性: 深睡∩wr≥0.65∩LHB双净买≈空集), 为空属正常。"
)

PB_LEGEND = (
    "列说明: 通道优先=获利盘≥0.65∩20日内LHB净买+机构双旗(置顶); 点火旗=T1首次≥2%/T2涨2~7%",
    "每股多行: 每行=一次命中日; 击中日期=该行日期(文本); 获利盘/20日获利盘Δ/换手列=命中当天画像",
    "通道优先/点火旗/当日涨幅=今日口径(当日停牌缺行→空/NaN); 次数10日/次数20日=过去10/20个交易日命中总数",
    "显示=今日有点火旗(T1/T2/T1+T2)的股, 其余只在后台表 preboard_watch_hits_*.csv (全量)",
    "获利盘=命中日收盘获利盘; 20日获利盘Δ=近20日获利盘升幅(签名腿之一, ≥10pp)",
    "5日均换手=近5日换手均值(%); 5日/20日换手比=缩量腿(签名腿之一, ≤0.75)",
    "★ 用法: 通道优先行 + 点火旗 = '开始动了' → 人工看盘确认, 勿程序化追买",
    "",
    "【通道优先·白话】= 获利盘≥0.65 且 近20天上过龙虎榜, 且当天净买与机构席位净买都为正",
    "这类票未来7天出首板概率约21% (普通票约8%), 但只领先2~3天; 没上过龙虎榜的票 (如华瓷) 永远不会标",
    "它是排序提示, 不是买入信号",
)

# 列 → Excel number_format (写的是**实数**, 显示带符号百分号; 文本会被 Excel 按字典序排坏)
_PCT2 = "+0.00%;-0.00%"
_NUMFMT = {
    "当日涨幅": _PCT2,
    "r20": _PCT2,
    "r60": _PCT2,
    "r120": _PCT2,
    "5日回撤": _PCT2,
    "乖离MA10": "+0.0%;-0.0%",
    "获利盘": "0.0%",
    "主力筹码比例": "0.0",
    "换手率": "0.00%",
    "量比": "0.00",
    "集中度Δ10": "0.00",
    "控盘MA10斜率": "0.00",
    "SL翻正年龄": "0",
    "SL洗盘天数": "0",
    # 首板点名页 (0922; 0923 v2: T+3/T+5 双概率+校准档位) / 板前哨页 (0922)
    "排名": "0",
    "T+3板概率": "0.0%",
    "T+5板概率": "0.0%",
    "板前获利盘": "0.0%",
    "板块涨停数": "0",
    "昨日晋级率": "0.0%",
    "市场涨停数": "0",
    "20日获利盘Δ": "0.0%",
    "5日均换手": "0.00",
    "5日/20日换手比": "0.00",
    "次数10日": "0",
    "次数20日": "0",
}

log = logging.getLogger("genious")


def _setup_logging(tag: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    log.setLevel(logging.INFO)
    if log.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_DIR / f"genious_{tag}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)


def _state_path(tag: str) -> Path:
    return LOG_DIR / f"genious_{tag}.state.json"


def _write_state(tag: str, status: str, **extra) -> None:
    payload = {"tag": tag, "status": status, "ts": datetime.datetime.now().isoformat()}
    payload.update(extra)
    _state_path(tag).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── 新鲜度 ────────────────────────────────────────────────────────────────────


def _heal_rows(target: str) -> pd.DataFrame:
    """自拉当日截面 + winner_ratio, **只回内存不写面板** (面板是 1.3GB 共享产物)。

    异常 OHLCV 大声失败 (铁律: 不得静默丢弃); cyq 缺席只 WARN (只影响 T1 闸)。
    """
    from app.pipeline1.data_supply import DataSupplyChain

    snap = DataSupplyChain().fetch_daily(target, refresh=True)
    bad = snap[
        (snap["high"] < snap["low"])
        | (snap["high"] < snap["open"])
        | (snap["high"] < snap["close"])
        | (snap["low"] > snap["open"])
        | (snap["low"] > snap["close"])
        | (snap["volume"] < 0)
    ]
    if len(bad):
        syms = bad["symbol"].astype(str).head(20).tolist()
        raise RuntimeError(f"自拉 {target} OHLCV 校验失败 {len(bad)} 行: {syms}")

    out = snap[["symbol", "date", "board", "high", "low", "close", "volume"]].copy()
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["date"] = target

    pro = DataSupplyChain()._tushare_pro()
    wr = None
    if pro is not None:
        try:
            cyq = pro.cyq_perf(trade_date=target)
        except Exception as e:  # noqa: BLE001 — cyq 缺席不该阻断交付 (T2/T3 不依赖)
            log.warning("[heal] cyq_perf 拉取失败: %s", e)
            cyq = None
        if cyq is not None and len(cyq):
            cyq = cyq[["ts_code", "winner_rate"]].copy()
            cyq["symbol"] = (
                cyq["ts_code"].astype(str).str.split(".").str[0].str.zfill(6)
            )
            wr = cyq.set_index("symbol")["winner_rate"].astype(float) / 100.0
    if wr is None:
        log.warning(
            "[heal] %s winner_ratio 缺失 → T1 本日不触发 (T2/T3 不受影响)", target
        )
    out["winner_ratio"] = out["symbol"].map(wr) if wr is not None else np.nan
    return out[list(kt.PANEL_COLUMNS)]


def _heal_rows_bounded(target: str) -> pd.DataFrame:
    """自拉带硬超时。

    Tushare 的 fetch_daily / cyq_perf **不接受 timeout 参数**, 底层 socket 挂起会永久阻塞。
    0914 实测: 冒烟进程 0 CPU 挂死 40 分钟, 让计划任务永久停在 Running (schtasks 默认
    IgnoreNew) — 之后每个交易日的 20:30 都会被静默跳过, 且不写任何终态。
    故用守护线程包一层: 超时即抛错 → 调用方写 state=failed + exit 2, 挂死变大声失败。
    """
    box: dict = {}

    def work() -> None:
        try:
            box["df"] = _heal_rows(target)
        except BaseException as e:  # noqa: BLE001 — 转发给主线程统一处理
            box["err"] = e

    th = threading.Thread(target=work, name="genious-heal", daemon=True)
    th.start()
    th.join(HEAL_TIMEOUT_S)
    if th.is_alive():
        raise TimeoutError(
            f"自拉 {target} 超过 {HEAL_TIMEOUT_S}s 未返回 (Tushare 连接挂起); "
            "拒绝无限等待 — 不产文件"
        )
    if "err" in box:
        raise box["err"]
    return box["df"]


def _load_fresh_panel(target: str, no_fetch: bool, wait_min: int) -> pd.DataFrame:
    """要求面板含 target 当日行; 不足则等待 → 自愈; 都不行抛错。"""
    cal = load_trade_cal()
    expect, src = expected_trading_date(target, cal)

    deadline = time.monotonic() + wait_min * 60
    pmax = kt.panel_max_date(PANEL_V3_PATH)
    while True:
        lag = lag_trading_days(pmax, expect, cal)
        if pmax is not None and str(pmax) == str(expect) and (lag is None or lag == 0):
            log.info(
                "[fresh] 面板最新 %s = 目标 %s (cal_source=%s), 直接读",
                pmax,
                target,
                src,
            )
            break
        if time.monotonic() >= deadline:
            break
        log.warning(
            "[fresh] 面板最新 %s 落后目标 %s, 等待 %ss (19:15 抓取可能未完成)",
            pmax,
            target,
            WAIT_TICK_S,
        )
        time.sleep(WAIT_TICK_S)
        pmax = kt.panel_max_date(PANEL_V3_PATH)

    df = kt.load_panel(PANEL_V3_PATH, target)
    have = df[df["date"] == target]
    if len(have):
        return df

    if no_fetch:
        raise RuntimeError(f"面板无 {target} 当日行且 --no-fetch: 不产文件")
    log.warning("[fresh] 面板无 %s 当日行 → 自拉当日截面 (只拼内存, 不写面板)", target)
    healed = _heal_rows_bounded(target)
    # 次新口径说明 (2026-09-18 用户令「为 GENIOUS 单独放宽」的落地判定):
    # **本模块不设 min_list_days 闸, 也不改**。原因有三, 全部实测:
    #   1) 缺行只发生在历史追跑 (fetch_daily 早于次日面板重建时点); 常规 20:30 当日链
    #      面板已含全部次新 (/_daily_fetch.py::_build_new_base_panel 不适用 ingest gate,
    #      实测 9/17 面板中「上市交易日 <150」的票数 = 0)。
    #   2) 自愈路径 fetch_daily → _tushare_fetch_daily 内部带 ingest gate, 会**原样复现**
    #      同一条剔除规则 —— 单靠放宽 GENIOUS 侧的闸, 这些行根本不会出现在 healed 里。
    #   3) T1 的 wr_rise60 需 61 bar、r120 需 121 bar, 次新股上这两个量结构性地是 NaN
    #      (实测 301583@20260915/16/17: 48/49/50 bar, r60 与 wr_rise60 均不可算) ——
    #      即便准入, T1/T2/T3 也全为 False。放宽只会放大未验证样本占比。
    # 故 301583 这一类的缺行属**数据年缺口**, 不在交付层闸的职责内。
    log.info(
        "[heal] 当日截面 %d 行, 其中 winner_ratio 非空 %d",
        len(healed),
        int(healed["winner_ratio"].notna().sum()),
    )
    return pd.concat([df, healed], ignore_index=True)


# ── 输出 ──────────────────────────────────────────────────────────────────────


def _fmt_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """数值列写**实数** (Excel 要能排序/筛选), 显示格式交给 _NUMFMT 的 number_format。

    写成 '+2.08%' 这类字符串时 Excel 按字典序排 (所有 '+…' 排在 '-…' 前), 排序即错。
    """
    out = df.copy()
    if "乖离MA10" in out.columns:
        out["乖离MA10"] = out["乖离MA10"] - 1  # ext10 1.05 → 显示 +5.0%
    return out


BIGDROP_HIGH = "大跌风险"
BIGDROP_VOL = "波动风险"
BIGDROP_NONE = "无风险"
BIGDROP_UNSCORED = "未评分"


def _norm_sym(s) -> str:
    """清单侧代码 → 面板键: 去交易所后缀 + 补零到 6 位 (920075.BJ → 920075)。"""
    return str(s).strip().split(".")[0].zfill(6)


def _bigdrop_labels(sc, p, th: float) -> np.ndarray:
    """(规则分, 模型概率) → 标注。**模型支优先于规则支**, 先写规则再被模型覆盖。

    顺序反了 (或把两边并成一个布尔) 就会把零方向的规则支标成 大跌风险 —— 规则支
    占报警面七成体量、次日均收益 +0.047%, 标成"高风险"会被读成"次日要跌", 正是
    bigdrop 0915 拆分分支要修掉的误读。这个守卫不为覆盖率, 为语义。
    """
    out = np.full(len(sc), BIGDROP_NONE, dtype=object)
    out[np.asarray(sc) >= 1] = BIGDROP_VOL
    out[np.asarray(p) >= th] = BIGDROP_HIGH
    return out


def _bigdrop_cells(labels, p, base: float) -> list[str]:
    """标注 → 单元格文本。**只有方向支 (大跌风险) 带倍数**, 另两态原样出。

    倍数 = 该股模型概率 ÷ OOS 市场基准大跌率, 即"次日大跌概率是市场的几倍"。用**逐股**
    概率而非分支常数, 是为了让高危档内部还能分出轻重 (实测 2.1x~9.4x)。

    波动风险 不带倍数 —— 不是省事, 是那个数在该分支上**恒为"比市场安全"**: 该分支按
    定义就是 p < th(0.10), 而 base=5.128%, 故倍数上限 = th/base = **1.95x**, 实测中位
    0.31x、87% 落在 1x 以下。挂一个"风险"标签却显示 0.3x, 读起来就是"风险只有市场的
    三成"= 比平均安全 —— 标签与数字自相矛盾 (用户 0916 报"0.2/0.3 不正常"即此)。
    该分支本就零方向, 摆一个方向性数字只会误导, 故只出标签。

    无风险 同样不带倍数; 它本身已是一个完整读数。
    """
    return [
        f"{x} {safe_divide(y, base):.1f}x" if x == BIGDROP_HIGH else x
        for x, y in zip(labels, p)
    ]


def _bigdrop_module_run() -> bool:
    """bigdrop **模块运行** (建包) —— 交付链里与 genious 顺序执行的第二步。

    此前 GENIOUS 只**读** bigdrop 的包, 没有任何入口跑它的建包, 盘上的包会一直停在
    上次手工 --build 那天。实测不是无害的 (tmp_t/_bigdrop_stale_delta_0916.py,
    0915 包 vs 0916 重建, 同一日截面 5,257 行): 标注类别变 11 行 (9 只 大跌风险
    → 波动风险)、倍数变 375 行, alarm/capture/branches 的 OOS 兑现整体漂移
    (报警格精度 8.96% → 7.65%)。所以顺序步骤是"重跑模块", 不是"读一个越来越旧的包"。

    **非致命**: 建包失败只记 ERROR 并回退盘上旧包 —— bigdrop 是旁路标注, 不许掀翻
    交付链 (同 _bigdrop_scan 契约)。包已同源时零成本跳过 (只读面板的 date 一列)。
    """
    try:
        from scripts import bigdrop_check as bc

        if not (bc.BUNDLE_DIR / "bundle_latest.joblib").exists():
            log.error("[genious] bigdrop 包缺失, 本次不跑模块也不产 BIGDROP SCAN 列")
            return False
        pmax = kt.panel_max_date(PANEL_V3_PATH)
        if pmax is None:
            log.error("[genious] 面板 date 列读失败 → 判不了包新鲜度, 跳过模块运行")
            return False
        data_date = pmax.strftime("%Y%m%d")
        b = bc.load_bundle()
        if not bc.bundle_is_stale(b, data_date):
            log.info(
                "[genious] bigdrop 包已同源 (tag=%s data_date=%s), 跳过建包",
                b.get("tag"),
                data_date,
            )
            return True
        log.info(
            "[genious] bigdrop 模块运行: 包 tag=%s data_date=%s vs 面板 %s → 重建",
            b.get("tag"),
            b.get("data_date"),
            data_date,
        )
        t0 = time.monotonic()
        nb = bc.build()
        log.info(
            "[genious] bigdrop 模块运行完成 %.1fs → tag=%s data_date=%s (OOS 基准 %.3f%%)",
            time.monotonic() - t0,
            nb.get("tag"),
            nb.get("data_date"),
            float(nb["oos_base"]) * 100,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — 旁路步骤, 失败回退旧包, 不许掀翻交付链
        log.error("[genious] bigdrop 模块运行失败, 回退盘上旧包: %s", exc)
        return False


def _bigdrop_scan(symbols) -> dict[str, str] | None:
    """把当日清单送 bigdrop 次日大跌模块过一遍 → 【BIGDROP SCAN】标注列。

    分档沿用 bigdrop_check 的**分支**口径 (0915 拆分), 不压成一个布尔:
      大跌风险 N.Nx = 模型支 (模型概率 >= 报警线) —— 唯一带看跌方向的一支
      波动风险 N.Nx = 仅规则支 (规则分>=1 而模型未达线) —— 零方向 (次日均 +0.047%)
      空            = 两边都没举手 (不出倍数)
    倍数是**逐股**读数 (见 _bigdrop_cells), 不是分支常数 —— 高危档内部靠它分轻重。
    压成单一 大跌风险 正是该模块 0915 重建要修掉的误读: 规则支占报警面七成却
    没有方向信息, 标成"高风险"会被读成"次日要跌"。

    依赖缺失不拖累交付: bigdrop 是**旁路标注**, 拿不到就 log 大声并返回 None
    (普通列照出, 名单一只不少)。load_bundle() 在缺包时 sys.exit, 那是 BaseException
    不是 Exception, 故先查文件存在性再进去。
    """
    try:
        from scripts import bigdrop_check as bc

        b = bc.load_bundle()
        d = bc.load_frame()
        day = d[d.groupby("symbol")["date"].transform("max") == d["date"]].reset_index(
            drop=True
        )
        _F, sc, P = bc.score_frame(day, b)
        p = np.asarray(P["模型(isotonic校准)"], dtype=float)
        th = float((b.get("alarm") or {}).get("th", bc.MODEL_ALARM))
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — 标注旁路, 不许掀翻交付链
        # load_bundle 在缺包时 sys.exit (SystemExit 是 BaseException, 需显式捕获);
        # 文件不存在或扫描异常均回退 None, 不拖累交付。
        log.error("[genious] bigdrop 扫描失败, 本次不产 BIGDROP SCAN 列: %s", exc)
        return None

    lab = _bigdrop_labels(sc, p, th)
    base = float(b["oos_base"])
    cells = _bigdrop_cells(lab, p, base)
    hit = dict(zip(day["symbol"].map(_norm_sym), cells))
    # 不在面板的票给 未评分 而不是 "" —— 空格与"查过且没问题"在表上长得一样,
    # 而含义相反 (未测 vs 测过无风险)。留空等于把没测的票静默读成安全。
    out = {_norm_sym(s): hit.get(_norm_sym(s), BIGDROP_UNSCORED) for s in symbols}
    unscored = [s for s, v in out.items() if v == BIGDROP_UNSCORED]
    # 只在 out 里数: lab 是全市场截面 (5000+ 只), 数它就把"送扫的 51 只"报成全市场。
    # 值已带倍数尾巴, 故用 startswith 而非等值比较。
    log.info(
        "[genious] BIGDROP SCAN [包 %s]: %d 只送扫, %s %d, %s %d, %s %d, %s %d (基准 %.2f%%)",
        b.get("tag"),
        len(out),
        BIGDROP_HIGH,
        sum(1 for v in out.values() if v.startswith(BIGDROP_HIGH)),
        BIGDROP_VOL,
        sum(1 for v in out.values() if v.startswith(BIGDROP_VOL)),
        BIGDROP_NONE,
        sum(1 for v in out.values() if v == BIGDROP_NONE),
        BIGDROP_UNSCORED,
        len(unscored),
        base * 100,
    )
    if unscored:
        log.warning(
            "[genious] %d 只送扫票不在面板, 没评上分 (列内标 %s): %s",
            len(unscored),
            BIGDROP_UNSCORED,
            unscored[:20],
        )
    return out


def write_stocklist_csv(
    s1: pd.DataFrame, date: str, list_dir=STOCK_LIST_DIR
) -> Path | None:
    """冠军四段 → genious_stocklist_{date}__{HHMMSS}.csv (WORM), 给 THS 推送当第三源。

    Sheet2 观察池不落这张 CSV: 推送侧只认个股买入名单, 观察池进去会污染自选股。
    空榜不落文件 (推送侧缺源即跳过, 不推空单)。
    """
    if not len(s1):
        log.warning("[genious] %s 冠军四段为空, 不落推送边车", date)
        return None
    stamp = datetime.datetime.now().strftime("%H%M%S")
    fp = Path(list_dir) / f"genious_stocklist_{date}__{stamp}.csv"
    # 0922 用户令: 边车也带记录 — 横盘提示/涨停提示 追加在 symbol 后 (推送侧按列名
    # 读 symbol, 加列不影响; 按 _norm_sym 归一的清单键在第一列不动)
    cols = ["排名", "symbol"] + [c for c in ("横盘提示", "涨停提示") if c in s1.columns]
    s1[cols].to_csv(fp, index=False)
    return fp


def _spawn_ths_push(date: str) -> None:
    """分离子进程跑同花顺推送 — 不与交付链同生共死。

    推送是 UI 自动化 (空闲闸 + 多轮补推 + 轮间静默), 单次可跑十几分钟, 且用户
    在场时按设计 fail-closed 退出; 挂在交付链上会把 GENIOUS 的 ok 终态拖住。
    """
    script = Path(__file__).with_name("_ths_watchlist_push.py")
    if not script.exists():
        log.warning("[genious] 找不到推送脚本 %s, 只落边车不推", script)
        return
    opts = 0
    if os.name == "nt":
        opts = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    try:
        subprocess.Popen(  # noqa: S603 — 固定脚本路径 + 日期参数, 无 shell
            [sys.executable, str(script), date],
            cwd=str(Path(__file__).resolve().parent.parent),
            creationflags=opts,
        )
        log.info("[genious] 已分离启动同花顺推送 (date=%s)", date)
    except Exception as exc:  # noqa: BLE001 — 推送失败不影响交付物
        log.warning("[genious] 启动推送失败: %s", exc)


def _build_firstboard_sheets(target: str):
    """首板点名页 + 板前哨页 (数据层在 scripts/_firstboard_pages.py)。

    面板 max ≠ 交付日 → (None, None): 这两页点的是"当日首板/当日深睡状态",
    日期错一天整页语义全错, 宁缺勿错; 冠军四段不受影响 (kt 路径有自己的新鲜度闸)。
    """
    from scripts import _firstboard_pages as fbp

    pdf = fbp.load_mainboard()
    d0 = pdf["date"].max()
    if d0 is None or d0.strftime("%Y%m%d") != target:
        log.error(
            "[genious] 面板最新 %s ≠ 交付日 %s → 首板两页本次不出 (勿点昨天的板)",
            d0,
            target,
        )
        return None, None
    # 板前哨后台全量表 (含次数0/1与当日停牌缺行): record_csv 契约见 _firstboard_pages.serve_preboard
    pb_csv = Path(STOCK_LIST_DIR) / f"preboard_watch_hits_{target}.csv"
    return fbp.serve_firstboard(pdf), fbp.serve_preboard(pdf, record_csv=pb_csv)


def write_xlsx(
    sheet1: pd.DataFrame,
    date: str,
    list_dir=STOCK_LIST_DIR,
    extra_sheets: list | None = None,
) -> Path:
    """WORM: GENIOUS_{date}.xlsx; 已存在 → GENIOUS_{date}__{HHMMSS}.xlsx (绝不覆盖)。

    0919 用户令: 观察池与全量表都删 — 只出冠军单表; 0922 起追加 首板点名/板前哨 两页
    (extra_sheets = [(name, df, banner, legend), ...], 旁路页, 允许为空)。"""
    fp = Path(list_dir) / f"{GENIOUS['filename_prefix']}_{date}.xlsx"
    if fp.exists():
        stamp = datetime.datetime.now().strftime("%H%M%S")
        fp = Path(list_dir) / f"{GENIOUS['filename_prefix']}_{date}__{stamp}.xlsx"
    sheets = [("冠军四段", sheet1, BANNER1, kt.sheet1_legend())] + [
        tuple(x) for x in (extra_sheets or [])
    ]
    with pd.ExcelWriter(fp, engine="openpyxl") as xw:
        for name, df, banner, legend in sheets:
            raw = df if len(df) else pd.DataFrame(columns=list(df.columns))
            raw.to_excel(xw, sheet_name=name, index=False, startrow=2)
            ws = xw.sheets[name]
            ws.cell(row=1, column=1, value=banner).font = Font(bold=True, size=9)
            ws.merge_cells(
                start_row=1,
                start_column=1,
                end_row=1,
                end_column=max(len(raw.columns), 2),
            )
            for i, col in enumerate(raw.columns, start=1):
                cell = ws.cell(row=3, column=i)
                cell.fill = PatternFill("solid", fgColor="D9E1F2")
                cell.font = Font(bold=True)
                vals = [len(str(v)) for v in raw[col]] if len(raw) else []
                ws.column_dimensions[cell.column_letter].width = min(
                    max([len(str(col)), *vals]) + 4, 40
                )
                nf = _NUMFMT.get(col)
                if nf:
                    for r in range(4, 4 + len(raw)):
                        ws.cell(row=r, column=i).number_format = nf
            # 图例写数据下方: 保持上方是干净表格 (筛选/排序不被打断), A 列右侧留空
            # 供文本溢出显示 — 不合并单元格, 免得挡住用户自己加行。
            for j, line in enumerate(legend):
                cell = ws.cell(row=4 + len(raw) + 1 + j, column=1, value=line)
                if line.startswith(("段位说明", "层说明", "列说明", "类型说明", "★")):
                    cell.font = Font(bold=True, size=10)
            ws.freeze_panes = "A4"
    return fp


# ── 验收闸: 全历史复现研究层表 ────────────────────────────────────────────────


# 大涨口径 (2026-09-18 用户令): 前瞻收益 > +8% 即大涨, **对所有股一视同仁, 不按板分档**。
# 旧值 0.098 是"次日涨停"口径 (10% 板), 加入 20% 板后不再成立 —— 且用户明确大涨是
# 幅度口径而非涨停口径。大跌对称为 < -5%。温火上沿固定 5%, 同样不随 20% 板放宽。
BIG_RISE_TH = 0.08
BIG_DROP_TH = -0.05


# (层名, 火/日, 真赢/日, 大涨/日, 胜率, 中位火/日, 零票天%) — 0914 续13 互斥口径
# 2026-09-18 刷新: 宇宙加入科创板 (68) + 涨停阈值改按板分档 (见 kt.UNIVERSE_PREFIXES
# 与 kt._limit_up_threshold)。旧值是在 主板+创业板 上量的, 直接沿用则 verify 必然 FAIL。
_EXPECT = (
    (kt.CH3_T3_DEEP_QUIET, 3.7, 2.94, 2.02, 0.786, 0, 0.73),
    (kt.CH2_T2_DEEP, 3.8, 2.27, 1.00, 0.593, 0, 0.54),
    (kt.CH1_T1_LONGBASE, 1.9, 1.18, 0.25, 0.606, 0, 0.66),
    (kt.CH2B_T2_STEADY, 8.3, 4.35, 1.36, 0.527, 1, 0.46),
    (kt.T1_REST, 9.5, 4.92, 0.88, 0.516, 3, 0.25),
    (kt.BAND_T2_WARM, 27.5, 13.64, 2.75, 0.497, 7, 0.21),
    (kt.BAND_T2_LIMIT, 1.0, 0.48, 0.17, 0.500, 0, 0.62),
    (kt.BAND_T3_WARM, 9.6, 4.84, 1.56, 0.505, 2, 0.35),
    (kt.BAND_T3_LIMIT, 5.0, 2.34, 0.90, 0.471, 2, 0.30),
)

# 用户 0911 时间线案例: (symbol, 日期, 期望层或 None, 期望触发器子串)
# 601869 r60≈-14% 不满足 CH2 的 r60<=-30 闸, 记忆里它是 "T2 命中" 而非冠军段 → 只验触发器。
_CASES = (
    ("000978", "20260903", kt.CH1_T1_LONGBASE, "T1"),
    ("601869", "20260907", None, "T2"),
    ("002815", "20260907", kt.BAND_T3_LIMIT, "T3"),
    ("603421", "20260908", kt.BAND_T2_WARM, "T2"),
)


def verify() -> int:
    """全历史跑一遍, 对 续13 层表与案例命中逐项 PASS/FAIL。"""
    log.info("[verify] 读全量面板 ...")
    df = kt.load_panel(
        PANEL_V3_PATH, datetime.date.today().strftime("%Y%m%d"), lookback_days=0
    )
    df = kt.compute_features(df)
    df = kt.compute_triggers(df)
    df["层"] = kt.assign_layers(df)
    g = df["symbol"]
    df["f5"] = df["close"].groupby(g, sort=False).shift(-5) / df["close"] - 1

    all_days = df["date"].nunique()
    day_index = sorted(df["date"].unique())
    fired = df[df["层"] != ""]
    matured = fired[fired["f5"].notna()]
    log.info(
        "[verify] %d 交易日, 起火 %d 行 (成熟 %d)", all_days, len(fired), len(matured)
    )

    print(
        f"\n全量: {all_days} 交易日; 日频分母 = 全交易日 (同 续13 '互斥口径全896日均')"
    )
    print(
        f"  {'层':<18}{'火/日':>8}{'真赢/日':>9}{'大涨/日':>9}{'胜率':>8}{'中位':>6}{'零票天':>8}{'判定':>6}"
    )
    results, fails = {}, []
    for name, e_fire, e_win, e_big, e_rate, e_med, e_zero in _EXPECT:
        sub = matured[matured["层"] == name]
        per = sub.groupby("date").size().reindex(day_index, fill_value=0)
        got = (
            len(sub) / all_days,
            (sub["f5"] > 0).sum() / all_days,
            (sub["f5"] > BIG_RISE_TH).sum() / all_days,
            (sub["f5"] > 0).mean() if len(sub) else float("nan"),
            float(per.median()),
            1 - per.gt(0).mean(),
        )
        ok = (
            abs(got[0] - e_fire) <= 0.2
            and abs(got[1] - e_win) <= 0.2
            and abs(got[2] - e_big) <= 0.2
            and abs(got[3] - e_rate) <= 0.015
            and abs(got[5] - e_zero) <= 0.05
        )
        fails += [] if ok else [name]
        results[name] = {
            "got": got,
            "expect": (e_fire, e_win, e_big, e_rate, e_med, e_zero),
            "ok": ok,
        }
        print(
            f"  {name:<18}{got[0]:>8.1f}{got[1]:>9.2f}{got[2]:>9.2f}{got[3]:>8.1%}"
            f"{got[4]:>6.0f}{got[5]:>8.0%}{'  PASS' if ok else '  FAIL':>6}"
            f"   期望 {e_fire:.1f}/{e_win:.2f}/{e_big:.2f}/{e_rate:.1%}/{e_med:.0f}/{e_zero:.0%}"
        )

    # 全榜总闸 (交叉验证分层总和; 2026-09-18 宇宙扩容 63.3→70.3; 2026-09-23 随
    # 0922 删票闸/BJ剔除+行情漂移刷新: 70.3→72.6 火/日)
    per_all = matured.groupby("date").size().reindex(day_index, fill_value=0)
    board = (
        len(matured) / all_days,
        (matured["f5"] > 0).sum() / all_days,
        (matured["f5"] > BIG_RISE_TH).sum() / all_days,
        (matured["f5"] > 0).mean(),
        float(per_all.median()),
    )
    board_ok = (
        abs(board[0] - 72.6) <= 0.5
        and abs(board[1] - 38.3) <= 0.5
        and abs(board[2] - 11.7) <= 0.5
        and abs(board[3] - 0.528) <= 0.01
    )
    if not board_ok:
        fails.append("全榜总数")
    print(
        f"\n  {'全榜 (冠军+余+带)':<18}{board[0]:>8.1f}{board[1]:>9.2f}{board[2]:>9.2f}"
        f"{board[3]:>8.1%}{board[4]:>6.0f}{'':>8}{'PASS' if board_ok else 'FAIL':>6}"
        f"   期望 72.6/38.3/11.7/52.8%"
    )
    results["_全榜"] = {"got": board, "ok": board_ok}

    print(f"\n案例命中 ({len(_CASES)}):")
    case_res = {}
    for sym, date, want_layer, want_trig in _CASES:
        row = df[(df["symbol"] == sym) & (df["date"] == date)]
        got_layer = row["层"].iloc[0] if len(row) else "(缺行)"
        trig = kt.trigger_name(row).iloc[0] if len(row) else ""
        ok = want_trig in trig and (want_layer is None or got_layer == want_layer)
        fails += [] if ok else [f"{sym}@{date}"]
        case_res[f"{sym}@{date}"] = {
            "got": got_layer,
            "want": want_layer,
            "trigger": trig,
            "want_trigger": want_trig,
            "ok": ok,
        }
        print(
            f"  {sym}@{date}  触发器={trig or '-'} (期望含 {want_trig})  层={got_layer}"
            f"  期望={want_layer or '(只验触发器)'}  {'PASS' if ok else 'FAIL'}"
        )

    DIAG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = DIAG_DIR / f"genious_verify_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "ts": ts,
                "days": all_days,
                "layers": results,
                "cases": case_res,
                "fails": fails,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n{'全部 PASS' if not fails else 'FAIL: ' + ', '.join(fails)}  → {out}")
    return 0 if not fails else 1


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("date", nargs="?", default=None, help="YYYYMMDD (缺省=当日)")
    ap.add_argument("--verify", action="store_true", help="全历史复现研究层表 (验收闸)")
    ap.add_argument(
        "--no-fetch", action="store_true", help="面板缺当日行时不自拉, 直接失败"
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印不落文件")
    ap.add_argument("--no-push", action="store_true", help="落边车但不推同花顺自选股")
    ap.add_argument("--wait-min", type=int, default=10, help="等面板更新的上限分钟")
    args = ap.parse_args()

    today = datetime.date.today().strftime("%Y%m%d")
    tag = args.date or today
    _setup_logging(tag)

    if args.verify:
        return verify()

    if not GENIOUS["enable"]:
        log.info("[genious] settings.GENIOUS.enable=False, 跳过")
        _write_state(tag, "disabled")
        return 0

    cal = load_trade_cal()
    if args.date is None:
        if cal is not None and pd.Timestamp(today) not in pd.DatetimeIndex(cal):
            log.info("[genious] %s 非交易日, 跳过", today)
            _write_state(tag, "skipped", reason="not_trading_day")
            return 0
        tag = today
    target = tag

    _write_state(tag, "running")
    try:
        df = _load_fresh_panel(target, args.no_fetch, args.wait_min)
    except Exception as e:  # noqa: BLE001 — 交付链失败一律显式终态, 不产旧数据清单
        log.error("[genious] 新鲜度自愈失败: %s", e)
        _write_state(tag, "failed", reason="freshness")
        return 2

    # bigdrop 模块运行 (用户 0916 令: 交付链 = genious 运行 + bigdrop 运行, 顺序执行)。
    # 放在新鲜度闸之后 —— 建包用的面板必须含交付当日行, 否则包一落地就是旧的。
    # --dry-run 的契约是"只打印不落文件", 而建包要写 models/bigdrop → 跳过建包,
    # 仍用盘上现有包出标注列 (读是干净的, 写才违约)。
    if args.dry_run:
        log.info("[genious] --dry-run: 跳过 bigdrop 模块运行 (不落包), 用盘上现有包")
    else:
        _bigdrop_module_run()

    df = kt.compute_features(df)
    df = kt.compute_triggers(df)
    s1, _ = kt.build_delivery(df, target)
    # 旁路标注列 (用户 0915 令): 清单过一遍 bigdrop 次日大跌模块。三张表都加,
    # 语义见 _bigdrop_scan —— 三态 (大跌风险/波动风险/无风险) + 未评分。
    # 查表一律走 _norm_sym: 直接 str(s).zfill(6) 会漏掉北交所的 `.BJ` 后缀, 那只票
    # 就静默变空 —— 与"没评上分"撞脸, 而分其实算过。
    # 缺省值同 _bigdrop_scan: 拿不到读数标 未评分, **不留空** (空格与"查过且无风险"同形)。
    # 附加列一律 insert(0) **放最前** (用户 0915 令) —— 追加到末尾会被列宽/横向滚动吞掉,
    # 后面新增的旁路列照此办理。
    scan = _bigdrop_scan(s1["symbol"])
    if scan is not None:
        s1.insert(
            0,
            "BIGDROP SCAN",
            s1["symbol"].map(lambda s: scan.get(_norm_sym(s), BIGDROP_UNSCORED)),
        )
    # 筹码水位标注 (0922 用户令, 第四线同源): chip_wr<0.5 低获利，涨 / ≥0.5 高获利，跌。
    # 冠军表 0919 精简后不含获利盘数值列, 故走与密度/LEGACY/PARALLEL 同一条
    # apply_chip_gate (load_chip_features 读 cyq_panel) — 四线同源同切分, 勿在此
    # 另写阈值; cyq 缺 → fail-open 只不加列, 名单一只不少 (同 BIGDROP 旁路契约)。
    chipped = apply_chip_gate(s1, pd.Timestamp(target))
    if "chip_flag" in chipped.columns:
        flags = chipped["chip_flag"].to_numpy()
        s1.insert(1 if "BIGDROP SCAN" in s1.columns else 0, "筹码标注", flags)
        log.info(
            "[genious] 筹码水位标注: %s %d / %s %d / 空 %d",
            CHIP_FLAG_LOW,
            int((flags == CHIP_FLAG_LOW).sum()),
            CHIP_FLAG_HIGH,
            int((flags == CHIP_FLAG_HIGH).sum()),
            int((flags == "").sum()),
        )
    # 横盘提示 (0922 用户令: 最终 STOCKLIST 必须带记录, 光日志无用): 近10日涨幅<2%
    # 且 冷静市 → "近10日未涨·冷静市". 股票级附加列 insert(0) 放最前 (同 BIGDROP SCAN
    # 惯例); 日级 市场温度/参与建议 每行同值 → 垫表尾; 中间量 (涨幅/入选次数/board)
    # 不进表。genious 史自 0918 起, "近20日入选次数" 已不参与判定纯参考。
    s1 = stall_marker(s1, target, "genious_stocklist_")
    if len(s1) and "近10日涨幅" in s1.columns and s1["近10日涨幅"].isna().all():
        log.warning(
            "[genious] 面板缺 %s 当日行 (自愈路径滞后?), 横盘提示/涨停提示 全空", target
        )
    s1 = s1.drop(
        columns=[
            c
            for c in ("近10日涨幅", "昨日涨幅", "近20日入选次数", "board")
            if c in s1.columns
        ]
    )
    _front = [c for c in ("横盘提示", "涨停提示") if c in s1.columns]
    s1 = s1[_front + [c for c in s1.columns if c not in _front]]
    n_stall = int((s1["横盘提示"] != "").sum()) if len(s1) else 0
    _adv = s1["参与建议"].iloc[0] if len(s1) and "参与建议" in s1.columns else ""
    _temp = s1["市场温度"].iloc[0] if len(s1) and "市场温度" in s1.columns else None
    if _temp is not None and pd.isna(_temp):
        _temp = None
    log.info("[genious] 横盘提示 %d 只; %s", n_stall, _adv)

    counts = s1["层"].value_counts().to_dict()
    n_pass = int((s1["涨闸"] == "过闸").sum())
    log.info(
        "[genious] %s 冠军表 %d 票 (涨闸过 %d); 层分布 %s",
        target,
        len(s1),
        n_pass,
        counts,
    )

    # 首板点名页 + 板前哨页 (0922 用户令): 当日全部首板点名 + 深睡监视名单。
    # 旁路契约同 BIGDROP: 构建失败只丢这两页, 冠军四段照常交付。
    extra_sheets: list = []
    try:
        fb_df, pb_df = _build_firstboard_sheets(target)
        if fb_df is not None:
            extra_sheets.append(("首板点名", fb_df, FB_BANNER, FB_LEGEND))
        if pb_df is not None:
            extra_sheets.append(("板前哨", pb_df, PB_BANNER, PB_LEGEND))
        log.info(
            "[genious] 首板点名 %d 行 / 板前哨 %d 行",
            len(fb_df) if fb_df is not None else 0,
            len(pb_df) if pb_df is not None else 0,
        )
    except Exception as exc:  # noqa: BLE001 — 旁路页, 不许掀翻交付链
        log.error("[genious] 首板两页构建失败, 本次只交付冠军四段: %s", exc)

    if args.dry_run:
        for name, sheet in [("冠军四段", s1)] + [(t[0], t[1]) for t in extra_sheets]:
            print(f"\n===== {name} ({len(sheet)}) =====")
            print(sheet.to_string(index=False) if len(sheet) else "(空)")
        _write_state(
            tag,
            "dry_run",
            s1=len(s1),
            s1_pass=n_pass,
            layers=counts,
            n_stall=n_stall,
            market_temp=_temp,
        )
        return 0

    fp = write_xlsx(_fmt_sheet(s1), target, extra_sheets=extra_sheets)
    log.info("[genious] 写出 %s", fp)
    csv_fp = write_stocklist_csv(s1, target)
    if csv_fp is not None:
        log.info("[genious] 推送边车 %s", csv_fp)
    _write_state(
        tag,
        "ok",
        file=str(fp),
        s1=len(s1),
        s1_pass=n_pass,
        layers=counts,
        n_stall=n_stall,
        market_temp=_temp,
    )
    print(str(fp))
    if GENIOUS.get("push_to_ths") and csv_fp is not None and not args.no_push:
        _spawn_ths_push(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
