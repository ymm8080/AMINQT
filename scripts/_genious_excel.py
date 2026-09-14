"""GENIOUS — 链路狙击交付层 Excel (2026-09-14 用户令: 交易日 20:30 自动跑).

Sheet1 = 冠军四段 (CH3 T3深跌缩量 / CH2 T2深跌 / CH1 T1长基 / CH2B T2稳健)
Sheet2 = 观察池 (T1余 + 带双指纹四臂)
**不截断** — 任何逐日 cap 都删洪峰日的钱 (0914 续12 取证级判死), 层序+层内 r60
深→浅 即排名。层定义与阈值见 app/pipeline1/kongduo_triggers.py 与 settings.GENIOUS。

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

LOG_DIR = Path(PROJECT_ROOT) / "logs"
DIAG_DIR = Path(PROJECT_ROOT) / "diag"
WAIT_TICK_S = 60
HEAL_TIMEOUT_S = 180  # 自拉硬超时; 超时=大声失败, 不留挂死实例 (见 _heal_rows_bounded)

BANNER1 = (
    "GENIOUS 冠军四段 — 段位全留, 【大涨闸】列标出其中哪几只是 低动量+右侧拐头+缩量。"
    "末250日口径: 冠军四段 19.2票·日 里只有 1.3票·日 过闸 (93%被拦), "
    "过闸后 未来10日均值 +9.00% 胜81.3%, ≥10% 命中 53.0% = 2.05x, ≥20% 命中 12.7% = 1.94x; "
    "所以**只标注、不删票** — 过闸那几只是窄名单, 其余仍按层序读。"
    "「全样本口径」列含选段偏差(**80% 勿信**)。扣0.7%往返费后火群整体≈0, 钱在层头部, "
    "请按层序自上而下读。执行档: 温火/质量层=T+1开盘进; 涨停/深跌层=T+1仍涨确认→T+1收盘进"
)
BANNER2 = (
    f"GENIOUS 观察池 — 已按【观察分】从最好到最差排序, 取前 {GENIOUS['sheet2_top_n']} 名 "
    "(全部名次见「火群全量」表)。观察分 = 带宽窄 + 未偏离MA10 + 获利盘低 + 深跌 → "
    "越靠前越'还没涨透'; 已发挥完的票自动沉底。全 896 日实测: 前半档 +0.65% vs 末档 -0.03%, "
    "IC 0.077 (t 8.7), 前后半样本同号。期望≈50% 平水, 非全买清单。"
    "【大涨闸】列仅为标注, 不删除任何一行"
)
BANNER3 = (
    "GENIOUS 火群·全量 — 冠军四段 + 观察池**全部**触发票, 一票不丢。"
    "【大涨闸】列: 过闸 = 三条件全中 (十日涨幅≤0 且 近5日涨幅>0 且 量比≤1); "
    "被拦 = 未全中 (**排在最前**, 供回看)。**该列只标注不筛表** — 三张表都是全量, 别把它当筛选器。"
    "被拦者按层序排列, 深跌层 (CH2/CH3) 天然被拦比例最高 — 这是形态特征, 非数据错误"
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
    "SL翻正年龄": "0",
    "SL洗盘天数": "0",
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
    _state_path(tag).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
            cyq["symbol"] = cyq["ts_code"].astype(str).str.split(".").str[0].str.zfill(6)
            wr = cyq.set_index("symbol")["winner_rate"].astype(float) / 100.0
    if wr is None:
        log.warning("[heal] %s winner_ratio 缺失 → T1 本日不触发 (T2/T3 不受影响)", target)
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
            log.info("[fresh] 面板最新 %s = 目标 %s (cal_source=%s), 直接读", pmax, target, src)
            break
        if time.monotonic() >= deadline:
            break
        log.warning(
            "[fresh] 面板最新 %s 落后目标 %s, 等待 %ss (19:15 抓取可能未完成)",
            pmax, target, WAIT_TICK_S,
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
    log.info("[heal] 当日截面 %d 行, 其中 winner_ratio 非空 %d",
             len(healed), int(healed["winner_ratio"].notna().sum()))
    return pd.concat([df, healed], ignore_index=True)


# ── 输出 ──────────────────────────────────────────────────────────────────────


def _fmt_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """数值列写**实数** (Excel 要能排序/筛选), 显示格式交给 _NUMFMT 的 number_format。

    写成 '+2.08%' 这类字符串时 Excel 按字典序排 (所有 '+…' 排在 '-…' 前), 排序即错。
    """
    out = df.copy()
    out["乖离MA10"] = df["乖离MA10"] - 1  # ext10 1.05 → 显示 +5.0%
    return out


def write_stocklist_csv(s1: pd.DataFrame, date: str, list_dir=STOCK_LIST_DIR) -> Path | None:
    """冠军四段 → genious_stocklist_{date}__{HHMMSS}.csv (WORM), 给 THS 推送当第三源。

    Sheet2 观察池不落这张 CSV: 推送侧只认个股买入名单, 观察池进去会污染自选股。
    空榜不落文件 (推送侧缺源即跳过, 不推空单)。
    """
    if not len(s1):
        log.warning("[genious] %s 冠军四段为空, 不落推送边车", date)
        return None
    stamp = datetime.datetime.now().strftime("%H%M%S")
    fp = Path(list_dir) / f"genious_stocklist_{date}__{stamp}.csv"
    s1[["排名", "symbol"]].to_csv(fp, index=False)
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


def write_xlsx(
    sheet1: pd.DataFrame,
    sheet2: pd.DataFrame,
    sheet2_full: pd.DataFrame,
    date: str,
    list_dir=STOCK_LIST_DIR,
) -> Path:
    """WORM: GENIOUS_{date}.xlsx; 已存在 → GENIOUS_{date}__{HHMMSS}.xlsx (绝不覆盖)。"""
    fp = Path(list_dir) / f"{GENIOUS['filename_prefix']}_{date}.xlsx"
    if fp.exists():
        stamp = datetime.datetime.now().strftime("%H%M%S")
        fp = Path(list_dir) / f"{GENIOUS['filename_prefix']}_{date}__{stamp}.xlsx"
    with pd.ExcelWriter(fp, engine="openpyxl") as xw:
        for name, df, banner, legend in (
            ("冠军四段", sheet1, BANNER1, kt.sheet1_legend()),
            ("观察池", sheet2, BANNER2, kt.sheet2_legend()),
            ("火群全量", sheet2_full, BANNER3, kt.sheet2_legend()),
        ):
            raw = df if len(df) else pd.DataFrame(columns=list(df.columns))
            raw.to_excel(xw, sheet_name=name, index=False, startrow=2)
            ws = xw.sheets[name]
            ws.cell(row=1, column=1, value=banner).font = Font(bold=True, size=9)
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(raw.columns), 2))
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


# (层名, 火/日, 真赢/日, 大涨/日, 胜率, 中位火/日, 零票天%) — 0914 续13 互斥口径
_EXPECT = (
    (kt.CH3_T3_DEEP_QUIET, 3.0, 2.40, 1.49, 0.802, 0, 0.76),
    (kt.CH2_T2_DEEP, 3.1, 1.86, 0.67, 0.598, 0, 0.58),
    (kt.CH1_T1_LONGBASE, 1.7, 1.06, 0.15, 0.633, 0, 0.68),
    (kt.CH2B_T2_STEADY, 7.2, 3.87, 0.85, 0.536, 1, 0.48),
    (kt.T1_REST, 8.4, 4.41, 0.51, 0.523, 3, 0.26),
    (kt.BAND_T2_WARM, 24.5, 12.38, 1.72, 0.505, 6, 0.21),
    (kt.BAND_T2_LIMIT, 1.1, 0.52, 0.14, 0.492, 0, 0.61),
    (kt.BAND_T3_WARM, 7.8, 3.92, 0.92, 0.502, 1, 0.37),
    (kt.BAND_T3_LIMIT, 6.4, 3.05, 0.95, 0.473, 3, 0.27),
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
    df = kt.load_panel(PANEL_V3_PATH, datetime.date.today().strftime("%Y%m%d"), lookback_days=0)
    df = kt.compute_features(df)
    df = kt.compute_triggers(df)
    df["层"] = kt.assign_layers(df)
    g = df["symbol"]
    df["f5"] = df["close"].groupby(g, sort=False).shift(-5) / df["close"] - 1

    all_days = df["date"].nunique()
    day_index = sorted(df["date"].unique())
    fired = df[df["层"] != ""]
    matured = fired[fired["f5"].notna()]
    log.info("[verify] %d 交易日, 起火 %d 行 (成熟 %d)", all_days, len(fired), len(matured))

    print(f"\n全量: {all_days} 交易日; 日频分母 = 全交易日 (同 续13 '互斥口径全896日均')")
    print(f"  {'层':<18}{'火/日':>8}{'真赢/日':>9}{'大涨/日':>9}{'胜率':>8}{'中位':>6}{'零票天':>8}{'判定':>6}")
    results, fails = {}, []
    for name, e_fire, e_win, e_big, e_rate, e_med, e_zero in _EXPECT:
        sub = matured[matured["层"] == name]
        per = sub.groupby("date").size().reindex(day_index, fill_value=0)
        got = (
            len(sub) / all_days,
            (sub["f5"] > 0).sum() / all_days,
            (sub["f5"] > 0.098).sum() / all_days,
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
        results[name] = {"got": got, "expect": (e_fire, e_win, e_big, e_rate, e_med, e_zero), "ok": ok}
        print(
            f"  {name:<18}{got[0]:>8.1f}{got[1]:>9.2f}{got[2]:>9.2f}{got[3]:>8.1%}"
            f"{got[4]:>6.0f}{got[5]:>8.0%}{'  PASS' if ok else '  FAIL':>6}"
            f"   期望 {e_fire:.1f}/{e_win:.2f}/{e_big:.2f}/{e_rate:.1%}/{e_med:.0f}/{e_zero:.0%}"
        )

    # 全榜总闸 (续13 独立的第三个数字, 用来交叉验证分层总和): 63.3火/33.5真赢/7.4大涨/52.9%
    per_all = matured.groupby("date").size().reindex(day_index, fill_value=0)
    board = (
        len(matured) / all_days,
        (matured["f5"] > 0).sum() / all_days,
        (matured["f5"] > 0.098).sum() / all_days,
        (matured["f5"] > 0).mean(),
        float(per_all.median()),
    )
    board_ok = (
        abs(board[0] - 63.3) <= 0.5
        and abs(board[1] - 33.5) <= 0.5
        and abs(board[2] - 7.4) <= 0.3
        and abs(board[3] - 0.529) <= 0.01
    )
    if not board_ok:
        fails.append("全榜总数")
    print(
        f"\n  {'全榜 (冠军+余+带)':<18}{board[0]:>8.1f}{board[1]:>9.2f}{board[2]:>9.2f}"
        f"{board[3]:>8.1%}{board[4]:>6.0f}{'':>8}{'PASS' if board_ok else 'FAIL':>6}"
        f"   期望 63.3/33.5/7.4/52.9%"
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
            "got": got_layer, "want": want_layer, "trigger": trig, "want_trigger": want_trig, "ok": ok,
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
            {"ts": ts, "days": all_days, "layers": results, "cases": case_res, "fails": fails},
            ensure_ascii=False, indent=2, default=str,
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
    ap.add_argument("--no-fetch", action="store_true", help="面板缺当日行时不自拉, 直接失败")
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

    df = kt.compute_features(df)
    df = kt.compute_triggers(df)
    s1, s2, s2_full = kt.build_delivery(df, target)
    counts = pd.concat([s1["层"], s2_full["层"]]).value_counts().to_dict()
    n_pass = int((s1["大涨闸"] == "过闸").sum())
    log.info(
        "[genious] %s 冠军四段 %d 票 (大涨闸过 %d), 观察池 %d/%d 票 (截断/全量); 层分布 %s",
        target, len(s1), n_pass, len(s2), len(s2_full), counts,
    )

    if args.dry_run:
        for name, sheet in (("冠军四段", s1), ("观察池", s2), ("火群全量", s2_full)):
            print(f"\n===== {name} ({len(sheet)}) =====")
            print(sheet.to_string(index=False) if len(sheet) else "(空)")
        _write_state(
            tag, "dry_run", s1=len(s1), s1_pass=n_pass, s2=len(s2),
            s2_full=len(s2_full), layers=counts,
        )
        return 0

    fp = write_xlsx(_fmt_sheet(s1), _fmt_sheet(s2), _fmt_sheet(s2_full), target)
    log.info("[genious] 写出 %s", fp)
    csv_fp = write_stocklist_csv(s1, target)
    if csv_fp is not None:
        log.info("[genious] 推送边车 %s", csv_fp)
    _write_state(
        tag, "ok", file=str(fp), s1=len(s1), s1_pass=n_pass, s2=len(s2),
        s2_full=len(s2_full), layers=counts,
    )
    print(str(fp))
    if GENIOUS.get("push_to_ths") and csv_fp is not None and not args.no_push:
        _spawn_ths_push(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
