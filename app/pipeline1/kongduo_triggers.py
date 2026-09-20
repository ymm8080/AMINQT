"""链路狙击交付层引擎 — 空→多翻转 / 洗盘日 / 状态点火 三事件触发器 (2026-09-14).

来源: 0914 全池回放 (tmp_t/_kongduo_flip_replay_0914.py, tmp_t/_trigger_best_seg_0914.py,
主板+创业板 4381 只 20230103-20260911 354 万行) 的**逐字固化**。用户驱动案例:
000978 (空@9/3洗盘→多@9/4→四连板+49%) 与 000823 同款链路。

三触发器 (纯规则, 不吃任何模型产物; 只要面板 OHLCV + winner_ratio):
  T1 洗盘日 = 空图标 × r20>0 × 获利盘>0.6 × 60日筹码升>5pp × MA10向上 × 乖离≤5%
  T2 翻转日 = 前10日内出现过空图标 × 今日多图标(上穿) × 当日涨幅>2%
  T3 状态点火 = 多头态(gap>0, 非交叉) × 前5日回撤≥3% × 当日涨幅>5% × 昨日乖离≤5%

图标公式 (THS 多空趋势, 已破解并 4/4 真值验证, 勿改): VAR1 = 100+pos_vec(21,90)-90,
inner6 = 100-pos_vec(6,100), SIG = 34日均线再过6日均线。duo/kong 是**上穿事件**
(只在翻转当日为真), 故 T2 的"空窗口"用 rolling(10).max().shift(1)。

关键坑 (0914 同 bug 犯两次): rolling().max() 后必须 `.fillna(False).astype(bool)`
**两连** — 只 fillna 不 astype 仍是 float, `float & bool` 抛 TypeError。

口径警示: 用面板**原始 close/high/low** (非 close_hfq) = 回放口径, 换 hfq 会让整张
判词表失效; 代价是除权日会污染 r20/r60/r120。**扣 0.7% 往返费后火群整体≈0**, 钱只在
分层头部 (CH1 +2.30% / CH2 +6.38% / CH3 +10.10%) → 层优先排名是把"费用负"转成
"费用正"的那一步。逐日 cap 已判死 (洪峰日集中 68.6% 的火), 榜长不限。

纯函数、无 IO 副作用 (read-only 的 load_panel 除外), 阈值全部读 settings.GENIOUS。
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.settings import GENIOUS

# 宇宙: 沪深主板 + 创业板 + 科创板 (2026-09-18 用户令: 科创板不入表是缺陷).
# 仍排除北交所 (92) —— 用户 0918: "only 北交 is out of scope".
# 注意: 0914 全样本回放是在 主板+创业板 上做的, 加入 68 后各层数字会漂移
# (实测 火 63.60 → 72.76/日, 全榜胜率 52.5% → 52.4%, CH3 80.0% → 78.6%),
# 故 _genious_excel._EXPECT 验收闸必须同步刷新, 否则 verify 必然 FAIL。
UNIVERSE_PREFIXES = ("00", "30", "60", "68")

# 涨跌幅三档口径 (2026-09-18 用户令) —— **对所有股一视同仁, 不按板分档**:
#   前瞻收益 > +8%  → 大涨
#   前瞻收益 < -5%  → 大跌
#   温火上沿固定 5% (不随 20% 板放宽)
# 只有"涨停"这一件事按板分档 (10%板 vs 20%板), 见 _limit_up_threshold。
# 大涨/大跌口径落在 _genious_excel.BIG_RISE_TH / BIG_DROP_TH (前瞻收益是标注量, 不进交付表)。
_LIMIT_UP_GROWTH = 0.20
_LIMIT_UP_MAIN = 0.10
# 判定留白: 用 9.5%/19.5% 而非 10%/20%, 吸收浮点与价格取整 (对齐 settings.PRICE_LIMIT_PCT=9.5)
_LIMIT_UP_TOL = 0.005


def _limit_up_threshold(symbols: pd.Series) -> pd.Series:
    """逐股涨停阈值 (小数): 创业板/科创板 0.195, 主板 0.095。

    单值 cfg["limit_up"]=0.09 是 10% 板时代的产物; 科创板 (20%) 上它会把
    +9%~+19.5% 的普通上涨全判成"涨停"。返回 Series 与 df 对齐, 供 pct > lim 比较。
    """
    growth = symbols.str[:2].isin(("30", "68"))
    return pd.Series(
        np.where(
            growth, _LIMIT_UP_GROWTH - _LIMIT_UP_TOL, _LIMIT_UP_MAIN - _LIMIT_UP_TOL
        ),
        index=symbols.index,
        dtype=float,
    )


PANEL_COLUMNS = (
    "symbol",
    "date",
    "board",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover_rate",
    "winner_ratio",
    "cost_50pct",
    "cost_95pct",
)

# ── 主力筹码比例 (益盟 A04 红柱复刻) ────────────────────────────────────────────
# 网格与 app/indicators/chip_distribution.py 逐字一致: 固定对数包络 0.5~6000。
# 绝不用 per-stock min/max 网格 —— 未来价格会拉伸网格、改写历史行的获利盘
# (0912 前视修复 818bd386, 合成验证漂移最高 100pp)。
CHIP_GRID_BINS = 2000
CHIP_GRID_LO = 0.5
CHIP_GRID_HI = 6000.0
# 红柱取样价位 = 典型价 × 0.96; 绿柱顶上沿 = 典型价 × 1.04
CHIP_RED_BAR_OFF = 0.96
CHIP_GREEN_TOP_OFF = 1.04
# 筹码趋势列的平滑窗 (控盘 MA10 斜率 / 集中度带宽 Δ10 都取 10 个交易日 ≈ 两周)
CHIP_TREND_MA = 10

# ── 交付层段位 (层名用段位不用裸字母: 续13 的第5层"B1"数值上=T1余, 与类型标签 B1 易混) ──
CH3_T3_DEEP_QUIET = "CH3 T3深跌缩量"
CH2_T2_DEEP = "CH2 T2深跌"
CH1_T1_LONGBASE = "CH1 T1长基"
CH2B_T2_STEADY = "CH2B T2稳健"
CH4_TREND_QUIET = "CH4 中线缩量横盘"
T1_REST = "T1余"
BAND_T2_WARM = "带T2温火"
BAND_T2_LIMIT = "带T2涨停"
BAND_T3_WARM = "带T3温火"
BAND_T3_LIMIT = "带T3涨停"

# 0918 用户令: 冠军表裁到两段 (CH1长基期望-2.34%/CH2B稳健+0.91% 不及市值基线; CH2/CH3 OOS
# f10 口径 +4.3%/+4.6%, 胜率55%/52%)。CH1/CH2B 降级进 Sheet2 观察池 (段位列仍可辨),
# 票也全量落 Sheet3 (build_delivery 一票不丢)。
SHEET1_LAYERS = (CH3_T3_DEEP_QUIET, CH2_T2_DEEP)
SHEET2_LAYERS = (
    CH1_T1_LONGBASE,
    CH2B_T2_STEADY,
    CH4_TREND_QUIET,
    T1_REST,
    BAND_T2_WARM,
    BAND_T2_LIMIT,
    BAND_T3_WARM,
    BAND_T3_LIMIT,
)
ALL_LAYERS = SHEET1_LAYERS + SHEET2_LAYERS

EXEC_NEXT_OPEN = "T+1开盘进"
EXEC_CONFIRM = "T+1仍涨确认→T+1收盘进"

# 执行档: 20:30 已收盘, "fire日收盘进"实际只能落到 T+1 开盘
LAYER_EXEC = {
    CH3_T3_DEEP_QUIET: EXEC_CONFIRM,
    CH2_T2_DEEP: EXEC_CONFIRM,
    CH2B_T2_STEADY: EXEC_NEXT_OPEN,
    CH1_T1_LONGBASE: EXEC_NEXT_OPEN,
    CH4_TREND_QUIET: EXEC_NEXT_OPEN,
    T1_REST: EXEC_NEXT_OPEN,
    BAND_T2_WARM: EXEC_NEXT_OPEN,
    BAND_T2_LIMIT: EXEC_CONFIRM,
    BAND_T3_WARM: EXEC_NEXT_OPEN,
    BAND_T3_LIMIT: EXEC_CONFIRM,
}

# 当月样本口径 (滚动重算, 源 .cache/kongduo_layer_monthly_<月>.json); build 期覆写。
# 下面是 fallback 手填值 (无 json 时的兜底)。勿当预期收益
LAYER_RESEARCH_FALLBACK = {
    CH3_T3_DEEP_QUIET: "64.4% / +3.12%",
    CH2_T2_DEEP: "68.3% / +2.05%",
    CH1_T1_LONGBASE: "30.9% / -0.23%",
    CH2B_T2_STEADY: "61.8% / +0.93%",
    CH4_TREND_QUIET: "35.7% / +2.26%",
    T1_REST: "28.8% / -1.34%",
    BAND_T2_WARM: "32.8% / -0.73%",
    BAND_T2_LIMIT: "39.2% / +1.12%",
    BAND_T3_WARM: "34.9% / -0.22%",
    BAND_T3_LIMIT: "38.3% / +1.23%",
}


def load_layer_research(trade_date: str | None = None) -> dict[str, str]:
    """从 .cache/kongduo_layer_monthly_<当月>.json 读每层当月口径。

    json 缺失 (月度重算没跑) → 返回 fallback 並照常出表, 不硬断。
    """
    import json as _json
    from pathlib import Path

    ym = (trade_date or datetime.date.today().strftime("%Y%m%d"))[:6]
    path = (
        Path(__file__).resolve().parents[2]
        / ".cache"
        / f"kongduo_layer_monthly_{ym}.json"
    )
    if not path.exists():
        return dict(LAYER_RESEARCH_FALLBACK)
    try:
        d = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError, KeyError, TypeError):
        # 只捕读文件/解析失败; 其他异常冒泡 (符合铁律: try-except 但不过度吞异常)
        return dict(LAYER_RESEARCH_FALLBACK)
    raw = d.get("data_by_layer", {})
    out = {}
    for name in ALL_LAYERS:
        v = raw.get(name)
        if not v:
            out[name] = LAYER_RESEARCH_FALLBACK.get(name, "")
            continue
        win, exp = v.get("win"), v.get("exp")
        out[name] = (
            f"{win:.1%} / {exp:+.2%}" if win is not None and exp is not None else ""
        )
    return out


# build_delivery 期由 load_layer_research 覆写; 默认 fallback
LAYER_RESEARCH = dict(LAYER_RESEARCH_FALLBACK)


def _w(s: str) -> int:
    """显示宽度 (CJK 算 2 列)。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    """按**显示宽度**右侧补空格, 让 Excel 里说明列对齐。"""
    return s + " " * max(width - _w(s), 1)


def _seg_lines() -> tuple[str, ...]:
    """冠军四段条件 (冠军表底图例与观察池图例共用); 阈值一律从 GENIOUS 取。"""
    c = GENIOUS
    deep = f"{c['r60_deep']:.0%}"  # -30%
    base = "0" if c["r120_base"] == 0 else f"{c['r120_base']:.0%}"
    return (
        f"  {CH3_T3_DEEP_QUIET} = T3状态点火 且 r60≤{deep} 且 量比≤{c['vr_quiet']}",
        f"  {CH2_T2_DEEP} = T2翻转日 且 r60≤{deep}",
        f"  {CH1_T1_LONGBASE} = T1洗盘日 且 r120≤{base} 且 r20>+{c['r20_min']:.0%}",
        f"  {CH2B_T2_STEADY} = T2翻转日 且 昨日乖离MA10≤{c['t2b_ext_prev_max']:.2f}"
        f" 且 r120≤{base}",
        f"  {CH4_TREND_QUIET} = r60>+{c['ch4_r60_min']:.0%} 且 MA10上行"
        f" 且 量比≤{c['ch4_vr_max']} 且 距5日高点<{c['ch4_dd5_max']:.0%} (缩量回调横盘)",
    )


def _col_lines() -> tuple[tuple[str, str], ...]:
    """列说明 — 冠军四段与观察池的列完全相同, 只写一份, 免得两处漂移。"""
    c = GENIOUS
    deep = f"{c['r60_deep']:.0%}"  # -30%
    base = "0" if c["r120_base"] == 0 else f"{c['r120_base']:.0%}"
    return (
        ("排名", "表内序号; 先按段位序 (CH3→CH2→CH1→CH2B), 段内按 r60 从深到浅"),
        ("symbol", "6 位股票代码 (已去掉 .SH/.SZ 后缀)"),
        (
            "涨闸",
            "过闸 | 没过闸 — 涨闸 (右侧拐头 + 缩量) 的标注; "
            '**不删任何一行**, 见下方"涨闸说明"',
        ),
        ("层", '冠军段位 — 四段互斥, 见上方"段位说明"'),
        (
            "触发器",
            "今日命中的触发器: T1洗盘日 / T2翻转日 / T3状态点火 / T2+T3(同日双触发)",
        ),
        ("类型", 'r60/r120 分桶标签 — 见下方"类型说明"'),
        ("当日涨幅", "今日涨跌幅 = 今收 / 昨收 − 1"),
        (
            "执行档",
            "T+1开盘进 | T+1仍涨确认→T+1收盘进 (20:30 已收盘, 只能 T+1 买)",
        ),
        (
            "当月样本口径",
            "该段位**当月实绩** (滚动重算) 胜率 / 5日均收益; 源 .cache/kongduo_layer_monthly_<月>.json",
        ),
        ("r20", "最近 20 个交易日涨跌幅 (= 今收 / 20交易日前收 − 1)"),
        ("r60", f'最近 60 个交易日涨跌幅 (中期位置; ≤{deep} 即本表的"深跌")'),
        ("r120", f'最近 120 个交易日涨跌幅 (长期位置; ≤{base} 即"半年没涨")'),
        ("获利盘", "当日获利盘比例; 越高 = 上方套牢盘越少"),
        (
            "主力筹码比例",
            "益盟「主力筹码红柱」= 成本低于「典型价×0.96」的筹码占比 (%, 0~100); "
            "越高 = 筹码越聚在当前价下方 (主力控盘越实)。**当警戒读数用, 别当买入信号** —"
            "末 250 日 OOS 对未来10日 IC −0.059 (t −5.9), 五分位向下 (最低桶 +0.15% → "
            "最高桶 −0.02%); 深跌层 (r60≤−30%) 天然上方套牢重 ⇒ 红柱必然偏小, 属形态使然",
        ),
        ("换手率", "今日成交量 / 流通股本 (Tushare daily_basic 口径)"),
        ("量比", "今量 / 前 5 日均量; <1 缩量, >1 放量"),
        (
            "乖离MA10",
            '收盘 / 10日均线 − 1: 正 = 在均线上方, 越大越"追高/过热"; 负 = 均线下方',
        ),
        ("5日回撤", "近 5 日相对 20 日高点的最深回撤 (负值, 越负回撤越深)"),
    )


def _type_lines() -> tuple[tuple[str, str], ...]:
    """r60/r120 分桶标签 (A/C/B2/B1/B3/D1/D2) 说明。"""
    c = GENIOUS
    deep = f"{c['r60_deep']:.0%}"  # -30%
    base = "0" if c["r120_base"] == 0 else f"{c['r120_base']:.0%}"
    return (
        (
            TYPE_A,
            f"r60 ≤ {deep} — 已深跌 (最深一桶, 优先级最高)",
        ),
        (
            TYPE_C,
            f"r120 > +{TYPE_R120_C:.0%} 且 r60 ≤ {TYPE_R60_FLAT:.0%} — "
            "半年大涨过、近期回调",
        ),
        (
            TYPE_B2,
            f"r120 ≤ {base} 且 {TYPE_R60_FLAT:.0%} ≤ r60 ≤ +{TYPE_R60_D1:.0%} — "
            "长期没涨、近期横盘",
        ),
        (
            TYPE_B1,
            f"{deep} < r60 < {TYPE_R60_FLAT:.0%} — 中等跌幅",
        ),
        (TYPE_B3, f"{TYPE_R60_FLAT:.0%} ≤ r60 ≤ +{TYPE_R60_D1:.0%} — 浅跌/平"),
        (TYPE_D1, f"+{TYPE_R60_D1:.0%} < r60 ≤ +{TYPE_R60_D2:.0%} — 已涨"),
        (TYPE_D2, f"r60 > +{TYPE_R60_D2:.0%} — 大涨 (已发挥完, 多在观察池)"),
    )


def _band_lines() -> tuple[str, ...]:
    """观察池五臂条件 (T1余 + 带双指纹四臂)。"""
    c = GENIOUS
    base = "0" if c["r120_base"] == 0 else f"{c['r120_base']:.0%}"
    return (
        f"  {T1_REST} = T1洗盘日 且 r120≤{c['t1_wide_r120_max']:.0%}"
        f" 且 r60≥{c['t1_wide_r60_min']:.0%} (T1宽, 未进冠军段)",
        f"  {BAND_T2_WARM} = T2余 且 r120≤{base}"
        f" 且 {c['band_t2_lo']:.0%}<涨幅≤{c['band_t2_hi']:.0%} 且 量比≤{c['band_vr_max']:g}",
        f"  {BAND_T2_LIMIT} = T2余 且 r120≤{base} 且 涨幅>{_LIMIT_UP_MAIN - _LIMIT_UP_TOL:.1%}"
        f"(创业板/科创板 >{_LIMIT_UP_GROWTH - _LIMIT_UP_TOL:.1%})",
        f"  {BAND_T3_WARM} = T3余 且 r120≤{base}"
        f" 且 {c['band_t3_lo']:.0%}<涨幅≤{c['band_t3_hi']:.0%} 且 量比≤{c['band_vr_max']:g}",
        f"  {BAND_T3_LIMIT} = T3余 且 r120≤{base} 且 涨幅>{_LIMIT_UP_MAIN - _LIMIT_UP_TOL:.1%}"
        f"(创业板/科创板 >{_LIMIT_UP_GROWTH - _LIMIT_UP_TOL:.1%})",
    )


def _gate_lines() -> tuple[str, ...]:
    """涨闸说明 (三张表底部)。数=引擎真实口径末250日实测。"""
    c = GENIOUS
    if not c["dir_gate"]:
        return ("涨闸: 已关闭 (dir_gate=False) — 全部标「过闸」。",)
    return (
        "涨闸说明 (⚠ 只标注, 不删任何一行 — 名单一只不少, 请按「涨闸」列自查)",
        f"  过闸 = 右侧拐头 (近5日涨幅>0, 不做左侧) + 缩量 (量比≤{c['dir_gate_vr_max']:g}); "
        "0918 用户令: 列名统一「涨闸」, 不分低动量/缺口/高动量档 — 全部标过闸",
        "  OOS f10+5%止损口径 (新冠军二段池): 过闸全取 胜率58%/期望+5.2%/大涨15.6%; "
        "其中低动量子集胜率76%/+8.5%/大涨21%, 缺口档 76.8%/+8.9%/24.4% (样本极薄)。",
        "  ⚠ 只当标记用, 一筛表就只剩头几行 — 没过闸的票不删, 供回看复核。",
    )


def _tail_lines() -> tuple[str, ...]:
    """两表共用的口径警示 (放最后, 不被上面的说明淹没)。"""
    return (
        "r20/r60/r120 用面板未复权原价计算 (与全样本判词同口径), 除权日会注入假跌幅。",
        "当月样本口径 = 该层**当月实绩** (滚动重算, 每月更新), 小月样本少勿单月下注。",
        "扣 0.7% 往返费后火群整体 ≈0 — 钱只在冠军四段的头部, 请按层序读, 勿无脑全买。",
    )


def sheet1_legend() -> tuple[str, ...]:
    """冠军四段表底部图例 (0914 用户令: 段位中文含义 + r20 口径写进表里)。"""
    cond, cols, types = _seg_lines(), _col_lines(), _type_lines()
    w = max(_w(s) for s in cond)
    cw = max(_w(k) for k, _ in cols)
    tw = max(_w(k) for k, _ in types)
    return (
        "段位说明 (四段互斥, 优先级 CH3 > CH2 > CH1 > CH2B)",
        *(
            _pad(s, w) + f"[当月 {LAYER_RESEARCH[name]}]"
            for s, name in zip(cond, SHEET1_LAYERS)
        ),
        "",
        "列说明 (按表内从左到右)",
        *(_pad("  " + k, cw + 2) + "= " + v for k, v in cols),
        "",
        "类型说明 (r60/r120 分桶; 重叠时优先级 A > C > B2 > B1 > B3 > D1 > D2)",
        *(_pad("  " + k, tw + 2) + "= " + v for k, v in types),
        "",
        *_tail_lines(),
    )


def sheet2_legend() -> tuple[str, ...]:
    """观察池表底部图例 (0914 用户令: 观察池也要有 footer)。

    与冠军表图例同源 (_col_lines / _type_lines / _tail_lines), 只在层说明与排序键两处不同。
    """
    c = GENIOUS
    bands, cols, types = _band_lines(), _col_lines(), _type_lines()
    bw = max(_w(s) for s in bands)
    cw = max(_w(k) for k, _ in cols)
    tw = max(_w(k) for k, _ in types)
    vr3 = c["band3_vr_outer_max"]
    vr3_txt = "不限" if vr3 is None else f"{vr3:g}"
    return (
        "层说明 (五臂互斥; 冠军四段之外的余票, 与冠军表零重叠)",
        *(
            _pad(s, bw) + f"[当月 {LAYER_RESEARCH[name]}]"
            for s, name in zip(bands, SHEET2_LAYERS)
        ),
        "  余票口径: T2余/T3余 = 命中 T2/T3 但已进冠军段的票不再进观察带。",
        f'  四臂"剔毒"外闸: r60≤{c["band_r60_max"]:.0%}、量比≤{c["band2_vr_outer_max"]:g}'
        f"(T3臂 {vr3_txt})、昨日乖离≤{c['band2_ext10p_max']:.2f}"
        f" — 无此闸温火臂会吞掉任何 r120≤0 的当日上涨票 (24.5 → 140 票/日)。",
        "",
        "★ 排序键 = 观察分 (不显示在列里; 表已按它从高到低排好, 重复名次见「火群全量」)",
        f"  = 同日截面 z 分求和  -({' + '.join(RANK_Z_COLUMNS)})",
        '  含义: 带宽越窄 / 越贴 MA10 / 获利盘越低 / 跌得越深 → 分越高 = 越"还没涨透"; '
        "已发挥完的自动沉底。",
        "  实测 (全 896 日): 首档 +0.65% vs 末档 -0.03%, IC +0.077 (t 8.7), 前后半样本同号。",
        "  期望≈50% 平水 — 这是**观察**排序, 不是全买清单 (排名键换成段位反而伤 IC, 同日截面 IC -0.025)。",
        "",
        "列说明 (按表内从左到右; 末两列 SL* 仅本表有)",
        *(_pad("  " + k, cw + 2) + "= " + v for k, v in cols[:-1]),
        "  "
        + _pad("SL翻正年龄", cw)
        + "= 益盟 S-L 由负转正至今的交易日数 (0 = 今日刚翻正)",
        "  "
        + _pad("SL洗盘天数", cw)
        + "= 这次翻正之前那段负值持续了多少交易日 (越长洗得越久)",
        "  两列只作标注: 实测 IC 仅 0.013 (t 1.5), 80.6% 与 T2 图标翻转重叠, 故不当排序键。",
        *(_pad("  " + k, cw + 2) + "= " + v for k, v in cols[-1:]),
        "",
        "类型说明 (r60/r120 分桶; 重叠时优先级 A > C > B2 > B1 > B3 > D1 > D2)",
        *(_pad("  " + k, tw + 2) + "= " + v for k, v in types),
        "",
        *_gate_lines(),
        "",
        *_tail_lines(),
    )


# 类型标签 (续13 表); B3/D1 在研究中被引用但未给规则, 按回放 r60 分桶补齐
TYPE_A = "A深跌反转"
TYPE_C = "C强牛深回调"
TYPE_B2 = "B2长平底"
TYPE_B1 = "B1中跌带"
TYPE_B3 = "B3浅跌平"
TYPE_D1 = "D1已涨"
TYPE_D2 = "D2大涨"
TYPE_UNKNOWN = "?"


def _iso(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _date_filter(trade_date, days: int) -> list | None:
    """pyarrow 谓词下推谓词; days<=0 → 全量 (None = 不过滤, 供 --verify)。面板 date 为 timestamp[ns]。"""
    if days <= 0:
        return None
    return [("date", ">=", pd.Timestamp(trade_date) - pd.Timedelta(days=days))]


# ── IO ────────────────────────────────────────────────────────────────────────


def load_panel(path, trade_date, lookback_days: int | None = None) -> pd.DataFrame:
    """只读面板尾部窗口 (r120 + SIG 34+6 rolling 需 ~250 交易日历史).

    返回: symbol(6位) / date(YYYYMMDD str) / board / open / high / low / close /
    volume / turnover_rate / winner_ratio, 已按 (symbol, date) 排序且宇宙过滤。
    """
    days = GENIOUS["lookback_days"] if lookback_days is None else int(lookback_days)
    tbl = pq.read_table(
        str(path), columns=list(PANEL_COLUMNS), filters=_date_filter(trade_date, days)
    )
    df = tbl.to_pandas()
    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.replace(r"\.(SH|SZ|BJ)", "", regex=True)
        .str.zfill(6)
    )
    df = df[df["symbol"].str[:2].isin(UNIVERSE_PREFIXES)]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df[list(PANEL_COLUMNS)]


def _chip_grid() -> np.ndarray:
    return np.exp(
        np.linspace(np.log(CHIP_GRID_LO), np.log(CHIP_GRID_HI), CHIP_GRID_BINS)
    )


def _triangle(grid: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    """当日新增筹码的三角分布 (峰值在典型价); 全出界时退化为最近格点。"""
    w = np.zeros(grid.size)
    m = (grid >= low) & (grid <= high)
    if m.any():
        w[m] = np.where(
            grid[m] <= peak,
            (grid[m] - low) / max(peak - low, 1e-9),
            (high - grid[m]) / max(high - peak, 1e-9),
        )
        np.clip(w, 0, None, out=w)
        s = w.sum()
        if s > 0:
            return w / s
    w[np.argmin(np.abs(grid - peak))] = 1.0
    return w


def compute_main_chip_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """加三列益盟「主力筹码控盘程度N」复刻量 (%, 0~100):

      `主力筹码比例` = A04 红柱 = WINNER(典型价 × 0.96) × 100  — **成本低于今价 4% 以上**的筹码占比。
                       实测与面板 `winner_ratio`(获利盘) 秩相关 ~0.98 ⇒ 它是**获利盘口径**, 不是控盘。
      `绿顶筹码`     = A02 绿柱顶 = WINNER(典型价 × 1.04) × 100 — 含 ±4% 带上沿, 即"上方套牢"的补集。
      `主力筹码控盘` = A08 黄柱 = A02 − A03 = 成本落在**典型价 ±4% 带内**的筹码占比。

    ★ A08 才是"控盘": 筹码越集中在现价附近 (控盘越足) 就越大, 与"赚没赚钱"无关 — 与获利盘正交。
      用户 0914 点名 000978/000823: 爆发前夜 000823 的 A08 连续 P92~P98 (含 9/4 前收 0.057 的
      极窄成本带), 而同期 A04 只有 P55~P69; A04 门槛 30 会把 000823 直接闸掉, A08 不会。

    口径源头 = app/indicators/chip_distribution.py::ChipDistribution.build 的 A02/A03/A04/A08
    (换手率衰减迁移 + 三角分布); 命名源头 = app/core/ths_indicators.compute_chip_control
    的 tech_ths_ctrl_low ("低价区筹码%")。

    逐股递归 ⇒ 必须在**切当日之前**对整窗调用 (当日值依赖全部历史筹码迁移)。窗口取
    GENIOUS['lookback_days'] 已够: 累计换手上百 % 后初始分布被冲刷掉。换手率列是
    Tushare daily_basic 的百分比口径 (3.5 = 3.5%), 故先 /100。
    """
    grid = _chip_grid()
    res = df.copy()
    res["主力筹码比例"] = np.nan
    res["绿顶筹码"] = np.nan
    res["主力筹码控盘"] = np.nan
    for _, g in df.groupby("symbol", sort=False):
        o = g["open"].to_numpy(float)
        h = g["high"].to_numpy(float)
        low = g["low"].to_numpy(float)
        c = g["close"].to_numpy(float)
        t = np.clip(np.nan_to_num(g["turnover_rate"].to_numpy(float)) / 100.0, 0.0, 1.0)
        red = np.full(len(g), np.nan)
        green = np.full(len(g), np.nan)
        dist = np.zeros(grid.size)
        for i in range(len(g)):
            if not (np.isfinite(o[i]) and np.isfinite(c[i])):
                continue
            a01 = (c[i] + o[i] + low[i] + h[i]) / 4.0
            w = _triangle(grid, low[i], h[i], a01)
            dist = w if dist.sum() == 0 else dist * (1.0 - t[i]) + t[i] * w
            tot = dist.sum()
            if tot > 0:
                red[i] = dist[grid < a01 * CHIP_RED_BAR_OFF].sum() / tot * 100.0
                green[i] = dist[grid < a01 * CHIP_GREEN_TOP_OFF].sum() / tot * 100.0
        res.loc[g.index, "主力筹码比例"] = red
        res.loc[g.index, "绿顶筹码"] = green
        res.loc[g.index, "主力筹码控盘"] = green - red
    return res


def compute_chip_trend(df: pd.DataFrame) -> pd.DataFrame:
    """加两列筹码趋势量 (按 symbol 时序, rolling/shift 只吃 t 及更早):

      `控盘MA10斜率` = MA10(主力筹码控盘) 与 10 日前的差 (>0 = 筹码向现价收敛)。
                       **只作展示列** — 0914 全枚举实测它是最弱的条件 (冠军末250 单条件 ≥20% 仅
                       0.95x, 往闸里加每加一次都掉), 已从涨闸摘除。
      `集中度Δ10`    = 带宽 W 的 10 日变化, W = (cost_95pct − cost_50pct) / cost_50pct。
                       W↑ 即 `集中度Δ10` > 0 = 筹码带变宽(发散)。**只作展示列, 不作闸** —
                       实测方向与直觉相反且很强 (变发散 ≥10% 命中 1.19x / 变集中 0.62x), 但用户
                       两个种子案例 000978/000823 恰是强变集中, 硬闸会把它们筛掉。见 settings.GENIOUS。

    必须在 compute_main_chip_ratio **之后**调用 (依赖 `主力筹码控盘`)。两列都是 MA 级平滑,
    lookback_days 默认 400 日足够预热。
    """
    n = CHIP_TREND_MA
    res = df.copy()
    k = res.groupby("symbol", sort=False)["主力筹码控盘"].transform(
        lambda x: x.rolling(n, min_periods=n).mean()
    )
    res["控盘MA10斜率"] = k - k.groupby(res["symbol"]).shift(n)
    c50 = res["cost_50pct"].to_numpy(float)
    w = (res["cost_95pct"].to_numpy(float) - c50) / np.where(c50 == 0.0, np.nan, c50)
    res["集中度Δ10"] = pd.Series(w, index=res.index).groupby(res["symbol"]).diff(n)
    return res


# ── 特征 ──────────────────────────────────────────────────────────────────────


def _roll(s: pd.Series, n: int, how: str, symbols: pd.Series) -> pd.Series:
    return s.groupby(symbols, sort=False).transform(
        lambda x: getattr(x.rolling(n, min_periods=n), how)()
    )


def _pos_vec(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
    k: int,
    symbols: pd.Series,
) -> pd.Series:
    """THS 位置向量: 收盘在 N 日高低区间的相对位 (k=量纲), 缺失日向前填充。"""
    hhv = _roll(high, n, "max", symbols)
    llv = _roll(low, n, "min", symbols)
    den = (hhv - llv).to_numpy()
    num = (hhv - close).to_numpy()
    r = np.where(den <= 0, np.nan, k - k * num / den)
    return pd.Series(r, index=close.index).groupby(symbols, sort=False).ffill()


def _rsv(df: pd.DataFrame, n: int) -> pd.Series:
    """益盟 RSV: 收盘在 N 日高低区间的位置 (∈[-100,0]), 缺失日 ffill。"""
    g = df["symbol"]
    hhv = _roll(df["high"], n, "max", g)
    llv = _roll(df["low"], n, "min", g)
    den = (hhv - llv).to_numpy()
    num = (df["close"] - hhv).to_numpy()
    r = np.where(den <= 0, np.nan, 100 * num / den)
    return pd.Series(r, index=df.index).groupby(g, sort=False).ffill()


def _th_signal(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """THS 多空趋势图标 (已破解公式): 返回 (duo 上穿, kong 下穿, gap=VAR1-SIG)。"""
    symbols = df["symbol"]
    var1 = 100 + _pos_vec(df["high"], df["low"], df["close"], 21, 90, symbols) - 90
    inner6 = 100 - _pos_vec(df["high"], df["low"], df["close"], 6, 100, symbols)
    sig = inner6.groupby(symbols, sort=False).transform(
        lambda x: (
            (100 - x.rolling(34, min_periods=34).mean())
            .rolling(6, min_periods=6)
            .mean()
        )
    )
    v_prev = var1.groupby(symbols, sort=False).shift(1)
    s_prev = sig.groupby(symbols, sort=False).shift(1)
    duo = ((var1 > sig) & (v_prev <= s_prev)).fillna(False).astype(bool)
    kong = ((sig > var1) & (s_prev <= v_prev)).fillna(False).astype(bool)
    return duo, kong, var1 - sig


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """派生全部触发器所需特征 (只含 <=t 信息, shift/rolling 均因果)。"""
    out = df.copy()
    g = out["symbol"]
    close = out["close"]
    duo, kong, gap = _th_signal(out)
    out["duo"] = duo
    out["kong"] = kong
    out["gap"] = gap

    out["pct"] = close.groupby(g, sort=False).pct_change()
    for n in (5, 10, 20, 60, 120):
        out[f"r{n}"] = close.groupby(g, sort=False).pct_change(n)

    wr = out["winner_ratio"]
    out["wr_rise60"] = wr - wr.groupby(g, sort=False).shift(60)

    out["dh20"] = close / _roll(out["high"], 20, "max", g) - 1
    out["pb5"] = out.groupby(g, sort=False)["dh20"].transform(
        lambda x: x.rolling(5, min_periods=5).min().shift(1)
    )

    # 20 日高低带宽 (压缩度): Sheet2 排序键之一, 窄=未启动
    out["band20"] = (
        _roll(out["high"], 20, "max", g) / _roll(out["low"], 20, "min", g) - 1
    )

    # 益盟 S-L 标准化 = (短期线-长期线)/长期线, 短期线=RSV14+100, 长期线=MA(RSV34,19)+100。
    # 两者同加 100, 故 sign(SL) 等价于 RSV14 > MA(RSV34,19), 只取符号做状态机。
    # 仅作 Sheet2 **标注** (翻正年龄/洗盘天数), 不作排序主键: 全 897 日实测 IC 0.013 (t 1.5),
    # 且在翻正子集内会毁掉观察分单调性; 另有 80.6% 与 T2 图标翻转重叠 (点二列相关 0.274)。
    sl_pos = (
        (
            _rsv(out, 14)
            > _rsv(out, 34)
            .groupby(g, sort=False)
            .transform(lambda x: x.rolling(19, min_periods=19).mean())
        )
        .fillna(False)
        .astype(bool)
    )
    sl_prev = sl_pos.groupby(g, sort=False).shift(1, fill_value=False)
    sl_up = sl_pos & ~sl_prev
    sl_idx = out.groupby(g, sort=False).cumcount()
    out["sl_flip_age"] = (
        sl_idx
        - pd.Series(sl_idx, index=out.index).where(sl_up).groupby(g, sort=False).ffill()
    )
    # 该次翻正之前那段负段的天数: 负段内 = 与段首日差 +1, 再 ffill 到其后正段
    # (段首用 transform("min") 取, 不能用 cumcount — 分组里含段的那个正日会把长度 +1)
    sl_neg_idx = sl_idx.where(~sl_pos)
    sl_run_start = sl_neg_idx.groupby(
        [g, sl_pos.groupby(g, sort=False).cumsum()], sort=False
    ).transform("min")
    out["sl_wash_days"] = (sl_neg_idx - sl_run_start + 1).groupby(g, sort=False).ffill()

    out["kong10"] = (
        out.groupby(g, sort=False)["kong"]
        .transform(
            lambda x: (
                x.rolling(GENIOUS["t2_kong_lookback"], min_periods=1).max().shift(1)
            )
        )
        .fillna(False)
        .astype(bool)
    )

    s_idx = out.groupby(g, sort=False).cumcount()
    last_kong = (
        pd.Series(s_idx, index=out.index)
        .where(out["kong"])
        .groupby(g, sort=False)
        .ffill()
    )
    out["d_since_kong"] = s_idx - last_kong
    # 看多交叉后的年龄 (与 d_since_kong 对称)。作 Sheet2 标注列, 并供 T3 分支
    # 判定"点火是否发生在看多交叉之后" —— 交叉之前 T3 只是普通放量上涨。
    last_duo = (
        pd.Series(s_idx, index=out.index)
        .where(out["duo"])
        .groupby(g, sort=False)
        .ffill()
    )
    out["d_since_duo"] = s_idx - last_duo

    ma10 = close.groupby(g, sort=False).transform(
        lambda x: x.rolling(10, min_periods=10).mean()
    )
    out["ma10up"] = ma10 > ma10.groupby(g, sort=False).shift(1)
    out["ext10"] = close / ma10
    out["ext10p"] = out["ext10"].groupby(g, sort=False).shift(1)

    # CH4 中线缩量横盘 的回调深度: 当日收盘距 5 日高点的回撤 (<ch4_dd5_max 即"横着缩")
    # 5日高点 max 做分母; 停牌日 high=0 → 除零防护 (对齐 _pos_vec np.where 口径)
    dd5_denom = _roll(out["high"], 5, "max", g)
    out["dd5"] = (dd5_denom - close) / np.where(dd5_denom <= 0, np.nan, dd5_denom)

    vol = out["volume"].replace(0, np.nan)
    out["vr"] = vol / vol.groupby(g, sort=False).transform(
        lambda x: x.rolling(5, min_periods=5).mean().shift(1)
    )
    return out


def compute_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """加 T1/T2/T3 三个 bool 列。"""
    cfg = GENIOUS
    out = df.copy()
    out["T1"] = (
        (
            out["kong"]
            & (out["r20"] > 0)
            & (out["winner_ratio"] > cfg["wr_min"])
            & (out["wr_rise60"] > cfg["wr_rise_min"])
            & out["ma10up"]
            & (out["ext10"] <= cfg["t1_ma10_dev_max"])
        )
        .fillna(False)
        .astype(bool)
    )
    out["T2"] = (
        (out["kong10"] & out["duo"] & (out["pct"] > cfg["t2_gain_min"]))
        .fillna(False)
        .astype(bool)
    )
    out["T3"] = (
        (
            (out["gap"] > 0)
            & (out["pb5"] <= cfg["t3_pb5_max"])
            & (out["pct"] > cfg["t3_gain_min"])
            & (out["ext10p"] <= cfg["t3_ma10_dev_max"])
        )
        .fillna(False)
        .astype(bool)
    )
    return out


# 类型分桶边界 (r60 / r120); 单列出来供 sheet1_legend 复用, 免得图例与实算漂移
TYPE_R60_D2 = 0.20
TYPE_R60_D1 = 0.10
TYPE_R60_FLAT = -0.05
TYPE_R120_C = 0.40


def classify_type(df: pd.DataFrame) -> pd.Series:
    """类型标签。重叠时按 优先级 A > C > B2 > B1 > B3 > D1 > D2 定死。

    A 压 C: A 是最强层且冠军段直接用 r60<=-30; C 压 B1: 601869 型 r60≈-14% 归 C;
    B2 压 B3: B2 = B3 ∩ r120<=0 (长平底是浅跌平的子集)。
    """
    cfg = GENIOUS
    r60, r120 = df["r60"], df["r120"]
    deep, base = cfg["r60_deep"], cfg["r120_base"]
    lab = pd.Series(TYPE_UNKNOWN, index=df.index, dtype=object)
    lab[r60 > TYPE_R60_D2] = TYPE_D2
    lab[(r60 > TYPE_R60_D1) & (r60 <= TYPE_R60_D2)] = TYPE_D1
    lab[(r60 >= TYPE_R60_FLAT) & (r60 <= TYPE_R60_D1)] = TYPE_B3
    lab[(r60 > deep) & (r60 < TYPE_R60_FLAT)] = TYPE_B1
    lab[(r120 <= base) & (r60 >= TYPE_R60_FLAT) & (r60 <= TYPE_R60_D1)] = TYPE_B2
    lab[(r120 > TYPE_R120_C) & (r60 <= TYPE_R60_FLAT)] = TYPE_C
    lab[r60 <= deep] = TYPE_A
    lab[r60.isna()] = TYPE_UNKNOWN
    return lab


def assign_layers(df: pd.DataFrame) -> pd.Series:
    """互斥分层: 任一行只属于一层, 高层先占 (Sheet1 四段 > T1余 > 观察带四臂)。

    层名 / 顺序见 ALL_LAYERS。T1 与 T2/T3 天然互斥 (kong 时 gap<0, T3 要 gap>0);
    T2∩T3 重叠归高层 (CH3)。
    """
    cfg = GENIOUS
    r60, r120, vr, pct = df["r60"], df["r120"], df["vr"], df["pct"]
    t1, t2, t3 = df["T1"], df["T2"], df["T3"]
    # 涨停阈值按板分档 (2026-09-18): 科创板/创业板 ±20%, 主板 ±10%。
    # 单值 0.09 在 20% 板上会把普通上涨误判成"涨停" —— 实测 68 的 pct>0.09 有
    # 8,932 行, 其中真涨停 (>=19.5%) 只有 1,484 行, 764 行被错分进带涨停臂。
    # 复用 app.core.price_limits 的板块口径, 不自己写死数字。
    lim = _limit_up_threshold(df["symbol"])

    layer = pd.Series("", index=df.index, dtype=object)
    taken = pd.Series(False, index=df.index)

    def take(mask: pd.Series, name: str) -> None:
        nonlocal taken
        hit = mask.fillna(False).astype(bool) & ~taken
        layer[hit] = name
        taken = taken | hit

    take(t3 & (r60 <= cfg["r60_deep"]) & (vr <= cfg["vr_quiet"]), CH3_T3_DEEP_QUIET)
    take(t2 & (r60 <= cfg["r60_deep"]), CH2_T2_DEEP)
    take(
        t1 & (r120 <= cfg["r120_base"]) & (df["r20"] > cfg["r20_min"]), CH1_T1_LONGBASE
    )
    take(
        t2 & (df["ext10p"] < cfg["t2b_ext_prev_max"]) & (r120 <= cfg["r120_base"]),
        CH2B_T2_STEADY,
    )
    # ★0919 用户令新增: CH4 中线缩量横盘 — "不想要深跌股, 想要中线在涨+短线加速+缩量回调"
    # 的直接落地。域在 r60>+25% 强势区, 与 CH3/CH2 深跌域零重叠 (纯增量层, 不动现有四段)。
    # OOS 全面板 (tmp_t/_mid_final_0919): 569 票 / 期望+6.22% / 胜率49.7% / 日均2.1;
    # 与现役涨闸几乎互斥 (∩ 全年仅 14 票) — 闸作标注列照常记账, 不筛本层。
    # 已知风险: 202607/202609 两月哑火 (-1.7%/-0.7%), 行情依赖。
    take(
        (df["r60"] > cfg["ch4_r60_min"])
        & df["ma10up"].fillna(False)
        & (vr <= cfg["ch4_vr_max"])
        & (df["dd5"] < cfg["ch4_dd5_max"]),
        CH4_TREND_QUIET,
    )
    # T1余 走 **T1宽** (剔 r120>+40% 与 r60<-5%): 不加这两个闸 8.4 → 10.4 票/日, 胜率掉 1.6pp
    take(
        t1 & (r120 <= cfg["t1_wide_r120_max"]) & (r60 >= cfg["t1_wide_r60_min"]),
        T1_REST,
    )

    # 观察带四臂的底是 **T2余 / T3余** 且带宽版三剔毒 (续9 TierC: 剔 r60>+20%/vr>3/昨乖离>1.05),
    # 不是裸全池 — 否则温火臂会吞掉任何 r120<=0 的当日上涨票 (24.5/日 → 140/日)。
    t2_rest = t2 & ~taken
    t3_rest = t3 & ~taken
    quiet = vr <= cfg["band_vr_max"]
    # 外闸分族 (见 settings 注释): T2臂=T2宽 三剔毒; T3臂=T3宽 只剔 r60。
    band2 = (
        (r120 <= cfg["r120_base"])
        & (vr <= cfg["band2_vr_outer_max"])
        & (r60 <= cfg["band_r60_max"])
        & (df["ext10p"] <= cfg["band2_ext10p_max"])
    )
    band3 = r120 <= cfg["r120_base"]
    if cfg["band3_vr_outer_max"] is not None:
        band3 &= vr <= cfg["band3_vr_outer_max"]
    band3 &= r60 <= cfg["band_r60_max"]
    take(
        t2_rest
        & band2
        & quiet
        & (pct > cfg["band_t2_lo"])
        & (pct <= cfg["band_t2_hi"]),
        BAND_T2_WARM,
    )
    take(t2_rest & band2 & (pct > lim), BAND_T2_LIMIT)
    take(
        t3_rest
        & band3
        & quiet
        & (pct > cfg["band_t3_lo"])
        & (pct <= cfg["band_t3_hi"]),
        BAND_T3_WARM,
    )
    take(t3_rest & band3 & (pct > lim), BAND_T3_LIMIT)
    return layer


def trigger_name(df: pd.DataFrame) -> pd.Series:
    """该行的触发器 (T1/T2/T3/T2+T3), 供清单标注。T1 与 T2/T3 天然互斥。"""
    t1 = df["T1"].fillna(False).astype(bool)
    t2 = df["T2"].fillna(False).astype(bool)
    t3 = df["T3"].fillna(False).astype(bool)
    out = pd.Series("", index=df.index, dtype=object)
    out[t2 & t3] = "T2+T3"
    out[t2 & ~t3] = "T2"
    out[t3 & ~t2] = "T3"
    out[t1] = "T1"
    return out


DISPLAY_COLUMNS = (
    "symbol",
    "涨闸",
    "层",
    "触发器",
    "当月样本口径",
    "当日涨幅",
    "执行档",
)


def _zscore(s: pd.Series) -> pd.Series:
    """日内截面 z。std 为 0/NaN → 全 0 (纯中性), 不给排序注入假信号。"""
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


RANK_Z_COLUMNS = ("band20", "ext10", "winner_ratio", "r60", "r120")

# Sheet2 附加观察列 (插在 5日回撤 之后, 即数值块末尾): 把"正→负→正"形态显式写进表里,
# 用户在 Excel 里可自行按它排序; 默认排序键仍是观察分 (实测更强)。
SHEET2_EXTRA_COLUMNS = ("SL翻正年龄", "SL洗盘天数")


def _gate_passed(day: pd.DataFrame) -> pd.Series:
    """ACD 涨闸谓词 (0918 终版): 前置 r5>0 & vr≤dir_gate_vr_max, 档位
    (r10≤0.05 & r5≤0.08) | r10≤0; NaN (历史不足) 计 False, 不静默放行。
    供 build_delivery 与跨模块标注列共用 — 口径必须逐字一致 (含 NaN 语义)。
    """
    pre = (day["r5"] > 0.0) & (day["vr"] <= float(GENIOUS["dir_gate_vr_max"]))
    mom = float(GENIOUS["dir_gate_mom_max"])
    in_tier = ((day["r10"] <= mom) & (day["r5"] <= 0.08)) | (day["r10"] <= 0.0)
    return pre.fillna(False) & in_tier.fillna(False)


def build_delivery(
    df: pd.DataFrame, trade_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """取 trade_date 当日分层结果, 返回 (Sheet1 冠军表, duplicate)。

    0919 用户令: 原观察池独立表删除 — Sheet2/Sheet3 都由全量表承接。
    0919 用户令追加: 加 CH4 中线缩量横盘层 — 全量进冠军表, 与现役四段域零重叠
    (r60>+25% 强势区 vs 深跌区), 闸照常标注记账 (这格与涨闸几乎互斥, ∩仅14票/年)。

    **涨闸** (GENIOUS['dir_gate']) — 0918 终版不分档: 前置 = r5 > 0 (右侧拐头, 不做左侧) &
    vr <= dir_gate_vr_max (缩量), 二值标注 过闸/没过闸。**只标注, 不筛表** (0914 用户令:
    "只在现在的 STOCKLIST 上加上 A+C+D 的标记") —— 三张表一只票都不删, 只多一列「涨闸」;
    Sheet3 另把没过闸者排最前便于复核。NaN (历史不足) 判没过闸, 不静默放行。
    (r10 低动量条件与筹码MA10斜率均已摘除 — r10 档位标注 0918 实测对选择无实际帮助;
    斜率 0914 实测每加一次 ≥20% lift 都掉。两条件旧行为分别保留在 OOS 判词里备用。)

    Sheet2 主键 = **观察分** 高→低 = 日内截面 z 的 -(band20 + ext10 + winner_ratio + r60 + r120),
    即"带宽窄 / 未偏离MA10 / 获利盘低 / 深跌" 越足越靠前。全 896 日实测 IC +0.077 (t 8.7),
    五分位单调 (桶1 +0.65% → 桶5 -0.03%), 前半/后半 0.079/0.076。已涨透的票自动沉底 —
    这就是"滤掉已经发挥完"的机制, 不需要额外的硬闸。

    层序**不作主键**: 层内 IC 实测为负 (-0.025), 按层质量排序反而有害; r60 深→浅只作同分兜底。
    Sheet2 截断到 GENIOUS['sheet2_top_n'] 供阅读, 不截断的全量留在第三张表 (洪峰日的钱不丢)。
    """
    # 主力筹码比例是逐股递归量, 必须在切当日之前对整窗算 (当日值依赖全部历史迁移)
    df = compute_chip_trend(compute_main_chip_ratio(df))
    day = df[df["date"] == _iso(trade_date)].copy()
    day["层"] = assign_layers(day)
    day = day[day["层"] != ""]
    day["触发器"] = trigger_name(day)
    day["类型"] = classify_type(day)
    day["当日涨幅"] = day["pct"]
    day["获利盘"] = day["winner_ratio"]
    day["换手率"] = day["turnover_rate"] / 100.0
    day["量比"] = day["vr"]
    day["乖离MA10"] = day["ext10"]
    day["5日回撤"] = day["pb5"]
    day["执行档"] = day["层"].map(LAYER_EXEC)
    day["当月样本口径"] = day["层"].map(load_layer_research(trade_date))
    day["SL翻正年龄"] = day["sl_flip_age"]
    day["SL洗盘天数"] = day["sl_wash_days"]

    # 0918 用户令终版: 列名「涨闸」, 二值 过闸/没过闸, **不分档标注**。过闸 = 三段
    # 之一 (原现役闸 ∪ 缺口档 ∪ 低动量, 三档胜率都 >70%); 温和 (r10≤0.05&r5>8%, 59%)
    # 与高动量 (r10>0.05, 52%) **不在过闸内**。并集恰 = 拐头+缩量 且 r10≤0.05 且
    # r5≤0.08 或 r10≤0 — 即温和档 (r5>0.08) 被显式踢出。未过前置 (含 NaN 历史不足)
    # 不静默放行仍标没过闸。仍只标注不筛表。
    if GENIOUS["dir_gate"]:
        day["_passed"] = _gate_passed(day)
        day["涨闸"] = np.where(day["_passed"], "过闸", "没过闸")
    else:
        day["_passed"] = True
        day["涨闸"] = "过闸"

    order = {name: i for i, name in enumerate(ALL_LAYERS)}
    day["_layer_rank"] = day["层"].map(order)

    def sheet(sub: pd.DataFrame, extra: tuple = ()) -> pd.DataFrame:
        cols = list(DISPLAY_COLUMNS)
        if extra:
            cols[len(DISPLAY_COLUMNS) : len(DISPLAY_COLUMNS)] = list(extra)
        out = sub[cols].copy()
        out.insert(0, "排名", range(1, len(out) + 1))
        return out.reset_index(drop=True)

    # Sheet1 也不删 — 冠军表全留 + 「涨闸」标注列。层序 / 层内 r60 深→浅。
    s1 = day[day["层"].isin(SHEET1_LAYERS)].sort_values(
        ["_layer_rank", "r60"], ascending=[True, True], na_position="last"
    )
    # 0919 用户令: Sheet1 再加 CH2B ∩ 涨闸过闸 的交集集合 — OOS 里这格 54.1%/
    # +6.01%/大涨22.4% 是观察池最强一格, 提到冠军表。排序仍在其后; CH2B 没过闸
    # 的照旧不出表 (观察池/全量表已删)。
    _s1_extra = day[day["层"].eq(CH2B_T2_STEADY) & day["_passed"]].sort_values(
        "r60", ascending=True, na_position="last"
    )
    # 0919 用户令: CH4 中线缩量横盘全量进冠军表 — "不想要深跌, 要中线在涨+缩量回调"。
    # 与现役四段域零重叠; 闸照常标注记账。
    _s1_ch4 = day[day["层"].eq(CH4_TREND_QUIET)].sort_values(
        "r60", ascending=True, na_position="last"
    )
    s1 = pd.concat([s1, _s1_extra, _s1_ch4], ignore_index=False)

    return (
        sheet(s1),
        sheet(s1, SHEET2_EXTRA_COLUMNS),
    )


def panel_max_date(path) -> datetime.date | None:
    """面板最新数据日 (读失败/空 → None, 绝不静默放行)。"""
    try:
        tbl = pq.read_table(str(path), columns=["date"])
        s = pd.to_datetime(tbl.column("date").to_pandas(), errors="coerce").dropna()
        return None if s.empty else s.max().date()
    except Exception:  # noqa: BLE001 — 读失败统一 None, 由调用方按失败处理
        return None
