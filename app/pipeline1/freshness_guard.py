"""特征族新鲜度守卫 (2026-09-02).

事故背景 (2026-09-02 审计结论): 历史上 4 起"特征静默停更"全是事后数周才发现 —
  1. cyq_panel 停更@07-17 (pct_70_con 慢牛列静默断供, 08-19 才修);
  2. sw_daily_history 冻结@07-31 (dim28 上游 39 列死 1 个月, 09-02 才补挂日更);
  3. 面板 announce_date 冻结@08-14 (dim31, 用户暂缓修复);
  4. 面板 fina 列冻结 (财报季 4950 股只换血 15 只) + sw_ret_1d 在 08-27/09-01
     整列 NaN.
全链原只有 3 个点状检查, 无系统性新鲜度闸. 本模块 = 注册表驱动的全族新鲜度守卫
(防再犯的兜底), 纯判定函数与 IO 分离 (风格对齐 ram_guard.py / scripts/_run_guard.py):
单测纯注入不碰真数据, IO 辅助单独可测.

三种 kind (注册表 config/freshness_registry.yaml):
  file          — 单文件水位: date_col 最大值, 交易日 lag <= max_lag_days
  panel_columns — 面板列族: 回看 lookback_days 个交易日内, 任一日族内各列
                  非空数最小值 >= min_nonnull 即健康 (行照常日更、列静默全 NaN
                  是 4 起事故里最常见的形态, 纯 file 水位盖不住)
  dir_watermark — 目录水位: 文件名 pattern 首个捕获组 (YYYYMMDD) 取 max

告警式不阻断 (scripts/_freshness_check.py): 08-27 教训 — refresh 卡死全天零清单,
链对任何步骤失败一视同仁, 新鲜度告警绝不能走到阻断交付.
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field

import pandas as pd
import pyarrow.parquet as pq
import yaml

# 相对路径锚定仓库根 (app/pipeline1/ 上 2 层), 与 CWD 无关
# (risk_overlays.py 教训: 曾按 CWD 解析 → CWD != repo root 时读不到).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_KINDS = ("file", "panel_columns", "dir_watermark")
# 各 kind 必备键 (缺一即注册表写错, 启动时报大声而非跑起来静默漏检)
_REQUIRED_KEYS = {
    "file": ("name", "kind", "path", "date_col", "max_lag_days"),
    "panel_columns": ("name", "kind", "path", "columns", "lookback_days", "min_nonnull"),
    "dir_watermark": ("name", "kind", "path", "pattern", "max_lag_days"),
}


# ── 注册表加载 ───────────────────────────────────────────────────────────────


def load_registry(path: str) -> list[dict]:
    """加载并校验新鲜度注册表. 缺必备键 / 未知 kind → ValueError (带条目名).

    顶层为条目列表, 或 {"entries": [...]} (允许 yaml 头部放说明注释块).
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    if not isinstance(data, list):
        raise ValueError(f"新鲜度注册表须为条目列表: {path}")
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"新鲜度注册表条目须为映射: {entry!r}")
        name = entry.get("name")
        kind = entry.get("kind")
        if not name:
            raise ValueError(f"新鲜度注册表条目缺 name: {entry!r}")
        if kind not in _KINDS:
            raise ValueError(
                f"新鲜度注册表条目 {name} 未知 kind: {kind!r} (合法: {list(_KINDS)})"
            )
        for key in _REQUIRED_KEYS[kind]:
            if entry.get(key) is None:
                raise ValueError(f"新鲜度注册表条目 {name} 缺必备键: {key}")
        if kind == "panel_columns":
            cols = entry["columns"]
            if not isinstance(cols, list) or not cols:
                raise ValueError(f"新鲜度注册表条目 {name} columns 须为非空列表")
    return [dict(e) for e in data]


# ── 纯判定: 交易日历 / 期望日 / lag ─────────────────────────────────────────


def expected_trading_date(today, cal) -> tuple[datetime.date, str]:
    """期望的最新数据日 = cal 中 <=today 的最大开市日.

    cal None/空 (Tushare 挂了) → 回退"最近工作日(周一~五)", source="natural_fallback";
    正常回溯 source="trade_cal". 回退已知偏差: 不含法定节假日, 假日后会高估期望日 →
    判定偏严 (误报好过漏报, 且只是告警).
    """
    today = pd.Timestamp(today)
    if cal is not None and len(cal) > 0:
        cal_idx = pd.DatetimeIndex(cal)
        past = cal_idx[cal_idx <= today]
        if len(past) > 0:
            return past[-1].date(), "trade_cal"
    d = today.date()
    while d.weekday() >= 5:  # A股周末不开市, 回退到周五
        d -= datetime.timedelta(days=1)
    return d, "natural_fallback"


def lag_trading_days(observed, expected, cal) -> int | None:
    """observed → expected 的交易日距离 (expected 更新为正).

    cal None/空或两日不在 cal 覆盖内 (非交易日/超范围) → None, 调用方回退自然日.
    """
    if cal is None or len(cal) == 0:
        return None
    cal_idx = pd.DatetimeIndex(cal)
    pos = cal_idx.get_indexer([pd.Timestamp(observed), pd.Timestamp(expected)])
    if -1 in pos:
        return None
    return int(pos[1] - pos[0])


def _as_date(d) -> datetime.date:
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    return pd.Timestamp(d).date()


def _violation(entry: dict, observed, expected, lag, threshold, detail: str) -> dict:
    return {
        "name": entry["name"],
        "kind": entry["kind"],
        "observed": _as_date(observed).isoformat() if observed is not None else None,
        "expected": _as_date(expected).isoformat(),
        "lag": lag,
        "threshold": threshold,
        "critical": bool(entry.get("critical", False)),
        "detail": detail,
    }


def check_file_entry(entry: dict, observed_max, expected, cal) -> dict | None:
    """file 条目判定. 违规返回违规 dict, 健康返回 None.

    observed_max None (读失败/空列) 也判违规 — 读失败绝不静默放行, 这正是旧链级
    面板闸 (2026-09-02 前的 except-pass) 的漏洞. cal 不可用时回退自然日 lag,
    自然日阈值放宽 +2 (周末缓冲: 周一跑链面板停周五=自然日3, 交易日 lag 只 1).
    """
    threshold = int(entry["max_lag_days"])
    if observed_max is None:
        return _violation(
            entry, None, expected, None, threshold, "read_failed (读失败/空列, 绝不静默放行)"
        )
    obs = _as_date(observed_max)
    exp = _as_date(expected)
    lag = lag_trading_days(obs, exp, cal)
    if lag is None:  # cal 不可用/覆盖外 → 自然日回退 (周末缓冲 +2)
        lag = (exp - obs).days
        if lag > threshold + 2:
            return _violation(
                entry, obs, exp, lag, threshold,
                f"自然日落后 {lag} 天 (交易日历不可用, 阈值 {threshold}+2 周末缓冲)",
            )
        return None
    if lag > threshold:
        return _violation(
            entry, obs, exp, lag, threshold, f"交易日落后 {lag} 天 (阈值 {threshold})"
        )
    return None


def _window_dates(expected, cal, lookback: int) -> list[datetime.date]:
    """回看窗口: expected 往前 lookback 个观察点 (交易日优先, cal 不可用回退自然日), 升序."""
    exp = pd.Timestamp(expected)
    if cal is not None and len(cal) > 0:
        cal_idx = pd.DatetimeIndex(cal)
        past = cal_idx[cal_idx <= exp]
        if len(past) > 0:
            return [d.date() for d in past[-lookback:]]
    return [(exp - pd.Timedelta(days=i)).date() for i in reversed(range(lookback))]


def check_columns_entry(entry: dict, daily_nonnull, expected, cal) -> dict | None:
    """panel_columns 条目判定. daily_nonnull: [(date, count)] — count = 该日族内
    各列非空数的最小值 (族健康须各列都有数).

    回看窗口内任一日 count >= min_nonnull 即健康; 窗口内无任何达标日 (含列表为空
    = 面板在窗口内无行/列全 NaN) 判违规 — 08-27 sw_ret_1d 整列 NaN 就是这个形态.
    """
    threshold = int(entry["min_nonnull"])
    window = _window_dates(expected, cal, int(entry["lookback_days"]))
    wset = set(window)
    counts = {d: c for d, c in daily_nonnull if d in wset}
    best_day = max((d for d in window if counts.get(d, 0) >= threshold), default=None)
    if best_day is not None:
        return None
    best_count = max((counts.get(d, 0) for d in window), default=0)
    return _violation(
        entry, best_day, expected, None, threshold,
        f"回看 {len(window)} 个观察日 ({window[0]}..{window[-1]}) 内族内各列非空数"
        f"最小值最高仅 {best_count}, 阈值 {threshold} (行在列死/全 NaN 形态)",
    )


def check_watermark_entry(entry: dict, watermark_date, expected, cal) -> dict | None:
    """dir_watermark 条目判定. watermark_date None (目录缺失/无匹配文件) 也判违规."""
    threshold = int(entry["max_lag_days"])
    if watermark_date is None:
        return _violation(
            entry, None, expected, None, threshold,
            "read_failed (目录缺失/无匹配文件, 绝不静默放行)",
        )
    return check_file_entry(entry, watermark_date, expected, cal)


# ── panel_stale_gate: 链级 A1 闸复用的纯判定 ────────────────────────────────

# 阈值常量 (2026-09-02 集中到判定处, 不再硬编码在链里):
#   交易日 3 / 自然日 4 — 旧闸漏洞回指: "(today-pmax).days > 3" 自然日阈值下,
#   周一跑链面板停周五 = 自然日 3 恰好放行, 周二周三才拦 (滞后 1-2 天);
#   交易日口径下周一 lag=1 正常放行, 周四 lag=4 才拦, 边界一致.
_PANEL_MAX_LAG_TRADING = 3
_PANEL_MAX_LAG_NATURAL = 4


def panel_stale_gate(pmax, today, cal) -> tuple[bool, str]:
    """V3 面板新鲜度链级判定. 返回 (是否放行, 原因串).

    有交易日历: 交易日 lag > _PANEL_MAX_LAG_TRADING 拦截;
    cal 不可用 (Tushare 挂): 自然日 > _PANEL_MAX_LAG_NATURAL 拦截;
    pmax None (读失败): 不放行 — 读失败≠数据新鲜, 旧闸的 except-pass 漏洞 (09-02 修).
    """
    if pmax is None:
        return False, "面板最新日期不可读 (observed=None)"
    pmax_d = _as_date(pmax)
    expected, cal_source = expected_trading_date(today, cal)
    lag = lag_trading_days(pmax_d, expected, cal)
    if lag is not None:
        if lag > _PANEL_MAX_LAG_TRADING:
            return (
                False,
                f"V3 面板最新 {pmax_d} 交易日落后 {lag} 天 "
                f"(>{_PANEL_MAX_LAG_TRADING}, cal_source={cal_source})",
            )
        return True, f"V3 面板最新 {pmax_d} 交易日 lag={lag} (cal_source={cal_source})"
    natural = (pd.Timestamp(today).normalize() - pd.Timestamp(pmax_d).normalize()).days
    if natural > _PANEL_MAX_LAG_NATURAL:
        return (
            False,
            f"V3 面板最新 {pmax_d} 自然日落后 {natural} 天 "
            f"(>{_PANEL_MAX_LAG_NATURAL}, 交易日历不可用回退自然日)",
        )
    return (
        True,
        f"V3 面板最新 {pmax_d} 自然日 lag={natural} (交易日历不可用回退自然日)",
    )


# ── IO 辅助 (与纯判定分离, 单测用伪 IO 注入) ────────────────────────────────


def file_max_date(path, date_col: str):
    """pyarrow 只读 date_col 一列取 max; 任何读失败/空列 → None (绝不静默放行)."""
    try:
        tbl = pq.read_table(str(path), columns=[date_col])
        s = pd.to_datetime(tbl.column(date_col).to_pandas(), errors="coerce").dropna()
        if s.empty:
            return None
        return s.max().date()
    except Exception:  # noqa: BLE001 — 读失败统一返回 None, 判定层必须当违规处理
        return None


def panel_column_daily_nonnull(path, columns, cal, tail_days, date_col: str = "date"):
    """逐日统计族内各列非空数 (向量化). 返回 [(date, min_count_across_cols)] 升序.

    先读 date 列定尾部窗口 (文件 max date 往前 tail_days 自然日; cal 可用时窗口
    收窄到交易日跨度, 少读行), 再带 filters 读目标列, groupby 逐日 count.
    读失败/无行 → 返回 [] (判定层把空列表当违规 — 窗口内无达标日).
    """
    try:
        dmax = file_max_date(path, date_col)
        if dmax is None:
            return []
        start = pd.Timestamp(dmax) - pd.Timedelta(days=int(tail_days))
        if cal is not None and len(cal) > 0:
            cal_idx = pd.DatetimeIndex(cal)
            past = cal_idx[cal_idx <= pd.Timestamp(dmax)]
            if len(past) > 0:  # 交易日跨度 ≤ 自然日跨度 → 用更晚的窗口起点少读行
                w_start = past[-int(tail_days):]
                start = pd.Timestamp(w_start[0])
        tbl = pq.read_table(
            str(path),
            columns=[date_col, *columns],
            filters=[(date_col, ">=", start)],
        )
        df = tbl.to_pandas()
        if df.empty:
            return []
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        # groupby 逐日非空计数, 再取族内各列最小值 (族健康须各列都有数)
        counts = df.groupby(df[date_col].dt.normalize())[list(columns)].count()
        daily_min = counts.min(axis=1)
        return sorted((ts.date(), int(v)) for ts, v in daily_min.items())
    except Exception:  # noqa: BLE001 — 读失败返回 [], 判定层当违规
        return []


def dir_watermark(path, pattern: str):
    """目录内文件名 pattern 首个捕获组 (YYYYMMDD) 取 max 为水位日期.

    目录缺失/无匹配/读失败 → None (判定层当违规).
    """
    try:
        rx = re.compile(pattern)
        best = None
        for fn in os.listdir(str(path)):
            m = rx.search(fn)
            if not m:
                continue
            d = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
            if pd.notna(d) and (best is None or d > best):
                best = d
        return best.date() if best is not None else None
    except Exception:  # noqa: BLE001
        return None


def load_trade_cal():
    """Tushare 交易日历 (仅开市日). 复用 data_supply 的缓存实现, 失败返回 None 不抛
    — 调用方 (expected_trading_date / panel_stale_gate) 自动回退自然日."""
    try:
        from app.pipeline1.data_supply import DataSupplyChain

        chain = DataSupplyChain()
        pro = chain._tushare_pro()
        if pro is None:
            return None
        return chain._get_trade_cal_cached(pro)
    except Exception:  # noqa: BLE001 — 日历拿不到只降级, 不拦告警主流程
        return None


# ── 编排 ────────────────────────────────────────────────────────────────────


@dataclass
class FreshnessIO:
    """IO 接口束 — 单测注入伪实现 (不碰真数据), 生产用 _REAL_IO."""

    file_max_date: object
    panel_column_daily_nonnull: object
    dir_watermark: object


def _real_io() -> FreshnessIO:
    return FreshnessIO(
        file_max_date=file_max_date,
        panel_column_daily_nonnull=panel_column_daily_nonnull,
        dir_watermark=dir_watermark,
    )


@dataclass
class CheckResult:
    """run_checks 编排结果: violations + skipped + 健康条目观测值 (CLI 报告用)."""

    violations: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)

    @property
    def has_critical_violation(self) -> bool:
        return any(v["critical"] for v in self.violations)


def _resolve_path(p: str) -> str:
    return p if os.path.isabs(str(p)) else os.path.join(_REPO_ROOT, str(p))


def run_checks(registry, expected, cal, *, io_impl: FreshnessIO | None = None) -> CheckResult:
    """编排全部条目. enabled=false 跳过并记入 skipped; 其余逐条判定.

    io_impl 可注入伪 IO (单测不碰真数据); None 用真实 IO.
    """
    io = io_impl or _real_io()
    result = CheckResult()
    for entry in registry:
        name = entry["name"]
        kind = entry["kind"]
        if not entry.get("enabled", True):
            result.skipped.append({"name": name, "reason": "enabled=false"})
            continue
        if kind == "file":
            observed = io.file_max_date(_resolve_path(entry["path"]), entry["date_col"])
            violation = check_file_entry(entry, observed, expected, cal)
            obs = {
                "name": name,
                "kind": kind,
                "observed": _as_date(observed).isoformat() if observed else None,
                "threshold": int(entry["max_lag_days"]),
                "critical": bool(entry.get("critical", False)),
            }
        elif kind == "dir_watermark":
            observed = io.dir_watermark(_resolve_path(entry["path"]), entry["pattern"])
            violation = check_watermark_entry(entry, observed, expected, cal)
            obs = {
                "name": name,
                "kind": kind,
                "observed": _as_date(observed).isoformat() if observed else None,
                "threshold": int(entry["max_lag_days"]),
                "critical": bool(entry.get("critical", False)),
            }
        else:  # panel_columns
            lookback = int(entry["lookback_days"])
            # io 尾部窗口 = 回看窗口的自然日跨度 + 3 缓冲 (文件 max 可能略超 expected)
            window = _window_dates(expected, cal, lookback)
            tail_days = (pd.Timestamp(expected) - pd.Timestamp(window[0])).days + 3
            counts = io.panel_column_daily_nonnull(
                _resolve_path(entry["path"]),
                list(entry["columns"]),
                cal,
                tail_days,
                entry.get("date_col", "date"),
            )
            violation = check_columns_entry(entry, counts, expected, cal)
            in_window = [c for d, c in counts if d in set(window)]
            obs = {
                "name": name,
                "kind": kind,
                "observed": max(in_window) if in_window else None,
                "threshold": int(entry["min_nonnull"]),
                "critical": bool(entry.get("critical", False)),
            }
        if violation is not None:
            result.violations.append(violation)
        else:
            obs["status"] = "ok"
            result.observations.append(obs)
    return result
