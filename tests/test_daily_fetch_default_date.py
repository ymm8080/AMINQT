"""_daily_fetch 缺省取数日 = 最近一个已收盘的开市日 (2026-09-18 补).

事故背景: 计划任务动作不含日期参数, 所以缺省值是无人值守路径的唯一日期来源.
原先用 datetime.now() 取墙钟日, 而 StartWhenAvailable (错过触发即补跑) 恰恰在
"机器睡着" 时于凌晨补跑 —— 09-17 整链哑火就是电量 5% 休眠 16h 吞掉触发点.
凌晨补跑时墙钟日已翻页:

  - 周六凌晨: 非交易日 → Tushare 空 → fail-fast (安全)
  - **周一凌晨: 周一是交易日 → 拿到周一数据 → 把"周一"这行写进面板**,
    而它要等周一收盘后才真实存在 → 此后 panel_max_date 谎报最新日,
    整条链的新鲜度判据 (expected vs panel_max) 全被污染.

★ 关键: 取数日 ≠ freshness_guard.expected_trading_date. 后者语义是"面板此刻
*应该*有的最新数据日", 周一 00:30 返回周一 (对告警正确, 对取数错误 —— 那时
周一还没开盘). 取数要的是"此刻*拿得到*什么", 故须再加"已过收盘"闸.

测试直接 exec _daily_fetch.py 里 _default_trade_date 的**源码函数对象** (不跑
模块体 —— 该脚本导入即连 Tushare 并 sys.exit), 打桩其依赖的两个名字.
"""

from __future__ import annotations

import ast
import datetime
import pathlib

import pandas as pd
import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "_daily_fetch.py"

# 2026-09-18 是周五; 19/20 是周末, 21 是周一.
FRI = datetime.date(2026, 9, 18)
SAT = datetime.date(2026, 9, 19)
MON = datetime.date(2026, 9, 21)

CAL = pd.DatetimeIndex([FRI, MON])  # load_trade_cal 返回"仅开市日"


class _Log:
    def info(self, *a, **k):  # noqa: ARG002 — 吞掉日志
        pass


def _load_fn():
    """抽出 _default_trade_date 的函数对象 (不执行模块体)。

    ★ 用**真实模块的导入语句**建命名空间再 exec 函数定义 —— 这样注解若在导入期
    就会炸 (如 `datetime | None` 而 datetime 是模块), 测试同样炸。曾用假命名空间
    {"datetime": <class>} 喂进去, 把真实的 TypeError 掩盖过去了 (2026-09-18 实撞)。
    """
    tree = ast.parse(SRC.read_text(encoding="utf-8-sig"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_default_trade_date"
    )
    # 只取模块顶部的 import 语句, 跳过其余模块级副作用 (网络/sys.exit)
    imports = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(SRC), "exec"), ns)  # noqa: S102
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)  # noqa: S102
    return ns["_default_trade_date"]


def _run(monkeypatch, now: datetime.datetime, cal=CAL) -> str:
    """打桩 load_trade_cal, 把"此刻"直接喂给被测函数."""
    import app.pipeline1.freshness_guard as fg

    monkeypatch.setattr(fg, "load_trade_cal", lambda: cal)

    fn = _load_fn()
    fn.__globals__["log"] = _Log()
    fn.__globals__["MARKET_CLOSE_HOUR"] = 16  # 与 config.settings 同值
    return fn(now)


@pytest.mark.parametrize(
    "now, want, why",
    [
        (datetime.datetime(2026, 9, 18, 19, 15), "20260918", "周五夜链正常时刻 → 周五"),
        (
            datetime.datetime(2026, 9, 18, 9, 0),
            "20260917",
            "周五开市前 → 退到周四(自然日回退)",
        ),
        (datetime.datetime(2026, 9, 19, 2, 0), "20260918", "★周六凌晨补跑 → 回退周五"),
        (
            datetime.datetime(2026, 9, 21, 0, 30),
            "20260918",
            "★周一凌晨补跑 → 回退周五, 不是周一",
        ),
        (datetime.datetime(2026, 9, 21, 20, 0), "20260921", "周一夜链正常时刻 → 周一"),
    ],
)
def test_default_trade_date(monkeypatch, now, want, why):
    assert _run(monkeypatch, now) == want, why


def test_monday_wallclock_negative_control(monkeypatch):
    """负控: 墙钟日路线在周一凌晨会选到周一 —— 即被修掉的那个 bug."""
    monday_early = datetime.datetime(2026, 9, 21, 0, 30)
    assert monday_early.strftime("%Y%m%d") == "20260921"  # 旧路线取这个
    assert _run(monkeypatch, monday_early) == "20260918"  # 修后取这个


def test_cal_missing_still_rolls_back(monkeypatch):
    """cal 拿不到 (Tushare 挂) → 自然工作日回退, 周一凌晨仍须退到周五."""
    assert (
        _run(monkeypatch, datetime.datetime(2026, 9, 21, 0, 30), cal=None) == "20260918"
    )
