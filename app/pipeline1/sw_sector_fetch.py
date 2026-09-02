"""申万行业指数拉取 — 纯逻辑抽离自 _daily_fetch.py (可单测).

2026-09-02 事故: _daily_fetch sw 段旧逻辑拉取失败只 print FAILED + continue,
全空只打一行 WARN — 2026-08-27 与 09-01 两天全市场 sw_ret_1d/sw_index_close/
sw_index_vol 整列 NaN/0, 无任何拦截 (下游是行业动量类特征).

本模块把"逐指数重试 + 完整性判定"抽成纯函数, 由 _daily_fetch 调用并据
missing_industries() 结果大声失败 (sys.exit(1), 在面板追加写之前 → 宁可放弃
当日整次 fetch 也不写 sw 列半空的半拉子面板).
"""
import time


def fetch_sw_sector_map(
    fetch_one, ind2code, present_inds, trade_date, attempts=3, delay=2.0
):
    """逐行业拉取申万指数日行情, 每个失败代码重试 attempts 次.

    Args:
        fetch_one: 可注入拉取函数 (ts_code, start_date, end_date) -> DataFrame
            (生产传 pro.index_daily, 测试传伪函数; 单位换算见下方注释).
        ind2code: 行业名 → 申万代码 (如 '电气设备' → '801730' 老申万名别名).
        present_inds: 今日面板实际出现的行业名集合.
        trade_date: 'YYYYMMDD'.
        attempts: 每个指数的最大尝试次数 (Tushare 限流/超时是瞬时的).
        delay: 重试间隔秒 (测试注入 delay=0 不等待).

    Returns:
        dict {行业名: {sw_ret_1d, sw_index_close, sw_index_vol}}, 只含成功项 —
        调用方用 missing_industries() 判完整性, 缺失即 fail-fast.
    """
    sw_map = {}
    for ind in sorted(present_inds):
        code = ind2code[ind]
        idx = None
        for attempt in range(1, attempts + 1):
            try:
                idx = fetch_one(code + ".SI", trade_date, trade_date)
            except Exception as e:
                print(f"    sw {code} ({ind}) {trade_date}: attempt "
                      f"{attempt}/{attempts} FAILED ({e})")
                idx = None
            if idx is not None and len(idx):
                break
            if idx is not None and not len(idx):
                print(f"    sw {code} ({ind}) {trade_date}: attempt "
                      f"{attempt}/{attempts} empty")
                idx = None
            if attempt < attempts:
                time.sleep(delay)
        if idx is None:
            print(f"    sw {code} ({ind}) {trade_date}: giving up after "
                  f"{attempts} attempts")
            continue
        # 单位换算 (实测校验, 与 _daily_fetch 原逻辑一致):
        # sw_ret_1d = pct_chg/100 (小数), sw_index_close = close,
        # sw_index_vol = vol/1e6 (面板单位=百万手, Tushare vol 单位=手).
        r = idx.iloc[0]
        sw_map[ind] = {
            "sw_ret_1d": float(r["pct_chg"]) / 100.0,
            "sw_index_close": float(r["close"]),
            "sw_index_vol": round(float(r["vol"]) / 1e6, 2),
        }
    return sw_map


def missing_industries(present_inds, sw_map):
    """完整性判定: 返回缺失行业名 sorted 列表 (非空 → 调用方应 [FATAL] exit)."""
    return sorted(set(present_inds) - set(sw_map))
