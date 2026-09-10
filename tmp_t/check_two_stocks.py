# -*- coding: utf-8 -*-
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

CODES = sys.argv[1:] or ["300857", "301086"]

# ---------- 1) 实时盘口 ----------
live = {}
try:
    import akshare as ak
    for code in CODES:
        df = ak.stock_bid_ask_em(symbol=code)
        d = dict(zip(df["item"], df["value"]))
        live[code] = d
        print(f"== {code} 实时 ==")
        for k in ["最新价", "涨跌幅", "今开", "最高", "最低", "昨收", "成交量", "成交额", "换手", "量比", "总市值", "流通市值"]:
            if k in d:
                print(f"  {k}: {d[k]}")
except Exception as e:
    print("akshare failed:", repr(e))
    try:
        import tushare as ts
        df = ts.realtime_quote(ts_code=",".join(c + ".SZ" for c in CODES))
        print(df.to_string())
    except Exception as e2:
        print("tushare rt failed:", repr(e2))

# ---------- 2) 面板近期走势(截至最近入库交易日) ----------
try:
    import pandas as pd
    import pyarrow.parquet as pq
    PATH = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
    sch = pq.read_schema(PATH).names
    symcol = next((c for c in ["symbol", "ts_code", "code"] if c in sch), None)
    datecol = next((c for c in ["trade_date", "date", "ds"] if c in sch), None)
    want = [symcol, datecol] + [c for c in ["open", "high", "low", "close", "vol", "volume", "amount", "pct_chg"] if c in sch]
    df = pq.read_table(PATH, columns=want).to_pandas()
    print("\npanel max date:", df[datecol].max())
    for code in CODES:
        sub = df[df[symcol].astype(str).str.contains(code)].sort_values(datecol)
        if len(sub) == 0:
            print(f"\n== {code} 不在V3宇宙 ==")
            continue
        c = sub["close"].astype(float)
        sub = sub.assign(
            r1=c.pct_change(),
            r5=c.pct_change(5), r10=c.pct_change(10), r20=c.pct_change(20),
            ma5=c.rolling(5).mean(), ma10=c.rolling(10).mean(), ma20=c.rolling(20).mean(), ma60=c.rolling(60).mean(),
        )
        vol = sub["vol"] if "vol" in sub else sub["volume"]
        sub = sub.assign(volratio=vol.astype(float) / vol.rolling(20).mean().astype(float))
        last = sub.iloc[-1]
        hi60 = c.rolling(60).max().iloc[-1]
        print(f"\n== {code} 面板尾部(截至{last[datecol]}) ==")
        cols = [datecol, "open", "high", "low", "close", "r1", "volratio"]
        print(sub[cols].tail(10).to_string(index=False))
        print(f"  r5={last['r5']:.2%} r10={last['r10']:.2%} r20={last['r20']:.2%}")
        print(f"  close vs ma5/ma10/ma20/ma60: {last['close']/last['ma5']-1:+.1%} / {last['close']/last['ma10']-1:+.1%} / {last['close']/last['ma20']-1:+.1%} / {last['close']/last['ma60']-1:+.1%}")
        print(f"  距60日高点: {last['close']/hi60-1:+.1%}  当日量比(vs 20日均量): {last['volratio']:.2f}")
except Exception as e:
    print("panel failed:", repr(e))
