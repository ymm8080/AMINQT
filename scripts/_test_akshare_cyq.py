#!/usr/bin/env python3
"""测试 akshare stock_cyq_em 接口: 数据结构、历史深度、字段."""

import time

import akshare as ak

# 1. 测试单只股票
print("=" * 60)
print("测试 akshare stock_cyq_em")
print("=" * 60)

test_codes = ["000001", "600519", "002594"]
for code in test_codes:
    print(f"\n--- {code} ---")
    try:
        t0 = time.time()
        df = ak.stock_cyq_em(symbol=code, adjust="qfq")
        elapsed = time.time() - t0
        if df is None or df.empty:
            print("  返回空")
            continue
        print(f"  耗时: {elapsed:.2f}s, 行数: {len(df)}, 列数: {len(df.columns)}")
        print(f"  列名: {df.columns.tolist()}")
        print(f"  日期范围: {df.iloc[:, 0].min()} ~ {df.iloc[:, 0].max()}")
        print("  前3行:")
        print(df.head(3).to_string())
        print("  后3行:")
        print(df.tail(3).to_string())
    except Exception as e:
        print(f"  失败: {e}")

# 2. 测试速度: 连续拉 10 只
print("\n--- 速度测试: 连续拉 10 只 ---")
test10 = [
    "000001",
    "000002",
    "000063",
    "000100",
    "000333",
    "000425",
    "000538",
    "000651",
    "000661",
    "000725",
]
ok = 0
t0 = time.time()
for code in test10:
    try:
        df = ak.stock_cyq_em(symbol=code, adjust="qfq")
        if df is not None and len(df) > 0:
            ok += 1
    except:  # noqa: E722
        pass
elapsed = time.time() - t0
print(f"  成功 {ok}/10, 总耗时 {elapsed:.1f}s, 平均 {elapsed / 10:.2f}s/股")
print(f"  预计 3244 股: {elapsed / 10 * 3244 / 60:.1f} 分钟")
