# Daily Market Pipeline - 20260728

- **Trigger**: 2026-07-29 08:32:18
- **Total elapsed**: 16976.4s

| Source | Description | Rows | Status | Note |
|--------|-------------|------|--------|------|
| ohlcv | OHLCV 日线 | 5524 | ok | 5524 rows in 4.2s |
| daily_basic | 每日估值 (PE/PB/换手/市值) | 5524 | ok | 5524 rows in 2.3s |
| stk_limit | 涨跌停价格 | 7710 | ok | 7710 rows in 3.1s |
| margin | 融资融券 | 4417 | ok | 4417 rows in 2.4s |
| northbound | 北向资金 | 1 | ok | 1 rows in 31.1s |
| lhb | 龙虎榜 | 87 | ok | 87 rows in 3.5s |
| sector_index | 申万行业指数 | 0 | empty | 0 rows in 16929.9s |
| cyq_tushare | 筹码分布 (Tushare cyq_perf) | 0 | fail | HTTPConnectionPool(host='api.waditu.com', port=80): Max retries exceeded with url: /dataapi/cyq_perf (Caused by NameResolutionError("HTTPConnection(host='api.waditu.com', port=80): Failed to resolve 'api.waditu.com' ([Errno 11004] getaddrinfo failed)")) (0.0s) |
