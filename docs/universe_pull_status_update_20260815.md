# V3 宇宙扩建进度更新 (2026-08-15 23:00)

## 已完成的
- ✅ 基础数据拉取（daily, basic, adj_factor, stk_limit, suspend）：约 500/876 天 → 后台拉进中
- ✅ CYQ 筹码分布 18 列：合并完毕
- ✅ 另类数据合并（margin, lhb, bt）：合并完毕
- ✅ 新面板构建：panel_alt.parquet
- ✅ QC 检查：OHLCV 一致 100%；资金量非负 100%

## 待拉
- daily/basic/adj_factor/stk_limit/suspend: 376 天 (20250124..20260814)
  - 后台 _pull_select_sources.py 进行中
- margin/lhb/bt/cyq: 同样，需要从 20250124 拉至 end
- top_inst/fina: 未开始
- holdertrade/sw: 未开始

## 输出
- panel_alt.parquet (1290231 rows, 1684 新符号)
- 覆盖 500 天 (57% of 876)
- coverage min=1 max=804 mean=766.2

## 后续
1. 等后台拉完剩余 376 天
2. 再跑 QC 全量
3. 补 top_inst/fina/holdertrade/sw
4. 整合为新 universe 面板（统一符号/日期）