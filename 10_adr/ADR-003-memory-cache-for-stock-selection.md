# ADR-003: 内存缓存全市场数据用于选股

**状态**: Accepted  
**日期**: 2026-07-16

## 背景 (Context)

Phase 4 的 `/api/v1/select` 接口需要遍历全市场（或大股票池）进行选股。如果每次请求都逐个读取 CSV 文件：
- I/O 瓶颈：数百个文件读取耗时数十秒
- 响应延迟：用户体验差，定时任务可能超时
- 重复计算：同一文件被反复读取和解析

## 决策 (Decision)

在 FastAPI **启动时**将所有 CSV 数据**一次性加载**到内存字典 `Dict[str, pd.DataFrame]`：
- 启动时: 遍历 `data/raw/*.csv`，解析后存入 `DATA_CACHE`
- 选股时: 直接从 `DATA_CACHE[code]` 读取，无 I/O
- 目标响应时间: < 5 秒

## 备选方案 (Alternatives Considered)

1. **逐文件读取**: 每次 `/select` 请求逐个读取 CSV — 太慢，放弃
2. **SQLite 缓存**: 将 CSV 导入 SQLite — 增加依赖，查询仍比内存慢 — 备用
3. **Redis 缓存**: DataFrame 序列化到 Redis — 序列化开销大，过度设计 — 放弃
4. **内存字典 (选中)**: 启动时全量加载，O(1) 读取 — 简单高效

## 影响 (Consequences)

### 正面影响
- 选股响应时间从数十秒降到 < 5 秒
- 无额外依赖（纯 Python dict）
- DataFrame 在内存中可直接计算，无需反序列化

### 负面影响
- 启动时间变长（取决于股票池大小，5 只约 1 秒，全市场约 30 秒）
- 内存占用增加（每只股票约 200KB，全市场约 100MB）
- 数据更新后需重启服务或手动刷新缓存

### 缓解措施
- 提供缓存刷新接口: `POST /api/v1/cache/refresh`
- 日终数据更新后自动刷新（APScheduler 定时任务）
- 内存占用监控: 启动时打印 `DATA_CACHE` 总大小

## 合规要求 (Compliance)

- 缓存位置: `app/main.py` startup 事件
- 缓存结构: `DATA_CACHE: Dict[str, pd.DataFrame]`
- 刷新接口: `POST /api/v1/cache/refresh`
- 性能标准: `/select` 响应 < 5 秒

## 参考 (References)

- PROMPT_CONTENT §2.C 全市场遍历性能（防止卡死）
- IMPLEMENTATION PLAN Phase 4
