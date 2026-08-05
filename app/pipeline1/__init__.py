"""Legacy 生产选股模块 (2026-08-04 命名为 "legacy").

= 旧版生产链路 = 每日 16:00 选股主循环 (DailySelectionPipeline),
与 app/pipeline_parallel/ 的新并行系统 (sniper/fusion/third) 区分.

调用入口:
  python scripts/run_daily.py           # 每日选股 (数据拉取→清洗→特征→推理→清单)
  python scripts/select_features.py --board main --update --dry-run  # 特征选择
  python scripts/build_features.py --board main                     # Layer1 特征
  python app/main.py                    # FastAPI + APScheduler 定时

本模块是生产权威路径 (DESIGN V1.5 §14 / 实施计划 P14), 不得被并行系统修改;
并行系统只读数据/配置, 不 import 本模块的选股/训练逻辑.
"""

PIPELINE_NAME = "legacy"
PIPELINE_LABEL = "Legacy 生产链路 (V3.5)"
PIPELINE_DESC = "每日 16:00 选股: 拉取→清洗→特征→推理→校准→清单, 历史权威路径"
