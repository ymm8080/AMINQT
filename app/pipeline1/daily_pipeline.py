# -*- coding: utf-8 -*-
"""
Pipeline-1 每日选股编排 (P14 端到端, 生产主循环)
=====================================================
执行时序 (V3.5 月度重训解耦):
  15:00 前   数据拉取 (DataSupplyChain, 失败 → 三档降级)
  15:30      每月第一个交易日启动重训 (T-1 数据, 与清单生成并行)
  16:00      用当前模型生成当日清单 (绝不让重训阻塞清单)
  18:00 前   重训完成, OOS 合格切换, 次日生效

日流程: 拉取 → 清洗 0→4 → 特征 → 推理 → 校准 → 清单 schema V1.0 →
        Holding Bonus 回填 → 空仓触发 → 持久化 → 降级守卫.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from .cleaning_pipeline import CleaningPipeline
from .data_supply import DataSupplyChain, DataSupplyError
from .dual_track_trainer import DualTrackTrainer
from .feature_engine_v35 import FeatureEngineV35
from .label_engine import LabelEngine
from .list_generator import ListDeliveryGuard, ListGenerator, MarketEnv
from .predictor import V35Predictor

logger = logging.getLogger(__name__)


class DailySelectionPipeline:
    """每日 16:00 选股主循环.

    Args:
        supply:    DataSupplyChain (生产 akshare / 测试 mock fetcher)
        bundle_paths: {'main': path, 'dual': path} 当前生效模型包
        list_dir:  清单持久化目录
        float_shares_map: {symbol: 流通股本} (dim09 筹码分布, 可选)
    """

    def __init__(self, supply: DataSupplyChain, bundle_paths: dict[str, str],
                 list_dir: str = "data/lists", float_shares_map: dict | None = None):
        self.supply = supply
        self.predictor = V35Predictor(bundle_paths)
        self.cleaner = CleaningPipeline()
        self.features = FeatureEngineV35()
        self.lister = ListGenerator()
        self.guard = ListDeliveryGuard()
        self.list_dir = list_dir
        self.float_shares_map = float_shares_map
        os.makedirs(list_dir, exist_ok=True)

    # ---------------- 清单持久化 ----------------
    def _list_path(self, trade_date: str) -> str:
        return os.path.join(self.list_dir, f"list_{trade_date}.parquet")

    def load_list(self, trade_date: str) -> pd.DataFrame | None:
        path = self._list_path(str(trade_date).replace("-", ""))
        return pd.read_parquet(path) if os.path.exists(path) else None

    # ---------------- 主流程 ----------------
    def run(self, trade_date: str, panel: pd.DataFrame | None = None,
            env: MarketEnv | None = None, market_state: str = "range") -> dict:
        """生成当日清单.

        Args:
            trade_date: 'YYYYMMDD'
            panel: 全市场历史面板 (含当日). None → 由 supply 装配 (生产路径)
            env: 大盘环境 (D18 空仓触发); None → supply.fetch_market_sentiment

        Returns:
            {'mode', 'list', 'cap_position', 'empty', 'schema_version', 'valve_state'}
        """
        try:
            if panel is None:
                panel = self._assemble_panel(trade_date)
        except DataSupplyError as exc:
            logger.error("数据供应链失败: %s → 降级", exc)
            return self.guard.on_failure()

        # 清洗 0→4 (推理端含安全阀)
        main_df, dual_df, valve_state = self.cleaner.run_inference(panel)
        if valve_state == "empty":
            logger.error("流动性安全阀强制空清单")
            return {"mode": "valve_empty", "list": pd.DataFrame(), "empty": True}

        # 特征 (只需清洗后幸存股票的历史)
        feat_main = self.features.build(
            panel[panel["symbol"].isin(main_df["symbol"].unique())],
            self.float_shares_map) if len(main_df) else pd.DataFrame()
        feat_dual = self.features.build(
            panel[panel["symbol"].isin(dual_df["symbol"].unique())],
            self.float_shares_map) if len(dual_df) else pd.DataFrame()

        # 推理 + 校准
        frames = []
        for board, feat, survivors in (("main", feat_main, main_df), ("dual", feat_dual, dual_df)):
            if len(feat) == 0:
                continue
            latest_symbols = survivors[survivors["date"] == survivors["date"].max()]["symbol"]
            today_feat = feat[feat["symbol"].isin(set(latest_symbols))]
            if len(today_feat) == 0:
                continue
            frames.append(self.predictor.predict(today_feat, board))
        if not frames:
            return self.guard.on_failure()
        candidates = pd.concat(frames, ignore_index=True)

        # Holding Bonus 回填 (昨日清单)
        yesterday = self._load_yesterday(trade_date)
        candidates = V35Predictor.mark_yesterday_list(candidates, yesterday)

        # 清单生成 (含 D18 空仓触发)
        result = self.lister.emit(candidates, env=env, market_state=market_state)
        result["valve_state"] = valve_state

        # 持久化 + 守卫
        if not result["empty"] and len(result["list"]):
            result["list"].to_parquet(self._list_path(trade_date), index=False)
            self.guard.on_success(result["list"])
            result["mode"] = "normal"
        else:
            result["mode"] = "empty"
        return result

    def _assemble_panel(self, trade_date: str) -> pd.DataFrame:
        """生产路径: 由 supply 装配全市场历史面板 (720 日窗口)."""
        raise DataSupplyError("生产装配: 需接入全市场历史库 (fetch_history 批量)")

    def _load_yesterday(self, trade_date: str) -> pd.DataFrame | None:
        """加载上一交易日清单 (Holding Bonus)."""
        dates = sorted(f.replace("list_", "").replace(".parquet", "")
                       for f in os.listdir(self.list_dir) if f.startswith("list_"))
        prev = [d for d in dates if d < trade_date]
        return self.load_list(prev[-1]) if prev else None

    # ---------------- 月度重训 (解耦) ----------------
    @staticmethod
    def is_retrain_day(trade_date: str, trade_calendar: list[str]) -> bool:
        """每月第一个交易日 → 15:30 启动重训."""
        idx = trade_calendar.index(trade_date) if trade_date in trade_calendar else -1
        if idx <= 0:
            return False
        return trade_date[:6] != trade_calendar[idx - 1][:6]

    def monthly_retrain(self, panels: dict[str, pd.DataFrame],
                        feature_cols_by_board: dict[str, list[str]],
                        tag: str) -> dict:
        """委托 DualTrackTrainer.monthly_retrain; 与 16:00 清单生成并行 (调用方排程)."""
        trainer = DualTrackTrainer()
        return trainer.monthly_retrain(panels, feature_cols_by_board, tag)
