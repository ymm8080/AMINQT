"""
Pipeline-1 每日选股编排 (P14 端到端, 生产主循环)
=====================================================
执行时序 (V3.5 周度重训解耦, 用户 2026-07-22 裁决: 周频):
  15:00 前   数据拉取 (DataSupplyChain, 失败 → 三档降级)
  15:30      每周第一个交易日启动重训 (T-1 数据, 与清单生成并行)
  16:00      用当前模型生成当日清单 (绝不让重训阻塞清单)
  18:00 前   重训完成, OOS 合格切换, 次日生效

日流程: 拉取 → 清洗 0→4 (+步骤5[E6]) → 特征 → 推理 → 校准 → 清单 schema V1.2 →
        Holding Bonus 回填 → 空仓触发 → 持久化 → 降级守卫.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import (
    LEGACY_PARALLEL_FEATURES,
    LEGACY_PROB_GATE,
    PANEL_V3_PATH,
    data_others_path,
)

from .cleaning_pipeline import CleaningPipeline
from .data_supply import DataSupplyChain, DataSupplyError
from .dual_track_trainer import DualTrackTrainer
from .feature_engine_v35 import FeatureEngineV35
from .forecast_accuracy import compute_bias_buckets, compute_quality_metrics
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

    def __init__(
        self,
        supply: DataSupplyChain,
        bundle_paths: dict[str, str],
        list_dir: str = "data/lists",
        float_shares_map: dict | None = None,
    ):
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
    def run(
        self,
        trade_date: str,
        panel: pd.DataFrame | None = None,
        env: MarketEnv | None = None,
        market_state: str = "range",
    ) -> dict:
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

        # 清洗 0→4 (推理端含安全阀). pool_blend=True (08-20 定案): dual 不在此切池,
        # 返回全谱双创, 待预测后按 blend(池分, pred_ret_10d) 切池 (pool_blend_cut)
        main_df, dual_df, valve_state = self.cleaner.run_inference(
            panel, pool_blend=True
        )
        if valve_state == "empty":
            logger.error("流动性安全阀强制空清单")
            return {"mode": "valve_empty", "list": pd.DataFrame(), "empty": True}

        # 特征 (在清洗输出上构建: 清洗附加列如 turnover_stability_5 是模型特征,
        # 用原始面板切片会导致训练/推理特征列不一致)
        # 推理模式: 只生成模型需要的派生列, 跳过 5000→~200 列的无用计算
        main_cols = (
            self.predictor.bundles["main"]["feature_cols"]
            if "main" in self.predictor.bundles
            else None
        )
        dual_cols = (
            self.predictor.bundles["dual"]["feature_cols"]
            if "dual" in self.predictor.bundles
            else None
        )
        # [2026-08-21] 双板特征构建子进程并行 (config LEGACY_PARALLEL_FEATURES):
        # main/dual 帧股票集不相交 + build 无状态 → 拆板并行结果与串行逐字节一致
        # (tests/test_parallel_feat_worker.py). 串行=main+dual 相加, 并行≈max(main,dual).
        if LEGACY_PARALLEL_FEATURES and len(main_df) and len(dual_df):
            feat_main, feat_dual = self._build_features_parallel(
                main_df, dual_df, main_cols, dual_cols
            )
        else:
            feat_main, feat_dual = self._build_features_serial(
                main_df, dual_df, main_cols, dual_cols
            )

        # 推理 + 校准
        frames = []
        gate_feats: dict[str, pd.DataFrame] = {}
        for board, feat, survivors in (
            ("main", feat_main, main_df),
            ("dual", feat_dual, dual_df),
        ):
            if len(feat) == 0:
                continue
            latest_symbols = survivors[survivors["date"] == survivors["date"].max()][
                "symbol"
            ]
            today_feat = feat[feat["symbol"].isin(set(latest_symbols))]
            # LEGACY_PROB_GATE 概率头用当日截面 (仅最新交易日一行, 防全历史重复行)
            if "date" in feat.columns:
                today_feat = today_feat[today_feat["date"] == today_feat["date"].max()]
            if len(today_feat) == 0:
                continue
            gate_feats[board] = today_feat
            frames.append(self.predictor.predict(today_feat, board))
        if not frames:
            return self.guard.on_failure()
        candidates = pd.concat(frames, ignore_index=True)
        # [08-20 定案] dual blend 入池: per (date, board) w*池分+(1-w)*预测涨幅 前 N.
        # 全谱预测后切池 → 高预期涨幅股可入池; main 不限池直通. 排名键仍纯 pred_ret_10d.
        candidates = self.cleaner.pool_blend_cut(candidates)

        # Holding Bonus 回填 (昨日清单)
        yesterday = self._load_yesterday(trade_date)
        candidates = V35Predictor.mark_yesterday_list(candidates, yesterday)

        # 全量候选预测持久化 (WORM): 供任意 symbol 当日预测即时查询, 免重算 (2026-08-06)
        try:
            cand_path = os.path.join(self.list_dir, f"candidates_{trade_date}.parquet")
            candidates.to_parquet(cand_path, index=False)
        except Exception:
            logger.warning("候选预测落盘失败 (非阻塞)", exc_info=True)

        # [2026-08-06] 输出级时间平滑: 单股预测/概率逐日剧变 → EMA 衰减 (Layer 2).
        # 在 emit 之前平滑 forecast 列 (pred_ret_*/prob_up*/pred_q50*), 使 E7 准入 + d3
        # 排名用稳定值; raw 底稿 WORM 落盘 legacy_preds_raw_<date>__<module>.csv.
        try:
            from .model_meta import load_modules, module_id
            from .pred_smoothing import persist_raw_preds, smooth_preds

            mod = module_id(load_modules())
            persist_raw_preds(candidates, trade_date, mod)
            candidates = smooth_preds(candidates, trade_date, mod)
        except Exception:
            logger.warning("预测平滑失败 (非阻塞)", exc_info=True)

        # [LEGACY_PROB_GATE] 概率闸输入组装 (2026-08-15 定案接线): legacy 无 stage
        # 特征检查点 → 由本处传特征当日截面/清洗帧面板尾/面板全局交易日; 组装失败 →
        # None (闸跳过, fail-open 不杀清单).
        prob_gate = None
        if LEGACY_PROB_GATE.get("enable", True):
            try:
                prob_gate = self._prob_gate_inputs(panel, main_df, dual_df, gate_feats)
            except Exception as exc:
                logger.error(
                    "LEGACY_PROB_GATE 输入组装失败 -> 闸跳过 (fail-open): %s", exc
                )

        # 清单生成 (含 D18 空仓触发)
        result = self.lister.emit(
            candidates, env=env, market_state=market_state, prob_gate=prob_gate
        )
        result["valve_state"] = valve_state

        # 持久化 + 守卫 + DB入库
        if not result["empty"] and len(result["list"]):
            # 模块版本戳: 每行记录产生该预测的 bundle 版本 (回归测试按 module 分组评估)
            from .model_meta import board_tag, load_modules

            mods = load_modules()
            if "board" in result["list"].columns:
                result["list"]["model_version"] = result["list"]["board"].map(
                    lambda b: board_tag(mods, b)
                )
            result["list"].to_parquet(self._list_path(trade_date), index=False)
            self.guard.on_success(result["list"])
            result["mode"] = "normal"
            # P25.5: 入库 prediction DB
            try:
                from .prediction_db import PredictionDB

                PredictionDB().insert_run(trade_date, result["list"])
            except Exception:
                logger.warning("预测池DB入库失败 (非阻塞)", exc_info=True)
            # 双轨影子 (2026-08-07): 同一份平滑候选按 pred_ret_3d 幅度排名入库,
            # 与生产 prob_up 排名并存, 1~2 月后真实结局对比 (eval_legacy_dual_track.py).
            try:
                from .prediction_db import PredictionDB, shadow_pool_frame

                PredictionDB().insert_shadow(trade_date, shadow_pool_frame(candidates))
            except Exception:
                logger.warning("影子排名入库失败 (非阻塞)", exc_info=True)
            # 同步到 priority.json (交易看板下拉框)
            try:
                import json

                pq_path = str(data_others_path("data/priority.json"))
                existing = set()
                if os.path.exists(pq_path):
                    with open(pq_path, encoding="utf-8") as f:
                        existing = set(json.load(f).get("symbols", []))
                new_symbols = set(result["list"]["symbol"].tolist())
                merged = sorted(existing | new_symbols)
                with open(pq_path, "w", encoding="utf-8") as f:
                    json.dump({"symbols": merged}, f, ensure_ascii=False, indent=2)
                logger.info(
                    "priority.json 同步: %d → %d (新增 %d)",
                    len(existing),
                    len(merged),
                    len(new_symbols - existing),
                )
            except Exception:
                logger.warning("priority.json 同步失败 (非阻塞)", exc_info=True)
        else:
            result["mode"] = "empty"
        return result

    def _build_features_serial(
        self,
        main_df: pd.DataFrame,
        dual_df: pd.DataFrame,
        main_cols: list[str] | None,
        dual_cols: list[str] | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """串行双板特征构建 (原 run() 内联逻辑抽出, 并行失败回退路径与主路径同源)."""
        feat_main = (
            self.features.build(
                main_df, self.float_shares_map, inference_cols=main_cols
            )
            if len(main_df)
            else pd.DataFrame()
        )
        feat_dual = (
            self.features.build(
                dual_df,
                self.float_shares_map,
                inference_cols=dual_cols,
                cross_sectional_rank=True,
            )
            if len(dual_df)
            else pd.DataFrame()
        )
        return feat_main, feat_dual

    def _build_features_parallel(
        self,
        main_df: pd.DataFrame,
        dual_df: pd.DataFrame,
        main_cols: list[str] | None,
        dual_cols: list[str] | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """双板特征构建子进程并行 (2026-08-21, config LEGACY_PARALLEL_FEATURES).

        父进程保留 panel/main/dual 帧 (概率闸 tail 仍需), 双 worker 各读自己板块
        parquet 构建, 特征帧落盘回读. worker 失败/超时 → 日志后回退串行重建,
        清单不丢 (fail-open 与生产其他非关键步骤同语义).
        """
        import json
        import shutil
        import subprocess
        import tempfile

        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        tmp = tempfile.mkdtemp(prefix="legacy_feat_")
        boards = [
            ("main", main_df, main_cols, False),
            ("dual", dual_df, dual_cols, True),
        ]
        try:
            jobs = []
            for board, df, cols, cs_rank in boards:
                in_path = os.path.join(tmp, f"{board}_in.parquet")
                out_path = os.path.join(tmp, f"{board}_out.parquet")
                df.to_parquet(in_path, index=False)
                argv = [
                    sys.executable,
                    "-u",
                    "-m",
                    "app.pipeline1.parallel_feat_worker",
                    in_path,
                    out_path,
                    json.dumps(cols or []),
                    "1" if cs_rank else "0",
                    json.dumps(self.float_shares_map or {}),
                ]
                jobs.append((board, subprocess.Popen(argv, cwd=root), out_path))
            outs: dict[str, pd.DataFrame] = {}
            for board, proc, out_path in jobs:
                try:
                    rc = proc.wait(timeout=2 * 3600)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = 124
                if rc != 0 or not os.path.exists(out_path):
                    logger.error(
                        "[feat_parallel] %s worker rc=%s → 回退串行构建", board, rc
                    )
                    return self._build_features_serial(
                        main_df, dual_df, main_cols, dual_cols
                    )
                outs[board] = pd.read_parquet(out_path)
            return outs.get("main", pd.DataFrame()), outs.get("dual", pd.DataFrame())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _assemble_panel(self, trade_date: str) -> pd.DataFrame:
        """生产路径: 加载历史面板 + 追加当日数据.

        1. 加载缓存的 panel_full_enriched.parquet (历史面板)
        2. 拉取当日 OHLCV + margin + lhb
        3. 追加当日行并合并 alt data
        4. 返回完整面板 (含当日)
        """
        from .panel_builder import enrich_cyq

        # 加载历史面板: 优先 canonical PANEL_V3_PATH (D:\AMINQT\PARQUET),
        # 再回退仓库内相对路径 (历史位置)
        candidates = [
            str(PANEL_V3_PATH),
            "data/panel_full_enriched_v3.parquet",
            "data/panel_full_enriched_v2.parquet",
        ]
        panel = None
        for path in candidates:
            if os.path.exists(path):
                panel = pd.read_parquet(path)
                logger.info(
                    "加载历史面板: %s (%d stocks, %d rows)",
                    path,
                    panel["symbol"].nunique(),
                    len(panel),
                )
                break
        if panel is None:
            raise DataSupplyError("无可用历史面板缓存 (panel_*.parquet)")

        # 确保历史面板不含当日 (避免重复)
        panel = panel[panel["date"] < pd.to_datetime(trade_date)]

        # CYQ enrich (增量: 只算新股票)
        panel = enrich_cyq(panel, cyq_cache="data/cyq_panel.parquet")

        # 追加当日数据
        panel = self.supply.append_today_to_panel(
            panel,
            trade_date=trade_date,
            sources=["ohlcv", "margin", "lhb"],  # northbound 已移除
        )

        logger.info(
            "面板装配完成: %d stocks, %d rows, %d cols",
            panel["symbol"].nunique(),
            len(panel),
            len(panel.columns),
        )
        return panel

    def _load_yesterday(self, trade_date: str) -> pd.DataFrame | None:
        """加载上一交易日清单 (Holding Bonus)."""
        dates = sorted(
            f.replace("list_", "").replace(".parquet", "")
            for f in os.listdir(self.list_dir)
            if f.startswith("list_")
        )
        prev = [d for d in dates if d < trade_date]
        return self.load_list(prev[-1]) if prev else None

    # ---------------- LEGACY_PROB_GATE 输入组装 ----------------
    @staticmethod
    def _prob_gate_inputs(
        panel: pd.DataFrame,
        main_df: pd.DataFrame,
        dual_df: pd.DataFrame,
        feats_by_board: dict[str, pd.DataFrame],
    ) -> dict:
        """组装 legacy 并行式概率闸输入 (prob_head.apply_prob_gate 签名, 2026-08-16).

        legacy 无 stage 特征检查点 → 数据流与训练脚本 _train_legacy_prob_head 同构:
        - feats: 各板当日截面 V35 特征帧 (run() 已构建, bundle feat_cols ⊆ 模型
          inference_cols — 概率头与模型同 inference_cols 训练)
        - tail: 清洗帧面板尾 (symbol/date/close_hfq/high_hfq/adv20, 近 base_rate_days+14
          交易日); adv20 由 amount 20 日均值现算 (清洗帧无 adv20, 同 label_engine 口径,
          仅用历史 → 无前瞻)
        - panel_dates: 面板全局交易日 (bundle staleness 判定)
        清洗帧缺所需 raw 列 → raise (由 run() 捕获 → 闸跳过 fail-open).
        """
        from .label_engine import _ensure_sorted

        n = LEGACY_PROB_GATE["base_rate_days"] + 14
        tail_frames = []
        for df in (main_df, dual_df):
            if df is None or len(df) == 0:
                continue
            need = ["symbol", "date", "close_hfq", "high_hfq", "amount"]
            missing = [c for c in need if c not in df.columns]
            if missing:
                raise ValueError(f"清洗帧缺 {missing} 列, 无法组装概率闸 tail")
            # 清洗帧无 adv20 (特征引擎内部中间量), 按 label_engine 同口径从 amount 现算
            s = _ensure_sorted(df)
            adv20 = (
                s.groupby("symbol")["amount"]
                .rolling(20, min_periods=20)
                .mean()
                .reset_index(level=0, drop=True)
            )
            dates = pd.to_datetime(s["date"])
            uni = np.unique(dates.to_numpy())
            cut = uni[-n] if len(uni) >= n else uni[0]
            mask = dates.to_numpy() >= cut
            tail = s.loc[mask, ["symbol", "date", "close_hfq", "high_hfq"]].copy()
            tail["date"] = pd.to_datetime(tail["date"])
            tail["symbol"] = tail["symbol"].astype(str)
            tail["adv20"] = adv20[mask].to_numpy()
            tail_frames.append(tail)
        return {
            "feats": dict(feats_by_board),
            "tail": pd.concat(tail_frames, ignore_index=True),
            "panel_dates": np.unique(pd.to_datetime(panel["date"]).to_numpy()),
        }

    # ---------------- 持仓卖出信号 (预测驱动 + 价格硬止损) ----------------
    def predict_held(
        self,
        trade_date: str,
        held_symbols: list[str],
        entry_cost_map: dict | None = None,
        panel: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """对持仓股重跑预测并输出卖出信号 (16:00 主链之后运行).

        买入链路只预测当日候选; 持仓股需要每天用当日特征重算预测,
        才能拿到新鲜 pred_ret_3d / prob_up / pain_prob 做卖出裁决.

        Args:
            trade_date: 'YYYYMMDD'
            held_symbols: 当前持仓 symbol 列表
            entry_cost_map: {symbol: 买入成本价}; 提供时合并当日 close 计算
                            pnl (现价/成本-1), 触发价格硬止损 (-6%)
            panel: 完整面板 (含当日); None → _assemble_panel (生产路径)

        Returns:
            DataFrame(symbol, board, close, pred_ret_3d/5d/10d, prob_up,
                      pain_prob, pred_q10, [pnl], sell_signal, sell_reason);
            无持仓可预测时返回空 DataFrame.
        """
        if panel is None:
            panel = self._assemble_panel(trade_date)
        main_df, dual_df, _valve = self.cleaner.run_inference(panel)
        held = set(held_symbols)
        frames = []
        for board, df in (("main", main_df), ("dual", dual_df)):
            if len(df) == 0 or board not in self.predictor.bundles:
                continue
            cols = self.predictor.bundles[board]["feature_cols"]
            # dual 板需要全截面算 cross-sectional rank, 特征在全 df 上构建
            feat = self.features.build(
                df,
                self.float_shares_map,
                inference_cols=cols,
                cross_sectional_rank=(board == "dual"),
            )
            held_feat = feat[feat["symbol"].isin(held)]
            if len(held_feat) == 0:
                continue
            pred = self.predictor.predict(held_feat, board)
            latest_close = (
                df[df["symbol"].isin(held)]
                .sort_values("date")
                .groupby("symbol")["close"]
                .last()
            )
            pred["close"] = pred["symbol"].map(latest_close)
            if entry_cost_map:
                pred["pnl"] = pred["close"] / pred["symbol"].map(entry_cost_map) - 1
            frames.append(pred)
        if not frames:
            logger.warning("持仓股均不在可预测范围: %s", held_symbols)
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        from .sell_signal import evaluate_sell_signal

        return evaluate_sell_signal(out, pnl_col="pnl" if entry_cost_map else None)

    # ---------------- P25.4 每日质量报告 (D-26) ----------------
    def generate_quality_report(self, forecast_df, actual_returns, trade_date):
        """D-26: 推理后自动计算预测质量报告.

        Returns:
            quality_report dict (MAE/BIAS/方向准确率/分桶BIAS/红灯状态).
        """
        import json

        actual = actual_returns.get(
            "label_pm_3d_net", actual_returns.get("label_3d_net", None)
        )
        if actual is None or len(forecast_df) == 0:
            return None

        pred = forecast_df["pred_ret_3d"].values
        act = actual.values if hasattr(actual, "values") else actual

        metrics = compute_quality_metrics(act, pred)
        buckets = compute_bias_buckets(act, pred)
        from .oos_monitor import bias_traffic_light

        light = bias_traffic_light({**metrics, **buckets})

        report = {
            "date": trade_date,
            **metrics,
            **buckets,
            "traffic_light": light,
        }

        # WORM 入库
        try:
            worm_dir = str(data_others_path("data/quality_reports"))
            os.makedirs(worm_dir, exist_ok=True)
            path = os.path.join(worm_dir, f"quality_{trade_date}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
        except Exception:
            logger.warning("质量报告 WORM 入库失败 (非阻塞)", exc_info=True)

        if light == "RED":
            logger.critical("BIAS 红灯触发: %s → E4-L1 模型降级", trade_date)

        logger.info(
            "预测质量报告 %s (3d): MAE=%.4f BIAS=%+.4f DirAcc=%.2f%% Light=%s",
            trade_date,
            metrics["mae_1d"],
            metrics["bias_1d"],
            metrics["direction_accuracy"] * 100,
            light,
        )
        return report

    # ---------------- 周度重训 (解耦) ----------------
    @staticmethod
    def is_retrain_day(trade_date: str, trade_calendar: list[str]) -> bool:
        """每周第一个交易日 → 15:30 启动重训 (用户 2026-07-22 裁决: 周频).

        判定: 当日与上一交易日分属不同 ISO 周.
        """
        idx = trade_calendar.index(trade_date) if trade_date in trade_calendar else -1
        if idx <= 0:
            return False

        def iso_week(d: str) -> tuple[int, int]:
            return datetime.strptime(d, "%Y%m%d").isocalendar()[:2]

        return iso_week(trade_date) != iso_week(trade_calendar[idx - 1])

    def weekly_retrain(
        self,
        panels: dict[str, pd.DataFrame],
        feature_cols_by_board: dict[str, list[str]],
        tag: str,
    ) -> dict:
        """委托 DualTrackTrainer.weekly_retrain; 与 16:00 清单生成并行 (调用方排程)."""
        trainer = DualTrackTrainer()
        return trainer.weekly_retrain(panels, feature_cols_by_board, tag)
