"""
D.8 双清单与影子跟踪机制 (PIPELINE1_V3.8 附录D.8, 检查清单 D-6/D-7)
================================================================================
每日 21:00 推理后, 规则引擎对同一份模型输出分别按两档参数生成两份清单:
  执行清单 (aggressive): 真金白银执行 — A 级信号门槛 (grade_A_entry 全部满足,
                         缺一不可) + 阻尼排序选取 + 数量闸门前 2
  影子清单 (stable):     不投入资金, 每日记录"假如执行"的完整损益
                         (与执行清单完全相同的失效条件与成本口径, 否则对比失真)

规则:
  - 清单 schema 增加 profile 字段 (stable/aggressive), 两份均入 WORM 日志
  - 两档重叠票为正常现象 (攻击档 Top0-2 多为稳定档 Top15 子集)
  - 月度 GT-Score 双档对比 (D.6): 攻击档连续 3 个月 GT-Score 低于稳定档
    → 强制切回 stable 并书面归因
  - 资金不分仓: 任何时刻只允许一份清单进入真实执行, 严禁"两份都买"
  - 熊市协议 (安全网 #19) 对攻击档强制接管 (D.7), 无豁免
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

from app.config.profiles import get_profile

from .dynamic_engine import DynamicEngine
from .gt_score import dual_profile_verdict, gt_score
from .list_generator import ListGenerator

logger = logging.getLogger(__name__)

ADJUDICATE_STREAK = 3  # D.6: 连续 3 个月 GT-Score 落后 → 强制切回 stable


class DualListRunner:
    """D.8 双清单编排: 同一模型输出 → 执行清单 (aggressive) + 影子清单 (stable).

    用法:
        runner = DualListRunner("data/dual_lists")
        out = runner.emit(candidates, "2026-07-25", market_state="range")
        # out["execution"] (0~2 只, A级) / out["shadow"] (0~15 只, stable)
    """

    def __init__(
        self, store_dir: str | None = None, stable_lister: ListGenerator | None = None
    ):
        self.stable = stable_lister or ListGenerator()
        self.store_dir = store_dir
        self._hist_exec: dict[str, float] = {}  # D.6 月度 GT 历史 (执行清单)
        self._hist_shadow: dict[str, float] = {}  # D.6 月度 GT 历史 (影子清单)
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)

    # ---------------- 执行清单 (aggressive A 级信号) ----------------
    @staticmethod
    def aggressive_entry(candidates: pd.DataFrame) -> pd.DataFrame:
        """A 级信号门槛 (grade_A_entry, 全部满足缺一不可, D.1).

        筛选链: 主板限定 → prob_up_calibrated ≥ 0.68 → pain_prob < 0.15
        → 事件窗口黑名单 → 板块共振前 5 (有 sector_rank 列时)
        → 阻尼排序 (E.3) 取前 2 (数量闸门, 用户定稿: 一天 1-2 只).
        """
        g = get_profile("aggressive")["grade_A_entry"]
        df = candidates.copy()
        if g["main_board_only"] and "board" in df.columns:
            df = df[df["board"] == "main"]
        df = df[df["prob_up"] >= g["prob_up_calibrated"]]
        if "pain_prob" in df.columns:
            df = df[df["pain_prob"].fillna(1.0) < g["pain_prob_max"]]
        if g["event_window_blacklist"] and "event_window" in df.columns:
            df = df[~df["event_window"].astype(bool)]
        if "sector_rank" in df.columns:
            df = df[df["sector_rank"] <= g["sector_resonance_top"]]
        # E.3 阻尼排序选取 (有分布列时), 否则用 score/prob_up
        if {"score", "uncertainty_width"} <= set(df.columns) and len(df):
            df = df.assign(
                _sel=DynamicEngine.damped_score(df["score"], df["uncertainty_width"])
            )
        elif "score" in df.columns:
            df = df.assign(_sel=df["score"])
        else:
            df = df.assign(_sel=df["prob_up"])
        df = df.sort_values("_sel", ascending=False).head(g["rank_score_top"])
        df = df.drop(columns=["_sel"])
        if len(df) == 0:
            logger.warning(
                "D.8 执行清单: 0 只过 A 级门槛, 今日空仓 (特性非故障, "
                "预计 75%% 交易日空仓)"
            )
        return df

    # ---------------- 双清单总装 ----------------
    def emit(
        self,
        candidates: pd.DataFrame,
        trade_date: str,
        market_state: str = "range",
        **stable_kwargs,
    ) -> dict:
        """生成执行清单 + 影子清单 (schema 含 profile, 均入 WORM).

        stable_kwargs: 透传 ListGenerator.emit (env/capital/ret_window_20d).
        Returns:
            {'execution': DataFrame(profile=aggressive),
             'shadow': DataFrame(profile=stable),
             'bear_takeover': bool (DEFENSE → 攻击档强制只卖不买, D.7)}
        """
        bear_takeover = market_state == "bear"
        execution = self.aggressive_entry(candidates)
        shadow = self.stable.emit(
            candidates, market_state=market_state, **stable_kwargs
        )["list"]
        execution = execution.copy()
        shadow = shadow.copy()
        execution["profile"] = "aggressive"
        shadow["profile"] = "stable"
        # D.9/E.2: 执行清单每票附带 stop_price 与 position (动态引擎影子输出,
        # 阶段一至三只留痕不驱动交易, F.5)
        if len(execution):
            execution = self._attach_dynamic_outputs(execution)
        if bear_takeover and len(execution):
            logger.error("D.7 熊市协议强制接管: DEFENSE 状态下攻击档只卖不买")
            execution = execution.iloc[0:0]
        self._persist(trade_date, execution, shadow)
        return {
            "execution": execution,
            "shadow": shadow,
            "bear_takeover": bear_takeover,
            # 资金不分仓 (D.8): live_profile 是唯一允许进入真实执行的清单;
            # 熊市接管日 None (只卖不买). shadow 永不 live, 严禁"两份都买".
            "live_profile": None if bear_takeover else "aggressive",
        }

    # ---------------- D.9/E.2 执行清单动态输出 ----------------
    @staticmethod
    def _attach_dynamic_outputs(execution: pd.DataFrame) -> pd.DataFrame:
        """每票每日随清单输出 stop_price 与 position (D.9/E.2 计算链留痕).

        输入列齐备 (prob_up/pred_q50/ATR_pct[/pain_prob]) 时逐票反推;
        缺列跳过 (向后兼容). close 存在时给出 stop_price 绝对价.
        """
        required = {"prob_up", "pred_q50", "ATR_pct"}
        if not required <= set(execution.columns):
            return execution
        eng = DynamicEngine()
        calcs = [
            eng.per_stock_calc(
                p=float(r["prob_up"]),
                pred_q50=float(r["pred_q50"]),
                atr_pct=float(r["ATR_pct"]),
                pain_prob=float(r.get("pain_prob", 0.0) or 0.0),
            )
            for _, r in execution.iterrows()
        ]
        out = execution.copy()
        out["dyn_stop_pct"] = [c["stop"] for c in calcs]
        out["dyn_position"] = [c["position"] for c in calcs]
        out["dyn_rr"] = [c["rr"] for c in calcs]
        if "close" in out.columns:
            out["stop_price"] = (out["close"] * (1 - out["dyn_stop_pct"])).round(3)
        return out

    # ---------------- WORM 持久化 ----------------
    def _persist(
        self, trade_date: str, execution: pd.DataFrame, shadow: pd.DataFrame
    ) -> None:
        if not self.store_dir:
            return
        path = os.path.join(self.store_dir, f"dual_{trade_date}.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            for df in (execution, shadow):
                for row in df.to_dict("records"):
                    row["trade_date"] = str(trade_date)
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # ---------------- D.6 月度双档 GT-Score 裁决 ----------------
    def monthly_adjudication(
        self,
        exec_ics,
        exec_turnover,
        shadow_ics,
        shadow_turnover,
        month: str | None = None,
    ) -> dict:
        """D.6 档位裁决: 攻击档连续 3 个月 GT-Score 低于稳定档 → 强制切回 stable.

        用事实裁决档位, 不拍脑袋. 裁决逻辑单点维护于
        `gt_score.dual_profile_verdict` (重叠月份不足 3 个月不得裁决).
        month: 月份标签 ("YYYY-MM"), 缺省按调用顺序合成.
        """
        label = month or f"m{len(self._hist_exec) + 1}"
        self._hist_exec[label] = gt_score(exec_ics, exec_turnover)
        self._hist_shadow[label] = gt_score(shadow_ics, shadow_turnover)
        v = dual_profile_verdict(
            self._hist_shadow, self._hist_exec, consecutive=ADJUDICATE_STREAK
        )
        force = v["force_switch_to_stable"]
        if force:
            logger.critical(
                "D.6 档位裁决: 攻击档连续 %d 月 GT-Score 落后 (%.4f vs %.4f), "
                "强制切回 stable 并书面归因",
                v["trailing_below"],
                self._hist_exec[label],
                self._hist_shadow[label],
            )
        return {
            "gt_aggressive": round(self._hist_exec[label], 4),
            "gt_stable": round(self._hist_shadow[label], 4),
            "lose_streak": v["trailing_below"],
            "force_switch_to_stable": force,
        }


def resolve_live_list(out: dict) -> pd.DataFrame:
    """资金不分仓校验 (P21.4 / D.8): 返回唯一允许进入真实执行的清单.

    任何时刻只有 live_profile 指向的清单可下单; shadow 永不返回;
    熊市接管日 (live_profile=None) 返回空表 (只卖不买).
    """
    live = out.get("live_profile")
    assert live in (None, "aggressive"), f"非法 live 通道: {live}"
    if live is None:
        return out["execution"].iloc[0:0]
    df = out["execution"]
    assert (df["profile"] == live).all(), "执行清单 profile 与 live 通道不一致"
    shadow = out["shadow"]
    assert (shadow["profile"] != live).all(), "影子清单混入 live 通道 (两份都买?)"
    return df
