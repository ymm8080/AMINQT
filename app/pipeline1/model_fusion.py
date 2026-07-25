"""
双模型动态融合 (PIPELINE1_V3.8 §2.7, 安全网 #16)
====================================================
重训排程: 周六 20:00 双模型同步重训 (Long 620d + Short 180d);
          周三条件触发仅 Short; 平日不重训.
融合权重 (随每次重训更新):
  20 日 OOS Rank IC (净口径) → Softmax(τ=0.01) → 负IC保护
  → w_long ∈ [0.2, 0.8] → 惯性平滑 0.7/0.3 → WORM 日志.
双模型 IC 均 < 0.01 → 黄色告警 (与 E4 三色灯联动).

纪律: 融合权重只由 OOS IC 反推, 严禁人工指定 (安全网 #15).
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

TAU = 0.01  # Softmax 温度
W_LONG_MIN, W_LONG_MAX = 0.2, 0.8  # 权重边界
INERTIA_PREV, INERTIA_NEW = 0.7, 0.3  # 惯性平滑: 0.7×旧 + 0.3×新
IC_YELLOW = 0.01  # 双模型 IC 均低于此 → 黄色告警
LONG_WINDOW, SHORT_WINDOW = 620, 180  # 训练窗口 (V3.8 §2.7)


class DualModelFusion:
    """Long/Short 双模型动态融合权重状态机 (随重训周期更新).

    用法:
        fusion = DualModelFusion("data/fusion")
        w = fusion.update_weight(ic_long_20d=0.035, ic_short_20d=0.020)
        pred = fusion.fuse(pred_long, pred_short)
    """

    def __init__(self, log_path: str | None = None):
        self.w_long = 0.5  # 初始等权
        self.log_path = log_path
        if log_path:
            os.makedirs(log_path, exist_ok=True)

    # ---------------- 融合权重 ----------------
    def update_weight(self, ic_long_20d: float, ic_short_20d: float) -> float:
        """20 日 OOS Rank IC → 新融合权重 (全链路: softmax→负IC保护→边界→惯性).

        负IC保护: 负 IC 先 clip 到 0 (失效模型不配拿权重, 但边界 0.2 保底
        防全押单模型); 惯性平滑防权重日间跳变.
        """
        raw_long, raw_short = max(ic_long_20d, 0.0), max(ic_short_20d, 0.0)
        z = np.array([raw_long, raw_short]) / TAU
        z = z - z.max()  # 数值稳定
        exp = np.exp(z)
        w_new = float(exp[0] / exp.sum())
        w_new = float(np.clip(w_new, W_LONG_MIN, W_LONG_MAX))
        self.w_long = INERTIA_PREV * self.w_long + INERTIA_NEW * w_new
        if ic_long_20d < IC_YELLOW and ic_short_20d < IC_YELLOW:
            logger.error("双模型 20 日 OOS IC 均 < %.2f (long=%.4f short=%.4f), "
                         "黄色告警 → E4 三色灯联动", IC_YELLOW,
                         ic_long_20d, ic_short_20d)
        self._worm(ic_long_20d, ic_short_20d, w_new)
        logger.info("融合权重更新: w_long=%.3f (IC long=%.4f short=%.4f)",
                    self.w_long, ic_long_20d, ic_short_20d)
        return self.w_long

    def fuse(self, pred_long: np.ndarray, pred_short: np.ndarray) -> np.ndarray:
        """预测融合: w_long×pred_long + (1-w_long)×pred_short."""
        return self.w_long * np.asarray(pred_long) + (
            1 - self.w_long) * np.asarray(pred_short)

    # ---------------- WORM 日志 ----------------
    def _worm(self, ic_long: float, ic_short: float, w_new: float) -> None:
        if not self.log_path:
            return
        rec = {
            "ic_long_20d": round(ic_long, 5), "ic_short_20d": round(ic_short, 5),
            "w_softmax": round(w_new, 4), "w_long": round(self.w_long, 4),
        }
        path = os.path.join(self.log_path, "fusion_weights.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 重训排程 (周六双模型 / 周三条件 Short / 平日不重训)
# ============================================================
def retrain_schedule(weekday: int, short_ic_alert: bool = False) -> list[str]:
    """返回今日应重训的模型清单.

    Args:
        weekday: 0=周一 ... 2=周三 ... 5=周六
        short_ic_alert: Short 模型 IC 告警 (E4 三色灯黄/红 → 周三条件触发)
    """
    if weekday == 5:
        return ["long", "short"]  # 周六 20:00 双模型同步重训
    if weekday == 2 and short_ic_alert:
        return ["short"]  # 周三条件触发仅 Short
    return []  # 平日不重训
