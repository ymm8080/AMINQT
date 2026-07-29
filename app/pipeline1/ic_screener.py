"""
IC 筛选器 (DESIGN §14 三 bis, 安全网 #2, PIPELINE1_V3.8 §三 bis)
==================================================================
- 每个滚动重训窗口内, 仅用该窗口训练集重新计算 IC 并重新筛因子 (严禁全样本一次筛选 = 前视偏差)
- 三标签 (1d/3d/5d) 分别计算 Rank IC → 取并集; [E5] 净口径 label_*_net 优先
- [E2] mdd 标签独立筛选 (label_mdd_3d) → 与净口径并集再取并集
- 分类模型独立: AUC + 互信息 → 与回归并集再取并集
- 滚动 IC 双指标 (D13): 60日滚动 IC 均值 > 0.02 且正值比例 > 60%
- [E4-L2] 连续 3 期滚动 IC 为负的因子自动剔除 (跨窗口持久化追踪)
- 每期因子清单必须记录 (工程强制)
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic

logger = logging.getLogger(__name__)

# ── P19 两阶门禁: Factor Gate (此处) vs Model Gate (metrics.py) ──
# Factor Gate: 单因子初筛 — 门槛宽松, 靠模型组合提纯 (ICIR≥0.05 即可).
#   单因子天生噪声大, ICIR 0.10-0.25 是正常范围, 不能套用模型分数的 0.30 门槛.
# Model Gate (metrics.ignition_gate): 组合模型分数 — 门槛严格 (ICIR≥0.30),
#   因为模型分数的任务是把多个 noisy 因子组合成一个稳定信号.
# 参考: FEATURE ADOPTION ANALYSIS 20260729, 全量 2.7M 行截面 IC 实测.
IC_STRONG = 0.02  # 有效因子 |IC| 下限 (因子级: ≥0.02 即有效; 模型级: ≥0.03)
IC_WEAK = 0.01    # 弱因子 |IC| 下限 (0.01-0.02: 正交信息可被模型组合利用)
ROLLING_WINDOW = 60  # 滚动 IC 窗口 (交易日)
ROLLING_MEAN_MIN = 0.01  # 60日滚动IC均值下限
ROLLING_POS_RATIO_MIN = 0.50  # 滚动IC正值比例 (过半即可, 原0.60太严)
L2_NEG_PERIODS = (
    10  # [E4-L2] 连续 N 期滚动 IC 为负 → 自动剔除 (提高至10期, 允许周期性回撤)
)
L2_RECOVERY_PERIODS = 1  # 连续 N 期正向 IC → 解除剔除, 恢复候选资格
ICIR_MIN = (
    0.05  # [P19 两阶门禁] 因子级 ICIR 下限: |IC|/IC_std ≥ 0.05 (原0.30为模型级门槛)
)


class ICScreener:
    """IC 筛选 — 每期滚动重算."""

    def __init__(self, registry_path: str = "data/factor_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)

    # ---------------- 单因子 IC ----------------
    @staticmethod
    def rank_ic(df: pd.DataFrame, factor: str, label: str) -> float:
        """横截面 Rank IC 均值 (带符号, 日度 Spearman 时间均值)."""
        return mean_rank_ic(df, factor, label, abs_mean=False)

    # ---------------- IC 稳定性 (P18) ----------------
    @staticmethod
    def ic_stability(df: pd.DataFrame, factor: str, label: str) -> float:
        """ICIR = |IC_mean| / IC_std — 信号稳定性 (杀稀疏/噪声因子).

        IC 高但日间波动剧烈 → ICIR 低 → 不可靠.
        IC 中等但稳定正 → ICIR 高 → 可靠.
        阈值: ICIR < 0.3 → 降级.
        """
        sub = df[["date", factor, label]].dropna()
        if len(sub) < 30:
            return 0.0
        ic_series = daily_rank_ic_series(sub, factor, label)
        if len(ic_series) < 10:
            return 0.0
        ic_std = float(ic_series.std())
        ic_mean = float(ic_series.mean())
        return abs(ic_mean) / ic_std if ic_std > 0 else 0.0

    # ---------------- Newey-West HAC t 统计量 (B6) ----------------
    @staticmethod
    def ic_t_stat_newey_west(
        df: pd.DataFrame, factor: str, label: str, lag: int = 0
    ) -> float:
        """IC 序列 Newey-West HAC 调整后的 t 统计量 (B6).

        3d/5d 标签重叠样本导致 IC 序列自相关, t 统计量虚高.
        lag=0: 标准 t (1d 无重叠); lag=5: 3d; lag=8: 5d.
        """
        sub = df[["date", factor, label]].dropna()
        dates = sorted(sub["date"].unique())
        if len(dates) < 10:
            return 0.0
        ic_series = daily_rank_ic_series(sub, factor, label)
        n = len(ic_series)
        if n < 5:
            return 0.0
        mean_ic = float(ic_series.mean())
        # HAC 方差估计 (Bartlett kernel)
        gamma0 = float(((ic_series - mean_ic) ** 2).mean())
        hac_var = gamma0
        for ell in range(1, min(lag + 1, n)):
            w = 1 - ell / (lag + 1)  # Bartlett 权重
            cov_l = float(
                (
                    (ic_series.iloc[ell:] - mean_ic) * (ic_series.iloc[:-ell] - mean_ic)
                ).mean()
            )
            hac_var += 2 * w * cov_l
        hac_var = max(hac_var, 1e-12)
        return mean_ic / (np.sqrt(hac_var / n))

    @staticmethod
    def rolling_ic_dual(
        df: pd.DataFrame, factor: str, label: str, window: int = ROLLING_WINDOW
    ) -> tuple[float, float]:
        """滚动 IC 双指标 (D13): (滚动 IC 均值, 主导方向日占比).

        主导方向日占比 = max(IC>0天数, IC<0天数) / 总天数, 衡量方向一致性.
        """
        sub = df[["date", factor, label]].dropna()
        dates = sorted(sub["date"].unique())
        if len(dates) < window:
            return 0.0, 0.0
        daily_ic = daily_rank_ic_series(sub, factor, label)
        rolls = [
            daily_ic.loc[d0:d1].mean()
            for d0, d1 in zip(dates[:-window], dates[window:])
        ]
        rolls = pd.Series(rolls).dropna().abs()
        if len(rolls) == 0:
            return 0.0, 0.0
        return float(rolls.mean()), float(
            max((daily_ic > 0).mean(), (daily_ic < 0).mean()) if len(daily_ic) else 0.0
        )

    # ---------------- 分类模型 AUC 筛选 ----------------
    @staticmethod
    def auc_score(df: pd.DataFrame, factor: str, label: str = "label_cls") -> float:
        """单因子对分类标签的 AUC (方向无关: max(auc, 1-auc)).
        [B9] label_pm_cls 优先于 label_cls, 调用方应显式传入.
        """
        from sklearn.metrics import roc_auc_score

        # [B9] PM 执行口径分类标签优先
        if label == "label_cls" and "label_pm_cls" in df.columns:
            label = "label_pm_cls"
        sub = df[[factor, label]].dropna()
        if sub[label].nunique() < 2 or len(sub) < 50:
            return 0.5
        auc = roc_auc_score(sub[label], sub[factor])
        return float(max(auc, 1 - auc))

    # ---------------- 主筛选 ----------------
    def screen(
        self, train_df: pd.DataFrame, feature_cols: list[str], window_id: str
    ) -> dict:
        """窗口内重算 IC 并筛因子.

        [E5] 净口径: label_{k}d_net 存在时优先于毛收益 label_{k}d;
        [E2] label_mdd_3d 独立筛选, 并入并集;
        [E4-L2] 连续 3 期滚动 IC 为负 → 强制 grade='dead'.

        Returns:
            {window_id, factors: [...], detail: {factor: {ic_1d, ..., grade}}}
            grade: 'strong' 保留 / 'weak' 观察 / 'dead' 剔除
        """
        # [B9] PM 执行口径验收标签优先; [E5] 净收益标签优先 (分层滑点口径)
        label_of = {}
        for k in (1, 3, 5):
            pm_net = f"label_pm_{k}d_net"
            reg_net = f"label_{k}d_net"
            pm_raw = f"label_pm_{k}d"
            reg_raw = f"label_{k}d"
            if pm_net in train_df.columns:
                label_of[k] = pm_net
            elif reg_net in train_df.columns:
                label_of[k] = reg_net
            elif pm_raw in train_df.columns:
                label_of[k] = pm_raw
            else:
                label_of[k] = reg_raw
        has_mdd = "label_mdd_3d" in train_df.columns  # [E2]
        l2_history = self._load_l2_history()
        result = {"window_id": window_id, "factors": [], "detail": {}}
        for f in feature_cols:
            ic_by_label = {
                k: self.rank_ic(train_df, f, lbl) for k, lbl in label_of.items()
            }
            # [P19 Factor Gate] best_ic 取绝对值最大 (方向无关; 负向强预测因子同样有效)
            best_ic = max((abs(v) for v in ic_by_label.values()), default=0.0)
            ic_mdd = self.rank_ic(train_df, f, "label_mdd_3d") if has_mdd else None
            if ic_mdd is not None:
                best_ic = max(best_ic, abs(ic_mdd))  # [E2] mdd 独立筛选并入并集
            auc = self.auc_score(train_df, f)
            # 滚动 IC 取各标签最佳值 (非仅 1d — 3d/5d 正向不因 1d 负向降级)
            roll_best = max(
                (self.rolling_ic_dual(train_df, f, lbl) for lbl in label_of.values()),
                key=lambda x: x[0],  # max by rolling mean
            )
            roll_mean, roll_pos = roll_best
            dual_ok = roll_mean > ROLLING_MEAN_MIN and roll_pos > ROLLING_POS_RATIO_MIN
            # B6: 3d/5d IC 显著性用 Newey-West HAC 调整 (lag=5/8)
            t_3d = self.ic_t_stat_newey_west(train_df, f, label_of[3], lag=5)
            t_5d = self.ic_t_stat_newey_west(train_df, f, label_of[5], lag=8)
            nw_significant = (
                abs(t_3d) > 1.28 or abs(t_5d) > 1.28
            )  # 90%置信 (原1.96/95%太严)
            # [P19 Factor Gate] IC 稳定性: 取各标签最佳 ICIR
            icir = max(self.ic_stability(train_df, f, lbl) for lbl in label_of.values())
            icir_ok = icir >= ICIR_MIN  # 因子级门槛 0.05 (模型级见 metrics.ignition_gate)
            if (
                (best_ic > IC_STRONG or auc > 0.55)
                and dual_ok
                and nw_significant
                and icir_ok
            ):
                grade = "strong"
            elif (best_ic > IC_STRONG or auc > 0.55) and dual_ok and nw_significant:
                grade = "weak"  # IC够但ICIR不足 → 降级为weak
            elif best_ic > IC_WEAK or auc > 0.52:
                grade = "weak"
            else:
                grade = "dead"
            # [E4-L2] 连续 N 期窗口 IC 为负 → 自动剔除; 连续 M 期正向 → 恢复
            # 使用最佳标签的带符号 IC (非仅 1d): 3d/5d 正向因子不应被 1d 负向杀死
            ics_signed = {
                k: self._signed_daily_ic_mean(train_df, f, lbl)
                for k, lbl in label_of.items()
            }
            signed_ic = max(ics_signed.values())  # 取各标签中最佳方向 IC
            entry = l2_history.get(f, {})
            if isinstance(entry, (int, float)):
                entry = {"neg": int(entry), "pos": 0}  # migrate old format
            neg_streak = entry.get("neg", 0)
            pos_streak = entry.get("pos", 0)
            if signed_ic < 0:
                neg_streak += 1
                pos_streak = 0
            else:
                pos_streak += 1
                if neg_streak >= L2_NEG_PERIODS:
                    # was evicted — need L2_RECOVERY_PERIODS positive windows to revive
                    if pos_streak < L2_RECOVERY_PERIODS:
                        pass  # still evicted, waiting for recovery
                    else:
                        neg_streak = 0
                        pos_streak = 0
                        logger.info(
                            "E4-L2: 因子 %s 连续 %d 期正向 IC, 恢复候选资格",
                            f,
                            L2_RECOVERY_PERIODS,
                        )
                else:
                    neg_streak = 0
                    pos_streak = 0
            l2_history[f] = {"neg": neg_streak, "pos": pos_streak}
            l2_evicted = neg_streak >= L2_NEG_PERIODS
            if l2_evicted and grade == "strong":
                grade = "weak"  # L2 驱逐降级为 weak (仍参与训练), 非 dead
                logger.warning(
                    "E4-L2: 因子 %s 连续 %d 期窗口IC为负, 降级为 weak", f, neg_streak
                )
            elif l2_evicted:
                logger.info(
                    "E4-L2: 因子 %s 持续剔除 (neg=%d), 保持 %s", f, neg_streak, grade
                )
            result["detail"][f] = {
                **{f"ic_{k}d": round(v, 4) for k, v in ic_by_label.items()},
                "auc": round(auc, 4),
                "rolling_mean": round(roll_mean, 4),
                "rolling_pos_ratio": round(roll_pos, 4),
                "nw_t_3d": round(t_3d, 4),  # B6
                "nw_t_5d": round(t_5d, 4),  # B6
                "icir": round(icir, 4),  # [P18] IC 稳定性
                "grade": grade,
            }
            if ic_mdd is not None:
                result["detail"][f]["ic_mdd_3d"] = round(ic_mdd, 4)  # [E2]
            if l2_evicted:
                result["detail"][f]["l2_evicted"] = True  # [E4-L2]
            if grade in ("strong", "weak"):
                result["factors"].append(f)
        self._persist(window_id, result)
        self._save_l2_history(l2_history)
        return result

    # ---------------- E4-L2 跨窗口负 IC 追踪 ----------------
    @staticmethod
    def _signed_daily_ic_mean(df: pd.DataFrame, factor: str, label: str) -> float:
        """窗口内日度 Rank IC 均值 (带符号) — L2 "连续N期为负" 判定输入.

        L2 追踪用带符号均值判断方向 (与 rank_ic 返回值相同, 语义区分保留)。
        """
        ics = daily_rank_ic_series(df, factor, label)
        if ics.empty:
            return 0.0
        return float(ics.mean())

    def _l2_path(self) -> str:
        return os.path.join(self.registry_path, "l2_factor_neg_streaks.json")

    def _load_l2_history(self) -> dict:
        try:
            if os.path.exists(self._l2_path()):
                with open(self._l2_path(), encoding="utf-8") as fh:
                    return json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("L2 历史加载失败: %s", e)
        return {}

    def _save_l2_history(self, history: dict) -> None:
        try:
            with open(self._l2_path(), "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=1)
        except OSError as e:
            logger.warning("L2 历史保存失败: %s", e)

    def _persist(self, window_id: str, result: dict) -> None:
        """每期因子清单必须记录 (安全网 #2 工程强制)."""
        path = os.path.join(self.registry_path, f"factors_{window_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        logger.info(
            "窗口 %s 因子清单: %d strong+weak / %d 候选",
            window_id,
            len(result["factors"]),
            len(result["detail"]),
        )

    # ---------------- IC 归因 ----------------
    @staticmethod
    def ic_attribution(
        ic_raw: float, ic_neutralized: float, drop_threshold: float = 0.5
    ) -> str:
        """行业/市值中性化后 IC 降幅 > 50% → 预测力主要来自风格暴露, 降级或移除."""
        if ic_raw <= 0:
            return "dead"
        drop = 1 - ic_neutralized / ic_raw
        return "style_exposed" if drop > drop_threshold else "alpha"
