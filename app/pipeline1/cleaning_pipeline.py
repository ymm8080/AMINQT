"""
数据清洗管线 0→4 (DESIGN §14.2, 安全网 #3/#6/#8/#11/#13/#14)
==============================================================
顺序严格 0 -> 1 -> 2 -> 3 -> 4; 主板/双创分治 (独立清洗独立排名).
步骤 4 仅用于实盘推理端, 训练集不受限.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAIN_BOARD_PREFIXES = ("60", "000", "001", "002", "003")
GEM_PREFIXES = ("300", "301")
STAR_PREFIXES = ("68",)


def board_of(code: str) -> str:
    """板块识别: main / GEM / STAR (与 V3.5 清单 schema 的 board 字段一致)."""
    code = str(code).split(".")[0]
    if code.startswith(STAR_PREFIXES):
        return "STAR"
    if code.startswith(GEM_PREFIXES):
        return "GEM"
    return "main"


def get_limit_pct(board: str, date: pd.Timestamp) -> float:
    """涨跌幅分段 (安全网 #6): 创业板 2020-08-24 前 10% 后 20%."""
    if board == "main":
        return 0.10
    if board == "STAR":
        return 0.20
    if board == "GEM":
        return 0.10 if date < pd.Timestamp("2020-08-24") else 0.20
    raise ValueError(f"Unknown board: {board}")


def limit_up_price(pre_close: float, limit_pct: float) -> float:
    """涨停价精确计算: 按分四舍五入 (安全网 #6)."""
    return round(pre_close * (1 + limit_pct), 2)


def is_limit_up(close: float, pre_close: float, limit_pct: float) -> bool:
    """涨停判定: B5 相对容差 max(0.01, 涨停价×0.1%), 高价股0.1%低价股保底0.01."""
    lu = limit_up_price(pre_close, limit_pct)
    return abs(close - lu) < max(0.01, lu * 0.001)


@dataclass
class CleaningConfig:
    """清洗阈值 (边界+初始值, 可被调参层覆盖)."""

    min_list_days: int = 250  # 上市 >= 250 交易日 (>= 最长特征窗口)
    min_amount: float = 5e7  # T 日成交额 >= 5000 万
    liquidity_top_n: int = (
        400  # 板块内流动性 Score 前 N (原200太保守, 3年数据可支撑更大池)
    )
    score_w_amount: float = 0.5  # Score = w*rank(成交额) + (1-w)*rank(自由流通换手)
    stability_window: int = 5  # D24 换手稳定性窗口
    stability_max: float = 0.5  # std/mean > 0.5 → 对倒嫌疑
    new_stock_days: int = 5  # 注册制新股 (<5日无涨跌幅限制)
    abs_amount_floor: float = 8e7  # 步骤4 绝对流动性安全阀 8000 万
    bottom_amount_pct: float = 0.2  # 步骤5 [E6] 剔除成交额全市场后 20% (流动性黑洞预防)
    valve_full: int = 50  # 过滤后 >= 50: 正常
    valve_reduced: int = 15  # >= 15: 减仓输出; < 15: 强制空清单
    delisted_virtual_ret: float = -0.5  # 退市股虚拟 T+1 收益 (安全网 #14)


class CleaningPipeline:
    """清洗 0→4. 输入: 全市场日线面板 (多 symbol × 多 date, 含 data_supply 标准列)."""

    def __init__(self, cfg: CleaningConfig | None = None):
        self.cfg = cfg or CleaningConfig()

    # ---------------- 步骤 0: 板块分治 ----------------
    @staticmethod
    def step0_board_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """主板 / 双创板 (GEM+STAR) 拆分 — 后续全流程独立."""
        if "board" not in df.columns:
            df = df.copy()
            df["board"] = df["symbol"].map(board_of)
        main = df[df["board"] == "main"]
        dual = df[df["board"].isin(["GEM", "STAR"])]
        return main, dual

    # ---------------- 步骤 1: 基础状态 ----------------
    def step1_base_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔 ST/*ST; 剔上市 < 250 交易日 (>= 最长特征窗口 250 日均线)."""
        out = df[~df["is_st"].astype(bool)]
        out = out[out["list_days"] >= self.cfg.min_list_days]
        return out

    # ---------------- 步骤 2: 流动性底线 ----------------
    def step2_liquidity(
        self, df: pd.DataFrame, apply_top_n: bool = True
    ) -> pd.DataFrame:
        """成交额 >= 5000万; 板块内流动性 Score 前 N (按 date+board 独立排名).

        Score = w*rank_pct(turnover_value) + (1-w)*rank_pct(free_float_turnover_rate)
        D24 换手稳定性: std(turnover_5d)/mean(turnover_5d) > 0.5 → 对倒嫌疑,
        附加 turnover_stability_5 列供模型学习 (不硬剔除).

        apply_top_n: True=推理端截取前N (默认), False=训练端仅保留 amount 底线.
        训练端不截断: 模型需学习全谱股票 (含中小流动性), 仅推理端需要候选清单收敛.
        """
        cfg = self.cfg
        out = df[df["amount"] >= cfg.min_amount].copy()
        grp = [out["date"], out["board"]]
        out["rank_amount"] = (
            out.groupby(grp[0])["amount"].rank(pct=True)
            if "board" not in out
            else out.groupby(["date", "board"])["amount"].rank(pct=True)
        )
        ff = out.get("free_float_turnover_rate", out["turnover_rate"])
        out["rank_ff_turnover"] = ff.groupby([out["date"], out["board"]]).rank(pct=True)
        out["liquidity_score"] = (
            cfg.score_w_amount * out["rank_amount"]
            + (1 - cfg.score_w_amount) * out["rank_ff_turnover"]
        )

        # D24 换手稳定性 (groupby symbol!)
        out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
        g = out.groupby("symbol")["turnover_rate"]
        std5 = g.rolling(cfg.stability_window).std().reset_index(level=0, drop=True)
        mean5 = g.rolling(cfg.stability_window).mean().reset_index(level=0, drop=True)
        out["turnover_stability_5"] = (std5 / mean5.replace(0, np.nan)).fillna(0.0)
        out["churn_suspect"] = (out["turnover_stability_5"] > cfg.stability_max).astype(
            int
        )

        if not apply_top_n:
            return out

        # 板块内每个 date 取 Score 前 N (仅推理端)
        out["score_rank"] = out.groupby(["date", "board"])["liquidity_score"].rank(
            ascending=False, method="first"
        )
        return out[out["score_rank"] <= cfg.liquidity_top_n]

    # ---------------- 步骤 3: 极端数据 ----------------
    def step3_extreme(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔当日停牌 / 复牌首日 (安全网 #11) / 关键字段缺失 / 注册制新股 (<5日)."""
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        # 复牌首日必须在剔除停牌行之前判定 (否则 shift 看到的是上一未停牌行)
        prev_susp = df.groupby("symbol")["is_suspended"].shift(1).fillna(0).astype(bool)
        resume_first_day = prev_susp & (~df["is_suspended"].astype(bool))
        out = df[(~df["is_suspended"].astype(bool)) & (~resume_first_day)]
        key_cols = ["open", "high", "low", "close", "close_hfq", "volume", "amount"]
        out = out.dropna(subset=key_cols)
        out = out[out["list_days"] >= self.cfg.new_stock_days]
        return out

    # ---------------- 步骤 4: 可成交性 (仅推理端) ----------------
    def step4_tradability(
        self, df: pd.DataFrame, inference_only: bool = True
    ) -> tuple[pd.DataFrame, str]:
        """一字涨停剔除 + 8000万安全阀. 训练集 (inference_only=False) 不受限.

        Returns:
            (过滤后数据, 阀门状态): 'full' 正常 / 'reduced' 减仓输出 / 'empty' 强制空清单
        """
        if not inference_only:
            return df, "full"
        out = df.copy()
        out["limit_pct"] = [
            get_limit_pct(b, d) for b, d in zip(out["board"], out["date"])
        ]
        out["limit_up_price"] = (out["pre_close"] * (1 + out["limit_pct"])).round(2)
        tol = np.maximum(0.01, out["limit_up_price"] * 0.001)  # B5: 相对容差
        out["is_limit_up_close"] = (
            abs(out["close"] - out["limit_up_price"]) < tol
        ).astype(int)
        out["is_one_word_limit"] = (
            out["is_limit_up_close"].astype(bool)
            & (abs(out["open"] - out["limit_up_price"]) < 0.01)
            & (abs(out["high"] - out["low"]) < 0.01)
        ).astype(int)
        out = out[out["is_one_word_limit"] == 0]
        out = out[out["amount"] >= self.cfg.abs_amount_floor]

        n = out["symbol"].nunique() if len(out) else 0
        if n >= self.cfg.valve_full:
            state = "full"
        elif n >= self.cfg.valve_reduced:
            state = "reduced"
            logger.warning("流动性安全阀: 仅剩 %d 只, 减仓输出", n)
        else:
            state = "empty"
            logger.error(
                "流动性安全阀: 仅剩 %d 只 < %d, 强制空清单", n, self.cfg.valve_reduced
            )
        return out, state

    # ---------------- 步骤 5: 成交额后 20% 剔除 (E6, V3.8) ----------------
    def step5_amount_bottom(self, df: pd.DataFrame) -> pd.DataFrame:
        """[E6] 剔除当日成交额全市场后 20% 的票 (按 date 横截面 rank_pct).

        流动性黑洞预防: 买错后卖不掉是最致命死法. 训练端与推理端同步执行.
        """
        pct = self.cfg.bottom_amount_pct
        if pct <= 0:
            return df
        rank = df.groupby("date")["amount"].rank(pct=True)
        return df[rank > pct]

    # ---------------- 退市股虚拟归零 (安全网 #14, D20) ----------------
    def inject_delisted_virtual_rows(
        self, df: pd.DataFrame, delisted_symbols: list[str]
    ) -> pd.DataFrame:
        """退市股虚拟 T+1: 收盘价×0.5, label_1d=-50%, 让模型学到归零风险.

        [B18] 标记 is_virtual=1 (真实行=0): label_engine 缩尾时豁免,
        防止 -50% 被剪至当日 0.1% 分位.
        """
        if "is_virtual" not in df.columns:
            df["is_virtual"] = 0
        rows = []
        for sym in delisted_symbols:
            sub = df[df["symbol"] == sym]
            if len(sub) == 0:
                continue
            last = sub.loc[sub["date"].idxmax()].copy()
            last["date"] = last["date"] + pd.Timedelta(days=1)
            last["close"] = last["close"] * (1 + self.cfg.delisted_virtual_ret)
            last["close_hfq"] = last["close_hfq"] * (1 + self.cfg.delisted_virtual_ret)
            last["is_virtual"] = 1
            rows.append(last)
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        return df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # ---------------- 总装 ----------------
    def run_train(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """训练端清洗 (步骤 0→3 + 步骤5[E6], 不做步骤 4). 返回 (主板, 双创).

        训练端 step2 不截取 top-N: 模型需学习全谱股票 (含中小流动性),
        仅推理端需要候选清单收敛 (run_inference 保留 top-N).
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
        main, dual = self.step0_board_split(df)
        return (
            self.step5_amount_bottom(
                self.step3_extreme(
                    self.step2_liquidity(self.step1_base_state(main), apply_top_n=False)
                )
            ),
            self.step5_amount_bottom(
                self.step3_extreme(
                    self.step2_liquidity(self.step1_base_state(dual), apply_top_n=False)
                )
            ),
        )

    def run_inference(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """推理端清洗 (步骤 0→4 + 步骤5[E6]). 返回 (主板, 双创, 阀门状态).

        推理端 step2 截取 top-N: 候选清单需要收敛 (训练端不截断, 推理端截断).
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        main, dual = self.step0_board_split(df)
        main = self.step3_extreme(
            self.step2_liquidity(self.step1_base_state(main), apply_top_n=True)
        )
        dual = self.step3_extreme(
            self.step2_liquidity(self.step1_base_state(dual), apply_top_n=True)
        )
        both = pd.concat([main, dual], ignore_index=True)
        both, state = self.step4_tradability(both, inference_only=True)
        both = self.step5_amount_bottom(both)  # [E6] 成交额后 20% 剔除
        m, d = self.step0_board_split(both)
        return m, d, state
