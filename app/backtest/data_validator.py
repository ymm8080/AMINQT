"""模块2: DataValidator — 数据完整性校验.

回测前检查:
    - E001: 未复权价格检测
    - E002: 缺失必要列
    - E003: close <= 0
    - E004: high < low
    - E005: 停牌状态推断异常
    - E006: 次新股未过滤
    - E007: PIT 数据警告
    - E008: 基准数据缺失
    - E009: 成交量确认数据缺失 (V5.0)
    - E010: 大盘指数数据缺失 (V5.0)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """数据完整性校验器.

    在回测前对预测表和行情表进行全面检查.
    """

    ERROR_CODES = {
        "E001": "Unadjusted prices detected",
        "E002": "Missing required columns",
        "E003": "Dirty data: close <= 0",
        "E004": "Dirty data: high < low",
        "E005": "Halt status inference anomaly",
        "E006": "IPO stocks not filtered",
        "E007": "PIT data warning",
        "E008": "Benchmark data missing",
        "E009": "Volume confirmation data missing",
        "E010": "Market index data missing for drop filter",
    }

    # 后复权检测: 如果存在除权除息, 复权价格应连续
    # 原始价格在除权日会有跳空, 后复权价格不应有大幅跳空
    ADJUSTED_JUMP_THRESHOLD = 0.18  # 18% 单日跳空视为未复权

    def __init__(
        self,
        price_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        benchmark_df: pd.DataFrame | None = None,
        market_df: pd.DataFrame | None = None,
    ):
        """初始化校验器.

        Args:
            price_df: 行情表.
            pred_df: 预测表.
            benchmark_df: 基准表 (可选).
            market_df: 大盘指数表 (V5.0, 可选).
        """
        self.price_df = price_df
        self.pred_df = pred_df
        self.benchmark_df = benchmark_df
        self.market_df = market_df

    def validate_prices(self) -> list[tuple[str, str]]:
        """校验行情表.

        Returns:
            错误列表 [(error_code, detail), ...].
        """
        errors: list[tuple[str, str]] = []

        # E003: close <= 0
        bad_close = self.price_df[self.price_df["close"] <= 0]
        if not bad_close.empty:
            errors.append(("E003", f"{len(bad_close)} rows with close <= 0, dropped"))

        # E004: high < low
        bad_hl = self.price_df[self.price_df["high"] < self.price_df["low"]]
        if not bad_hl.empty:
            errors.append(("E004", f"{len(bad_hl)} rows with high < low, dropped"))

        # E004 补充: open > high 或 open < low
        bad_oh = self.price_df[
            (self.price_df["open"] > self.price_df["high"])
            | (self.price_df["open"] < self.price_df["low"])
        ]
        if not bad_oh.empty:
            errors.append(("E004", f"{len(bad_oh)} rows with open out of [low, high]"))

        # E005: 成交量为0但有收盘价 → 推断停牌
        vol_zero = self.price_df[
            (self.price_df["volume"] <= 0) & (self.price_df["close"] > 0)
        ]
        if not vol_zero.empty:
            errors.append(
                ("E005", f"{len(vol_zero)} rows vol=0 but close>0, mark halt")
            )

        logger.info("行情表校验: %d 个问题", len(errors))
        return errors

    def validate_predictions(self) -> list[tuple[str, str]]:
        """校验预测表.

        Returns:
            错误列表.
        """
        errors: list[tuple[str, str]] = []

        # 检查概率范围
        for h in [1, 2, 4]:
            col = f"prob_up_h{h}"
            if col in self.pred_df.columns:
                out_of_range = self.pred_df[
                    (self.pred_df[col] < 0) | (self.pred_df[col] > 1)
                ]
                if not out_of_range.empty:
                    errors.append(
                        ("E002", f"{len(out_of_range)} rows {col} out of [0,1]")
                    )

        logger.info("预测表校验: %d 个问题", len(errors))
        return errors

    def check_adjusted_prices(self) -> bool:
        """检测是否为后复权价格.

        方法: 检查是否存在单日跌幅 > 18% 且次日恢复的情况
        (除权除息的特征: 原始价格跳空, 后复权不跳空).

        Returns:
            True 如果检测到未复权 (有问题), False 如果看起来已复权.
        """
        if self.price_df.empty:
            return False

        issues = 0
        for _stock, grp in self.price_df.groupby("stock"):
            grp = grp.sort_values("date")
            if len(grp) < 3:
                continue
            rets = grp["close"].pct_change()
            # 除权日: 大幅下跌后次日恢复
            big_drops = rets < -self.ADJUSTED_JUMP_THRESHOLD
            if big_drops.any():
                for idx in big_drops[big_drops].index:
                    pos = grp.index.get_loc(idx)
                    if pos + 1 < len(grp):
                        next_ret = rets.iloc[pos + 1]
                        # 如果次日大涨 (恢复), 说明是除权
                        if next_ret > self.ADJUSTED_JUMP_THRESHOLD:
                            issues += 1

        if issues > 0:
            logger.warning("检测到 %d 处疑似未复权跳空 (E001)", issues)
            return True
        return False

    def check_pit_data(self) -> list[str]:
        """PIT 数据警告.

        检查 is_st / is_halt / board 是否可能包含未来信息.

        Returns:
            警告列表.
        """
        warnings: list[str] = []

        # 检查 board 列是否在预测表中 (应在 T 日快照)
        if "board" not in self.pred_df.columns:
            warnings.append("预测表缺少 board 列, 可能使用最新板块 (PIT 风险)")

        # 检查 is_st / is_halt 是否在行情表中 (应基于 T 日)
        for col in ["is_st", "is_halt"]:
            if col not in self.price_df.columns:
                warnings.append(f"行情表缺少 {col} 列")

        return warnings

    def infer_halt_status(self) -> pd.DataFrame:
        """推断停牌状态.

        规则: volume <= 0 且 close > 0 → is_halt = 1
        volume < 1000 (极低) → is_halt = 1

        Returns:
            修正后的 price_df.
        """
        df = self.price_df.copy()
        # Ensure is_halt is int (input may be bool — newer pandas rejects int→bool assignment)
        if "is_halt" in df.columns:
            df["is_halt"] = df["is_halt"].astype(int)
        else:
            df["is_halt"] = 0

        # volume <= 0 且 close > 0 → halt
        mask = (df["volume"] <= 0) & (df["close"] > 0)
        df.loc[mask, "is_halt"] = 1

        # volume 极低 (< 1000) → halt
        mask_low = (df["volume"] < 1000) & (df["volume"] > 0) & (df["close"] > 0)
        df.loc[mask_low, "is_halt"] = 1

        return df

    def filter_ipo_stocks(self, min_listing_days: int = 60) -> pd.DataFrame:
        """过滤次新股 (上市 < 60 交易日).

        Args:
            min_listing_days: 最小上市天数.

        Returns:
            过滤后的 price_df (删除上市不足 min_listing_days 的记录).
        """
        df = self.price_df.copy()
        to_drop: list[int] = []

        for _stock, grp in df.groupby("stock"):
            grp = grp.sort_values("date")
            if len(grp) < min_listing_days:
                # 整个股票数据不足, 删除前 min_listing_days 天
                to_drop.extend(grp.index[:min_listing_days].tolist())
            else:
                # 删除前 min_listing_days 天
                to_drop.extend(grp.index[:min_listing_days].tolist())

        df = df.drop(index=to_drop)
        logger.info(
            "IPO过滤: 删除 %d 行次新股数据 (min_listing_days=%d)",
            len(to_drop),
            min_listing_days,
        )
        return df

    def run_all_checks(self) -> list[tuple[str, str]]:
        """运行所有校验.

        Returns:
            所有错误和警告列表.
        """
        all_issues: list[tuple[str, str]] = []
        all_issues.extend(self.validate_prices())
        all_issues.extend(self.validate_predictions())

        if self.check_adjusted_prices():
            all_issues.append(("E001", "Unadjusted prices detected"))

        if self.benchmark_df is None or self.benchmark_df.empty:
            all_issues.append(("E008", "Benchmark data missing"))

        # V5.0 E009: 成交量确认数据检查
        if "volume" not in self.price_df.columns:
            all_issues.append(("E009", "Volume column missing for confirmation ratio"))

        # V5.0 E010: 大盘指数数据检查
        if self.market_df is None or self.market_df.empty:
            all_issues.append(("E010", "Market index data missing for drop filter"))

        return all_issues
