"""
特征注册中心 (P19 Auto-Adoption)
===============================
单一真源: data/factor_registry/feature_registry.json
ICScreener 每次筛选后更新 grade/active; FeatureEngine 读取注册中心决定
哪些 dim 执行、哪些特征列保留。

设计原则:
  - registry=None → 全量执行 (向后兼容)
  - registry 存在 → dim 级门控 + 特征列裁剪
  - 自动采纳新面板列 → grade="trial" → IC 验证 → 晋升/淘汰
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Registry JSON schema version ──
SCHEMA_VERSION = 2

# ── Dim group names (must match FeatureEngineV35.build() method calls) ──
DIM_GROUPS = [
    "dim01_price_volume",
    "dim02_volatility",
    "dim03_fundamentals",
    "dim07_limit_gene",
    "dim04_sector_effect",
    "dim05_turnover_liquidity",
    "dim06_valuation_size",
    "dim_active_pit",
    "dim08_calendar_month",
    "dim09_custom_formulas",
    "dim10_money_flow",
    "dim11_float_limits",
    "dim12_ma_system",
    "dim13_holiday",
    "dim14_market_sentiment",
    "dim15_alpha_factors",
    "dim16_candlestick",
    "dim17_extended_factors",
    "dim20_short_horizon",
    "dim18_lhb",
    "dim19_amihud",
    "dim21_chip_tushare",
    "dim22_fundamental_pit",
    "dim23_shareholder_structure",
    "dim24_margin_trading",
    "dim26_lhb_enhanced",
    "dim27_industry_flow",
    "dim28_sector_index",
    "dim29_holdertrade",
    "dim30_kline_geometry",
    "dim31_announcement",
    # Post-processing (not real dims but produce features)
    "_industry_neutralize",
    "_missingness_flags",
    "_time_series_changes",
    "_cross_sectional_ranks",
    "_auto_adopted",  # Phase 2 auto-adoption
]

# ── Template feature transforms for auto-adoption ──
ADOPTION_TEMPLATES = [
    "zscore_20d",
    "chg5d",
    "chg20d",
    "sector_rank",
    "ma5_cross",
    "vol_adj",
]

# ── Columns never treated as feature sources ──
NON_FEATURE_COLS = {
    "symbol", "date", "board", "industry", "name", "tradestatus",
    "announce_date", "report_period", "time", "market_state",
    "schema_version", "ts_code",
}


class FeatureRegistry:
    """特征注册中心 — 单一真源, 驱动 FeatureEngine 门控 + 自动采纳."""

    def __init__(self, path: str = "data/factor_registry/feature_registry.json"):
        self.path = path
        self._data: dict = {"version": SCHEMA_VERSION, "features": {}, "adoption": {}, "last_update": ""}
        if os.path.exists(path):
            self.load()
        else:
            logger.info("FeatureRegistry: 注册中心不存在 %s, 将创建新注册", path)

    # ── Persistence ──────────────────────────────────────────────

    def load(self) -> None:
        """从 JSON 加载注册中心."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                self._data = json.load(fh)
            # Ensure keys exist for backward compat with v1 schema
            self._data.setdefault("version", SCHEMA_VERSION)
            self._data.setdefault("features", {})
            self._data.setdefault("adoption", {})
            self._data.setdefault("last_update", "")
            # Migration: if adoption missing sub-keys, fill defaults
            adoption = self._data["adoption"]
            adoption.setdefault("enabled", False)
            adoption.setdefault("registered_source_cols", [])
            logger.info("FeatureRegistry: 已加载 %s (%d 特征)", self.path, len(self._data["features"]))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("FeatureRegistry: 加载 %s 失败: %s", self.path, exc)
            self._data = {"version": SCHEMA_VERSION, "features": {}, "adoption": {}, "last_update": ""}

    def save(self) -> None:
        """原子写入 (temp file + rename)."""
        self._data["last_update"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=".json", prefix="feature_registry_",
                dir=os.path.dirname(self.path) or ".",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=1)
            shutil.move(tmp, self.path)
            logger.info("FeatureRegistry: 已保存 %s (%d 特征)", self.path, len(self._data["features"]))
        except OSError as exc:
            logger.error("FeatureRegistry: 保存 %s 失败: %s", self.path, exc)

    def _save_as(self, path: str) -> None:
        """Save registry to a specific timestamped path (for Layer1 output)."""
        self._data["last_update"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            fd, tmp = tempfile.mkstemp(
                suffix=".json", prefix="feature_registry_",
                dir=os.path.dirname(path) or ".",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=1)
            shutil.move(tmp, path)
            logger.info("FeatureRegistry: 已保存副本 %s", path)
        except OSError as exc:
            logger.error("FeatureRegistry: 保存 %s 失败: %s", path, exc)

    # ── Queries ──────────────────────────────────────────────────

    @property
    def features(self) -> dict:
        return self._data["features"]

    def get_all(self) -> dict:
        """返回全部特征元数据 {name: meta}."""
        return self._data["features"]

    def get_active(self, dim_group: str | None = None) -> list[str]:
        """返回 active=True 的特征名列表, 可选按 dim_group 过滤."""
        feats = self._data["features"]
        if dim_group is None:
            return [n for n, m in feats.items() if m.get("active", True)]
        return [n for n, m in feats.items()
                if m.get("active", True) and m.get("dim_group") == dim_group]

    def get_by_grade(self, grade: str) -> list[str]:
        """返回指定 grade 的特征名列表."""
        return [n for n, m in self._data["features"].items() if m.get("grade") == grade]

    def get_dim_groups(self) -> set[str]:
        """返回注册中心中所有 dim_group."""
        return {m.get("dim_group", "unknown") for m in self._data["features"].values()}

    def get_active_dim_groups(self) -> set[str]:
        """返回至少有一个 active 特征的 dim_group."""
        active_groups: set[str] = set()
        for name, meta in self._data["features"].items():
            if meta.get("active", True):
                active_groups.add(meta.get("dim_group", "unknown"))
        return active_groups

    def has_dim_group(self, dim_name: str) -> bool:
        """dim 是否有 >=1 个 active 特征 (用于门控)."""
        return dim_name in self.get_active_dim_groups()

    def has_feature(self, name: str) -> bool:
        return name in self._data["features"]

    def get_meta(self, name: str) -> dict | None:
        return self._data["features"].get(name)

    def is_active(self, name: str) -> bool:
        meta = self._data["features"].get(name)
        if meta is None:
            return True  # unknown → active by default
        return meta.get("active", True)

    # ── Mutations ────────────────────────────────────────────────

    def register_new(self, name: str, meta: dict) -> None:
        """注册新特征 (upsert)."""
        meta.setdefault("dim_group", "unknown")
        meta.setdefault("active", True)
        meta.setdefault("grade", "unknown")
        meta.setdefault("created", datetime.now().strftime("%Y-%m-%d"))
        if name in self._data["features"]:
            existing = self._data["features"][name]
            # Preserve creation date on update
            meta.setdefault("created", existing.get("created", meta["created"]))
            existing.update(meta)
        else:
            self._data["features"][name] = meta

    def deactivate(self, names: list[str]) -> None:
        """批量停用特征."""
        for name in names:
            if name in self._data["features"]:
                self._data["features"][name]["active"] = False

    def activate(self, names: list[str]) -> None:
        """批量启用特征."""
        for name in names:
            if name in self._data["features"]:
                self._data["features"][name]["active"] = True

    def update_from_screen(self, result: dict, window_id: str) -> None:
        """从 ICScreener.screen() 结果同步 grade/ICIR/active.

        完整记录注册中心变化: 新登记 / 停用 / 重新激活 / 评级迁移 / IC 跳变.
        """
        detail = result.get("detail", {})
        factors_list = set(result.get("factors", []))

        # ── BEFORE snapshot ──
        before_total = len(self._data["features"])
        before_active = len(self.get_active())
        before_by_grade = {g: len(self.get_by_grade(g)) for g in ("strong", "weak", "trial", "dead", "unknown")}

        # ── Track all changes ──
        new_registrations: list[str] = []
        deactivated: list[tuple[str, str]] = []    # (name, old_grade→new_grade)
        reactivated: list[tuple[str, str]] = []     # (name, old_grade→new_grade)
        grade_changes: list[tuple[str, str, str]] = []  # (name, old_grade, new_grade)
        ic_jumps: list[tuple[str, float, float]] = []   # (name, old_ic, new_ic) — delta > 0.01

        updated = 0
        for fname, d in detail.items():
            grade = d.get("grade", "dead")
            new_ic = max(abs(d.get(f"ic_{k}d", 0.0)) for k in (1, 3, 5))
            old_entry = self._data["features"].get(fname, {})
            old_grade = old_entry.get("grade", "unknown")
            old_active = old_entry.get("active", True)
            old_ic = old_entry.get("ic_abs", 0.0)

            is_new = fname not in self._data["features"]

            meta = {
                "dim_group": old_entry.get("dim_group", "unknown"),
                "active": grade in ("strong", "weak", "trial"),
                "grade": grade,
                "icir": d.get("icir", 0.0),
                "ic_abs": new_ic,
                "rolling_mean": d.get("rolling_mean", 0.0),
                "rolling_pos_ratio": d.get("rolling_pos_ratio", 0.0),
                "last_eval": datetime.now().strftime("%Y-%m-%d"),
                "window_birth": window_id,
                "l2_evicted": d.get("l2_evicted", False),
            }
            self.register_new(fname, meta)
            updated += 1

            # ── Classify the change ──
            if is_new:
                new_registrations.append(fname)
            else:
                new_active = grade in ("strong", "weak", "trial")
                if old_active and not new_active:
                    deactivated.append((fname, f"{old_grade}→{grade}"))
                elif not old_active and new_active:
                    reactivated.append((fname, f"{old_grade}→{grade}"))
                if old_grade != grade:
                    grade_changes.append((fname, old_grade, grade))
                if abs(new_ic - old_ic) > 0.01:
                    ic_jumps.append((fname, old_ic, new_ic))

        # ── AFTER snapshot ──
        after_total = len(self._data["features"])
        after_active = len(self.get_active())
        after_by_grade = {g: len(self.get_by_grade(g)) for g in ("strong", "weak", "trial", "dead", "unknown")}

        # ── Log comprehensive change report ──
        logger.info(
            "═══ Registry Update [%s] ═══\n"
            "  Total:    %d → %d features (%+d)\n"
            "  Active:   %d → %d (%+d)\n"
            "  Grade:    strong %d→%d  weak %d→%d  trial %d→%d  dead %d→%d  unknown %d→%d",
            window_id,
            before_total, after_total, after_total - before_total,
            before_active, after_active, after_active - before_active,
            before_by_grade["strong"], after_by_grade["strong"],
            before_by_grade["weak"], after_by_grade["weak"],
            before_by_grade["trial"], after_by_grade["trial"],
            before_by_grade["dead"], after_by_grade["dead"],
            before_by_grade["unknown"], after_by_grade["unknown"],
        )

        if new_registrations:
            logger.info("  NEW [+%d]: %s", len(new_registrations), new_registrations[:15])
        if deactivated:
            names = [f"{n}({t})" for n, t in deactivated[:15]]
            logger.info("  DEACTIVATED [%d]: %s", len(deactivated), names)
        if reactivated:
            names = [f"{n}({t})" for n, t in reactivated[:15]]
            logger.info("  REACTIVATED [%d]: %s", len(reactivated), names)
        if grade_changes:
            by_transition: dict[str, list[str]] = {}
            for name, old_g, new_g in grade_changes:
                by_transition.setdefault(f"{old_g}→{new_g}", []).append(name)
            for trans, names in sorted(by_transition.items()):
                logger.info("  GRADE [%s]: %d — %s", trans, len(names), names[:10])
        if ic_jumps:
            jumps_str = ", ".join(f"{n}:{old:.4f}→{new:.4f}" for n, old, new in ic_jumps[:10])
            logger.info("  IC-JUMP [%d]: %s", len(ic_jumps), jumps_str)

        logger.info("═══ End Registry Update [%s] ═══", window_id)

    def enable_adoption(self) -> None:
        """开启自动采纳."""
        self._data["adoption"]["enabled"] = True
        self._data["adoption"].setdefault("registered_source_cols", [])
        logger.info("FeatureRegistry: auto-adoption ENABLED")

    def disable_adoption(self) -> None:
        """关闭自动采纳."""
        self._data["adoption"]["enabled"] = False
        logger.info("FeatureRegistry: auto-adoption DISABLED")

    def is_adoption_enabled(self) -> bool:
        return self._data.get("adoption", {}).get("enabled", False)

    def mark_source_cols_registered(self, cols: list[str]) -> None:
        """记录已被采纳的源列 (避免重复生成)."""
        existing = set(self._data["adoption"].get("registered_source_cols", []))
        existing.update(cols)
        self._data["adoption"]["registered_source_cols"] = sorted(existing)

    def get_registered_source_cols(self) -> set[str]:
        return set(self._data["adoption"].get("registered_source_cols", []))

    def prune_stale(self, min_windows: int = 4) -> int:
        """停用 stale 特征 (grade=dead 且从未被 training 使用的特征).

        Returns: 停用数量.
        """
        count = 0
        for name, meta in self._data["features"].items():
            if meta.get("grade") == "dead" and meta.get("active", True):
                meta["active"] = False
                count += 1
        if count:
            logger.info("FeatureRegistry: prune_stale — %d features deactivated", count)
        return count

    # ── Seeding ──────────────────────────────────────────────────

    def _seed(self, df_sample: pd.DataFrame) -> int:
        """一次性迁移: 在小样本上运行全量 FeatureEngine, 按 dim 发现特征→dim_group 映射.

        Args:
            df_sample: 小面板 (如 5 stocks × 200 days, 已含必要列).

        Returns:
            注册的特征数量.
        """
        from app.pipeline1.cleaning_pipeline import board_of
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        df_sample = df_sample.copy()
        # Normalize board: fix raw Tushare market codes (SZ/SH/BJ) → cleaned board names
        if "board" in df_sample.columns:
            raw_boards = df_sample["board"].unique()
            needs_fix = any(b in ("SZ", "SH", "BJ") for b in raw_boards)
            if needs_fix:
                logger.info("_seed: normalizing board values (SZ/SH/BJ → main/GEM/STAR)")
                board_map = {s: board_of(s) for s in df_sample["symbol"].unique()}
                df_sample["board"] = df_sample["symbol"].map(board_map)
                # Fallback: keep existing if already valid
                valid_boards = {"main", "GEM", "STAR"}
                existing_valid = df_sample["board"].isin(valid_boards)
                df_sample.loc[~existing_valid, "board"] = df_sample.loc[~existing_valid, "symbol"].map(board_map)

        fe = FeatureEngineV35()
        # Run full build to get all feature columns
        df_before = df_sample.copy()
        df_full = fe.build(df_before.copy())
        all_feature_cols = FeatureEngineV35.feature_columns(df_full)

        # Now run each dim in isolation to attribute columns
        df_iso = df_before.copy()
        dim_methods = [
            m for m in DIM_GROUPS
            if m in dir(fe) and not m.startswith("_industry") and not m.startswith("_missingness")
            and not m.startswith("_time_series") and not m.startswith("_cross_sectional")
            and not m.startswith("_auto_adopted")
        ]
        for dim_name in dim_methods:
            try:
                before = set(df_iso.columns)
                dim_func = getattr(fe, dim_name)
                df_iso = dim_func(df_iso)
                after = set(df_iso.columns)
                new_cols = after - before
                for col in sorted(new_cols):
                    if col in all_feature_cols or col in df_full.columns:
                        self.register_new(col, {
                            "dim_group": dim_name,
                            "active": True,
                            "grade": "unknown",
                            "source_cols": [],
                            "transform": dim_name,
                            "created": datetime.now().strftime("%Y-%m-%d"),
                            "last_eval": "",
                        })
            except Exception as exc:
                logger.warning("FeatureRegistry._seed: dim %s 失败, 跳过: %s", dim_name, exc)

        # Post-processing groups (industry_neutralize, missingness, time_series_changes, xrank)
        df_post = fe.industry_neutralize(df_iso)
        industry_cols = set(df_post.columns) - set(df_iso.columns)
        for col in sorted(industry_cols):
            self.register_new(col, {
                "dim_group": "_industry_neutralize",
                "active": True,
                "grade": "unknown",
                "source_cols": [],
                "transform": "industry_rank",
                "created": datetime.now().strftime("%Y-%m-%d"),
                "last_eval": "",
            })
        df_iso = df_post

        df_post = fe.add_missingness_flags(df_iso)
        miss_cols = set(df_post.columns) - set(df_iso.columns)
        for col in sorted(miss_cols):
            self.register_new(col, {
                "dim_group": "_missingness_flags",
                "active": True,
                "grade": "unknown",
                "source_cols": [],
                "transform": "isna_flag",
                "created": datetime.now().strftime("%Y-%m-%d"),
                "last_eval": "",
            })
        df_iso = df_post

        df_post = fe._add_time_series_changes(df_iso)
        ts_cols = set(df_post.columns) - set(df_iso.columns)
        for col in sorted(ts_cols):
            self.register_new(col, {
                "dim_group": "_time_series_changes",
                "active": True,
                "grade": "unknown",
                "source_cols": [],
                "transform": "time_series_chg",
                "created": datetime.now().strftime("%Y-%m-%d"),
                "last_eval": "",
            })
        df_iso = df_post

        df_post = fe._add_cross_sectional_ranks(df_iso)
        xr_cols = set(df_post.columns) - set(df_iso.columns)
        for col in sorted(xr_cols):
            self.register_new(col, {
                "dim_group": "_cross_sectional_ranks",
                "active": True,
                "grade": "unknown",
                "source_cols": [],
                "transform": "board_rank",
                "created": datetime.now().strftime("%Y-%m-%d"),
                "last_eval": "",
            })

        n = len(self._data["features"])
        logger.info(
            "FeatureRegistry._seed: %d features discovered across %d dim groups",
            n, len(self.get_dim_groups()),
        )
        self.save()
        return n

    # ── Stats ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """返回注册中心摘要统计."""
        feats = self._data["features"]
        grades = {"strong": 0, "weak": 0, "trial": 0, "dead": 0, "unknown": 0}
        active_count = 0
        for m in feats.values():
            g = m.get("grade", "unknown")
            grades[g] = grades.get(g, 0) + 1
            if m.get("active", True):
                active_count += 1
        return {
            "total_features": len(feats),
            "active": active_count,
            "inactive": len(feats) - active_count,
            "by_grade": grades,
            "dim_groups": len(self.get_dim_groups()),
            "active_dim_groups": len(self.get_active_dim_groups()),
            "adoption_enabled": self.is_adoption_enabled(),
            "registered_source_cols": len(self.get_registered_source_cols()),
            "last_update": self._data.get("last_update", ""),
        }
