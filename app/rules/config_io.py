# -*- coding: utf-8 -*-
"""
规则引擎 Config 持久化 (YAML roundtrip)
============================================
Config dataclass ↔ config/rules_config.yaml.
只持久化 TUNABLE 字段 (其余为协议常量); 加载时未知字段忽略并告警.
"""

from __future__ import annotations

import logging
import os
from dataclasses import fields

import yaml

from .config import Config, TUNABLE_BOUNDS

logger = logging.getLogger(__name__)

DEFAULT_PATH = "config/rules_config.yaml"


def save_config(cfg: Config, path: str = DEFAULT_PATH) -> None:
    """保存全部字段到 YAML."""
    data = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=True)
        logger.info("规则配置已保存: %s", path)
    except OSError as e:
        logger.exception("规则配置保存失败: %s (%s)", path, e)
        raise


def load_config(path: str = DEFAULT_PATH) -> Config:
    """从 YAML 加载 Config; 文件不存在返回默认值; 未知字段忽略并告警."""
    cfg = Config()
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError as e:
        logger.exception("规则配置加载失败: %s (%s)", path, e)
        return cfg
    valid = {f.name for f in fields(cfg)}
    for key, value in data.items():
        if key in valid:
            try:
                setattr(cfg, key, type(getattr(cfg, key))(value))
            except (TypeError, ValueError):
                logger.warning("配置字段 %s 类型转换失败, 使用默认值", key)
        else:
            logger.warning("未知配置字段忽略: %s", key)
    return cfg


def validate_tunable(updates: dict) -> tuple[dict, list[str]]:
    """校验 TUNABLE 更新: 边界内接受, 越界/未知拒绝.

    Returns:
        (accepted, rejected_reasons)
    """
    accepted, rejected = {}, []
    for name, value in updates.items():
        if name not in TUNABLE_BOUNDS:
            rejected.append(f"{name}: 非 TUNABLE 字段")
            continue
        lo, hi, _ = TUNABLE_BOUNDS[name]
        try:
            v = float(value)
        except (TypeError, ValueError):
            rejected.append(f"{name}: 无法转换为数值 ({value})")
            continue
        if not (lo <= v <= hi):
            rejected.append(f"{name}: {v} 越界 [{lo}, {hi}]")
            continue
        accepted[name] = v
    return accepted, rejected
