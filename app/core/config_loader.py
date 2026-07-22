# -*- coding: utf-8 -*-
"""统一配置加载器 (P8, ARCH §6.3).

加载 selection_config.yaml / trading_config.yaml / training_config.yaml /
adaptive_config.yaml / llm_config.yaml, 提供字典式访问与默认值合并。
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def load_config(name: str) -> dict:
    """加载指定 YAML 配置.

    Args:
        name: 配置名 (不含后缀), 如 "selection_config"。

    Returns:
        配置字典; 文件不存在或内容为空/非字典时返回 {} (不存在时告警)。

    Raises:
        yaml.YAMLError: 配置语法错误。
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        logger.warning("配置文件不存在: %s, 返回空配置 {}", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        logger.warning("配置文件为空: %s, 返回 {}", path)
        return {}
    if not isinstance(cfg, dict):
        logger.warning(
            "配置文件顶层非字典: %s (type=%s), 返回 {}", path, type(cfg).__name__
        )
        return {}

    logger.debug("配置加载完成: %s (%d 个顶层键)", path, len(cfg))
    return cfg


def get(config: dict, dotted_key: str, default: Any = None) -> Any:
    """点号路径取值: get(cfg, "scoring.model_weight", 0.6).

    Args:
        config: 配置字典。
        dotted_key: 点号分隔的嵌套键路径。
        default: 任一层缺失或中间层非字典时返回的默认值。

    Returns:
        取到的值或 default。
    """
    if not isinstance(config, dict):
        return default
    cur: Any = config
    for part in str(dotted_key).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur
