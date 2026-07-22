# -*- coding: utf-8 -*-
"""模型版本管理 (P10.5, ARCH §9.3).

版本注册/切换/回滚; 注册表为 JSON 文件 (默认 app/models/trained/registry.json),
测试可通过 registry_dir 指向 tmp_path。
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"


class ModelRegistry:
    """模型版本注册表.

    JSON 结构::

        {"models": {model_name: {"current": version,
                                  "versions": [{version, metrics, file_path,
                                                registered_at}, ...]}}}
    """

    def __init__(self, registry_dir: str = "app/models/trained") -> None:
        """初始化 (目录不存在则创建; 已有 registry.json 则加载).

        Args:
            registry_dir: 注册表目录 (可配置, 测试传 tmp_path)。
        """
        self.registry_dir = str(registry_dir)
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_path = os.path.join(self.registry_dir, REGISTRY_FILENAME)
        self._data = self._load()

    def _load(self) -> dict:
        """从磁盘加载注册表.

        Returns:
            注册表字典; 文件不存在或损坏时返回空结构。
        """
        if not os.path.exists(self.registry_path):
            return {"models": {}}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "models" not in data:
                raise ValueError("missing 'models' key")
            return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("registry.json 损坏 (%s), 重置为空注册表", exc)
            return {"models": {}}

    def _save(self) -> None:
        """原子写回注册表 (tmp + replace, utf-8)."""
        tmp_path = self.registry_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.registry_path)

    def _entry(self, model_name: str) -> dict:
        """取模型条目 (不存在则报错).

        Args:
            model_name: 模型名。

        Returns:
            {"current": ..., "versions": [...]}。

        Raises:
            KeyError: 模型未注册。
        """
        models = self._data["models"]
        if model_name not in models:
            raise KeyError(f"模型 {model_name} 未注册. 已注册: {sorted(models)}")
        return models[model_name]

    def register(
        self, model_name: str, version: str, metrics: dict, file_path: str
    ) -> None:
        """注册新版本 (含回测指标), 并置为当前生效版本.

        Args:
            model_name: 模型名。
            version: 版本号 (如 "v20260721")。
            metrics: 回测指标 (至少含 oos_ic 以支持 get_best_version)。
            file_path: 模型文件路径。
        """
        models = self._data["models"]
        entry = models.setdefault(model_name, {"current": None, "versions": []})
        entry["versions"] = [v for v in entry["versions"] if v["version"] != version]
        entry["versions"].append(
            {
                "version": version,
                "metrics": dict(metrics),
                "file_path": str(file_path),
                "registered_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        entry["current"] = version
        self._save()
        logger.info(
            "注册模型版本: %s@%s (oos_ic=%s)",
            model_name,
            version,
            metrics.get("oos_ic"),
        )

    def get_current(self, model_name: Optional[str] = None) -> dict:
        """当前生效版本.

        Args:
            model_name: 指定模型; None 时返回所有模型的当前版本映射。

        Returns:
            指定模型: {model_name, version, metrics, file_path};
            未指定: {model_name: {...}}。
        """
        if model_name is None:
            return {
                name: self.get_current(name)
                for name in self._data["models"]
                if self._data["models"][name]["current"] is not None
            }
        entry = self._entry(model_name)
        current = entry["current"]
        if current is None:
            raise KeyError(f"模型 {model_name} 无当前生效版本")
        for v in entry["versions"]:
            if v["version"] == current:
                return {"model_name": model_name, **v}
        raise KeyError(  # pragma: no cover — 防御性
            f"模型 {model_name} 当前版本 {current} 不在版本列表中"
        )

    def set_current(self, version: str, model_name: Optional[str] = None) -> None:
        """切换生效版本.

        Args:
            version: 目标版本号。
            model_name: 目标模型; None 时若注册表仅一个模型则自动推断。

        Raises:
            ValueError: model_name 缺失且无法推断。
            KeyError: 版本不存在。
        """
        if model_name is None:
            names = list(self._data["models"])
            if len(names) != 1:
                raise ValueError(f"注册表含 {len(names)} 个模型, 必须指定 model_name")
            model_name = names[0]
        entry = self._entry(model_name)
        if not any(v["version"] == version for v in entry["versions"]):
            raise KeyError(
                f"模型 {model_name} 无版本 {version}. "
                f"已有: {[v['version'] for v in entry['versions']]}"
            )
        entry["current"] = version
        self._save()
        logger.info("切换生效版本: %s → %s", model_name, version)

    def rollback(self, model_name: str) -> str:
        """回滚到上一版本 (按注册时间), 返回版本号.

        Args:
            model_name: 模型名。

        Returns:
            回滚后的生效版本号。

        Raises:
            ValueError: 历史版本 <2, 无法回滚。
        """
        entry = self._entry(model_name)
        versions = entry["versions"]
        if len(versions) < 2:
            raise ValueError(f"模型 {model_name} 仅 {len(versions)} 个版本, 无法回滚")
        idx = next(
            i for i, v in enumerate(versions) if v["version"] == entry["current"]
        )
        target = versions[idx - 1]["version"] if idx > 0 else versions[-2]["version"]
        entry["current"] = target
        self._save()
        logger.warning("模型 %s 回滚 → %s", model_name, target)
        return target

    def list_versions(self, model_name: str) -> List[dict]:
        """版本列表 (按注册时间倒序, 最新在前).

        Args:
            model_name: 模型名。

        Returns:
            [{version, metrics, file_path, registered_at}, ...]。
        """
        entry = self._entry(model_name)
        return sorted(entry["versions"], key=lambda v: v["registered_at"], reverse=True)

    def get_best_version(self, model_name: str) -> dict:
        """按 OOS IC 取历史最优版本.

        Args:
            model_name: 模型名。

        Returns:
            {version, metrics, file_path, registered_at}。

        Raises:
            ValueError: 无任何带 oos_ic 指标的版本。
        """
        entry = self._entry(model_name)
        scored = [
            v for v in entry["versions"] if v["metrics"].get("oos_ic") is not None
        ]
        if not scored:
            raise ValueError(f"模型 {model_name} 无含 oos_ic 的版本")
        best = max(scored, key=lambda v: v["metrics"]["oos_ic"])
        logger.info(
            "模型 %s 历史最优: %s (oos_ic=%.4f)",
            model_name,
            best["version"],
            best["metrics"]["oos_ic"],
        )
        return best
