"""关键数据文件备份 — 防 automation git 操作误删.

背景: 主仓库目录的 automation 会执行 checkout/reset --hard, 工作区中
被 git 跟踪的 parquet 会被物理删除 (2026-07-31 三波删除实录).
每日 pipeline 结束后, 将 config/data_pipeline_config.yaml::backup.keepers
列出的文件复制到仓库外备份目录.

规则:
- 备份文件名 <src_stem>__<trade_date><suffix> (双下划线分隔, WORM 不覆盖)
- keeper 支持 glob, 取 mtime 最新的匹配文件 (适配带时间戳的文件名)
- retention: 每个 keeper 家族只保留最新 N 份, 清理最旧
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _newest_match(root: Path, pattern: str) -> Path | None:
    """返回匹配 pattern 的文件中 mtime 最新者, 无匹配返回 None.

    pattern 为仓库内相对路径时相对 root 解析; 绝对路径 (仓库外主数据,
    如 D:/AMINQT/PARQUET/) 按自身解析, 同样支持 glob.
    """
    p = Path(pattern)
    if p.is_absolute():
        matches = [x for x in p.parent.glob(p.name) if x.is_file()]
    else:
        matches = [x for x in root.glob(pattern) if x.is_file()]
    return max(matches, key=lambda x: x.stat().st_mtime) if matches else None


def _prune_glob(pattern: str, suffix: str) -> str:
    """由 keeper pattern 推导备份家族的清理 glob.

    静态名 'data/cyq_panel.parquet' -> 'cyq_panel__*.parquet'
      (精确到双下划线, 不会误伤 cyq_panel_full 家族)
    通配名 '.../features_main_*.parquet' -> 'features_main_*__*.parquet'
      (覆盖不同时间戳源文件产生的所有备份)
    """
    stem = Path(pattern).stem
    if "*" in stem:
        prefix = stem.split("*")[0]
        return f"{prefix}*__*{suffix}"
    return f"{stem}__*{suffix}"


def backup_keepers(
    root: str | Path,
    backup_dir: str | Path,
    keepers: list[str],
    trade_date: str,
    retention: int = 2,
) -> dict[str, str]:
    """把 keepers 列出的关键文件复制到 backup_dir.

    Args:
        root: 仓库根目录 (keeper 路径相对于它).
        backup_dir: 备份目录 (必须在仓库外), 不存在则创建.
        keepers: 相对路径列表, 支持 glob.
        trade_date: 'YYYYMMDD', 追加到备份文件名 (WORM).
        retention: 每个 keeper 家族保留最新 N 份.

    Returns:
        {keeper_pattern: "ok (<name>)" | "exists (<name>)" | "skip (not found)"}
    """
    root = Path(root)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for pattern in keepers:
        src = _newest_match(root, pattern)
        if src is None:
            results[pattern] = "skip (not found)"
            logger.warning("backup skip, not found: %s", pattern)
            continue

        dst = backup_dir / f"{src.stem}__{trade_date}{src.suffix}"
        if dst.exists():
            results[pattern] = f"exists ({dst.name})"  # WORM: 不覆盖
            logger.info("backup exists, skip: %s", dst.name)
        else:
            shutil.copy2(src, dst)
            results[pattern] = f"ok ({dst.name})"
            logger.info("backup: %s -> %s", src, dst)

        # retention: 清理该 keeper 家族最旧的备份
        family = sorted(
            backup_dir.glob(_prune_glob(pattern, src.suffix)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in family[retention:]:
            old.unlink()
            logger.info("backup prune: %s", old.name)

    return results
