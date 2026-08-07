"""双向同步 backtest_design.md.

副本1: docs/backtest_design.md (项目代码库)
副本2: d:/AMINQT/REFERENCE/Design All/Function Spec/BACKTESTING/backtest_design.md

同步逻辑: 比较两文件修改时间, 较新的覆盖较旧的.
如果内容不同, 打印diff摘要.

用法:
    python scripts/sync_backtest_design.py          # 同步
    python scripts/sync_backtest_design.py --check   # 仅检查, 不修改
"""

import argparse
import hashlib
import logging
import os
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 两份副本的绝对路径
FILE_A = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "backtest_design.md",
)
FILE_B = r"d:\AMINQT\REFERENCE\Design All\Function Spec\BACKTESTING\backtest_design.md"


def file_hash(path: str) -> str:
    """计算文件内容的 SHA256 哈希."""
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def sync(check_only: bool = False) -> int:
    """同步两个文件.

    Args:
        check_only: True=仅检查不修改, False=执行同步.

    Returns:
        0=已同步, 1=已同步(执行了复制), 2=文件缺失.
    """
    exists_a = os.path.exists(FILE_A)
    exists_b = os.path.exists(FILE_B)

    if not exists_a and not exists_b:
        logger.error("两份文件都不存在!")
        return 2

    if not exists_a:
        logger.warning("副本A不存在, 从B复制到A")
        if check_only:
            return 2
        os.makedirs(os.path.dirname(FILE_A), exist_ok=True)
        shutil.copy2(FILE_B, FILE_A)
        logger.info("已复制 B → A")
        return 1

    if not exists_b:
        logger.warning("副本B不存在, 从A复制到B")
        if check_only:
            return 2
        os.makedirs(os.path.dirname(FILE_B), exist_ok=True)
        shutil.copy2(FILE_A, FILE_B)
        logger.info("已复制 A → B")
        return 1

    # 两份都存在
    hash_a = file_hash(FILE_A)
    hash_b = file_hash(FILE_B)

    if hash_a == hash_b:
        logger.info("两份文件内容一致, 无需同步 (hash=%s)", hash_a)
        return 0

    # 内容不同, 按修改时间决定方向
    mtime_a = os.path.getmtime(FILE_A)
    mtime_b = os.path.getmtime(FILE_B)

    if check_only:
        logger.warning(
            "文件不一致! A=%s(%s) B=%s(%s)", hash_a, mtime_a, hash_b, mtime_b
        )
        return 1

    if mtime_a > mtime_b:
        # A较新 → A覆盖B
        shutil.copy2(FILE_A, FILE_B)
        logger.info("同步: A → B (A较新, hash=%s)", hash_a)
    else:
        # B较新 → B覆盖A
        shutil.copy2(FILE_B, FILE_A)
        logger.info("同步: B → A (B较新, hash=%s)", hash_b)

    return 1


def main() -> int:
    """主函数."""
    parser = argparse.ArgumentParser(
        description="双向同步 backtest_design.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查一致性, 不执行复制",
    )
    args = parser.parse_args()

    logger.info("副本A: %s", FILE_A)
    logger.info("副本B: %s", FILE_B)

    result = sync(check_only=args.check)
    if result == 0:
        logger.info("✓ 同步完成, 无需修改")
    elif result == 1:
        if args.check:
            logger.warning("⚠ 文件不一致, 需要同步 (使用不带 --check 运行)")
        else:
            logger.info("✓ 同步完成")
    else:
        logger.error("✗ 文件缺失")
    return 0 if result != 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
