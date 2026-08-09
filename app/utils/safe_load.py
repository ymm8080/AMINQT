"""Safe model loading utilities — mitigate pickle deserialization RCE.

pickle.load() can execute arbitrary Python code.  These helpers use
joblib (which wraps pickle but is more controlled) and add integrity
logging so that model file tampering is at least traceable.

For maximum safety, prefer:
  1. torch.load(path, weights_only=True)  for PyTorch models
  2. joblib.load(path)                     for sklearn / LightGBM models
  3. This module's safe_pickle_load()      only when raw pickle is unavoidable

Usage:
    from app.utils.safe_load import safe_pickle_load
    bundle = safe_pickle_load("models/pipeline1/main_v38.pkl")
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# 常见 pickle RCE gadget 模块 (M5): find_class 拦截其类引用, 阻止
# os.system / subprocess.Popen / builtins.eval / codecs.escape 等标准攻击链.
# 模型类 (numpy / lightgbm / sklearn / app) 均不受影响; 按根模块拦截,
# 覆盖 os.path / importlib.util 等子模块.
_UNSAFE_PICKLE_MODULES = frozenset(
    {
        "os",
        "posix",
        "nt",
        "posixpath",
        "ntpath",
        "genericpath",
        "subprocess",
        "builtins",
        "codecs",
        "_codecs",
        "importlib",
        "ctypes",
        "marshal",
        "sys",
        "types",
        "threading",
        "multiprocessing",
        "runpy",
        "pkgutil",
        "shlex",
        "webbrowser",
        "tempfile",
        "platform",
    }
)


class _SafeUnpickler(pickle.Unpickler):
    """阻断危险模块类引用的 Unpickler (M5).

    pickle.load 可执行任意代码; 此版在 find_class 时拦截已知 RCE gadget
    模块, 其余类照常加载. 仅加防护, 不改审计日志行为.
    """

    def find_class(self, module: str, name: str) -> type:
        root = module.split(".")[0]
        if root in _UNSAFE_PICKLE_MODULES:
            raise pickle.UnpicklingError(
                f"blocked unsafe pickle module: {module}.{name}"
            )
        return super().find_class(module, name)


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 hash of a file (first 64 chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:64]


def safe_pickle_load(path: str | Path) -> object:
    """Load a pickle file with safety checks.

    Mitigations:
      - Logs file path, size, and SHA-256 hash for audit trail.
      - Wraps in try/except to avoid silent corruption.

    Args:
        path: Path to the pickle file.

    Returns:
        The unpickled object.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If the file is not valid pickle.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    size = p.stat().st_size
    sha = file_sha256(p)
    logger.info("Loading pickle: %s (%d bytes, sha256=%s...)", p, size, sha[:16])

    try:
        with open(p, "rb") as fh:
            return _SafeUnpickler(fh).load()
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        logger.error("Failed to load pickle %s: %s (sha256=%s)", p, exc, sha)
        raise


def safe_joblib_load(path: str | Path) -> object:
    """Load a model via joblib (preferred over raw pickle).

    joblib uses pickle internally but is the standard for sklearn models
    and provides better compression / numpy array handling.

    Args:
        path: Path to the joblib/pickle file.

    Returns:
        The loaded model object.
    """
    import joblib

    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    size = p.stat().st_size
    sha = file_sha256(p)
    logger.info("Loading joblib: %s (%d bytes, sha256=%s...)", p, size, sha[:16])

    try:
        return joblib.load(p)
    except Exception as exc:
        logger.error("Failed to load joblib %s: %s (sha256=%s)", p, exc, sha)
        raise
