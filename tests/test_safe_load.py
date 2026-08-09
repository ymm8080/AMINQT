"""safe_load 安全加载测试 (M5)."""

from __future__ import annotations

import os
import pickle
import subprocess

import numpy as np
import pytest

from app.utils.safe_load import safe_pickle_load


def _dump(payload, tmp_path):
    p = tmp_path / "m.pkl"
    p.write_bytes(pickle.dumps(payload))
    return p


class _EvilOs:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


class _EvilSubprocess:
    def __reduce__(self):
        return (subprocess.Popen, ("echo pwned",))


class _EvilEval:
    def __reduce__(self):
        return (eval, ("1+1",))


def test_blocks_rce_gadgets(tmp_path):
    for evil in (_EvilOs(), _EvilSubprocess(), _EvilEval()):
        p = _dump(evil, tmp_path)
        with pytest.raises(pickle.UnpicklingError):
            safe_pickle_load(p)


def test_loads_legit_payload(tmp_path):
    payload = {"models": {"main": 1}, "arr": np.arange(3), "meta": {"a": [1, 2]}}
    p = _dump(payload, tmp_path)
    loaded = safe_pickle_load(p)
    assert isinstance(loaded, dict)
    assert (loaded["arr"] == np.arange(3)).all()
