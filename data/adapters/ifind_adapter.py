# -*- coding: utf-8 -*-
"""iFinD (同花顺) data adapter — production source.

Requires the iFinD terminal + iFinDPy SDK (NOT pip-installable) and account
credentials. All real calls are gated behind a login check so this module
imports safely even without iFinDPy installed. See README for setup.
"""
import logging

import pandas as pd

from config import settings
from .base import DataAdapter

logger = logging.getLogger(__name__)


class IfindAdapter(DataAdapter):
    """Daily + intraday data via 同花顺 iFinD (iFinDPy)."""

    def __init__(self) -> None:
        try:
            import iFinDPy  # noqa: F401
            self._api = __import__("iFinDPy")
        except ImportError as exc:
            raise RuntimeError(
                "iFinDPy not installed. Install via 同花顺 iFinD terminal."
            ) from exc
        self._login()

    def _login(self) -> None:
        """Log into iFinD using env credentials."""
        user, pwd = settings.IFIND_USER, settings.IFIND_PASSWORD
        if not user or not pwd:
            raise RuntimeError("IFIND_USER / IFIND_PASSWORD not set in env.")
        ret = self._api.THS_iFinDlogin(user, pwd)
        logger.info("iFinD login result: %s", ret)

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily K-line via iFinD.

        TODO(Phase 1+): wire real iFinD high-frequency/basic-data call and
        map iFinD field names to canonical columns via _canonicalize.
        """
        raise NotImplementedError("iFinD daily fetch — wire with creds.")

    def fetch_intraday(self, symbol: str, trade_date: str) -> pd.DataFrame:
        """Fetch intraday bars via iFinD for M2 pattern learning.

        TODO(M2): wire real iFinD high-frequency call.
        """
        raise NotImplementedError("iFinD intraday fetch — wire in M2.")
