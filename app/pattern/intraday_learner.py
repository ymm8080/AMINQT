# -*- coding: utf-8 -*-
"""M2 — Intraday pattern learner.

Learns recurring within-day patterns from minute/tick bars and emits
candidate trading rules consumed by app/rules/rule_engine. This is the
module that turns "learn pattern within a day" into a set of executable
trading rules for the trading system.

TODO(M2): implement clustering / sequence model over intraday bars to
discover valid patterns → rule candidates with entry/exit/stop + confidence.
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def learn_pattern(intraday_df: pd.DataFrame) -> List[dict]:
    """Learn intraday patterns and return candidate trading rules.

    Args:
        intraday_df: Intraday bars for one symbol/day (canonical cols).

    Returns:
        List of rule dicts, e.g.
        {'symbol': str, 'entry': <condition>, 'exit': <condition>,
         'stop': <condition>, 'confidence': float}.
    """
    raise NotImplementedError("intraday_learner.learn_pattern — implement in M2")
