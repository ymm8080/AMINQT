# -*- coding: utf-8 -*-
"""3 个月 / 300 股 smoke test driver.

Patches _train_predict_dual_1y.py constants, then execs it verbatim so the
smoke run exercises the exact same production code path (feature_engine_v35 /
panel_builder / dual_track_trainer / predictor):
  LOOKBACK_DAYS 250 -> 63        (3 trading months)
  symbol universe -> 300 确定性抽样 (random_state=0, 可复现)
  model tag "_1y"  -> "_3m"      (不覆盖真实 1y 模型包)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_train_predict_dual_1y.py")
_src = open(_src_path, encoding="utf-8").read()

_src = _src.replace(
    "LOOKBACK_DAYS = 250  # 1 calendar year ≈ 250 A-share trading days",
    "LOOKBACK_DAYS = 63   # 3 trading months (smoke)",
)
_src = _src.replace(
    'panel_1y = panel[panel["date"].isin(dates[-LOOKBACK_DAYS:])].copy()',
    'panel_1y = panel[panel["date"].isin(dates[-LOOKBACK_DAYS:])].copy()\n'
    '_syms = pd.Series(panel_1y["symbol"].unique()).sample(300, random_state=0)\n'
    'panel_1y = panel_1y[panel_1y["symbol"].isin(_syms)].copy()',
)
_src = _src.replace(
    'tag = time.strftime("%GW%V") + "_1y"',
    'tag = time.strftime("%GW%V") + "_3m"',
)

exec(compile(_src, "_train_predict_dual_1y.py", "exec"), {"__file__": _src_path, "__name__": "__main__"})
