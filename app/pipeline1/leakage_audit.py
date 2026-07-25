"""
未来函数审计 (P19.0 W2, 安全网 #4, PIPELINE1_V3.8 §六)
==============================================================
两道防线:
  1. 源码静态扫描: ZIG/PEAK/TROUGHBARS 类未来函数与 shift(-k)/REF(X,-k)
     前瞻引用零使用 (特征引擎源码硬约束)
  2. IC 上限哨兵 (安全网 #4): 任一特征对标签的 |Rank IC| > 0.15 → 泄漏嫌疑,
     触发复核 (A 股日频横截面 alpha 不可能稳定超过此水平)

铁律: 回测只能用 t-1 及更早的数据, 特征计算不得引用未来信息.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

IC_SENTINEL = 0.15  # 安全网 #4: |IC| > 0.15 触发未来函数审计

# 禁用模式 (特征计算源码): 未来函数 + 负向 shift/REF 前瞻
FORBIDDEN_PATTERNS = (
    r"\bZIG\b",
    r"\bPEAK\b",
    r"\bPEAKBARS\b",
    r"\bTROUGH\b",
    r"\bTROUGHBARS\b",
    r"\.shift\(\s*-\d",  # shift(-k)
    r"\bREF\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*-\d",  # REF(X,-k)
)


# ============================================================
# 防线 1: 源码静态扫描
# ============================================================
def audit_source(source: str, filename: str = "<source>") -> list[dict]:
    """扫描特征计算源码中的未来函数模式. 返回违规清单 (空=通过).

    例外: label_engine.py 的 _label_reference/_future_window_min 是标签专用
    (标签合法引用未来), 由 ALLOW_LABEL_MODULES 豁免 — 但特征模块绝不豁免.
    """
    hits = []
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.split("#", 1)[0]  # 忽略注释
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, stripped):
                hits.append(
                    {
                        "file": filename,
                        "line": i,
                        "pattern": pat,
                        "code": line.strip()[:80],
                    }
                )
    for h in hits:
        logger.error(
            "未来函数嫌疑: %s:%d [%s] %s", h["file"], h["line"], h["pattern"], h["code"]
        )
    return hits


def audit_feature_modules(paths: list[str]) -> dict:
    """扫描特征模块文件清单. 任一命中 → 不通过."""
    all_hits = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            all_hits.extend(audit_source(fh.read(), p))
    return {"pass": len(all_hits) == 0, "violations": all_hits}


# ============================================================
# 防线 2: IC 上限哨兵
# ============================================================
def ic_sentinel(
    df: pd.DataFrame, feature_cols: list[str], label: str = "label_1d"
) -> dict:
    """|Rank IC| > 0.15 的特征 → 泄漏嫌疑清单 (安全网 #4).

    Returns:
        {'pass': bool, 'suspects': {feature: ic}, 'max_abs_ic': float}
    """
    suspects = {}
    max_ic = 0.0
    for f in feature_cols:
        sub = df[["date", f, label]].dropna()
        if sub["date"].nunique() < 5:
            continue
        ics = sub.groupby("date").apply(
            lambda g: (
                spearmanr(g[f], g[label]).statistic
                if g[f].nunique() > 5 and g[label].nunique() > 1
                else np.nan
            )
        )
        ic = float(np.nanmean(ics.values))
        max_ic = max(max_ic, abs(ic))
        if abs(ic) > IC_SENTINEL:
            suspects[f] = round(ic, 4)
    for f, ic in suspects.items():
        logger.error(
            "IC 上限哨兵: %s IC=%.4f > %.2f, 未来函数嫌疑, 触发复核", f, ic, IC_SENTINEL
        )
    return {
        "pass": len(suspects) == 0,
        "suspects": suspects,
        "max_abs_ic": round(max_ic, 4),
    }
