"""Safe division utility — zero division protection for scalar arithmetic."""

from __future__ import annotations


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe scalar division: returns *default* when denominator is zero.

    Used throughout v51 to satisfy the ``safe_divide`` code-review rule
    (rule #8: Division without safe_divide — zero division risk).
    """
    if denominator == 0:
        return default
    return numerator / denominator
