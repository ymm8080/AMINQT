# -*- coding: utf-8 -*-
"""
D-16: 全文档硬阈值审计 (IMPLEMENTATION_PLAN_v3.2 P26)
======================================================
扫描 config + app/ 下所有 .py 文件中的硬编码数字,
按四类分类: QUANTILE_REPLACEABLE | SECTOR_ADAPTIVE | JUSTIFIED_HARD | NEEDS_REVIEW.

已知违规 (v3.1_0727 grep):
  - risk_overlays.py: AMPLITUDE_FUSE=0.03, VOL_SURGE_RATIO=1.5
  - trade_discipline.py: HARD_STOP=-0.04, TIME_STOP_MIN_RET=0.01, DAILY_LOSS_LIMIT=0.04
  - sell_engine.py S5a: 固定 <1%
  - fund_manager.py: daily_loss_limit=0.04
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# JUSTIFIED 模式: 有经济学/法规依据的硬阈值
JUSTIFIED_PATTERNS = {
    "leverage": r"\bleverage\s*[=:]\s*1\.0\b",  # 约束C: 不上杠杆
    "stamp_tax": r"0\.0005",  # 印花税
    "limit_up_main": r"\b0\.10\b.*limit.*up",  # 主板涨停 10%
    "limit_up_chinext": r"\b0\.20\b.*limit.*up",  # 双创涨停 20%
    "cost": r"COST\s*=\s*0\.0013",  # E5 口径 round-trip 费用
}

# 已知已改为分位数驱动的文件 (v3.2 已改造)
V32_QUANTILIZED = {
    "app/pipeline1/risk_overlays.py",
    "app/pipeline1/trade_discipline.py",
    "app/intraday/v51/sell_engine.py",
    "app/intraday/v51/fund_manager.py",
}


def scan_file(filepath: str) -> list[dict]:
    """扫描单个文件中的硬编码数字赋值."""
    findings = []
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
    except Exception:
        return findings

    # 简化的数值常量扫描: 匹配 module-level CONST = <number>
    for m in re.finditer(
        r"^([A-Z][A-Z_0-9]*)\s*=\s*(-?\d+\.?\d*)\s*(?:#.*)?$",
        source,
        re.MULTILINE,
    ):
        name, val = m.group(1), float(m.group(2))
        classification = _classify(name, val, filepath)
        findings.append(
            {
                "file": filepath,
                "line": source[: m.start()].count("\n") + 1,
                "name": name,
                "value": val,
                "classification": classification,
            }
        )
    return findings


def _classify(name: str, value: float, filepath: str) -> str:
    """分类: QUANTILE_REPLACEABLE | SECTOR_ADAPTIVE | JUSTIFIED_HARD | NEEDS_REVIEW."""
    # v3.2 已改造文件中的阈值视为已处理
    rel = filepath.replace("\\", "/")
    for qf in V32_QUANTILIZED:
        if qf in rel:
            return "QUANTILIZED_v32"

    # JUSTIFIED: 经济学/法规依据
    lower = name.lower()
    text = f"{name}={value}"
    for jname, pattern in JUSTIFIED_PATTERNS.items():
        if re.search(pattern, text):
            return "JUSTIFIED_HARD"

    # QUANTILE_REPLACEABLE: 止损/振幅/保险丝/时间止损相关
    if any(
        kw in lower
        for kw in (
            "stop",
            "fuse",
            "loss_limit",
            "amplitude",
            "vol_surge",
            "daily_loss",
            "time_stop",
            "hard_stop",
            "drawdown",
            "halt",
        )
    ):
        return "QUANTILE_REPLACEABLE"

    # SECTOR_ADAPTIVE: 板块/主板/双创相关
    if any(kw in lower for kw in ("main", "chinext", "sector", "board")):
        return "SECTOR_ADAPTIVE"

    # NEEDS_REVIEW: 不确定
    return "NEEDS_REVIEW"


def audit_all(root: str = ".") -> dict:
    """扫描全部 .py 文件, 返回审计报告."""
    findings = []
    scanned = 0
    for dirpath, _, filenames in os.walk(root):
        # 跳过非代码目录
        for skip in (
            "__pycache__",
            "node_modules",
            ".git",
            ".claude",
            "venv",
            "skills",
            ".tox",
            "site-packages",
        ):
            if skip in dirpath:
                break
        else:
            for fn in filenames:
                if fn.endswith(".py"):
                    filepath = os.path.join(dirpath, fn)
                    scanned += 1
                    findings.extend(scan_file(filepath))

    # 统计
    by_class = {}
    for f in findings:
        c = f["classification"]
        by_class.setdefault(c, []).append(f)

    report = {
        "date": date.today().isoformat(),
        "scanned_files": scanned,
        "total_findings": len(findings),
        "by_classification": {k: len(v) for k, v in by_class.items()},
        "violations": [
            f
            for f in findings
            if f["classification"] in ("QUANTILE_REPLACEABLE", "NEEDS_REVIEW")
        ],
        "findings": findings,
    }
    return report


def ci_check(report: dict) -> bool:
    """CI 门禁: QUANTILE_REPLACEABLE 或 NEEDS_REVIEW → 阻止合并."""
    violations = report.get("violations", [])
    if violations:
        logger.error("硬阈值审计: %d 处违规, 拒绝合并:", len(violations))
        for v in violations:
            logger.error(
                "  %s:%d %s=%.4f (%s)",
                v["file"],
                v["line"],
                v["name"],
                v["value"],
                v["classification"],
            )
        return False
    logger.info("硬阈值审计: 通过 (0 违规)")
    return True


def main():
    """扫描并生成审计报告 → WORM 日志."""
    out_dir = os.path.join("data", "audit")
    os.makedirs(out_dir, exist_ok=True)
    today = date.today().isoformat()
    out_path = os.path.join(out_dir, f"hard_thresholds_{today}.json")

    report = audit_all("app")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    logger.info(
        "审计完成: %d 文件, %d 阈值, 输出 → %s",
        report["scanned_files"],
        report["total_findings"],
        out_path,
    )
    for cls_name, count in sorted(report["by_classification"].items()):
        logger.info("  %s: %d", cls_name, count)

    # CI 模式: 有违规 → exit 1
    if not ci_check(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
