# -*- coding: utf-8 -*-
"""筹码 chip4 族入 main force_include 接线测试 (2026-09-12 族级 A/B 8格矩阵).

判词 (WORM: diag/ab_families_0912_main_20260912_110635.json):
  main×reg  top10 5.15%→6.82% (+1.67pp 达线) / IC +0.41pp → 入
  main×cls  top10 7.14%→6.91% (−0.23pp 留观带内) → 共列净正
  dual 已由 gate_d 选择收录 (harness 自动跳过), 勿 force_include 重复注入
同批否决: ps_ttm (reg IC +1.34pp 但 top10 −2.53pp) / quality_factor 合成列
(main reg −3.52pp) — 两族不得出现在任一板 force_include.
"""

import unittest

from app.pipeline1.feature_selector import FeatureSelector

CHIP4 = ["cost_bias", "peak_roc_20d", "chip_gini", "chip_entropy"]
REJECTED = ["ps_ttm", "quality_factor"]


class TestChip4ForceIncludeWiring(unittest.TestCase):
    def test_main_includes_chip4(self):
        inc = set(FeatureSelector.DEFAULT_CONFIG["main"]["force_include"])
        for name in CHIP4:
            self.assertIn(name, inc, f"main force_include 缺 {name}")

    def test_dual_not_duplicated(self):
        inc = set(FeatureSelector.DEFAULT_CONFIG["dual"]["gate_d"]["force_include"])
        for name in CHIP4:
            self.assertNotIn(
                name, inc, f"dual 已由 pin 选择收录 {name}, force_include 勿重复"
            )

    def test_rejected_families_absent_both_boards(self):
        main = set(FeatureSelector.DEFAULT_CONFIG["main"]["force_include"])
        dual = set(FeatureSelector.DEFAULT_CONFIG["dual"]["gate_d"]["force_include"])
        for name in REJECTED:
            self.assertNotIn(name, main, f"已判摘 {name} 不得回 main")
            self.assertNotIn(name, dual, f"已判摘 {name} 不得回 dual")


if __name__ == "__main__":
    unittest.main()
