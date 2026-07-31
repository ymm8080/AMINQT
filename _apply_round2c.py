# -*- coding: utf-8 -*-
"""Round 2 fix: add _safe_register helper to wrap register_new with try-except."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_engine_v35.py')
text = p.read_text(encoding='utf-8')

# Add _safe_register helper method right before _generate_adopted_features
old_method_start = '''    def _generate_adopted_features(
        self, df: pd.DataFrame, col: str, registry
    ) -> pd.DataFrame:
        """为一个新面板列生成 6 个模板特征, 并注册到 registry."""'''

new_method_start = '''    @staticmethod
    def _safe_register(registry, name: str, meta: dict) -> None:
        """Wrap registry.register_new with try-except (file I/O safety)."""
        try:
            registry.register_new(name, meta)
        except Exception as exc:
            logger.warning("Auto-adopt: register_new(%s) failed: %s", name, exc)

    def _generate_adopted_features(
        self, df: pd.DataFrame, col: str, registry
    ) -> pd.DataFrame:
        """为一个新面板列生成 6 个模板特征, 并注册到 registry."""'''

assert old_method_start in text, '_generate_adopted_features start not found'
text = text.replace(old_method_start, new_method_start, 1)

# Replace all registry.register_new( with self._safe_register(registry,
text = text.replace('registry.register_new(', 'self._safe_register(registry, ')

p.write_text(text, encoding='utf-8')
print(f'Applied _safe_register wrapper. Replaced {text.count("self._safe_register(registry,")} call sites')
