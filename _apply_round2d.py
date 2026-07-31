# -*- coding: utf-8 -*-
"""Fix _safe_register: remove @staticmethod, restore register_new call."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_engine_v35.py')
text = p.read_text(encoding='utf-8')

old = '''    @staticmethod
    def _safe_register(registry, name: str, meta: dict) -> None:
        """Wrap registry.register_new with try-except (file I/O safety)."""
        try:
            self._safe_register(registry, name, meta)
        except Exception as exc:
            logger.warning("Auto-adopt: register_new(%s) failed: %s", name, exc)'''

new = '''    def _safe_register(self, registry, name: str, meta: dict) -> None:
        """Wrap registry.register_new with try-except (file I/O safety)."""
        try:
            registry.register_new(name, meta)
        except Exception as exc:
            logger.warning("Auto-adopt: register_new(%s) failed: %s", name, exc)'''

assert old in text, 'old _safe_register not found'
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print('Fixed _safe_register method')
