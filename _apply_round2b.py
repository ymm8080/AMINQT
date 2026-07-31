# -*- coding: utf-8 -*-
"""Round 2 fix: broaden exception handling in feature_registry.py save()/load()."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_registry.py')
text = p.read_text(encoding='utf-8')

# Fix load(): broaden OSError, JSONDecodeError -> Exception
old_load = '''        except (OSError, json.JSONDecodeError) as exc:
            logger.error("FeatureRegistry: 加载 %s 失败: %s", self.path, exc)'''
new_load = '''        except Exception as exc:
            logger.error("FeatureRegistry: 加载 %s 失败: %s", self.path, exc)'''
assert old_load in text, 'load except not found'
text = text.replace(old_load, new_load, 1)

# Fix save(): broaden OSError -> Exception
old_save = '''        except OSError as exc:
            logger.error("FeatureRegistry: 保存 %s 失败: %s", self.path, exc)'''
new_save = '''        except Exception as exc:
            logger.error("FeatureRegistry: 保存 %s 失败: %s", self.path, exc)'''
assert old_save in text, 'save except not found'
text = text.replace(old_save, new_save, 1)

p.write_text(text, encoding='utf-8')
print('feature_registry.py: broadened exception handling in load()/save()')