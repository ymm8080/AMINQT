# -*- coding: utf-8 -*-
"""Apply E402 fix: move relative imports above logger init."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_engine_v35.py')
text = p.read_text(encoding='utf-8')

old_block = """import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

from .cleaning_pipeline import get_limit_pct
from .label_engine import _label_reference
"""

new_block = """import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .cleaning_pipeline import get_limit_pct
from .label_engine import _label_reference

logger = logging.getLogger(__name__)
"""

assert old_block in text, 'old block not found'
text = text.replace(old_block, new_block, 1)
p.write_text(text, encoding='utf-8')
print('E402 fix applied')
