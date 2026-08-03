#!/usr/bin/env python3
import pytest
import tempfile
from tests.test_end_to_end_v35 import chain

# Create a temporary directory and run the test
tmp_path_factory = pytest.TempPathFactory(tempfile.mkdtemp(), "tmp", False)
result = chain(tmp_path_factory)
panel, pipe = result['panel'], result['pipe']
r1 = pipe.run('20260720', panel=panel)
print('Mode:', r1['mode'])
print('List length:', len(r1.get('list', [])))
if len(r1.get('list', [])) > 0:
    print('First symbol:', r1['list'].iloc[0]['symbol'])