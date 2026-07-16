# -*- coding: utf-8 -*-
"""Pytest shared fixtures + project root on sys.path.

Ensures `from config import settings`, `from app...`, `from data...`,
`from services...` resolve when running `pytest` from the project root.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
