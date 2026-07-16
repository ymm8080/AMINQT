# -*- coding: utf-8 -*-
"""Data adapters: pluggable data sources (iFinD primary, akshare fallback)."""
from .base import DataAdapter, get_adapter

__all__ = ["DataAdapter", "get_adapter"]
