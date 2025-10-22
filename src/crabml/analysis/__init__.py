"""
Statistical analysis tools for molecular evolution.

This module provides high-level functions for hypothesis testing,
including tests for positive selection using likelihood ratio tests.
"""

from .positive_selection import (
    positive_selection,
    m1a_vs_m2a,
    m7_vs_m8,
    m8a_vs_m8,
)
from .results import LRTResult, compare_results

__all__ = [
    "positive_selection",
    "m1a_vs_m2a",
    "m7_vs_m8",
    "m8a_vs_m8",
    "LRTResult",
    "compare_results",
]
