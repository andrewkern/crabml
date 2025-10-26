"""
Sequence simulation module for crabML.

This module provides tools for simulating molecular sequences under various
evolutionary models. Useful for:
- Validating parameter estimation methods
- Power analysis for detecting selection
- Generating test datasets
- Benchmarking performance

Available simulators:
- M0CodonSimulator: Single omega (dN/dS) model
- M1aSimulator: Nearly neutral model (2 classes)
- M2aSimulator: Positive selection model (3 classes)
- M7Simulator: Beta distribution model
- M8Simulator: Beta + positive selection model
"""

from .base import SequenceSimulator
from .codon import M0CodonSimulator

__all__ = [
    'SequenceSimulator',
    'M0CodonSimulator',
]
