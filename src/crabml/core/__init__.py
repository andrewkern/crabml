"""
Core algorithms for phylogenetic likelihood calculation.

This module provides low-level computational routines:

- **Likelihood calculation**: Felsenstein's pruning algorithm
- **Matrix operations**: Eigendecomposition and matrix exponential
- **Performance**: Rust backend via PyO3 for high-speed computation

These are expert-level functions typically not needed by end users.
The high-level API (:mod:`crabml.api`) provides easier access.
"""

from crabml.core.likelihood import LikelihoodCalculator
from crabml.core.matrix import eigen_decompose_rev, matrix_exponential

__all__ = ["LikelihoodCalculator", "matrix_exponential", "eigen_decompose_rev"]
