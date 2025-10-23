"""
Optimization routines for maximum likelihood parameter estimation.

This module provides optimizer classes for fitting codon substitution models:

- **Site-class models**: M0, M1a, M2a, M3, M7, M8, M8a, etc.
- **Branch models**: Free-ratio and multi-ratio models
- **Branch-site models**: Model A and variants

Each optimizer class handles parameter initialization, likelihood calculation,
and numerical optimization using scipy.optimize.
"""

from crabml.optimize.optimizer import M0Optimizer, M1aOptimizer, M2aOptimizer
from crabml.optimize.branch_site import BranchSiteModelAOptimizer
from crabml.optimize.branch import BranchModelOptimizer

__all__ = [
    "M0Optimizer",
    "M1aOptimizer",
    "M2aOptimizer",
    "BranchSiteModelAOptimizer",
    "BranchModelOptimizer",
]
