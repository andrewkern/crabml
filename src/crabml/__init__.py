"""
py-codeml: Python reimplementation of PAML's codeml

A modern, fast implementation of phylogenetic maximum likelihood analysis
with a Python frontend and Rust computational core.
"""

__version__ = "0.1.0"

from crabml.core.likelihood import LikelihoodCalculator
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree

__all__ = [
    "LikelihoodCalculator",
    "Alignment",
    "Tree",
    "__version__",
]
