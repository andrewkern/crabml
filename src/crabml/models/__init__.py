"""
Molecular evolution models for sequence analysis.

This module provides substitution models for phylogenetic likelihood calculation:

- **Codon models**: Site-class models (M0-M9) with varying ω across sites
- **Branch models**: Models with varying ω across phylogenetic lineages
- **Branch-site models**: Models combining site and branch variation

Each model class computes transition probability matrices and handles
rate variation across sites and/or branches.
"""

from crabml.models.codon import (
    M0CodonModel,
    M1aCodonModel,
    M2aCodonModel,
    M3CodonModel,
    M7CodonModel,
    M8CodonModel,
    M8aCodonModel,
    compute_codon_frequencies_f3x4,
)
from crabml.models.codon_branch import CodonBranchModel

__all__ = [
    "M0CodonModel",
    "M1aCodonModel",
    "M2aCodonModel",
    "M3CodonModel",
    "M7CodonModel",
    "M8CodonModel",
    "M8aCodonModel",
    "CodonBranchModel",
    "compute_codon_frequencies_f3x4",
]
