"""
crabML: Modern Python implementation of PAML's codeml.

A fast, user-friendly implementation of phylogenetic maximum likelihood
analysis with a Python frontend and Rust computational core.

Quick Start
-----------
Fit a single model:

>>> from crabml import optimize_model
>>> result = optimize_model("M0", "alignment.fasta", "tree.nwk")
>>> print(result.summary())
>>> print(f"omega = {result.omega:.4f}")

Test for positive selection:

>>> from crabml import positive_selection
>>> results = positive_selection("alignment.fasta", "tree.nwk", test="both")
>>> print(results['M1a_vs_M2a'].summary())
>>> print(results['M7_vs_M8'].summary())

Examples
--------
>>> # Fit M2a model with custom parameters
>>> result = optimize_model("M2a", "data.fasta", "tree.nwk", maxiter=500)
>>> if result.params['omega2'] > 1:
...     print(f"Positive selection detected: omega = {result.params['omega2']:.2f}")

>>> # Run M1a vs M2a test with BEB analysis
>>> from crabml import m1a_vs_m2a
>>> result = m1a_vs_m2a("data.fasta", "tree.nwk", compute_beb=True)
>>> if result.significant(0.05) and result.beb:
...     sig_sites = result.beb.significant_sites(threshold=0.95)
...     print(f"Sites under selection: {sig_sites}")
"""

__version__ = "0.2.0"

# High-level API (simple interface)
from .api import (
    optimize_model,
    optimize_branch_model,
    optimize_branch_site_model,
    ModelResult,  # Backwards compatibility alias
    SiteModelResult,
    BranchModelResult,
    BranchSiteModelResult,
    ModelResultBase,
)

# Analysis functions (hypothesis testing)
from .analysis import (
    positive_selection,
    m1a_vs_m2a,
    m7_vs_m8,
    m8a_vs_m8,
    branch_site_test,
    branch_model_test,
    free_ratio_test,
    LRTResult,
    compare_results,
)

# I/O classes (for advanced users)
from .io.sequences import Alignment
from .io.trees import Tree

# Core likelihood calculator (expert use)
from .core.likelihood import LikelihoodCalculator

__all__ = [
    # Simple API - Start here!
    "optimize_model",
    "optimize_branch_model",
    "optimize_branch_site_model",
    "positive_selection",

    # Result objects
    "ModelResult",  # Backwards compatibility (alias for SiteModelResult)
    "SiteModelResult",
    "BranchModelResult",
    "BranchSiteModelResult",
    "ModelResultBase",
    "LRTResult",

    # Specific hypothesis tests
    "m1a_vs_m2a",
    "m7_vs_m8",
    "m8a_vs_m8",
    "branch_site_test",
    "branch_model_test",
    "free_ratio_test",

    # Utilities
    "compare_results",

    # I/O (advanced)
    "Alignment",
    "Tree",

    # Core (expert)
    "LikelihoodCalculator",

    # Version
    "__version__",
]
