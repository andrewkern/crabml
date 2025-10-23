"""
Hypothesis tests for lineage-specific selection using branch models.
"""

from typing import Union, Dict, Optional
import warnings
import numpy as np

from ..io.sequences import Alignment
from ..io.trees import Tree
from ..optimize.optimizer import M0Optimizer
from ..optimize.branch import BranchModelOptimizer
from .results import LRTResult


def branch_model_test(
    alignment: Union[str, Alignment],
    tree: Union[str, Tree],
    verbose: bool = True,
    optimize_branch_lengths: bool = True,
) -> LRTResult:
    """
    Test for lineage-specific selection using branch model vs M0.

    This test compares a multi-ratio branch model (where labeled branches
    have different omega) against M0 (one-ratio model where all branches
    share the same omega). A significant result indicates different
    selection pressures on different lineages.

    The tree must have branch labels (e.g., #0 for background, #1 for
    foreground) to specify which branches belong to which group.

    Parameters
    ----------
    alignment : str or Alignment
        Path to FASTA file or Alignment object
    tree : str or Tree
        Path to Newick file or Tree object with branch labels
        Example: "((human,chimp) #1, (mouse,rat));" tests if the
        human-chimp clade has different ω than the rest of the tree
    verbose : bool, default=True
        Print optimization progress
    optimize_branch_lengths : bool, default=True
        Optimize individual branch lengths (True) or use global scaling (False)

    Returns
    -------
    LRTResult
        Test results including LRT statistic, p-value, and model parameters

    Examples
    --------
    Test for positive selection on primate lineage:

    >>> from crabml.analysis import branch_model_test
    >>> tree_str = "((human,chimp) #1, (mouse,rat));"
    >>> result = branch_model_test('alignment.fasta', tree_str)
    >>> print(result.summary())
    >>> if result.significant(0.05):
    ...     omega_fg = result.alt_params['omega1']
    ...     omega_bg = result.alt_params['omega0']
    ...     print(f"Foreground ω={omega_fg:.3f}, Background ω={omega_bg:.3f}")

    Notes
    -----
    The test uses a likelihood ratio test where degrees of freedom equal
    the number of omega parameters in the alternative model minus 1.

    For a two-ratio model (background + foreground), df = 1.
    For a three-ratio model (background + 2 foreground groups), df = 2.

    This test is appropriate for detecting lineage-specific selection.
    If you want to test for positive selection on specific sites on
    specific lineages, use branch_site_test instead.

    References
    ----------
    Yang, Z. (1998). Likelihood ratio tests for detecting positive
    selection and application to primate lysozyme evolution.
    Molecular Biology and Evolution, 15(5), 568-573.

    Yang, Z., & Nielsen, R. (2002). Codon-substitution models for
    detecting molecular adaptation at individual sites along specific
    lineages. Molecular Biology and Evolution, 19(6), 908-917.
    """
    # Load data
    if isinstance(alignment, str):
        align = Alignment.from_fasta(alignment, seqtype='codon')
    else:
        align = alignment

    if isinstance(tree, str):
        tree_obj = Tree.from_newick(tree)
    else:
        tree_obj = tree

    # Optimize M0 (null model - one omega for all branches)
    if verbose:
        print("=" * 80)
        print("Optimizing M0 (One-ratio) model...")
        print("=" * 80)

    m0_optimizer = M0Optimizer(align, tree_obj)
    try:
        kappa_m0, omega_m0, lnL_m0 = m0_optimizer.optimize()
        m0_success = True
    except Exception as e:
        warnings.warn(f"M0 optimization failed: {e}", UserWarning)
        kappa_m0, omega_m0, lnL_m0 = 0, 0, -np.inf
        m0_success = False

    # Optimize branch model (alternative - different omega for labeled branches)
    if verbose:
        print("\n" + "=" * 80)
        print("Optimizing Branch Model (Multi-ratio)...")
        print("=" * 80)

    branch_optimizer = BranchModelOptimizer(
        alignment=align,
        tree=tree_obj,
        use_f3x4=True,
        optimize_branch_lengths=optimize_branch_lengths,
        free_ratio=False,  # Multi-ratio model
    )

    try:
        kappa_branch, omega_dict, lnL_branch = branch_optimizer.optimize()
        branch_success = True
        n_omega = branch_optimizer.model.n_omega
    except Exception as e:
        warnings.warn(f"Branch model optimization failed: {e}", UserWarning)
        kappa_branch, omega_dict, lnL_branch = 0, {}, -np.inf
        branch_success = False
        n_omega = 0

    # Calculate degrees of freedom
    # M0 has 1 omega parameter, branch model has n_omega parameters
    df = n_omega - 1

    # Create result object
    result = LRTResult(
        test_name="Branch Model vs M0",
        null_model="M0 (one-ratio)",
        alt_model=f"Branch Model ({n_omega}-ratio)",
        lnL_null=lnL_m0,
        lnL_alt=lnL_branch,
        df=df,
        null_params={
            'kappa': kappa_m0,
            'omega': omega_m0,
        },
        alt_params={
            'kappa': kappa_branch,
            **omega_dict,
        },
        null_optimization_success=m0_success,
        alt_optimization_success=branch_success,
    )

    if verbose:
        print("\n" + "=" * 80)
        print(result.summary())

    return result


def free_ratio_test(
    alignment: Union[str, Alignment],
    tree: Union[str, Tree],
    verbose: bool = True,
    optimize_branch_lengths: bool = True,
) -> LRTResult:
    """
    Test for heterogeneous selection using free-ratio model vs M0.

    This test compares a free-ratio branch model (where each branch has
    independent omega) against M0 (one-ratio model). This is a very
    parameter-rich model that can detect heterogeneous selection across
    the phylogeny but is prone to overfitting.

    WARNING: Free-ratio model has many parameters (n_branches omega values)
    and may overfit with small datasets. Multi-ratio models (with branch
    labels) are generally recommended instead.

    Parameters
    ----------
    alignment : str or Alignment
        Path to FASTA file or Alignment object
    tree : str or Tree
        Path to Newick file or Tree object (no branch labels needed)
    verbose : bool, default=True
        Print optimization progress
    optimize_branch_lengths : bool, default=True
        Optimize individual branch lengths (True) or use global scaling (False)

    Returns
    -------
    LRTResult
        Test results including LRT statistic, p-value, and model parameters

    Examples
    --------
    >>> from crabml.analysis import free_ratio_test
    >>> result = free_ratio_test('alignment.fasta', 'tree.nwk')
    >>> print(result.summary())
    >>> if result.significant(0.05):
    ...     print("Heterogeneous selection detected across branches")

    Notes
    -----
    The test uses a likelihood ratio test where degrees of freedom equal
    the number of branches minus 1 (n_species - 2 for n_species).

    This test is exploratory and should be interpreted with caution due to
    the large number of parameters. Multi-ratio models with biologically
    motivated branch labels are preferred.

    References
    ----------
    Yang, Z. (1998). Likelihood ratio tests for detecting positive
    selection and application to primate lysozyme evolution.
    Molecular Biology and Evolution, 15(5), 568-573.
    """
    # Load data
    if isinstance(alignment, str):
        align = Alignment.from_fasta(alignment, seqtype='codon')
    else:
        align = alignment

    if isinstance(tree, str):
        tree_obj = Tree.from_newick(tree)
    else:
        tree_obj = tree

    # Optimize M0 (null model)
    if verbose:
        print("=" * 80)
        print("Optimizing M0 (One-ratio) model...")
        print("=" * 80)

    m0_optimizer = M0Optimizer(align, tree_obj)
    try:
        kappa_m0, omega_m0, lnL_m0 = m0_optimizer.optimize()
        m0_success = True
    except Exception as e:
        warnings.warn(f"M0 optimization failed: {e}", UserWarning)
        kappa_m0, omega_m0, lnL_m0 = 0, 0, -np.inf
        m0_success = False

    # Optimize free-ratio model (alternative)
    if verbose:
        print("\n" + "=" * 80)
        print("Optimizing Free-Ratio Branch Model...")
        print("=" * 80)

    branch_optimizer = BranchModelOptimizer(
        alignment=align,
        tree=tree_obj,
        use_f3x4=True,
        optimize_branch_lengths=optimize_branch_lengths,
        free_ratio=True,  # Free-ratio model
    )

    try:
        kappa_branch, omega_dict, lnL_branch = branch_optimizer.optimize()
        branch_success = True
        n_omega = branch_optimizer.model.n_omega
    except Exception as e:
        warnings.warn(f"Free-ratio model optimization failed: {e}", UserWarning)
        kappa_branch, omega_dict, lnL_branch = 0, {}, -np.inf
        branch_success = False
        n_omega = 0

    # Calculate degrees of freedom
    df = n_omega - 1

    # Create result object
    result = LRTResult(
        test_name="Free-Ratio vs M0",
        null_model="M0 (one-ratio)",
        alt_model=f"Free-Ratio ({n_omega} omegas)",
        lnL_null=lnL_m0,
        lnL_alt=lnL_branch,
        df=df,
        null_params={
            'kappa': kappa_m0,
            'omega': omega_m0,
        },
        alt_params={
            'kappa': kappa_branch,
            **omega_dict,
        },
        null_optimization_success=m0_success,
        alt_optimization_success=branch_success,
    )

    if verbose:
        print("\n" + "=" * 80)
        print(result.summary())
        print("\nWARNING: Free-ratio model has many parameters and may overfit.")
        print("Consider using multi-ratio model with branch labels instead.")

    return result
