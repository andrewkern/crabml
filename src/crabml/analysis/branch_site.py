"""
Branch-site model hypothesis tests for detecting positive selection.
"""

from typing import Tuple, Optional
import numpy as np

from ..io.sequences import Alignment
from ..io.trees import Tree
from ..optimize.branch_site import BranchSiteModelAOptimizer
from .lrt import calculate_lrt


def branch_site_test(
    alignment: Alignment,
    tree: Tree,
    use_f3x4: bool = True,
    optimize_branch_lengths: bool = True,
    init_kappa: float = 3.0,
    init_omega0: float = 0.05,
    init_omega2: float = 3.0,
    init_p0: float = 0.35,
    init_p1: float = 0.30,
    method: str = 'L-BFGS-B',
    maxiter: int = 500
) -> dict:
    """
    Perform branch-site test of positive selection.

    This test compares Branch-Site Model A (alternative) against
    Branch-Site Model A with omega2=1 (null) to detect positive selection
    affecting specific sites on foreground branches.

    The likelihood ratio test uses df=1 with standard chi-square distribution.

    Parameters
    ----------
    alignment : Alignment
        Codon sequence alignment
    tree : Tree
        Phylogenetic tree with branch labels (0=background, 1=foreground)
    use_f3x4 : bool
        Use F3X4 codon frequencies (default: True)
    optimize_branch_lengths : bool
        Optimize individual branch lengths (default: True)
    init_kappa : float
        Initial kappa value (default: 3.0)
    init_omega0 : float
        Initial omega0 value (default: 0.05)
    init_omega2 : float
        Initial omega2 value for alternative model (default: 3.0)
    init_p0 : float
        Initial p0 proportion (default: 0.35)
    init_p1 : float
        Initial p1 proportion (default: 0.30)
    method : str
        Optimization method (default: 'L-BFGS-B')
    maxiter : int
        Maximum iterations (default: 500)

    Returns
    -------
    dict
        Results dictionary containing:
        - null_lnL: log-likelihood of null model (omega2=1)
        - null_params: dict of null model parameters
        - alt_lnL: log-likelihood of alternative model (omega2 free)
        - alt_params: dict of alternative model parameters
        - lrt_statistic: 2*(alt_lnL - null_lnL)
        - pvalue: p-value from chi-square distribution (df=1)
        - df: degrees of freedom (always 1)
        - significant: boolean, True if p < 0.05
        - interpretation: string describing the result

    Notes
    -----
    The branch-site test is the recommended method for detecting positive
    selection on specific lineages (Yang & Nielsen 2002; Zhang et al. 2005).

    The test statistic follows a standard chi-square distribution with df=1
    (NOT a mixture distribution like M8a vs M8).

    Critical values for significance:
    - α = 0.05: LRT > 3.84
    - α = 0.01: LRT > 6.63

    References
    ----------
    Yang, Z., & Nielsen, R. (2002). Codon-substitution models for detecting
    molecular adaptation at individual sites along specific lineages.
    Molecular Biology and Evolution, 19(6), 908-917.

    Zhang, J., Nielsen, R., & Yang, Z. (2005). Evaluation of an improved
    branch-site likelihood method for detecting positive selection at the
    molecular level. Molecular Biology and Evolution, 22(12), 2472-2479.
    """
    print("=" * 70)
    print("BRANCH-SITE TEST OF POSITIVE SELECTION")
    print("=" * 70)
    print()
    print("This test compares:")
    print("  H0 (null):        Model A with omega2 = 1 (no positive selection)")
    print("  H1 (alternative): Model A with omega2 free (positive selection allowed)")
    print()

    # Validate tree has branch labels
    tree.validate_branch_site_labels()
    n_foreground = sum(1 for label in tree.get_branch_labels() if label == 1)
    n_background = sum(1 for label in tree.get_branch_labels() if label == 0)
    print(f"Tree structure:")
    print(f"  {n_foreground} foreground branch(es) (tested for positive selection)")
    print(f"  {n_background} background branch(es)")
    print()

    # ===== Fit NULL MODEL (omega2 = 1) =====
    print("=" * 70)
    print("STEP 1: Fitting NULL model (omega2 = 1)")
    print("=" * 70)
    print()

    # Create a copy of the tree for null model to avoid interference
    import copy
    tree_null = copy.deepcopy(tree)

    null_optimizer = BranchSiteModelAOptimizer(
        alignment=alignment,
        tree=tree_null,
        use_f3x4=use_f3x4,
        optimize_branch_lengths=optimize_branch_lengths,
        fix_omega=True  # NULL MODEL: fix omega2 = 1
    )

    null_kappa, null_omega0, null_omega2, null_p0, null_p1, null_lnL = null_optimizer.optimize(
        init_kappa=init_kappa,
        init_omega0=init_omega0,
        init_omega2=1.0,  # Not used, but pass 1.0 for clarity
        init_p0=init_p0,
        init_p1=init_p1,
        method=method,
        maxiter=maxiter
    )

    print()

    # ===== Fit ALTERNATIVE MODEL (omega2 free) =====
    print("=" * 70)
    print("STEP 2: Fitting ALTERNATIVE model (omega2 free)")
    print("=" * 70)
    print()

    # Create a copy of the tree for alternative model to avoid interference
    tree_alt = copy.deepcopy(tree)

    alt_optimizer = BranchSiteModelAOptimizer(
        alignment=alignment,
        tree=tree_alt,
        use_f3x4=use_f3x4,
        optimize_branch_lengths=optimize_branch_lengths,
        fix_omega=False  # ALTERNATIVE MODEL: omega2 free
    )

    alt_kappa, alt_omega0, alt_omega2, alt_p0, alt_p1, alt_lnL = alt_optimizer.optimize(
        init_kappa=init_kappa,
        init_omega0=init_omega0,
        init_omega2=init_omega2,
        init_p0=init_p0,
        init_p1=init_p1,
        method=method,
        maxiter=maxiter
    )

    print()

    # ===== LIKELIHOOD RATIO TEST =====
    print("=" * 70)
    print("STEP 3: Likelihood Ratio Test")
    print("=" * 70)
    print()

    lrt_statistic, pvalue = calculate_lrt(null_lnL, alt_lnL, df=1)

    print(f"Null model (omega2=1):")
    print(f"  lnL = {null_lnL:.6f}")
    print(f"  Parameters: kappa={null_kappa:.4f}, omega0={null_omega0:.4f}, "
          f"omega2=1.0000, p0={null_p0:.4f}, p1={null_p1:.4f}")
    print()

    print(f"Alternative model (omega2 free):")
    print(f"  lnL = {alt_lnL:.6f}")
    print(f"  Parameters: kappa={alt_kappa:.4f}, omega0={alt_omega0:.4f}, "
          f"omega2={alt_omega2:.4f}, p0={alt_p0:.4f}, p1={alt_p1:.4f}")
    print()

    print(f"Likelihood Ratio Test:")
    print(f"  LRT statistic = 2 * (alt_lnL - null_lnL) = {lrt_statistic:.6f}")
    print(f"  Degrees of freedom = 1")
    print(f"  P-value = {pvalue:.6f}")
    print()

    # Determine significance
    significant = pvalue < 0.05

    if significant:
        if pvalue < 0.01:
            interpretation = "HIGHLY SIGNIFICANT evidence of positive selection on foreground branches (p < 0.01)"
        else:
            interpretation = "SIGNIFICANT evidence of positive selection on foreground branches (p < 0.05)"
    else:
        interpretation = "NO significant evidence of positive selection on foreground branches (p >= 0.05)"

    print("=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(interpretation)
    print()

    if significant and alt_omega2 > 1:
        alt_p2 = 1 - alt_p0 - alt_p1
        print(f"Estimated proportion of positively selected sites on foreground: {alt_p2:.4f}")
        print(f"Estimated omega for positively selected sites: {alt_omega2:.4f}")
        print()

    # Return results
    return {
        'null_lnL': null_lnL,
        'null_params': {
            'kappa': null_kappa,
            'omega0': null_omega0,
            'omega2': null_omega2,  # Always 1.0
            'p0': null_p0,
            'p1': null_p1,
            'p2': 1 - null_p0 - null_p1
        },
        'alt_lnL': alt_lnL,
        'alt_params': {
            'kappa': alt_kappa,
            'omega0': alt_omega0,
            'omega2': alt_omega2,
            'p0': alt_p0,
            'p1': alt_p1,
            'p2': 1 - alt_p0 - alt_p1
        },
        'lrt_statistic': lrt_statistic,
        'pvalue': pvalue,
        'df': 1,
        'significant': significant,
        'interpretation': interpretation
    }
