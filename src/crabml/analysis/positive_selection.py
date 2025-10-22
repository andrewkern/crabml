"""
Hypothesis tests for positive selection using likelihood ratio tests.
"""

from typing import Union, Dict
import warnings
import numpy as np

from ..io.sequences import Alignment
from ..io.trees import Tree
from ..optimize.optimizer import M1aOptimizer, M2aOptimizer, M7Optimizer, M8Optimizer, M8aOptimizer
from .results import LRTResult


def m1a_vs_m2a(
    alignment: Union[str, Alignment],
    tree: Union[str, Tree],
    verbose: bool = True,
    compute_beb: bool = False,
    beb_threshold: float = 0.95
) -> LRTResult:
    """
    Test for positive selection using M1a (null) vs M2a (alternative).

    This is one of the two standard tests for positive selection.
    M1a allows sites to be under purifying selection (ω < 1) or neutral (ω = 1),
    while M2a adds a third class allowing positive selection (ω > 1).

    Parameters
    ----------
    alignment : str or Alignment
        Path to FASTA file or Alignment object
    tree : str or Tree
        Path to Newick file or Tree object
    verbose : bool, default=True
        Print optimization progress
    compute_beb : bool, default=False
        Whether to compute Bayes Empirical Bayes analysis if test is significant
    beb_threshold : float, default=0.95
        Posterior probability threshold for reporting significant sites

    Returns
    -------
    LRTResult
        Test results including LRT statistic, p-value, and model parameters.
        If compute_beb=True and test is significant, result.beb will contain
        BEBResult object.

    Examples
    --------
    >>> from crabml.analysis import m1a_vs_m2a
    >>> result = m1a_vs_m2a('alignment.fasta', 'tree.nwk')
    >>> print(result.summary())
    >>> if result.significant(0.05):
    ...     print(f"Positive selection detected! ω = {result.omega_positive:.2f}")

    >>> # With BEB analysis
    >>> result = m1a_vs_m2a('alignment.fasta', 'tree.nwk', compute_beb=True)
    >>> if result.beb is not None:
    ...     sig_sites = result.beb.significant_sites(threshold=0.95)
    ...     print(f"Sites under positive selection: {sig_sites}")

    Notes
    -----
    The test uses a likelihood ratio test with 2 degrees of freedom
    (two additional parameters in M2a: p2 and ω2).

    References
    ----------
    Yang, Z., Nielsen, R., Goldman, N., & Pedersen, A. M. K. (2000).
    Codon-substitution models for heterogeneous selection pressure at
    amino acid sites. Genetics, 155(1), 431-449.

    Yang, Z., Wong, W.S.W., & Nielsen, R. (2005). Bayes empirical Bayes
    inference of amino acid sites under positive selection. Molecular Biology
    and Evolution, 22(4), 1107-1118.
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

    # Optimize M1a (null model)
    if verbose:
        print("=" * 80)
        print("Optimizing M1a (Nearly Neutral) model...")
        print("=" * 80)

    m1a_optimizer = M1aOptimizer(align, tree_obj)
    try:
        kappa_m1a, omega0_m1a, p0_m1a, lnL_m1a = m1a_optimizer.optimize()
        m1a_success = True
    except Exception as e:
        warnings.warn(f"M1a optimization failed: {e}", UserWarning)
        kappa_m1a, omega0_m1a, p0_m1a, lnL_m1a = 0, 0, 0, -np.inf
        m1a_success = False

    # Optimize M2a (alternative model)
    if verbose:
        print("\n" + "=" * 80)
        print("Optimizing M2a (Positive Selection) model...")
        print("=" * 80)

    m2a_optimizer = M2aOptimizer(align, tree_obj)
    try:
        kappa_m2a, omega0_m2a, omega2_m2a, p0_m2a, p1_m2a, lnL_m2a = m2a_optimizer.optimize()
        m2a_success = True
    except Exception as e:
        warnings.warn(f"M2a optimization failed: {e}", UserWarning)
        kappa_m2a, omega0_m2a, omega2_m2a, p0_m2a, p1_m2a, lnL_m2a = 0, 0, 0, 0, 0, -np.inf
        m2a_success = False

    # Calculate p2 for M2a
    p2_m2a = 1.0 - p0_m2a - p1_m2a if m2a_success else 0

    # Create result object
    result = LRTResult(
        test_name="M1a vs M2a",
        null_model="M1a",
        alt_model="M2a",
        lnL_null=lnL_m1a,
        lnL_alt=lnL_m2a,
        df=2,  # Two extra parameters in M2a: p2 and ω2
        null_params={
            'kappa': kappa_m1a,
            'omega_0': omega0_m1a,
            'p0': p0_m1a,
            'p1': 1.0 - p0_m1a,
        },
        alt_params={
            'kappa': kappa_m2a,
            'omega_0': omega0_m2a,
            'omega_2': omega2_m2a,
            'p0': p0_m2a,
            'p1': p1_m2a,
            'p2': p2_m2a,
        },
        null_optimization_success=m1a_success,
        alt_optimization_success=m2a_success,
    )

    # Compute BEB if requested and test is significant
    if compute_beb and result.significant(0.05) and m2a_success:
        if verbose:
            print("\n" + "=" * 80)
            print("Computing Bayes Empirical Bayes (BEB) analysis...")
            print("=" * 80)

        try:
            from .beb import BEBCalculator
            from ..models.codon import M2aCodonModel

            # Create BEB calculator for M2a
            m2a_mle_params = {
                'kappa': kappa_m2a,
                'omega0': omega0_m2a,
                'omega2': omega2_m2a,
                'p0': p0_m2a,
                'p1': p1_m2a,
            }

            beb_calc = BEBCalculator(
                mle_params=m2a_mle_params,
                alignment=align,
                tree=tree_obj,
                model_class=M2aCodonModel,
                param_names=['kappa', 'omega0', 'omega2', 'p0', 'p1'],
                fix_branch_lengths=True,
                n_grid_points=5
            )

            # Run BEB
            beb_result = beb_calc.calculate_beb()
            result.beb = beb_result

            # Print summary
            if verbose:
                print("\n")
                print(beb_result.summary(threshold_95=beb_threshold))

        except Exception as e:
            warnings.warn(f"BEB calculation failed: {e}", UserWarning)
            result.beb = None
    else:
        result.beb = None

    if verbose:
        print("\n" + "=" * 80)
        print(result.summary())

    return result


def m7_vs_m8(
    alignment: Union[str, Alignment],
    tree: Union[str, Tree],
    verbose: bool = True
) -> LRTResult:
    """
    Test for positive selection using M7 (null) vs M8 (alternative).

    This is one of the two standard tests for positive selection.
    M7 uses a beta distribution for ω constrained to (0, 1), while M8
    adds an additional site class allowing ω > 1.

    Parameters
    ----------
    alignment : str or Alignment
        Path to FASTA file or Alignment object
    tree : str or Tree
        Path to Newick file or Tree object
    verbose : bool, default=True
        Print optimization progress

    Returns
    -------
    LRTResult
        Test results including LRT statistic, p-value, and model parameters

    Examples
    --------
    >>> from crabml.analysis import m7_vs_m8
    >>> result = m7_vs_m8('alignment.fasta', 'tree.nwk')
    >>> print(result.summary())
    >>> if result.significant(0.05):
    ...     print(f"Positive selection detected! ω = {result.omega_positive:.2f}")

    Notes
    -----
    The test uses a likelihood ratio test with 2 degrees of freedom
    (two additional parameters in M8: p0 and ωs).

    M7 vs M8 is generally more powerful than M1a vs M2a for detecting
    positive selection.

    References
    ----------
    Yang, Z., Wong, W. S., & Nielsen, R. (2005). Bayes empirical Bayes
    inference of amino acid sites under positive selection. Molecular
    Biology and Evolution, 22(4), 1107-1118.
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

    # Optimize M7 (null model)
    if verbose:
        print("=" * 80)
        print("Optimizing M7 (Beta) model...")
        print("=" * 80)

    m7_optimizer = M7Optimizer(align, tree_obj)
    try:
        kappa_m7, p_beta_m7, q_beta_m7, lnL_m7 = m7_optimizer.optimize()
        m7_success = True
    except Exception as e:
        warnings.warn(f"M7 optimization failed: {e}", UserWarning)
        kappa_m7, p_beta_m7, q_beta_m7, lnL_m7 = 0, 0, 0, -np.inf
        m7_success = False

    # Optimize M8 (alternative model)
    if verbose:
        print("\n" + "=" * 80)
        print("Optimizing M8 (Beta & ω>1) model...")
        print("=" * 80)

    m8_optimizer = M8Optimizer(align, tree_obj)
    try:
        kappa_m8, p0_m8, p_beta_m8, q_beta_m8, omega_s_m8, lnL_m8 = m8_optimizer.optimize()
        m8_success = True
    except Exception as e:
        warnings.warn(f"M8 optimization failed: {e}", UserWarning)
        kappa_m8, p0_m8, p_beta_m8, q_beta_m8, omega_s_m8, lnL_m8 = 0, 0, 0, 0, 0, -np.inf
        m8_success = False

    # Create result object
    result = LRTResult(
        test_name="M7 vs M8",
        null_model="M7",
        alt_model="M8",
        lnL_null=lnL_m7,
        lnL_alt=lnL_m8,
        df=2,  # Two extra parameters in M8: p0 and ωs
        null_params={
            'kappa': kappa_m7,
            'p_beta': p_beta_m7,
            'q_beta': q_beta_m7,
        },
        alt_params={
            'kappa': kappa_m8,
            'p0': p0_m8,
            'p_beta': p_beta_m8,
            'q_beta': q_beta_m8,
            'omega_s': omega_s_m8,
        },
        null_optimization_success=m7_success,
        alt_optimization_success=m8_success,
    )

    if verbose:
        print("\n" + "=" * 80)
        print(result.summary())

    return result


def m8a_vs_m8(
    alignment: Union[str, Alignment],
    tree: Union[str, Tree],
    verbose: bool = True
) -> LRTResult:
    """
    Test for positive selection using M8a (null) vs M8 (alternative).

    This is an alternative to M7 vs M8 that uses a 50:50 mixture chi-square
    null distribution. M8a is M8 with omega_s fixed to 1 (neutral), while
    M8 allows omega_s > 1 (positive selection).

    The null distribution is a 50:50 mixture of a point mass at 0 and χ²(1)
    because the null hypothesis (ω = 1) is on the boundary of the parameter
    space (ω >= 1).

    Parameters
    ----------
    alignment : str or Alignment
        Path to FASTA file or Alignment object
    tree : str or Tree
        Path to Newick file or Tree object
    verbose : bool, default=True
        Print optimization progress

    Returns
    -------
    LRTResult
        Test results including LRT statistic, p-value, and model parameters

    Examples
    --------
    >>> from crabml.analysis import m8a_vs_m8
    >>> result = m8a_vs_m8('alignment.fasta', 'tree.nwk')
    >>> print(result.summary())
    >>> if result.significant(0.05):
    ...     print(f"Positive selection detected! ω = {result.omega_positive:.2f}")

    Notes
    -----
    The test uses a likelihood ratio test with 1 degree of freedom
    (one additional parameter in M8: ωs is free vs fixed to 1).

    Because ω=1 is on the boundary of the parameter space, the null
    distribution is a 50:50 mixture of point mass 0 and χ²(1).

    Critical values for α levels (from Self & Liang 1987):
    - α = 0.05: LRT > 2.71
    - α = 0.01: LRT > 5.41

    References
    ----------
    Self, S. G., & Liang, K. Y. (1987). Asymptotic properties of maximum
    likelihood estimators and likelihood ratio tests under nonstandard
    conditions. Journal of the American Statistical Association, 82(398), 605-610.

    Yang, Z., Nielsen, R., Goldman, N., & Pedersen, A. M. K. (2000).
    Codon-substitution models for heterogeneous selection pressure at amino
    acid sites. Genetics, 155(1), 431-449.
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

    # Optimize M8a (null model - omega_s fixed to 1)
    if verbose:
        print("=" * 80)
        print("Optimizing M8a (Beta & ω=1) model...")
        print("=" * 80)

    m8a_optimizer = M8aOptimizer(align, tree_obj)
    try:
        kappa_m8a, p0_m8a, p_beta_m8a, q_beta_m8a, lnL_m8a = m8a_optimizer.optimize()
        m8a_success = True
    except Exception as e:
        warnings.warn(f"M8a optimization failed: {e}", UserWarning)
        kappa_m8a, p0_m8a, p_beta_m8a, q_beta_m8a, lnL_m8a = 0, 0, 0, 0, -np.inf
        m8a_success = False

    # Optimize M8 (alternative model - omega_s > 1 allowed)
    if verbose:
        print("\n" + "=" * 80)
        print("Optimizing M8 (Beta & ω>1) model...")
        print("=" * 80)

    m8_optimizer = M8Optimizer(align, tree_obj)
    try:
        kappa_m8, p0_m8, p_beta_m8, q_beta_m8, omega_s_m8, lnL_m8 = m8_optimizer.optimize()
        m8_success = True
    except Exception as e:
        warnings.warn(f"M8 optimization failed: {e}", UserWarning)
        kappa_m8, p0_m8, p_beta_m8, q_beta_m8, omega_s_m8, lnL_m8 = 0, 0, 0, 0, 0, -np.inf
        m8_success = False

    # Create result object with 50:50 mixture null distribution
    from .lrt import calculate_lrt_mixture

    lrt_stat, pval = calculate_lrt_mixture(lnL_m8a, lnL_m8, df=1)

    result = LRTResult(
        test_name="M8a vs M8",
        null_model="M8a",
        alt_model="M8",
        lnL_null=lnL_m8a,
        lnL_alt=lnL_m8,
        df=1,  # One extra parameter in M8: ωs (free vs fixed to 1)
        null_params={
            'kappa': kappa_m8a,
            'p0': p0_m8a,
            'p_beta': p_beta_m8a,
            'q_beta': q_beta_m8a,
            'omega_s': 1.0,  # Fixed in M8a
        },
        alt_params={
            'kappa': kappa_m8,
            'p0': p0_m8,
            'p_beta': p_beta_m8,
            'q_beta': q_beta_m8,
            'omega_s': omega_s_m8,
        },
        null_optimization_success=m8a_success,
        alt_optimization_success=m8_success,
        _override_lrt=lrt_stat,
        _override_pvalue=pval,
    )

    if verbose:
        print("\n" + "=" * 80)
        print(result.summary())
        print("\nNote: This test uses a 50:50 mixture chi-square null distribution")
        print("because ω=1 is on the boundary of the parameter space.")
        print("Critical values: α=0.05 → LRT>2.71, α=0.01 → LRT>5.41")

    return result


def positive_selection(
    alignment: Union[str, Alignment],
    tree: Union[str, Tree],
    test: str = 'both',
    verbose: bool = True
) -> Union[LRTResult, Dict[str, LRTResult]]:
    """
    Test for positive selection using standard likelihood ratio tests.

    This is a convenience function that runs one or both of the standard
    tests for positive selection: M1a vs M2a and M7 vs M8.

    Parameters
    ----------
    alignment : str or Alignment
        Path to FASTA file or Alignment object with codon sequences
    tree : str or Tree
        Path to Newick file or Tree object
    test : str, default='both'
        Which test(s) to run:
        - 'M1a_vs_M2a' or 'm1a_vs_m2a': Test M1a vs M2a only
        - 'M7_vs_M8' or 'm7_vs_m8': Test M7 vs M8 only
        - 'both': Run both tests (default)
    verbose : bool, default=True
        Print optimization progress and results

    Returns
    -------
    LRTResult or dict of LRTResult
        If test='both', returns dict with keys 'M1a_vs_M2a' and 'M7_vs_M8'
        Otherwise, returns single LRTResult object

    Examples
    --------
    Run both tests:

    >>> from crabml.analysis import positive_selection
    >>> results = positive_selection('lysozyme.fasta', 'lysozyme.tree')
    >>> print(results['M1a_vs_M2a'].summary())
    >>> print(results['M7_vs_M8'].summary())

    Run single test:

    >>> result = positive_selection('lysozyme.fasta', 'lysozyme.tree',
    ...                              test='M1a_vs_M2a')
    >>> if result.significant(0.05):
    ...     print("Positive selection detected!")

    With custom parameters:

    >>> result = positive_selection('lysozyme.fasta', 'lysozyme.tree',
    ...                              test='M7_vs_M8',
    ...                              verbose=False)

    Notes
    -----
    Both tests use likelihood ratio tests with 2 degrees of freedom.
    The M7 vs M8 test is generally more powerful than M1a vs M2a.

    It is recommended to run both tests. Positive selection is well-supported
    if both tests are significant at α = 0.05.

    See Also
    --------
    m1a_vs_m2a : Run M1a vs M2a test only
    m7_vs_m8 : Run M7 vs M8 test only

    References
    ----------
    Yang, Z. (2007). PAML 4: phylogenetic analysis by maximum likelihood.
    Molecular Biology and Evolution, 24(8), 1586-1591.
    """
    # Normalize test name
    test = test.lower().replace('_', '').replace('-', '')

    if test == 'both':
        if verbose:
            print("\n" + "=" * 80)
            print("Running both M1a vs M2a and M7 vs M8 tests")
            print("=" * 80 + "\n")

        result_m1a_m2a = m1a_vs_m2a(
            alignment, tree, verbose=verbose
        )

        if verbose:
            print("\n\n")

        result_m7_m8 = m7_vs_m8(
            alignment, tree, verbose=verbose
        )

        return {
            'M1a_vs_M2a': result_m1a_m2a,
            'M7_vs_M8': result_m7_m8,
        }

    elif test in ('m1avsm2a', 'm1avm2a'):
        return m1a_vs_m2a(alignment, tree, verbose=verbose)

    elif test in ('m7vsm8', 'm7vm8'):
        return m7_vs_m8(alignment, tree, verbose=verbose)

    else:
        raise ValueError(
            f"Unknown test: '{test}'. "
            "Must be 'M1a_vs_M2a', 'M7_vs_M8', or 'both'."
        )
