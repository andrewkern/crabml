"""
Generic likelihood ratio test utilities.
"""

from scipy.stats import chi2


def calculate_lrt(lnL_null, lnL_alt, df):
    """
    Calculate likelihood ratio test statistic and p-value.

    The likelihood ratio test statistic is:
        LRT = 2 * (lnL_alt - lnL_null)

    Under the null hypothesis, LRT follows a chi-square distribution
    with degrees of freedom equal to the difference in the number of
    free parameters between the models.

    Parameters
    ----------
    lnL_null : float
        Log-likelihood of the null (simpler) model
    lnL_alt : float
        Log-likelihood of the alternative (more complex) model
    df : int
        Degrees of freedom (difference in number of parameters)

    Returns
    -------
    lrt_statistic : float
        The LRT statistic (should be >= 0)
    pvalue : float
        P-value from chi-square distribution

    Notes
    -----
    If lnL_alt < lnL_null (negative LRT), this indicates a problem
    with the optimization or model comparison. The function will
    return LRT=0 and p-value=1.0 as a conservative estimate.
    """
    lrt_statistic = 2 * (lnL_alt - lnL_null)

    # Handle edge cases
    if lrt_statistic < 0:
        # This shouldn't happen in theory, but can occur due to
        # numerical issues or failed optimization
        import warnings
        warnings.warn(
            f"Negative LRT detected (LRT={lrt_statistic:.6f}). "
            "This suggests the null model fits better than the alternative, "
            "which is unexpected. Setting LRT=0 and p-value=1.0. "
            "Check model optimization convergence.",
            UserWarning
        )
        lrt_statistic = 0.0
        pvalue = 1.0
    elif lrt_statistic == 0:
        # Models are identical
        pvalue = 1.0
    else:
        # Standard case: calculate p-value from chi-square distribution
        pvalue = chi2.sf(lrt_statistic, df)

    return lrt_statistic, pvalue


def calculate_lrt_mixture(lnL_null, lnL_alt, df):
    """
    Calculate likelihood ratio test with 50:50 mixture chi-square null distribution.

    This is used for tests where the null hypothesis is on the boundary of the
    parameter space (e.g., M8a vs M8 where omega is tested against the boundary ω=1).

    The null distribution is a 50:50 mixture of a point mass at 0 and χ²(df).
    This means:
    - P(LRT = 0) = 0.5 (null hypothesis is true)
    - P(LRT > 0 | LRT > 0) ~ χ²(df) with probability 0.5

    The p-value is calculated as:
        p = 0.5 * P(χ² > LRT) if LRT > 0
        p = 1.0 if LRT ≤ 0

    Parameters
    ----------
    lnL_null : float
        Log-likelihood of the null (simpler) model
    lnL_alt : float
        Log-likelihood of the alternative (more complex) model
    df : int
        Degrees of freedom (difference in number of parameters)

    Returns
    -------
    lrt_statistic : float
        The LRT statistic (should be >= 0)
    pvalue : float
        P-value from 50:50 mixture distribution

    Notes
    -----
    The 50:50 mixture is appropriate when testing parameters on the boundary
    of the parameter space. See:

    Self, S. G., & Liang, K. Y. (1987). Asymptotic properties of maximum
    likelihood estimators and likelihood ratio tests under nonstandard
    conditions. Journal of the American Statistical Association, 82(398), 605-610.

    Yang, Z., Nielsen, R., Goldman, N., & Pedersen, A. M. K. (2000).
    Codon-substitution models for heterogeneous selection pressure at amino
    acid sites. Genetics, 155(1), 431-449.

    For M8a vs M8 (df=1), critical values at significance levels:
    - α = 0.05: LRT > 2.71
    - α = 0.01: LRT > 5.41
    """
    lrt_statistic = 2 * (lnL_alt - lnL_null)

    # Handle edge cases
    if lrt_statistic < 0:
        # This shouldn't happen in theory
        import warnings
        warnings.warn(
            f"Negative LRT detected (LRT={lrt_statistic:.6f}). "
            "This suggests the null model fits better than the alternative, "
            "which is unexpected. Setting LRT=0 and p-value=1.0. "
            "Check model optimization convergence.",
            UserWarning
        )
        lrt_statistic = 0.0
        pvalue = 1.0
    elif lrt_statistic == 0:
        # On the boundary, maximum p-value
        pvalue = 1.0
    else:
        # 50:50 mixture: p = 0.5 * P(χ² > LRT | χ²(df))
        pvalue = 0.5 * chi2.sf(lrt_statistic, df)

    return lrt_statistic, pvalue
