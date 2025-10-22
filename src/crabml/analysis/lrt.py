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
