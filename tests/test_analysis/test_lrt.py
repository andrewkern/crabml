"""
Tests for likelihood ratio test calculations.
"""

import pytest
import numpy as np
from scipy.stats import chi2

from crabml.analysis.lrt import calculate_lrt


class TestCalculateLRT:
    """Tests for the calculate_lrt function."""

    def test_basic_lrt_calculation(self):
        """Test basic LRT calculation with known values."""
        lnL_null = -1000.0
        lnL_alt = -995.0
        df = 2

        lrt_stat, pvalue = calculate_lrt(lnL_null, lnL_alt, df)

        # LRT should be 2 * (lnL_alt - lnL_null)
        expected_lrt = 2 * (-995.0 - (-1000.0))
        assert np.isclose(lrt_stat, expected_lrt)
        assert lrt_stat == 10.0

        # P-value should match chi-square distribution
        expected_pvalue = chi2.sf(10.0, 2)
        assert np.isclose(pvalue, expected_pvalue)

    def test_identical_likelihoods(self):
        """Test when null and alternative have same likelihood."""
        lnL = -900.0
        lrt_stat, pvalue = calculate_lrt(lnL, lnL, df=2)

        assert lrt_stat == 0.0
        assert pvalue == 1.0

    def test_negative_lrt_warning(self):
        """Test that negative LRT raises warning and returns conservative values."""
        lnL_null = -995.0
        lnL_alt = -1000.0  # Alternative worse than null (shouldn't happen)

        with pytest.warns(UserWarning, match="Negative LRT detected"):
            lrt_stat, pvalue = calculate_lrt(lnL_null, lnL_alt, df=2)

        # Should return conservative estimates
        assert lrt_stat == 0.0
        assert pvalue == 1.0

    def test_different_df(self):
        """Test LRT with different degrees of freedom."""
        lnL_null = -1000.0
        lnL_alt = -995.0

        # Test with df=1
        lrt_stat, pvalue1 = calculate_lrt(lnL_null, lnL_alt, df=1)
        expected_pvalue1 = chi2.sf(10.0, 1)
        assert np.isclose(pvalue1, expected_pvalue1)

        # Test with df=3
        lrt_stat, pvalue3 = calculate_lrt(lnL_null, lnL_alt, df=3)
        expected_pvalue3 = chi2.sf(10.0, 3)
        assert np.isclose(pvalue3, expected_pvalue3)

        # P-value should decrease as df increases for same LRT
        assert pvalue3 > pvalue1

    def test_strong_signal(self):
        """Test with strong signal (large difference in likelihoods)."""
        lnL_null = -1000.0
        lnL_alt = -950.0  # Big improvement
        df = 2

        lrt_stat, pvalue = calculate_lrt(lnL_null, lnL_alt, df)

        assert lrt_stat == 100.0
        assert pvalue < 0.001  # Should be highly significant

    def test_weak_signal(self):
        """Test with weak signal (small difference in likelihoods)."""
        lnL_null = -1000.0
        lnL_alt = -999.5  # Small improvement
        df = 2

        lrt_stat, pvalue = calculate_lrt(lnL_null, lnL_alt, df)

        assert lrt_stat == 1.0
        assert pvalue > 0.05  # Should not be significant
