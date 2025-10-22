"""
Tests for LRTResult class.
"""

import pytest
import numpy as np

from crabml.analysis.results import LRTResult


class TestLRTResult:
    """Tests for the LRTResult dataclass."""

    def test_basic_creation(self):
        """Test basic creation of LRTResult."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={'kappa': 2.0, 'omega_0': 0.5},
            alt_params={'kappa': 2.1, 'omega_2': 2.5},
        )

        assert result.test_name == "M1a vs M2a"
        assert result.lnL_null == -1000.0
        assert result.lnL_alt == -995.0
        assert result.df == 2

    def test_lrt_property(self):
        """Test LRT statistic calculation."""
        result = LRTResult(
            test_name="Test",
            null_model="Null",
            alt_model="Alt",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={},
        )

        assert result.LRT == 10.0

    def test_pvalue_property(self):
        """Test p-value calculation."""
        result = LRTResult(
            test_name="Test",
            null_model="Null",
            alt_model="Alt",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={},
        )

        # Should be between 0 and 1
        assert 0 <= result.pvalue <= 1

        # For LRT=10 with df=2, p-value should be around 0.0067
        assert 0.005 < result.pvalue < 0.01

    def test_significant_method(self):
        """Test significance testing at different alpha levels."""
        # Create result with p-value around 0.005 (LRT = 10.6, df=2)
        result = LRTResult(
            test_name="Test",
            null_model="Null",
            alt_model="Alt",
            lnL_null=-1000.0,
            lnL_alt=-994.7,  # LRT = 10.6
            df=2,
            null_params={},
            alt_params={},
        )

        # Should be significant at 0.05
        assert result.significant(alpha=0.05)

        # Should be significant at 0.01
        assert result.significant(alpha=0.01)

        # Should not be significant at 0.001
        assert not result.significant(alpha=0.001)

    def test_omega_positive_m2a(self):
        """Test extraction of omega_positive for M2a model."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={'omega_2': 3.5},
        )

        assert result.omega_positive == 3.5

    def test_omega_positive_m8(self):
        """Test extraction of omega_positive for M8 model."""
        result = LRTResult(
            test_name="M7 vs M8",
            null_model="M7",
            alt_model="M8",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={'omega_s': 2.8},
        )

        assert result.omega_positive == 2.8

    def test_omega_positive_none(self):
        """Test omega_positive when not applicable."""
        result = LRTResult(
            test_name="Test",
            null_model="Null",
            alt_model="Alt",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={},
        )

        assert result.omega_positive is None

    def test_proportion_positive_m2a(self):
        """Test extraction of proportion_positive for M2a model."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={'p2': 0.25},
        )

        assert result.proportion_positive == 0.25

    def test_proportion_positive_m8(self):
        """Test extraction of proportion_positive for M8 model."""
        result = LRTResult(
            test_name="M7 vs M8",
            null_model="M7",
            alt_model="M8",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={'p0': 0.7},  # 1 - 0.7 = 0.3 in positive class
        )

        assert np.isclose(result.proportion_positive, 0.3)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={'kappa': 2.0},
            alt_params={'kappa': 2.1, 'omega_2': 3.5, 'p2': 0.25},
        )

        result_dict = result.to_dict()

        assert result_dict['test_name'] == "M1a vs M2a"
        assert result_dict['lnL_null'] == -1000.0
        assert result_dict['lnL_alt'] == -995.0
        assert result_dict['LRT'] == 10.0
        assert 'pvalue' in result_dict
        assert result_dict['omega_positive'] == 3.5
        assert result_dict['proportion_positive'] == 0.25

    def test_summary_output(self):
        """Test that summary() produces formatted output."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={'kappa': 2.0, 'omega_0': 0.5},
            alt_params={'kappa': 2.1, 'omega_2': 3.5, 'p2': 0.25},
        )

        summary = result.summary()

        # Check that key information is in the summary
        assert "M1a vs M2a" in summary
        assert "Log-likelihood" in summary
        assert "LRT statistic" in summary
        assert "P-value" in summary
        assert "CONCLUSION" in summary

    def test_convergence_warning(self):
        """Test that failed optimization is indicated in summary."""
        result = LRTResult(
            test_name="Test",
            null_model="Null",
            alt_model="Alt",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={},
            null_optimization_success=False,
        )

        summary = result.summary()
        assert "WARNING" in summary
        assert "not converge" in summary

    def test_to_json(self):
        """Test JSON export."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={'kappa': 2.0},
            alt_params={'omega_2': 3.5},
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert '"test_name": "M1a vs M2a"' in json_str
        assert '"lnL_null": -1000.0' in json_str

    def test_to_markdown_table(self):
        """Test markdown table export."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={'omega_2': 3.5, 'p2': 0.25},
        )

        markdown = result.to_markdown_table()
        assert "M1a vs M2a" in markdown
        assert "M1a (null)" in markdown
        assert "M2a (alternative)" in markdown
        assert "|" in markdown  # Table formatting

    def test_to_csv_row(self):
        """Test CSV row export."""
        result = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={},
        )

        # Without header
        csv = result.to_csv_row(include_header=False)
        assert csv.startswith("M1a vs M2a")
        assert "-1000.0" in csv

        # With header
        csv_with_header = result.to_csv_row(include_header=True)
        assert "test_name" in csv_with_header
        assert "\n" in csv_with_header
