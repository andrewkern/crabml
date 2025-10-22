"""
Integration tests for positive selection analysis using PAML reference data.

These tests verify that our hypothesis testing functions work correctly
with real data and produce results consistent with PAML.
"""

import pytest
import os

from crabml.analysis import m1a_vs_m2a, m7_vs_m8, positive_selection
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


@pytest.fixture
def lysozyme_files():
    """Provide paths to lysozyme test data."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_examples_dir = os.path.join(base_dir, 'data', 'paml_examples', 'lysozyme')
    data_dir = os.path.join(base_dir, 'data')

    return {
        'phylip': os.path.join(data_examples_dir, 'lysozymeSmall.nuc'),
        'tree': os.path.join(data_dir, 'lysozyme.tree'),  # Simple tree with names
    }


@pytest.fixture
def lysozyme_data(lysozyme_files):
    """Load and return lysozyme alignment and tree."""
    alignment = Alignment.from_phylip(lysozyme_files['phylip'], seqtype='codon')
    with open(lysozyme_files['tree']) as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    return {
        'alignment': alignment,
        'tree': tree,
    }


@pytest.fixture
def paml_reference_likelihoods():
    """PAML reference log-likelihoods for lysozyme dataset."""
    return {
        'M1a': -902.503872,
        'M2a': -899.998568,
        'M7': -902.510018,
        'M8': -899.999237,
    }


class TestM1aVsM2aIntegration:
    """Integration tests for M1a vs M2a test."""

    @pytest.mark.slow
    def test_m1a_vs_m2a_runs(self, lysozyme_data):
        """Test that M1a vs M2a test runs without errors."""
        result = m1a_vs_m2a(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        # Check that result is returned
        assert result is not None
        assert result.test_name == "M1a vs M2a"

        # Check that likelihoods are reasonable
        assert result.lnL_null < 0  # Log-likelihood should be negative
        assert result.lnL_alt < 0
        assert result.lnL_alt >= result.lnL_null  # Alternative should fit better

        # Check that parameters are reasonable
        assert result.alt_params['omega_2'] > 1  # Positive selection class

    @pytest.mark.slow
    def test_m1a_vs_m2a_matches_paml(self, lysozyme_data, paml_reference_likelihoods):
        """Test that M1a vs M2a likelihoods match PAML reference values."""
        result = m1a_vs_m2a(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        # Check that likelihoods are close to PAML values
        # Allow some tolerance for optimization differences
        assert abs(result.lnL_null - paml_reference_likelihoods['M1a']) < 0.01
        assert abs(result.lnL_alt - paml_reference_likelihoods['M2a']) < 0.01

    @pytest.mark.slow
    def test_m1a_vs_m2a_lrt_statistics(self, lysozyme_data):
        """Test that LRT statistics are calculated correctly."""
        result = m1a_vs_m2a(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        # Check LRT calculation
        expected_lrt = 2 * (result.lnL_alt - result.lnL_null)
        assert abs(result.LRT - expected_lrt) < 1e-6

        # Check degrees of freedom
        assert result.df == 2

        # Check p-value is valid
        assert 0 <= result.pvalue <= 1


class TestM7VsM8Integration:
    """Integration tests for M7 vs M8 test."""

    @pytest.mark.slow
    def test_m7_vs_m8_runs(self, lysozyme_data):
        """Test that M7 vs M8 test runs without errors."""
        result = m7_vs_m8(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        # Check that result is returned
        assert result is not None
        assert result.test_name == "M7 vs M8"

        # Check that likelihoods are reasonable
        assert result.lnL_null < 0
        assert result.lnL_alt < 0
        assert result.lnL_alt >= result.lnL_null

        # Check that M8 has positive selection parameter
        assert result.alt_params['omega_s'] > 1

    @pytest.mark.slow
    def test_m7_vs_m8_matches_paml(self, lysozyme_data, paml_reference_likelihoods):
        """Test that M7 vs M8 likelihoods match PAML reference values."""
        result = m7_vs_m8(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        # Check that likelihoods are close to PAML values
        assert abs(result.lnL_null - paml_reference_likelihoods['M7']) < 0.01
        assert abs(result.lnL_alt - paml_reference_likelihoods['M8']) < 0.01

    @pytest.mark.slow
    def test_m7_vs_m8_lrt_statistics(self, lysozyme_data):
        """Test that LRT statistics are calculated correctly."""
        result = m7_vs_m8(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        # Check LRT calculation
        expected_lrt = 2 * (result.lnL_alt - result.lnL_null)
        assert abs(result.LRT - expected_lrt) < 1e-6

        # Check degrees of freedom
        assert result.df == 2


class TestPositiveSelectionWrapper:
    """Integration tests for unified test_positive_selection function."""

    @pytest.mark.slow
    def test_both_tests(self, lysozyme_data):
        """Test running both tests with 'both' option."""
        results = positive_selection(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            test='both',
            verbose=False
        )

        # Should return dictionary
        assert isinstance(results, dict)
        assert 'M1a_vs_M2a' in results
        assert 'M7_vs_M8' in results

        # Both results should be valid
        assert results['M1a_vs_M2a'].lnL_null < 0
        assert results['M7_vs_M8'].lnL_null < 0

    @pytest.mark.slow
    def test_single_test_m1a_m2a(self, lysozyme_data):
        """Test running M1a vs M2a only."""
        result = positive_selection(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            test='M1a_vs_M2a',
            verbose=False
        )

        # Should return single result, not dict
        assert not isinstance(result, dict)
        assert result.test_name == "M1a vs M2a"

    @pytest.mark.slow
    def test_single_test_m7_m8(self, lysozyme_data):
        """Test running M7 vs M8 only."""
        result = positive_selection(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            test='M7_vs_M8',
            verbose=False
        )

        # Should return single result, not dict
        assert not isinstance(result, dict)
        assert result.test_name == "M7 vs M8"

    def test_invalid_test_name(self, lysozyme_data):
        """Test that invalid test name raises error."""
        with pytest.raises(ValueError, match="Unknown test"):
            positive_selection(
                lysozyme_data['alignment'],
                lysozyme_data['tree'],
                test='invalid_test',
                verbose=False
            )

    @pytest.mark.slow
    def test_case_insensitive_test_names(self, lysozyme_data):
        """Test that test names are case-insensitive."""
        # All these should work
        result1 = positive_selection(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            test='m1a_vs_m2a',
            verbose=False
        )

        result2 = positive_selection(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            test='M1A_VS_M2A',
            verbose=False
        )

        # Should both run successfully
        assert result1.test_name == "M1a vs M2a"
        assert result2.test_name == "M1a vs M2a"


class TestResultOutput:
    """Test that result objects have useful output methods."""

    @pytest.mark.slow
    def test_summary_output(self, lysozyme_data):
        """Test that summary() produces human-readable output."""
        result = m1a_vs_m2a(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        summary = result.summary()

        # Summary should be a string
        assert isinstance(summary, str)

        # Should contain key information
        assert "M1a" in summary
        assert "M2a" in summary
        assert "Log-likelihood" in summary

    @pytest.mark.slow
    def test_to_dict_output(self, lysozyme_data):
        """Test that to_dict() produces valid dictionary."""
        result = m1a_vs_m2a(
            lysozyme_data['alignment'],
            lysozyme_data['tree'],
            verbose=False
        )

        result_dict = result.to_dict()

        # Should be a dictionary
        assert isinstance(result_dict, dict)

        # Should contain expected keys
        assert 'lnL_null' in result_dict
        assert 'lnL_alt' in result_dict
        assert 'LRT' in result_dict
        assert 'pvalue' in result_dict
