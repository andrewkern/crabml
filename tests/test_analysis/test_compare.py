"""
Tests for compare_results function.
"""

import pytest
from crabml.analysis.results import LRTResult, compare_results


class TestCompareResults:
    """Tests for the compare_results function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        result1 = LRTResult(
            test_name="M1a vs M2a",
            null_model="M1a",
            alt_model="M2a",
            lnL_null=-1000.0,
            lnL_alt=-995.0,
            df=2,
            null_params={},
            alt_params={},
        )

        result2 = LRTResult(
            test_name="M7 vs M8",
            null_model="M7",
            alt_model="M8",
            lnL_null=-1000.0,
            lnL_alt=-990.0,
            df=2,
            null_params={},
            alt_params={},
        )

        return [result1, result2]

    def test_compare_table_format(self, sample_results):
        """Test table format comparison."""
        output = compare_results(sample_results, format='table')

        assert "COMPARISON" in output
        assert "M1a vs M2a" in output
        assert "M7 vs M8" in output
        assert "LRT" in output
        assert "P-value" in output

    def test_compare_markdown_format(self, sample_results):
        """Test markdown format comparison."""
        output = compare_results(sample_results, format='markdown')

        assert "##" in output  # Markdown header
        assert "|" in output  # Table formatting
        assert "M1a vs M2a" in output
        assert "M7 vs M8" in output

    def test_compare_csv_format(self, sample_results):
        """Test CSV format comparison."""
        output = compare_results(sample_results, format='csv')

        lines = output.split('\n')
        assert len(lines) >= 3  # Header + 2 data rows

        # Check header
        assert "test_name" in lines[0]
        assert "lnL_null" in lines[0]

        # Check data
        assert "M1a vs M2a" in lines[1]
        assert "M7 vs M8" in lines[2]

    def test_compare_default_format(self, sample_results):
        """Test that default format is table."""
        output_default = compare_results(sample_results)
        output_table = compare_results(sample_results, format='table')

        assert output_default == output_table

    def test_compare_single_result(self):
        """Test comparison with single result."""
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

        output = compare_results([result])
        assert "M1a vs M2a" in output

    def test_compare_significance_summary(self, sample_results):
        """Test that significance summary is included."""
        output = compare_results(sample_results, format='table')

        assert "Summary:" in output
        assert "/2 tests" in output  # Shows fraction of significant tests
