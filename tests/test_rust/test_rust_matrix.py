"""
Test Rust matrix exponential against Python implementation.

Test-driven development: These tests define the contract between
Python and Rust implementations.
"""

import numpy as np
import pytest

# Will be available after maturin build
try:
    import pycodeml_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    pytest.skip("Rust extension not built yet", allow_module_level=True)

from pycodeml.core.matrix import matrix_exponential as python_matrix_exponential


class TestRustMatrixExponential:
    """Test Rust matrix exponential matches Python exactly."""

    def test_matrix_exponential_identity(self):
        """Test exp(0 * Q) = Identity"""
        n = 61  # Codon states
        q = np.eye(n, dtype=np.float64)
        pi = np.ones(n, dtype=np.float64) / n
        t = 0.0

        # Python version
        p_python = python_matrix_exponential(q, t)

        # Rust version
        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        # Should be exact match for identity
        np.testing.assert_array_almost_equal(p_python, p_rust, decimal=12)

    def test_matrix_exponential_small_time(self):
        """Test matrix exponential for small t"""
        n = 4  # Simplified for testing

        # Create symmetric rate matrix
        q = np.array([
            [-2.0,  1.0,  0.5,  0.5],
            [ 1.0, -2.0,  0.5,  0.5],
            [ 0.5,  0.5, -2.0,  1.0],
            [ 0.5,  0.5,  1.0, -2.0]
        ], dtype=np.float64)

        pi = np.ones(n, dtype=np.float64) / n
        t = 0.01

        # Python version
        p_python = python_matrix_exponential(q, t)

        # Rust version
        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        # Should match within numerical precision
        np.testing.assert_array_almost_equal(p_python, p_rust, decimal=10,
            err_msg="Rust and Python matrix exponentials should match")

    def test_matrix_exponential_moderate_time(self):
        """Test matrix exponential for moderate t"""
        n = 4
        q = np.array([
            [-2.0,  1.0,  0.5,  0.5],
            [ 1.0, -2.0,  0.5,  0.5],
            [ 0.5,  0.5, -2.0,  1.0],
            [ 0.5,  0.5,  1.0, -2.0]
        ], dtype=np.float64)

        pi = np.ones(n, dtype=np.float64) / n
        t = 0.1

        p_python = python_matrix_exponential(q, t)
        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        np.testing.assert_array_almost_equal(p_python, p_rust, decimal=9)

    def test_matrix_exponential_rows_sum_to_one(self):
        """Test that P(t) rows sum to 1 (stochastic matrix property)"""
        n = 4
        q = np.array([
            [-2.0,  1.0,  0.5,  0.5],
            [ 1.0, -2.0,  0.5,  0.5],
            [ 0.5,  0.5, -2.0,  1.0],
            [ 0.5,  0.5,  1.0, -2.0]
        ], dtype=np.float64)

        pi = np.ones(n, dtype=np.float64) / n
        t = 0.1

        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        # Each row should sum to 1
        row_sums = p_rust.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(n), decimal=12)

    def test_matrix_exponential_non_negative(self):
        """Test that P(t) has all non-negative entries"""
        n = 4
        q = np.array([
            [-2.0,  1.0,  0.5,  0.5],
            [ 1.0, -2.0,  0.5,  0.5],
            [ 0.5,  0.5, -2.0,  1.0],
            [ 0.5,  0.5,  1.0, -2.0]
        ], dtype=np.float64)

        pi = np.ones(n, dtype=np.float64) / n
        t = 0.1

        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        # All entries should be >= 0
        assert np.all(p_rust >= -1e-10), "P(t) should have non-negative entries"

    def test_matrix_exponential_detailed_balance(self):
        """Test detailed balance: π_i P_{ij} = π_j P_{ji}"""
        n = 3
        pi = np.array([0.3, 0.5, 0.2], dtype=np.float64)

        # Create reversible Q matrix
        q = np.array([
            [-1.2,  1.0,  0.2],
            [ 0.6, -1.1,  0.5],
            [ 0.3,  1.25, -1.55]
        ], dtype=np.float64)

        t = 0.1
        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        # Check detailed balance
        for i in range(n):
            for j in range(n):
                lhs = pi[i] * p_rust[i, j]
                rhs = pi[j] * p_rust[j, i]
                np.testing.assert_almost_equal(lhs, rhs, decimal=8,
                    err_msg=f"Detailed balance violated at ({i},{j})")

    @pytest.mark.slow
    def test_matrix_exponential_codon_sized(self):
        """Test with full codon-sized matrix (61x61)"""
        n = 61
        np.random.seed(42)

        # Create random symmetric matrix
        q_sym = np.random.randn(n, n)
        q_sym = (q_sym + q_sym.T) / 2

        # Make rows sum to zero
        q = q_sym - np.diag(q_sym.sum(axis=1))

        pi = np.ones(n, dtype=np.float64) / n
        t = 0.05

        p_python = python_matrix_exponential(q, t)
        p_rust = pycodeml_rust.matrix_exponential(q, pi, t)

        # Should match within numerical precision
        np.testing.assert_array_almost_equal(p_python, p_rust, decimal=8)

    def test_matrix_exponential_performance(self, benchmark):
        """Benchmark Rust vs Python matrix exponential"""
        n = 61
        np.random.seed(42)

        q_sym = np.random.randn(n, n)
        q_sym = (q_sym + q_sym.T) / 2
        q = q_sym - np.diag(q_sym.sum(axis=1))

        pi = np.ones(n, dtype=np.float64) / n
        t = 0.1

        # Benchmark Rust version
        result = benchmark(pycodeml_rust.matrix_exponential, q, pi, t)

        # Verify correctness
        p_python = python_matrix_exponential(q, t)
        np.testing.assert_array_almost_equal(result, p_python, decimal=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
