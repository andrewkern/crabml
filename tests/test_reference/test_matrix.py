"""
Reference tests for matrix operations.

These tests validate mathematical correctness against analytical solutions
and known properties of transition probability matrices.
"""

import numpy as np
import pytest
from pycodeml.core.matrix import (
    check_detailed_balance,
    create_reversible_Q,
    eigen_decompose_rev,
    matrix_exponential,
)


class TestMatrixExponential:
    """Test matrix exponential computation."""

    def test_jc69_analytical(self):
        """Test matrix exponential against analytical JC69 solution."""
        # JC69 model: all substitution rates equal
        alpha = 0.25
        Q = np.array(
            [
                [-3 * alpha, alpha, alpha, alpha],
                [alpha, -3 * alpha, alpha, alpha],
                [alpha, alpha, -3 * alpha, alpha],
                [alpha, alpha, alpha, -3 * alpha],
            ]
        )
        t = 0.1

        P = matrix_exponential(Q, t)

        # Analytical solution for JC69: P(t) = 1/4 + 3/4 * exp(-4αt) on diagonal
        #                                      1/4 - 1/4 * exp(-4αt) off-diagonal
        e_term = np.exp(-4 * alpha * t)
        expected_diag = 0.25 + 0.75 * e_term
        expected_off = 0.25 - 0.25 * e_term

        np.testing.assert_allclose(np.diag(P), expected_diag, rtol=1e-10)
        np.testing.assert_allclose(P[0, 1], expected_off, rtol=1e-10)
        np.testing.assert_allclose(P[0, 2], expected_off, rtol=1e-10)

    def test_row_sums_one(self):
        """Test that transition probability matrix rows sum to 1."""
        Q = self._random_rate_matrix(4)
        t = 0.1
        P = matrix_exponential(Q, t)

        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_all_positive(self):
        """Test that all entries are non-negative."""
        Q = self._random_rate_matrix(4)
        t = 0.1
        P = matrix_exponential(Q, t)

        assert np.all(P >= -1e-10)  # Allow tiny numerical errors

    def test_identity_at_zero(self):
        """Test P(0) = I (identity matrix)."""
        Q = self._random_rate_matrix(4)
        P0 = matrix_exponential(Q, 0.0)

        np.testing.assert_allclose(P0, np.eye(4), rtol=1e-10)

    def test_semigroup_property(self):
        """Test P(t1 + t2) = P(t1) @ P(t2)."""
        Q = self._random_rate_matrix(4)
        t1, t2 = 0.05, 0.15

        P_sum = matrix_exponential(Q, t1 + t2)
        P_prod = matrix_exponential(Q, t1) @ matrix_exponential(Q, t2)

        np.testing.assert_allclose(P_sum, P_prod, rtol=1e-8)

    def test_large_time(self):
        """Test convergence to stationary distribution at large t."""
        # For reversible matrix with uniform stationary distribution
        pi = np.ones(4) / 4
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi)

        # At large t, all rows should converge to stationary distribution
        P = matrix_exponential(Q, 100.0)

        for i in range(4):
            np.testing.assert_allclose(P[i, :], pi, rtol=1e-6)

    @staticmethod
    def _random_rate_matrix(n: int, seed: int = 42) -> np.ndarray:
        """Create a random reversible rate matrix."""
        rng = np.random.RandomState(seed)
        pi = rng.dirichlet(np.ones(n))
        rates = rng.uniform(0, 1, (n, n))
        rates = (rates + rates.T) / 2  # Make symmetric
        return create_reversible_Q(rates, pi)


class TestEigenDecomposition:
    """Test eigendecomposition of reversible rate matrices."""

    def test_reconstruction(self):
        """Test Q = U @ diag(eigenvalues) @ V."""
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi)

        eigenvalues, U, V = eigen_decompose_rev(Q, pi)

        # Reconstruct Q
        Q_reconstructed = U @ np.diag(eigenvalues) @ V

        np.testing.assert_allclose(Q, Q_reconstructed, rtol=1e-10)

    def test_largest_eigenvalue_zero(self):
        """Test that largest eigenvalue is approximately zero."""
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi)

        eigenvalues, U, V = eigen_decompose_rev(Q, pi)

        # Largest eigenvalue (last element) should be ~0
        assert np.abs(eigenvalues[-1]) < 1e-10

    def test_stationary_distribution(self):
        """Test eigenvector corresponding to eigenvalue 0 gives stationary distribution."""
        pi = np.array([0.3, 0.2, 0.4, 0.1])
        rates = np.array(
            [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
        )
        Q = create_reversible_Q(rates, pi)

        eigenvalues, U, V = eigen_decompose_rev(Q, pi)

        # Eigenvector for eigenvalue 0 (largest, last index)
        stationary = U[:, -1] * V[-1, :]
        stationary = stationary / stationary.sum()  # Normalize

        np.testing.assert_allclose(stationary, pi, rtol=1e-10)

    def test_eigenvalues_real_negative(self):
        """Test all eigenvalues except largest are real and negative."""
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi)

        eigenvalues, U, V = eigen_decompose_rev(Q, pi)

        # All should be real (no imaginary part)
        assert np.all(np.imag(eigenvalues) == 0)

        # All except last should be negative
        assert np.all(eigenvalues[:-1] < 0)

    def test_transition_via_eigen(self):
        """Test P(t) = U @ diag(exp(eigenvalues*t)) @ V."""
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi)
        t = 0.1

        # Method 1: Direct matrix exponential
        P_direct = matrix_exponential(Q, t)

        # Method 2: Via eigendecomposition
        eigenvalues, U, V = eigen_decompose_rev(Q, pi)
        exp_lambda_t = np.exp(eigenvalues * t)
        P_eigen = U @ np.diag(exp_lambda_t) @ V

        np.testing.assert_allclose(P_direct, P_eigen, rtol=1e-10)


class TestReversibleQ:
    """Test creation and properties of reversible rate matrices."""

    def test_detailed_balance(self):
        """Test detailed balance condition."""
        pi = np.array([0.3, 0.2, 0.4, 0.1])
        rates = np.array(
            [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]]
        )
        Q = create_reversible_Q(rates, pi)

        assert check_detailed_balance(Q, pi)

    def test_row_sums_zero(self):
        """Test that row sums are zero (rate matrix property)."""
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi)

        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-15)

    def test_normalization(self):
        """Test that normalized Q has expected rate of 1."""
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi, normalize=True)

        # Expected rate = -sum(π_i * Q[i,i])
        expected_rate = -np.dot(pi, np.diag(Q))
        np.testing.assert_allclose(expected_rate, 1.0, rtol=1e-10)

    def test_jc69_construction(self):
        """Test constructing JC69 model."""
        pi = np.ones(4) / 4
        rates = np.ones((4, 4))
        Q = create_reversible_Q(rates, pi, normalize=True)

        # JC69: all off-diagonal entries should be equal within each row
        # and diagonal should be -3 * off_diag
        for i in range(4):
            off_diags = [Q[i, j] for j in range(4) if i != j]
            # All off-diagonal elements should be equal
            np.testing.assert_allclose(off_diags, off_diags[0], rtol=1e-10)
            # Diagonal should be -3 times off-diagonal
            np.testing.assert_allclose(Q[i, i], -3 * off_diags[0], rtol=1e-10)

        # All off-diagonal rates should be identical across the whole matrix
        all_off_diags = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    all_off_diags.append(Q[i, j])
        np.testing.assert_allclose(all_off_diags, all_off_diags[0], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
