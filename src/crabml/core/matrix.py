"""
Matrix operations for phylogenetic likelihood calculations.

This module provides core matrix operations needed for computing transition
probabilities and likelihood calculations.
"""

import numpy as np
from scipy.linalg import expm


def matrix_exponential(Q: np.ndarray, t: float) -> np.ndarray:
    """
    Compute transition probability matrix P(t) = exp(Q*t).

    Uses scipy's highly optimized matrix exponential implementation
    based on Padé approximation with scaling and squaring.

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        Rate matrix (instantaneous substitution rate matrix)
    t : float
        Branch length (time)

    Returns
    -------
    P : ndarray, shape (n, n)
        Transition probability matrix where P[i,j] is the probability
        of state i transitioning to state j over time t

    Notes
    -----
    The transition probability matrix satisfies:
    - Row sums equal 1 (stochastic matrix)
    - All entries are non-negative
    - P(0) = I (identity matrix)
    - P(t1 + t2) = P(t1) @ P(t2) (semigroup property)

    Examples
    --------
    >>> # JC69 model (equal rates)
    >>> alpha = 0.25
    >>> Q = np.array([[-3*alpha, alpha, alpha, alpha],
    ...               [alpha, -3*alpha, alpha, alpha],
    ...               [alpha, alpha, -3*alpha, alpha],
    ...               [alpha, alpha, alpha, -3*alpha]])
    >>> P = matrix_exponential(Q, 0.1)
    >>> np.sum(P[0])  # Row sum should be 1
    1.0
    """
    return expm(Q * t)


def eigen_decompose_rev(Q: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eigendecompose reversible rate matrix Q = U @ diag(eigenvalues) @ V.

    Uses symmetrization trick for reversible (time-reversible) rate matrices:
    Transform Q to symmetric matrix Q' = √D @ Q @ √D^(-1), where D = diag(pi),
    then eigendecompose Q' and transform back.

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        Reversible rate matrix satisfying detailed balance: π_i * Q[i,j] = π_j * Q[j,i]
    pi : ndarray, shape (n,)
        Stationary distribution (equilibrium frequencies)

    Returns
    -------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues of Q, sorted in ascending order
    U : ndarray, shape (n, n)
        Left eigenvector matrix
    V : ndarray, shape (n, n)
        Right eigenvector matrix

    Notes
    -----
    The decomposition satisfies:
    - Q = U @ diag(eigenvalues) @ V
    - P(t) = U @ diag(exp(eigenvalues * t)) @ V
    - Largest eigenvalue is 0 (stationary distribution)
    - pi = U[:, -1] * V[-1, :] (up to normalization)

    The symmetrization exploits the reversibility property for numerical stability
    and efficiency.

    Examples
    --------
    >>> pi = np.array([0.25, 0.25, 0.25, 0.25])
    >>> Q = create_reversible_Q(pi)  # Must satisfy detailed balance
    >>> eigenvalues, U, V = eigen_decompose_rev(Q, pi)
    >>> Q_reconstructed = U @ np.diag(eigenvalues) @ V
    >>> np.allclose(Q, Q_reconstructed)
    True
    """
    n = len(pi)
    sqrt_pi = np.sqrt(pi)

    # Symmetrize: Q' = √D @ Q @ √D^(-1)
    # This is symmetric if Q satisfies detailed balance
    Q_sym = Q * sqrt_pi[:, np.newaxis] / sqrt_pi[np.newaxis, :]

    # Eigendecompose symmetric matrix (numerically stable)
    # Returns eigenvalues in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(Q_sym)

    # Transform back to original basis
    # U = eigenvectors / √π (row-wise)
    # V = eigenvectors^T * √π (column-wise)
    U = eigenvectors / sqrt_pi[:, np.newaxis]
    V = eigenvectors.T * sqrt_pi[np.newaxis, :]

    return eigenvalues, U, V


def create_reversible_Q(
    rates: np.ndarray, pi: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """
    Create a reversible rate matrix from exchangeability rates and stationary distribution.

    Parameters
    ----------
    rates : ndarray, shape (n, n)
        Symmetric exchangeability matrix (r[i,j] = r[j,i])
    pi : ndarray, shape (n,)
        Stationary distribution (equilibrium frequencies)
    normalize : bool, default=True
        If True, scale Q so that expected rate is 1 substitution per time unit

    Returns
    -------
    Q : ndarray, shape (n, n)
        Reversible rate matrix satisfying detailed balance

    Notes
    -----
    The rate matrix is constructed as Q[i,j] = r[i,j] * pi[j] for i ≠ j,
    and Q[i,i] = -sum(Q[i,j] for j ≠ i).

    This satisfies detailed balance: pi[i] * Q[i,j] = pi[j] * Q[j,i]

    Examples
    --------
    >>> # JC69 model
    >>> rates = np.ones((4, 4)) - np.eye(4)  # All rates equal
    >>> pi = np.ones(4) / 4
    >>> Q = create_reversible_Q(rates, pi)
    """
    # Vectorized: Q[i,j] = r[i,j] * π_j (broadcast pi across columns)
    Q = rates * pi[np.newaxis, :]

    # Diagonal: Q[i,i] = -sum(Q[i,j] for j ≠ i)
    # Zero out diagonal first (in case rates has non-zero diagonal)
    np.fill_diagonal(Q, 0.0)
    # Compute row sums and negate for diagonal
    row_sums = np.sum(Q, axis=1)
    np.fill_diagonal(Q, -row_sums)

    # Normalize if requested
    if normalize:
        # Expected rate = -sum(π_i * Q[i,i])
        # Use diagonal() to get view instead of np.diag() which creates a copy
        expected_rate = -np.dot(pi, Q.diagonal())
        Q /= expected_rate  # In-place division

    return Q


def check_detailed_balance(Q: np.ndarray, pi: np.ndarray, rtol: float = 1e-10) -> bool:
    """
    Test if rate matrix Q satisfies detailed balance with stationary distribution pi.

    Parameters
    ----------
    Q : ndarray, shape (n, n)
        Rate matrix
    pi : ndarray, shape (n,)
        Proposed stationary distribution
    rtol : float
        Relative tolerance for comparison

    Returns
    -------
    bool
        True if detailed balance is satisfied

    Notes
    -----
    Detailed balance: π_i * Q[i,j] = π_j * Q[j,i] for all i, j
    """
    n = len(pi)
    for i in range(n):
        for j in range(i + 1, n):
            forward = pi[i] * Q[i, j]
            reverse = pi[j] * Q[j, i]
            if not np.isclose(forward, reverse, rtol=rtol):
                return False
    return True
