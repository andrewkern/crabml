"""
Branch codon models for detecting positive selection on specific lineages.

Implements:
- Free-ratio model (model=1): Independent omega for each branch
- Two-ratio model (model=2): Different omega for labeled branch groups

References:
    Yang (1998). Mol. Biol. Evol. 15:568-573
"""

from typing import Optional, Tuple, Dict
import numpy as np
from .codon import build_codon_Q_matrix


class CodonBranchModel:
    """
    Branch model allowing omega (dN/dS) to vary among branches.

    Unlike site models where omega varies across sites, branch models
    allow omega to vary across lineages while keeping it constant across
    sites. This enables detection of positive selection on specific branches.

    Two modes:
    1. Free-ratio (model=1): Each branch gets independent omega
    2. Multi-ratio (model=2): Branches with same label share omega

    Parameters
    ----------
    codon_frequencies : np.ndarray, shape (61,)
        Codon equilibrium frequencies
    branch_labels : np.ndarray, shape (n_branches,)
        Integer labels for each branch. In free-ratio mode, labels are
        ignored (each branch gets unique omega). In multi-ratio mode,
        branches with same label share omega.
    free_ratio : bool, default=False
        If True, use free-ratio model (independent omega per branch).
        If False, use multi-ratio model (omega per label).
    n_branches : int
        Total number of branches in the tree

    Examples
    --------
    Multi-ratio model (2 omega ratios):
        >>> labels = np.array([0, 0, 1, 0, 1, 0])  # 6 branches, 2 groups
        >>> model = CodonBranchModel(pi, labels, free_ratio=False, n_branches=6)
        >>> # This estimates omega0 (labels 0) and omega1 (labels 1)

    Free-ratio model:
        >>> model = CodonBranchModel(pi, labels, free_ratio=True, n_branches=6)
        >>> # This estimates 6 independent omega values (ignores labels)
    """

    def __init__(
        self,
        codon_frequencies: np.ndarray,
        branch_labels: np.ndarray,
        free_ratio: bool = False,
        n_nodes: int = None,  # Total nodes including root
    ):
        """Initialize branch model."""
        self.pi = np.array(codon_frequencies)
        self.branch_labels = np.array(branch_labels, dtype=int)
        self.free_ratio = free_ratio

        # n_nodes is total number of nodes (including root)
        # branch_labels has one entry per node (even root has label for consistency)
        if n_nodes is None:
            n_nodes = len(branch_labels)
        self.n_nodes = n_nodes

        if len(self.branch_labels) != n_nodes:
            raise ValueError(
                f"branch_labels length ({len(self.branch_labels)}) must match "
                f"n_nodes ({n_nodes})"
            )

        # Determine number of omega parameters
        # Note: root's label doesn't matter for parameters since it has no parent branch
        if free_ratio:
            # Free-ratio: one omega per branch (excluding root)
            self.n_omega = n_nodes - 1  # Root has no parent branch
            # Create omega labels: each node gets its own omega index, except root gets 0 (arbitrary)
            # Root's label doesn't matter since root has no parent branch
            self.omega_labels = np.arange(n_nodes)
            self.omega_labels[-1] = 0  # Assign root (last node) to omega 0 (unused but valid index)
            print(f"Free-ratio model initialized: {self.n_omega} independent ω parameters")
        else:
            # Multi-ratio: one omega per unique label
            unique_labels = np.unique(self.branch_labels)
            self.n_omega = len(unique_labels)

            # Map branch labels to omega indices
            self.omega_labels = np.zeros(n_nodes, dtype=int)
            for i, label in enumerate(unique_labels):
                self.omega_labels[self.branch_labels == label] = i

            print(f"Multi-ratio model initialized:")
            print(f"  - {n_nodes} nodes")
            print(f"  - {self.n_omega} ω parameters")
            for i, label in enumerate(unique_labels):
                n_branches_in_class = np.sum(self.branch_labels == label)
                print(f"  - ω{i} (label #{label}): {n_branches_in_class} nodes")

    def get_parameters(self) -> dict:
        """
        Get parameter names and bounds.

        Returns
        -------
        dict
            Parameter names mapped to (lower_bound, upper_bound) tuples
        """
        params = {'kappa': (0.1, 20.0)}

        # Add omega parameters
        for i in range(self.n_omega):
            params[f'omega{i}'] = (1e-6, 20.0)

        return params

    def get_param_names(self) -> list[str]:
        """Get list of parameter names in order."""
        names = ['kappa']
        names.extend([f'omega{i}' for i in range(self.n_omega)])
        return names

    def get_Q_matrices(
        self,
        kappa: float,
        omega_values: np.ndarray,
    ) -> np.ndarray:
        """
        Build Q matrix for each node.

        Parameters
        ----------
        kappa : float
            Transition/transversion ratio
        omega_values : np.ndarray, shape (n_omega,)
            Omega values (one per omega parameter)

        Returns
        -------
        np.ndarray, shape (n_nodes, 61, 61)
            Q matrix for each node
        """
        if len(omega_values) != self.n_omega:
            raise ValueError(
                f"Expected {self.n_omega} omega values, got {len(omega_values)}"
            )

        Q_matrices = np.zeros((self.n_nodes, 61, 61))

        for node_idx in range(self.n_nodes):
            omega_idx = self.omega_labels[node_idx]
            omega = omega_values[omega_idx]
            Q_matrices[node_idx] = build_codon_Q_matrix(kappa, omega, self.pi)

        return Q_matrices

    def compute_log_likelihood(
        self,
        rust_calc,  # RustLikelihoodCalculator instance
        kappa: float,
        omega_values: np.ndarray,
        branch_lengths: np.ndarray,
        use_rust: bool = True,
    ) -> float:
        """
        Compute log-likelihood for branch model.

        Parameters
        ----------
        rust_calc : RustLikelihoodCalculator
            Pre-initialized calculator with alignment and tree
        kappa : float
            Transition/transversion ratio
        omega_values : np.ndarray, shape (n_omega,)
            Omega values for each parameter
        branch_lengths : np.ndarray, shape (n_branches,)
            Branch lengths
        use_rust : bool, default=True
            Use Rust backend if available

        Returns
        -------
        float
            Log-likelihood
        """
        # Get Q matrices for all branches
        Q_matrices = self.get_Q_matrices(kappa, omega_values)

        # Compute likelihood
        if use_rust:
            try:
                from crabml.crabml_rust import compute_log_likelihood_branch

                # Build parent indices from tree structure
                parent_indices = []
                for node_id, parent_id in rust_calc.tree_structure:
                    if parent_id is None:
                        parent_indices.append(-1)  # Root has no parent
                    else:
                        parent_indices.append(parent_id)

                lnL = compute_log_likelihood_branch(
                    sequences=rust_calc.sequences,
                    parent_indices=parent_indices,
                    leaf_node_ids=rust_calc.leaf_node_ids,
                    Q_matrices=Q_matrices,
                    branch_lengths=branch_lengths,
                    pi=self.pi,
                )
                return lnL
            except ImportError:
                print("Warning: Rust backend not available, falling back to Python")
                use_rust = False

        if not use_rust:
            # Python fallback
            raise NotImplementedError(
                "Python backend for branch models not yet implemented. "
                "Please install Rust backend: uv sync --all-extras --reinstall-package crabml-rust"
            )

        return lnL
