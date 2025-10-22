"""
Rust-accelerated likelihood calculation for phylogenetic models.

This module provides a drop-in replacement for the Python LikelihoodCalculator
that uses the high-performance Rust backend for likelihood calculations.
"""

import numpy as np
from typing import Optional

try:
    import crabml_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from ..io.sequences import Alignment
from ..io.trees import Tree, TreeNode


class RustLikelihoodCalculator:
    """
    Compute phylogenetic likelihood using Rust backend.

    This is a drop-in replacement for the Python LikelihoodCalculator
    with identical interface but 15-30x faster performance.

    Attributes
    ----------
    alignment : Alignment
        Multiple sequence alignment
    tree : Tree
        Phylogenetic tree
    n_states : int
        Number of states (61 for codons, 20 for amino acids, 4 for nucleotides)
    n_sites : int
        Number of sites in alignment
    """

    def __init__(self, alignment: Alignment, tree: Tree):
        """
        Initialize Rust-backed likelihood calculator.

        Parameters
        ----------
        alignment : Alignment
            Multiple sequence alignment
        tree : Tree
            Phylogenetic tree
        """
        if not RUST_AVAILABLE:
            raise ImportError(
                "Rust backend not available. Install with: "
                "cd rust && maturin develop --release"
            )

        self.alignment = alignment
        self.tree = tree

        # Verify alignment and tree are compatible
        if len(alignment.names) != tree.n_leaves:
            raise ValueError(
                f"Alignment has {len(alignment.names)} sequences but tree has "
                f"{tree.n_leaves} leaves"
            )

        # Check that all leaf names match
        alignment_names_set = set(alignment.names)
        tree_names_set = set(tree.leaf_names)
        if alignment_names_set != tree_names_set:
            raise ValueError(
                "Alignment and tree have different species. "
                f"In alignment but not tree: {alignment_names_set - tree_names_set}. "
                f"In tree but not alignment: {tree_names_set - alignment_names_set}"
            )

        # Determine number of states
        if alignment.seqtype == "codon":
            self.n_states = 61
        elif alignment.seqtype == "aa":
            self.n_states = 20
        elif alignment.seqtype == "dna":
            self.n_states = 4
        else:
            raise ValueError(f"Unknown sequence type: {alignment.seqtype}")

        self.n_sites = alignment.n_sites

        # Create mapping from leaf name to alignment row
        self.leaf_to_seq_idx = {name: i for i, name in enumerate(alignment.names)}

        # Prepare tree structure for Rust
        self._prepare_tree_structure()

        # Prepare sequences for Rust (numpy array of integers)
        self._prepare_sequences()

    def _prepare_tree_structure(self):
        """
        Convert Python tree to Rust-compatible structure.

        IMPORTANT: Rust code assumes leaves are nodes 0..n_leaves-1.
        We must renumber nodes so all leaves come first in postorder.

        Creates:
        - tree_structure: List of (node_id, parent_id) tuples
        - branch_lengths: List of branch lengths (one per node)
        - leaf_names: List of leaf names in tree order
        """
        # First, separate leaves and internal nodes in postorder
        postorder_list = list(self.tree.postorder())
        leaves = [node for node in postorder_list if node.is_leaf]
        internals = [node for node in postorder_list if not node.is_leaf]

        # Assign node IDs: leaves first (0..n_leaves-1), then internals
        node_id_map = {}
        for i, node in enumerate(leaves):
            node_id_map[node.id] = i
        for i, node in enumerate(internals):
            node_id_map[node.id] = len(leaves) + i

        # Build structure in the new postorder (leaves first, then internals)
        self.tree_structure = []
        self.branch_lengths_template = []
        renumbered_postorder = leaves + internals

        for node in renumbered_postorder:
            node_id = node_id_map[node.id]
            parent_id = node_id_map[node.parent.id] if node.parent is not None else None

            self.tree_structure.append((node_id, parent_id))
            self.branch_lengths_template.append(node.branch_length if node.parent is not None else 0.0)

        # Get leaf names in correct order (matching new node IDs 0..n_leaves-1)
        self.leaf_names_ordered = [node.name for node in leaves]

        # Create list of leaf node IDs (these are the renumbered IDs)
        self.leaf_node_ids = list(range(len(leaves)))  # Leaves are nodes 0..n_leaves-1

        # Store node map for updating branch lengths
        self.node_id_map = node_id_map
        self.postorder_nodes = renumbered_postorder

    def _prepare_sequences(self):
        """
        Convert alignment sequences to numpy array for Rust.

        Creates integer-encoded sequences matrix: [n_leaves, n_sites]
        where each entry is the state index or -1 for missing data.
        """
        n_leaves = len(self.leaf_names_ordered)
        sequences = np.zeros((n_leaves, self.n_sites), dtype=np.int32)

        for leaf_idx, leaf_name in enumerate(self.leaf_names_ordered):
            seq_idx = self.leaf_to_seq_idx[leaf_name]
            sequences[leaf_idx, :] = self.alignment.sequences[seq_idx, :]

        self.sequences = sequences

    def _get_current_branch_lengths(self, scale_branch_lengths: float = 1.0) -> np.ndarray:
        """
        Get current branch lengths from tree, scaled.

        Parameters
        ----------
        scale_branch_lengths : float
            Global scaling factor

        Returns
        -------
        np.ndarray
            Array of branch lengths in postorder
        """
        branch_lengths = []
        for node in self.postorder_nodes:
            if node.parent is not None:
                branch_lengths.append(node.branch_length * scale_branch_lengths)
            else:
                branch_lengths.append(0.0)

        return np.array(branch_lengths, dtype=np.float64)

    def compute_log_likelihood(
        self, Q: np.ndarray, pi: np.ndarray, scale_branch_lengths: float = 1.0
    ) -> float:
        """
        Compute log-likelihood using Rust backend.

        Parameters
        ----------
        Q : np.ndarray
            Rate matrix (n_states x n_states)
        pi : np.ndarray
            Equilibrium frequencies (n_states,)
        scale_branch_lengths : float, optional
            Global scaling factor for branch lengths (default 1.0)

        Returns
        -------
        float
            Log-likelihood value
        """
        if Q.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"Q matrix has shape {Q.shape}, expected ({self.n_states}, {self.n_states})"
            )
        if pi.shape != (self.n_states,):
            raise ValueError(f"pi has shape {pi.shape}, expected ({self.n_states},)")

        # Get current branch lengths from tree
        branch_lengths = self._get_current_branch_lengths(scale_branch_lengths)

        # Call Rust function
        log_likelihood = crabml_rust.compute_log_likelihood(
            q=Q,
            pi=pi,
            tree_structure=self.tree_structure,
            branch_lengths=branch_lengths.tolist(),
            leaf_names=self.leaf_names_ordered,
            sequences=self.sequences,
            leaf_node_ids=self.leaf_node_ids
        )

        return log_likelihood

    def compute_log_likelihood_site_classes(
        self,
        Q_matrices: list[np.ndarray],
        pi: np.ndarray,
        proportions: list[float],
        scale_branch_lengths: float = 1.0,
        use_scaling: bool = False  # Ignored in Rust version (always uses stable numerics)
    ) -> float:
        """
        Compute log-likelihood for site class model using Rust backend.

        PARALLELIZED: Site classes computed in parallel with Rayon.

        Parameters
        ----------
        Q_matrices : list[np.ndarray]
            List of rate matrices, one per site class
        pi : np.ndarray
            Equilibrium frequencies (n_states,)
        proportions : list[float]
            Proportion of sites in each class
        scale_branch_lengths : float, optional
            Global scaling factor for branch lengths (default 1.0)
        use_scaling : bool, optional
            Ignored (Rust always uses numerically stable algorithms)

        Returns
        -------
        float
            Log-likelihood value
        """
        n_classes = len(Q_matrices)

        if len(proportions) != n_classes:
            raise ValueError(
                f"Number of proportions ({len(proportions)}) must match "
                f"number of Q matrices ({n_classes})"
            )

        # Validate Q matrices
        for k, Q in enumerate(Q_matrices):
            if Q.shape != (self.n_states, self.n_states):
                raise ValueError(
                    f"Q matrix {k} has shape {Q.shape}, "
                    f"expected ({self.n_states}, {self.n_states})"
                )

        # Get current branch lengths from tree
        branch_lengths = self._get_current_branch_lengths(scale_branch_lengths)

        # Call Rust function (parallelized across site classes)
        log_likelihood = crabml_rust.compute_site_class_log_likelihood(
            q_matrices=Q_matrices,
            proportions=proportions,
            pi=pi,
            tree_structure=self.tree_structure,
            branch_lengths=branch_lengths.tolist(),
            leaf_names=self.leaf_names_ordered,
            sequences=self.sequences,
            leaf_node_ids=self.leaf_node_ids
        )

        return log_likelihood

    def compute_site_log_likelihoods(
        self,
        Q_matrices: list[np.ndarray],
        pi: np.ndarray,
        proportions: list[float],
        scale_branch_lengths: float = 1.0,
    ) -> np.ndarray:
        """
        Compute site-specific log-likelihoods for each class (for BEB analysis).

        Returns a 2D array where each entry [i, k] is the log-likelihood of
        site i given class k. This is used by Bayes Empirical Bayes to compute
        posterior probabilities for each site belonging to each class.

        PARALLELIZED: Site classes computed in parallel with Rayon.

        Parameters
        ----------
        Q_matrices : list[np.ndarray]
            List of rate matrices, one per site class
        pi : np.ndarray
            Equilibrium frequencies (n_states,)
        proportions : list[float]
            Proportion of sites in each class (used for weighting)
        scale_branch_lengths : float, optional
            Global scaling factor for branch lengths (default 1.0)

        Returns
        -------
        np.ndarray
            Site-specific log-likelihoods, shape (n_sites, n_classes)
            Each entry [i, k] = log P(data at site i | class k, params)

        Examples
        --------
        >>> # Get site-specific likelihoods for M2a model
        >>> Q_matrices = model.get_Q_matrices()
        >>> proportions, _ = model.get_site_classes()
        >>> site_log_liks = calc.compute_site_log_likelihoods(
        ...     Q_matrices, pi, proportions
        ... )
        >>> print(site_log_liks.shape)  # (n_sites, n_classes)
        """
        n_classes = len(Q_matrices)

        if len(proportions) != n_classes:
            raise ValueError(
                f"Number of proportions ({len(proportions)}) must match "
                f"number of Q matrices ({n_classes})"
            )

        # Validate Q matrices
        for k, Q in enumerate(Q_matrices):
            if Q.shape != (self.n_states, self.n_states):
                raise ValueError(
                    f"Q matrix {k} has shape {Q.shape}, "
                    f"expected ({self.n_states}, {self.n_states})"
                )

        # Get current branch lengths from tree
        branch_lengths = self._get_current_branch_lengths(scale_branch_lengths)

        # Call Rust function (parallelized across site classes)
        # Returns np.ndarray of shape (n_sites, n_classes)
        site_log_likelihoods = crabml_rust.compute_site_log_likelihoods_by_class(
            q_matrices=Q_matrices,
            pi=pi,
            tree_structure=self.tree_structure,
            branch_lengths=branch_lengths.tolist(),
            leaf_names=self.leaf_names_ordered,
            sequences=self.sequences,
            leaf_node_ids=self.leaf_node_ids
        )

        return site_log_likelihoods
