"""
Likelihood calculation for phylogenetic models.

This module implements the Felsenstein pruning algorithm for computing
phylogenetic likelihood on a tree.
"""

import numpy as np
from typing import Optional

from ..io.sequences import Alignment
from ..io.trees import Tree, TreeNode
from .matrix import matrix_exponential


class LikelihoodCalculator:
    """
    Compute phylogenetic likelihood using Felsenstein's pruning algorithm.

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
        Initialize likelihood calculator.

        Parameters
        ----------
        alignment : Alignment
            Multiple sequence alignment
        tree : Tree
            Phylogenetic tree
        """
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

    def compute_log_likelihood(
        self, Q: np.ndarray, pi: np.ndarray, scale_branch_lengths: float = 1.0
    ) -> float:
        """
        Compute log-likelihood for given substitution model.

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

        # Compute transition probability matrices for each branch
        # Store in a dictionary keyed by node id
        P_matrices = {}
        for node in self.tree.postorder():
            if node.parent is not None:
                branch_length = node.branch_length * scale_branch_lengths
                P_matrices[node.id] = matrix_exponential(Q, branch_length)

        # Initialize conditional likelihood arrays
        # L[node_id][site, state] = P(data below node | state at node)
        L = {}

        # Post-order traversal (leaves to root)
        for node in self.tree.postorder():
            L[node.id] = np.zeros((self.n_sites, self.n_states))

            if node.is_leaf:
                # For leaves, set likelihood based on observed data
                seq_idx = self.leaf_to_seq_idx[node.name]
                for site in range(self.n_sites):
                    obs_state = self.alignment.sequences[seq_idx, site]
                    if obs_state >= 0:  # Valid codon
                        # Likelihood is 1 if state matches observation, 0 otherwise
                        L[node.id][site, obs_state] = 1.0
                    else:
                        # Missing data or ambiguous - all states equally likely
                        L[node.id][site, :] = 1.0
            else:
                # For internal nodes, compute from children
                for site in range(self.n_sites):
                    for state in range(self.n_states):
                        # Product over all children
                        likelihood = 1.0
                        for child in node.children:
                            # Sum over all possible child states
                            P = P_matrices[child.id]
                            child_sum = np.sum(P[state, :] * L[child.id][site, :])
                            likelihood *= child_sum
                        L[node.id][site, state] = likelihood

        # At root, compute total likelihood for each site
        root_L = L[self.tree.root.id]
        site_likelihoods = np.sum(root_L * pi[np.newaxis, :], axis=1)

        # Return log-likelihood (sum over sites)
        # Add small constant to avoid log(0)
        log_likelihood = np.sum(np.log(site_likelihoods + 1e-100))

        return log_likelihood

    def compute_log_likelihood_site_classes(
        self,
        Q_matrices: list[np.ndarray],
        pi: np.ndarray,
        proportions: list[float],
        scale_branch_lengths: float = 1.0,
        use_scaling: bool = False
    ) -> float:
        """
        Compute log-likelihood for site class model.

        For models with site classes (M1a, M2a, M3), the likelihood is a
        mixture over site classes:
        P(data) = sum_k p_k * P(data | Q_k)

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
            Use PAML-style scaling to prevent underflow (default False)

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

        # Compute likelihood for each site class
        class_likelihoods = np.zeros((self.n_sites, n_classes))

        for k, Q in enumerate(Q_matrices):
            if Q.shape != (self.n_states, self.n_states):
                raise ValueError(
                    f"Q matrix {k} has shape {Q.shape}, "
                    f"expected ({self.n_states}, {self.n_states})"
                )

            # Compute transition probability matrices for each branch
            P_matrices = {}
            for node in self.tree.postorder():
                if node.parent is not None:
                    branch_length = node.branch_length * scale_branch_lengths
                    P_matrices[node.id] = matrix_exponential(Q, branch_length)

            # Initialize conditional likelihood arrays and scale factors
            L = {}
            scale_factors = np.zeros((self.tree.n_nodes, self.n_sites)) if use_scaling else None

            # Post-order traversal (leaves to root)
            for node in self.tree.postorder():
                L[node.id] = np.zeros((self.n_sites, self.n_states))

                if node.is_leaf:
                    # For leaves, set likelihood based on observed data
                    seq_idx = self.leaf_to_seq_idx[node.name]
                    for site in range(self.n_sites):
                        obs_state = self.alignment.sequences[seq_idx, site]
                        if obs_state >= 0:  # Valid codon
                            L[node.id][site, obs_state] = 1.0
                        else:
                            # Missing data - all states equally likely
                            L[node.id][site, :] = 1.0
                else:
                    # For internal nodes, compute from children
                    for site in range(self.n_sites):
                        for state in range(self.n_states):
                            likelihood = 1.0
                            for child in node.children:
                                P = P_matrices[child.id]
                                child_sum = np.sum(P[state, :] * L[child.id][site, :])
                                likelihood *= child_sum
                            L[node.id][site, state] = likelihood

                # Apply PAML-style scaling if enabled
                if use_scaling and not node.is_leaf:
                    for site in range(self.n_sites):
                        max_val = np.max(L[node.id][site, :])
                        if max_val < 1e-300:
                            L[node.id][site, :] = 1.0
                            scale_factors[node.id, site] = -800.0
                        else:
                            L[node.id][site, :] /= max_val
                            scale_factors[node.id, site] = np.log(max_val)

            # At root, compute total likelihood for each site
            root_L = L[self.tree.root.id]
            site_likelihoods_k = np.sum(root_L * pi[np.newaxis, :], axis=1)

            # Add back scale factors if using scaling
            if use_scaling:
                log_site_likelihoods_k = np.log(site_likelihoods_k + 1e-100)
                # Sum scale factors across all scaled nodes
                for node in self.tree.postorder():
                    if not node.is_leaf:
                        log_site_likelihoods_k += scale_factors[node.id, :]
                class_likelihoods[:, k] = np.exp(log_site_likelihoods_k)
            else:
                class_likelihoods[:, k] = site_likelihoods_k

        # Mix over site classes using log-sum-exp trick (like PAML)
        # Convert to log-space
        log_class_likelihoods = np.log(class_likelihoods + 1e-100)
        log_proportions = np.log(np.array(proportions))

        # Add log proportions: log(p_k * L_k) = log(p_k) + log(L_k)
        log_weighted = log_class_likelihoods + log_proportions[np.newaxis, :]

        # Use log-sum-exp trick to compute log(sum(p_k * L_k))
        # log(sum(exp(x_i))) = max(x_i) + log(sum(exp(x_i - max(x_i))))
        max_log = np.max(log_weighted, axis=1, keepdims=True)
        log_site_likelihoods = max_log.squeeze() + np.log(
            np.sum(np.exp(log_weighted - max_log), axis=1)
        )

        # Return total log-likelihood (sum over sites)
        log_likelihood = np.sum(log_site_likelihoods)

        return log_likelihood
