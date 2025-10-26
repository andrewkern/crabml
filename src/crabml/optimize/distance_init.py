"""
Least-squares distance initialization for branch lengths.

Implements PAML's LSDistance approach to initialize branch lengths from sequence data.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

from ..io.sequences import Alignment
from ..io.trees import Tree, TreeNode


def compute_pairwise_distances(alignment: Alignment) -> np.ndarray:
    """
    Compute pairwise Jukes-Cantor distances between sequences.

    Parameters
    ----------
    alignment : Alignment
        Sequence alignment

    Returns
    -------
    np.ndarray
        Matrix of pairwise distances (n_species x n_species)
    """
    n_seqs = alignment.n_species
    sequences = alignment.sequences  # shape: (n_seqs, n_sites)
    distances = np.zeros((n_seqs, n_seqs))

    # Precompute valid site mask (non-gap codons)
    valid_mask = (sequences >= 0) & (sequences < 61)

    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            # Use numpy vectorization for site comparisons
            both_valid = valid_mask[i] & valid_mask[j]
            valid_sites = both_valid.sum()

            if valid_sites > 0:
                # Count differences at valid sites
                differences = ((sequences[i] != sequences[j]) & both_valid).sum()
                p_dist = differences / valid_sites

                # Jukes-Cantor correction for multiple substitutions
                # d = -3/4 * log(1 - 4p/3)
                if p_dist < 0.74:  # avoid log of negative
                    jc_dist = -0.75 * np.log(1.0 - (4.0 * p_dist / 3.0))
                else:
                    # Saturated - use large distance
                    jc_dist = 5.0

                distances[i, j] = jc_dist
                distances[j, i] = jc_dist
            else:
                # No valid sites - use small distance
                distances[i, j] = 0.1
                distances[j, i] = 0.1

    return distances


def find_path_between_leaves(tree: Tree, leaf1: TreeNode, leaf2: TreeNode) -> List[TreeNode]:
    """
    Find all nodes on the path from leaf1 to leaf2.

    Returns list of nodes with branches that contribute to the path distance.

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree
    leaf1, leaf2 : TreeNode
        Two leaf nodes

    Returns
    -------
    List[TreeNode]
        Nodes whose branches are on the path (excludes LCA)
    """
    # Get path from leaf1 to root
    path1 = []
    node = leaf1
    while node.parent is not None:
        path1.append(node)
        node = node.parent
    path1.append(node)  # root

    # Get path from leaf2 to root
    path2 = []
    node = leaf2
    while node.parent is not None:
        path2.append(node)
        node = node.parent
    path2.append(node)  # root

    # Find LCA (lowest common ancestor) using node IDs
    path1_ids = set(n.id for n in path1)
    lca = None
    for node in path2:
        if node.id in path1_ids:
            lca = node
            break

    # Build path from leaf1 to LCA, then LCA to leaf2
    path_nodes = []

    # leaf1 to LCA (exclude LCA itself)
    node = leaf1
    while node.id != lca.id:
        path_nodes.append(node)
        node = node.parent

    # LCA to leaf2 (exclude LCA itself)
    path2_to_lca = []
    node = leaf2
    while node.id != lca.id:
        path2_to_lca.append(node)
        node = node.parent

    path_nodes.extend(path2_to_lca)

    return path_nodes


def compute_tree_path_distance(
    branch_lengths: np.ndarray,
    node_to_index: dict,
    tree: Tree,
    leaf1: TreeNode,
    leaf2: TreeNode
) -> float:
    """
    Compute distance between two leaves as sum of branch lengths along path.

    Parameters
    ----------
    branch_lengths : np.ndarray
        Array of branch lengths
    node_to_index : dict
        Mapping from node ID to index in branch_lengths array
    tree : Tree
        Phylogenetic tree
    leaf1, leaf2 : TreeNode
        Two leaf nodes

    Returns
    -------
    float
        Sum of branch lengths along path
    """
    if leaf1 == leaf2:
        return 0.0

    path_nodes = find_path_between_leaves(tree, leaf1, leaf2)

    total_distance = 0.0
    for node in path_nodes:
        if node.id in node_to_index:
            total_distance += branch_lengths[node_to_index[node.id]]

    return total_distance


def ls_distance_objective(
    branch_lengths: np.ndarray,
    pairwise_dists: np.ndarray,
    node_to_index: dict,
    tree: Tree,
    leaf_nodes: List[TreeNode]
) -> float:
    """
    Least-squares objective: sum of squared differences between
    observed pairwise distances and tree-implied distances.

    Parameters
    ----------
    branch_lengths : np.ndarray
        Branch length values to optimize
    pairwise_dists : np.ndarray
        Observed pairwise distances
    node_to_index : dict
        Mapping from node ID to branch index
    tree : Tree
        Phylogenetic tree
    leaf_nodes : List[TreeNode]
        List of leaf nodes

    Returns
    -------
    float
        Sum of squared errors
    """
    # Ensure non-negative
    branch_lengths = np.abs(branch_lengths)

    sse = 0.0
    n_leaves = len(leaf_nodes)

    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            obs_dist = pairwise_dists[i, j]
            tree_dist = compute_tree_path_distance(
                branch_lengths, node_to_index, tree, leaf_nodes[i], leaf_nodes[j]
            )

            diff = obs_dist - tree_dist
            sse += diff * diff

    return sse


def initialize_branch_lengths_ls(
    alignment: Alignment,
    tree: Tree,
    branch_nodes: List[TreeNode]
) -> np.ndarray:
    """
    Initialize branch lengths using least-squares distance fitting.

    This implements PAML's LSDistance approach: fit branch lengths to minimize
    the difference between observed pairwise sequence distances and tree-implied
    distances.

    Parameters
    ----------
    alignment : Alignment
        Sequence alignment
    tree : Tree
        Phylogenetic tree (topology only, branch lengths ignored)
    branch_nodes : List[TreeNode]
        Nodes with branches to optimize (excludes root)

    Returns
    -------
    np.ndarray
        Initial branch length values
    """
    # Compute pairwise sequence distances
    pairwise_dists = compute_pairwise_distances(alignment)

    # Get leaf nodes in same order as alignment
    leaf_nodes = [node for node in tree.postorder() if node.is_leaf]

    # Create mapping from node ID to index in branch_nodes
    node_to_index = {node.id: i for i, node in enumerate(branch_nodes)}

    # Initial guess: use mean pairwise distance divided by rough tree depth
    mean_dist = np.mean(pairwise_dists[pairwise_dists > 0])
    n_branches = len(branch_nodes)

    # Initial branch lengths: scale down mean distance
    init_branch_lengths = np.ones(n_branches) * (mean_dist / 10.0)

    # Optimize using L-BFGS-B
    result = minimize(
        ls_distance_objective,
        init_branch_lengths,
        args=(pairwise_dists, node_to_index, tree, leaf_nodes),
        method='L-BFGS-B',
        bounds=[(0.0001, 5.0)] * n_branches,
        options={'maxiter': 100, 'ftol': 1e-4}
    )

    # Return optimized branch lengths (ensure non-negative)
    return np.abs(result.x)
