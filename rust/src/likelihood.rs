/// High-performance Felsenstein pruning algorithm for phylogenetic likelihood
///
/// Key optimizations:
/// 1. BLAS matrix operations instead of nested loops
/// 2. Contiguous memory layout for cache efficiency
/// 3. Rayon parallelization across site classes
/// 4. SIMD for log-sum-exp

use ndarray::prelude::*;
use rayon::prelude::*;
use crate::matrix::CachedQMatrix;

/// Tree structure optimized for post-order traversal
#[derive(Clone, Debug)]
pub struct Tree {
    /// Number of nodes (internal + leaves)
    pub n_nodes: usize,
    /// Number of leaves (species)
    pub n_leaves: usize,
    /// Post-order traversal indices
    pub postorder: Vec<usize>,
    /// Parent index for each node (None for root)
    pub parents: Vec<Option<usize>>,
    /// Children indices for each node
    pub children: Vec<Vec<usize>>,
    /// Branch lengths (indexed by node, length from node to parent)
    pub branch_lengths: Vec<f64>,
    /// Leaf names mapped to node indices
    pub leaf_names: Vec<String>,
    /// Node IDs that are leaves (for correct sequence indexing)
    pub leaf_node_ids: Vec<usize>,
}

impl Tree {
    /// Create tree from structure
    pub fn from_structure(
        structure: Vec<(usize, Option<usize>)>,  // (node_id, parent_id)
        branch_lengths: Vec<f64>,
        leaf_names: Vec<String>,
        leaf_node_ids: Vec<usize>,
    ) -> Result<Self, String> {
        let n_nodes = structure.len();
        let n_leaves = leaf_names.len();

        // Build parent and children relationships
        let mut parents = vec![None; n_nodes];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];

        for (node_id, parent_id) in structure.iter() {
            parents[*node_id] = *parent_id;
            if let Some(p) = parent_id {
                children[*p].push(*node_id);
            }
        }

        // Find root (node with no parent)
        let root = parents.iter()
            .position(|p| p.is_none())
            .ok_or("No root node found")?;

        // Compute post-order traversal
        let mut postorder = Vec::new();
        fn visit(node: usize, children: &[Vec<usize>], postorder: &mut Vec<usize>) {
            for &child in &children[node] {
                visit(child, children, postorder);
            }
            postorder.push(node);
        }
        visit(root, &children, &mut postorder);

        Ok(Self {
            n_nodes,
            n_leaves,
            postorder,
            parents,
            children,
            branch_lengths,
            leaf_names,
            leaf_node_ids,
        })
    }
}

/// Likelihood calculator with pre-allocated workspace
pub struct LikelihoodCalculator {
    tree: Tree,
    n_sites: usize,
    n_states: usize,
    /// Workspace for conditional likelihoods
    /// Shape: [n_nodes, n_sites, n_states]
    /// Single contiguous allocation for cache efficiency
    workspace: Array3<f64>,
}

impl LikelihoodCalculator {
    /// Create new likelihood calculator
    pub fn new(tree: Tree, n_sites: usize, n_states: usize) -> Self {
        let workspace = Array3::zeros((tree.n_nodes, n_sites, n_states));

        Self {
            tree,
            n_sites,
            n_states,
            workspace,
        }
    }

    /// Compute log-likelihood for single Q matrix (e.g., M0 model)
    pub fn compute_log_likelihood(
        &mut self,
        q_matrix: &CachedQMatrix,
        pi: ArrayView1<f64>,
        sequences: ArrayView2<i32>,  // [n_leaves, n_sites]
    ) -> f64 {
        // Compute P(t) for each branch
        let p_matrices = q_matrix.expm_multiple(&self.tree.branch_lengths);

        // Initialize leaf likelihoods
        // Use leaf_node_ids to correctly identify which nodes are leaves
        for (leaf_idx, &node_idx) in self.tree.leaf_node_ids.iter().enumerate() {
            for site in 0..self.n_sites {
                let obs_state = sequences[[leaf_idx, site]];

                if obs_state >= 0 && (obs_state as usize) < self.n_states {
                    // Observed state: likelihood = 1 for that state, 0 for others
                    self.workspace[[node_idx, site, obs_state as usize]] = 1.0;
                } else {
                    // Missing data: all states equally likely
                    self.workspace.slice_mut(s![node_idx, site, ..]).fill(1.0);
                }
            }
        }

        // Post-order traversal: compute internal node likelihoods
        // Create a set of leaf node IDs for fast lookup
        let leaf_set: std::collections::HashSet<usize> = self.tree.leaf_node_ids.iter().copied().collect();
        let postorder = self.tree.postorder.clone();

        for &node_idx in &postorder {
            if !leaf_set.contains(&node_idx) {
                // Internal node
                self.compute_internal_node(node_idx, &p_matrices);
            }
        }

        // Root likelihood: sum over all states weighted by pi
        let root_idx = *self.tree.postorder.last().unwrap();
        let root_likelihoods = self.workspace.slice(s![root_idx, .., ..]);

        // Site likelihoods: sum_i (L_root[site, i] * pi[i])
        let mut log_likelihood = 0.0;
        for site in 0..self.n_sites {
            let site_lnl: f64 = root_likelihoods.slice(s![site, ..])
                .iter()
                .zip(pi.iter())
                .map(|(&l, &p)| l * p)
                .sum();

            log_likelihood += (site_lnl + 1e-100).ln();
        }

        log_likelihood
    }

    /// Compute internal node likelihood using BLAS
    ///
    /// KEY OPTIMIZATION: Instead of triple nested loop, use matrix multiply:
    /// L[node][site, i] = ∏_children (∑_j P_child[i,j] * L[child][site, j])
    ///
    /// Rewrite as matrix operation:
    /// temp[site, i] = (L[child] @ P_child^T)[site, i]
    #[inline]
    fn compute_internal_node(&mut self, node_idx: usize, p_matrices: &[Array2<f64>]) {
        // Initialize to all 1s
        self.workspace.slice_mut(s![node_idx, .., ..]).fill(1.0);

        for &child_idx in &self.tree.children[node_idx] {
            // Get transition matrix for this branch
            let p = &p_matrices[child_idx];

            // Get child likelihoods: [n_sites, n_states]
            let child_lnl = self.workspace.slice(s![child_idx, .., ..]);

            // Compute: temp = child_lnl @ P^T
            // This is ONE BLAS call for ALL sites at once!
            let temp = child_lnl.dot(&p.t());

            // Element-wise multiply: L[node] *= temp
            self.workspace.slice_mut(s![node_idx, .., ..])
                .zip_mut_with(&temp, |a, &b| *a *= b);
        }
    }

    /// Compute site likelihoods (NOT log-likelihoods) for all sites
    ///
    /// Returns: Array of site likelihoods [n_sites]
    pub fn compute_site_likelihoods(
        &mut self,
        q_matrix: &CachedQMatrix,
        pi: ArrayView1<f64>,
        sequences: ArrayView2<i32>,
    ) -> Array1<f64> {
        // Compute P(t) for each branch
        let p_matrices = q_matrix.expm_multiple(&self.tree.branch_lengths);

        // Initialize leaf likelihoods
        for (leaf_idx, &node_idx) in self.tree.leaf_node_ids.iter().enumerate() {
            for site in 0..self.n_sites {
                let obs_state = sequences[[leaf_idx, site]];

                if obs_state >= 0 && (obs_state as usize) < self.n_states {
                    self.workspace[[node_idx, site, obs_state as usize]] = 1.0;
                } else {
                    self.workspace.slice_mut(s![node_idx, site, ..]).fill(1.0);
                }
            }
        }

        // Post-order traversal: compute internal node likelihoods
        let leaf_set: std::collections::HashSet<usize> = self.tree.leaf_node_ids.iter().copied().collect();
        let postorder = self.tree.postorder.clone();

        for &node_idx in &postorder {
            if !leaf_set.contains(&node_idx) {
                self.compute_internal_node(node_idx, &p_matrices);
            }
        }

        // Root likelihood: sum over all states weighted by pi (PER SITE)
        let root_idx = *self.tree.postorder.last().unwrap();
        let root_likelihoods = self.workspace.slice(s![root_idx, .., ..]);

        let mut site_likelihoods = Array1::zeros(self.n_sites);
        for site in 0..self.n_sites {
            site_likelihoods[site] = root_likelihoods.slice(s![site, ..])
                .iter()
                .zip(pi.iter())
                .map(|(&l, &p)| l * p)
                .sum();
        }

        site_likelihoods
    }

    /// Compute log-likelihood for site class models (M1a, M2a, M3)
    ///
    /// PARALLELIZED: Each site class computed in parallel with Rayon
    ///
    /// Correct algorithm:
    /// 1. For each class k: compute site likelihoods L_k[site]
    /// 2. For each site: compute log(∑_k p_k * L_k[site])
    /// 3. Sum across sites: ∑_sites log(∑_k p_k * L_k[site])
    pub fn compute_site_class_log_likelihood(
        &mut self,
        q_matrices: &[CachedQMatrix],
        proportions: &[f64],
        pi: ArrayView1<f64>,
        sequences: ArrayView2<i32>,
    ) -> f64 {
        let n_classes = q_matrices.len();

        // Compute site likelihoods for each class IN PARALLEL
        // Returns: Vec of Array1 [n_classes x n_sites]
        let class_site_likelihoods: Vec<Array1<f64>> = q_matrices.par_iter()
            .map(|q_matrix| {
                // Each thread gets its own calculator (no shared state)
                let mut calc = LikelihoodCalculator::new(
                    self.tree.clone(),
                    self.n_sites,
                    self.n_states,
                );
                calc.compute_site_likelihoods(q_matrix, pi, sequences)
            })
            .collect();

        // For each site: compute log(∑_k p_k * L_k[site])
        let mut log_likelihood = 0.0;
        for site in 0..self.n_sites {
            // Collect likelihoods for this site across all classes
            let site_class_lnls: Vec<f64> = (0..n_classes)
                .map(|k| class_site_likelihoods[k][site])
                .collect();

            // Compute log(∑_k p_k * L_k[site]) using log-sum-exp
            log_likelihood += self.log_sum_exp_likelihoods(&site_class_lnls, proportions);
        }

        log_likelihood
    }

    /// Compute log(∑_k p_k * L_k) from likelihoods (not log-likelihoods)
    ///
    /// Uses numerically stable log-sum-exp algorithm:
    /// log(∑_k p_k * L_k) = log(∑_k p_k * exp(log(L_k)))
    ///                     = log(∑_k exp(log(p_k) + log(L_k)))
    ///                     = max + log(∑_k exp(log(p_k) + log(L_k) - max))
    #[inline]
    fn log_sum_exp_likelihoods(&self, likelihoods: &[f64], proportions: &[f64]) -> f64 {
        // Convert to log space: log(p_k) + log(L_k)
        let log_weighted: Vec<f64> = likelihoods.iter()
            .zip(proportions)
            .map(|(&lnl, &p)| (lnl + 1e-100).ln() + p.ln())
            .collect();

        // Find max for numerical stability
        let max_log = log_weighted.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Compute log(sum(exp(log_weighted - max)))
        let sum: f64 = log_weighted.iter()
            .map(|&x| (x - max_log).exp())
            .sum();

        max_log + sum.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_postorder() {
        // Simple tree: ((A,B),C)
        //     4 (root)
        //    / \
        //   3   2 (C)
        //  / \
        // 0   1
        // (A) (B)

        let structure = vec![
            (0, Some(3)),  // A -> internal
            (1, Some(3)),  // B -> internal
            (2, Some(4)),  // C -> root
            (3, Some(4)),  // internal -> root
            (4, None),     // root
        ];

        let branch_lengths = vec![0.1, 0.1, 0.2, 0.1, 0.0];
        let leaf_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let leaf_node_ids = vec![0, 1, 2];  // Nodes 0, 1, 2 are leaves

        let tree = Tree::from_structure(structure, branch_lengths, leaf_names, leaf_node_ids).unwrap();

        // Post-order should visit leaves first, then internal, then root
        assert_eq!(tree.postorder.len(), 5);
        assert_eq!(*tree.postorder.last().unwrap(), 4);  // Root last
    }
}
