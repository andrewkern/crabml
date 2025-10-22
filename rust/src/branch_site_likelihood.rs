/// Branch-Site Model likelihood calculation
///
/// Key difference from site-class models:
/// - Each BRANCH can have different omega values depending on branch label
/// - Need to compute P matrices per-branch, per-site-class
///
/// Architecture:
/// 1. Precompute eigendecompositions for all unique omega values (ω₀, 1, ω₂)
/// 2. During pruning, select correct eigen for (branch_label, site_class) combination
/// 3. Parallelize across site classes using Rayon
///
/// Memory layout:
/// - Branch labels: Vec<u8> [n_branches], 0=background, 1=foreground
/// - Eigen cache: HashMap<(branch_label, site_class), EigenSystem>
/// - P matrices: computed on-the-fly from eigen + branch_length + Qfactor

use ndarray::prelude::*;
use rayon::prelude::*;
use crate::matrix::CachedQMatrix;
use crate::likelihood::Tree;

/// Branch-site likelihood calculator
pub struct BranchSiteLikelihoodCalculator {
    tree: Tree,
    n_sites: usize,
    n_states: usize,
    /// Branch labels (0=background, 1=foreground)
    branch_labels: Vec<u8>,
    /// Workspace for conditional likelihoods [n_nodes, n_sites, n_states]
    workspace: Array3<f64>,
}

impl BranchSiteLikelihoodCalculator {
    /// Create new branch-site likelihood calculator
    pub fn new(
        tree: Tree,
        n_sites: usize,
        n_states: usize,
        branch_labels: Vec<u8>,
    ) -> Result<Self, String> {
        if branch_labels.len() != tree.n_nodes - 1 {
            return Err(format!(
                "Branch labels length {} must equal n_nodes - 1 = {}",
                branch_labels.len(),
                tree.n_nodes - 1
            ));
        }

        let workspace = Array3::zeros((tree.n_nodes, n_sites, n_states));

        Ok(Self {
            tree,
            n_sites,
            n_states,
            branch_labels,
            workspace,
        })
    }

    /// Get branch label for a given node (label of branch leading TO this node)
    fn get_branch_label(&self, node_idx: usize) -> u8 {
        // Find this node's position in the branch_labels array
        // Branch i connects node i to its parent
        if node_idx >= self.branch_labels.len() {
            // Root has no branch label
            0
        } else {
            self.branch_labels[node_idx]
        }
    }

    /// Select omega for (branch_label, site_class) combination
    ///
    /// Branch-Site Model A omega structure:
    /// - Background (label=0):
    ///   - Class 0: ω₀
    ///   - Class 1: 1.0
    ///   - Class 2a: ω₀
    ///   - Class 2b: 1.0
    /// - Foreground (label=1):
    ///   - Class 0: ω₀
    ///   - Class 1: 1.0
    ///   - Class 2a: ω₂
    ///   - Class 2b: ω₂
    fn select_omega(
        branch_label: u8,
        site_class: usize,
        omega0: f64,
        omega2: f64,
    ) -> f64 {
        match (branch_label, site_class) {
            (0, 0) => omega0,      // Background, class 0
            (0, 1) => 1.0,         // Background, class 1
            (0, 2) => omega0,      // Background, class 2a
            (0, 3) => 1.0,         // Background, class 2b
            (1, 0) => omega0,      // Foreground, class 0
            (1, 1) => 1.0,         // Foreground, class 1
            (1, 2) => omega2,      // Foreground, class 2a
            (1, 3) => omega2,      // Foreground, class 2b
            _ => panic!("Invalid branch_label or site_class"),
        }
    }

    /// Compute log-likelihood for branch-site Model A
    ///
    /// Parameters:
    /// - q_omega0: Cached Q matrix eigendecomposition for ω₀
    /// - q_omega1: Cached Q matrix eigendecomposition for ω = 1
    /// - q_omega2: Cached Q matrix eigendecomposition for ω₂
    /// - qfactor_back: Q normalization factor for background branches
    /// - qfactor_fore: Q normalization factor for foreground branches
    /// - site_class_freqs: Array of 4 site class frequencies [p0, p1, p2a, p2b]
    /// - pi: Codon frequencies (stationary distribution)
    /// - sequences: Observed sequences [n_species, n_sites]
    pub fn compute_log_likelihood(
        &mut self,
        q_omega0: &CachedQMatrix,
        q_omega1: &CachedQMatrix,
        q_omega2: &CachedQMatrix,
        qfactor_back: f64,
        qfactor_fore: f64,
        omega0: f64,
        omega2: f64,
        site_class_freqs: &[f64; 4],
        pi: ArrayView1<f64>,
        sequences: ArrayView2<i32>,
    ) -> f64 {
        let n_classes = 4;

        // Compute site likelihoods for each site class IN PARALLEL
        let class_site_likelihoods: Vec<Array1<f64>> = (0..n_classes)
            .into_par_iter()
            .map(|site_class| {
                // Each thread gets its own calculator
                let mut calc = Self::new(
                    self.tree.clone(),
                    self.n_sites,
                    self.n_states,
                    self.branch_labels.clone(),
                ).unwrap();

                // Compute P matrices for this site class
                // For each branch, select correct (Q, Qfactor) based on branch label
                let p_matrices = calc.compute_p_matrices_for_class(
                    site_class,
                    q_omega0,
                    q_omega1,
                    q_omega2,
                    qfactor_back,
                    qfactor_fore,
                    omega0,
                    omega2,
                );

                // Standard Felsenstein pruning with these P matrices
                calc.compute_site_likelihoods_with_p_matrices(&p_matrices, pi, sequences)
            })
            .collect();

        // For each site: compute log(∑_k freq_k * L_k[site])
        let mut log_likelihood = 0.0;
        for site in 0..self.n_sites {
            let mut site_likelihood = 0.0;
            for k in 0..n_classes {
                site_likelihood += site_class_freqs[k] * class_site_likelihoods[k][site];
            }
            log_likelihood += site_likelihood.ln();
        }

        log_likelihood
    }

    /// Compute P matrices for all branches for a given site class
    fn compute_p_matrices_for_class(
        &self,
        site_class: usize,
        q_omega0: &CachedQMatrix,
        q_omega1: &CachedQMatrix,
        q_omega2: &CachedQMatrix,
        qfactor_back: f64,
        qfactor_fore: f64,
        omega0: f64,
        omega2: f64,
    ) -> Vec<Array2<f64>> {
        let mut p_matrices = Vec::with_capacity(self.tree.n_nodes);

        for node_idx in 0..self.tree.n_nodes {
            let branch_len = self.tree.branch_lengths[node_idx];
            let branch_label = self.get_branch_label(node_idx);

            // Select omega for this (branch, site_class) combination
            let omega = Self::select_omega(branch_label, site_class, omega0, omega2);

            // Select Q matrix eigendecomposition
            let q_matrix = if omega == omega0 {
                q_omega0
            } else if omega == 1.0 {
                q_omega1
            } else {
                q_omega2
            };

            // Select Qfactor
            let qfactor = if branch_label == 0 {
                qfactor_back
            } else {
                qfactor_fore
            };

            // Compute P(t) = exp(Q * (t * qfactor))
            // PAML multiplies branch length by Qfactor (not divides!)
            // See treesub.c lines 7587-7588: t *= Qfactor; PMatUVRoot(Pt, t, ...)
            // CachedQMatrix.expm() computes U * exp(D * t) * U^{-1}
            let p_matrix = q_matrix.expm(branch_len * qfactor);
            p_matrices.push(p_matrix);
        }

        p_matrices
    }

    /// Standard Felsenstein pruning with precomputed P matrices
    fn compute_site_likelihoods_with_p_matrices(
        &mut self,
        p_matrices: &[Array2<f64>],
        pi: ArrayView1<f64>,
        sequences: ArrayView2<i32>,
    ) -> Array1<f64> {
        // Reset workspace
        self.workspace.fill(0.0);

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
        let leaf_set: std::collections::HashSet<usize> =
            self.tree.leaf_node_ids.iter().copied().collect();

        // Clone postorder to avoid borrow checker issues
        let postorder = self.tree.postorder.clone();
        for &node_idx in &postorder {
            if !leaf_set.contains(&node_idx) {
                self.compute_internal_node(node_idx, p_matrices);
            }
        }

        // Root likelihood: sum over all states weighted by pi
        let root_idx = *self.tree.postorder.last().unwrap();
        let root_likelihoods = self.workspace.slice(s![root_idx, .., ..]);

        let mut site_likelihoods = Array1::zeros(self.n_sites);
        for site in 0..self.n_sites {
            site_likelihoods[site] = root_likelihoods
                .slice(s![site, ..])
                .iter()
                .zip(pi.iter())
                .map(|(&l, &p)| l * p)
                .sum();
        }

        site_likelihoods
    }

    /// Compute conditional likelihood for internal node
    fn compute_internal_node(&mut self, node_idx: usize, p_matrices: &[Array2<f64>]) {
        for &child_idx in &self.tree.children[node_idx] {
            let p_matrix = &p_matrices[child_idx];

            for site in 0..self.n_sites {
                // Copy child likelihood to avoid borrow checker issues
                let child_like = self.workspace.slice(s![child_idx, site, ..]).to_owned();

                // L_parent[i] = ∑_j P(i,j) * L_child[j]
                for i in 0..self.n_states {
                    let mut sum = 0.0;
                    for j in 0..self.n_states {
                        sum += p_matrix[[i, j]] * child_like[j];
                    }

                    if child_idx == self.tree.children[node_idx][0] {
                        // First child: initialize
                        self.workspace[[node_idx, site, i]] = sum;
                    } else {
                        // Subsequent children: multiply
                        self.workspace[[node_idx, site, i]] *= sum;
                    }
                }
            }
        }
    }
}
