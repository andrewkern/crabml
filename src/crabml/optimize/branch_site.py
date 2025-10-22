"""
Parameter optimization for Branch-Site Model A.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple

from ..core.likelihood_rust import RustLikelihoodCalculator
from ..models.codon import build_codon_Q_matrix, compute_codon_frequencies_f3x4
from ..models.codon_branch_site import BranchSiteModelA
from ..io.sequences import Alignment
from ..io.trees import Tree
import crabml_rust


class BranchSiteModelAOptimizer:
    """
    Optimize parameters for Branch-Site Model A.

    This optimizer estimates:
    - kappa (transition/transversion ratio)
    - p0 (proportion of conserved sites)
    - p1 (proportion of neutral sites)
    - omega0 (dN/dS for purifying selection, < 1)
    - omega2 (dN/dS for positive selection on foreground, > 1)
    - branch lengths

    Using maximum likelihood estimation.
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """
        Initialize Branch-Site Model A optimizer.

        Parameters
        ----------
        alignment : Alignment
            Codon alignment
        tree : Tree
            Phylogenetic tree with branch labels (0=background, 1=foreground)
        use_f3x4 : bool
            Use F3X4 codon frequencies (True) or uniform (False)
        optimize_branch_lengths : bool
            Optimize individual branch lengths (True) or use global scaling (False)
        """
        self.alignment = alignment
        self.tree = tree
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        # Validate tree has branch labels
        tree.validate_branch_site_labels()
        self.branch_labels = tree.get_branch_labels()

        # Compute codon frequencies
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)

        # Get branch nodes
        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)

        # Remap branch labels to renumbered tree structure
        branches_original = tree.get_branches()
        branch_labels_original = tree.get_branch_labels()
        self.branch_labels_renumbered = [0] * (len(self.calc.tree_structure) - 1)

        for i, (parent, child) in enumerate(branches_original):
            renumbered_child_id = self.calc.node_id_map[child.id]
            if renumbered_child_id >= len(self.branch_labels_renumbered):
                continue
            self.branch_labels_renumbered[renumbered_child_id] = branch_labels_original[i]

        self.branch_labels_u8 = [int(x) for x in self.branch_labels_renumbered]

        # Create model
        self.model = BranchSiteModelA(self.pi, self.branch_labels)

        # Store optimization history
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for optimization.

        Parameters
        ----------
        params : np.ndarray
            If optimize_branch_lengths is True:
                [log(kappa), log(omega0), log(omega2), logit(p0), logit(p1), log(branch1), ...]
            If optimize_branch_lengths is False:
                [log(kappa), log(omega0), log(omega2), logit(p0), logit(p1), log(branch_scale)]

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Extract parameters
        kappa = np.exp(params[0])
        omega0 = np.exp(params[1])
        omega2 = np.exp(params[2])

        # Use sigmoid for p0 and p1 to keep in [0,1]
        # Then enforce p0 + p1 < 1
        p0_raw = 1.0 / (1.0 + np.exp(-params[3]))
        p1_raw = 1.0 / (1.0 + np.exp(-params[4]))

        # Rescale to ensure p0 + p1 < 1
        total = p0_raw + p1_raw
        if total >= 0.999:  # Leave room for p2
            p0 = p0_raw / total * 0.999
            p1 = p1_raw / total * 0.999
        else:
            p0 = p0_raw
            p1 = p1_raw

        # Update branch lengths
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[5 + i])
        else:
            # Scale all branch lengths
            scale = np.exp(params[5])
            for node in self.branch_nodes:
                node.branch_length *= scale

        # Compute site class frequencies
        try:
            site_class_freqs = self.model.compute_site_class_frequencies(p0, p1)
        except ValueError as e:
            # Invalid parameters (e.g., p0 + p1 >= 1)
            return 1e10

        # Compute Qfactors
        qfactor_back, qfactor_fore = self.model.compute_qfactors(kappa, p0, p1, omega0, omega2)

        # Build Q matrices
        Q_omega0 = build_codon_Q_matrix(kappa=kappa, omega=omega0, pi=self.pi, normalization_factor=1.0)
        Q_omega1 = build_codon_Q_matrix(kappa=kappa, omega=1.0, pi=self.pi, normalization_factor=1.0)
        Q_omega2 = build_codon_Q_matrix(kappa=kappa, omega=omega2, pi=self.pi, normalization_factor=1.0)

        # Compute log-likelihood
        try:
            log_likelihood = crabml_rust.compute_branch_site_log_likelihood(
                Q_omega0,
                Q_omega1,
                Q_omega2,
                qfactor_back,
                qfactor_fore,
                omega0,
                omega2,
                site_class_freqs.tolist(),
                self.pi,
                self.calc.tree_structure,
                self.calc.branch_lengths_template,
                self.branch_labels_u8,
                self.calc.leaf_names_ordered,
                self.calc.sequences,
                self.calc.leaf_node_ids,
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        # Store in history
        hist_entry = {
            'kappa': kappa,
            'omega0': omega0,
            'omega2': omega2,
            'p0': p0,
            'p1': p1,
            'log_likelihood': log_likelihood
        }
        if self.optimize_branch_lengths:
            hist_entry['branch_lengths'] = [node.branch_length for node in self.branch_nodes]
        self.history.append(hist_entry)

        # Return negative for minimization
        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_omega0: float = 0.1,
        init_omega2: float = 2.0,
        init_p0: float = 0.3,
        init_p1: float = 0.3,
        method: str = 'L-BFGS-B',
        maxiter: int = 300
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Optimize parameters to maximize likelihood.

        Parameters
        ----------
        init_kappa : float
            Initial kappa value
        init_omega0 : float
            Initial omega0 value (should be < 1)
        init_omega2 : float
            Initial omega2 value (should be > 1)
        init_p0 : float
            Initial p0 proportion
        init_p1 : float
            Initial p1 proportion
        method : str
            Optimization method (default 'L-BFGS-B')
        maxiter : int
            Maximum number of iterations

        Returns
        -------
        tuple
            (kappa, omega0, omega2, p0, p1, log_likelihood)
            Branch lengths are updated in the tree directly
        """
        # Initial parameters (in log/logit space)
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa),
                 np.log(init_omega0),
                 np.log(init_omega2),
                 np.log(init_p0 / (1 - init_p0)),  # Logit transform
                 np.log(init_p1 / (1 - init_p1))] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )

            # Bounds: kappa [0.1, 100], omega0 [0.001, 0.999], omega2 [1.001, 20],
            #         p0, p1 logits [-10, 10], branches [0.0001, 50]
            bounds = (
                [(np.log(0.1), np.log(100)),        # kappa
                 (np.log(0.001), np.log(0.999)),    # omega0
                 (np.log(1.001), np.log(20)),       # omega2
                 (-10, 10), (-10, 10)] +            # p0, p1 logits
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )

            n_foreground = sum(1 for x in self.branch_labels if x == 1)
            n_background = sum(1 for x in self.branch_labels if x == 0)

            print(f"Starting Branch-Site Model A optimization")
            print(f"Method: {method}, max iterations: {maxiter}")
            print(f"Tree: {n_background} background branches, {n_foreground} foreground branches")
            print(f"Initial parameters:")
            print(f"  kappa  = {init_kappa:.4f}")
            print(f"  omega0 = {init_omega0:.4f} (purifying)")
            print(f"  omega2 = {init_omega2:.4f} (positive selection)")
            print(f"  p0     = {init_p0:.4f}")
            print(f"  p1     = {init_p1:.4f}")
            print(f"  p2     = {1 - init_p0 - init_p1:.4f}")
            print(f"Optimizing {self.n_branches} branch lengths")
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_omega0),
                np.log(init_omega2),
                np.log(init_p0 / (1 - init_p0)),
                np.log(init_p1 / (1 - init_p1)),
                np.log(1.0)  # Initial branch scale
            ])

            bounds = [
                (np.log(0.1), np.log(100)),       # kappa
                (np.log(0.001), np.log(0.999)),   # omega0
                (np.log(1.001), np.log(20)),      # omega2
                (-10, 10), (-10, 10),             # p0, p1 logits
                (np.log(0.01), np.log(100))       # branch_scale
            ]

            print(f"Starting Branch-Site Model A optimization (branch scaling mode)")
            print(f"Initial: kappa={init_kappa:.4f}, omega0={init_omega0:.4f}, "
                  f"omega2={init_omega2:.4f}, p0={init_p0:.4f}, p1={init_p1:.4f}")

        # Clear history
        self.history = []

        # Optimize
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )

        # Extract optimal parameters
        opt_kappa = np.exp(result.x[0])
        opt_omega0 = np.exp(result.x[1])
        opt_omega2 = np.exp(result.x[2])

        # Extract p0 and p1 from sigmoid
        p0_raw = 1.0 / (1.0 + np.exp(-result.x[3]))
        p1_raw = 1.0 / (1.0 + np.exp(-result.x[4]))
        total = p0_raw + p1_raw
        if total >= 0.999:
            opt_p0 = p0_raw / total * 0.999
            opt_p1 = p1_raw / total * 0.999
        else:
            opt_p0 = p0_raw
            opt_p1 = p1_raw

        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final parameters:")
        print(f"  kappa  = {opt_kappa:.6f}")
        print(f"  omega0 = {opt_omega0:.6f}")
        print(f"  omega2 = {opt_omega2:.6f}")
        print(f"  p0     = {opt_p0:.6f}")
        print(f"  p1     = {opt_p1:.6f}")
        print(f"  p2     = {1 - opt_p0 - opt_p1:.6f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Function evaluations: {len(self.history)}")
        print(f"Success: {result.success}")
        if not result.success:
            print(f"Message: {result.message}")

        if self.optimize_branch_lengths:
            print(f"Branch lengths optimized")
        else:
            opt_branch_scale = np.exp(result.x[5])
            print(f"Branch scale: {opt_branch_scale:.4f}")

        return opt_kappa, opt_omega0, opt_omega2, opt_p0, opt_p1, opt_log_likelihood
