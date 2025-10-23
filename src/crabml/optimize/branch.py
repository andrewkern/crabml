"""
Parameter optimization for branch codon models.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict

from ..models.codon import compute_codon_frequencies_f3x4
from ..models.codon_branch import CodonBranchModel
from ..core.likelihood_rust import RustLikelihoodCalculator
from ..io.sequences import Alignment
from ..io.trees import Tree


class BranchModelOptimizer:
    """
    Optimize parameters for branch codon models.

    This optimizer estimates:
    - kappa (transition/transversion ratio)
    - omega values (one per branch or per label group)
    - branch lengths (individual or global scaling)

    Two modes:
    1. Free-ratio (free_ratio=True): Independent omega for each branch
    2. Multi-ratio (free_ratio=False): Shared omega for labeled branch groups

    Parameters
    ----------
    alignment : Alignment
        Codon alignment
    tree : Tree
        Phylogenetic tree with branch labels
    use_f3x4 : bool, default=True
        Use F3X4 codon frequencies (True) or uniform (False)
    optimize_branch_lengths : bool, default=True
        Optimize individual branch lengths (True) or use global scaling (False)
    free_ratio : bool, default=False
        If True, use free-ratio model (independent omega per branch).
        If False, use multi-ratio model (omega per label).

    Examples
    --------
    Two-ratio model with branch labels:
        >>> tree_str = "((1,2) #1, ((3,4), 5), (6,7));"
        >>> tree = Tree.from_newick(tree_str)
        >>> optimizer = BranchModelOptimizer(alignment, tree, free_ratio=False)
        >>> kappa, omega_dict, lnL = optimizer.optimize()

    Free-ratio model:
        >>> optimizer = BranchModelOptimizer(alignment, tree, free_ratio=True)
        >>> kappa, omega_dict, lnL = optimizer.optimize()
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True,
        free_ratio: bool = False,
    ):
        """Initialize branch model optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths
        self.free_ratio = free_ratio

        # Compute codon frequencies
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)

        # Get number of nodes
        postorder_nodes = list(tree.postorder())
        n_nodes = len(postorder_nodes)

        # Build node labels array matching RustLikelihoodCalculator's node ordering
        # RustLikelihoodCalculator renumbers nodes: leaves first, then internals
        # We need to assign a label to each node based on its branch to parent

        # Build mapping from node.id to label (from branch to parent)
        node_label_map = {}
        for node in postorder_nodes:
            if node.parent is not None and node.label is not None:
                # Parse label like '#1' -> 1
                label_str = node.label.lstrip('#')
                try:
                    node_label_map[node.id] = int(label_str)
                except ValueError:
                    raise ValueError(f"Invalid branch label: {node.label}")
            else:
                # Default to background (0)
                node_label_map[node.id] = 0

        # Apply same renumbering as RustLikelihoodCalculator
        leaves = [node for node in postorder_nodes if node.is_leaf]
        internals = [node for node in postorder_nodes if not node.is_leaf]
        renumbered_postorder = leaves + internals

        # Build label array in renumbered order
        branch_labels = np.array([node_label_map[node.id] for node in renumbered_postorder])

        # Get list of nodes with branches (exclude root)
        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)

        # Create branch model (needs all nodes including root)
        self.model = CodonBranchModel(
            codon_frequencies=self.pi,
            branch_labels=branch_labels,
            free_ratio=free_ratio,
            n_nodes=n_nodes,
        )

        # Store optimization history
        self.history = []

        print(f"BranchModelOptimizer initialized:")
        print(f"  - Alignment: {alignment.n_species} species, {alignment.n_sites} sites")
        print(f"  - Tree: {n_nodes} nodes, {self.n_branches} branches")
        print(f"  - Model: {'Free-ratio' if free_ratio else 'Multi-ratio'}")
        print(f"  - Omega parameters: {self.model.n_omega}")

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for optimization.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector in log space:
            [log(kappa), log(omega_0), log(omega_1), ..., log(branch_lengths...)]

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Transform parameters from log scale
        kappa = np.exp(params[0])

        # Extract omega values
        omega_values = np.exp(params[1:1 + self.model.n_omega])

        # Extract branch lengths - match RustLikelihoodCalculator's tree structure
        if self.optimize_branch_lengths:
            # Use the template as base
            branch_lengths = np.array(self.calc.branch_lengths_template, dtype=float)

            # Update with optimized values
            # The template already has correct length and order (n_nodes)
            # We optimize n_branches parameters (excluding root which is last)
            optimized_lengths = np.exp(params[1 + self.model.n_omega:])
            if len(optimized_lengths) != self.n_branches:
                raise ValueError(
                    f"Expected {self.n_branches} branch lengths, "
                    f"got {len(optimized_lengths)}"
                )

            # Update all but the last entry (root has no parent branch)
            branch_lengths[:-1] = optimized_lengths
        else:
            # Global scaling (not implemented yet)
            raise NotImplementedError("Global branch scaling not yet implemented")

        # Compute log-likelihood using Rust backend
        try:
            log_likelihood = self.model.compute_log_likelihood(
                rust_calc=self.calc,
                kappa=kappa,
                omega_values=omega_values,
                branch_lengths=branch_lengths,
                use_rust=True,
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        # Store in history
        hist_entry = {
            'kappa': kappa,
            'omega': {f'omega{i}': omega_values[i] for i in range(self.model.n_omega)},
            'log_likelihood': log_likelihood,
        }
        if len(self.history) % 10 == 0:  # Print every 10 iterations
            omega_str = ', '.join([f'ω{i}={omega_values[i]:.4f}' for i in range(min(3, self.model.n_omega))])
            if self.model.n_omega > 3:
                omega_str += ', ...'
            print(f"  Iteration {len(self.history)}: lnL={log_likelihood:.4f}, κ={kappa:.4f}, {omega_str}")
        self.history.append(hist_entry)

        # Return negative for minimization
        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 3.0,
        init_omega: float = 0.4,
        method: str = 'L-BFGS-B',
        maxiter: int = 500,
    ) -> Tuple[float, Dict[str, float], float]:
        """
        Optimize parameters to maximize likelihood.

        Parameters
        ----------
        init_kappa : float, default=3.0
            Initial kappa value
        init_omega : float, default=0.4
            Initial omega value (used for all omega parameters)
        method : str, default='L-BFGS-B'
            Optimization method
        maxiter : int, default=500
            Maximum number of iterations

        Returns
        -------
        tuple
            (kappa, omega_dict, log_likelihood)
            - kappa: transition/transversion ratio
            - omega_dict: dictionary mapping omega names to values
            - log_likelihood: final log-likelihood
            Branch lengths are updated in the tree directly
        """
        print(f"\nStarting optimization:")
        print(f"  Method: {method}")
        print(f"  Max iterations: {maxiter}")
        print(f"  Initial kappa: {init_kappa:.4f}")
        print(f"  Initial omega: {init_omega:.4f}")

        # Get initial branch lengths from tree
        init_branch_lengths = [node.branch_length for node in self.branch_nodes]

        # Build initial parameter vector (in log space)
        init_params = (
            [np.log(init_kappa)] +
            [np.log(init_omega)] * self.model.n_omega +
            [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
        )
        init_params = np.array(init_params)

        # Set bounds (match PAML bounds)
        bounds = (
            [(np.log(0.1), np.log(999))] +  # kappa (PAML: 0.1-999)
            [(np.log(1e-6), np.log(999))] * self.model.n_omega +  # omega values (PAML: 0.0001-999)
            [(np.log(0.0001), np.log(50))] * self.n_branches  # branch lengths
        )

        print(f"  Total parameters: {len(init_params)}")
        print(f"    - kappa: 1")
        print(f"    - omega: {self.model.n_omega}")
        print(f"    - branch lengths: {self.n_branches}")

        # Clear history
        self.history = []

        # Optimize
        print("\nOptimizing...")
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )

        if not result.success:
            print(f"\nWarning: Optimization did not converge: {result.message}")

        # Extract final parameters
        kappa = np.exp(result.x[0])
        omega_values = np.exp(result.x[1:1 + self.model.n_omega])
        branch_lengths = np.exp(result.x[1 + self.model.n_omega:])

        # Update tree with final branch lengths
        for i, node in enumerate(self.branch_nodes):
            node.branch_length = branch_lengths[i]

        # Compute final likelihood
        final_lnL = -result.fun

        # Create omega dictionary
        omega_dict = {f'omega{i}': omega_values[i] for i in range(self.model.n_omega)}

        print(f"\nOptimization complete:")
        print(f"  Log-likelihood: {final_lnL:.6f}")
        print(f"  kappa: {kappa:.6f}")
        for name, value in omega_dict.items():
            print(f"  {name}: {value:.6f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")

        return kappa, omega_dict, final_lnL
