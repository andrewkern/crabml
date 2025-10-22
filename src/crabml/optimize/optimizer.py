"""
Parameter optimization for phylogenetic models.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple

from ..core.likelihood_rust import RustLikelihoodCalculator
from ..models.codon import (
    M0CodonModel,
    M1aCodonModel,
    M2aCodonModel,
    M3CodonModel,
    M7CodonModel,
    M8CodonModel,
    compute_codon_frequencies_f3x4,
)
from ..io.sequences import Alignment
from ..io.trees import Tree


class M0Optimizer:
    """
    Optimize parameters for M0 codon model.

    This optimizer estimates:
    - kappa (transition/transversion ratio)
    - omega (dN/dS ratio)
    - branch lengths (individual or global scaling)

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
        Initialize optimizer.

        Parameters
        ----------
        alignment : Alignment
            Codon alignment
        tree : Tree
            Phylogenetic tree
        use_f3x4 : bool
            Use F3X4 codon frequencies (True) or uniform (False)
        optimize_branch_lengths : bool
            Optimize individual branch lengths (True) or use global scaling (False)
        """
        self.alignment = alignment
        self.tree = tree
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        # Compute codon frequencies
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)

        # Get list of nodes with branches (exclude root)
        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)

        # Store optimization history
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for optimization.

        Parameters
        ----------
        params : np.ndarray
            If optimize_branch_lengths is True:
                [log(kappa), log(omega), log(branch1), log(branch2), ...]
            If optimize_branch_lengths is False:
                [log(kappa), log(omega), log(branch_scale)]

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Transform parameters from log scale
        kappa = np.exp(params[0])
        omega = np.exp(params[1])

        # Update branch lengths
        if self.optimize_branch_lengths:
            # Individual branch lengths
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[2 + i])
            branch_scale = 1.0  # Don't scale when using individual lengths
        else:
            # Global scaling factor
            branch_scale = np.exp(params[2])

        # Create model with these parameters
        model = M0CodonModel(kappa=kappa, omega=omega, pi=self.pi)
        Q = model.get_Q_matrix()

        # Compute log-likelihood
        try:
            log_likelihood = self.calc.compute_log_likelihood(
                Q, self.pi, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            # If computation fails, return very bad likelihood
            print(f"Error computing likelihood: {e}")
            return 1e10

        # Store in history
        hist_entry = {
            'kappa': kappa,
            'omega': omega,
            'log_likelihood': log_likelihood
        }
        if self.optimize_branch_lengths:
            hist_entry['branch_lengths'] = [node.branch_length for node in self.branch_nodes]
        else:
            hist_entry['branch_scale'] = branch_scale
        self.history.append(hist_entry)

        # Return negative for minimization
        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_omega: float = 0.4,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float]:
        """
        Optimize parameters to maximize likelihood.

        Parameters
        ----------
        init_kappa : float
            Initial kappa value
        init_omega : float
            Initial omega value
        method : str
            Optimization method (default 'L-BFGS-B')
        maxiter : int
            Maximum number of iterations

        Returns
        -------
        tuple
            (kappa, omega, log_likelihood) if optimize_branch_lengths is True
            Branch lengths are updated in the tree directly
        """
        # Initial parameters (in log space)
        if self.optimize_branch_lengths:
            # Start with current branch lengths from tree
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa), np.log(init_omega)] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )

            # Bounds: kappa [0.1, 100], omega [0.001, 10], branches [0.0001, 50]
            bounds = (
                [(np.log(0.1), np.log(100)), (np.log(0.001), np.log(10))] +
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )

            print(f"Starting optimization with method={method}, maxiter={maxiter}")
            print(f"Initial: kappa={init_kappa:.4f}, omega={init_omega:.4f}")
            print(f"Optimizing {self.n_branches} branch lengths")
        else:
            # Use global branch scaling
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_omega),
                np.log(1.0)  # Initial branch scale
            ])

            bounds = [
                (np.log(0.1), np.log(100)),   # log(kappa)
                (np.log(0.001), np.log(10)),  # log(omega)
                (np.log(0.01), np.log(100))   # log(branch_scale)
            ]

            print(f"Starting optimization with method={method}, maxiter={maxiter}")
            print(f"Initial: kappa={init_kappa:.4f}, omega={init_omega:.4f}")

        # Clear history
        self.history = []

        # Optimize
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        # Extract optimal parameters
        opt_kappa = np.exp(result.x[0])
        opt_omega = np.exp(result.x[1])
        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, omega={opt_omega:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")

        if self.optimize_branch_lengths:
            print(f"Branch lengths updated in tree")
        else:
            opt_branch_scale = np.exp(result.x[2])
            print(f"Branch scale: {opt_branch_scale:.4f}")

        return opt_kappa, opt_omega, opt_log_likelihood


class M1aOptimizer:
    """
    Optimize parameters for M1a (NearlyNeutral) codon model.

    Estimates:
    - kappa (transition/transversion ratio)
    - omega0 (dN/dS for purifying class, constrained < 1)
    - p0 (proportion in purifying class)
    - branch lengths (individual or global scaling)
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M1a optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        # Extract parameters
        kappa = np.exp(params[0])
        omega0 = np.exp(params[1])
        p0 = 1.0 / (1.0 + np.exp(-params[2]))  # Sigmoid for [0,1]

        # Update branch lengths
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[3 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[3])

        # Create model
        model = M1aCodonModel(kappa=kappa, omega0=omega0, p0=p0, pi=self.pi)

        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        # Store in history
        self.history.append({
            'kappa': kappa,
            'omega0': omega0,
            'p0': p0,
            'log_likelihood': log_likelihood
        })

        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_omega0: float = 0.5,
        init_p0: float = 0.7,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float]:
        """Optimize M1a parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa), np.log(init_omega0),
                 np.log(init_p0 / (1 - init_p0))] +  # Logit transform
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),      # kappa
                 (np.log(0.001), np.log(0.999)),  # omega0
                 (-10, 10)] +                      # p0 (logit)
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_omega0),
                np.log(init_p0 / (1 - init_p0)),
                np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (np.log(0.001), np.log(0.999)),
                (-10, 10),
                (np.log(0.01), np.log(100))
            ]

        print(f"Starting M1a optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, omega0={init_omega0:.4f}, p0={init_p0:.4f}")

        self.history = []

        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        opt_kappa = np.exp(result.x[0])
        opt_omega0 = np.exp(result.x[1])
        opt_p0 = 1.0 / (1.0 + np.exp(-result.x[2]))
        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, omega0={opt_omega0:.4f}, p0={opt_p0:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")

        return opt_kappa, opt_omega0, opt_p0, opt_log_likelihood


class M2aOptimizer:
    """
    Optimize parameters for M2a (PositiveSelection) codon model.

    Estimates:
    - kappa
    - omega0 (purifying, < 1)
    - omega2 (positive selection, > 1)
    - p0, p1 (proportions)
    - branch lengths
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M2a optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        kappa = np.exp(params[0])
        omega0 = np.exp(params[1])
        omega2 = np.exp(params[2])

        # Use softmax for proportions (ensures sum to 1)
        logits = params[3:5]  # p0 and p1 logits
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        props = exp_logits / (exp_logits.sum() + 1)  # Reserve space for p2
        p0, p1 = props[0], props[1]

        # Update branch lengths
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[5 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[5])

        # Create model
        model = M2aCodonModel(
            kappa=kappa, omega0=omega0, omega2=omega2,
            p0=p0, p1=p1, pi=self.pi
        )

        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        self.history.append({
            'kappa': kappa,
            'omega0': omega0,
            'omega2': omega2,
            'p0': p0,
            'p1': p1,
            'log_likelihood': log_likelihood
        })

        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_omega0: float = 0.5,
        init_omega2: float = 2.0,
        init_p0: float = 0.5,
        init_p1: float = 0.3,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float, float, float]:
        """Optimize M2a parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa), np.log(init_omega0), np.log(init_omega2),
                 0.0, 0.0] +  # p0, p1 logits (will be normalized)
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),      # kappa
                 (np.log(0.001), np.log(0.999)),  # omega0
                 (np.log(1.001), np.log(20)),     # omega2
                 (-10, 10), (-10, 10)] +          # p0, p1 logits
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa), np.log(init_omega0), np.log(init_omega2),
                0.0, 0.0, np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (np.log(0.001), np.log(0.999)),
                (np.log(1.001), np.log(20)),
                (-10, 10), (-10, 10),
                (np.log(0.01), np.log(100))
            ]

        print(f"Starting M2a optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, omega0={init_omega0:.4f}, "
              f"omega2={init_omega2:.4f}")

        self.history = []

        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        opt_kappa = np.exp(result.x[0])
        opt_omega0 = np.exp(result.x[1])
        opt_omega2 = np.exp(result.x[2])

        logits = result.x[3:5]
        exp_logits = np.exp(logits - np.max(logits))
        props = exp_logits / (exp_logits.sum() + 1)
        opt_p0, opt_p1 = props[0], props[1]

        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, omega0={opt_omega0:.4f}, "
              f"omega2={opt_omega2:.4f}")
        print(f"Proportions: p0={opt_p0:.4f}, p1={opt_p1:.4f}, p2={1-opt_p0-opt_p1:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")

        return opt_kappa, opt_omega0, opt_omega2, opt_p0, opt_p1, opt_log_likelihood


class M3Optimizer:
    """
    Optimize parameters for M3 (Discrete) codon model.

    Estimates:
    - kappa
    - K omega values (one per site class)
    - K-1 proportion parameters (K-th is 1 - sum of others)
    - branch lengths
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        n_classes: int = 3,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """
        Initialize M3 optimizer.

        Parameters
        ----------
        alignment : Alignment
            Codon alignment
        tree : Tree
            Phylogenetic tree
        n_classes : int
            Number of site classes (default 3)
        use_f3x4 : bool
            Use F3X4 codon frequencies
        optimize_branch_lengths : bool
            Optimize individual branch lengths
        """
        self.alignment = alignment
        self.tree = tree
        self.n_classes = n_classes
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        # Extract kappa
        kappa = np.exp(params[0])

        # Extract K omega values
        omegas = [np.exp(params[i]) for i in range(1, 1 + self.n_classes)]

        # Extract K-1 proportion logits and compute proportions using softmax
        # params layout: [log(kappa), log(omega1), ..., log(omegaK), logit1, ..., logitK-1, branches...]
        logit_start = 1 + self.n_classes
        logit_end = logit_start + (self.n_classes - 1)
        logits = params[logit_start:logit_end]

        # Use softmax to ensure proportions sum to 1
        # Reserve space for the last proportion
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        proportions_unnorm = np.append(exp_logits, 1.0)  # Add 1.0 for last class
        proportions = proportions_unnorm / proportions_unnorm.sum()

        # Update branch lengths
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[logit_end + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[logit_end])

        # Create model
        model = M3CodonModel(
            kappa=kappa,
            omegas=omegas,
            proportions=proportions.tolist(),
            pi=self.pi
        )

        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            prop_list, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, prop_list, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        # Store in history
        hist_entry = {
            'kappa': kappa,
            'omegas': omegas,
            'proportions': proportions.tolist(),
            'log_likelihood': log_likelihood
        }
        self.history.append(hist_entry)

        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_omegas: list[float] = None,
        init_proportions: list[float] = None,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, list[float], list[float], float]:
        """
        Optimize M3 parameters.

        Parameters
        ----------
        init_kappa : float
            Initial kappa value
        init_omegas : list[float], optional
            Initial omega values (default: evenly spaced from 0.1 to 2.0)
        init_proportions : list[float], optional
            Initial proportions (default: equal proportions)
        method : str
            Optimization method
        maxiter : int
            Maximum iterations

        Returns
        -------
        tuple
            (kappa, omegas, proportions, log_likelihood)
        """
        # Set defaults
        if init_omegas is None:
            # Evenly spaced omega values from 0.1 to 2.0
            init_omegas = np.linspace(0.1, 2.0, self.n_classes).tolist()
        if init_proportions is None:
            # Equal proportions
            init_proportions = [1.0 / self.n_classes] * self.n_classes

        # Validate inputs
        if len(init_omegas) != self.n_classes:
            raise ValueError(f"init_omegas must have {self.n_classes} elements")
        if len(init_proportions) != self.n_classes:
            raise ValueError(f"init_proportions must have {self.n_classes} elements")
        if not np.isclose(sum(init_proportions), 1.0):
            raise ValueError("init_proportions must sum to 1")

        # Build initial parameters
        # Layout: [log(kappa), log(omega1), ..., log(omegaK), logit1, ..., logitK-1, branches...]
        init_params_list = [np.log(init_kappa)]
        init_params_list.extend([np.log(w) for w in init_omegas])

        # Initialize proportion logits (K-1 values, last is implicit)
        # Use zeros which will give equal proportions after softmax
        init_params_list.extend([0.0] * (self.n_classes - 1))

        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params_list.extend([np.log(max(bl, 0.001)) for bl in init_branch_lengths])
        else:
            init_params_list.append(np.log(1.0))

        init_params = np.array(init_params_list)

        # Set bounds
        bounds = [(np.log(0.1), np.log(100))]  # kappa
        bounds.extend([(np.log(0.001), np.log(20))] * self.n_classes)  # omegas
        bounds.extend([(-10, 10)] * (self.n_classes - 1))  # proportion logits

        if self.optimize_branch_lengths:
            bounds.extend([(np.log(0.0001), np.log(50))] * self.n_branches)
        else:
            bounds.append((np.log(0.01), np.log(100)))

        print(f"Starting M3 optimization with method={method}, maxiter={maxiter}")
        print(f"Number of site classes: {self.n_classes}")
        print(f"Initial: kappa={init_kappa:.4f}")
        print(f"Initial omegas: {', '.join([f'{w:.4f}' for w in init_omegas])}")
        print(f"Initial proportions: {', '.join([f'{p:.4f}' for p in init_proportions])}")

        self.history = []

        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        # Extract optimized parameters
        opt_kappa = np.exp(result.x[0])
        opt_omegas = [np.exp(result.x[i]) for i in range(1, 1 + self.n_classes)]

        # Extract proportions
        logit_start = 1 + self.n_classes
        logit_end = logit_start + (self.n_classes - 1)
        logits = result.x[logit_start:logit_end]
        exp_logits = np.exp(logits - np.max(logits))
        proportions_unnorm = np.append(exp_logits, 1.0)
        opt_proportions = (proportions_unnorm / proportions_unnorm.sum()).tolist()

        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}")
        print(f"Final omegas: {', '.join([f'{w:.4f}' for w in opt_omegas])}")
        print(f"Final proportions: {', '.join([f'{p:.4f}' for p in opt_proportions])}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")

        return opt_kappa, opt_omegas, opt_proportions, opt_log_likelihood


class M7Optimizer:
    """
    Optimize parameters for M7 (beta) codon model.

    Estimates:
    - kappa (transition/transversion ratio)
    - p_beta (beta distribution shape parameter 1)
    - q_beta (beta distribution shape parameter 2)
    - branch lengths
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        ncatG: int = 10,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M7 optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.ncatG = ncatG
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)

        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        kappa = np.exp(params[0])
        p_beta = np.exp(params[1])
        q_beta = np.exp(params[2])

        # Update branch lengths
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[3 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[3])

        # Create model
        model = M7CodonModel(
            kappa=kappa,
            p_beta=p_beta,
            q_beta=q_beta,
            ncatG=self.ncatG,
            pi=self.pi
        )

        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        self.history.append({
            'kappa': kappa,
            'p_beta': p_beta,
            'q_beta': q_beta,
            'log_likelihood': log_likelihood
        })

        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_p_beta: float = 0.5,
        init_q_beta: float = 0.5,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float]:
        """Optimize M7 parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa), np.log(init_p_beta), np.log(init_q_beta)] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),      # kappa
                 (np.log(0.005), np.log(99)),     # p_beta
                 (np.log(0.005), np.log(99))] +   # q_beta
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_p_beta),
                np.log(init_q_beta),
                np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.01), np.log(100))
            ]

        print(f"Starting M7 optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, p={init_p_beta:.4f}, q={init_q_beta:.4f}")
        print(f"Beta distribution discretized into {self.ncatG} categories")

        self.history = []

        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        opt_kappa = np.exp(result.x[0])
        opt_p_beta = np.exp(result.x[1])
        opt_q_beta = np.exp(result.x[2])
        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, p={opt_p_beta:.4f}, q={opt_q_beta:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")

        return opt_kappa, opt_p_beta, opt_q_beta, opt_log_likelihood


class M8Optimizer:
    """
    Optimize parameters for M8 (beta & omega>1) codon model.

    Estimates:
    - kappa (transition/transversion ratio)
    - p0 (proportion in beta distribution)
    - p_beta (beta shape parameter 1)
    - q_beta (beta shape parameter 2)
    - omega_s (omega for positive selection class, > 1)
    - branch lengths
    """

    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        ncatG: int = 10,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M8 optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.ncatG = ncatG
        self.use_f3x4 = use_f3x4
        self.optimize_branch_lengths = optimize_branch_lengths

        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61

        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)

        self.branch_nodes = [node for node in tree.postorder() if node.parent is not None]
        self.n_branches = len(self.branch_nodes)
        self.history = []

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        kappa = np.exp(params[0])
        p0 = 1.0 / (1.0 + np.exp(-params[1]))  # Sigmoid for [0,1]
        p_beta = np.exp(params[2])
        q_beta = np.exp(params[3])
        omega_s = np.exp(params[4])

        # Update branch lengths
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[5 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[5])

        # Create model
        model = M8CodonModel(
            kappa=kappa,
            p0=p0,
            p_beta=p_beta,
            q_beta=q_beta,
            omega_s=omega_s,
            ncatG=self.ncatG,
            pi=self.pi
        )

        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10

        self.history.append({
            'kappa': kappa,
            'p0': p0,
            'p_beta': p_beta,
            'q_beta': q_beta,
            'omega_s': omega_s,
            'log_likelihood': log_likelihood
        })

        return -log_likelihood

    def optimize(
        self,
        init_kappa: float = 2.0,
        init_p0: float = 0.9,
        init_p_beta: float = 0.5,
        init_q_beta: float = 0.5,
        init_omega_s: float = 2.0,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float, float, float]:
        """Optimize M8 parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa),
                 np.log(init_p0 / (1 - init_p0)),  # Logit transform
                 np.log(init_p_beta),
                 np.log(init_q_beta),
                 np.log(init_omega_s)] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),       # kappa
                 (-10, 10),                         # p0 (logit)
                 (np.log(0.005), np.log(99)),       # p_beta
                 (np.log(0.005), np.log(99)),       # q_beta
                 (np.log(1.001), np.log(20))] +     # omega_s
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_p0 / (1 - init_p0)),
                np.log(init_p_beta),
                np.log(init_q_beta),
                np.log(init_omega_s),
                np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (-10, 10),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(1.001), np.log(20)),
                (np.log(0.01), np.log(100))
            ]

        print(f"Starting M8 optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, p0={init_p0:.4f}, p={init_p_beta:.4f}, "
              f"q={init_q_beta:.4f}, omega_s={init_omega_s:.4f}")
        print(f"Beta distribution discretized into {self.ncatG} categories + 1 positive selection class")

        self.history = []

        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        opt_kappa = np.exp(result.x[0])
        opt_p0 = 1.0 / (1.0 + np.exp(-result.x[1]))
        opt_p_beta = np.exp(result.x[2])
        opt_q_beta = np.exp(result.x[3])
        opt_omega_s = np.exp(result.x[4])
        opt_log_likelihood = -result.fun

        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, p0={opt_p0:.4f}, p={opt_p_beta:.4f}, "
              f"q={opt_q_beta:.4f}, omega_s={opt_omega_s:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")

        return opt_kappa, opt_p0, opt_p_beta, opt_q_beta, opt_omega_s, opt_log_likelihood


class M5Optimizer:
    """
    Optimizer for M5 (gamma) codon model.
    
    Optimizes kappa, alpha, beta parameters and optionally branch lengths.
    """
    
    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        ncatG: int = 10,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M5 optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.ncatG = ncatG
        self.optimize_branch_lengths = optimize_branch_lengths
        
        # Compute codon frequencies
        from ..models.codon import compute_codon_frequencies_f3x4
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61
        
        # Get branch nodes if optimizing branch lengths
        if optimize_branch_lengths:
            self.branch_nodes = []
            for node in tree.traverse():
                if node.branch_length is not None:
                    self.branch_nodes.append(node)
            self.n_branches = len(self.branch_nodes)
        else:
            self.n_branches = 0
        
        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
    
    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        # Extract parameters (log-transformed)
        kappa = np.exp(params[0])
        alpha = np.exp(params[1])
        beta = np.exp(params[2])
        
        # Update branch lengths if optimizing
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[3 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[3])
        
        # Create model and compute likelihood
        from ..models.codon import M5CodonModel
        model = M5CodonModel(kappa=kappa, alpha=alpha, beta=beta, ncatG=self.ncatG, pi=self.pi)
        
        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10
        
        self.history.append({
            'kappa': kappa,
            'alpha': alpha,
            'beta': beta,
            'log_likelihood': log_likelihood
        })
        
        return -log_likelihood
    
    def optimize(
        self,
        init_kappa: float = 2.0,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float]:
        """Optimize M5 parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa),
                 np.log(init_alpha),
                 np.log(init_beta)] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),       # kappa
                 (np.log(0.005), np.log(99)),      # alpha
                 (np.log(0.005), np.log(99))] +    # beta
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_alpha),
                np.log(init_beta),
                np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.01), np.log(100))
            ]
        
        print(f"Starting M5 optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, alpha={init_alpha:.4f}, beta={init_beta:.4f}")
        print(f"Gamma distribution discretized into {self.ncatG} categories")
        
        self.history = []
        
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        opt_kappa = np.exp(result.x[0])
        opt_alpha = np.exp(result.x[1])
        opt_beta = np.exp(result.x[2])
        opt_log_likelihood = -result.fun
        
        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, alpha={opt_alpha:.4f}, beta={opt_beta:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")
        
        return opt_kappa, opt_alpha, opt_beta, opt_log_likelihood


class M9Optimizer:
    """
    Optimizer for M9 (beta & gamma) codon model.
    
    Optimizes kappa, p0, p_beta, q_beta, alpha, beta_gamma parameters 
    and optionally branch lengths.
    """
    
    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        ncatG: int = 10,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M9 optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.ncatG = ncatG
        self.optimize_branch_lengths = optimize_branch_lengths
        
        # Compute codon frequencies
        from ..models.codon import compute_codon_frequencies_f3x4
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61
        
        # Get branch nodes if optimizing branch lengths
        if optimize_branch_lengths:
            self.branch_nodes = []
            for node in tree.traverse():
                if node.branch_length is not None:
                    self.branch_nodes.append(node)
            self.n_branches = len(self.branch_nodes)
        else:
            self.n_branches = 0
        
        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
    
    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        # Extract parameters
        kappa = np.exp(params[0])
        p0 = 1.0 / (1.0 + np.exp(-params[1]))  # Sigmoid for [0,1]
        p_beta = np.exp(params[2])
        q_beta = np.exp(params[3])
        alpha = np.exp(params[4])
        beta_gamma = np.exp(params[5])
        
        # Update branch lengths if optimizing
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[6 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[6])
        
        # Create model and compute likelihood
        from ..models.codon import M9CodonModel
        model = M9CodonModel(
            kappa=kappa, p0=p0, p_beta=p_beta, q_beta=q_beta,
            alpha=alpha, beta_gamma=beta_gamma, ncatG=self.ncatG, pi=self.pi
        )
        
        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10
        
        self.history.append({
            'kappa': kappa,
            'p0': p0,
            'p_beta': p_beta,
            'q_beta': q_beta,
            'alpha': alpha,
            'beta_gamma': beta_gamma,
            'log_likelihood': log_likelihood
        })
        
        return -log_likelihood
    
    def optimize(
        self,
        init_kappa: float = 2.0,
        init_p0: float = 0.5,
        init_p_beta: float = 0.5,
        init_q_beta: float = 0.5,
        init_alpha: float = 1.0,
        init_beta_gamma: float = 1.0,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Optimize M9 parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa),
                 np.log(init_p0 / (1 - init_p0)),  # Logit transform
                 np.log(init_p_beta),
                 np.log(init_q_beta),
                 np.log(init_alpha),
                 np.log(init_beta_gamma)] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),       # kappa
                 (-10, 10),                         # p0 (logit)
                 (np.log(0.005), np.log(99)),       # p_beta
                 (np.log(0.005), np.log(99)),       # q_beta
                 (np.log(0.005), np.log(99)),       # alpha
                 (np.log(0.005), np.log(99))] +     # beta_gamma
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_p0 / (1 - init_p0)),
                np.log(init_p_beta),
                np.log(init_q_beta),
                np.log(init_alpha),
                np.log(init_beta_gamma),
                np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (-10, 10),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.01), np.log(100))
            ]
        
        print(f"Starting M9 optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, p0={init_p0:.4f}, p_beta={init_p_beta:.4f}, "
              f"q_beta={init_q_beta:.4f}, alpha={init_alpha:.4f}, beta={init_beta_gamma:.4f}")
        print(f"Beta distribution ({self.ncatG} categories) + Gamma distribution ({self.ncatG} categories)")
        
        self.history = []
        
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        opt_kappa = np.exp(result.x[0])
        opt_p0 = 1.0 / (1.0 + np.exp(-result.x[1]))
        opt_p_beta = np.exp(result.x[2])
        opt_q_beta = np.exp(result.x[3])
        opt_alpha = np.exp(result.x[4])
        opt_beta_gamma = np.exp(result.x[5])
        opt_log_likelihood = -result.fun
        
        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, p0={opt_p0:.4f}, p_beta={opt_p_beta:.4f}, "
              f"q_beta={opt_q_beta:.4f}, alpha={opt_alpha:.4f}, beta={opt_beta_gamma:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")
        
        return opt_kappa, opt_p0, opt_p_beta, opt_q_beta, opt_alpha, opt_beta_gamma, opt_log_likelihood


class M4Optimizer:
    """
    Optimizer for M4 (freqs) codon model.
    
    Optimizes kappa and proportions (p0, p1, p2, p3) with p4 = 1 - sum.
    Omega values are fixed at {0, 1/3, 2/3, 1, 3}.
    """
    
    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M4 optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.optimize_branch_lengths = optimize_branch_lengths
        
        # Compute codon frequencies
        from ..models.codon import compute_codon_frequencies_f3x4
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61
        
        # Get branch nodes if optimizing branch lengths
        if optimize_branch_lengths:
            self.branch_nodes = []
            for node in tree.traverse():
                if node.branch_length is not None:
                    self.branch_nodes.append(node)
            self.n_branches = len(self.branch_nodes)
        else:
            self.n_branches = 0
        
        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
    
    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        # Extract parameters
        kappa = np.exp(params[0])
        
        # Extract proportions using softmax transformation (ensures sum to 1)
        # We have 4 free parameters for 5 proportions
        logits = params[1:5]
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        proportions_raw = np.concatenate([exp_logits, [1.0]])
        proportions = proportions_raw / proportions_raw.sum()
        
        # Update branch lengths if optimizing
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[5 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[5])
        
        # Create model and compute likelihood
        from ..models.codon import M4CodonModel
        model = M4CodonModel(kappa=kappa, proportions=proportions.tolist(), pi=self.pi)
        
        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            props, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, props, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10
        
        self.history.append({
            'kappa': kappa,
            'proportions': proportions.tolist(),
            'log_likelihood': log_likelihood
        })
        
        return -log_likelihood
    
    def optimize(
        self,
        init_kappa: float = 2.0,
        init_proportions: list[float] = None,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, list[float], float]:
        """Optimize M4 parameters."""
        # Default proportions (equal)
        if init_proportions is None:
            init_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            if len(init_proportions) != 5:
                raise ValueError("M4 requires exactly 5 initial proportions")
            # Normalize
            init_proportions = np.array(init_proportions)
            init_proportions = init_proportions / init_proportions.sum()
        
        # Convert proportions to logits (inverse softmax for 4 free params)
        # We use first 4 proportions and compute 5th as 1 - sum
        logits = np.log(init_proportions[:4])
        
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa)] + 
                logits.tolist() +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100))] +        # kappa
                [(-10, 10)] * 4 +                      # logits for proportions
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array(
                [np.log(init_kappa)] + 
                logits.tolist() +
                [np.log(1.0)]
            )
            bounds = (
                [(np.log(0.1), np.log(100))] +        # kappa
                [(-10, 10)] * 4 +                      # logits for proportions
                [(np.log(0.01), np.log(100))]         # scale
            )
        
        print(f"Starting M4 optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}")
        print(f"Initial proportions: {[f'{p:.4f}' for p in init_proportions]}")
        print(f"Fixed omegas: [0.0, 0.333, 0.667, 1.0, 3.0]")
        
        self.history = []
        
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        opt_kappa = np.exp(result.x[0])
        
        # Extract optimized proportions
        logits = result.x[1:5]
        exp_logits = np.exp(logits - np.max(logits))
        proportions_raw = np.concatenate([exp_logits, [1.0]])
        opt_proportions = (proportions_raw / proportions_raw.sum()).tolist()
        
        opt_log_likelihood = -result.fun
        
        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}")
        print(f"Final proportions: {[f'{p:.4f}' for p in opt_proportions]}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")
        
        return opt_kappa, opt_proportions, opt_log_likelihood


class M6Optimizer:
    """
    Optimizer for M6 (2gamma) codon model.
    
    Optimizes kappa, p0, alpha1, beta1, alpha2 (where alpha2 = beta2).
    """
    
    def __init__(
        self,
        alignment: Alignment,
        tree: Tree,
        ncatG: int = 10,
        use_f3x4: bool = True,
        optimize_branch_lengths: bool = True
    ):
        """Initialize M6 optimizer."""
        self.alignment = alignment
        self.tree = tree
        self.ncatG = ncatG
        self.optimize_branch_lengths = optimize_branch_lengths
        
        # Compute codon frequencies
        from ..models.codon import compute_codon_frequencies_f3x4
        if use_f3x4:
            self.pi = compute_codon_frequencies_f3x4(alignment)
        else:
            self.pi = np.ones(61) / 61
        
        # Get branch nodes if optimizing branch lengths
        if optimize_branch_lengths:
            self.branch_nodes = []
            for node in tree.traverse():
                if node.branch_length is not None:
                    self.branch_nodes.append(node)
            self.n_branches = len(self.branch_nodes)
        else:
            self.n_branches = 0
        
        # Create Rust likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)
    
    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Compute negative log-likelihood for optimization."""
        # Extract parameters (log-transformed)
        kappa = np.exp(params[0])
        p0 = 1.0 / (1.0 + np.exp(-params[1]))  # Sigmoid for [0,1]
        alpha1 = np.exp(params[2])
        beta1 = np.exp(params[3])
        alpha2 = np.exp(params[4])  # alpha2 = beta2
        
        # Update branch lengths if optimizing
        if self.optimize_branch_lengths:
            for i, node in enumerate(self.branch_nodes):
                node.branch_length = np.exp(params[5 + i])
            branch_scale = 1.0
        else:
            branch_scale = np.exp(params[5])
        
        # Create model and compute likelihood
        from ..models.codon import M6CodonModel
        model = M6CodonModel(
            kappa=kappa, p0=p0, alpha1=alpha1, beta1=beta1, alpha2=alpha2,
            ncatG=self.ncatG, pi=self.pi
        )
        
        # Compute likelihood
        try:
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()
            log_likelihood = self.calc.compute_log_likelihood_site_classes(
                Q_matrices, self.pi, proportions, scale_branch_lengths=branch_scale
            )
        except Exception as e:
            print(f"Error computing likelihood: {e}")
            return 1e10
        
        self.history.append({
            'kappa': kappa,
            'p0': p0,
            'alpha1': alpha1,
            'beta1': beta1,
            'alpha2': alpha2,
            'log_likelihood': log_likelihood
        })
        
        return -log_likelihood
    
    def optimize(
        self,
        init_kappa: float = 2.0,
        init_p0: float = 0.5,
        init_alpha1: float = 1.0,
        init_beta1: float = 1.0,
        init_alpha2: float = 1.0,
        method: str = 'L-BFGS-B',
        maxiter: int = 200
    ) -> Tuple[float, float, float, float, float, float]:
        """Optimize M6 parameters."""
        # Initial parameters
        if self.optimize_branch_lengths:
            init_branch_lengths = [node.branch_length for node in self.branch_nodes]
            init_params = np.array(
                [np.log(init_kappa),
                 np.log(init_p0 / (1 - init_p0)),  # Logit transform
                 np.log(init_alpha1),
                 np.log(init_beta1),
                 np.log(init_alpha2)] +
                [np.log(max(bl, 0.001)) for bl in init_branch_lengths]
            )
            bounds = (
                [(np.log(0.1), np.log(100)),       # kappa
                 (-10, 10),                         # p0 (logit)
                 (np.log(0.005), np.log(99)),       # alpha1
                 (np.log(0.005), np.log(99)),       # beta1
                 (np.log(0.005), np.log(99))] +     # alpha2
                [(np.log(0.0001), np.log(50))] * self.n_branches
            )
        else:
            init_params = np.array([
                np.log(init_kappa),
                np.log(init_p0 / (1 - init_p0)),
                np.log(init_alpha1),
                np.log(init_beta1),
                np.log(init_alpha2),
                np.log(1.0)
            ])
            bounds = [
                (np.log(0.1), np.log(100)),
                (-10, 10),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.005), np.log(99)),
                (np.log(0.01), np.log(100))
            ]
        
        print(f"Starting M6 optimization with method={method}, maxiter={maxiter}")
        print(f"Initial: kappa={init_kappa:.4f}, p0={init_p0:.4f}, alpha1={init_alpha1:.4f}, "
              f"beta1={init_beta1:.4f}, alpha2={init_alpha2:.4f}")
        print(f"Mixture of 2 gamma distributions discretized into {self.ncatG} categories")
        
        self.history = []
        
        result = minimize(
            self.compute_log_likelihood,
            init_params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter}
        )
        
        opt_kappa = np.exp(result.x[0])
        opt_p0 = 1.0 / (1.0 + np.exp(-result.x[1]))
        opt_alpha1 = np.exp(result.x[2])
        opt_beta1 = np.exp(result.x[3])
        opt_alpha2 = np.exp(result.x[4])
        opt_log_likelihood = -result.fun
        
        print(f"\nOptimization complete!")
        print(f"Final: kappa={opt_kappa:.4f}, p0={opt_p0:.4f}, alpha1={opt_alpha1:.4f}, "
              f"beta1={opt_beta1:.4f}, alpha2={opt_alpha2:.4f}")
        print(f"Log-likelihood: {opt_log_likelihood:.6f}")
        print(f"Iterations: {len(self.history)}")
        
        return opt_kappa, opt_p0, opt_alpha1, opt_beta1, opt_alpha2, opt_log_likelihood
