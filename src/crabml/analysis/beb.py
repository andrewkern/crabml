"""
Bayes Empirical Bayes (BEB) analysis for site-specific selection inference.

Implements the algorithm from Yang, Wong & Nielsen (2005) MBE 22:1107-1118.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import multivariate_normal
from itertools import product

from ..io.sequences import Alignment
from ..io.trees import Tree
from ..models.codon import compute_codon_frequencies_f3x4
from ..core.likelihood_rust import RustLikelihoodCalculator


@dataclass
class BEBResult:
    """
    Results from Bayes Empirical Bayes analysis.

    Attributes
    ----------
    site_numbers : np.ndarray, shape (n_sites,)
        Site positions (1-indexed)
    posterior_probs : np.ndarray, shape (n_sites, n_classes)
        Posterior probability that each site belongs to each class
    posterior_omega : np.ndarray, shape (n_sites,)
        Mean posterior ω for each site
    posterior_omega_se : np.ndarray, shape (n_sites,)
        Standard error of posterior ω
    model_name : str
        Name of the model used (M2a, M8, etc.)
    site_classes : List[str]
        Names of site classes
    class_omegas : np.ndarray
        ω values for each class
    class_proportions : np.ndarray
        Proportion of sites in each class (from MLE)
    """

    site_numbers: np.ndarray
    posterior_probs: np.ndarray
    posterior_omega: np.ndarray
    posterior_omega_se: np.ndarray
    model_name: str
    site_classes: List[str]
    class_omegas: np.ndarray
    class_proportions: np.ndarray

    def significant_sites(
        self,
        threshold: float = 0.95,
        class_idx: int = -1
    ) -> np.ndarray:
        """
        Return sites with posterior probability > threshold.

        Parameters
        ----------
        threshold : float, default=0.95
            Posterior probability threshold
        class_idx : int, default=-1
            Index of site class to test (default: last class, usually ω>1)

        Returns
        -------
        np.ndarray
            Site numbers (1-indexed) with P > threshold
        """
        mask = self.posterior_probs[:, class_idx] > threshold
        return self.site_numbers[mask]

    def summary(self, threshold_95: float = 0.95, threshold_99: float = 0.99) -> str:
        """
        Generate formatted summary matching PAML output style.

        Returns
        -------
        str
            Formatted summary with significant sites
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Bayes Empirical Bayes (BEB) Analysis")
        lines.append("Yang, Wong & Nielsen (2005) Mol. Biol. Evol. 22:1107-1118")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Model: {self.model_name}")
        lines.append(f"Site classes: {', '.join(self.site_classes)}")
        lines.append("")

        # Count significant sites
        sig_95 = len(self.significant_sites(threshold_95))
        sig_99 = len(self.significant_sites(threshold_99))

        lines.append(f"Positively selected sites (*: P>{threshold_95}; **: P>{threshold_99})")
        lines.append(f"  {sig_95} sites with P > {threshold_95}")
        lines.append(f"  {sig_99} sites with P > {threshold_99}")
        lines.append("")

        if sig_95 > 0:
            lines.append(f"{'Site':>6}  {'P(ω>1)':>8}  {'Mean ω':>8}  {'±SE':>8}")
            lines.append("-" * 40)

            # Show sites sorted by posterior probability (descending)
            indices = np.argsort(self.posterior_probs[:, -1])[::-1]

            for idx in indices:
                site_num = self.site_numbers[idx]
                prob = self.posterior_probs[idx, -1]
                omega = self.posterior_omega[idx]
                se = self.posterior_omega_se[idx]

                if prob >= threshold_95:
                    marker = "**" if prob >= threshold_99 else "*"
                    lines.append(
                        f"{site_num:6d}  {prob:8.3f}{marker:2s}  "
                        f"{omega:8.3f}  ±{se:6.3f}"
                    )
        else:
            lines.append("No sites with significant evidence for positive selection.")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


class BEBCalculator:
    """
    Bayes Empirical Bayes calculator for site-specific selection analysis.

    Implements the algorithm from Yang et al. (2005) MBE 22:1107-1118.

    Parameters
    ----------
    mle_params : Dict[str, float]
        Maximum likelihood estimates of model parameters
    alignment : Alignment
        Sequence alignment
    tree : Tree
        Phylogenetic tree
    model_class : class
        Model class (M2aCodonModel or M8CodonModel)
    param_names : List[str]
        Names of parameters to integrate over
    fix_branch_lengths : bool, default=True
        Whether to fix branch lengths during BEB (recommended)
    n_grid_points : int, default=5
        Number of grid points per dimension for integration (adaptive grid only)
    grid_type : str, default='adaptive'
        Grid construction method:
        - 'adaptive': Eigendecomposition-based adaptive grid (default)
        - 'paml': Fixed uniform grids matching PAML exactly
    """

    def __init__(
        self,
        mle_params: Dict[str, float],
        alignment: Alignment,
        tree: Tree,
        model_class,
        param_names: List[str],
        fix_branch_lengths: bool = True,
        n_grid_points: int = 5,
        grid_type: str = 'adaptive'
    ):
        self.mle_params = mle_params
        self.alignment = alignment
        self.tree = tree
        self.model_class = model_class
        self.param_names = param_names
        self.fix_branch_lengths = fix_branch_lengths
        self.n_grid_points = n_grid_points
        self.grid_type = grid_type  # 'adaptive' or 'paml'

        # Pre-compute codon frequencies
        self.pi = compute_codon_frequencies_f3x4(alignment)

        # Create likelihood calculator
        self.calc = RustLikelihoodCalculator(alignment, tree)

        # Create model instance at MLE
        self.mle_model = self.model_class(**mle_params, pi=self.pi)

    def _params_to_array(self, params_dict: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array for optimization."""
        return np.array([params_dict[name] for name in self.param_names])

    def _array_to_params(self, params_array: np.ndarray) -> Dict[str, float]:
        """Convert parameter array back to dict."""
        params = self.mle_params.copy()
        for i, name in enumerate(self.param_names):
            params[name] = params_array[i]
        return params

    def _check_bounds(self, params_array: np.ndarray) -> bool:
        """
        Check if parameters are within valid bounds.

        Returns
        -------
        bool
            True if all parameters are valid
        """
        params = self._array_to_params(params_array)

        # Check kappa > 0
        if 'kappa' in params and params['kappa'] <= 0:
            return False

        # Check proportions in [0, 1]
        for key in params:
            if key.startswith('p') and not key.endswith('beta'):
                if params[key] < 0 or params[key] > 1:
                    return False

        # Check beta parameters > 0
        for key in ['p_beta', 'q_beta']:
            if key in params and params[key] <= 0:
                return False

        # Check omega values >= 0
        for key in params:
            if key.startswith('omega') and params[key] < 0:
                return False

        return True

    def compute_hessian(self) -> np.ndarray:
        """
        Compute Hessian matrix at MLE using numerical differentiation.

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Hessian matrix (second derivatives)

        Notes
        -----
        Uses central finite differences for stability.
        """
        def neg_log_likelihood(params_array):
            """Negative log-likelihood for a parameter vector."""
            if not self._check_bounds(params_array):
                return np.inf

            params_dict = self._array_to_params(params_array)

            try:
                model = self.model_class(**params_dict, pi=self.pi)
                Q_matrices = model.get_Q_matrices()
                proportions, _ = model.get_site_classes()
                lnL = self.calc.compute_log_likelihood_site_classes(
                    Q_matrices, self.pi, proportions
                )
                return -lnL
            except Exception:
                return np.inf

        # Convert MLE to array
        param_array = self._params_to_array(self.mle_params)
        n_params = len(param_array)

        # Compute Hessian using finite differences
        hessian = np.zeros((n_params, n_params))
        h = 1e-5  # Step size

        print(f"    Computing Hessian: {n_params} × {n_params} parameters...")

        for i in range(n_params):
            for j in range(i, n_params):
                # Central difference approximation
                # ∂²f/∂xi∂xj ≈ [f(x+hi+hj) - f(x+hi) - f(x+hj) + f(x)] / (hi*hj)

                # Create perturbation vectors
                ei = np.zeros(n_params)
                ei[i] = h
                ej = np.zeros(n_params)
                ej[j] = h

                # Evaluate at four points
                f_00 = neg_log_likelihood(param_array)
                f_10 = neg_log_likelihood(param_array + ei)
                f_01 = neg_log_likelihood(param_array + ej)
                f_11 = neg_log_likelihood(param_array + ei + ej)

                # Central difference
                hessian[i, j] = (f_11 - f_10 - f_01 + f_00) / (h * h)
                hessian[j, i] = hessian[i, j]  # Symmetric

        return hessian

    def setup_integration_grid(
        self,
        hessian: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Set up grid of parameter values for numerical integration.

        Uses eigendecomposition of inverse Hessian to find principal
        axes of parameter uncertainty, then samples along these axes.

        Parameters
        ----------
        hessian : np.ndarray
            Hessian matrix at MLE

        Returns
        -------
        grid_points : List[np.ndarray]
            List of parameter vectors for grid points
        weights : List[float]
            Prior probability weights for each grid point
        """
        # Regularize Hessian if needed
        try:
            # Covariance = inverse Hessian
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # Singular - use pseudoinverse with regularization
            print("    Warning: Hessian is singular, using regularized inverse")
            eigenvals, eigenvecs = np.linalg.eigh(hessian)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Regularize
            covariance = eigenvecs @ np.diag(1/eigenvals) @ eigenvecs.T

        # Eigendecomposition for principal axes
        eigenvals, eigenvecs = np.linalg.eigh(covariance)

        # Ensure positive semidefinite by clipping negative eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-10)
        std_devs = np.sqrt(eigenvals)

        # Reconstruct positive semidefinite covariance
        covariance = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Create grid along each axis
        # For n_grid_points=5: [-2σ, -σ, 0, +σ, +2σ]
        n = self.n_grid_points
        if n == 5:
            offsets = [-2, -1, 0, 1, 2]
        elif n == 3:
            offsets = [-1, 0, 1]
        else:
            # General case: equally spaced from -2σ to +2σ
            offsets = np.linspace(-2, 2, n)

        # Generate all combinations (Cartesian product)
        param_array = self._params_to_array(self.mle_params)
        n_params = len(param_array)

        grid_points = []
        weights = []

        # Create multivariate normal for weighting
        prior = multivariate_normal(mean=param_array, cov=covariance)

        for offset_combo in product(offsets, repeat=n_params):
            # Compute point in parameter space
            offset_vector = np.zeros(n_params)
            for i, offset in enumerate(offset_combo):
                offset_vector += offset * std_devs[i] * eigenvecs[:, i]

            point = param_array + offset_vector

            # Check parameter bounds (all parameters must be valid)
            if self._check_bounds(point):
                grid_points.append(point)
                # Weight by prior probability
                weights.append(prior.pdf(point))

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Fallback: uniform weights
            weights = [1.0 / len(grid_points)] * len(grid_points)

        print(f"    BEB grid: {len(grid_points)} valid points "
              f"(from {n**n_params} possible)")

        return grid_points, weights

    def _get_index_ternary(self, itriangle: int, K: int) -> Tuple[int, int, float, float]:
        """
        PAML's GetIndexTernary function for triangular grid mapping.

        Maps triangle index to (p0, p1) coordinates on a ternary/simplex grid.
        From PAML's tools.c lines 4617-4637.

        The ternary graph (0-1 on each axis) is partitioned into K*K equal-sized
        triangles:
        - Row ix=0: 1 triangle (iy=0)
        - Row ix=1: 3 triangles (iy=0,1,2)
        - Row ix=i: 2*i+1 triangles (iy=0,1,...,2*i)

        Parameters
        ----------
        itriangle : int
            Triangle index from 0 to K*K-1
        K : int
            Grid resolution (typically 10)

        Returns
        -------
        ix : int
            Row index
        iy : int
            Column index within row
        x : float
            p0 coordinate (centroid x-coordinate)
        y : float
            p1 coordinate (centroid y-coordinate)
        """
        ix = int(np.sqrt(itriangle))
        iy = itriangle - ix**2

        x = (1 + (iy // 2) * 3 + (iy % 2)) / (3.0 * K)
        y = (1 + (K - 1 - ix) * 3 + (iy % 2)) / (3.0 * K)

        return ix, iy, x, y

    def setup_paml_grid(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Set up grid matching PAML's exact BEB algorithm with triangular grid.

        PAML uses a triangular grid (ternary mode) for the (p0, p1) plane,
        ensuring uniform coverage of the simplex {p0+p1+p2=1, all ≥ 0}.

        For M2a model:
        - (p0, p1): Triangular grid with K*K points (100 for K=10)
        - omega0: 10 points from 0.05 to 0.95
        - omega2: 10 points from 1.5 to 10.5
        - kappa: Fixed at MLE (not integrated)

        Total grid: 100 × 10 × 10 = 10,000 points

        Returns
        -------
        grid_points : List[np.ndarray]
            List of parameter vectors for grid points
        weights : List[float]
            Prior probability weights for each grid point (uniform)
        """
        param_array = self._params_to_array(self.mle_params)
        param_dict = dict(zip(self.param_names, param_array))

        # PAML uses K=10 for grid resolution
        n1d = 10

        # omega0: 10 bin midpoints in [0, 1]
        w0_grid = [(i + 0.5) / n1d for i in range(n1d)] if 'omega0' in param_dict else [None]

        # omega2: 10 bin midpoints in [1, 11]
        w2_grid = [1.0 + (i + 0.5) * 10.0 / n1d for i in range(n1d)] if 'omega2' in param_dict else [None]

        # kappa: Fixed at MLE
        kappa_val = param_dict.get('kappa', None)

        grid_points = []
        weights = []

        # Iterate over triangular grid for (p0, p1)
        # Total: n1d * n1d = 100 triangles
        for itriangle in range(n1d * n1d):
            ix, iy, p0, p1 = self._get_index_ternary(itriangle, n1d)
            p2 = 1.0 - p0 - p1  # Automatically valid by construction

            # For each (p0, p1, p2), iterate over omega0 and omega2
            for w0 in w0_grid:
                for w2 in w2_grid:
                    # Build parameter array in correct order
                    param_values = []
                    for name in self.param_names:
                        if name == 'kappa':
                            param_values.append(kappa_val)
                        elif name == 'p0':
                            param_values.append(p0)
                        elif name == 'p1':
                            param_values.append(p1)
                        elif name == 'omega0':
                            param_values.append(w0)
                        elif name == 'omega2':
                            param_values.append(w2)
                        else:
                            # For other parameters, use MLE value
                            param_values.append(param_dict[name])

                    point = np.array(param_values)

                    # Bounds checking
                    if self._check_bounds(point):
                        grid_points.append(point)
                        # PAML uses uniform prior
                        weights.append(1.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(grid_points)] * len(grid_points)

        print(f"    PAML-style triangular grid: {len(grid_points)} points")

        return grid_points, weights

    def compute_site_posteriors_paml(
        self,
        grid_points: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Compute posterior probabilities using PAML's exact BEB algorithm.

        PAML's approach (for M2a):
        1. Compute Q matrices ONCE for 21 fixed omega values
        2. Compute site likelihoods fhK[h, k] for each site h and omega k
        3. Integrate over grid of (p0, p1, w0_idx, w2_idx) parameters

        This is much more efficient than recomputing Q matrices for each grid point.

        Parameters
        ----------
        grid_points : List[np.ndarray]
            Parameter values (p0, p1, omega0, omega2) for integration
        weights : List[float]
            Prior weights for each grid point (uniform for PAML)

        Returns
        -------
        np.ndarray, shape (n_sites, n_classes)
            Posterior probabilities
        """
        from ..models.codon import build_codon_Q_matrix

        n_sites = self.alignment.n_sites
        n_classes = 3  # M2a has 3 classes

        print("    Computing Qfactor_NS from MLE...")
        # Step 1: Compute Qfactor_NS from MLE (weighted-average normalization)
        proportions_mle, omegas_mle = self.mle_model.get_site_classes()

        Qfactor_NS_sum = 0.0
        for omega, prop in zip(omegas_mle, proportions_mle):
            Q_unnorm = build_codon_Q_matrix(
                self.mle_params['kappa'], omega, self.pi, normalization_factor=1.0
            )
            mr = -np.dot(self.pi, np.diag(Q_unnorm))
            Qfactor_NS_sum += prop * mr

        Qfactor_NS = 1.0 / Qfactor_NS_sum
        print(f"      Qfactor_NS = {Qfactor_NS:.6f}")

        # Step 2: Build Q matrices for 21 BEB omega values
        print("    Building Q matrices for 21 omega values...")
        n1d = 10
        w0_grid = [(i + 0.5) / n1d for i in range(n1d)]
        w2_grid = [1.0 + (i + 0.5) * 10.0 / n1d for i in range(n1d)]
        omega_values = w0_grid + [1.0] + w2_grid  # 21 values

        Q_matrices = []
        for omega in omega_values:
            Q_unnorm = build_codon_Q_matrix(
                self.mle_params['kappa'], omega, self.pi, normalization_factor=1.0
            )
            Q_scaled = Q_unnorm * Qfactor_NS
            Q_matrices.append(Q_scaled)

        # Step 3: Compute site likelihoods fhK[h, k]
        print("    Computing site likelihoods...")
        site_log_liks = self.calc.compute_site_log_likelihoods(
            Q_matrices, self.pi, [1.0/21] * 21
        )

        # Normalize per site (PAML lines 6287-6302)
        fhK = np.zeros_like(site_log_liks)
        for h in range(n_sites):
            max_log_lik = np.max(site_log_liks[h, :])
            fhK[h, :] = np.exp(site_log_liks[h, :] - max_log_lik)

        # Step 4: Compute lnfXs for each grid point
        print(f"    Computing marginal likelihoods for {len(grid_points)} grid points...")
        lnfXs = np.zeros(len(grid_points))

        for igrid, (point, weight) in enumerate(zip(grid_points, weights)):
            params = self._array_to_params(point)
            p0 = params['p0']
            p1 = params['p1']
            p2 = 1.0 - p0 - p1

            # Find omega indices
            w0 = params['omega0']
            w2 = params['omega2']
            w0_idx = w0_grid.index(w0)
            w2_idx = 11 + w2_grid.index(w2)
            w1_idx = 10  # omega=1.0

            for h in range(n_sites):
                fh = p0 * fhK[h, w0_idx] + p1 * fhK[h, w1_idx] + p2 * fhK[h, w2_idx]
                if fh > 1e-300:
                    lnfXs[igrid] += np.log(fh)

        # Step 5: Compute fX
        S2 = np.max(lnfXs)
        fX = np.log(np.sum(np.exp(lnfXs - S2))) + S2
        print(f"      log(fX) = {fX:.6f}")

        # Step 6: Compute posteriors for each site
        print("    Computing site posteriors...")
        posteriors = np.zeros((n_sites, n_classes))

        for h in range(n_sites):
            post_site = np.zeros(n_classes)
            S1 = -1e300

            for igrid, point in enumerate(grid_points):
                params = self._array_to_params(point)
                p0 = params['p0']
                p1 = params['p1']
                p2 = 1.0 - p0 - p1

                w0 = params['omega0']
                w2 = params['omega2']
                w0_idx = w0_grid.index(w0)
                w2_idx = 11 + w2_grid.index(w2)
                w1_idx = 10

                fh = p0 * fhK[h, w0_idx] + p1 * fhK[h, w1_idx] + p2 * fhK[h, w2_idx]
                if fh < 1e-300:
                    continue

                for iclass in range(n_classes):
                    if iclass == 0:
                        fh1site = p0 * fhK[h, w0_idx]
                    elif iclass == 1:
                        fh1site = p1 * fhK[h, w1_idx]
                    else:
                        fh1site = p2 * fhK[h, w2_idx]

                    if fh1site < 1e-300:
                        continue

                    fh1site /= fh
                    t = np.log(fh1site) + lnfXs[igrid]

                    if t > S1:
                        post_site *= np.exp(S1 - t)
                        S1 = t

                    post_site[iclass] += np.exp(t - S1)

            post_site *= np.exp(S1 - fX)
            posteriors[h, :] = post_site

        return posteriors

    def compute_site_posteriors(
        self,
        grid_points: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Compute posterior probabilities for each site.

        For each site i and class k:
            P(site i in class k | data) =
                Σ_θ P(site i in class k | data, θ) * P(θ | data)

        where the sum is over grid points θ with weights P(θ | data).

        Parameters
        ----------
        grid_points : List[np.ndarray]
            Parameter values for integration
        weights : List[float]
            Prior weights for each grid point

        Returns
        -------
        np.ndarray, shape (n_sites, n_classes)
            Posterior probabilities
        """
        n_sites = self.alignment.n_sites

        # Determine number of classes from model
        proportions, omegas = self.mle_model.get_site_classes()
        n_classes = len(proportions)

        # Initialize posterior accumulator
        posteriors = np.zeros((n_sites, n_classes))

        # For each grid point
        print(f"    Evaluating {len(grid_points)} grid points...")
        for point_idx, (point, weight) in enumerate(zip(grid_points, weights)):
            if (point_idx + 1) % 100 == 0:
                print(f"      Progress: {point_idx + 1}/{len(grid_points)}")

            params = self._array_to_params(point)

            # Create model at this parameter value
            model = self.model_class(**params, pi=self.pi)
            Q_matrices = model.get_Q_matrices()
            proportions, _ = model.get_site_classes()

            # Get site-specific likelihoods [n_sites × n_classes]
            site_log_liks = self.calc.compute_site_log_likelihoods(
                Q_matrices, self.pi, proportions
            )

            # Convert to probabilities and weight by class proportions
            # P(site i in class k | params) ∝ P(data_i | class k) * P(class k)
            site_liks = np.exp(site_log_liks)

            for k in range(n_classes):
                site_liks[:, k] *= proportions[k]

            # Normalize across classes (for this parameter value)
            row_sums = site_liks.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-300)  # Prevent division by zero
            class_posteriors_given_params = site_liks / row_sums

            # Add weighted contribution to overall posterior
            posteriors += weight * class_posteriors_given_params

        return posteriors

    def calculate_beb(self) -> BEBResult:
        """
        Main BEB calculation pipeline.

        Returns
        -------
        BEBResult
            Complete BEB results with posterior probabilities
        """
        print("Computing Bayes Empirical Bayes analysis...")
        print(f"  Alignment: {self.alignment.n_species} sequences, "
              f"{self.alignment.n_sites} sites")

        # Step 1: Compute Hessian (needed for adaptive grid)
        if self.grid_type == 'adaptive':
            print("  Step 1: Computing Hessian matrix...")
            hessian = self.compute_hessian()
            n_params = hessian.shape[0]

            # Step 2: Set up integration grid
            print(f"  Step 2: Setting up integration grid "
                  f"({self.n_grid_points} points per dimension)...")
            grid_points, weights = self.setup_integration_grid(hessian)
        elif self.grid_type == 'paml':
            # Use PAML's fixed grid (doesn't need Hessian)
            print("  Step 1: Setting up PAML-style fixed grid...")
            grid_points, weights = self.setup_paml_grid()
        else:
            raise ValueError(f"Unknown grid_type: {self.grid_type}. "
                           "Must be 'adaptive' or 'paml'.")

        # Step 3: Compute posteriors
        print(f"  Step 3: Computing posterior probabilities...")
        if self.grid_type == 'paml':
            # Use PAML-specific algorithm (more efficient)
            posteriors = self.compute_site_posteriors_paml(grid_points, weights)
        else:
            # Use general adaptive grid algorithm
            posteriors = self.compute_site_posteriors(grid_points, weights)

        # Step 4: Compute posterior omega values
        print("  Step 4: Computing posterior ω statistics...")
        proportions, omegas = self.mle_model.get_site_classes()

        # Mean posterior ω for each site
        posterior_omega = posteriors @ np.array(omegas)

        # SE of posterior ω
        posterior_omega_sq = posteriors @ np.array([w**2 for w in omegas])
        posterior_omega_se = np.sqrt(
            np.maximum(posterior_omega_sq - posterior_omega**2, 0)
        )

        # Create result object
        result = BEBResult(
            site_numbers=np.arange(1, self.alignment.n_sites + 1),
            posterior_probs=posteriors,
            posterior_omega=posterior_omega,
            posterior_omega_se=posterior_omega_se,
            model_name=self.model_class.__name__,
            site_classes=[f"ω={w:.2f}" for w in omegas],
            class_omegas=np.array(omegas),
            class_proportions=np.array(proportions)
        )

        print("  BEB analysis complete!")
        return result
