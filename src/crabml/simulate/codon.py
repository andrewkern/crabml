"""
Codon sequence simulators.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from scipy.stats import beta as beta_dist

from .base import SequenceSimulator
from ..models.codon import build_codon_Q_matrix
from ..core.matrix import eigen_decompose_rev
from ..io.trees import Tree


class M0CodonSimulator(SequenceSimulator):
    """
    Simulate codon sequences under M0 model (single omega).

    The M0 model has a single dN/dS (omega) parameter for all sites and branches.
    Substitution rates depend on:
    - kappa: transition/transversion ratio
    - omega: dN/dS ratio
    - codon_freqs: equilibrium codon frequencies

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree (rooted, with branch lengths)
    sequence_length : int
        Number of codons to simulate
    kappa : float
        Transition/transversion ratio (default 2.0)
    omega : float
        dN/dS ratio (default 0.4)
    codon_freqs : np.ndarray, shape (61,)
        Codon equilibrium frequencies (must sum to 1)
    genetic_code : int, default=0
        Genetic code table (0=universal)
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> from crabml.io.trees import Tree
    >>> import numpy as np
    >>> tree = Tree.from_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.05);")
    >>> sim = M0CodonSimulator(
    ...     tree=tree,
    ...     sequence_length=1000,
    ...     kappa=2.5,
    ...     omega=0.3,
    ...     codon_freqs=np.ones(61) / 61,
    ...     seed=42
    ... )
    >>> sequences = sim.simulate()
    >>> len(sequences)  # 4 species
    4
    >>> sequences['A'].shape  # 1000 codons
    (1000,)
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        omega: float,
        codon_freqs: np.ndarray,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        super().__init__(tree, sequence_length, seed)

        self.kappa = kappa
        self.omega = omega
        self.codon_freqs = codon_freqs
        self.genetic_code = genetic_code

        # Validate codon frequencies
        if len(codon_freqs) != 61:
            raise ValueError(f"codon_freqs must have length 61, got {len(codon_freqs)}")
        if not np.isclose(codon_freqs.sum(), 1.0):
            raise ValueError(f"codon_freqs must sum to 1, got {codon_freqs.sum()}")

        # Build Q matrix and precompute eigendecomposition (ONCE)
        self._setup_substitution_model()

    def _setup_substitution_model(self):
        """
        Build Q matrix and compute eigendecomposition.

        This is done once during initialization and reused for all branches.
        Eigendecomposition allows fast P(t) computation via:
            P(t) = U @ diag(exp(eigenvalues * t)) @ V
        """
        # Build Q matrix (reuses existing optimized code!)
        Q = build_codon_Q_matrix(self.kappa, self.omega, self.codon_freqs)

        # Precompute eigendecomposition: Q = U @ diag(eigenvalues) @ V
        self.eigenvalues, self.U, self.V = eigen_decompose_rev(Q, self.codon_freqs)

    def _compute_transition_matrix(self, t: float) -> np.ndarray:
        """
        Compute P(t) = exp(Qt) using precomputed eigendecomposition.

        This is O(n²) instead of O(n³) for direct expm.

        Parameters
        ----------
        t : float
            Branch length

        Returns
        -------
        np.ndarray, shape (61, 61)
            Transition probability matrix
        """
        # P(t) = U @ diag(exp(eigenvalues * t)) @ V
        exp_eigvals = np.exp(self.eigenvalues * t)
        P = self.U @ np.diag(exp_eigvals) @ self.V

        # Handle numerical precision issues
        # Ensure all probabilities are non-negative (can get tiny negative values from floating point errors)
        P = np.maximum(P, 0.0)

        # Ensure rows sum to 1
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / row_sums

        return P

    def _generate_ancestral_sequence(self) -> np.ndarray:
        """
        Sample ancestral sequence at root from equilibrium frequencies.

        Returns
        -------
        np.ndarray, shape (sequence_length,), dtype=uint8
            Sequence of codon indices (0-60)
        """
        return self.rng.choice(
            61,  # 61 non-stop codons
            size=self.sequence_length,
            p=self.codon_freqs
        ).astype(np.uint8)

    def _evolve_sequence(
        self,
        parent_seq: np.ndarray,
        branch_length: float,
        **kwargs
    ) -> np.ndarray:
        """
        Evolve sequence along a branch using P(t).

        For each site, sample child codon from transition probabilities:
            child_codon ~ Categorical(P[parent_codon, :])

        Parameters
        ----------
        parent_seq : np.ndarray
            Parent sequence (codon indices)
        branch_length : float
            Branch length

        Returns
        -------
        np.ndarray
            Child sequence (codon indices)
        """
        # Compute transition probability matrix for this branch
        P = self._compute_transition_matrix(branch_length)

        # Evolve each site independently
        child_seq = np.empty(self.sequence_length, dtype=np.uint8)

        for i, parent_codon in enumerate(parent_seq):
            # Sample child codon from P[parent_codon, :]
            child_seq[i] = self.rng.choice(61, p=P[parent_codon, :])

        return child_seq

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get simulation parameters for metadata output.

        Returns
        -------
        dict
            Model parameters including:
            - model: 'M0'
            - kappa: transition/transversion ratio
            - omega: dN/dS ratio
            - sequence_length: number of codons
            - genetic_code: genetic code table ID
        """
        return {
            'model': 'M0',
            'kappa': float(self.kappa),
            'omega': float(self.omega),
            'sequence_length': int(self.sequence_length),
            'genetic_code': int(self.genetic_code),
            'tree_length': sum(
                node.branch_length for node in self.tree.postorder()
                if node.parent is not None
            )
        }


class SiteClassCodonSimulator(SequenceSimulator):
    """
    Base class for site-class codon models (M1a, M2a, M7, M8).

    Site-class models allow omega (dN/dS) to vary across sites but not branches.
    Sites are assigned to classes, each with a specific omega value.

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree (rooted, with branch lengths)
    sequence_length : int
        Number of codons to simulate
    kappa : float
        Transition/transversion ratio
    site_class_freqs : np.ndarray
        Proportion of sites in each class (must sum to 1)
    site_class_omegas : np.ndarray
        Omega value for each site class
    codon_freqs : np.ndarray, shape (61,)
        Codon equilibrium frequencies
    genetic_code : int, default=0
        Genetic code table
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        site_class_freqs: np.ndarray,
        site_class_omegas: np.ndarray,
        codon_freqs: np.ndarray,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        super().__init__(tree, sequence_length, seed)

        self.kappa = kappa
        self.site_class_freqs = site_class_freqs
        self.site_class_omegas = site_class_omegas
        self.codon_freqs = codon_freqs
        self.genetic_code = genetic_code

        # Validate
        if len(site_class_freqs) != len(site_class_omegas):
            raise ValueError("site_class_freqs and site_class_omegas must have same length")
        if not np.isclose(site_class_freqs.sum(), 1.0):
            raise ValueError(f"site_class_freqs must sum to 1, got {site_class_freqs.sum()}")
        if len(codon_freqs) != 61:
            raise ValueError(f"codon_freqs must have length 61, got {len(codon_freqs)}")

        self.n_classes = len(site_class_freqs)

        # Assign sites to classes (once!)
        self._assign_site_classes()

        # Build Q matrices for each class (once!)
        self._setup_substitution_models()

    def _assign_site_classes(self):
        """
        Assign each site to a class based on frequencies.

        Stores site_class_ids: array of class IDs (0 to n_classes-1) for each site.
        """
        self.site_class_ids = self.rng.choice(
            self.n_classes,
            size=self.sequence_length,
            p=self.site_class_freqs
        ).astype(np.uint8)

    def _setup_substitution_models(self):
        """Build Q matrix and eigendecomposition for EACH site class."""
        # Build Q matrices for each class
        Q_matrices = [
            build_codon_Q_matrix(self.kappa, omega, self.codon_freqs)
            for omega in self.site_class_omegas
        ]

        # Precompute eigendecomposition for each class
        self.eigenvalues_list = []
        self.U_list = []
        self.V_list = []

        for Q in Q_matrices:
            eigenvalues, U, V = eigen_decompose_rev(Q, self.codon_freqs)
            self.eigenvalues_list.append(eigenvalues)
            self.U_list.append(U)
            self.V_list.append(V)

    def _compute_transition_matrices(self, t: float) -> List[np.ndarray]:
        """
        Compute P(t) for each site class.

        Returns
        -------
        list of np.ndarray
            One P(t) matrix per site class
        """
        P_matrices = []

        for eigenvalues, U, V in zip(self.eigenvalues_list, self.U_list, self.V_list):
            exp_eigvals = np.exp(eigenvalues * t)
            P = U @ np.diag(exp_eigvals) @ V

            # Handle numerical precision
            P = np.maximum(P, 0.0)
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / row_sums

            P_matrices.append(P)

        return P_matrices

    def _generate_ancestral_sequence(self) -> np.ndarray:
        """Sample from equilibrium (same for all classes)."""
        return self.rng.choice(
            61,
            size=self.sequence_length,
            p=self.codon_freqs
        ).astype(np.uint8)

    def _evolve_sequence(
        self,
        parent_seq: np.ndarray,
        branch_length: float,
        **kwargs
    ) -> np.ndarray:
        """
        Evolve sequence with site-specific omegas.

        Each site uses its assigned class's P matrix.
        """
        child_seq = np.empty(self.sequence_length, dtype=np.uint8)

        # Compute P(t) for each class (cache within this branch)
        P_matrices = self._compute_transition_matrices(branch_length)

        # Evolve each site with its class-specific P matrix
        for site_idx in range(self.sequence_length):
            class_id = self.site_class_ids[site_idx]
            parent_codon = parent_seq[site_idx]
            P = P_matrices[class_id]

            # Sample child codon
            child_seq[site_idx] = self.rng.choice(61, p=P[parent_codon, :])

        return child_seq

    def get_site_classes(self) -> Dict[str, Any]:
        """
        Get site class assignments for output.

        Returns
        -------
        dict
            Dictionary with:
            - site_class_ids: class ID for each site
            - site_class_freqs: proportion of each class
            - site_class_omegas: omega for each class
            - positively_selected_sites: sites with omega > 1
        """
        # Find positively selected sites
        ps_sites = []
        for site_idx, class_id in enumerate(self.site_class_ids):
            omega = self.site_class_omegas[int(class_id)]
            if omega > 1:
                ps_sites.append(site_idx)

        return {
            'site_class_ids': self.site_class_ids.tolist(),
            'site_class_freqs': self.site_class_freqs.tolist(),
            'site_class_omegas': self.site_class_omegas.tolist(),
            'positively_selected_sites': ps_sites
        }


class M1aSimulator(SiteClassCodonSimulator):
    """
    M1a (Nearly Neutral) model: 2 site classes.

    - Class 0: omega_0 < 1 (purifying selection), proportion p_0
    - Class 1: omega_1 = 1 (neutral), proportion 1 - p_0

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree
    sequence_length : int
        Number of codons
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion in purifying class (0 < p0 < 1)
    omega0 : float
        dN/dS for purifying class (must be < 1)
    codon_freqs : np.ndarray
        Codon frequencies
    genetic_code : int, default=0
        Genetic code table
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        p0: float,
        omega0: float,
        codon_freqs: np.ndarray,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        if not (0 < p0 < 1):
            raise ValueError(f"p0 must be in (0, 1), got {p0}")
        if omega0 >= 1:
            raise ValueError(f"omega0 must be < 1 for M1a, got {omega0}")

        site_class_freqs = np.array([p0, 1 - p0])
        site_class_omegas = np.array([omega0, 1.0])

        super().__init__(
            tree, sequence_length, kappa,
            site_class_freqs, site_class_omegas,
            codon_freqs, genetic_code, seed
        )

        self.p0 = p0
        self.omega0 = omega0

    def get_parameters(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        return {
            'model': 'M1a',
            'kappa': float(self.kappa),
            'p0': float(self.p0),
            'omega0': float(self.omega0),
            'sequence_length': int(self.sequence_length),
            'genetic_code': int(self.genetic_code),
            'tree_length': sum(
                node.branch_length for node in self.tree.postorder()
                if node.parent is not None
            )
        }


class M2aSimulator(SiteClassCodonSimulator):
    """
    M2a (Positive Selection) model: 3 site classes.

    - Class 0: omega_0 < 1 (purifying), proportion p_0
    - Class 1: omega_1 = 1 (neutral), proportion p_1
    - Class 2: omega_2 > 1 (positive selection), proportion p_2 = 1 - p_0 - p_1

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree
    sequence_length : int
        Number of codons
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion in purifying class
    p1 : float
        Proportion in neutral class (p2 = 1 - p0 - p1)
    omega0 : float
        dN/dS for purifying class (< 1)
    omega2 : float
        dN/dS for positive selection class (> 1)
    codon_freqs : np.ndarray
        Codon frequencies
    genetic_code : int, default=0
        Genetic code table
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        p0: float,
        p1: float,
        omega0: float,
        omega2: float,
        codon_freqs: np.ndarray,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        if not (0 < p0 + p1 < 1):
            raise ValueError(f"p0 + p1 must be in (0, 1), got {p0 + p1}")
        if omega0 >= 1:
            raise ValueError(f"omega0 must be < 1, got {omega0}")
        if omega2 <= 1:
            raise ValueError(f"omega2 must be > 1 for M2a, got {omega2}")

        p2 = 1 - p0 - p1
        site_class_freqs = np.array([p0, p1, p2])
        site_class_omegas = np.array([omega0, 1.0, omega2])

        super().__init__(
            tree, sequence_length, kappa,
            site_class_freqs, site_class_omegas,
            codon_freqs, genetic_code, seed
        )

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.omega0 = omega0
        self.omega2 = omega2

    def get_parameters(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        return {
            'model': 'M2a',
            'kappa': float(self.kappa),
            'p0': float(self.p0),
            'p1': float(self.p1),
            'p2': float(self.p2),
            'omega0': float(self.omega0),
            'omega2': float(self.omega2),
            'sequence_length': int(self.sequence_length),
            'genetic_code': int(self.genetic_code),
            'tree_length': sum(
                node.branch_length for node in self.tree.postorder()
                if node.parent is not None
            )
        }


class M7Simulator(SiteClassCodonSimulator):
    """
    M7 (Beta) model: Beta distribution for omega in (0, 1).

    The beta distribution is discretized into K categories.

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree
    sequence_length : int
        Number of codons
    kappa : float
        Transition/transversion ratio
    p : float
        Beta shape parameter p (alpha)
    q : float
        Beta shape parameter q (beta)
    n_categories : int, default=10
        Number of discrete categories for beta distribution
    codon_freqs : np.ndarray
        Codon frequencies
    genetic_code : int, default=0
        Genetic code table
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        p: float,
        q: float,
        n_categories: int = 10,
        codon_freqs: np.ndarray = None,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        if p <= 0 or q <= 0:
            raise ValueError(f"p and q must be > 0, got p={p}, q={q}")

        if codon_freqs is None:
            codon_freqs = np.ones(61) / 61

        # Discretize beta distribution
        proportions, omegas = self._discretize_beta(p, q, n_categories)

        super().__init__(
            tree, sequence_length, kappa,
            proportions, omegas,
            codon_freqs, genetic_code, seed
        )

        self.p = p
        self.q = q
        self.n_categories = n_categories

    @staticmethod
    def _discretize_beta(p: float, q: float, K: int) -> tuple:
        """
        Discretize beta(p, q) into K categories using equal quantiles.

        Returns
        -------
        tuple
            (proportions, omegas) as np.ndarray
        """
        # Equal proportions
        proportions = np.ones(K) / K

        # Omega values at quantile midpoints
        quantiles = (np.arange(K) + 0.5) / K
        omegas = beta_dist.ppf(quantiles, p, q)

        return proportions, omegas

    def get_parameters(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        return {
            'model': 'M7',
            'kappa': float(self.kappa),
            'p': float(self.p),
            'q': float(self.q),
            'n_categories': int(self.n_categories),
            'sequence_length': int(self.sequence_length),
            'genetic_code': int(self.genetic_code),
            'tree_length': sum(
                node.branch_length for node in self.tree.postorder()
                if node.parent is not None
            )
        }


class M8Simulator(SiteClassCodonSimulator):
    """
    M8 (Beta & omega) model: Beta distribution + positive selection class.

    - Classes 0-(K-1): Beta distribution (proportion p_0)
    - Class K: omega_s > 1 (positive selection), proportion 1 - p_0

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree
    sequence_length : int
        Number of codons
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion in beta distribution
    p : float
        Beta shape parameter p
    q : float
        Beta shape parameter q
    omega_s : float
        Omega for selection class (> 1)
    n_beta_categories : int, default=10
        Number of categories for beta distribution
    codon_freqs : np.ndarray
        Codon frequencies
    genetic_code : int, default=0
        Genetic code table
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        p0: float,
        p: float,
        q: float,
        omega_s: float,
        n_beta_categories: int = 10,
        codon_freqs: np.ndarray = None,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        if not (0 < p0 < 1):
            raise ValueError(f"p0 must be in (0, 1), got {p0}")
        if omega_s <= 1:
            raise ValueError(f"omega_s must be > 1 for M8, got {omega_s}")
        if p <= 0 or q <= 0:
            raise ValueError(f"p and q must be > 0, got p={p}, q={q}")

        if codon_freqs is None:
            codon_freqs = np.ones(61) / 61

        # Discretize beta distribution
        beta_props, beta_omegas = M7Simulator._discretize_beta(p, q, n_beta_categories)

        # Combine beta categories with selection category
        proportions = np.concatenate([beta_props * p0, [1 - p0]])
        omegas = np.concatenate([beta_omegas, [omega_s]])

        super().__init__(
            tree, sequence_length, kappa,
            proportions, omegas,
            codon_freqs, genetic_code, seed
        )

        self.p0 = p0
        self.p = p
        self.q = q
        self.omega_s = omega_s
        self.n_beta_categories = n_beta_categories

    def get_parameters(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        return {
            'model': 'M8',
            'kappa': float(self.kappa),
            'p0': float(self.p0),
            'p': float(self.p),
            'q': float(self.q),
            'omega_s': float(self.omega_s),
            'n_beta_categories': int(self.n_beta_categories),
            'sequence_length': int(self.sequence_length),
            'genetic_code': int(self.genetic_code),
            'tree_length': sum(
                node.branch_length for node in self.tree.postorder()
                if node.parent is not None
            )
        }


class M8aSimulator(SiteClassCodonSimulator):
    """
    M8a (Beta & omega=1) model: Beta distribution + neutral class.

    This is the null model for M8a vs M8 test. It's identical to M8
    but with omega_s fixed to 1.0 (neutral).

    - Classes 0-(K-1): Beta distribution (proportion p_0)
    - Class K: omega_s = 1.0 (neutral), proportion 1 - p_0

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree
    sequence_length : int
        Number of codons
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion in beta distribution
    p : float
        Beta shape parameter p
    q : float
        Beta shape parameter q
    n_beta_categories : int, default=10
        Number of categories for beta distribution
    codon_freqs : np.ndarray
        Codon frequencies
    genetic_code : int, default=0
        Genetic code table
    seed : int, optional
        Random seed
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        kappa: float,
        p0: float,
        p: float,
        q: float,
        n_beta_categories: int = 10,
        codon_freqs: np.ndarray = None,
        genetic_code: int = 0,
        seed: Optional[int] = None
    ):
        if not (0 < p0 < 1):
            raise ValueError(f"p0 must be in (0, 1), got {p0}")
        if p <= 0 or q <= 0:
            raise ValueError(f"p and q must be > 0, got p={p}, q={q}")

        if codon_freqs is None:
            codon_freqs = np.ones(61) / 61

        # Discretize beta distribution
        beta_props, beta_omegas = M7Simulator._discretize_beta(p, q, n_beta_categories)

        # Combine beta categories with neutral category (omega_s = 1.0)
        proportions = np.concatenate([beta_props * p0, [1 - p0]])
        omegas = np.concatenate([beta_omegas, [1.0]])

        super().__init__(
            tree, sequence_length, kappa,
            proportions, omegas,
            codon_freqs, genetic_code, seed
        )

        self.p0 = p0
        self.p = p
        self.q = q
        self.omega_s = 1.0  # Fixed to 1.0 for M8a
        self.n_beta_categories = n_beta_categories

    def get_parameters(self) -> Dict[str, Any]:
        """Get simulation parameters."""
        return {
            'model': 'M8a',
            'kappa': float(self.kappa),
            'p0': float(self.p0),
            'p': float(self.p),
            'q': float(self.q),
            'omega_s': 1.0,  # Always 1.0 for M8a
            'n_beta_categories': int(self.n_beta_categories),
            'sequence_length': int(self.sequence_length),
            'genetic_code': int(self.genetic_code),
            'tree_length': sum(
                node.branch_length for node in self.tree.postorder()
                if node.parent is not None
            )
        }
