"""
Codon sequence simulators.
"""

from typing import Optional, Dict, Any
import numpy as np

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
