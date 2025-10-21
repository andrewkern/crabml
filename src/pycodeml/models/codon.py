"""
Codon substitution models.
"""

import numpy as np

from ..io.sequences import CODONS, GENETIC_CODE, Alignment
from ..core.matrix import create_reversible_Q


def compute_codon_frequencies_f3x4(alignment: Alignment) -> np.ndarray:
    """
    Compute F3X4 codon frequencies from alignment.

    F3X4 estimates codon frequencies from the nucleotide frequencies
    at each of the three codon positions.

    Parameters
    ----------
    alignment : Alignment
        Codon alignment

    Returns
    -------
    np.ndarray, shape (61,)
        Codon frequencies
    """
    if alignment.seqtype != "codon":
        raise ValueError("F3X4 requires codon alignment")

    # Get sequences as strings
    sequences = []
    for i in range(alignment.n_species):
        seq = []
        for j in range(alignment.n_sites):
            codon_idx = alignment.sequences[i, j]
            if codon_idx >= 0:  # Valid codon
                from ..io.sequences import INDEX_TO_CODON
                seq.append(INDEX_TO_CODON[codon_idx])
        sequences.append(''.join(seq))

    # Count nucleotide frequencies at each codon position
    nuc_counts = np.zeros((3, 4))  # 3 positions x 4 nucleotides (T, C, A, G)
    nuc_map = {'T': 0, 'C': 1, 'A': 2, 'G': 3}

    total_codons = 0
    for seq in sequences:
        for i in range(0, len(seq), 3):
            if i + 2 < len(seq):
                codon = seq[i:i+3]
                if len(codon) == 3 and all(n in nuc_map for n in codon):
                    for pos, nuc in enumerate(codon):
                        nuc_counts[pos, nuc_map[nuc]] += 1
                    total_codons += 1

    # Compute frequencies at each position
    pi_nuc = nuc_counts / nuc_counts.sum(axis=1, keepdims=True)

    # Compute codon frequencies as product of nucleotide frequencies
    pi_codon = np.zeros(61)
    for idx, codon in enumerate(CODONS):
        pi_codon[idx] = (
            pi_nuc[0, nuc_map[codon[0]]] *
            pi_nuc[1, nuc_map[codon[1]]] *
            pi_nuc[2, nuc_map[codon[2]]]
        )

    # Normalize
    pi_codon /= pi_codon.sum()

    return pi_codon


def is_transition(nuc1: str, nuc2: str) -> bool:
    """Check if nucleotide change is a transition (A<->G or C<->T)."""
    transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    return (nuc1, nuc2) in transitions


def is_synonymous(codon1: str, codon2: str) -> bool:
    """Check if two codons code for the same amino acid."""
    return GENETIC_CODE[codon1] == GENETIC_CODE[codon2]


def build_codon_Q_matrix(kappa: float, omega: float, pi: np.ndarray) -> np.ndarray:
    """
    Build a codon rate matrix Q for given kappa, omega, and pi.

    This is a helper function used by all codon models.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    omega : float
        dN/dS ratio
    pi : np.ndarray, shape (61,)
        Codon frequencies

    Returns
    -------
    np.ndarray, shape (61, 61)
        Rate matrix Q
    """
    # Build exchangeability matrix S
    S = np.zeros((61, 61))

    for i, codon_i in enumerate(CODONS):
        for j, codon_j in enumerate(CODONS):
            if i == j:
                continue

            # Count nucleotide differences
            diffs = sum(1 for k in range(3) if codon_i[k] != codon_j[k])

            if diffs != 1:
                # Only single nucleotide changes allowed
                continue

            # Find which position differs
            diff_pos = next(k for k in range(3) if codon_i[k] != codon_j[k])
            nuc_i = codon_i[diff_pos]
            nuc_j = codon_j[diff_pos]

            # Base exchangeability is 1
            s = 1.0

            # Multiply by kappa if transition
            if is_transition(nuc_i, nuc_j):
                s *= kappa

            # Multiply by omega if non-synonymous
            if not is_synonymous(codon_i, codon_j):
                s *= omega

            S[i, j] = s

    # Create reversible Q matrix
    Q = create_reversible_Q(S, pi, normalize=True)

    return Q


class M0CodonModel:
    """
    M0 codon substitution model (one dN/dS ratio).

    The M0 model has a single omega (dN/dS) parameter for all branches
    and sites. The substitution rate between codons depends on:
    - kappa (transition/transversion ratio)
    - omega (non-synonymous/synonymous ratio)
    - codon frequencies (pi)

    Parameters
    ----------
    kappa : float
        Transition/transversion rate ratio (default 2.0)
    omega : float
        dN/dS ratio (default 0.4)
    pi : np.ndarray, shape (61,), optional
        Codon equilibrium frequencies. If None, uniform frequencies are used.
    """

    def __init__(
        self,
        kappa: float = 2.0,
        omega: float = 0.4,
        pi: np.ndarray = None
    ):
        """Initialize M0 codon model."""
        self.kappa = kappa
        self.omega = omega

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()  # Ensure normalized

    def get_Q_matrix(self) -> np.ndarray:
        """
        Construct the rate matrix Q for the M0 model.

        Returns
        -------
        np.ndarray, shape (61, 61)
            Rate matrix Q (normalized to one substitution per time unit)
        """
        return build_codon_Q_matrix(self.kappa, self.omega, self.pi)


class M1aCodonModel:
    """
    M1a (NearlyNeutral) codon model.

    Two site classes:
    - Class 0: omega_0 < 1 (purifying selection), proportion p_0
    - Class 1: omega_1 = 1 (neutral), proportion p_1 = 1 - p_0

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    omega0 : float
        dN/dS for purifying selection class (must be < 1)
    p0 : float
        Proportion of sites under purifying selection
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        omega0: float = 0.5,
        p0: float = 0.7,
        pi: np.ndarray = None
    ):
        """Initialize M1a model."""
        self.kappa = kappa
        self.omega0 = min(omega0, 0.999)  # Ensure < 1
        self.p0 = np.clip(p0, 0.001, 0.999)

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """
        Get site class proportions and omega values.

        Returns
        -------
        tuple
            (proportions, omegas) where each is a list
        """
        proportions = [self.p0, 1.0 - self.p0]
        omegas = [self.omega0, 1.0]
        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Returns
        -------
        list[np.ndarray]
            List of Q matrices, one per site class
        """
        _, omegas = self.get_site_classes()
        return [build_codon_Q_matrix(self.kappa, omega, self.pi) for omega in omegas]


class M2aCodonModel:
    """
    M2a (PositiveSelection) codon model.

    Three site classes:
    - Class 0: omega_0 < 1 (purifying), proportion p_0
    - Class 1: omega_1 = 1 (neutral), proportion p_1
    - Class 2: omega_2 > 1 (positive selection), proportion p_2 = 1 - p_0 - p_1

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    omega0 : float
        dN/dS for purifying class (< 1)
    omega2 : float
        dN/dS for positive selection class (> 1)
    p0 : float
        Proportion of purifying sites
    p1 : float
        Proportion of neutral sites
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        omega0: float = 0.5,
        omega2: float = 2.0,
        p0: float = 0.5,
        p1: float = 0.3,
        pi: np.ndarray = None
    ):
        """Initialize M2a model."""
        self.kappa = kappa
        self.omega0 = min(omega0, 0.999)
        self.omega2 = max(omega2, 1.001)

        # Ensure proportions sum to 1
        total = p0 + p1
        if total >= 1.0:
            # Rescale
            p0 = p0 / (total + 0.01) * 0.99
            p1 = p1 / (total + 0.01) * 0.99

        self.p0 = np.clip(p0, 0.001, 0.998)
        self.p1 = np.clip(p1, 0.001, 0.998 - self.p0)

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """Get site class proportions and omega values."""
        proportions = [self.p0, self.p1, 1.0 - self.p0 - self.p1]
        omegas = [self.omega0, 1.0, self.omega2]
        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """Get Q matrices for each site class."""
        _, omegas = self.get_site_classes()
        return [build_codon_Q_matrix(self.kappa, omega, self.pi) for omega in omegas]


class M3CodonModel:
    """
    M3 (discrete) codon model.

    K site classes with different omega values.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    omegas : list[float]
        Omega value for each site class
    proportions : list[float]
        Proportion of sites in each class (must sum to 1)
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        omegas: list[float] = None,
        proportions: list[float] = None,
        pi: np.ndarray = None
    ):
        """Initialize M3 model."""
        self.kappa = kappa

        if omegas is None:
            omegas = [0.5, 1.0, 2.0]
        if proportions is None:
            proportions = [1.0 / len(omegas)] * len(omegas)

        if len(omegas) != len(proportions):
            raise ValueError("omegas and proportions must have same length")

        # Normalize proportions
        proportions = np.array(proportions)
        proportions = proportions / proportions.sum()

        self.omegas = omegas
        self.proportions = proportions.tolist()

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """Get site class proportions and omega values."""
        return self.proportions, self.omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """Get Q matrices for each site class."""
        return [build_codon_Q_matrix(self.kappa, omega, self.pi) for omega in self.omegas]
