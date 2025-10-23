"""
Codon substitution models.
"""

import numpy as np

from ..io.sequences import CODONS, GENETIC_CODE, Alignment, GAP_CODE
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

    # Get sequences as strings (skip gaps and invalid codons)
    sequences = []
    for i in range(alignment.n_species):
        seq = []
        for j in range(alignment.n_sites):
            codon_idx = alignment.sequences[i, j]
            # Skip gaps (64) and invalid codons (< 0)
            if 0 <= codon_idx < 61:  # Valid codon (0-60)
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


def build_codon_Q_matrix(kappa: float, omega: float, pi: np.ndarray, normalization_factor: float = None) -> np.ndarray:
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
    normalization_factor : float, optional
        If provided, use this specific normalization factor instead of computing one.
        This is used for site class models to ensure all Q matrices use the same scale.

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
    if normalization_factor is not None:
        # Use provided normalization factor (for site class models)
        Q = create_reversible_Q(S, pi, normalize=False)
        Q = Q / normalization_factor
    else:
        # Compute normalization from this Q matrix alone
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

        Following PAML's approach, each Q matrix is normalized independently,
        then scaled so all share the same effective branch length scale.

        Returns
        -------
        list[np.ndarray]
            List of Q matrices, one per site class
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        # This makes all Q matrices share the same effective time scale
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


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
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


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
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in self.omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(self.proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M7CodonModel:
    """
    M7 (beta) codon model.

    Beta distribution for omega (0 < omega < 1), discretized into K categories.
    This model assumes all sites are under purifying or neutral selection.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    p_beta : float
        First shape parameter of beta distribution
    q_beta : float
        Second shape parameter of beta distribution
    ncatG : int
        Number of site classes for discretizing the beta distribution
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        p_beta: float = 0.5,
        q_beta: float = 0.5,
        ncatG: int = 10,
        pi: np.ndarray = None
    ):
        """Initialize M7 model."""
        self.kappa = kappa
        self.p_beta = max(p_beta, 0.005)  # Ensure > 0
        self.q_beta = max(q_beta, 0.005)  # Ensure > 0
        self.ncatG = ncatG

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """
        Get site class proportions and omega values.

        Uses beta distribution quantiles at median points of K equal bins,
        following PAML's DiscreteNSsites implementation.

        Returns
        -------
        tuple
            (proportions, omegas) where each is a list
        """
        from scipy.stats import beta

        K = self.ncatG
        proportions = [1.0 / K] * K
        omegas = []

        # Use median method: quantile at center of each bin
        for j in range(K):
            p = (j * 2.0 + 1) / (2.0 * K)
            omega = beta.ppf(p, self.p_beta, self.q_beta)
            omegas.append(omega)

        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M8CodonModel:
    """
    M8 (beta & omega > 1) codon model.

    Beta distribution for omega (0 < omega < 1) with proportion p0,
    plus an additional omega class (omega_s > 1) for positive selection
    with proportion (1 - p0).

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion of sites from beta distribution
    p_beta : float
        First shape parameter of beta distribution
    q_beta : float
        Second shape parameter of beta distribution
    omega_s : float
        Omega value for positive selection class (must be > 1)
    ncatG : int
        Number of site classes for discretizing the beta distribution
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        p0: float = 0.9,
        p_beta: float = 0.5,
        q_beta: float = 0.5,
        omega_s: float = 2.0,
        ncatG: int = 10,
        pi: np.ndarray = None
    ):
        """Initialize M8 model."""
        self.kappa = kappa
        self.p0 = np.clip(p0, 0.001, 0.999)
        self.p_beta = max(p_beta, 0.005)
        self.q_beta = max(q_beta, 0.005)
        self.omega_s = max(omega_s, 1.001)  # Must be > 1
        self.ncatG = ncatG

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
        from scipy.stats import beta

        K = self.ncatG  # Number of beta classes
        proportions = []
        omegas = []

        # Beta distribution classes (K classes total)
        for j in range(K):
            p = (j * 2.0 + 1) / (2.0 * K)
            omega = beta.ppf(p, self.p_beta, self.q_beta)
            omegas.append(omega)
            proportions.append(self.p0 / K)

        # Additional omega > 1 class
        omegas.append(self.omega_s)
        proportions.append(1.0 - self.p0)

        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M8aCodonModel:
    """
    M8a (beta & omega = 1) codon model.

    Beta distribution for omega (0 < omega < 1) with proportion p0,
    plus an additional omega=1 class (neutral) with proportion (1 - p0).

    This is the null model for the M8a vs M8 likelihood ratio test.
    The only difference from M8 is that omega_s is fixed to 1.0.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion of sites from beta distribution
    p_beta : float
        First shape parameter of beta distribution
    q_beta : float
        Second shape parameter of beta distribution
    ncatG : int
        Number of site classes for discretizing the beta distribution
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        p0: float = 0.9,
        p_beta: float = 0.5,
        q_beta: float = 0.5,
        ncatG: int = 10,
        pi: np.ndarray = None
    ):
        """Initialize M8a model."""
        self.kappa = kappa
        self.p0 = np.clip(p0, 0.001, 0.999)
        self.p_beta = max(p_beta, 0.005)
        self.q_beta = max(q_beta, 0.005)
        self.omega_s = 1.0  # Fixed to 1.0 (neutral)
        self.ncatG = ncatG

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
        from scipy.stats import beta

        K = self.ncatG  # Number of beta classes
        proportions = []
        omegas = []

        # Beta distribution classes (K classes total)
        for j in range(K):
            p = (j * 2.0 + 1) / (2.0 * K)
            omega = beta.ppf(p, self.p_beta, self.q_beta)
            omegas.append(omega)
            proportions.append(self.p0 / K)

        # Additional omega = 1 class (neutral, NOT positive selection)
        omegas.append(self.omega_s)  # Fixed to 1.0
        proportions.append(1.0 - self.p0)

        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M5CodonModel:
    """
    M5 (gamma) codon model.

    Gamma distribution for omega (allowing omega > 1), discretized into K categories.
    This model allows for variation in selective pressure including positive selection.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    alpha : float
        Shape parameter of gamma distribution
    beta : float
        Rate parameter of gamma distribution
    ncatG : int
        Number of site classes for discretizing the gamma distribution
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        ncatG: int = 10,
        pi: np.ndarray = None
    ):
        """Initialize M5 model."""
        self.kappa = kappa
        self.alpha = max(alpha, 0.005)  # Ensure > 0
        self.beta = max(beta, 0.005)    # Ensure > 0
        self.ncatG = ncatG

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """
        Get site class proportions and omega values.

        Uses gamma distribution quantiles at median points of K equal bins,
        following PAML's DiscreteNSsites implementation.

        Returns
        -------
        tuple
            (proportions, omegas) where each is a list
        """
        from scipy.stats import gamma

        K = self.ncatG
        proportions = [1.0 / K] * K
        omegas = []

        # Use median method: quantile at center of each bin
        # Gamma distribution with shape=alpha, scale=1/beta
        for j in range(K):
            p = (j * 2.0 + 1) / (2.0 * K)
            omega = gamma.ppf(p, self.alpha, scale=1.0/self.beta)
            omegas.append(omega)

        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M9CodonModel:
    """
    M9 (beta & gamma) codon model.

    Mixture of beta distribution for omega (0 < omega < 1) with proportion p0,
    plus a gamma distribution for omega (omega > 0) with proportion (1 - p0).

    This is a more flexible version of M8, allowing the positive selection
    class to follow a gamma distribution rather than a single omega value.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion of sites from beta distribution
    p_beta : float
        First shape parameter of beta distribution
    q_beta : float
        Second shape parameter of beta distribution
    alpha : float
        Shape parameter of gamma distribution
    beta_gamma : float
        Rate parameter of gamma distribution
    ncatG : int
        Number of site classes for discretizing each distribution
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        p0: float = 0.5,
        p_beta: float = 0.5,
        q_beta: float = 0.5,
        alpha: float = 1.0,
        beta_gamma: float = 1.0,
        ncatG: int = 10,
        pi: np.ndarray = None
    ):
        """Initialize M9 model."""
        self.kappa = kappa
        self.p0 = np.clip(p0, 0.001, 0.999)
        self.p_beta = max(p_beta, 0.005)
        self.q_beta = max(q_beta, 0.005)
        self.alpha = max(alpha, 0.005)
        self.beta_gamma = max(beta_gamma, 0.005)
        self.ncatG = ncatG

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """
        Get site class proportions and omega values.

        Following PAML, this discretizes the MIXTURE distribution (not each
        component separately) into K categories using the median method.

        Returns
        -------
        tuple
            (proportions, omegas) where each is a list
        """
        from scipy.stats import beta, gamma
        from scipy.optimize import brentq

        K = self.ncatG
        proportions = [1.0 / K] * K
        omegas = []

        # Define the mixture CDF
        def mixture_cdf(x):
            """Mixture CDF: p0 * Beta + (1-p0) * Gamma"""
            beta_cdf = beta.cdf(x, self.p_beta, self.q_beta)
            gamma_cdf = gamma.cdf(x, self.alpha, scale=1.0/self.beta_gamma)
            return self.p0 * beta_cdf + (1.0 - self.p0) * gamma_cdf

        # Discretize using median method: find quantiles of the mixture
        for j in range(K):
            p = (j * 2.0 + 1) / (2.0 * K)

            # Find omega such that mixture_cdf(omega) = p
            # Search in a reasonable range
            try:
                omega = brentq(lambda x: mixture_cdf(x) - p, 0.0001, 99.0)
            except ValueError:
                # If search fails, use a fallback
                omega = p * 10  # Simple fallback

            omegas.append(omega)

        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M4CodonModel:
    """
    M4 (freqs) codon model.

    Discrete model with 5 fixed omega values and variable proportions.
    Omega values are fixed at: {0, 1/3, 2/3, 1, 3}
    Proportions are estimated parameters.

    This is a variant of M3 with predefined omega values to test specific
    selection scenarios.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    proportions : list[float], length 5
        Proportions for each of the 5 site classes (must sum to 1)
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        proportions: list[float] = None,
        pi: np.ndarray = None
    ):
        """Initialize M4 model."""
        self.kappa = kappa
        
        # Fixed omega values for M4 (from PAML)
        self.omegas = [0.0, 1.0/3.0, 2.0/3.0, 1.0, 3.0]
        
        # Default proportions (equal)
        if proportions is None:
            self.proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            if len(proportions) != 5:
                raise ValueError(f"M4 requires exactly 5 proportions, got {len(proportions)}")
            # Normalize proportions
            proportions = np.array(proportions)
            self.proportions = (proportions / proportions.sum()).tolist()
        
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
        return self.proportions, self.omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]


class M6CodonModel:
    """
    M6 (2gamma) codon model.

    Mixture of two gamma distributions for omega, discretized into K categories.
    The second gamma has a constraint: shape = rate (alpha2 = beta2).

    This model allows for more flexibility than a single gamma distribution.

    Parameters
    ----------
    kappa : float
        Transition/transversion ratio
    p0 : float
        Proportion of sites from first gamma distribution
    alpha1 : float
        Shape parameter of first gamma distribution
    beta1 : float
        Rate parameter of first gamma distribution
    alpha2 : float
        Shape (and rate) parameter of second gamma distribution (alpha2 = beta2)
    ncatG : int
        Number of site classes for discretizing the mixture distribution
    pi : np.ndarray, shape (61,), optional
        Codon frequencies
    """

    def __init__(
        self,
        kappa: float = 2.0,
        p0: float = 0.5,
        alpha1: float = 1.0,
        beta1: float = 1.0,
        alpha2: float = 1.0,
        ncatG: int = 10,
        pi: np.ndarray = None
    ):
        """Initialize M6 model."""
        self.kappa = kappa
        self.p0 = np.clip(p0, 0.001, 0.999)
        self.alpha1 = max(alpha1, 0.005)
        self.beta1 = max(beta1, 0.005)
        self.alpha2 = max(alpha2, 0.005)  # alpha2 = beta2
        self.ncatG = ncatG

        if pi is None:
            self.pi = np.ones(61) / 61
        else:
            if len(pi) != 61:
                raise ValueError(f"pi must have length 61, got {len(pi)}")
            self.pi = pi / pi.sum()

    def get_site_classes(self) -> tuple[list[float], list[float]]:
        """
        Get site class proportions and omega values.

        Following PAML, this discretizes the MIXTURE distribution into K categories
        using the median method.

        Returns
        -------
        tuple
            (proportions, omegas) where each is a list
        """
        from scipy.stats import gamma
        from scipy.optimize import brentq

        K = self.ncatG
        proportions = [1.0 / K] * K
        omegas = []

        # Define the mixture CDF
        def mixture_cdf(x):
            """Mixture CDF: p0 * Gamma1 + (1-p0) * Gamma2"""
            gamma1_cdf = gamma.cdf(x, self.alpha1, scale=1.0/self.beta1)
            # Second gamma has shape = rate (alpha2 = beta2)
            gamma2_cdf = gamma.cdf(x, self.alpha2, scale=1.0/self.alpha2)
            return self.p0 * gamma1_cdf + (1.0 - self.p0) * gamma2_cdf

        # Discretize using median method: find quantiles of the mixture
        for j in range(K):
            p = (j * 2.0 + 1) / (2.0 * K)

            # Find omega such that mixture_cdf(omega) = p
            # Search in a reasonable range
            try:
                omega = brentq(lambda x: mixture_cdf(x) - p, 0.0001, 99.0)
            except ValueError:
                # If search fails, use a fallback
                omega = p * 10  # Simple fallback

            omegas.append(omega)

        return proportions, omegas

    def get_Q_matrices(self) -> list[np.ndarray]:
        """
        Get Q matrices for each site class.

        Following PAML's approach, all Q matrices are normalized by the same
        weighted-average normalization factor.
        """
        proportions, omegas = self.get_site_classes()

        # Build each Q matrix with its own omega, WITHOUT normalization
        Q_list_unnorm = [build_codon_Q_matrix(self.kappa, omega, self.pi, normalization_factor=1.0)
                         for omega in omegas]

        # Compute normalization factors for each UNNORMALIZED Q matrix
        norm_factors = [-np.dot(self.pi, np.diag(Q)) for Q in Q_list_unnorm]

        # Compute weighted average normalization (PAML's Qfactor_NS)
        weighted_avg_norm = sum(p * norm for p, norm in zip(proportions, norm_factors))

        # Normalize all Q matrices by the weighted average normalization
        return [Q / weighted_avg_norm for Q in Q_list_unnorm]
