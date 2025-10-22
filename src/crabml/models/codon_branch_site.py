"""
Branch-Site codon models for detecting positive selection on specific lineages.

Implements Branch-Site Model A (Yang & Nielsen 2002; Yang et al. 2005)
with exact PAML correspondence.
"""

from typing import Optional, Tuple
import numpy as np
from .codon import build_codon_Q_matrix


class BranchSiteModelA:
    """
    Branch-Site Model A (Zhang, Nielsen & Yang 2005).

    Tests for positive selection on foreground branches by allowing
    omega to vary both across sites and across branches.

    Site class structure:
        Class 0:  conserved on both background and foreground (ω₀ < 1)
        Class 1:  neutral on both background and foreground (ω = 1)
        Class 2a: conserved on background (ω₀), positive on foreground (ω₂)
        Class 2b: neutral on background (ω = 1), positive on foreground (ω₂)

    Parameters:
        kappa: transition/transversion ratio
        p0: proportion of site class 0 (conserved)
        p1: proportion of site class 1 (neutral)
        omega0: dN/dS for conserved sites (0 < ω₀ < 1)
        omega2: dN/dS for foreground class 2 (ω₂ ≥ 1, or =1 for null model)

    References:
        Yang & Nielsen (2002). Mol. Biol. Evol. 19:908-917
        Yang et al. (2005). Mol. Biol. Evol. 22:1107-1118
        Zhang et al. (2005). Mol. Biol. Evol. 22:2472-2479
    """

    def __init__(
        self,
        codon_frequencies: np.ndarray,
        branch_labels: np.ndarray,
        fix_omega: bool = False,
    ):
        """
        Initialize Branch-Site Model A.

        Parameters
        ----------
        codon_frequencies : np.ndarray
            Codon frequencies (61 sense codons)
        branch_labels : np.ndarray
            Integer labels for each branch (0=background, 1=foreground)
        fix_omega : bool
            If True, fix omega2=1 (null model A1)
        """
        self.pi = np.array(codon_frequencies)
        self.branch_labels = np.array(branch_labels, dtype=int)
        self.fix_omega = fix_omega
        self.n_site_classes = 4

        # Validate branch labels
        unique_labels = np.unique(self.branch_labels)
        if not np.array_equal(unique_labels, np.array([0, 1])):
            raise ValueError(
                f"Branch-site models require exactly 2 branch types (0 and 1). "
                f"Found: {unique_labels}"
            )

        n_foreground = np.sum(self.branch_labels == 1)
        if n_foreground == 0:
            raise ValueError("No foreground branches marked with label 1")

        print(f"Branch-Site Model A initialized:")
        print(f"  - {len(self.branch_labels)} branches")
        print(f"  - {np.sum(self.branch_labels == 0)} background branches")
        print(f"  - {n_foreground} foreground branches")
        print(f"  - Omega2 {'fixed at 1.0' if fix_omega else 'free to vary'}")

    def get_parameters(self) -> dict:
        """
        Get parameter names and bounds.

        Returns
        -------
        dict
            Parameter names mapped to (lower_bound, upper_bound) tuples
        """
        params = {
            'kappa': (0.1, 20.0),
            'p0': (0.0, 0.999),  # Must leave room for p1
            'p1': (0.0, 0.999),  # Must leave room for p0
            'omega0': (1e-6, 0.999),  # Constrained < 1
        }

        if not self.fix_omega:
            params['omega2'] = (1.0, 20.0)  # ω₂ ≥ 1 for alternative model

        return params

    def get_param_names(self) -> list[str]:
        """Get list of parameter names in order."""
        names = ['kappa', 'p0', 'p1', 'omega0']
        if not self.fix_omega:
            names.append('omega2')
        return names

    def compute_site_class_frequencies(
        self,
        p0: float,
        p1: float,
    ) -> np.ndarray:
        """
        Compute frequencies for the 4 site classes.

        Site class structure:
            Class 0:  p0
            Class 1:  p1
            Class 2a: p2 * p0/(p0+p1)
            Class 2b: p2 * p1/(p0+p1)

        where p2 = 1 - p0 - p1

        Parameters
        ----------
        p0 : float
            Proportion of site class 0
        p1 : float
            Proportion of site class 1

        Returns
        -------
        np.ndarray
            Array of 4 site class frequencies [p0, p1, p2a, p2b]

        Raises
        ------
        ValueError
            If p0 + p1 >= 1 (no room for class 2)
        """
        if p0 + p1 >= 1.0:
            raise ValueError(
                f"p0 + p1 must be < 1 (got {p0} + {p1} = {p0+p1}). "
                f"No probability mass left for class 2."
            )

        if p0 + p1 < 1e-10:
            raise ValueError(
                f"p0 + p1 too small ({p0+p1}). "
                f"Cannot compute class 2 frequencies."
            )

        p2 = 1.0 - p0 - p1
        t = p0 + p1

        return np.array([
            p0,              # Class 0: conserved everywhere
            p1,              # Class 1: neutral everywhere
            p2 * p0 / t,     # Class 2a: conserved on back, positive on fore
            p2 * p1 / t,     # Class 2b: neutral on back, positive on fore
        ])

    def compute_qfactors(
        self,
        kappa: float,
        p0: float,
        p1: float,
        omega0: float,
        omega2: float,
    ) -> Tuple[float, float]:
        """
        Compute Q matrix normalization factors for background and foreground.

        Background branches have 2 site classes (0, 1) with normalized frequencies.
        Foreground branches have 3 site classes (0, 1, 2) with MLE frequencies.

        This follows PAML's exact computation (codeml.c lines 2608-2630).

        Parameters
        ----------
        kappa : float
            Transition/transversion ratio
        p0, p1 : float
            Site class proportions
        omega0, omega2 : float
            Omega values

        Returns
        -------
        Tuple[float, float]
            (Qfactor_background, Qfactor_foreground)
        """
        p2 = 1.0 - p0 - p1
        t = p0 + p1

        if t < 1e-10:
            raise ValueError("p0 + p1 too small for Qfactor computation")

        # Background: 2 site classes with normalized frequencies
        # freq[0] = p0/t, freq[1] = p1/t
        # Weighted omega: mr_back = (p0/t)*omega0 + (p1/t)*1.0
        mr_back = (p0 / t) * omega0 + (p1 / t) * 1.0

        # Build Q matrix with weighted omega for background
        Q_back = build_codon_Q_matrix(
            kappa=kappa,
            omega=mr_back,
            pi=self.pi,
            normalization_factor=1.0  # No normalization yet
        )
        # Get mean rate from Q diagonal: mr = -sum(pi * Q_ii)
        mr_from_Q_back = -np.dot(self.pi, np.diag(Q_back))
        Qfactor_back = 1.0 / mr_from_Q_back

        # Foreground: 3 site classes with MLE frequencies
        # freq[0] = p0, freq[1] = p1, freq[2] = p2
        # Weighted omega: mr_fore = p0*omega0 + p1*1.0 + p2*omega2
        mr_fore = p0 * omega0 + p1 * 1.0 + p2 * omega2

        # Build Q matrix with weighted omega for foreground
        Q_fore = build_codon_Q_matrix(
            kappa=kappa,
            omega=mr_fore,
            pi=self.pi,
            normalization_factor=1.0  # No normalization yet
        )
        # Get mean rate from Q diagonal
        mr_from_Q_fore = -np.dot(self.pi, np.diag(Q_fore))
        Qfactor_fore = 1.0 / mr_from_Q_fore

        return Qfactor_back, Qfactor_fore

    def get_omega_for_branch_class(
        self,
        branch_label: int,
        site_class: int,
        omega0: float,
        omega2: float,
    ) -> float:
        """
        Get omega value for a specific (branch_label, site_class) combination.

        Parameters
        ----------
        branch_label : int
            0 (background) or 1 (foreground)
        site_class : int
            Site class index (0, 1, 2, or 3)
        omega0, omega2 : float
            Model parameters

        Returns
        -------
        float
            Omega value for this combination

        Raises
        ------
        ValueError
            If invalid branch_label or site_class
        """
        if branch_label not in [0, 1]:
            raise ValueError(f"Invalid branch_label: {branch_label}")

        if site_class not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid site_class: {site_class}")

        # Site class 0: omega0 everywhere
        if site_class == 0:
            return omega0

        # Site class 1: omega=1 everywhere
        elif site_class == 1:
            return 1.0

        # Site class 2a or 2b: depends on branch type
        else:
            if branch_label == 0:  # Background
                # Class 2a: omega0 on background
                # Class 2b: omega=1 on background
                return omega0 if site_class == 2 else 1.0
            else:  # Foreground (label = 1)
                # Both class 2a and 2b: omega2 on foreground
                return omega2

    def validate_parameters(
        self,
        kappa: float,
        p0: float,
        p1: float,
        omega0: float,
        omega2: Optional[float] = None,
    ) -> None:
        """
        Validate parameter values.

        Parameters
        ----------
        kappa, p0, p1, omega0, omega2 : float
            Model parameters

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")

        if not (0 <= p0 < 1):
            raise ValueError(f"p0 must be in [0, 1), got {p0}")

        if not (0 <= p1 < 1):
            raise ValueError(f"p1 must be in [0, 1), got {p1}")

        if p0 + p1 >= 1.0:
            raise ValueError(f"p0 + p1 must be < 1, got {p0 + p1}")

        if not (0 < omega0 < 1):
            raise ValueError(f"omega0 must be in (0, 1), got {omega0}")

        if not self.fix_omega:
            if omega2 is None:
                raise ValueError("omega2 is required when not fixed")
            if omega2 < 1.0:
                raise ValueError(f"omega2 must be >= 1, got {omega2}")

    def __repr__(self) -> str:
        """String representation."""
        model_type = "A1 (null)" if self.fix_omega else "A (alternative)"
        return (
            f"BranchSiteModelA({model_type}, "
            f"{len(self.branch_labels)} branches, "
            f"{self.n_site_classes} site classes)"
        )


class BranchSiteModelA1(BranchSiteModelA):
    """
    Branch-Site Model A1 (null model).

    Identical to Model A except omega2 is fixed at 1.0.
    Used as null hypothesis in likelihood ratio test.

    This tests H₀: no positive selection on foreground branches (ω₂ ≤ 1).
    """

    def __init__(
        self,
        codon_frequencies: np.ndarray,
        branch_labels: np.ndarray,
    ):
        """
        Initialize null model with omega2 fixed at 1.

        Parameters
        ----------
        codon_frequencies : np.ndarray
            Codon frequencies
        branch_labels : np.ndarray
            Branch labels (0=background, 1=foreground)
        """
        super().__init__(
            codon_frequencies=codon_frequencies,
            branch_labels=branch_labels,
            fix_omega=True,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BranchSiteModelA1(null model, omega2=1, "
            f"{len(self.branch_labels)} branches)"
        )
