"""
High-level API for crabML codon model optimization.

This module provides a simplified interface for fitting codon models,
with unified result objects and automatic file format detection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import warnings
import numpy as np

from .io.sequences import Alignment
from .io.trees import Tree
from .optimize.optimizer import (
    M0Optimizer, M1aOptimizer, M2aOptimizer, M3Optimizer,
    M7Optimizer, M8Optimizer, M8aOptimizer
)
from .optimize.branch import BranchModelOptimizer


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from .optimize.branch_site import BranchSiteModelAOptimizer


@dataclass
class ModelResultBase:
    """
    Base class for all codon model optimization results.

    This class provides common functionality shared across all model types
    (site-class, branch, branch-site), including export and display methods.

    Attributes
    ----------
    model_name : str
        Name of the codon model (e.g., "M0", "M1a", "Branch-Site Model A")
    lnL : float
        Log-likelihood of the fitted model
    kappa : float
        Transition/transversion ratio
    params : Dict[str, Any]
        Model-specific parameters
    tree : Tree
        Phylogenetic tree with optimized branch lengths
    alignment : Alignment
        Codon alignment used for optimization
    n_params : int
        Number of free parameters in the model
    convergence_info : Optional[Dict[str, Any]]
        Information about optimizer convergence (reserved for future use)
    """

    model_name: str
    lnL: float
    kappa: float
    params: Dict[str, Any]
    tree: Tree
    alignment: Alignment
    n_params: int
    convergence_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Export results as a dictionary.

        The tree and alignment objects are not included in the dictionary
        export to keep it JSON-serializable.

        Returns
        -------
        dict
            Dictionary containing model name, likelihood, parameters, etc.

        Examples
        --------
        >>> result = optimize_model("M0", "data.fasta", "tree.nwk")
        >>> data = result.to_dict()
        >>> print(data['lnL'])
        """
        return {
            'model_name': self.model_name,
            'lnL': float(self.lnL),
            'kappa': float(self.kappa),
            'params': self.params,
            'n_params': int(self.n_params),
            'convergence_info': self.convergence_info,
        }

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Export results as JSON.

        Parameters
        ----------
        filepath : str, optional
            If provided, write JSON to this file
        indent : int, default=2
            Indentation level for pretty printing

        Returns
        -------
        str
            JSON string representation

        Examples
        --------
        >>> result = optimize_model("M0", "data.fasta", "tree.nwk")
        >>> result.to_json("results.json")
        >>> json_str = result.to_json()
        """
        result_dict = self.to_dict()
        json_str = json.dumps(result_dict, indent=indent, cls=NumpyEncoder)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def __str__(self) -> str:
        """String representation shows summary."""
        return self.summary()

    def __repr__(self) -> str:
        """Concise representation for interactive use."""
        return f"{self.__class__.__name__}(model='{self.model_name}', lnL={self.lnL:.2f}, kappa={self.kappa:.2f})"

    def summary(self) -> str:
        """
        Generate human-readable summary of optimization results.

        Must be implemented by subclasses.

        Returns
        -------
        str
            Formatted multi-line summary
        """
        raise NotImplementedError("Subclasses must implement summary()")


@dataclass
class SiteModelResult(ModelResultBase):
    """
    Result object for site-class codon models.

    Site-class models allow omega (dN/dS) to vary across sites but not
    across branches. This includes models M0, M1a, M2a, M3, M7, M8, M8a, etc.

    Attributes
    ----------
    model_name : str
        Name of the codon model (e.g., "M0", "M1A", "M2A")
    lnL : float
        Log-likelihood of the fitted model
    kappa : float
        Transition/transversion ratio
    params : Dict[str, Any]
        Model-specific parameters (omega, proportions, etc.)
    tree : Tree
        Phylogenetic tree with optimized branch lengths
    alignment : Alignment
        Codon alignment used for optimization
    n_params : int
        Number of free parameters in the model
    convergence_info : Optional[Dict[str, Any]]
        Information about optimizer convergence (reserved for future use)

    Examples
    --------
    >>> from crabml import optimize_model
    >>> result = optimize_model("M0", "alignment.fasta", "tree.nwk")
    >>> print(result.summary())
    >>> print(f"omega = {result.omega:.4f}")
    >>> result.to_json("results.json")
    """

    @property
    def omega(self) -> Optional[float]:
        """
        Single omega value (for M0 only).

        Returns
        -------
        float or None
            The omega (dN/dS) value for M0 model, None for other models
        """
        return self.params.get('omega')

    @property
    def omegas(self) -> Optional[List[float]]:
        """
        Multiple omega values across site classes.

        For models with discrete site classes (M1a, M2a, M3), returns
        the list of omega values. For continuous models (M7, M8), returns None.

        Returns
        -------
        list of float or None
            List of omega values for each site class, or None if not applicable
        """
        # Some models store omegas directly
        if 'omegas' in self.params:
            return self.params['omegas']

        # Construct from model-specific parameters
        if self.model_name == 'M1A':
            return [self.params.get('omega0'), 1.0]
        elif self.model_name == 'M2A':
            return [self.params.get('omega0'), 1.0, self.params.get('omega2')]

        return None

    @property
    def proportions(self) -> Optional[List[float]]:
        """
        Site class proportions.

        Returns
        -------
        list of float or None
            Proportion of sites in each class, or None if not applicable
        """
        # Some models store proportions directly
        if 'proportions' in self.params:
            return self.params['proportions']

        # Construct from model-specific parameters
        if self.model_name == 'M1A':
            p0 = self.params.get('p0')
            return [p0, 1-p0] if p0 is not None else None
        elif self.model_name == 'M2A':
            p0, p1 = self.params.get('p0'), self.params.get('p1')
            if p0 is not None and p1 is not None:
                return [p0, p1, 1-p0-p1]
        elif self.model_name == 'M8':
            p0 = self.params.get('p0')
            return [p0, 1-p0] if p0 is not None else None
        elif self.model_name == 'M8A':
            p0 = self.params.get('p0')
            return [p0, 1-p0] if p0 is not None else None

        return None

    @property
    def n_site_classes(self) -> int:
        """
        Number of discrete site classes in the model.

        Returns
        -------
        int
            Number of site classes
        """
        if self.proportions:
            return len(self.proportions)
        return 1  # M0 has single class

    def summary(self) -> str:
        """
        Generate human-readable summary of optimization results.

        Returns
        -------
        str
            Formatted multi-line summary with model parameters and statistics

        Examples
        --------
        >>> result = optimize_model("M0", "data.fasta", "tree.nwk")
        >>> print(result.summary())
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"MODEL: {self.model_name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Log-likelihood:       {self.lnL:.6f}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append("")
        lines.append("PARAMETERS:")
        lines.append(f"  kappa (ts/tv) = {self.kappa:.4f}")

        # Model-specific parameter display
        if self.model_name == 'M0':
            lines.append(f"  omega (dN/dS) = {self.params['omega']:.4f}")

        elif self.model_name in ('M1A', 'M2A'):
            omegas = self.omegas
            props = self.proportions
            lines.append("")
            lines.append("  Site classes:")
            for i, (omega, prop) in enumerate(zip(omegas, props)):
                lines.append(f"    Class {i}: omega = {omega:.4f}, proportion = {prop:.4f}")

        elif self.model_name == 'M3':
            omegas = self.params.get('omegas', [])
            props = self.params.get('proportions', [])
            lines.append("")
            lines.append("  Site classes:")
            for i, (omega, prop) in enumerate(zip(omegas, props)):
                lines.append(f"    Class {i}: omega = {omega:.4f}, proportion = {prop:.4f}")

        elif self.model_name == 'M7':
            lines.append(f"  Beta distribution:")
            lines.append(f"    p = {self.params['p']:.4f}")
            lines.append(f"    q = {self.params['q']:.4f}")

        elif self.model_name == 'M8':
            lines.append(f"  Beta distribution:")
            lines.append(f"    p = {self.params['p']:.4f}")
            lines.append(f"    q = {self.params['q']:.4f}")
            lines.append(f"  Additional site class:")
            lines.append(f"    omega_s = {self.params['omega_s']:.4f}")
            lines.append(f"    proportion = {1-self.params['p0']:.4f}")

        elif self.model_name == 'M8A':
            lines.append(f"  Beta distribution:")
            lines.append(f"    p = {self.params['p']:.4f}")
            lines.append(f"    q = {self.params['q']:.4f}")
            lines.append(f"  Additional site class:")
            lines.append(f"    omega = 1.0000 (fixed)")
            lines.append(f"    proportion = {1-self.params['p0']:.4f}")

        lines.append("")
        lines.append("TREE:")
        n_leaves = self.tree.n_leaves
        n_branches = sum(1 for node in self.tree.postorder() if node.parent is not None)
        lines.append(f"  {n_leaves} sequences")
        lines.append(f"  {n_branches} branches (optimized)")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


@dataclass
class BranchModelResult(ModelResultBase):
    """
    Result object for branch codon models.

    Branch models allow omega (dN/dS) to vary across branches but not
    across sites. This includes free-ratio and multi-ratio models.

    Attributes
    ----------
    model_name : str
        Name of the branch model (e.g., "Free-ratio", "Multi-ratio")
    lnL : float
        Log-likelihood of the fitted model
    kappa : float
        Transition/transversion ratio
    params : Dict[str, Any]
        Model parameters including omega_dict
    tree : Tree
        Phylogenetic tree with optimized branch lengths and branch labels
    alignment : Alignment
        Codon alignment used for optimization
    n_params : int
        Number of free parameters in the model
    convergence_info : Optional[Dict[str, Any]]
        Information about optimizer convergence (reserved for future use)

    Examples
    --------
    >>> from crabml import optimize_branch_model
    >>> result = optimize_branch_model("multi-ratio", "alignment.fasta", "tree.nwk")
    >>> print(result.summary())
    >>> print(f"Foreground omega = {result.foreground_omega:.4f}")
    >>> print(f"Background omega = {result.background_omega:.4f}")
    """

    @property
    def omega_dict(self) -> Dict[str, float]:
        """
        Dictionary mapping omega names to values.

        Returns
        -------
        Dict[str, float]
            Dictionary like {'omega0': 0.5, 'omega1': 2.3}
        """
        return self.params.get('omega_dict', {})

    @property
    def foreground_omega(self) -> Optional[float]:
        """
        Omega value for foreground branches (label #1).

        Returns
        -------
        float or None
            Foreground omega if available, None otherwise
        """
        return self.omega_dict.get('omega1')

    @property
    def background_omega(self) -> Optional[float]:
        """
        Omega value for background branches (label #0).

        Returns
        -------
        float or None
            Background omega if available, None otherwise
        """
        return self.omega_dict.get('omega0')

    def summary(self) -> str:
        """
        Generate human-readable summary of branch model results.

        Returns
        -------
        str
            Formatted multi-line summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"MODEL: {self.model_name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Log-likelihood:       {self.lnL:.6f}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append("")
        lines.append("PARAMETERS:")
        lines.append(f"  kappa (ts/tv) = {self.kappa:.4f}")
        lines.append("")
        lines.append("  Branch-specific omega values:")

        omega_dict = self.omega_dict
        for omega_name in sorted(omega_dict.keys()):
            omega_val = omega_dict[omega_name]
            lines.append(f"    {omega_name} = {omega_val:.4f}")

        lines.append("")
        lines.append("TREE:")
        n_leaves = self.tree.n_leaves
        n_branches = sum(1 for node in self.tree.postorder() if node.parent is not None)
        lines.append(f"  {n_leaves} sequences")
        lines.append(f"  {n_branches} branches (optimized)")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


@dataclass
class BranchSiteModelResult(ModelResultBase):
    """
    Result object for branch-site codon models.

    Branch-site models allow omega (dN/dS) to vary across both sites AND
    branches. This includes Branch-Site Model A for detecting positive
    selection on specific lineages.

    Attributes
    ----------
    model_name : str
        Name of the branch-site model (e.g., "Branch-Site Model A")
    lnL : float
        Log-likelihood of the fitted model
    kappa : float
        Transition/transversion ratio
    params : Dict[str, Any]
        Model parameters including omega0, omega2, p0, p1
    tree : Tree
        Phylogenetic tree with optimized branch lengths and branch labels
    alignment : Alignment
        Codon alignment used for optimization
    n_params : int
        Number of free parameters in the model
    convergence_info : Optional[Dict[str, Any]]
        Information about optimizer convergence (reserved for future use)

    Examples
    --------
    >>> from crabml import optimize_branch_site_model
    >>> result = optimize_branch_site_model("model-a", "alignment.fasta", "tree.nwk")
    >>> print(result.summary())
    >>> print(f"Positive selection omega = {result.omega2:.4f}")
    >>> print(f"Proportion under selection = {result.foreground_positive_proportion:.4f}")
    """

    @property
    def omega0(self) -> float:
        """
        Omega value for conserved sites (class 0).

        Returns
        -------
        float
            Conserved omega (typically < 1)
        """
        return self.params['omega0']

    @property
    def omega2(self) -> float:
        """
        Omega value for positive selection sites on foreground (class 2a/2b).

        Returns
        -------
        float
            Positive selection omega (can be > 1)
        """
        return self.params['omega2']

    @property
    def proportions(self) -> List[float]:
        """
        Site class proportions [p0, p1, p2a, p2b].

        Returns
        -------
        List[float]
            Four proportions for site classes 0, 1, 2a, 2b
        """
        p0 = self.params['p0']
        p1 = self.params['p1']
        # p2 is split into p2a and p2b (each gets half)
        p2 = 1.0 - p0 - p1
        return [p0, p1, p2/2, p2/2]

    @property
    def foreground_positive_proportion(self) -> float:
        """
        Proportion of sites under positive selection on foreground branches.

        This combines class 2a and 2b proportions (sites with omega2 on foreground).

        Returns
        -------
        float
            Proportion of positively selected sites on foreground
        """
        p0 = self.params['p0']
        p1 = self.params['p1']
        return 1.0 - p0 - p1  # p2a + p2b

    def summary(self) -> str:
        """
        Generate human-readable summary of branch-site model results.

        Returns
        -------
        str
            Formatted multi-line summary
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"MODEL: {self.model_name}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Log-likelihood:       {self.lnL:.6f}")
        lines.append(f"Number of parameters: {self.n_params}")
        lines.append("")
        lines.append("PARAMETERS:")
        lines.append(f"  kappa (ts/tv) = {self.kappa:.4f}")
        lines.append("")
        lines.append("  Site classes (background | foreground):")

        props = self.proportions
        lines.append(f"    Class 0 (conserved):     ω₀ = {self.omega0:.4f} | ω₀ = {self.omega0:.4f}  (p = {props[0]:.4f})")
        lines.append(f"    Class 1 (neutral):       ω = 1.0000 | ω = 1.0000  (p = {props[1]:.4f})")
        lines.append(f"    Class 2a (pos. sel.):    ω₀ = {self.omega0:.4f} | ω₂ = {self.omega2:.4f}  (p = {props[2]:.4f})")
        lines.append(f"    Class 2b (pos. sel.):    ω = 1.0000 | ω₂ = {self.omega2:.4f}  (p = {props[3]:.4f})")
        lines.append("")
        lines.append(f"  Foreground positive selection:")
        lines.append(f"    ω₂ = {self.omega2:.4f}")
        lines.append(f"    Proportion = {self.foreground_positive_proportion:.4f} (classes 2a + 2b)")

        lines.append("")
        lines.append("TREE:")
        n_leaves = self.tree.n_leaves
        n_branches = sum(1 for node in self.tree.postorder() if node.parent is not None)
        lines.append(f"  {n_leaves} sequences")
        lines.append(f"  {n_branches} branches (optimized)")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Backwards compatibility alias
# =============================================================================

# Provide ModelResult as alias to SiteModelResult for backwards compatibility
ModelResult = SiteModelResult


# =============================================================================
# Parser functions: Convert optimizer result tuples to result objects
# =============================================================================

def _parse_m0_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M0 optimizer result: (kappa, omega, lnL)"""
    kappa, omega, lnL = result_tuple

    # Count parameters: kappa + omega + branch lengths
    n_params = 2
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1  # global branch scale

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={'omega': omega},
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_m1a_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M1a optimizer result: (kappa, p0, omega0, lnL)"""
    kappa, p0, omega0, lnL = result_tuple

    # Count parameters: kappa + p0 + omega0 + branch lengths
    n_params = 3
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p0': p0,
            'p1': 1.0 - p0,
            'omega0': omega0,
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_m2a_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M2a optimizer result: (kappa, p0, p1, omega0, omega2, lnL)"""
    kappa, p0, p1, omega0, omega2, lnL = result_tuple

    # Count parameters: kappa + p0 + p1 + omega0 + omega2 + branch lengths
    n_params = 5
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p0': p0,
            'p1': p1,
            'p2': 1.0 - p0 - p1,
            'omega0': omega0,
            'omega2': omega2,
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_m3_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M3 optimizer result: (kappa, omegas_list, proportions_list, lnL)"""
    kappa, omegas, proportions, lnL = result_tuple

    # Count parameters: kappa + (K-1) proportions + K omegas + branch lengths
    # where K is number of site classes (default 3)
    K = len(omegas)
    n_params = 1 + (K - 1) + K  # kappa + proportions + omegas
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'omegas': list(omegas),
            'proportions': list(proportions),
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_m7_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M7 optimizer result: (kappa, p_beta, q_beta, lnL)"""
    kappa, p_beta, q_beta, lnL = result_tuple

    # Count parameters: kappa + p + q + branch lengths
    n_params = 3
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p': p_beta,  # Use 'p' for consistency
            'q': q_beta,  # Use 'q' for consistency
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_m8_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M8 optimizer result: (kappa, p0, p_beta, q_beta, omega_s, lnL)"""
    kappa, p0, p_beta, q_beta, omega_s, lnL = result_tuple

    # Count parameters: kappa + p0 + p + q + omega_s + branch lengths
    n_params = 5
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p0': p0,
            'p': p_beta,  # Use 'p' for consistency with M7/M8a
            'q': q_beta,  # Use 'q' for consistency with M7/M8a
            'omega_s': omega_s,
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_m8a_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> SiteModelResult:
    """Parse M8a optimizer result: (kappa, p0, p_beta, q_beta, lnL)"""
    kappa, p0, p_beta, q_beta, lnL = result_tuple

    # Count parameters: kappa + p0 + p + q + branch lengths
    # Note: omega_s is fixed at 1.0, so not counted
    n_params = 4
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return SiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p0': p0,
            'p': p_beta,  # Use 'p' for consistency with M7/M8
            'q': q_beta,  # Use 'q' for consistency with M7/M8
            'omega_s': 1.0,  # Fixed at 1.0 for M8a (not optimized)
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_branch_model_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> BranchModelResult:
    """Parse branch model optimizer result: (kappa, omega_dict, lnL)"""
    kappa, omega_dict, lnL = result_tuple

    # Count parameters: kappa + number of omega parameters + branch lengths
    n_omegas = len(omega_dict)
    n_params = 1 + n_omegas  # kappa + omegas

    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1  # global branch scale

    return BranchModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={'omega_dict': omega_dict},
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


def _parse_branch_site_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> BranchSiteModelResult:
    """Parse branch-site model optimizer result: (kappa, omega0, omega2, p0, p1, lnL)"""
    kappa, omega0, omega2, p0, p1, lnL = result_tuple

    # Count parameters: kappa + omega0 + omega2 + p0 + p1 + branch lengths
    n_params = 5  # kappa + omega0 + omega2 + p0 + p1

    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1  # global branch scale

    return BranchSiteModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'omega0': omega0,
            'omega2': omega2,
            'p0': p0,
            'p1': p1,
        },
        tree=tree,
        alignment=alignment,
        n_params=n_params
    )


# =============================================================================
# File loading helpers with automatic format detection
# =============================================================================

def _load_alignment(alignment: Union[str, Path, Alignment]) -> Alignment:
    """
    Load alignment with automatic format detection.

    Tries to detect format based on:
    1. File extension (.fa/.fasta → FASTA, .phy/.phylip → PHYLIP)
    2. File content inspection (fallback)

    Parameters
    ----------
    alignment : str, Path, or Alignment
        Path to alignment file or Alignment object

    Returns
    -------
    Alignment
        Loaded alignment object

    Raises
    ------
    FileNotFoundError
        If the alignment file doesn't exist
    ValueError
        If the format cannot be detected or parsing fails
    """
    # Already an Alignment object
    if isinstance(alignment, Alignment):
        return alignment

    path = Path(alignment)

    if not path.exists():
        raise FileNotFoundError(f"Alignment file not found: {path}")

    # Try extension-based detection
    suffix = path.suffix.lower()

    if suffix in ('.fa', '.fasta', '.fna'):
        try:
            return Alignment.from_fasta(str(path), seqtype='codon')
        except Exception as e:
            raise ValueError(f"Failed to load FASTA alignment from {path}: {e}")

    elif suffix in ('.phy', '.phylip', '.txt'):
        # Try PHYLIP first for these extensions
        try:
            return Alignment.from_phylip(str(path), seqtype='codon')
        except:
            # Fallback to FASTA
            try:
                return Alignment.from_fasta(str(path), seqtype='codon')
            except Exception as e:
                raise ValueError(f"Failed to load alignment from {path}: {e}")

    else:
        # Unknown extension - try FASTA first (most common)
        try:
            return Alignment.from_fasta(str(path), seqtype='codon')
        except:
            try:
                return Alignment.from_phylip(str(path), seqtype='codon')
            except Exception as e:
                raise ValueError(
                    f"Failed to auto-detect alignment format for {path}. "
                    f"Please specify format explicitly using Alignment.from_fasta() "
                    f"or Alignment.from_phylip(). Error: {e}"
                )


def _load_tree(tree: Union[str, Path, Tree]) -> Tree:
    """
    Load tree from Newick file or string.

    Parameters
    ----------
    tree : str, Path, or Tree
        Path to tree file, Newick string, or Tree object

    Returns
    -------
    Tree
        Loaded tree object

    Raises
    ------
    ValueError
        If tree parsing fails
    """
    # Already a Tree object
    if isinstance(tree, Tree):
        return tree

    path_or_str = str(tree)

    # Check if it's a file path
    path = Path(path_or_str)
    if path.exists():
        with open(path) as f:
            newick_str = f.read().strip()
    else:
        # Assume it's a Newick string
        newick_str = path_or_str

    try:
        return Tree.from_newick(newick_str)
    except Exception as e:
        raise ValueError(f"Failed to parse tree: {e}")


# =============================================================================
# Main API function: optimize_model()
# =============================================================================

def optimize_model(
    model: str,
    alignment: Union[str, Path, Alignment],
    tree: Union[str, Path, Tree],
    use_f3x4: bool = True,
    optimize_branch_lengths: bool = True,
    **optimizer_kwargs
) -> SiteModelResult:
    """
    Optimize a codon model by name.

    This is the main entry point for fitting codon substitution models.
    It provides a unified interface across all models, with automatic
    file format detection and consistent result objects.

    Parameters
    ----------
    model : str
        Model name (case-insensitive):

        - "M0": One-ratio model (single omega for all sites)
        - "M1a": Nearly neutral (2 classes: omega<1 and omega=1)
        - "M2a": Positive selection (3 classes: omega<1, omega=1, omega>1)
        - "M3": Discrete (K=3 discrete omega classes)
        - "M7": Beta distribution (omega constrained to 0-1)
        - "M8": Beta + omega>1 (beta distribution + positive selection class)
        - "M8a": Beta + omega=1 (beta distribution + neutral class)

    alignment : str, Path, or Alignment
        Codon alignment. If string/Path, will auto-detect FASTA vs PHYLIP format.
        Supported formats:

        - FASTA (.fa, .fasta, .fna)
        - PHYLIP (.phy, .phylip, .txt)

    tree : str, Path, or Tree
        Phylogenetic tree in Newick format. Can be:

        - Path to file
        - Newick string
        - Tree object

    use_f3x4 : bool, default=True
        Use F3X4 codon frequency model (position-specific nucleotide frequencies).
        This is the recommended setting and matches PAML defaults.

    optimize_branch_lengths : bool, default=True
        Optimize individual branch lengths. If False, uses global scaling factor.

    **optimizer_kwargs
        Additional keyword arguments passed to the optimizer's optimize() method:

        - maxiter : int - Maximum iterations (default: 200 for most models)
        - method : str - Optimization method (default: 'L-BFGS-B')
        - init_kappa : float - Initial kappa value
        - Model-specific initialization parameters (see optimizer docs)

    Returns
    -------
    ModelResult
        Optimization results containing:

        - model_name : Model identifier
        - lnL : Log-likelihood
        - kappa : Transition/transversion ratio
        - params : Model-specific parameters
        - tree : Tree with optimized branch lengths
        - alignment : Input alignment
        - n_params : Number of free parameters

    Raises
    ------
    ValueError
        If model name is invalid or file format cannot be detected
    FileNotFoundError
        If alignment or tree file doesn't exist

    Examples
    --------
    Basic usage with file paths:

    >>> from crabml import optimize_model
    >>> result = optimize_model("M0", "alignment.fasta", "tree.nwk")
    >>> print(result.summary())
    >>> print(f"omega = {result.omega:.4f}")

    With custom optimization parameters:

    >>> result = optimize_model(
    ...     "M2a",
    ...     "alignment.fasta",
    ...     "tree.nwk",
    ...     maxiter=500,
    ...     init_omega2=3.0
    ... )
    >>> print(f"Positive selection omega = {result.params['omega2']:.2f}")

    Using Alignment and Tree objects directly:

    >>> from crabml.io import Alignment, Tree
    >>> align = Alignment.from_fasta("data.fasta", seqtype="codon")
    >>> tree = Tree.from_newick("tree.nwk")
    >>> result = optimize_model("M7", align, tree)

    Export results:

    >>> result.to_json("results.json")
    >>> data = result.to_dict()

    Notes
    -----
    All site-class models (M1a, M2a, M7, M8, M8a) automatically run M0
    optimization first to initialize branch lengths. This dramatically
    improves convergence, especially on gapped alignments.

    The tree object is modified in-place with optimized branch lengths.
    If you need to preserve the original tree, make a copy before calling
    this function.

    See Also
    --------
    positive_selection : Run hypothesis tests for positive selection
    m1a_vs_m2a : Test M1a vs M2a
    m7_vs_m8 : Test M7 vs M8

    References
    ----------
    Yang, Z. (2007). PAML 4: phylogenetic analysis by maximum likelihood.
    Molecular Biology and Evolution, 24(8), 1586-1591.
    """
    # 1. Load data with auto-detection
    align = _load_alignment(alignment)
    tree_obj = _load_tree(tree)

    # 2. Normalize model name
    model_upper = model.upper()

    # 3. Model registry with optimizer classes and parser functions
    OPTIMIZER_REGISTRY = {
        'M0': (M0Optimizer, _parse_m0_result),
        'M1A': (M1aOptimizer, _parse_m1a_result),
        'M2A': (M2aOptimizer, _parse_m2a_result),
        'M3': (M3Optimizer, _parse_m3_result),
        'M7': (M7Optimizer, _parse_m7_result),
        'M8': (M8Optimizer, _parse_m8_result),
        'M8A': (M8aOptimizer, _parse_m8a_result),
    }

    if model_upper not in OPTIMIZER_REGISTRY:
        valid_models = ', '.join(sorted(OPTIMIZER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model: '{model}'. "
            f"Valid models are: {valid_models}"
        )

    # 4. Create optimizer
    optimizer_class, parser_func = OPTIMIZER_REGISTRY[model_upper]

    # Separate init_with_m0 from other optimizer_kwargs
    init_with_m0 = optimizer_kwargs.pop('init_with_m0', True)  # Default True

    # Build optimizer kwargs - M0 and M3 don't accept init_with_m0
    optimizer_init_kwargs = {
        'use_f3x4': use_f3x4,
        'optimize_branch_lengths': optimize_branch_lengths,
    }
    if model_upper not in ('M0', 'M3'):
        optimizer_init_kwargs['init_with_m0'] = init_with_m0

    optimizer = optimizer_class(
        align,
        tree_obj,
        **optimizer_init_kwargs
    )

    # 5. Run optimization
    result_tuple = optimizer.optimize(**optimizer_kwargs)

    # 6. Parse result with model-specific parser
    model_result = parser_func(
        model_name=model_upper,
        result_tuple=result_tuple,
        tree=tree_obj,
        alignment=align,
        optimize_branch_lengths=optimize_branch_lengths
    )

    return model_result


def optimize_branch_model(
    model: str,
    alignment: Union[str, Path, Alignment],
    tree: Union[str, Path, Tree],
    use_f3x4: bool = True,
    optimize_branch_lengths: bool = True,
    **optimizer_kwargs
) -> BranchModelResult:
    """
    Optimize a branch codon model.

    Branch models allow omega (dN/dS) to vary across branches but not across sites.

    Parameters
    ----------
    model : str
        Model type (case-insensitive):

        - "free-ratio" or "model1": Independent omega for each branch
        - "multi-ratio" or "model2": Different omega for labeled branch groups

    alignment : str, Path, or Alignment
        Codon alignment (auto-detects FASTA vs PHYLIP format)

    tree : str, Path, or Tree
        Phylogenetic tree in Newick format.
        For multi-ratio model, use branch labels like #0, #1, #2 to specify groups.
        Example: "((human,chimp) #1, (mouse,rat) #0);" tests if primates have
        different omega than rodents.

    use_f3x4 : bool, default=True
        Use F3X4 codon frequency model

    optimize_branch_lengths : bool, default=True
        Optimize individual branch lengths

    **optimizer_kwargs
        Additional optimizer arguments (maxiter, etc.)

    Returns
    -------
    BranchModelResult
        Optimization results with branch-specific omega values

    Examples
    --------
    Test for lineage-specific selection:

    >>> from crabml import optimize_branch_model
    >>> tree_str = "((human,chimp) #1, (mouse,rat) #0);"
    >>> result = optimize_branch_model("multi-ratio", "align.fasta", tree_str)
    >>> print(f"Primate omega: {result.foreground_omega:.3f}")
    >>> print(f"Rodent omega: {result.background_omega:.3f}")

    Free-ratio model (exploratory analysis):

    >>> result = optimize_branch_model("free-ratio", "align.fasta", "tree.nwk")
    >>> print(result.summary())
    >>> print(result.omega_dict)

    See Also
    --------
    branch_model_test : Test for lineage-specific selection
    optimize_model : Fit site-class models
    """
    # 1. Load data
    align = _load_alignment(alignment)
    tree_obj = _load_tree(tree)

    # 2. Normalize model name
    model_lower = model.lower()

    if model_lower in ('free-ratio', 'model1'):
        free_ratio = True
        model_name = 'Free-ratio'
    elif model_lower in ('multi-ratio', 'model2'):
        free_ratio = False
        model_name = 'Multi-ratio'
    else:
        raise ValueError(
            f"Unknown branch model: '{model}'. "
            f"Valid models are: 'free-ratio', 'multi-ratio'"
        )

    # 3. Run M0 first to get good starting values (like PAML does)
    m0_optimizer = M0Optimizer(align, tree_obj, use_f3x4=use_f3x4)
    kappa_m0, omega_m0, lnL_m0 = m0_optimizer.optimize()

    # 4. Create branch optimizer (tree now has M0-optimized branch lengths)
    optimizer = BranchModelOptimizer(
        align,
        tree_obj,
        use_f3x4=use_f3x4,
        free_ratio=free_ratio,
        optimize_branch_lengths=optimize_branch_lengths
    )

    # 5. Run optimization starting from M0 estimates
    optimizer_kwargs.setdefault('init_kappa', kappa_m0)
    optimizer_kwargs.setdefault('init_omega', omega_m0)
    result_tuple = optimizer.optimize(**optimizer_kwargs)

    # 6. Parse result
    return _parse_branch_model_result(
        model_name=model_name,
        result_tuple=result_tuple,
        tree=tree_obj,
        alignment=align,
        optimize_branch_lengths=optimize_branch_lengths
    )


def optimize_branch_site_model(
    model: str,
    alignment: Union[str, Path, Alignment],
    tree: Union[str, Path, Tree],
    use_f3x4: bool = True,
    optimize_branch_lengths: bool = True,
    fix_omega: bool = False,
    **optimizer_kwargs
) -> BranchSiteModelResult:
    """
    Optimize a branch-site codon model.

    Branch-site models allow omega (dN/dS) to vary across both sites AND branches,
    enabling detection of positive selection on specific lineages.

    Parameters
    ----------
    model : str
        Model type (case-insensitive):

        - "model-a" or "branch-site-a": Branch-Site Model A

    alignment : str, Path, or Alignment
        Codon alignment (auto-detects FASTA vs PHYLIP format)

    tree : str, Path, or Tree
        Phylogenetic tree in Newick format with branch labels.
        Use #0 for background branches and #1 for foreground branches.
        Example: "((human,chimp) #1, (mouse,rat) #0);" tests for positive
        selection on the primate lineage.

    use_f3x4 : bool, default=True
        Use F3X4 codon frequency model

    optimize_branch_lengths : bool, default=True
        Optimize individual branch lengths

    fix_omega : bool, default=False
        Fix omega2 to 1.0 for the null model.
        If False, omega2 is free to vary (alternative model).

    **optimizer_kwargs
        Additional optimizer arguments (maxiter, init_omega2, etc.)

    Returns
    -------
    BranchSiteModelResult
        Optimization results with site-class parameters

    Examples
    --------
    Test for positive selection on primates:

    >>> from crabml import optimize_branch_site_model
    >>> tree_str = "((human,chimp) #1, (mouse,rat) #0);"
    >>> result = optimize_branch_site_model("model-a", "align.fasta", tree_str)
    >>> print(f"Positive selection omega: {result.omega2:.3f}")
    >>> print(f"Proportion under selection: {result.foreground_positive_proportion:.3f}")

    Fit null model (omega2 = 1):

    >>> null = optimize_branch_site_model("model-a", "align.fasta", tree_str, fix_omega=True)

    See Also
    --------
    branch_site_test : Test for positive selection on foreground branches
    optimize_model : Fit site-class models
    """
    # 1. Load data
    align = _load_alignment(alignment)
    tree_obj = _load_tree(tree)

    # 2. Normalize model name
    model_lower = model.lower()

    if model_lower in ('model-a', 'branch-site-a', 'modela'):
        model_name = 'Branch-Site Model A'
    else:
        raise ValueError(
            f"Unknown branch-site model: '{model}'. "
            f"Valid models are: 'model-a', 'branch-site-a'"
        )

    # 3. Create optimizer
    optimizer = BranchSiteModelAOptimizer(
        align,
        tree_obj,
        use_f3x4=use_f3x4,
        optimize_branch_lengths=optimize_branch_lengths,
        fix_omega=fix_omega
    )

    # 4. Run optimization
    result_tuple = optimizer.optimize(**optimizer_kwargs)

    # 5. Parse result
    result = _parse_branch_site_result(
        model_name=model_name,
        result_tuple=result_tuple,
        tree=tree_obj,
        alignment=align,
        optimize_branch_lengths=optimize_branch_lengths
    )

    # Add null/alt information to model name if omega2 is fixed
    if fix_omega:
        result.model_name = f"{model_name} (null, ω₂=1)"

    return result
