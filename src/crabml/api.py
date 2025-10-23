"""
High-level API for crabML codon model optimization.

This module provides a simplified interface for fitting codon models,
with unified result objects and automatic file format detection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json

from .io.sequences import Alignment
from .io.trees import Tree
from .optimize.optimizer import (
    M0Optimizer, M1aOptimizer, M2aOptimizer, M3Optimizer,
    M7Optimizer, M8Optimizer, M8aOptimizer
)


@dataclass
class ModelResult:
    """
    Unified result object for codon model optimization.

    This class provides a consistent interface for accessing results
    from any codon model, abstracting away the differences in parameter
    sets across models.

    Attributes
    ----------
    model_name : str
        Name of the codon model (e.g., "M0", "M1a", "M2a")
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

    model_name: str
    lnL: float
    kappa: float
    params: Dict[str, Any]
    tree: Tree
    alignment: Alignment
    n_params: int
    convergence_info: Optional[Dict[str, Any]] = None

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
            lines.append(f"    p = {self.params['p_beta']:.4f}")
            lines.append(f"    q = {self.params['q_beta']:.4f}")

        elif self.model_name == 'M8':
            lines.append(f"  Beta distribution:")
            lines.append(f"    p = {self.params['p_beta']:.4f}")
            lines.append(f"    q = {self.params['q_beta']:.4f}")
            lines.append(f"  Additional site class:")
            lines.append(f"    omega_s = {self.params['omega_s']:.4f}")
            lines.append(f"    proportion = {1-self.params['p0']:.4f}")

        elif self.model_name == 'M8A':
            lines.append(f"  Beta distribution:")
            lines.append(f"    p = {self.params['p_beta']:.4f}")
            lines.append(f"    q = {self.params['q_beta']:.4f}")
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
        json_str = json.dumps(result_dict, indent=indent)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def __str__(self) -> str:
        """String representation shows summary."""
        return self.summary()

    def __repr__(self) -> str:
        """Concise representation for interactive use."""
        return f"ModelResult(model='{self.model_name}', lnL={self.lnL:.2f}, kappa={self.kappa:.2f})"


# =============================================================================
# Parser functions: Convert optimizer result tuples to ModelResult objects
# =============================================================================

def _parse_m0_result(
    model_name: str,
    result_tuple: tuple,
    tree: Tree,
    alignment: Alignment,
    optimize_branch_lengths: bool
) -> ModelResult:
    """Parse M0 optimizer result: (kappa, omega, lnL)"""
    kappa, omega, lnL = result_tuple

    # Count parameters: kappa + omega + branch lengths
    n_params = 2
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1  # global branch scale

    return ModelResult(
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
) -> ModelResult:
    """Parse M1a optimizer result: (kappa, p0, omega0, lnL)"""
    kappa, p0, omega0, lnL = result_tuple

    # Count parameters: kappa + p0 + omega0 + branch lengths
    n_params = 3
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return ModelResult(
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
) -> ModelResult:
    """Parse M2a optimizer result: (kappa, p0, p1, omega0, omega2, lnL)"""
    kappa, p0, p1, omega0, omega2, lnL = result_tuple

    # Count parameters: kappa + p0 + p1 + omega0 + omega2 + branch lengths
    n_params = 5
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return ModelResult(
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
) -> ModelResult:
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

    return ModelResult(
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
) -> ModelResult:
    """Parse M7 optimizer result: (kappa, p_beta, q_beta, lnL)"""
    kappa, p_beta, q_beta, lnL = result_tuple

    # Count parameters: kappa + p + q + branch lengths
    n_params = 3
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return ModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p_beta': p_beta,
            'q_beta': q_beta,
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
) -> ModelResult:
    """Parse M8 optimizer result: (kappa, p0, p_beta, q_beta, omega_s, lnL)"""
    kappa, p0, p_beta, q_beta, omega_s, lnL = result_tuple

    # Count parameters: kappa + p0 + p + q + omega_s + branch lengths
    n_params = 5
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return ModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p0': p0,
            'p_beta': p_beta,
            'q_beta': q_beta,
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
) -> ModelResult:
    """Parse M8a optimizer result: (kappa, p0, p_beta, q_beta, lnL)"""
    kappa, p0, p_beta, q_beta, lnL = result_tuple

    # Count parameters: kappa + p0 + p + q + branch lengths
    # Note: omega_s is fixed at 1.0, so not counted
    n_params = 4
    if optimize_branch_lengths:
        n_params += sum(1 for node in tree.postorder() if node.parent is not None)
    else:
        n_params += 1

    return ModelResult(
        model_name=model_name,
        lnL=lnL,
        kappa=kappa,
        params={
            'p0': p0,
            'p_beta': p_beta,
            'q_beta': q_beta,
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
) -> ModelResult:
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

    optimizer = optimizer_class(
        align,
        tree_obj,
        use_f3x4=use_f3x4,
        optimize_branch_lengths=optimize_branch_lengths
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
