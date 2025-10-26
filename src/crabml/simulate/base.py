"""
Base class for sequence simulators.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np

from ..io.trees import Tree


class SequenceSimulator(ABC):
    """
    Abstract base class for sequence simulators.

    This class defines the interface for simulating molecular sequences
    under evolutionary models. Subclasses implement specific models
    (e.g., M0, M1a, M2a).

    Parameters
    ----------
    tree : Tree
        Phylogenetic tree with branch lengths (must be rooted)
    sequence_length : int
        Length of sequence to simulate (in codons for codon models)
    seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    tree : Tree
        The phylogenetic tree
    sequence_length : int
        Sequence length
    rng : numpy.random.Generator
        Random number generator (seeded for reproducibility)
    """

    def __init__(
        self,
        tree: Tree,
        sequence_length: int,
        seed: Optional[int] = None
    ):
        self.tree = tree
        self.sequence_length = sequence_length

        # Create seeded random number generator (new API, faster than RandomState)
        self.rng = np.random.default_rng(seed)

        # Validate tree
        self._validate_tree()

    def _validate_tree(self):
        """Ensure tree is suitable for simulation."""
        # Check if tree is rooted
        if self.tree.root is None:
            raise ValueError("Tree must be rooted for simulation")

        # Check if tree has branch lengths
        for node in self.tree.postorder():
            if node.parent is not None and node.branch_length is None:
                raise ValueError(
                    f"Node {node.name if node.name else 'unnamed'} missing branch length"
                )

    @abstractmethod
    def _generate_ancestral_sequence(self) -> np.ndarray:
        """
        Generate sequence at root node.

        Returns
        -------
        np.ndarray
            Ancestral sequence (array of state indices)
        """
        pass

    @abstractmethod
    def _evolve_sequence(
        self,
        parent_seq: np.ndarray,
        branch_length: float,
        **kwargs
    ) -> np.ndarray:
        """
        Evolve sequence along a branch.

        Parameters
        ----------
        parent_seq : np.ndarray
            Parent sequence (array of state indices)
        branch_length : float
            Length of branch
        **kwargs
            Additional model-specific parameters

        Returns
        -------
        np.ndarray
            Child sequence (array of state indices)
        """
        pass

    def simulate(self) -> Dict[str, np.ndarray]:
        """
        Simulate sequences on the tree.

        This is the main simulation algorithm:
        1. Generate ancestral sequence at root
        2. Traverse tree from root to tips
        3. Evolve sequences along each branch
        4. Return tip sequences

        Returns
        -------
        dict
            Mapping from species name to sequence array.
            Keys are species names (str), values are arrays of state indices.
        """
        # Generate root sequence
        root_seq = self._generate_ancestral_sequence()

        # Store sequences at all nodes using node id() as key (hashable)
        # This allows us to store ancestral sequences for potential output
        sequences = {id(self.tree.root): root_seq}

        # Recursively traverse tree from root to tips
        def evolve_subtree(node):
            """Recursively evolve sequences from parent to children."""
            for child in node.children:
                # Get parent sequence
                parent_seq = sequences[id(node)]

                # Evolve along branch
                child_seq = self._evolve_sequence(parent_seq, child.branch_length)

                # Store child sequence
                sequences[id(child)] = child_seq

                # Recursively process child's subtree
                evolve_subtree(child)

        # Start recursive traversal from root
        evolve_subtree(self.tree.root)

        # Extract tip sequences (only leaf nodes)
        tip_sequences = {}
        for node in self.tree.postorder():
            # Check if leaf (no children)
            if not node.children:
                # Use node name or generate one if missing
                name = node.name if node.name else f"seq_{id(node)}"
                tip_sequences[name] = sequences[id(node)]

        return tip_sequences

    @abstractmethod
    def get_parameters(self) -> Dict:
        """
        Get simulation parameters for output metadata.

        Returns
        -------
        dict
            Dictionary of model parameters
        """
        pass
