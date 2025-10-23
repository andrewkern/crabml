"""
Input/Output modules for sequence alignments and phylogenetic trees.

This module provides classes for reading and working with:

- **Sequence alignments**: FASTA and PHYLIP formats
- **Phylogenetic trees**: Newick format

The main classes handle file parsing, format detection, and data validation.
"""

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree, TreeNode

__all__ = ["Alignment", "Tree", "TreeNode"]
