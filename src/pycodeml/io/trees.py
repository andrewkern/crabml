"""
Phylogenetic tree parsing and manipulation.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TreeNode:
    """
    Phylogenetic tree node.

    Attributes
    ----------
    id : int
        Node identifier
    name : Optional[str]
        Node name (for leaves)
    parent : Optional[TreeNode]
        Parent node
    children : list[TreeNode]
        Child nodes
    branch_length : float
        Branch length to parent
    label : Optional[str]
        Branch label (e.g., '#1' for model specification)
    """

    id: int
    name: Optional[str] = None
    parent: Optional["TreeNode"] = None
    children: list["TreeNode"] = field(default_factory=list)
    branch_length: float = 0.0
    label: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0


@dataclass
class Tree:
    """
    Phylogenetic tree.

    Attributes
    ----------
    root : TreeNode
        Root node of the tree
    n_nodes : int
        Total number of nodes
    n_leaves : int
        Number of leaf nodes
    leaf_names : list[str]
        Names of leaf nodes
    """

    root: TreeNode
    n_nodes: int
    n_leaves: int
    leaf_names: list[str]

    @classmethod
    def from_newick(cls, newick_string: str) -> "Tree":
        """
        Parse Newick format tree string.

        Parameters
        ----------
        newick_string : str
            Newick format tree

        Returns
        -------
        Tree
            Parsed tree
        """
        # Clean the input: remove comments and whitespace
        # Remove // comments
        newick = re.sub(r'//.*', '', newick_string)
        # Remove /* */ style comments (PAML uses / * with space)
        newick = re.sub(r'/\s*\*.*?\*\s*/', '', newick)
        # Remove extra whitespace but preserve species names
        newick = newick.strip()

        # Find the tree string (ends with semicolon)
        if ';' not in newick:
            raise ValueError("Invalid Newick format: missing semicolon")

        # Extract just the tree part (first line with semicolon)
        lines = newick.split('\n')
        tree_line = None
        for line in lines:
            if ';' in line:
                tree_line = line[:line.index(';') + 1]
                break

        if tree_line is None:
            raise ValueError("Invalid Newick format: no tree found")

        # Remove remaining whitespace from tree line
        tree_line = re.sub(r'\s+', '', tree_line)

        # Parse the tree recursively
        node_id_counter = [0]  # Use list for mutable counter

        def parse_node(s: str, start: int, parent: Optional[TreeNode] = None) -> tuple[TreeNode, int]:
            """Parse a node from position start in string s."""
            node = TreeNode(id=node_id_counter[0])
            node_id_counter[0] += 1
            node.parent = parent
            pos = start

            # Check if this is an internal node (starts with '(')
            if pos < len(s) and s[pos] == '(':
                pos += 1  # skip '('
                # Parse children
                while True:
                    child, pos = parse_node(s, pos, node)
                    node.children.append(child)

                    if pos < len(s) and s[pos] == ',':
                        pos += 1  # skip ','
                        continue
                    elif pos < len(s) and s[pos] == ')':
                        pos += 1  # skip ')'
                        break
                    else:
                        raise ValueError(f"Expected ',' or ')' at position {pos}")

            # Parse node name/label (for leaves or labeled internal nodes)
            name_start = pos
            while pos < len(s) and s[pos] not in ',:();#':
                pos += 1
            if pos > name_start:
                node.name = s[name_start:pos]

            # Parse branch label (e.g., #1, #2)
            if pos < len(s) and s[pos] == '#':
                pos += 1
                label_start = pos
                while pos < len(s) and s[pos] not in ',:();':
                    pos += 1
                node.label = '#' + s[label_start:pos]

            # Parse branch length (e.g., :0.123)
            if pos < len(s) and s[pos] == ':':
                pos += 1
                length_start = pos
                while pos < len(s) and s[pos] not in ',();':
                    pos += 1
                try:
                    node.branch_length = float(s[length_start:pos])
                except ValueError:
                    raise ValueError(f"Invalid branch length: {s[length_start:pos]}")

            return node, pos

        # Parse the root
        root, pos = parse_node(tree_line, 0, None)

        # Count nodes and leaves
        def count_nodes(node: TreeNode) -> tuple[int, int, list[str]]:
            """Count total nodes, leaves, and collect leaf names."""
            if node.is_leaf:
                leaf_name = node.name if node.name else str(node.id)
                return 1, 1, [leaf_name]
            else:
                total_nodes = 1
                total_leaves = 0
                leaf_names = []
                for child in node.children:
                    n, l, names = count_nodes(child)
                    total_nodes += n
                    total_leaves += l
                    leaf_names.extend(names)
                return total_nodes, total_leaves, leaf_names

        n_nodes, n_leaves, leaf_names = count_nodes(root)

        return cls(
            root=root,
            n_nodes=n_nodes,
            n_leaves=n_leaves,
            leaf_names=leaf_names
        )

    def postorder(self) -> list[TreeNode]:
        """
        Return nodes in post-order traversal (leaves to root).

        Returns
        -------
        list[TreeNode]
            Nodes in post-order
        """
        result = []

        def traverse(node: TreeNode) -> None:
            """Recursively traverse in post-order."""
            for child in node.children:
                traverse(child)
            result.append(node)

        traverse(self.root)
        return result
