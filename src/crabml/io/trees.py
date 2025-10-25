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

        # Extract just the tree part (all lines with content until semicolon)
        lines = newick.split('\n')
        tree_lines = []
        for line in lines:
            # Skip lines that look like PAML headers (just numbers)
            if line.strip() and not re.match(r'^\s*\d+\s+\d+\s*$', line):
                tree_lines.append(line)
                if ';' in line:
                    break

        if not tree_lines:
            raise ValueError("Invalid Newick format: no tree found")

        # Join all tree lines and remove newlines/tabs (but keep spaces before #)
        tree_line = ''.join(tree_lines)
        # Remove only newlines and tabs, preserve spaces
        tree_line = tree_line.replace('\n', '').replace('\t', '').replace('\r', '')

        # Parse the tree recursively
        node_id_counter = [0]  # Use list for mutable counter

        def skip_whitespace(s: str, pos: int) -> int:
            """Skip whitespace characters."""
            while pos < len(s) and s[pos] in ' \t\n\r':
                pos += 1
            return pos

        def parse_node(s: str, start: int, parent: Optional[TreeNode] = None) -> tuple[TreeNode, int]:
            """Parse a node from position start in string s."""
            node = TreeNode(id=node_id_counter[0])
            node_id_counter[0] += 1
            node.parent = parent
            pos = skip_whitespace(s, start)

            # Check if this is an internal node (starts with '(')
            if pos < len(s) and s[pos] == '(':
                pos = skip_whitespace(s, pos + 1)  # skip '(' and whitespace
                # Parse children
                while True:
                    child, pos = parse_node(s, pos, node)
                    node.children.append(child)
                    pos = skip_whitespace(s, pos)

                    if pos < len(s) and s[pos] == ',':
                        pos = skip_whitespace(s, pos + 1)  # skip ',' and whitespace
                        continue
                    elif pos < len(s) and s[pos] == ')':
                        pos = skip_whitespace(s, pos + 1)  # skip ')' and whitespace
                        break
                    else:
                        raise ValueError(f"Expected ',' or ')' at position {pos}")

            # Parse node name/label (for leaves or labeled internal nodes)
            name_start = pos
            while pos < len(s) and s[pos] not in ',:();# \t\n\r':
                pos += 1
            if pos > name_start:
                node.name = s[name_start:pos]

            pos = skip_whitespace(s, pos)

            # Parse branch label (e.g., #1, #2)
            if pos < len(s) and s[pos] == '#':
                pos += 1
                label_start = pos
                while pos < len(s) and s[pos] not in ',:(); \t\n\r':
                    pos += 1
                node.label = '#' + s[label_start:pos]

            pos = skip_whitespace(s, pos)

            # Parse branch length (e.g., :0.123 or : 0.123)
            if pos < len(s) and s[pos] == ':':
                pos += 1
                pos = skip_whitespace(s, pos)  # Skip whitespace after colon
                length_start = pos
                while pos < len(s) and s[pos] not in ',(); \t\n\r':
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

    def get_branches(self) -> list[tuple[TreeNode, TreeNode]]:
        """
        Get all branches as (parent, child) pairs.

        Returns
        -------
        list[tuple[TreeNode, TreeNode]]
            List of (parent, child) tuples for each branch
        """
        branches = []

        def traverse(node: TreeNode) -> None:
            """Recursively collect branches."""
            for child in node.children:
                branches.append((node, child))
                traverse(child)

        traverse(self.root)
        return branches

    def get_branch_labels(self) -> list[int]:
        """
        Get integer branch labels for branch-site models.

        Converts string labels like '#0', '#1' to integers.
        Branches without labels are assigned 0 (background).

        Returns
        -------
        list[int]
            Branch labels as integers (0=background, 1=foreground, etc.)
        """
        branches = self.get_branches()
        labels = []

        for parent, child in branches:
            if child.label is not None:
                # Parse label like '#1' -> 1
                label_str = child.label.lstrip('#')
                try:
                    labels.append(int(label_str))
                except ValueError:
                    raise ValueError(f"Invalid branch label: {child.label}")
            else:
                # Default to background (0)
                labels.append(0)

        return labels

    def validate_branch_site_labels(self) -> None:
        """
        Validate branch labels for branch-site models.

        Branch-site models (Model A, A1) require exactly 2 label types:
        - 0 (background)
        - 1 (foreground)

        Raises
        ------
        ValueError
            If labels are not valid for branch-site models
        """
        labels = self.get_branch_labels()
        unique_labels = sorted(set(labels))

        if unique_labels != [0, 1]:
            raise ValueError(
                f"Branch-site models require exactly 2 label types (0 and 1). "
                f"Found: {unique_labels}. "
                f"Mark foreground branches with '#1' in the tree."
            )

        n_foreground = sum(1 for label in labels if label == 1)
        if n_foreground == 0:
            raise ValueError("No foreground branches marked with '#1'")

        print(f"✓ Tree validation passed: {n_foreground} foreground branch(es) marked")
