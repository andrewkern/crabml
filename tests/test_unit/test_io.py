"""
Unit tests for I/O modules (sequence and tree parsing).
"""

import numpy as np
import pytest
from pathlib import Path

from crabml.io.sequences import Alignment, CODON_TO_INDEX
from crabml.io.trees import Tree, TreeNode


class TestSequenceParsing:
    """Test PHYLIP format sequence parsing."""

    def test_parse_lysozyme_alignment(self, lysozyme_small_files):
        """Test parsing lysozyme sequence file."""
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        assert aln.n_species == 7
        assert aln.n_sites == 130  # 390 nucleotides / 3
        assert len(aln.names) == 7
        assert aln.names[0] == "Hsa_Human"
        assert aln.seqtype == "codon"

        # Verify sequence encoding
        assert aln.sequences.dtype == np.int8
        assert aln.sequences.shape == (7, 130)
        assert np.all(aln.sequences >= -1)  # -1 for invalid codons
        assert np.all(aln.sequences < 61)  # 61 sense codons (0-60)

    def test_sequence_names(self, lysozyme_small_files):
        """Test that all sequence names are parsed correctly."""
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        expected_names = [
            "Hsa_Human",
            "Hla_gibbon",
            "Cgu/Can_colobus",
            "Pne_langur",
            "Mmu_rhesus",
            "Ssc_squirrelM",
            "Cja_marmoset",
        ]

        assert aln.names == expected_names

    def test_codon_encoding(self):
        """Test codon encoding/decoding."""
        # Create simple test sequences
        sequences = ["ATGATGCCC", "TTTAAAGGG"]  # 3 codons each

        # Manually create alignment
        from crabml.io.sequences import INDEX_TO_CODON

        encoded = Alignment._encode_codons(sequences)

        assert encoded.shape == (2, 3)
        assert encoded[0, 0] == CODON_TO_INDEX["ATG"]
        assert encoded[0, 1] == CODON_TO_INDEX["ATG"]
        assert encoded[0, 2] == CODON_TO_INDEX["CCC"]
        assert encoded[1, 0] == CODON_TO_INDEX["TTT"]
        assert encoded[1, 1] == CODON_TO_INDEX["AAA"]
        assert encoded[1, 2] == CODON_TO_INDEX["GGG"]

        # Test round-trip
        for i, seq_encoded in enumerate(encoded):
            reconstructed = "".join(INDEX_TO_CODON[idx] for idx in seq_encoded)
            assert reconstructed == sequences[i]

    def test_invalid_codon_length(self, tmp_path):
        """Test error handling for sequences not divisible by 3."""
        # Create test file with invalid length
        test_file = tmp_path / "invalid.txt"
        with open(test_file, "w") as f:
            f.write("  2   10\n")  # 10 nucleotides - not divisible by 3
            f.write("seq1\nATGATGATGA\n")
            f.write("seq2\nTTTTTTTTTT\n")

        with pytest.raises(ValueError, match="not divisible by 3"):
            Alignment.from_phylip(test_file, seqtype="codon")

    def test_write_read_round_trip(self, tmp_path):
        """Test that we can write and read back an alignment."""
        # Create simple alignment
        names = ["seq1", "seq2"]
        sequences = np.array(
            [
                [CODON_TO_INDEX["ATG"], CODON_TO_INDEX["CCC"]],
                [CODON_TO_INDEX["TTT"], CODON_TO_INDEX["GGG"]],
            ],
            dtype=np.int8,
        )

        aln = Alignment(
            names=names,
            sequences=sequences,
            n_species=2,
            n_sites=2,
            seqtype="codon",
        )

        # Write
        output_file = tmp_path / "test.phy"
        aln.to_phylip(output_file)

        # Read back
        aln2 = Alignment.from_phylip(output_file, seqtype="codon")

        # Compare
        assert aln2.names == aln.names
        assert aln2.n_species == aln.n_species
        assert aln2.n_sites == aln.n_sites
        np.testing.assert_array_equal(aln2.sequences, aln.sequences)

    def test_repr(self):
        """Test string representation."""
        aln = Alignment(
            names=["a", "b"],
            sequences=np.zeros((2, 10), dtype=np.int8),
            n_species=2,
            n_sites=10,
            seqtype="codon",
        )

        repr_str = repr(aln)
        assert "n_species=2" in repr_str
        assert "n_sites=10" in repr_str
        assert "seqtype='codon'" in repr_str


class TestTreeParsing:
    """Test Newick format tree parsing."""

    def test_simple_tree(self):
        """Test parsing a simple 3-taxon tree."""
        newick = "((1,2),3);"
        tree = Tree.from_newick(newick)

        assert tree.n_nodes == 5  # 3 leaves + 2 internal nodes
        assert tree.n_leaves == 3
        assert tree.leaf_names == ["1", "2", "3"]
        assert not tree.root.is_leaf
        assert len(tree.root.children) == 2

    def test_tree_with_branch_lengths(self):
        """Test parsing tree with branch lengths."""
        newick = "((1:0.1, 2:0.2):0.12, 3:0.3, 4:0.4);"
        tree = Tree.from_newick(newick)

        assert tree.n_leaves == 4
        assert tree.leaf_names == ["1", "2", "3", "4"]

        # Find leaf nodes and check branch lengths
        leaves = [node for node in tree.postorder() if node.is_leaf]
        leaf_dict = {node.name: node for node in leaves}

        assert leaf_dict["1"].branch_length == pytest.approx(0.1)
        assert leaf_dict["2"].branch_length == pytest.approx(0.2)
        assert leaf_dict["3"].branch_length == pytest.approx(0.3)
        assert leaf_dict["4"].branch_length == pytest.approx(0.4)

        # Check internal node branch length
        internal_nodes = [node for node in tree.postorder() if not node.is_leaf]
        # The first internal node (parent of 1,2) should have branch length 0.12
        parent_of_1_2 = leaf_dict["1"].parent
        assert parent_of_1_2.branch_length == pytest.approx(0.12)

    def test_tree_with_labels(self):
        """Test parsing tree with branch labels (PAML model specification)."""
        newick = "((1,2) #1, ((3,4), 5), (6,7));"
        tree = Tree.from_newick(newick)

        assert tree.n_leaves == 7

        # Find the node with label #1
        nodes = tree.postorder()
        labeled_nodes = [node for node in nodes if node.label is not None]
        assert len(labeled_nodes) == 1
        assert labeled_nodes[0].label == "#1"

        # Verify it's the parent of leaves 1 and 2
        leaves = [node for node in nodes if node.is_leaf]
        leaf_dict = {node.name: node for node in leaves}
        parent_of_1_2 = leaf_dict["1"].parent
        assert parent_of_1_2.label == "#1"

    def test_tree_with_species_names(self):
        """Test parsing tree with actual species names."""
        newick = "((Hsa_Human, Hla_gibbon), (Cgu/Can_colobus, Pne_langur));"
        tree = Tree.from_newick(newick)

        assert tree.n_leaves == 4
        expected_names = ["Hsa_Human", "Hla_gibbon", "Cgu/Can_colobus", "Pne_langur"]
        assert tree.leaf_names == expected_names

    def test_postorder_traversal(self):
        """Test post-order traversal returns leaves before parents."""
        newick = "((1,2),3);"
        tree = Tree.from_newick(newick)

        nodes = tree.postorder()
        assert len(nodes) == 5

        # In post-order, children come before parents
        # So we should see leaves first, then internal nodes, then root
        leaves_seen = []
        for node in nodes:
            if node.is_leaf:
                leaves_seen.append(node.name)
            else:
                # All children should already be in the list
                for child in node.children:
                    assert child in nodes[:nodes.index(node)]

        # Root should be last
        assert nodes[-1] == tree.root

    def test_lysozyme_tree(self, lysozyme_small_files):
        """Test parsing the lysozyme tree file."""
        tree_file = lysozyme_small_files["tree"]

        # Read the file and parse the first tree
        with open(tree_file) as f:
            content = f.read()

        # The file has multiple trees, let's parse one with species names
        # Line 11: ((Hsa_Human, Hla_gibbon),((Cgu/Can_colobus, Pne_langur) #1, Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));
        newick = "((Hsa_Human, Hla_gibbon),((Cgu/Can_colobus, Pne_langur) #1, Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));"
        tree = Tree.from_newick(newick)

        assert tree.n_leaves == 7
        expected_names = [
            "Hsa_Human",
            "Hla_gibbon",
            "Cgu/Can_colobus",
            "Pne_langur",
            "Mmu_rhesus",
            "Ssc_squirrelM",
            "Cja_marmoset",
        ]
        assert tree.leaf_names == expected_names

        # Check that the branch label is present
        nodes = tree.postorder()
        labeled_nodes = [node for node in nodes if node.label == "#1"]
        assert len(labeled_nodes) == 1

    def test_comments_removed(self):
        """Test that comments are properly removed."""
        newick = "((1,2),3); // this is a comment"
        tree = Tree.from_newick(newick)
        assert tree.n_leaves == 3

        newick2 = "((1,2),3); / * another comment * /"
        tree2 = Tree.from_newick(newick2)
        assert tree2.n_leaves == 3

    def test_multiline_format(self):
        """Test parsing trees from multiline format files."""
        multiline = """  3  4

(1,2,3);
((1,2),3);"""
        tree = Tree.from_newick(multiline)
        # Should parse the first tree
        assert tree.n_leaves == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
