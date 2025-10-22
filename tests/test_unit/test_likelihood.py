"""
Unit tests for likelihood calculation.
"""

import numpy as np
import pytest

from crabml.core.likelihood import LikelihoodCalculator
from crabml.core.matrix import create_reversible_Q
from crabml.io.sequences import Alignment, CODON_TO_INDEX
from crabml.io.trees import Tree


class TestLikelihoodCalculator:
    """Test likelihood calculation."""

    def test_simple_two_sequence_tree(self):
        """Test likelihood on a simple two-sequence tree."""
        # Create simple alignment: two sequences, one codon site
        # Both sequences have ATG (methionine)
        names = ["seq1", "seq2"]
        atg_idx = CODON_TO_INDEX["ATG"]
        sequences = np.array([[atg_idx], [atg_idx]], dtype=np.int8)

        aln = Alignment(
            names=names,
            sequences=sequences,
            n_species=2,
            n_sites=1,
            seqtype="codon",
        )

        # Simple tree: (seq1:0.1, seq2:0.1);
        tree = Tree.from_newick("(seq1:0.1, seq2:0.1);")

        # Create simple JC69-like model for codons (equal rates)
        pi = np.ones(61) / 61  # Uniform frequencies
        rates = np.ones((61, 61))  # Equal exchangeability
        Q = create_reversible_Q(rates, pi, normalize=True)

        # Compute likelihood
        calc = LikelihoodCalculator(aln, tree)
        log_likelihood = calc.compute_log_likelihood(Q, pi)

        # With identical sequences and short branches, likelihood should be high
        # (i.e., log-likelihood should be close to 0, not very negative)
        assert log_likelihood < 0  # Log-likelihood is always <= 0
        assert log_likelihood > -10  # Should be reasonably high

    def test_different_sequences_lower_likelihood(self):
        """Test that different sequences give lower likelihood than identical ones."""
        # Create two alignments: one with identical sequences, one with different
        names = ["seq1", "seq2"]

        # Identical sequences: ATG, ATG
        atg_idx = CODON_TO_INDEX["ATG"]
        aaa_idx = CODON_TO_INDEX["AAA"]

        identical_seqs = np.array([[atg_idx], [atg_idx]], dtype=np.int8)
        different_seqs = np.array([[atg_idx], [aaa_idx]], dtype=np.int8)

        aln_identical = Alignment(
            names=names,
            sequences=identical_seqs,
            n_species=2,
            n_sites=1,
            seqtype="codon",
        )

        aln_different = Alignment(
            names=names,
            sequences=different_seqs,
            n_species=2,
            n_sites=1,
            seqtype="codon",
        )

        # Same tree for both
        tree = Tree.from_newick("(seq1:0.1, seq2:0.1);")

        # Same model
        pi = np.ones(61) / 61
        rates = np.ones((61, 61))
        Q = create_reversible_Q(rates, pi, normalize=True)

        # Compute likelihoods
        calc_identical = LikelihoodCalculator(aln_identical, tree)
        calc_different = LikelihoodCalculator(aln_different, tree)

        ll_identical = calc_identical.compute_log_likelihood(Q, pi)
        ll_different = calc_different.compute_log_likelihood(Q, pi)

        # Identical sequences should have higher likelihood
        assert ll_identical > ll_different

    def test_longer_branch_lower_likelihood_for_identical(self):
        """Test that longer branches give lower likelihood for identical sequences."""
        # With identical sequences, longer branches are less likely
        names = ["seq1", "seq2"]
        atg_idx = CODON_TO_INDEX["ATG"]
        sequences = np.array([[atg_idx], [atg_idx]], dtype=np.int8)

        aln = Alignment(
            names=names,
            sequences=sequences,
            n_species=2,
            n_sites=1,
            seqtype="codon",
        )

        # Two trees with different branch lengths
        tree_short = Tree.from_newick("(seq1:0.01, seq2:0.01);")
        tree_long = Tree.from_newick("(seq1:1.0, seq2:1.0);")

        # Same model
        pi = np.ones(61) / 61
        rates = np.ones((61, 61))
        Q = create_reversible_Q(rates, pi, normalize=True)

        # Compute likelihoods
        calc_short = LikelihoodCalculator(aln, tree_short)
        calc_long = LikelihoodCalculator(aln, tree_long)

        ll_short = calc_short.compute_log_likelihood(Q, pi)
        ll_long = calc_long.compute_log_likelihood(Q, pi)

        # Short branches should have higher likelihood for identical sequences
        assert ll_short > ll_long

    def test_multiple_sites(self):
        """Test likelihood calculation with multiple sites."""
        names = ["seq1", "seq2"]
        # Three codon sites
        sequences = np.array(
            [
                [CODON_TO_INDEX["ATG"], CODON_TO_INDEX["CCC"], CODON_TO_INDEX["GGG"]],
                [CODON_TO_INDEX["ATG"], CODON_TO_INDEX["CCC"], CODON_TO_INDEX["GGG"]],
            ],
            dtype=np.int8,
        )

        aln = Alignment(
            names=names,
            sequences=sequences,
            n_species=2,
            n_sites=3,
            seqtype="codon",
        )

        tree = Tree.from_newick("(seq1:0.1, seq2:0.1);")

        pi = np.ones(61) / 61
        rates = np.ones((61, 61))
        Q = create_reversible_Q(rates, pi, normalize=True)

        calc = LikelihoodCalculator(aln, tree)
        log_likelihood = calc.compute_log_likelihood(Q, pi)

        # Should be finite and negative
        assert np.isfinite(log_likelihood)
        assert log_likelihood < 0

    def test_mismatched_alignment_tree_raises(self):
        """Test that mismatched alignment and tree raise an error."""
        # Alignment with 2 sequences
        names = ["seq1", "seq2"]
        sequences = np.array([[0], [0]], dtype=np.int8)
        aln = Alignment(
            names=names,
            sequences=sequences,
            n_species=2,
            n_sites=1,
            seqtype="codon",
        )

        # Tree with 3 leaves
        tree = Tree.from_newick("((seq1,seq2),seq3);")

        # Should raise ValueError
        with pytest.raises(ValueError, match="sequences but tree has"):
            LikelihoodCalculator(aln, tree)

    def test_scale_branch_lengths(self):
        """Test branch length scaling parameter."""
        names = ["seq1", "seq2"]
        atg_idx = CODON_TO_INDEX["ATG"]
        sequences = np.array([[atg_idx], [atg_idx]], dtype=np.int8)

        aln = Alignment(
            names=names,
            sequences=sequences,
            n_species=2,
            n_sites=1,
            seqtype="codon",
        )

        tree = Tree.from_newick("(seq1:0.1, seq2:0.1);")

        pi = np.ones(61) / 61
        rates = np.ones((61, 61))
        Q = create_reversible_Q(rates, pi, normalize=True)

        calc = LikelihoodCalculator(aln, tree)

        # Compute with different scaling factors
        ll_scale_1 = calc.compute_log_likelihood(Q, pi, scale_branch_lengths=1.0)
        ll_scale_2 = calc.compute_log_likelihood(Q, pi, scale_branch_lengths=2.0)

        # Scaling by 2 should give same result as doubling branch lengths
        # So for identical sequences, higher scale should give lower likelihood
        assert ll_scale_1 > ll_scale_2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
