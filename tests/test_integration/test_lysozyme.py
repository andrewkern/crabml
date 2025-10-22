"""
Integration tests using the lysozyme dataset.

These tests verify that our implementation works on real data
and produces reasonable results.
"""

import numpy as np
import pytest
from pathlib import Path

from crabml.core.likelihood import LikelihoodCalculator
from crabml.core.matrix import create_reversible_Q
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


class TestLysozyme:
    """Integration tests with lysozyme dataset."""

    def test_load_lysozyme_data(self, lysozyme_small_files):
        """Test that we can load the lysozyme dataset."""
        # Load alignment
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        assert aln.n_species == 7
        assert aln.n_sites == 130
        assert aln.seqtype == "codon"

        # Load tree
        # Use the simple tree from the file (no branch labels)
        tree_str = "((Hsa_Human, Hla_gibbon), ((Cgu/Can_colobus, Pne_langur), Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));"
        tree = Tree.from_newick(tree_str)

        assert tree.n_leaves == 7
        assert set(tree.leaf_names) == set(aln.names)

    def test_compute_likelihood_simple_model(self, lysozyme_small_files):
        """Test likelihood computation on lysozyme with simple equal-rates model."""
        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        # Use simple tree with arbitrary branch lengths
        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree = Tree.from_newick(tree_str)

        # Create simple equal-rates model for 61 codons
        pi = np.ones(61) / 61  # Uniform frequencies
        rates = np.ones((61, 61))  # Equal exchangeability
        Q = create_reversible_Q(rates, pi, normalize=True)

        # Create likelihood calculator
        calc = LikelihoodCalculator(aln, tree)

        # Compute likelihood
        log_likelihood = calc.compute_log_likelihood(Q, pi)

        # Verify likelihood is finite and negative
        assert np.isfinite(log_likelihood)
        assert log_likelihood < 0

        # With 130 sites and equal-rates model, log-likelihood should be
        # somewhere in the reasonable range (rough sanity check)
        # For reference, PAML gives lnL ≈ -906 with optimized parameters
        # Our simple model should give much worse likelihood
        assert log_likelihood < -1000  # Should be worse than optimized model
        assert log_likelihood > -5000  # But not completely terrible

    def test_likelihood_varies_with_branch_lengths(self, lysozyme_small_files):
        """Test that likelihood changes with branch length scaling."""
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree = Tree.from_newick(tree_str)

        pi = np.ones(61) / 61
        rates = np.ones((61, 61))
        Q = create_reversible_Q(rates, pi, normalize=True)

        calc = LikelihoodCalculator(aln, tree)

        # Compute with different branch length scales
        ll_scale_05 = calc.compute_log_likelihood(Q, pi, scale_branch_lengths=0.5)
        ll_scale_10 = calc.compute_log_likelihood(Q, pi, scale_branch_lengths=1.0)
        ll_scale_20 = calc.compute_log_likelihood(Q, pi, scale_branch_lengths=2.0)

        # All should be finite
        assert np.isfinite(ll_scale_05)
        assert np.isfinite(ll_scale_10)
        assert np.isfinite(ll_scale_20)

        # They should be different
        assert ll_scale_05 != ll_scale_10
        assert ll_scale_10 != ll_scale_20

    def test_likelihood_computation_is_reproducible(self, lysozyme_small_files):
        """Test that likelihood computation gives consistent results."""
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree = Tree.from_newick(tree_str)

        pi = np.ones(61) / 61
        rates = np.ones((61, 61))
        Q = create_reversible_Q(rates, pi, normalize=True)

        calc = LikelihoodCalculator(aln, tree)

        # Compute multiple times
        ll1 = calc.compute_log_likelihood(Q, pi)
        ll2 = calc.compute_log_likelihood(Q, pi)
        ll3 = calc.compute_log_likelihood(Q, pi)

        # Should be exactly the same
        np.testing.assert_equal(ll1, ll2)
        np.testing.assert_equal(ll2, ll3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
