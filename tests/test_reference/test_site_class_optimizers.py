"""
Reference tests for site class model optimizers.
"""

import numpy as np
import pytest

from crabml.optimize.optimizer import M1aOptimizer, M2aOptimizer, M3Optimizer
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


class TestSiteClassOptimizers:
    """Test M1a and M2a optimizers."""

    def test_m1a_optimizer_converges(self, lysozyme_small_files):
        """Test that M1a optimizer converges."""
        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree = Tree.from_newick(tree_str)

        # Use branch scaling for faster test
        optimizer = M1aOptimizer(aln, tree, use_f3x4=True, optimize_branch_lengths=False)

        # Run short optimization
        kappa, omega0, p0, log_likelihood = optimizer.optimize(
            init_kappa=2.0,
            init_omega0=0.5,
            init_p0=0.7,
            maxiter=20
        )

        # Basic sanity checks
        assert kappa > 0
        assert 0 < omega0 < 1  # Should be constrained
        assert 0 < p0 < 1
        assert log_likelihood < 0
        assert np.isfinite(log_likelihood)

        # Should improve from initial
        assert len(optimizer.history) > 0
        initial_ll = optimizer.history[0]['log_likelihood']
        assert log_likelihood >= initial_ll

        print(f"\nM1a optimization results:")
        print(f"  kappa = {kappa:.4f}")
        print(f"  omega0 = {omega0:.4f}")
        print(f"  p0 = {p0:.4f}")
        print(f"  lnL = {log_likelihood:.6f}")

    def test_m2a_optimizer_converges(self, lysozyme_small_files):
        """Test that M2a optimizer converges."""
        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree = Tree.from_newick(tree_str)

        # Use branch scaling for faster test
        optimizer = M2aOptimizer(aln, tree, use_f3x4=True, optimize_branch_lengths=False)

        # Run short optimization
        kappa, omega0, omega2, p0, p1, log_likelihood = optimizer.optimize(
            init_kappa=2.0,
            init_omega0=0.5,
            init_omega2=2.0,
            init_p0=0.5,
            init_p1=0.3,
            maxiter=20
        )

        # Basic sanity checks
        assert kappa > 0
        assert 0 < omega0 < 1  # Purifying
        assert omega2 > 1  # Positive selection
        assert 0 < p0 < 1
        assert 0 < p1 < 1
        assert 0 < (1 - p0 - p1) < 1  # p2
        assert log_likelihood < 0
        assert np.isfinite(log_likelihood)

        # Should improve from initial
        assert len(optimizer.history) > 0
        initial_ll = optimizer.history[0]['log_likelihood']
        assert log_likelihood >= initial_ll

        print(f"\nM2a optimization results:")
        print(f"  kappa = {kappa:.4f}")
        print(f"  omega0 = {omega0:.4f}")
        print(f"  omega2 = {omega2:.4f}")
        print(f"  p0 = {p0:.4f}, p1 = {p1:.4f}, p2 = {1-p0-p1:.4f}")
        print(f"  lnL = {log_likelihood:.6f}")

    @pytest.mark.slow
    def test_m1a_vs_m0_likelihood(self, lysozyme_small_files):
        """Test that M1a should fit at least as well as M0."""
        from crabml.optimize.optimizer import M0Optimizer

        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree_m0 = Tree.from_newick(tree_str)
        tree_m1a = Tree.from_newick(tree_str)

        # Optimize M0
        opt_m0 = M0Optimizer(aln, tree_m0, use_f3x4=True, optimize_branch_lengths=False)
        _, _, ll_m0 = opt_m0.optimize(maxiter=50)

        # Optimize M1a
        opt_m1a = M1aOptimizer(aln, tree_m1a, use_f3x4=True, optimize_branch_lengths=False)
        _, _, _, ll_m1a = opt_m1a.optimize(maxiter=50)

        print(f"\nModel comparison:")
        print(f"  M0 lnL: {ll_m0:.6f}")
        print(f"  M1a lnL: {ll_m1a:.6f}")
        print(f"  Difference: {ll_m1a - ll_m0:.6f}")

        # M1a should fit at least as well as M0 (more parameters)
        # Allow small numerical tolerance
        assert ll_m1a >= ll_m0 - 0.1

    @pytest.mark.slow
    def test_m2a_vs_m1a_likelihood(self, lysozyme_small_files):
        """Test that M2a should fit at least as well as M1a."""
        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree_m1a = Tree.from_newick(tree_str)
        tree_m2a = Tree.from_newick(tree_str)

        # Optimize M1a
        opt_m1a = M1aOptimizer(aln, tree_m1a, use_f3x4=True, optimize_branch_lengths=False)
        _, _, _, ll_m1a = opt_m1a.optimize(maxiter=50)

        # Optimize M2a
        opt_m2a = M2aOptimizer(aln, tree_m2a, use_f3x4=True, optimize_branch_lengths=False)
        _, _, _, _, _, ll_m2a = opt_m2a.optimize(maxiter=50)

        print(f"\nModel comparison:")
        print(f"  M1a lnL: {ll_m1a:.6f}")
        print(f"  M2a lnL: {ll_m2a:.6f}")
        print(f"  Difference: {ll_m2a - ll_m1a:.6f}")

        # M2a should fit at least as well as M1a (more parameters)
        # Allow small numerical tolerance
        assert ll_m2a >= ll_m1a - 0.1

    def test_m3_optimizer_converges(self, lysozyme_small_files):
        """Test that M3 optimizer converges."""
        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree = Tree.from_newick(tree_str)

        # Use branch scaling for faster test
        optimizer = M3Optimizer(
            aln, tree, n_classes=3, use_f3x4=True, optimize_branch_lengths=False
        )

        # Run short optimization
        kappa, omegas, proportions, log_likelihood = optimizer.optimize(
            init_kappa=2.0,
            maxiter=20
        )

        # Basic sanity checks
        assert kappa > 0
        assert len(omegas) == 3
        assert all(w > 0 for w in omegas)
        assert len(proportions) == 3
        assert all(0 < p < 1 for p in proportions)
        assert np.isclose(sum(proportions), 1.0)
        assert log_likelihood < 0
        assert np.isfinite(log_likelihood)

        # Should improve from initial
        assert len(optimizer.history) > 0
        initial_ll = optimizer.history[0]['log_likelihood']
        assert log_likelihood >= initial_ll

        print(f"\nM3 optimization results:")
        print(f"  kappa = {kappa:.4f}")
        print(f"  omegas = {', '.join([f'{w:.4f}' for w in omegas])}")
        print(f"  proportions = {', '.join([f'{p:.4f}' for p in proportions])}")
        print(f"  lnL = {log_likelihood:.6f}")

    @pytest.mark.slow
    def test_m3_vs_m0_likelihood(self, lysozyme_small_files):
        """Test that M3 should fit at least as well as M0."""
        from crabml.optimize.optimizer import M0Optimizer

        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.05):0.05, "
            "((Cgu/Can_colobus:0.05, Pne_langur:0.05):0.05, Mmu_rhesus:0.05):0.05, "
            "(Ssc_squirrelM:0.05, Cja_marmoset:0.05):0.05);"
        )
        tree_m0 = Tree.from_newick(tree_str)
        tree_m3 = Tree.from_newick(tree_str)

        # Optimize M0
        opt_m0 = M0Optimizer(aln, tree_m0, use_f3x4=True, optimize_branch_lengths=False)
        _, _, ll_m0 = opt_m0.optimize(maxiter=50)

        # Optimize M3
        opt_m3 = M3Optimizer(
            aln, tree_m3, n_classes=3, use_f3x4=True, optimize_branch_lengths=False
        )
        _, _, _, ll_m3 = opt_m3.optimize(maxiter=50)

        print(f"\nModel comparison:")
        print(f"  M0 lnL: {ll_m0:.6f}")
        print(f"  M3 lnL: {ll_m3:.6f}")
        print(f"  Difference: {ll_m3 - ll_m0:.6f}")

        # M3 should fit at least as well as M0 (more parameters)
        # Allow small numerical tolerance
        assert ll_m3 >= ll_m0 - 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
