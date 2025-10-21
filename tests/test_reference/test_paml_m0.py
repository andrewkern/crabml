"""
Reference tests comparing against PAML M0 model output.
"""

import numpy as np
import pytest

from pycodeml.optimize.optimizer import M0Optimizer
from pycodeml.io.sequences import Alignment
from pycodeml.io.trees import Tree


class TestPAMLM0:
    """Test M0 model optimization against PAML reference."""

    @pytest.mark.slow
    def test_lysozyme_m0_optimization(self, lysozyme_small_files):
        """
        Test M0 optimization on lysozyme dataset.

        PAML reference (from lysozyme_m0_out.txt):
        lnL = -906.017440

        This is a slow test as optimization can take time.
        """
        # Load data
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )

        # Use a simple tree topology
        # Start with reasonable branch lengths
        tree_str = (
            "((Hsa_Human:0.05, Hla_gibbon:0.03):0.03, "
            "((Cgu/Can_colobus:0.03, Pne_langur:0.03):0.03, Mmu_rhesus:0.06):0.03, "
            "(Ssc_squirrelM:0.03, Cja_marmoset:0.06):0.03);"
        )
        tree = Tree.from_newick(tree_str)

        # Create optimizer with F3X4 frequencies and individual branch optimization
        optimizer = M0Optimizer(aln, tree, use_f3x4=True, optimize_branch_lengths=True)

        # Optimize parameters
        # PAML initial values: kappa=2, omega=0.4
        kappa, omega, log_likelihood = optimizer.optimize(
            init_kappa=2.0,
            init_omega=0.4,
            maxiter=100
        )

        print(f"\nOptimized parameters:")
        print(f"  kappa = {kappa:.6f}")
        print(f"  omega = {omega:.6f}")
        print(f"  lnL = {log_likelihood:.6f}")
        print(f"\nPAML reference: lnL = -906.017440")
        print(f"Difference: {abs(log_likelihood - (-906.017440)):.6f}")

        # Basic sanity checks
        assert kappa > 0, "kappa should be positive"
        assert omega > 0, "omega should be positive"
        assert log_likelihood < 0, "log-likelihood should be negative"
        assert np.isfinite(log_likelihood), "log-likelihood should be finite"

        # The likelihood should be reasonable
        assert log_likelihood > -2000, "likelihood should not be terrible"
        assert log_likelihood < -500, "likelihood should be in reasonable range"

        # With individual branch optimization, we should get close to PAML
        # Allow some tolerance due to different optimization strategies
        assert abs(log_likelihood - (-906.017440)) < 10, \
            f"Should be within 10 log-likelihood units of PAML (got {log_likelihood:.6f})"

    def test_simple_optimization_converges(self, lysozyme_small_files):
        """Test that optimization converges on a simple problem."""
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

        # Test with global branch scaling (faster)
        optimizer = M0Optimizer(aln, tree, use_f3x4=True, optimize_branch_lengths=False)

        # Run short optimization
        kappa, omega, log_likelihood = optimizer.optimize(
            init_kappa=2.0,
            init_omega=0.4,
            maxiter=20
        )

        # Should improve from initial likelihood
        assert len(optimizer.history) > 0
        initial_ll = optimizer.history[0]['log_likelihood']
        final_ll = log_likelihood

        print(f"\nInitial lnL: {initial_ll:.6f}")
        print(f"Final lnL: {final_ll:.6f}")
        print(f"Improvement: {final_ll - initial_ll:.6f}")

        # Final should be better than or equal to initial
        assert final_ll >= initial_ll


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
