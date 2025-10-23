"""
Test Branch-Site Model A null model (omega2 = 1).

Tests that the null model correctly fixes omega2 = 1 and optimizes
all other parameters.
"""
import pytest
import numpy as np
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.branch_site import BranchSiteModelAOptimizer


def test_branch_site_model_a_null():
    """Test Branch-Site Model A null model optimization (omega2 = 1)."""

    # Load alignment
    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )

    # Load tree
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    # Create optimizer with fix_omega=True
    optimizer = BranchSiteModelAOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        fix_omega=True  # NULL MODEL
    )

    # Verify model has fix_omega enabled
    assert optimizer.fix_omega is True
    assert optimizer.model.fix_omega is True

    # Run optimization
    kappa, omega0, omega2, p0, p1, lnL = optimizer.optimize(
        init_kappa=3.0,
        init_omega0=0.05,
        init_omega2=1.0,  # Not used, but pass for clarity
        init_p0=0.35,
        init_p1=0.30,
        method='L-BFGS-B',
        maxiter=500
    )

    # Verify omega2 is fixed at 1.0
    assert omega2 == 1.0, f"omega2 should be fixed at 1.0, got {omega2}"

    # Verify other parameters are reasonable
    assert 0.1 < kappa < 20.0, f"kappa out of range: {kappa}"
    assert 0 < omega0 < 1.0, f"omega0 should be < 1: {omega0}"
    assert 0 < p0 < 1.0, f"p0 out of range: {p0}"
    assert 0 < p1 < 1.0, f"p1 out of range: {p1}"
    assert 0 < (1 - p0 - p1) < 1.0, f"p2 out of range: {1 - p0 - p1}"

    # Verify likelihood is reasonable (should be worse than alternative model)
    # Expected null model lnL around -1036.3
    assert -1050 < lnL < -1030, f"Likelihood out of expected range: {lnL}"

    # Verify optimization converged
    assert len(optimizer.history) > 0, "Optimization history is empty"

    print(f"\n✓ Branch-Site Model A null optimization successful!")
    print(f"  Likelihood: {lnL:.6f}")
    print(f"  kappa: {kappa:.6f}")
    print(f"  omega0: {omega0:.6f}")
    print(f"  omega2: {omega2:.6f} (FIXED)")
    print(f"  p0: {p0:.6f}, p1: {p1:.6f}, p2: {1-p0-p1:.6f}")


def test_branch_site_null_vs_alternative():
    """Test that null model has worse likelihood than alternative model."""

    # Load data
    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()

    # Test NULL model
    tree_null = Tree.from_newick(tree_str)
    optimizer_null = BranchSiteModelAOptimizer(
        alignment=alignment,
        tree=tree_null,
        use_f3x4=True,
        optimize_branch_lengths=True,
        fix_omega=True
    )
    _, _, _, _, _, lnL_null = optimizer_null.optimize(
        init_kappa=3.0, init_omega0=0.05, init_omega2=1.0,
        init_p0=0.35, init_p1=0.30,
        method='L-BFGS-B', maxiter=500
    )

    # Test ALTERNATIVE model
    tree_alt = Tree.from_newick(tree_str)
    optimizer_alt = BranchSiteModelAOptimizer(
        alignment=alignment,
        tree=tree_alt,
        use_f3x4=True,
        optimize_branch_lengths=True,
        fix_omega=False
    )
    _, _, omega2_alt, _, _, lnL_alt = optimizer_alt.optimize(
        init_kappa=3.0, init_omega0=0.05, init_omega2=3.0,
        init_p0=0.35, init_p1=0.30,
        method='L-BFGS-B', maxiter=500
    )

    # Null model should have WORSE (lower) likelihood than alternative
    assert lnL_alt > lnL_null, (
        f"Alternative model should fit better than null: "
        f"alt_lnL = {lnL_alt:.6f}, null_lnL = {lnL_null:.6f}"
    )

    # Alternative model should have omega2 > 1 (if there's positive selection)
    # (May not always be true, but expected for this dataset)
    print(f"\n✓ Null vs Alternative comparison:")
    print(f"  Null lnL:        {lnL_null:.6f} (omega2 = 1.0)")
    print(f"  Alternative lnL: {lnL_alt:.6f} (omega2 = {omega2_alt:.4f})")
    print(f"  Difference:      {lnL_alt - lnL_null:.6f}")


def test_branch_site_null_parameter_count():
    """Test that null model has one fewer parameter than alternative."""

    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    # Null model: kappa, omega0, p0, p1 + branch_lengths
    # Alternative: kappa, omega0, omega2, p0, p1 + branch_lengths
    # Difference: 1 parameter (omega2)

    # This is implicitly tested by the optimizer implementation,
    # but we verify the model's parameter specification
    from crabml.models.codon_branch_site import BranchSiteModelA

    pi = np.ones(61) / 61
    labels = [0] * 32 + [1]  # 32 background, 1 foreground

    # Null model parameters: kappa, omega0, p0, p1 (omega2 is fixed, not optimized)
    model_null = BranchSiteModelA(pi, labels, fix_omega=True)
    params_null = model_null.get_parameters()
    assert 'omega2' not in params_null, "Null model should not have omega2 parameter"
    assert len(params_null) == 4, f"Null model should have 4 params (kappa, omega0, p0, p1), got {len(params_null)}"

    # Alternative model parameters: kappa, omega0, omega2, p0, p1
    model_alt = BranchSiteModelA(pi, labels, fix_omega=False)
    params_alt = model_alt.get_parameters()
    assert 'omega2' in params_alt, "Alternative model should have omega2 parameter"
    assert len(params_alt) == 5, f"Alternative model should have 5 params (kappa, omega0, omega2, p0, p1), got {len(params_alt)}"

    # Verify the difference is exactly 1 (omega2)
    assert len(params_alt) - len(params_null) == 1, "Should differ by exactly 1 parameter (omega2)"

    print("\n✓ Parameter count verified:")
    print(f"  Null model:        {list(params_null.keys())} ({len(params_null)} params)")
    print(f"  Alternative model: {list(params_alt.keys())} ({len(params_alt)} params)")
    print(f"  Difference: {len(params_alt) - len(params_null)} parameter (omega2)")


if __name__ == "__main__":
    test_branch_site_model_a_null()
    test_branch_site_null_vs_alternative()
    test_branch_site_null_parameter_count()
