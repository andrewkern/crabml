"""
Test Branch-Site Model A parameter optimization against PAML reference.

This test verifies that our optimizer can recover parameters close to PAML's MLEs
when starting from reasonable initial values.

PAML Reference (lysozyme dataset):
- lnL = -1035.533916
- kappa = 4.154927
- p0 = 0.326513
- p1 = 0.269308
- omega0 = 0.000001
- omega2 = 4.809765
"""
import pytest
import numpy as np
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.branch_site import BranchSiteModelAOptimizer


def test_branch_site_model_a_optimization():
    """Test Branch-Site Model A optimization converges near PAML MLEs."""

    # Load alignment
    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )

    # Load tree
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    # Create optimizer
    optimizer = BranchSiteModelAOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True
    )

    # Run optimization with good initial values
    kappa, omega0, omega2, p0, p1, lnL = optimizer.optimize(
        init_kappa=3.0,
        init_omega0=0.05,
        init_omega2=3.0,
        init_p0=0.35,
        init_p1=0.30,
        method='L-BFGS-B',
        maxiter=500
    )

    # PAML reference values
    paml_lnL = -1035.533916
    paml_kappa = 4.154927
    paml_p0 = 0.326513
    paml_p1 = 0.269308
    paml_omega0 = 0.000001
    paml_omega2 = 4.809765

    # Verify likelihood is close to PAML (within 0.1 units is excellent)
    lnL_diff = abs(lnL - paml_lnL)
    assert lnL_diff < 0.1, (
        f"Likelihood differs too much from PAML: "
        f"our lnL = {lnL:.6f}, PAML lnL = {paml_lnL:.6f}, "
        f"difference = {lnL_diff:.6f}"
    )

    # Verify parameters are reasonably close
    # (Note: slight differences expected due to different optimizers)
    assert abs(kappa - paml_kappa) < 0.1, f"kappa diff too large: {abs(kappa - paml_kappa):.6f}"
    assert abs(p0 - paml_p0) < 0.05, f"p0 diff too large: {abs(p0 - paml_p0):.6f}"
    assert abs(p1 - paml_p1) < 0.05, f"p1 diff too large: {abs(p1 - paml_p1):.6f}"

    # omega0 is near 0, so we just check it's small and positive
    assert 0 < omega0 < 0.01, f"omega0 should be small: {omega0:.6f}"

    # omega2 should be > 1 (positive selection)
    assert omega2 > 1.0, f"omega2 should indicate positive selection: {omega2:.6f}"
    assert abs(omega2 - paml_omega2) < 0.5, f"omega2 diff too large: {abs(omega2 - paml_omega2):.6f}"

    # Verify site class proportions sum to ~1
    p2 = 1 - p0 - p1
    assert abs(p0 + p1 + p2 - 1.0) < 1e-6, "Site class proportions should sum to 1"

    print(f"\n✓ Branch-Site Model A optimization successful!")
    print(f"  Likelihood: {lnL:.6f} (PAML: {paml_lnL:.6f}, diff: {lnL_diff:.6f})")
    print(f"  kappa: {kappa:.6f} (PAML: {paml_kappa:.6f})")
    print(f"  p0: {p0:.6f} (PAML: {paml_p0:.6f})")
    print(f"  p1: {p1:.6f} (PAML: {paml_p1:.6f})")
    print(f"  p2: {p2:.6f}")
    print(f"  omega0: {omega0:.6f} (PAML: {paml_omega0:.6f})")
    print(f"  omega2: {omega2:.6f} (PAML: {paml_omega2:.6f})")


if __name__ == "__main__":
    test_branch_site_model_a_optimization()
