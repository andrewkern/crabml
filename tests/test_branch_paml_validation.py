"""
Validate branch models against PAML reference outputs.

Tests branch model implementation against PAML codeml results
for the lysozyme dataset (Yang 1998).
"""

import pytest
import numpy as np
from pathlib import Path

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.branch import BranchModelOptimizer


@pytest.mark.slow
def test_branch_model_two_ratio_vs_paml():
    """
    Test two-ratio branch model against PAML reference.

    PAML reference (codeml with model=2, NSsites=0):
        lnL = -903.076551
        kappa = 4.56831
        omega0 (background) = 0.67535
        omega1 (foreground) = 999.0 (hit bound)

    Tree: ((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset))

    One branch (#1) is foreground, all others are background.
    """
    # Load alignment
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Two-ratio tree (same as PAML)
    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    # Create optimizer
    optimizer = BranchModelOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        free_ratio=False,
    )

    # Run optimization (more iterations for convergence)
    kappa, omega_dict, lnL = optimizer.optimize(
        init_kappa=3.0,
        init_omega=0.4,
        method='L-BFGS-B',
        maxiter=1000,
    )

    # PAML reference values
    paml_lnL = -903.076551
    paml_kappa = 4.56831
    paml_omega0 = 0.67535
    # paml_omega1 = 999.0 (but this hit PAML's upper bound)

    # Validate log-likelihood (should be very close)
    lnL_diff = abs(lnL - paml_lnL)
    print(f"\nValidation Results:")
    print(f"  lnL:    {lnL:.6f} vs PAML {paml_lnL:.6f} (diff: {lnL_diff:.6f})")
    print(f"  kappa:  {kappa:.6f} vs PAML {paml_kappa:.6f} (diff: {abs(kappa - paml_kappa):.6f})")
    print(f"  omega0: {omega_dict['omega0']:.6f} vs PAML {paml_omega0:.6f} (diff: {abs(omega_dict['omega0'] - paml_omega0):.6f})")
    print(f"  omega1: {omega_dict['omega1']:.6f} (PAML hit bound at 999.0)")

    # Assertions
    assert lnL_diff < 0.5, f"lnL differs too much from PAML: {lnL_diff}"
    assert abs(kappa - paml_kappa) < 0.1, f"kappa differs from PAML: {abs(kappa - paml_kappa)}"
    assert abs(omega_dict['omega0'] - paml_omega0) < 0.05, f"omega0 differs from PAML"

    # omega1 should be large (foreground under positive selection)
    assert omega_dict['omega1'] > 5.0, f"omega1 should be large (positive selection)"

    print("\n✓ PAML validation PASSED!")


@pytest.mark.slow
def test_branch_model_parameter_interpretation():
    """
    Test that branch model parameters are interpretable.

    For lysozyme with one foreground branch:
    - omega0 should be < 1 (purifying selection on background)
    - omega1 should be > 1 (positive selection on foreground)
    - kappa should be 3-5 (typical ts/tv ratio)
    """
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    optimizer = BranchModelOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        free_ratio=False,
    )

    kappa, omega_dict, lnL = optimizer.optimize(maxiter=500)

    # Biological interpretation
    print(f"\nParameter interpretation:")
    print(f"  kappa = {kappa:.4f} (transition/transversion ratio)")
    print(f"  omega0 = {omega_dict['omega0']:.4f} ({'purifying' if omega_dict['omega0'] < 1 else 'neutral/positive'})")
    print(f"  omega1 = {omega_dict['omega1']:.4f} ({'purifying' if omega_dict['omega1'] < 1 else 'positive selection'})")

    # Sanity checks
    assert 2.0 < kappa < 10.0, f"kappa seems unreasonable: {kappa}"
    assert 0.0 < omega_dict['omega0'] < 2.0, f"omega0 seems unreasonable"
    assert omega_dict['omega1'] > omega_dict['omega0'], "Foreground should have higher omega"


if __name__ == "__main__":
    print("=" * 80)
    print("Branch Model PAML Validation")
    print("=" * 80)

    test_branch_model_two_ratio_vs_paml()
    test_branch_model_parameter_interpretation()

    print("\n" + "=" * 80)
    print("All PAML validation tests passed!")
    print("=" * 80)
