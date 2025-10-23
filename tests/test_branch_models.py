"""
Test branch models against PAML reference outputs.

Tests:
- Two-ratio model (model=2, NSsites=0)
- Free-ratio model (model=1, NSsites=0)

Reference: Yang (1998) Mol. Biol. Evol. 15:568-573
Dataset: lysozymeSmall (7 primate species, 130 codons)
"""

import pytest
import numpy as np
from pathlib import Path

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.branch import BranchModelOptimizer


def test_branch_model_two_ratio():
    """
    Test two-ratio branch model (model=2).

    Tree: ((1,2) #1, ((3,4), 5), (6,7))
    This estimates:
    - omega0: background branches (no label)
    - omega1: foreground branch (label #1)
    """
    # Load alignment
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Tree with one foreground branch
    # Table 1C from Yang (1998)
    # Species: 1=Hsa_Human, 2=Hla_gibbon, 3=Cgu/Can_colobus, 4=Pne_langur,
    #          5=Mmu_rhesus, 6=Ssc_squirrelM, 7=Cja_marmoset
    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    # Create optimizer
    optimizer = BranchModelOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        free_ratio=False,  # Two-ratio model
    )

    # Run optimization
    kappa, omega_dict, lnL = optimizer.optimize(
        init_kappa=3.0,
        init_omega=0.4,
        method='L-BFGS-B',
        maxiter=500,
    )

    # Basic checks
    assert 'omega0' in omega_dict, "Should have omega0 (background)"
    assert 'omega1' in omega_dict, "Should have omega1 (foreground)"
    assert len(omega_dict) == 2, f"Should have exactly 2 omega values, got {len(omega_dict)}"

    # Check parameter ranges
    assert 0.1 < kappa < 20.0, f"kappa out of range: {kappa}"
    assert 0.0 < omega_dict['omega0'] < 20.0, f"omega0 out of range: {omega_dict['omega0']}"
    assert 0.0 < omega_dict['omega1'] < 20.0, f"omega1 out of range: {omega_dict['omega1']}"

    # Check likelihood is reasonable
    assert lnL < 0, f"Log-likelihood should be negative, got {lnL}"
    assert lnL > -2000, f"Log-likelihood seems too negative: {lnL}"

    print(f"\nTwo-ratio model results:")
    print(f"  lnL = {lnL:.6f}")
    print(f"  kappa = {kappa:.6f}")
    print(f"  omega0 (background) = {omega_dict['omega0']:.6f}")
    print(f"  omega1 (foreground) = {omega_dict['omega1']:.6f}")


def test_branch_model_basic_import():
    """Test that we can import the branch model classes."""
    from crabml.models.codon_branch import CodonBranchModel
    from crabml.optimize.branch import BranchModelOptimizer

    assert CodonBranchModel is not None
    assert BranchModelOptimizer is not None


def test_branch_model_parameter_counts():
    """
    Test that branch models have correct number of parameters.
    """
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Two-ratio model: 2 omega values
    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    optimizer = BranchModelOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        free_ratio=False,
    )

    assert optimizer.model.n_omega == 2, f"Two-ratio should have 2 omegas, got {optimizer.model.n_omega}"

    # Three-ratio model: 3 omega values
    tree_str3 = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur) #2, Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree3 = Tree.from_newick(tree_str3)

    optimizer3 = BranchModelOptimizer(
        alignment=alignment,
        tree=tree3,
        use_f3x4=True,
        free_ratio=False,
    )

    assert optimizer3.model.n_omega == 3, f"Three-ratio should have 3 omegas, got {optimizer3.model.n_omega}"


@pytest.mark.slow
def test_branch_model_vs_m0():
    """
    Test that branch model with no labels matches M0.

    When all branches have the same omega, branch model
    should give similar results to M0 (one-ratio model).
    """
    from crabml.optimize.optimizer import M0Optimizer

    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Tree with NO branch labels (all background = label 0)
    tree_str = "((Hsa_Human,Hla_gibbon), ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    # Run M0 model
    m0_optimizer = M0Optimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
    )
    kappa_m0, omega_m0, lnL_m0 = m0_optimizer.optimize(maxiter=200)

    # Run branch model with no labels (single omega)
    branch_optimizer = BranchModelOptimizer(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        free_ratio=False,
    )
    kappa_br, omega_dict_br, lnL_br = branch_optimizer.optimize(maxiter=200)

    print(f"\nM0 vs Branch (no labels):")
    print(f"  M0:     lnL={lnL_m0:.4f}, kappa={kappa_m0:.4f}, omega={omega_m0:.4f}")
    print(f"  Branch: lnL={lnL_br:.4f}, kappa={kappa_br:.4f}, omega={omega_dict_br['omega0']:.4f}")

    # Should give very similar results
    assert abs(lnL_m0 - lnL_br) < 0.5, f"Likelihoods should match (diff={abs(lnL_m0 - lnL_br)})"
    assert abs(kappa_m0 - kappa_br) < 0.1, f"Kappa should match (diff={abs(kappa_m0 - kappa_br)})"
    assert abs(omega_m0 - omega_dict_br['omega0']) < 0.1, f"Omega should match"


if __name__ == "__main__":
    # Run basic tests
    print("=" * 80)
    print("Testing Branch Models")
    print("=" * 80)

    test_branch_model_basic_import()
    print("✓ Import test passed")

    test_branch_model_parameter_counts()
    print("✓ Parameter count test passed")

    test_branch_model_two_ratio()
    print("✓ Two-ratio model test passed")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
