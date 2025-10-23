"""
Test hypothesis testing framework for branch models.
"""

import pytest
from pathlib import Path

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.analysis import branch_model_test, free_ratio_test


def test_branch_model_test_two_ratio():
    """
    Test branch model hypothesis test (two-ratio vs M0).

    Uses lysozyme dataset with one foreground branch.
    """
    # Load alignment
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Two-ratio tree (background=#0, foreground=#1)
    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    # Run test
    result = branch_model_test(
        alignment=alignment,
        tree=tree,
        verbose=True,
    )

    # Check result structure
    assert result is not None
    assert result.test_name == "Branch Model vs M0"
    assert result.null_model == "M0 (one-ratio)"
    assert "2-ratio" in result.alt_model
    assert result.df == 1  # 2 omegas - 1

    # Check parameters
    assert 'kappa' in result.null_params
    assert 'omega' in result.null_params
    assert 'kappa' in result.alt_params
    assert 'omega0' in result.alt_params
    assert 'omega1' in result.alt_params

    # Log-likelihoods should be finite
    assert result.lnL_null < 0
    assert result.lnL_alt < 0

    # Alternative should fit better
    assert result.lnL_alt > result.lnL_null, "Branch model should fit better than M0"

    # LRT should be positive
    assert result.LRT >= 0

    # P-value should be between 0 and 1
    assert 0 <= result.pvalue <= 1

    print(f"\n✓ Branch model test completed successfully!")
    print(f"  LRT statistic: {result.LRT:.4f}")
    print(f"  P-value: {result.pvalue:.6f}")
    print(f"  Significant at α=0.05: {result.significant(0.05)}")
    print(f"  M0 ω: {result.null_params['omega']:.4f}")
    print(f"  Branch model ω0 (background): {result.alt_params['omega0']:.4f}")
    print(f"  Branch model ω1 (foreground): {result.alt_params['omega1']:.4f}")


def test_branch_model_test_three_ratio():
    """
    Test branch model hypothesis test with three omega categories.
    """
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Three-ratio tree (background=#0, foreground1=#1, foreground2=#2)
    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur) #2, Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    # Run test
    result = branch_model_test(
        alignment=alignment,
        tree=tree,
        verbose=True,
    )

    # Check result structure
    assert result is not None
    assert "3-ratio" in result.alt_model
    assert result.df == 2  # 3 omegas - 1

    # Check parameters
    assert 'omega0' in result.alt_params
    assert 'omega1' in result.alt_params
    assert 'omega2' in result.alt_params

    # Alternative should fit better
    assert result.lnL_alt > result.lnL_null

    print(f"\n✓ Three-ratio test completed successfully!")
    print(f"  df: {result.df}")
    print(f"  LRT statistic: {result.LRT:.4f}")


def test_branch_model_result_methods():
    """
    Test LRTResult methods with branch model results.
    """
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    tree_str = "((Hsa_Human,Hla_gibbon) #1, ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    result = branch_model_test(alignment=alignment, tree=tree, verbose=False)

    # Test significance at different thresholds
    sig_05 = result.significant(0.05)
    sig_01 = result.significant(0.01)

    print(f"\n✓ Result methods:")
    print(f"  Significant at α=0.05: {sig_05}")
    print(f"  Significant at α=0.01: {sig_01}")

    # If significant at 0.01, should be significant at 0.05
    if sig_01:
        assert sig_05, "If significant at 0.01, must be significant at 0.05"

    # Test summary output
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "Branch Model vs M0" in summary

    # Test to_dict output
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert 'test_name' in result_dict
    assert 'LRT' in result_dict
    assert 'pvalue' in result_dict


def test_free_ratio_model_test():
    """
    Test free-ratio model hypothesis test.

    WARNING: This test may be slow due to the large number of parameters.
    """
    data_dir = Path(__file__).parent / "data" / "branch_models"
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Tree without branch labels (will use free-ratio)
    tree_str = "((Hsa_Human,Hla_gibbon), ((Cgu/Can_colobus,Pne_langur), Mmu_rhesus), (Ssc_squirrelM,Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    # Run test
    result = free_ratio_test(
        alignment=alignment,
        tree=tree,
        verbose=True,
    )

    # Check result structure
    assert result is not None
    assert result.test_name == "Free-Ratio vs M0"
    assert "Free-Ratio" in result.alt_model

    # Free-ratio should have many omegas (one per branch)
    # For 7 species, tree has 11 branches
    assert result.df > 5, "Free-ratio should have many parameters"

    # Alternative should fit better
    assert result.lnL_alt > result.lnL_null

    print(f"\n✓ Free-ratio test completed successfully!")
    print(f"  Number of omega parameters: {result.df + 1}")
    print(f"  LRT statistic: {result.LRT:.4f}")
    print(f"  P-value: {result.pvalue:.6f}")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Branch Model Hypothesis Tests")
    print("=" * 80)

    test_branch_model_test_two_ratio()
    test_branch_model_test_three_ratio()
    test_branch_model_result_methods()

    print("\n" + "=" * 80)
    print("Testing Free-Ratio Model (may be slow)")
    print("=" * 80)
    test_free_ratio_model_test()

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
