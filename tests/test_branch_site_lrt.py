"""
Test branch-site likelihood ratio test for detecting positive selection.
"""
import pytest
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.analysis import branch_site_test


def test_branch_site_test_function():
    """Test branch_site_test function runs successfully."""

    # Load test data
    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    # Run branch-site test
    results = branch_site_test(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        init_kappa=3.0,
        init_omega0=0.05,
        init_omega2=3.0,
        init_p0=0.35,
        init_p1=0.30,
        method='L-BFGS-B',
        maxiter=500
    )

    # Verify results structure
    assert 'null_lnL' in results
    assert 'null_params' in results
    assert 'alt_lnL' in results
    assert 'alt_params' in results
    assert 'lrt_statistic' in results
    assert 'pvalue' in results
    assert 'df' in results
    assert 'significant' in results
    assert 'interpretation' in results

    # Verify degrees of freedom
    assert results['df'] == 1, f"df should be 1, got {results['df']}"

    # Verify alternative has better likelihood
    assert results['alt_lnL'] > results['null_lnL'], (
        f"Alternative should fit better: "
        f"alt={results['alt_lnL']:.6f}, null={results['null_lnL']:.6f}"
    )

    # Verify LRT statistic is non-negative
    assert results['lrt_statistic'] >= 0, (
        f"LRT statistic should be >= 0, got {results['lrt_statistic']:.6f}"
    )

    # Verify p-value is in [0, 1]
    assert 0 <= results['pvalue'] <= 1, (
        f"P-value should be in [0, 1], got {results['pvalue']:.6f}"
    )

    # Verify null model has omega2 = 1
    assert results['null_params']['omega2'] == 1.0, (
        f"Null model omega2 should be 1.0, got {results['null_params']['omega2']}"
    )

    # Verify significance matches p-value
    expected_sig = results['pvalue'] < 0.05
    assert results['significant'] == expected_sig, (
        f"Significance flag incorrect: p={results['pvalue']:.6f}, "
        f"significant={results['significant']}"
    )

    print("\n✓ Branch-site test completed successfully!")
    print(f"  Null lnL:        {results['null_lnL']:.6f}")
    print(f"  Alternative lnL: {results['alt_lnL']:.6f}")
    print(f"  LRT statistic:   {results['lrt_statistic']:.6f}")
    print(f"  P-value:         {results['pvalue']:.6f}")
    print(f"  Significant:     {results['significant']}")
    print(f"  {results['interpretation']}")


def test_branch_site_test_parameter_values():
    """Test that parameter values from branch-site test are reasonable."""

    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    results = branch_site_test(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        init_kappa=3.0,
        init_omega0=0.05,
        init_omega2=3.0,
        init_p0=0.35,
        init_p1=0.30,
        method='L-BFGS-B',
        maxiter=500
    )

    # Check null model parameters
    null = results['null_params']
    assert 0.1 < null['kappa'] < 20, f"Null kappa out of range: {null['kappa']}"
    assert 0 < null['omega0'] < 1, f"Null omega0 should be < 1: {null['omega0']}"
    assert null['omega2'] == 1.0, f"Null omega2 should be 1.0: {null['omega2']}"
    assert 0 < null['p0'] < 1, f"Null p0 out of range: {null['p0']}"
    assert 0 < null['p1'] < 1, f"Null p1 out of range: {null['p1']}"
    assert 0 < null['p2'] < 1, f"Null p2 out of range: {null['p2']}"
    assert abs(null['p0'] + null['p1'] + null['p2'] - 1.0) < 1e-6, (
        f"Null proportions don't sum to 1: {null['p0'] + null['p1'] + null['p2']}"
    )

    # Check alternative model parameters
    alt = results['alt_params']
    assert 0.1 < alt['kappa'] < 20, f"Alt kappa out of range: {alt['kappa']}"
    assert 0 < alt['omega0'] < 1, f"Alt omega0 should be < 1: {alt['omega0']}"
    assert alt['omega2'] >= 1.0, f"Alt omega2 should be >= 1: {alt['omega2']}"
    assert 0 < alt['p0'] < 1, f"Alt p0 out of range: {alt['p0']}"
    assert 0 < alt['p1'] < 1, f"Alt p1 out of range: {alt['p1']}"
    assert 0 < alt['p2'] < 1, f"Alt p2 out of range: {alt['p2']}"
    assert abs(alt['p0'] + alt['p1'] + alt['p2'] - 1.0) < 1e-6, (
        f"Alt proportions don't sum to 1: {alt['p0'] + alt['p1'] + alt['p2']}"
    )

    print("\n✓ All parameter values are within expected ranges")


def test_branch_site_test_matches_paml_alternative():
    """Test that alternative model matches our previous PAML-validated results."""

    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    results = branch_site_test(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        init_kappa=3.0,
        init_omega0=0.05,
        init_omega2=3.0,
        init_p0=0.35,
        init_p1=0.30,
        method='L-BFGS-B',
        maxiter=500
    )

    # Our previous Model A validation: -1035.568 vs PAML -1035.534 (diff: 0.034)
    expected_lnL = -1035.568
    tolerance = 0.1

    lnL_diff = abs(results['alt_lnL'] - expected_lnL)
    assert lnL_diff < tolerance, (
        f"Alternative model lnL differs from expected: "
        f"got {results['alt_lnL']:.6f}, expected ~{expected_lnL:.6f}, "
        f"diff = {lnL_diff:.6f}"
    )

    # Also check it's close to PAML
    paml_lnL = -1035.533916
    paml_diff = abs(results['alt_lnL'] - paml_lnL)
    assert paml_diff < 0.5, (
        f"Alternative model should be close to PAML: "
        f"got {results['alt_lnL']:.6f}, PAML = {paml_lnL:.6f}, "
        f"diff = {paml_diff:.6f}"
    )

    print(f"\n✓ Alternative model matches previous validation:")
    print(f"  Our lnL:      {results['alt_lnL']:.6f}")
    print(f"  Expected lnL: {expected_lnL:.6f}")
    print(f"  PAML lnL:     {paml_lnL:.6f}")
    print(f"  Difference from PAML: {paml_diff:.6f}")


def test_branch_site_lrt_calculation():
    """Test that LRT statistic is calculated correctly."""

    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    results = branch_site_test(
        alignment=alignment,
        tree=tree,
        use_f3x4=True,
        optimize_branch_lengths=True,
        method='L-BFGS-B',
        maxiter=500
    )

    # Manually calculate LRT
    expected_lrt = 2 * (results['alt_lnL'] - results['null_lnL'])

    # Should match (within floating point precision)
    lrt_diff = abs(results['lrt_statistic'] - expected_lrt)
    assert lrt_diff < 1e-10, (
        f"LRT calculation incorrect: "
        f"got {results['lrt_statistic']:.10f}, "
        f"expected {expected_lrt:.10f}"
    )

    print(f"\n✓ LRT statistic correctly calculated:")
    print(f"  2 * (alt_lnL - null_lnL) = {expected_lrt:.6f}")
    print(f"  LRT statistic = {results['lrt_statistic']:.6f}")


if __name__ == "__main__":
    test_branch_site_test_function()
    test_branch_site_test_parameter_values()
    test_branch_site_test_matches_paml_alternative()
    test_branch_site_lrt_calculation()
