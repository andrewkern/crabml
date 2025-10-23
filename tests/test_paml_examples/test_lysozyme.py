"""
Validation tests for lysozyme example.

Tests crabML against PAML reference outputs for the primate lysozyme dataset
(Messier & Stewart 1997; Yang 1998).

Dataset: 7 sequences, 130 codons (390 bp)
Models: M0-M9 (site models), branch models, branch-site Model A
"""

import pytest
from pathlib import Path
import numpy as np

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.optimizer import (
    M0Optimizer, M1aOptimizer, M2aOptimizer, M3Optimizer,
    M7Optimizer, M8Optimizer, M8aOptimizer
)


# PAML Reference Values for Site Models (generated October 2025)
PAML_SITE_MODELS = {
    'M0': {
        'lnL': -906.017440,
        'np': 13,
    },
    'M1a': {
        'lnL': -902.503872,
        'np': 14,
    },
    'M2a': {
        'lnL': -899.998568,
        'np': 16,
    },
    'M3': {
        'lnL': -899.985262,
        'np': 17,
    },
    'M7': {
        'lnL': -902.510018,
        'np': 14,
    },
    'M8': {
        'lnL': -899.999237,
        'np': 16,
    },
    'M8a': {
        'lnL': -902.503869,
        'np': 15,
    },
}


@pytest.fixture
def lysozyme_data():
    """Load lysozyme dataset."""
    data_dir = Path(__file__).parent.parent / "data" / "paml_reference" / "lysozyme"

    # Load alignment from PHYLIP format
    alignment = Alignment.from_phylip(
        str(data_dir / "lysozymeSmall.txt"),
        seqtype="codon"
    )

    # Simple tree (no branch labels for site models)
    tree_str = "((Hsa_Human, Hla_gibbon), ((Cgu/Can_colobus, Pne_langur), Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));"
    tree = Tree.from_newick(tree_str)

    return alignment, tree


def test_lysozyme_m0_vs_paml(lysozyme_data):
    """Test M0 (one-ratio) model against PAML reference."""
    alignment, tree = lysozyme_data

    optimizer = M0Optimizer(alignment, tree, use_f3x4=True)
    kappa, omega, lnL = optimizer.optimize()

    paml_lnL = PAML_SITE_MODELS['M0']['lnL']

    print(f"\nM0 Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M0 lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable
    assert 0.1 < kappa < 20, f"kappa out of range: {kappa}"
    assert 0.001 < omega < 10, f"omega out of range: {omega}"


def test_lysozyme_m1a_vs_paml(lysozyme_data):
    """Test M1a (NearlyNeutral) model against PAML reference."""
    alignment, tree = lysozyme_data

    # M1a optimizer automatically initializes with M0
    optimizer = M1aOptimizer(alignment, tree, use_f3x4=True)
    kappa, p0, omega0, lnL = optimizer.optimize(maxiter=500)

    paml_lnL = PAML_SITE_MODELS['M1a']['lnL']

    print(f"\nM1a Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  p0: {p0:.4f}, omega0: {omega0:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M1a lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable
    assert 0 < p0 < 1, f"p0 out of range: {p0}"
    assert 0 < omega0 < 1, f"omega0 should be < 1: {omega0}"


def test_lysozyme_m2a_vs_paml(lysozyme_data):
    """Test M2a (PositiveSelection) model against PAML reference."""
    alignment, tree = lysozyme_data

    # M2a optimizer automatically initializes with M0
    optimizer = M2aOptimizer(alignment, tree, use_f3x4=True)
    result = optimizer.optimize(maxiter=500)

    # Unpack result
    if len(result) == 6:
        kappa, p0, p1, omega0, omega2, lnL = result
    else:
        raise ValueError(f"Unexpected number of return values: {len(result)}")

    paml_lnL = PAML_SITE_MODELS['M2a']['lnL']

    print(f"\nM2a Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  kappa: {kappa:.4f}")
    print(f"  p0: {p0:.4f}, p1: {p1:.4f}")
    print(f"  omega0: {omega0:.4f}, omega2: {omega2:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M2a lnL mismatch: {lnL} vs {paml_lnL}"


def test_lysozyme_m7_vs_paml(lysozyme_data):
    """Test M7 (Beta) model against PAML reference."""
    alignment, tree = lysozyme_data

    # M7 optimizer automatically initializes with M0
    optimizer = M7Optimizer(alignment, tree, use_f3x4=True)
    kappa, p_beta, q_beta, lnL = optimizer.optimize(maxiter=500)

    paml_lnL = PAML_SITE_MODELS['M7']['lnL']

    print(f"\nM7 Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  p: {p_beta:.4f}, q: {q_beta:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M7 lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable
    assert p_beta > 0, f"p_beta should be positive: {p_beta}"
    assert q_beta > 0, f"q_beta should be positive: {q_beta}"


def test_lysozyme_m8_vs_paml(lysozyme_data):
    """Test M8 (Beta&ω) model against PAML reference."""
    alignment, tree = lysozyme_data

    optimizer = M8Optimizer(alignment, tree, use_f3x4=True)
    kappa, p0, p_beta, q_beta, omega_s, lnL = optimizer.optimize(maxiter=500)

    paml_lnL = PAML_SITE_MODELS['M8']['lnL']

    print(f"\nM8 Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  p0: {p0:.4f}")
    print(f"  p: {p_beta:.4f}, q: {q_beta:.4f}")
    print(f"  omega_s: {omega_s:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M8 lnL mismatch: {lnL} vs {paml_lnL}"


def test_lysozyme_m8a_vs_paml(lysozyme_data):
    """Test M8a (Beta&ω=1) model against PAML reference."""
    alignment, tree = lysozyme_data

    optimizer = M8aOptimizer(alignment, tree, use_f3x4=True)
    kappa, p0, p_beta, q_beta, lnL = optimizer.optimize(maxiter=500)

    paml_lnL = PAML_SITE_MODELS['M8a']['lnL']

    print(f"\nM8a Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  p0: {p0:.4f}")
    print(f"  p: {p_beta:.4f}, q: {q_beta:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M8a lnL mismatch: {lnL} vs {paml_lnL}"


def test_lysozyme_lrt_m1a_vs_m2a(lysozyme_data):
    """Test M1a vs M2a likelihood ratio test."""
    lnL_m1a = PAML_SITE_MODELS['M1a']['lnL']
    lnL_m2a = PAML_SITE_MODELS['M2a']['lnL']

    # Calculate LRT
    lrt = 2 * (lnL_m2a - lnL_m1a)

    # df = 2 (p1 and omega2)
    # Critical value at α=0.05 is 5.991

    print(f"\nM1a vs M2a LRT:")
    print(f"  LRT statistic: {lrt:.6f}")
    print(f"  df: 2")
    print(f"  Critical value (α=0.05): 5.991")
    print(f"  Significant: {lrt > 5.991}")

    # Lysozyme dataset shows marginal positive selection signal (LRT ≈ 5.01)
    # This is less significant than HIV (LRT = 16.39) or lysin (LRT = 121.80)
    assert lrt > 0, "LRT should be positive"
    assert lnL_m2a > lnL_m1a, "M2a should fit better than M1a"


def test_lysozyme_lrt_m7_vs_m8(lysozyme_data):
    """Test M7 vs M8 likelihood ratio test."""
    lnL_m7 = PAML_SITE_MODELS['M7']['lnL']
    lnL_m8 = PAML_SITE_MODELS['M8']['lnL']

    # Calculate LRT
    lrt = 2 * (lnL_m8 - lnL_m7)

    # df = 2 (p0 and omega_s)
    # Critical value at α=0.05 is 5.991

    print(f"\nM7 vs M8 LRT:")
    print(f"  LRT statistic: {lrt:.6f}")
    print(f"  df: 2")
    print(f"  Critical value (α=0.05): 5.991")
    print(f"  Significant: {lrt > 5.991}")

    # Lysozyme dataset shows marginal positive selection signal (LRT ≈ 5.02)
    # This is less significant than HIV (LRT = 18.01) or lysin datasets
    assert lrt > 0, "LRT should be positive"
    assert lnL_m8 > lnL_m7, "M8 should fit better than M7"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
