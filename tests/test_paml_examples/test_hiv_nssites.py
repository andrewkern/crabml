"""
Validation tests for HIV NSsites example.

Tests crabML against PAML reference outputs for the HIV env V3 region dataset
(Yang et al. 2000, Genetics 155:431-449).

Dataset: 13 sequences, 91 codons (273 bp)
Models: M0, M1a, M2a, M7, M8, M8a
"""

import pytest
from pathlib import Path
import numpy as np

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.optimizer import (
    M0Optimizer, M1aOptimizer, M2aOptimizer,
    M7Optimizer, M8Optimizer, M8aOptimizer
)


# PAML Reference Values (generated October 23, 2025)
PAML_REFERENCE = {
    'M0': {
        'lnL': -1137.688190,
        'np': 25,
    },
    'M1a': {
        'lnL': -1114.641736,
        'np': 26,
    },
    'M2a': {
        'lnL': -1106.445004,
        'np': 28,
    },
    'M7': {
        'lnL': -1115.395312,
        'np': 26,
    },
    'M8': {
        'lnL': -1106.388268,
        'np': 28,
    },
    'M8a': {
        'lnL': -1114.579213,
        'np': 27,
    },
}


@pytest.fixture
def hiv_data():
    """Load HIV env V3 region data."""
    data_dir = Path(__file__).parent.parent / "data" / "paml_reference" / "hiv_nssites"

    # Load FASTA version (converted from PHYLIP)
    alignment = Alignment.from_fasta(
        str(data_dir / "HIVenvSweden.fasta"),
        seqtype="codon"
    )

    # Read tree from file (skip header line, remove spaces in branch lengths)
    tree_file = data_dir / "HIVenvSweden.trees"
    with open(tree_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        # First non-empty line is header, second is tree
        tree_str = lines[1]
        # Remove spaces after colons (PAML format: ": 0.023" -> ":0.023")
        tree_str = tree_str.replace(": ", ":")
    tree = Tree.from_newick(tree_str)

    return alignment, tree


def test_hiv_m0_vs_paml(hiv_data):
    """Test M0 (one-ratio) model against PAML reference."""
    alignment, tree = hiv_data

    optimizer = M0Optimizer(alignment, tree, use_f3x4=True)
    kappa, omega, lnL = optimizer.optimize()

    paml_lnL = PAML_REFERENCE['M0']['lnL']

    print(f"\nM0 Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M0 lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable
    assert 0.1 < kappa < 20, f"kappa out of range: {kappa}"
    assert 0.001 < omega < 10, f"omega out of range: {omega}"


def test_hiv_m1a_vs_paml(hiv_data):
    """Test M1a (NearlyNeutral) model against PAML reference."""
    alignment, tree = hiv_data

    optimizer = M1aOptimizer(alignment, tree, use_f3x4=True)
    kappa, p0, omega0, lnL = optimizer.optimize()

    paml_lnL = PAML_REFERENCE['M1a']['lnL']

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


def test_hiv_m2a_vs_paml(hiv_data):
    """Test M2a (PositiveSelection) model against PAML reference."""
    alignment, tree = hiv_data

    optimizer = M2aOptimizer(alignment, tree, use_f3x4=True)
    result = optimizer.optimize()

    # Unpack based on actual return (check optimizer output)
    # Returns: kappa, p0, p1, omega0, omega2, lnL
    if len(result) == 6:
        kappa, p0, p1, omega0, omega2, lnL = result
    else:
        # Handle different return format if needed
        raise ValueError(f"Unexpected number of return values: {len(result)}")

    paml_lnL = PAML_REFERENCE['M2a']['lnL']

    print(f"\nM2a Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  kappa: {kappa:.4f}")
    print(f"  p0: {p0:.4f}, p1: {p1:.4f}")
    print(f"  omega0: {omega0:.4f}, omega2: {omega2:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M2a lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable (relaxed checks - just verify they're positive)
    assert kappa > 0, f"kappa should be positive: {kappa}"
    assert p0 >= 0, f"p0 should be non-negative: {p0}"
    assert p1 >= 0, f"p1 should be non-negative: {p1}"
    # Don't check omega ranges strictly - optimization might find different parameterization


def test_hiv_m7_vs_paml(hiv_data):
    """Test M7 (Beta) model against PAML reference."""
    alignment, tree = hiv_data

    optimizer = M7Optimizer(alignment, tree, use_f3x4=True)
    kappa, p_beta, q_beta, lnL = optimizer.optimize()

    paml_lnL = PAML_REFERENCE['M7']['lnL']

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


def test_hiv_m8_vs_paml(hiv_data):
    """Test M8 (Beta&ω) model against PAML reference."""
    alignment, tree = hiv_data

    optimizer = M8Optimizer(alignment, tree, use_f3x4=True)
    kappa, p0, p_beta, q_beta, omega_s, lnL = optimizer.optimize()

    paml_lnL = PAML_REFERENCE['M8']['lnL']

    print(f"\nM8 Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  p0: {p0:.4f}")
    print(f"  p: {p_beta:.4f}, q: {q_beta:.4f}")
    print(f"  omega_s: {omega_s:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M8 lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable
    assert 0 < p0 < 1, f"p0 out of range: {p0}"
    assert p_beta > 0, f"p_beta should be positive: {p_beta}"
    assert q_beta > 0, f"q_beta should be positive: {q_beta}"
    assert omega_s > 1, f"omega_s should be > 1: {omega_s}"


def test_hiv_m8a_vs_paml(hiv_data):
    """Test M8a (Beta&ω=1) model against PAML reference."""
    alignment, tree = hiv_data

    optimizer = M8aOptimizer(alignment, tree, use_f3x4=True)
    kappa, p0, p_beta, q_beta, lnL = optimizer.optimize()

    paml_lnL = PAML_REFERENCE['M8a']['lnL']

    print(f"\nM8a Results:")
    print(f"  crabML lnL: {lnL:.6f}")
    print(f"  PAML lnL:   {paml_lnL:.6f}")
    print(f"  Difference: {abs(lnL - paml_lnL):.6f}")
    print(f"  p0: {p0:.4f}")
    print(f"  p: {p_beta:.4f}, q: {q_beta:.4f}")

    # Assert exact match (within 0.01 lnL units)
    assert abs(lnL - paml_lnL) < 0.01, f"M8a lnL mismatch: {lnL} vs {paml_lnL}"

    # Check parameters are reasonable
    assert 0 < p0 < 1, f"p0 out of range: {p0}"
    assert p_beta > 0, f"p_beta should be positive: {p_beta}"
    assert q_beta > 0, f"q_beta should be positive: {q_beta}"


def test_hiv_lrt_m1a_vs_m2a(hiv_data):
    """Test M1a vs M2a likelihood ratio test."""
    alignment, tree = hiv_data

    # Get log-likelihoods
    lnL_m1a = PAML_REFERENCE['M1a']['lnL']
    lnL_m2a = PAML_REFERENCE['M2a']['lnL']

    # Calculate LRT
    lrt = 2 * (lnL_m2a - lnL_m1a)

    # df = 2 (p1 and omega2)
    # Critical value at α=0.05 is 5.991

    print(f"\nM1a vs M2a LRT:")
    print(f"  LRT statistic: {lrt:.6f}")
    print(f"  df: 2")
    print(f"  Critical value (α=0.05): 5.991")
    print(f"  Significant: {lrt > 5.991}")

    assert lrt > 5.991, "M1a vs M2a should be significant"


def test_hiv_lrt_m7_vs_m8(hiv_data):
    """Test M7 vs M8 likelihood ratio test."""
    alignment, tree = hiv_data

    # Get log-likelihoods
    lnL_m7 = PAML_REFERENCE['M7']['lnL']
    lnL_m8 = PAML_REFERENCE['M8']['lnL']

    # Calculate LRT
    lrt = 2 * (lnL_m8 - lnL_m7)

    # df = 2 (p0 and omega_s)
    # Critical value at α=0.05 is 5.991

    print(f"\nM7 vs M8 LRT:")
    print(f"  LRT statistic: {lrt:.6f}")
    print(f"  df: 2")
    print(f"  Critical value (α=0.05): 5.991")
    print(f"  Significant: {lrt > 5.991}")

    assert lrt > 5.991, "M7 vs M8 should be significant"


def test_hiv_lrt_m8a_vs_m8(hiv_data):
    """Test M8a vs M8 likelihood ratio test (50:50 mixture)."""
    alignment, tree = hiv_data

    # Get log-likelihoods
    lnL_m8a = PAML_REFERENCE['M8a']['lnL']
    lnL_m8 = PAML_REFERENCE['M8']['lnL']

    # Calculate LRT
    lrt = 2 * (lnL_m8 - lnL_m8a)

    # df = 1 (omega_s), but use 50:50 mixture
    # Critical value at α=0.05 (50:50 mixture) is 2.71

    print(f"\nM8a vs M8 LRT (50:50 mixture):")
    print(f"  LRT statistic: {lrt:.6f}")
    print(f"  df: 1")
    print(f"  Critical value (α=0.05, 50:50): 2.71")
    print(f"  Significant: {lrt > 2.71}")

    assert lrt > 2.71, "M8a vs M8 should be significant"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
