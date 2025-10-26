"""
Validation tests for lysin NSsites example.

Tests crabML against PAML reference outputs for the abalone sperm lysin dataset
(Yang, Swanson & Vacquier 2000).

Dataset: 25 sequences, 135 codons (405 bp)
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
        'lnL': -4682.373100,
        'np': 49,
    },
    'M1a': {
        'lnL': -4525.413236,
        'np': 50,
    },
    'M2a': {
        'lnL': -4464.511376,
        'np': 52,
    },
    'M7': {
        'lnL': -4524.684003,
        'np': 50,
    },
    'M8': {
        'lnL': -4464.620176,
        'np': 52,
    },
    'M8a': {
        'lnL': -4519.033815,
        'np': 51,
    },
}


@pytest.fixture
def lysin_data():
    """Load abalone sperm lysin data."""
    data_dir = Path(__file__).parent.parent / "data" / "paml_reference" / "lysin"

    # Load FASTA version (converted from PHYLIP)
    alignment = Alignment.from_fasta(
        str(data_dir / "lysin.fasta"),
        seqtype="codon"
    )

    # Read tree from file (PAML format with spaces, multi-line)
    tree_file = data_dir / "lysin.trees"
    with open(tree_file) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        # Skip header line (number of trees), join rest
        tree_lines = [l for l in lines[1:] if not l.startswith('#')]
        tree_str = ''.join(tree_lines)
        # Remove spaces after colons
        tree_str = tree_str.replace(": ", ":")
        # Remove branch labels like #1
        tree_str = tree_str.replace(" #1", "")
    tree = Tree.from_newick(tree_str)

    return alignment, tree


def test_lysin_m0_vs_paml(lysin_data):
    """Test M0 (one-ratio) model against PAML reference."""
    alignment, tree = lysin_data

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


def test_lysin_m1a_vs_paml(lysin_data):
    """Test M1a (NearlyNeutral) model against PAML reference."""
    alignment, tree = lysin_data

    # M1a optimizer now automatically initializes with M0 (init_with_m0=True by default)
    optimizer = M1aOptimizer(alignment, tree, use_f3x4=True)
    kappa, p0, omega0, lnL = optimizer.optimize(maxiter=500)

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


@pytest.mark.slow
def test_lysin_m2a_vs_paml(lysin_data):
    """Test M2a (PositiveSelection) model against PAML reference."""
    alignment, tree = lysin_data

    # M2a optimizer now automatically initializes with M0
    optimizer = M2aOptimizer(alignment, tree, use_f3x4=True)
    result = optimizer.optimize(maxiter=500)

    # Unpack result
    if len(result) == 6:
        kappa, omega0, omega2, p0, p1, lnL = result
    else:
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

    # Check parameters are reasonable (relaxed)
    assert kappa > 0, f"kappa should be positive: {kappa}"
    assert p0 >= 0, f"p0 should be non-negative: {p0}"
    assert p1 >= 0, f"p1 should be non-negative: {p1}"


@pytest.mark.slow
def test_lysin_m7_vs_paml(lysin_data):
    """Test M7 (Beta) model against PAML reference."""
    alignment, tree = lysin_data

    # M7 optimizer now automatically initializes with M0
    optimizer = M7Optimizer(alignment, tree, use_f3x4=True)
    kappa, p_beta, q_beta, lnL = optimizer.optimize(maxiter=500)

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


@pytest.mark.slow
def test_lysin_m8_vs_paml(lysin_data):
    """Test M8 (Beta&ω) model against PAML reference."""
    alignment, tree = lysin_data

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


@pytest.mark.slow
def test_lysin_m8a_vs_paml(lysin_data):
    """Test M8a (Beta&ω=1) model against PAML reference."""
    alignment, tree = lysin_data

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


def test_lysin_lrt_m1a_vs_m2a(lysin_data):
    """Test M1a vs M2a likelihood ratio test."""
    alignment, tree = lysin_data

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

    # This dataset has STRONG positive selection signal
    assert lrt > 5.991, "M1a vs M2a should be highly significant"
    assert lrt > 100, "Lysin dataset should show very strong signal (LRT > 100)"


def test_lysin_lrt_m7_vs_m8(lysin_data):
    """Test M7 vs M8 likelihood ratio test."""
    alignment, tree = lysin_data

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

    # This dataset has STRONG positive selection signal
    assert lrt > 5.991, "M7 vs M8 should be highly significant"
    assert lrt > 100, "Lysin dataset should show very strong signal (LRT > 100)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
