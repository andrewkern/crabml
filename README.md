# crabML

High-performance reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis, powered by Rust.

## Status

Production-ready with 10 PAML codon models fully validated against reference outputs.

### Features

- **Codon substitution models**: M0, M1a, M2a, M3, M4, M5, M6, M7, M8, M9
- **Parameter optimization**: Complete MLE optimization for all 10 models
- **High-performance Rust backend**: 3-10x faster than PAML for full optimization
- **PAML validation**: All models produce exact numerical matches to PAML
- **Test coverage**: 12 validation tests passing, exact agreement with reference outputs

## Installation

**Requirements:**
- Python 3.11+
- Rust toolchain (install from https://rustup.rs)
- OpenBLAS or similar BLAS/LAPACK implementation

```bash
uv sync --all-extras
```

This single command installs all Python dependencies and builds the Rust extension.

## Quick Start

### Testing for Positive Selection (Recommended)

The easiest way to test for positive selection:

```python
from crabml.analysis import test_positive_selection

# Run standard likelihood ratio tests
results = test_positive_selection(
    alignment='lysozyme.fasta',
    tree='lysozyme.tree',
    test='both'  # Runs both M1a vs M2a and M7 vs M8
)

# Check results
print(results['M1a_vs_M2a'].summary())
print(results['M7_vs_M8'].summary())

# Get p-values
if results['M1a_vs_M2a'].significant(0.05):
    print(f"Positive selection detected! ω = {results['M1a_vs_M2a'].omega_positive:.2f}")
```

### Advanced: Direct Model Optimization

For more control over optimization:

```python
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize import M2aOptimizer

# Load data
alignment = Alignment.from_fasta("alignment.fasta", seqtype='codon')
tree = Tree.from_newick("tree.nwk")

# Run M2a model optimization
optimizer = M2aOptimizer(alignment, tree)
kappa, omega0, omega2, p0, p1, lnL = optimizer.optimize()

print(f"Log-likelihood: {lnL:.6f}")
print(f"ω for positive selection: {omega2:.4f}")
```

## Hypothesis Testing

crabML provides publication-ready hypothesis tests for detecting positive selection:

### Standard Tests

- **M1a vs M2a**: Tests for positive selection against nearly neutral null model
- **M7 vs M8**: Tests for positive selection using beta distribution models

Both tests:
- Calculate likelihood ratio test (LRT) statistics automatically
- Provide p-values from chi-square distribution
- Include formatted output suitable for publications
- Export results to dict/JSON for further analysis

### Example Output

```
================================================================================
Likelihood Ratio Test for Positive Selection
================================================================================

Test: M1a vs M2a

NULL MODEL (M1a):
  Log-likelihood: -902.503872
  Parameters:
    p0 = 0.4923 (proportion ω < 1)
    ω0 = 0.0538
    κ  = 2.2945

ALTERNATIVE MODEL (M2a):
  Log-likelihood: -899.998568
  Parameters:
    p2 = 0.2075 (proportion ω > 1)
    ω2 = 3.4472

LIKELIHOOD RATIO TEST:
  LRT statistic: 5.0106
  Degrees of freedom: 2
  P-value: 0.0817

CONCLUSION:
  No significant evidence for positive selection (α = 0.05)
================================================================================
```

## Supported Models

All models validated against PAML reference outputs with exact numerical agreement:

- **M0** (one-ratio): Single dN/dS ratio across all sites
- **M1a** (NearlyNeutral): Two site classes (purifying, neutral)
- **M2a** (PositiveSelection): Three site classes (purifying, neutral, positive)
- **M3** (discrete): K discrete omega categories
- **M4** (freqs): Five fixed omegas with variable proportions
- **M5** (gamma): Gamma distribution for omega
- **M6** (2gamma): Mixture of two gamma distributions
- **M7** (beta): Beta distribution for omega (0 < omega < 1)
- **M8** (beta&omega): Beta distribution plus positive selection class
- **M9** (beta&gamma): Mixture of beta and gamma distributions

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=pycodeml --cov-report=html

# Run specific test
uv run pytest tests/test_reference/test_matrix.py -v

# Run Rust tests
cd rust
cargo test
cd ..

# Run PAML validation tests
uv run pytest tests/test_rust/test_paml_rust_validation.py -v
```

## License

GPL-3.0-or-later
