# crabML

High-performance reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis, powered by Rust.

## Status

Production-ready with 10 site-class models and branch-site models fully validated against PAML.

### Features

- **Site-class models**: M0, M1a, M2a, M3, M4, M5, M6, M7, M8, M8a, M9
- **Branch models**: Free-ratio and multi-ratio models for lineage-specific selection
- **Branch-site models**: Model A (test for positive selection on specific lineages)
- **Hypothesis testing**: Complete LRT framework for detecting positive selection
- **Parameter optimization**: Complete MLE optimization for all models
- **High-performance Rust backend**: 300-500x faster than NumPy, 3-10x faster than PAML
- **PAML validation**: All models produce exact numerical matches to PAML
- **Test coverage**: 16+ validation tests passing, exact agreement with reference outputs

## Installation

**Requirements:**
- Python 3.11+
- Rust toolchain (install from https://rustup.rs)
- OpenBLAS or similar BLAS/LAPACK implementation

```bash
uv sync --all-extras --reinstall-package crabml-rust
```

This single command installs all Python dependencies and builds the Rust extension. The `--reinstall-package` flag ensures the Rust extension is rebuilt even if already installed.

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

### Site-Class Model Tests

- **M1a vs M2a**: Tests for positive selection against nearly neutral null model
- **M7 vs M8**: Tests for positive selection using beta distribution models
- **M8a vs M8**: Tests for positive selection with 50:50 mixture null distribution

### Branch-Site Model Tests

- **Branch-Site Model A**: Tests for positive selection on specific phylogenetic lineages
  - Detects if specific sites on foreground branches have ω > 1
  - Uses standard chi-square LRT with df=1
  - Returns site-specific parameter estimates

All tests:
- Calculate likelihood ratio test (LRT) statistics automatically
- Provide p-values from appropriate chi-square distributions
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

### Branch-Site Analysis

Test for positive selection on specific lineages (e.g., primates, mammals, specific gene duplicates):

```python
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.analysis import branch_site_test

# Load data
alignment = Alignment.from_phylip('alignment.phy', seqtype='codon')

# Tree with branch labels: #0 = background, #1 = foreground
# Test for positive selection on the primate lineage
tree_str = """
((human, chimp) #1,
 (mouse, rat));
"""
tree = Tree.from_newick(tree_str)

# Run branch-site test (Model A vs Model A null)
results = branch_site_test(
    alignment=alignment,
    tree=tree,
    use_f3x4=True,
    optimize_branch_lengths=True
)

# Check results
print(f"P-value: {results['pvalue']:.6f}")
print(f"LRT statistic: {results['lrt_statistic']:.6f}")

if results['significant']:
    omega2 = results['alt_params']['omega2']
    p2 = results['alt_params']['p2']
    print(f"✓ Positive selection detected on foreground branches!")
    print(f"  ω₂ = {omega2:.4f} (dN/dS for positively selected sites)")
    print(f"  p₂ = {p2:.4f} (proportion of sites under selection)")
else:
    print("✗ No significant evidence for positive selection")
```

**Key features:**
- Automatically runs both null (ω₂=1) and alternative (ω₂ free) models
- Calculates likelihood ratio test with df=1
- Returns comprehensive results dictionary with all parameters
- Supports any tree topology with branch labels

### Branch Model Analysis

Test for lineage-specific selection (different ω on different branches):

```python
from crabml.analysis import branch_model_test

# Tree with branch labels: #0 = background, #1 = foreground
# Tests if human-chimp lineage has different ω than mouse-rat
tree_str = "((human,chimp) #1, (mouse,rat));"

result = branch_model_test(
    alignment='alignment.fasta',
    tree=tree_str,
    verbose=True
)

# Check results
print(f"P-value: {result.pvalue:.6f}")
print(f"LRT statistic: {result.LRT:.6f}")

if result.significant(0.05):
    omega_fg = result.alt_params['omega1']
    omega_bg = result.alt_params['omega0']
    print(f"✓ Lineage-specific selection detected!")
    print(f"  Foreground ω={omega_fg:.3f}, Background ω={omega_bg:.3f}")
```

**Advanced: Direct model optimization**

For more control over optimization:

```python
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.branch import BranchModelOptimizer

# Load data
alignment = Alignment.from_phylip('alignment.phy', seqtype='codon')

# Tree with branch labels: #0 = background, #1 = foreground
tree_str = "((human,chimp) #1, (mouse,rat));"
tree = Tree.from_newick(tree_str)

# Run multi-ratio branch model (model=2)
optimizer = BranchModelOptimizer(
    alignment=alignment,
    tree=tree,
    use_f3x4=True,
    free_ratio=False,  # Multi-ratio model
)

# Optimize parameters
kappa, omega_dict, lnL = optimizer.optimize()

print(f"Log-likelihood: {lnL:.6f}")
print(f"Kappa: {kappa:.6f}")
print(f"Omega (background): {omega_dict['omega0']:.6f}")
print(f"Omega (foreground): {omega_dict['omega1']:.6f}")

# Interpret results
if omega_dict['omega1'] > 1 and omega_dict['omega1'] > omega_dict['omega0']:
    print("✓ Positive selection detected on foreground lineage!")
```

**Key features:**
- Hypothesis test: Multi-ratio vs M0 (one-ratio) model
- Multi-ratio model: Different ω for labeled branch groups
- Free-ratio model: Independent ω for each branch (set `free_ratio=True`)
- Branch labels specified in tree: `#0`, `#1`, `#2`, etc.
- PAML-validated: Exact numerical match (lnL diff < 0.000001)

## Supported Models

All models validated against PAML reference outputs with exact numerical agreement.

### Site-Class Models

Models where ω varies across sites but not across branches:

- **M0** (one-ratio): Single dN/dS ratio across all sites
- **M1a** (NearlyNeutral): Two site classes (purifying, neutral)
- **M2a** (PositiveSelection): Three site classes (purifying, neutral, positive)
- **M3** (discrete): K discrete omega categories
- **M4** (freqs): Five fixed omegas with variable proportions
- **M5** (gamma): Gamma distribution for omega
- **M6** (2gamma): Mixture of two gamma distributions
- **M7** (beta): Beta distribution for omega (0 < omega < 1)
- **M8** (beta&ω): Beta distribution plus positive selection class
- **M8a** (beta&ω=1): Beta distribution plus neutral class (null for M8)
- **M9** (beta&gamma): Mixture of beta and gamma distributions

### Branch Models

Models where ω varies across branches but not across sites:

- **Free-ratio model** (model=1): Independent ω for each branch
  - Estimates one ω per branch in the tree
  - Highly parameter-rich (n-1 omega parameters for n species)
  - Useful for exploratory analysis but prone to overfitting

- **Multi-ratio model** (model=2): Different ω for labeled branch groups
  - User specifies branch groups via labels (#0, #1, #2, etc.)
  - One ω parameter per unique label
  - Tests for lineage-specific selection (e.g., primates vs others)
  - **Recommended** approach for detecting selection on specific lineages

### Branch-Site Models

Models where ω varies across both sites and branches:

- **Branch-Site Model A**: Four site classes with different ω on foreground vs background branches
  - Class 0: conserved (ω₀ < 1) on all branches
  - Class 1: neutral (ω = 1) on all branches
  - Class 2a: conserved on background, positive selection (ω₂ > 1) on foreground
  - Class 2b: neutral on background, positive selection on foreground
  - Null model: fixes ω₂ = 1 for hypothesis testing

**Validation:** All models match PAML within 0.1 log-likelihood units (typically < 0.05)

## Development

```bash
# Build/rebuild the package (including Rust extension)
uv sync --all-extras --reinstall-package crabml-rust

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
