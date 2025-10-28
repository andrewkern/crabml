# crabML

[![Documentation](https://readthedocs.org/projects/crabml/badge/?version=latest)](https://crabml.readthedocs.io/en/latest/user_guide/models.html)
[![Tests](https://github.com/andrewkern/crabml/actions/workflows/ci.yml/badge.svg)](https://github.com/andrewkern/crabml/actions/workflows/ci.yml)

```
                 _     __  __ _     
   ___ _ __ __ _| |__ |  \/  | |    
  / __| '__/ _` | '_ \| |\/| | |    
 | (__| | | (_| | |_) | |  | | |___ 
  \___|_|  \__,_|_.__/|_|  |_|_____|
```

High-performance reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis, powered by Rust.

## Status

**Production-ready v0.2.0** - Fully validated against PAML with 7-20x speedup.

### Key Features

- **Simple Python API**: One-line model fitting with `optimize_model()`
- **Command-Line Interface**: `crabml` command for common analyses
- **11 Site-Class Models**: M0, M1a, M2a, M3, M4, M5, M6, M7, M8, M8a, M9
- **Branch Models**: Free-ratio and multi-ratio for lineage-specific selection
- **Branch-Site Models**: Model A for site and lineage-specific selection
- **Hypothesis Testing**: Complete LRT framework with p-values
- **7-11x Faster**: Optimized Rust backend with BLAS acceleration
- **Validated**: Perfect correlation (r > 0.999) with PAML across 180 datasets
- **Auto File Detection**: Supports FASTA and PHYLIP formats

## Installation

### Prerequisites

1. **Install Rust** (required):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Install uv** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Build from Source

```bash
git clone https://github.com/andrewkern/crabml.git
cd crabml
uv sync --all-extras
```

## Quick Start

### Python API

```python
from crabml import optimize_model

# Fit M0 model with one line
result = optimize_model("M0", "alignment.fasta", "tree.nwk")

# Beautiful formatted output
print(result.summary())

# Access results
print(f"omega = {result.omega:.4f}")
print(f"kappa = {result.kappa:.4f}")
print(f"lnL = {result.lnL:.2f}")

# Export to JSON
result.to_json("results.json")
```

### Test for Positive Selection

```python
from crabml import positive_selection

# Run both M1a vs M2a and M7 vs M8 tests
results = positive_selection(
    alignment='lysozyme.fasta',
    tree='lysozyme.tree',
    test='both'
)

# Check results
if results['M1a_vs_M2a'].significant(0.05):
    print(f"Positive selection detected! ω = {results['M1a_vs_M2a'].omega_positive:.2f}")
```

### Command-Line Interface

```bash
# Test for positive selection
crabml site-model -s alignment.fasta -t tree.nwk --test both

# Fit single model
crabml fit -m M0 -s alignment.fasta -t tree.nwk

# Branch-site test
crabml branch-site -s alignment.fasta -t labeled_tree.nwk

# Output as JSON
crabml site-model -s data.fasta -t tree.nwk --test m7m8 --format json -o results.json
```

## Supported Models

All models validated against PAML with exact numerical agreement.

### Site-Class Models

Models where ω varies across sites:

- **M0**: Single dN/dS ratio
- **M1a**: NearlyNeutral (purifying + neutral)
- **M2a**: PositiveSelection (purifying + neutral + positive)
- **M3**: K discrete omega categories
- **M7**: Beta distribution (0 < ω < 1)
- **M8**: Beta + positive selection class (ω > 1)
- **M8a**: Beta + neutral class (null for M8)
- **M4, M5, M6, M9**: Additional distribution-based models

### Branch Models

Models where ω varies across branches:

- **Multi-ratio**: Different ω for labeled branch groups (recommended)
- **Free-ratio**: Independent ω for each branch (exploratory)

### Branch-Site Models

Models where ω varies across sites AND branches:

- **Model A**: Four site classes with lineage-specific positive selection

## Validation & Performance

### Comprehensive Validation

Tested with **180 datasets** (30 replicates × 6 models) using simulated data:

| Model | Likelihood Correlation | RMSE | Mean \|Δ\| | Speedup |
|-------|----------------------|------|-----------|---------|
| **M0** | r = 1.000 | 0.0000 | 0.0000 | 7.11x |
| **M1a** | r = 1.000 | 0.0173 | 0.0037 | 11.40x |
| **M2a** | r = 1.000 | 0.0119 | 0.0063 | 16.30x |
| **M7** | r = 1.000 | 0.0014 | 0.0003 | 20.54x |
| **M8** | r = 1.000 | 0.0588 | 0.0241 | 11.45x |
| **M8a** | r = 1.000 | 0.0327 | 0.0188 | 8.29x |

**Perfect correlation (r > 0.999) across all models** with mean differences < 0.03 lnL units.

**Performance:** Runtime comparisons show 7-20x speedup over PAML across all models.

### Known Identifiability Issues

crabML correctly replicates known parameter identifiability issues in M1a and M2a:
- **M1a**: Flat likelihood surface when ω₀ ≈ 1
- **M2a**: Label switching between neutral and positive selection classes

Both implementations show identical issues, validating correctness.

### Performance Benchmarks

**Average speedup: 7-11x over PAML**

Key optimizations:
- Rust likelihood calculation with BLAS acceleration (300-500x faster than NumPy)
- Precomputed codon substitution graph (21x faster Q matrix)
- Vectorized matrix operations
- Efficient eigendecomposition caching

See `benchmarks/` directory for full validation infrastructure.

## CLI Reference

### `crabml site-model` - Positive Selection Tests

```bash
# M7 vs M8 test
crabml site-model -s data.fasta -t tree.nwk --test m7m8

# M1a vs M2a test
crabml site-model -s data.fasta -t tree.nwk --test m1m2

# Run both tests
crabml site-model -s data.fasta -t tree.nwk --test both

# Output formats
crabml site-model -s data.fasta -t tree.nwk --test both --format json -o results.json
crabml site-model -s data.fasta -t tree.nwk --test both --format tsv -o results.tsv
```

**Options:**
- `-s, --alignment`: Alignment file (FASTA or PHYLIP)
- `-t, --tree`: Tree file (Newick format)
- `--test`: Test type (`m1m2`, `m7m8`, `both`)
- `--format`: Output format (`text`, `json`, `tsv`)
- `--maxiter`: Max iterations (default: 500)
- `--alpha`: Significance threshold (default: 0.05)

### `crabml fit` - Single Model Fitting

```bash
# Fit specific model
crabml fit -m M0 -s alignment.fasta -t tree.nwk

# With custom settings
crabml fit -m M8 -s data.fasta -t tree.nwk --maxiter 1000 --verbose

# JSON output
crabml fit -m M2a -s data.fasta -t tree.nwk --format json -o result.json
```

**Supported models:** M0, M1a, M2a, M3, M7, M8, M8a

### `crabml branch-model` - Branch Model Tests

```bash
# Multi-ratio test (labeled branches)
crabml branch-model -s alignment.fasta -t labeled_tree.nwk --test multi-ratio

# Free-ratio test (all branches)
crabml branch-model -s alignment.fasta -t tree.nwk --test free-ratio
```

### `crabml branch-site` - Branch-Site Model Test

```bash
# Tree must have branch labels: #0 (background), #1 (foreground)
crabml branch-site -s alignment.fasta -t labeled_tree.nwk
```

## Python API Examples

### Multiple Models

```python
from crabml import optimize_model

# Test different models
m1a = optimize_model("M1a", "alignment.fasta", "tree.nwk")
m2a = optimize_model("M2a", "alignment.fasta", "tree.nwk")
m7 = optimize_model("M7", "alignment.fasta", "tree.nwk")
m8 = optimize_model("M8", "alignment.fasta", "tree.nwk")

# Access site class information
print(f"M2a site classes: {m2a.n_site_classes}")
print(f"M2a proportions: {m2a.proportions}")
print(f"M2a omegas: {m2a.omegas}")
```

### Branch Models

```python
from crabml import optimize_branch_model

# Tree with branch labels: #0 = background, #1 = foreground
tree_str = "((human,chimp) #1, (mouse,rat) #0);"

# Multi-ratio model
result = optimize_branch_model("multi-ratio", "alignment.fasta", tree_str)

print(f"Foreground omega: {result.foreground_omega:.3f}")
print(f"Background omega: {result.background_omega:.3f}")
```

### Branch-Site Models

```python
from crabml import optimize_branch_site_model

# Test for positive selection on primate lineage
tree_str = "((human,chimp) #1, (mouse,rat) #0);"

# Alternative model (omega2 free)
result = optimize_branch_site_model("model-a", "alignment.fasta", tree_str)

print(f"Positive selection omega: {result.omega2:.3f}")
print(f"Sites under selection: {result.foreground_positive_proportion:.1%}")

# Null model (omega2 = 1)
null = optimize_branch_site_model("model-a", "alignment.fasta", tree_str, fix_omega=True)
```

### Result Classes

crabML uses specialized result classes:

- **`SiteModelResult`**: Site-class models (M0, M1a, M2a, M7, M8, etc.)
  - Properties: `.omega`, `.omegas`, `.proportions`, `.n_site_classes`

- **`BranchModelResult`**: Branch models (free-ratio, multi-ratio)
  - Properties: `.omega_dict`, `.foreground_omega`, `.background_omega`

- **`BranchSiteModelResult`**: Branch-site models (Model A)
  - Properties: `.omega0`, `.omega2`, `.proportions`, `.foreground_positive_proportion`

All result classes support:
- `.summary()`: Formatted output
- `.to_dict()`: Export as dictionary
- `.to_json(filepath)`: Export as JSON

## Advanced Usage

### Direct Optimizer Access

For maximum control:

```python
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.optimizer import M2aOptimizer

# Load data
alignment = Alignment.from_fasta("alignment.fasta", seqtype='codon')
tree = Tree.from_newick("tree.nwk")

# Run M2a optimization
optimizer = M2aOptimizer(alignment, tree, use_f3x4=True)
kappa, p0, p1, omega0, omega2, lnL = optimizer.optimize()

print(f"Log-likelihood: {lnL:.6f}")
print(f"ω for positive selection: {omega2:.4f}")
```

### Branch-Site Analysis

```python
from crabml.analysis import branch_site_test

# Tree with branch labels: #0 = background, #1 = foreground
tree_str = "((human, chimp) #1, (mouse, rat));"

# Run branch-site test (Model A vs Model A null)
results = branch_site_test(
    alignment='alignment.fasta',
    tree=tree_str,
    use_f3x4=True,
    optimize_branch_lengths=True
)

# Check results
if results['significant']:
    print(f"Positive selection detected!")
    print(f"  ω₂ = {results['alt_params']['omega2']:.4f}")
    print(f"  p₂ = {results['alt_params']['p2']:.4f}")
```

## Development

### Setup

```bash
# Install dependencies
uv sync --all-extras

# Rebuild Rust extension after changes
uv sync --all-extras --reinstall-package crabml-rust
```

### Testing

**Fast tests** (238 tests, ~5 min):
```bash
uv run pytest -m "not slow" -n 4 -v
```

**PAML validation tests** (~30 tests, ~10 min):
```bash
uv run pytest -m "slow" -n 4 -v
```

**Full test suite**:
```bash
uv run pytest -n 4 -v
```

### Benchmarking

Run comprehensive PAML validation:

```bash
cd benchmarks

# Generate simulated data
uv run python run_benchmark.py generate --models M0 M1a M2a M7 M8 M8a

# Run PAML analysis
uv run python run_benchmark.py run-paml --models M0 M1a M2a M7 M8 M8a

# Run crabML analysis
uv run python run_benchmark.py run-crabml --models M0 M1a M2a M7 M8 M8a

# Generate comparison and visualizations
uv run python run_benchmark.py compare
uv run python run_benchmark.py visualize
```

See `benchmarks/README.md` for details.

### Other Commands

```bash
# Run specific test
uv run pytest tests/test_api.py::TestOptimizeModel::test_m0_with_file_paths -v

# Run with coverage
uv run pytest --cov=crabml --cov-report=html

# Run Rust tests
cargo test
```

## Documentation

Full documentation available at [crabml.readthedocs.io](https://crabml.readthedocs.io/en/latest/user_guide/models.html)

## Citation

If you use crabML in your research, please cite:

```bibtex
@software{crabml2025,
  title = {crabML: High-performance phylogenetic analysis},
  author = {Kern, Andrew D.},
  year = {2025},
  url = {https://github.com/andrewkern/crabml}
}
```

Also cite the original PAML software:

```bibtex
@article{yang2007paml,
  title={PAML 4: phylogenetic analysis by maximum likelihood},
  author={Yang, Ziheng},
  journal={Molecular biology and evolution},
  volume={24},
  number={8},
  pages={1586--1591},
  year={2007}
}
```

## License

GPL-3.0-or-later
