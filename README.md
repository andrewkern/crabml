# py-codeml

Python reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis.

## Status

Production-ready with 10 PAML codon models fully validated against reference outputs.

### Features

- **Codon substitution models**: M0, M1a, M2a, M3, M4, M5, M6, M7, M8, M9
- **Parameter optimization**: Complete MLE optimization for all 10 models
- **High-performance Rust backend**: 3-10x faster than PAML for full optimization
- **PAML validation**: All models produce exact numerical matches to PAML
- **Test coverage**: 12 validation tests passing, exact agreement with reference outputs

## Installation

### Python-only (NumPy/SciPy)

```bash
uv pip install -e ".[dev]"
```

### With Rust backend (recommended)

The Rust backend provides 3-10x speedup for full parameter optimization.

**Requirements:**
- Rust toolchain (install from https://rustup.rs)
- OpenBLAS or similar BLAS/LAPACK implementation

```bash
# Install Python package
uv pip install -e ".[dev]"

# Build Rust extension
cd rust
uv pip install maturin
uv run maturin develop --release
cd ..
```

**Verify Rust installation:**
```bash
python -c "import pycodeml_rust; print('Rust backend available')"
```

## Usage

```python
from pycodeml.io.sequences import read_phylip
from pycodeml.io.trees import read_newick
from pycodeml.optimize import M0Optimizer

# Load data
alignment = read_phylip("alignment.phy")
tree = read_newick("tree.nwk")

# Run M0 model with Rust backend (3-10x faster than PAML)
optimizer = M0Optimizer(alignment, tree, use_rust=True)
result = optimizer.optimize()

print(f"Log-likelihood: {result['log_likelihood']:.6f}")
print(f"Parameters: kappa={result['kappa']:.4f}, omega={result['omega']:.4f}")
```

Set `use_rust=False` to use pure Python implementation.

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
# Run tests (Python only)
uv run pytest

# Run tests with coverage
uv run pytest --cov=pycodeml --cov-report=html

# Run specific test
uv run pytest tests/test_reference/test_matrix.py -v

# Run Rust tests
cd rust
cargo test
cd ..

# Benchmark Rust vs Python
uv run pytest tests/test_rust/ -v
```

## License

GPL-3.0-or-later
