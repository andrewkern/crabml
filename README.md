# py-codeml

Python reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis.

## Status

✅ **Core functionality complete** - M0, M1a, M2a, M3 models implemented with optional Rust acceleration

### Features

- **Codon substitution models**: M0, M1a (NearlyNeutral), M2a (PositiveSelection), M3 (Discrete)
- **High-performance Rust backend**: 15-30x faster than pure Python (optional)
- **PAML compatibility**: Validated against PAML reference outputs
- **Test coverage**: 67 tests passing, 79% code coverage

## Installation

### Python-only (NumPy/SciPy)

```bash
uv pip install -e ".[dev]"
```

### With Rust backend (recommended for performance)

The Rust backend provides 15-30x speedup for likelihood calculations.

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

# Run M0 model with Rust backend (15-25x faster)
optimizer = M0Optimizer(alignment, tree, use_rust=True)
result = optimizer.optimize()

print(f"Log-likelihood: {result['log_likelihood']:.6f}")
print(f"kappa: {result['kappa']:.4f}")
print(f"omega: {result['omega']:.4f}")
```

Set `use_rust=False` to use pure Python implementation.

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
