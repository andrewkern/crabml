# py-codeml

Python reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis.

## Status

🚧 **Under Development** - Phase 1 implementation in progress

## Installation

```bash
uv pip install -e ".[dev]"
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=pycodeml --cov-report=html

# Run specific test
pytest tests/test_reference/test_matrix.py -v
```

## License

GPL-3.0-or-later
