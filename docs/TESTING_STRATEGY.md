# Testing Strategy for crabML

This document describes the two-tier testing strategy for crabML, designed to balance comprehensive validation with fast CI/CD feedback loops.

## Overview

- **Tier 1: Fast CI Tests** - Quick unit tests and basic correctness checks (~203 tests, ~5-8 min)
- **Tier 2: Slow/PAML Validation Tests** - Comprehensive validation against PAML reference outputs (~30 tests, marked as `@pytest.mark.slow`)

## Tier 1: Fast CI Tests (Default)

**Purpose**: Rapid feedback during development, suitable for CI/CD pipelines

**Selection**: All tests EXCEPT those marked with `@pytest.mark.slow`

**Characteristics**:
- Fast execution (~5-10 minutes with 2 workers)
- Core unit tests
- Basic integration tests
- Quick correctness checks
- No expensive optimization tests

**Run Command**:
```bash
# CI/CD (2 workers, matching GitHub Actions)
uv run pytest -m "not slow" -n 2 -v

# Local development (more workers)
uv run pytest -m "not slow" -n 4 -v
```

**Test Count**: ~203 tests

**Coverage**:
- ✅ Core likelihood calculations
- ✅ Model parameterization
- ✅ Tree parsing and manipulation
- ✅ Sequence I/O
- ✅ Matrix operations (Q matrix, eigendecomposition)
- ✅ Basic optimization (M0, simple models)
- ✅ API functionality
- ✅ CLI interface
- ✅ Rust backend integration (where applicable)

## Tier 2: Slow/PAML Validation Tests

**Purpose**: Comprehensive correctness validation against PAML reference outputs

**Selection**: Tests marked with `@pytest.mark.slow`

**Characteristics**:
- Longer execution time (additional 10-20 minutes)
- Full optimization runs (500-1000 iterations)
- PAML reference comparisons
- Branch model tests (computationally expensive)
- Branch-site model tests
- Free-ratio models (many parameters)

**Run Command**:
```bash
# Run ONLY slow tests
uv run pytest -m "slow" -n 4 -v

# Run ALL tests (fast + slow)
uv run pytest -n 4 -v
```

**Test Count**: ~30 tests

**Slow Tests Include**:

### Branch Models (`tests/test_branch_model_analysis.py`)
- `test_branch_model_test_two_ratio` - Two-ratio vs M0 hypothesis test
- `test_branch_model_test_three_ratio` - Three-ratio vs M0 hypothesis test
- `test_free_ratio_model_test` - Free-ratio model (11 omega parameters)

### Branch Model Validation (`tests/test_branch_models.py`)
- `test_branch_model_two_ratio` - Two-ratio model optimization
- `test_branch_model_vs_m0` - Branch model without labels vs M0

### PAML Reference Validation (`tests/test_branch_paml_validation.py`)
- `test_branch_model_two_ratio_vs_paml` - Compare against PAML reference (lnL, kappa, omega)
- `test_branch_model_parameter_interpretation` - Biological parameter validation

### Branch-Site Models (`tests/test_branch_site_likelihood.py`)
- `test_branch_site_model_a_likelihood` - Branch-site model A likelihood vs PAML

### Site-Class Models (various files)
- Full M7 vs M8 tests
- Full M1a vs M2a tests
- Other expensive site-class model tests

## GitHub Actions CI Configuration

**Workflow**: `.github/workflows/ci.yml`

**Trigger**:
- Push to `main` branch
- Pull requests to `main`
- Manual dispatch

**Configuration**:
- **OS**: Ubuntu latest
- **Python**: 3.11
- **Workers**: 2 (matches GitHub's 2-core runners)
- **Tests**: `-m "not slow"` (Tier 1 only)
- **Rust**: Built in release mode for performance

**Expected CI Time**: ~12-15 minutes total
- Build Rust extension: ~3 min
- Install dependencies: ~1 min
- Run fast tests: ~8-10 min

## Local Development Workflow

### Quick Development Cycle
```bash
# Run just the tests for the module you're working on
uv run pytest tests/test_models/ -v

# Run fast tests with more workers
uv run pytest -m "not slow" -n 8 -v
```

### Pre-Commit Validation
```bash
# Run all fast tests
uv run pytest -m "not slow" -n 4 -v

# This should complete in ~5-8 minutes
```

### Before Creating PR
```bash
# Run FULL test suite (fast + slow)
uv run pytest -n 4 -v

# This ensures all PAML validation tests pass
# Expected time: ~15-20 minutes total
```

### PAML Reference Validation
```bash
# Run only slow PAML validation tests
uv run pytest -m "slow" -n 4 -v

# Useful after changing optimization or likelihood code
```

## Adding New Tests

### When to Mark as `@pytest.mark.slow`:

1. **Expensive Optimization**: Tests that run full optimization (>500 iterations)
2. **Complex Models**: Branch models, free-ratio models, branch-site models
3. **PAML Validation**: Tests that compare against PAML reference outputs
4. **Long Runtime**: Any test taking >10 seconds to execute

### Example:
```python
import pytest

@pytest.mark.slow
def test_expensive_optimization():
    """Test full optimization against PAML reference."""
    result = optimize_model("M8", alignment, tree, maxiter=1000)
    assert_paml_match(result)
```

## Test Execution Times

Based on empirical measurements with 2 workers:

| Test Suite | Count | Time | Notes |
|------------|-------|------|-------|
| Fast tests | 203 | ~8-10 min | Tier 1, runs in CI |
| Slow tests | 30 | ~10-15 min | Tier 2, local only |
| **Total** | **233** | **~18-25 min** | Full validation |

## Marking Tests

Tests are marked using pytest markers defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

## Future Enhancements

### Potential Third Tier: Nightly Tests
For very expensive tests that don't need to run on every commit:
- Extensive parameter sweeps
- Large dataset tests
- Performance benchmarking
- Multiple dataset validation

**Implementation**:
```python
@pytest.mark.nightly
def test_extensive_validation():
    # Very expensive test
    pass
```

**GitHub Actions**:
```yaml
# .github/workflows/nightly.yml
name: Nightly Tests
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - # ... setup
      - run: uv run pytest -m "slow or nightly" -n 2 -v
```

## Summary

- **CI/CD (GitHub Actions)**: Fast tests only (`-m "not slow"`) → ~12-15 min
- **Local Pre-Commit**: Fast tests (`-m "not slow"`) → ~5-8 min
- **Before PR**: All tests (no marker filter) → ~18-25 min
- **PAML Validation**: Slow tests only (`-m "slow"`) → ~10-15 min

This strategy ensures fast feedback loops during development while maintaining comprehensive validation against PAML reference outputs.
