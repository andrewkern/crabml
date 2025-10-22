#!/bin/bash
cd /home/adkern/crabml
echo "Syncing dependencies and rebuilding..."
uv sync --all-extras --reinstall 2>&1 | tail -50
echo ""
echo "Testing import..."
uv run python -c "import crabml_rust; print('✓ Rust backend ready')" 2>&1
echo ""
echo "Running quick validation test..."
uv run pytest tests/test_rust/test_paml_rust_validation.py::TestRustPAMLValidation::test_rust_m0_likelihood_matches_paml -v 2>&1 | tail -20
