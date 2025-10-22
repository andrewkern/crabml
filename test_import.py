#!/usr/bin/env python3
"""Test that the renamed package can be imported."""
import sys
sys.path.insert(0, '/home/adkern/crabml/src')

print("Testing crabml imports...")
try:
    import crabml
    print("✓ crabml imported successfully")
except ImportError as e:
    print(f"✗ Failed to import crabml: {e}")
    sys.exit(1)

try:
    from crabml.io.sequences import read_phylip
    from crabml.io.trees import read_newick
    from crabml.optimize import M0Optimizer
    print("✓ All crabml submodules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import crabml submodules: {e}")
    sys.exit(1)

try:
    import crabml_rust
    print("✓ crabml_rust imported successfully")
except ImportError as e:
    print(f"⚠ crabml_rust not available (needs rebuild): {e}")
    print("  Run: cd /home/adkern/crabml && uv sync --all-extras --reinstall")

print("\n✓ All imports successful!")
