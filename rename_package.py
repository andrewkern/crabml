#!/usr/bin/env python3
"""Script to rename pycodeml to crabml throughout the codebase."""

import os
import shutil
from pathlib import Path

# Base directory
base_dir = Path("/home/adkern/crabml")

# 1. Rename src directory
src_old = base_dir / "src" / "pycodeml"
src_new = base_dir / "src" / "crabml"
if src_old.exists() and not src_new.exists():
    print(f"Renaming {src_old} -> {src_new}")
    shutil.move(str(src_old), str(src_new))
    print("✓ Renamed source directory")
else:
    print(f"Source directory already renamed or doesn't exist")

# 2. Update all Python imports
print("\nUpdating Python imports...")
for root, dirs, files in os.walk(base_dir):
    # Skip hidden directories and build artifacts
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'target', 'build', 'dist']]

    for file in files:
        if file.endswith('.py'):
            file_path = Path(root) / file
            try:
                content = file_path.read_text()
                original = content

                # Replace imports
                content = content.replace('from crabml.', 'from crabml.')
                content = content.replace('import crabml.', 'import crabml.')
                content = content.replace('from ..crabml', 'from ..crabml')

                if content != original:
                    file_path.write_text(content)
                    print(f"  Updated: {file_path.relative_to(base_dir)}")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

print("\n✓ Python imports updated")
print("\nDone! Manual steps remaining:")
print("- Update Rust package name in rust/Cargo.toml")
print("- Update Rust code references")
print("- Update README.md")
