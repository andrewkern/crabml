"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path


@pytest.fixture
def paml_examples_dir():
    """Path to PAML example datasets."""
    return Path(__file__).parent / "data" / "paml_examples"


@pytest.fixture
def paml_reference_dir():
    """Path to PAML reference outputs."""
    return Path(__file__).parent / "reference"


@pytest.fixture
def lysozyme_small_files(paml_examples_dir):
    """Lysozyme small dataset files."""
    base = paml_examples_dir / "lysozyme"
    return {
        "sequences": base / "lysozymeSmall.txt",
        "tree": base / "lysozymeSmall.trees",
        "control": base / "codeml.ctl",
    }
