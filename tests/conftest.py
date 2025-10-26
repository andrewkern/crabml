"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner


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


@pytest.fixture
def cli_runner():
    """CLI test runner for Typer apps."""
    return CliRunner()


@pytest.fixture
def lysozyme_tree_file(tmp_path):
    """Create a temporary tree file for lysozyme dataset."""
    tree_content = "((Hsa_Human:0.03,Hla_gibbon:0.04):0.07,((Cgu/Can_colobus:0.04,Pne_langur:0.05):0.08,Mmu_rhesus:0.02):0.04,(Ssc_squirrelM:0.04,Cja_marmoset:0.02):0.13);\n"
    tree_file = tmp_path / "test_tree.nwk"
    tree_file.write_text(tree_content)
    return tree_file
