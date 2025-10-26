"""Main CLI application for crabML."""

import typer
from pathlib import Path
from typing import Optional
from enum import Enum

from .commands import simulate as simulate_cmd

app = typer.Typer(
    name="crabml",
    help="Fast phylogenetic analysis for detecting positive selection",
    no_args_is_help=True,
)

# Add simulate subcommand
app.add_typer(simulate_cmd.app, name="simulate")


class TestType(str, Enum):
    """Type of positive selection test to run."""
    M1M2 = "m1m2"
    M7M8 = "m7m8"
    BOTH = "both"
    ALL = "all"


class OutputFormat(str, Enum):
    """Output format."""
    TEXT = "text"
    JSON = "json"
    TSV = "tsv"


class BranchModelTest(str, Enum):
    """Type of branch model test to run."""
    MULTI_RATIO = "multi-ratio"
    FREE_RATIO = "free-ratio"


@app.command(name="site-model")
def site_model(
    alignment: Path = typer.Option(
        ...,
        "--alignment", "-s",
        help="Codon alignment file (FASTA or PHYLIP)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    tree: Path = typer.Option(
        ...,
        "--tree", "-t",
        help="Phylogenetic tree file (Newick format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    test_type: TestType = typer.Option(
        TestType.BOTH,
        "--test",
        help="Which test(s) to run",
    ),
    alpha: float = typer.Option(
        0.05,
        "--alpha",
        help="Significance level for hypothesis tests",
        min=0.0,
        max=1.0,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file (default: stdout)",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        help="Output format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show optimization progress",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Minimal output",
    ),
    maxiter: int = typer.Option(
        1000,
        "--maxiter",
        help="Maximum optimization iterations",
        min=100,
    ),
    no_m0_init: bool = typer.Option(
        False,
        "--no-m0-init",
        help="Skip M0 initialization (faster but less robust)",
    ),
):
    """
    Test for positive selection using site-class models.

    Runs standard likelihood ratio tests:
    - M1a (Nearly Neutral) vs M2a (Positive Selection)
    - M7 (Beta) vs M8 (Beta + positive selection)

    Example:
        crabml site-model -s alignment.fasta -t tree.nwk
        crabml site-model -s alignment.fasta -t tree.nwk --test m7m8 --alpha 0.01
    """
    from .commands.test import run_test

    run_test(
        alignment=alignment,
        tree=tree,
        test_type=test_type.value,
        alpha=alpha,
        output=output,
        format=format.value,
        verbose=verbose,
        quiet=quiet,
        maxiter=maxiter,
        init_with_m0=not no_m0_init,
    )


@app.command()
def fit(
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model to fit (M0, M1a, M2a, M7, M8, etc.)",
    ),
    alignment: Path = typer.Option(
        ...,
        "--alignment", "-s",
        help="Codon alignment file (FASTA or PHYLIP)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    tree: Path = typer.Option(
        ...,
        "--tree", "-t",
        help="Phylogenetic tree file (Newick format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file (default: stdout)",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        help="Output format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show optimization progress",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Minimal output",
    ),
    maxiter: int = typer.Option(
        1000,
        "--maxiter",
        help="Maximum optimization iterations",
        min=100,
    ),
    no_m0_init: bool = typer.Option(
        False,
        "--no-m0-init",
        help="Skip M0 initialization (faster but less robust)",
    ),
):
    """
    Fit a single codon substitution model.

    Example:
        crabml fit -m M8 -s alignment.fasta -t tree.nwk
        crabml fit -m M0 -s alignment.fasta -t tree.nwk --format json
    """
    from .commands.fit import run_fit

    run_fit(
        model=model,
        alignment=alignment,
        tree=tree,
        output=output,
        format=format.value,
        verbose=verbose,
        quiet=quiet,
        maxiter=maxiter,
        init_with_m0=not no_m0_init,
    )


@app.command(name="branch-model")
def branch_model(
    alignment: Path = typer.Option(
        ...,
        "--alignment", "-s",
        help="Codon alignment file (FASTA or PHYLIP)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    tree: Path = typer.Option(
        ...,
        "--tree", "-t",
        help="Phylogenetic tree file (Newick format, with branch labels for multi-ratio)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    test_type: BranchModelTest = typer.Option(
        BranchModelTest.MULTI_RATIO,
        "--test",
        help="Which branch model test to run",
    ),
    alpha: float = typer.Option(
        0.05,
        "--alpha",
        help="Significance threshold for hypothesis test",
        min=0.001,
        max=0.5,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file (default: stdout)",
        file_okay=True,
        dir_okay=False,
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        help="Output format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show optimization progress",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Minimal output",
    ),
    maxiter: int = typer.Option(
        1000,
        "--maxiter",
        help="Maximum optimization iterations",
        min=100,
    ),
):
    """
    Test for lineage-specific selection using branch models.

    Runs likelihood ratio tests:
    - multi-ratio vs M0: Different omega for labeled branch groups
    - free-ratio vs M0: Independent omega for each branch (exploratory)

    For multi-ratio, tree must have branch labels (#0, #1, etc.).

    Example:
        crabml branch-model -s alignment.fasta -t labeled_tree.nwk --test multi-ratio
        crabml branch-model -s alignment.fasta -t tree.nwk --test free-ratio
    """
    from .commands.branch_model import run_branch_model

    run_branch_model(
        alignment=alignment,
        tree=tree,
        test_type=test_type.value,
        alpha=alpha,
        output=output,
        format=format.value,
        verbose=verbose,
        quiet=quiet,
        maxiter=maxiter,
    )


@app.command(name="branch-site")
def branch_site(
    alignment: Path = typer.Option(
        ...,
        "--alignment", "-s",
        help="Codon alignment file (FASTA or PHYLIP)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    tree: Path = typer.Option(
        ...,
        "--tree", "-t",
        help="Phylogenetic tree file with branch labels (Newick format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    alpha: float = typer.Option(
        0.05,
        "--alpha",
        help="Significance level for hypothesis test",
        min=0.0,
        max=1.0,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file (default: stdout)",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TEXT,
        "--format",
        help="Output format",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show optimization progress",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Minimal output",
    ),
    maxiter: int = typer.Option(
        1000,
        "--maxiter",
        help="Maximum optimization iterations",
        min=100,
    ),
):
    """
    Test for positive selection on specific lineages using branch-site model A.

    The tree must have branch labels (#0 for background, #1 for foreground).
    Tests if specific sites on foreground branches have ω > 1.

    Example:
        crabml branch-site -s alignment.fasta -t labeled_tree.nwk
    """
    from .commands.branch_site import run_branch_site

    run_branch_site(
        alignment=alignment,
        tree=tree,
        alpha=alpha,
        output=output,
        format=format.value,
        verbose=verbose,
        quiet=quiet,
        maxiter=maxiter,
    )


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
