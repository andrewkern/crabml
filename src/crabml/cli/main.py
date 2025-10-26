"""Main CLI application for crabML."""

import typer
from pathlib import Path
from typing import Optional
from enum import Enum

app = typer.Typer(
    name="crabml",
    help="Fast phylogenetic analysis for detecting positive selection",
    no_args_is_help=True,
)


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


@app.command()
def test(
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
    Test for positive selection using site models.

    Runs standard likelihood ratio tests:
    - M1a (Nearly Neutral) vs M2a (Positive Selection)
    - M7 (Beta) vs M8 (Beta + positive selection)

    Example:
        crabml test -s alignment.fasta -t tree.nwk
        crabml test -s alignment.fasta -t tree.nwk --test m7m8 --alpha 0.01
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
