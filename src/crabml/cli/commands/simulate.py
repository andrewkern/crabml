"""Simulate command for crabML CLI."""

import typer
from pathlib import Path
from typing import Optional
import numpy as np
import json

from ...simulate.codon import M0CodonSimulator
from ...simulate.output import SimulationOutput
from ...io.trees import Tree

app = typer.Typer(help="Simulate sequences under evolutionary models")


@app.command(name="m0")
def simulate_m0(
    tree: Path = typer.Option(
        ...,
        "--tree", "-t",
        help="Tree file (Newick format with branch lengths)",
        exists=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output FASTA file",
    ),
    length: int = typer.Option(
        ...,
        "--length", "-l",
        help="Sequence length (number of codons)",
    ),
    omega: float = typer.Option(
        ...,
        "--omega",
        help="dN/dS ratio",
    ),
    kappa: float = typer.Option(
        2.0,
        "--kappa",
        help="Transition/transversion ratio",
    ),
    replicates: int = typer.Option(
        1,
        "--replicates", "-r",
        help="Number of replicates to simulate",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility",
    ),
    codon_freqs_file: Optional[Path] = typer.Option(
        None,
        "--codon-freqs",
        help="Codon frequencies JSON file (default: uniform)",
        exists=True,
    ),
    output_params: bool = typer.Option(
        True,
        "--output-params/--no-output-params",
        help="Write parameters to JSON file",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress messages",
    ),
):
    """
    Simulate codon sequences under M0 model (single omega).

    The M0 model has a single dN/dS ratio for all sites and branches.
    This is useful for testing parameter recovery and generating
    null datasets.

    Examples:

        \b
        # Simulate 1000 codons with omega=0.3
        crabml simulate m0 -t tree.nwk -o sim.fasta -l 1000 --omega 0.3

        \b
        # Simulate 10 replicates with custom kappa
        crabml simulate m0 -t tree.nwk -o sim.fasta -l 500 --omega 0.5 --kappa 2.5 -r 10

        \b
        # Use reproducible seed
        crabml simulate m0 -t tree.nwk -o sim.fasta -l 1000 --omega 0.3 --seed 42
    """
    if not quiet:
        typer.echo("crabML Sequence Simulator - M0 Model")
        typer.echo("=" * 50)

    # Load tree
    if not quiet:
        typer.echo(f"Loading tree from {tree}...")
    with open(tree) as f:
        tree_str = f.read()
    tree_obj = Tree.from_newick(tree_str)

    # Validate tree
    if tree_obj.root is None:
        typer.echo("Error: Tree must be rooted", err=True)
        raise typer.Exit(code=1)

    # Load or generate codon frequencies
    if codon_freqs_file:
        if not quiet:
            typer.echo(f"Loading codon frequencies from {codon_freqs_file}...")
        with open(codon_freqs_file) as f:
            codon_freqs = np.array(json.load(f))
        if len(codon_freqs) != 61:
            typer.echo(f"Error: Expected 61 codon frequencies, got {len(codon_freqs)}", err=True)
            raise typer.Exit(code=1)
    else:
        if not quiet:
            typer.echo("Using uniform codon frequencies")
        codon_freqs = np.ones(61) / 61

    # Setup simulator
    if not quiet:
        typer.echo(f"\nSimulation parameters:")
        typer.echo(f"  Model: M0")
        typer.echo(f"  Sequence length: {length} codons")
        typer.echo(f"  Omega (dN/dS): {omega:.4f}")
        typer.echo(f"  Kappa (ts/tv): {kappa:.4f}")
        typer.echo(f"  Replicates: {replicates}")
        if seed is not None:
            typer.echo(f"  Seed: {seed}")

    try:
        simulator = M0CodonSimulator(
            tree=tree_obj,
            sequence_length=length,
            kappa=kappa,
            omega=omega,
            codon_freqs=codon_freqs,
            seed=seed
        )
    except Exception as e:
        typer.echo(f"Error creating simulator: {e}", err=True)
        raise typer.Exit(code=1)

    # Run simulation
    if not quiet:
        typer.echo(f"\nSimulating {replicates} replicate(s)...")

    for rep in range(replicates):
        # Simulate sequences
        try:
            sequences = simulator.simulate()
        except Exception as e:
            typer.echo(f"Error simulating replicate {rep+1}: {e}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if replicates == 1:
            out_path = output
        else:
            # Add replicate number to filename
            out_path = output.parent / f"{output.stem}_rep{rep+1}{output.suffix}"

        # Write sequences
        try:
            SimulationOutput.write_fasta(sequences, out_path, replicate_id=rep+1 if replicates > 1 else None)
        except Exception as e:
            typer.echo(f"Error writing output: {e}", err=True)
            raise typer.Exit(code=1)

        if not quiet:
            typer.echo(f"  Replicate {rep+1} -> {out_path}")

    # Write parameters
    if output_params:
        params_path = output.parent / f"{output.stem}.params.json"
        try:
            params = simulator.get_parameters()
            params['seed'] = seed
            params['replicates'] = replicates
            SimulationOutput.write_parameters(params, params_path)
            if not quiet:
                typer.echo(f"\nParameters -> {params_path}")
        except Exception as e:
            typer.echo(f"Warning: Could not write parameters: {e}", err=True)

    if not quiet:
        typer.echo("\nSimulation complete!")


if __name__ == "__main__":
    app()
