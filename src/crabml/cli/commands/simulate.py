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


@app.command(name="m2a")
def simulate_m2a(
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
    kappa: float = typer.Option(
        2.0,
        "--kappa",
        help="Transition/transversion ratio",
    ),
    p0: float = typer.Option(
        ...,
        "--p0",
        help="Proportion in purifying class (omega < 1)",
    ),
    p1: float = typer.Option(
        ...,
        "--p1",
        help="Proportion in neutral class (omega = 1)",
    ),
    omega0: float = typer.Option(
        ...,
        "--omega0",
        help="dN/dS for purifying class (must be < 1)",
    ),
    omega2: float = typer.Option(
        ...,
        "--omega2",
        help="dN/dS for positive selection class (must be > 1)",
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
    output_site_classes: bool = typer.Option(
        True,
        "--output-site-classes/--no-output-site-classes",
        help="Write site class assignments to file",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress messages",
    ),
):
    """
    Simulate under M2a model (positive selection).

    The M2a model has three site classes:
    - Class 0: omega_0 < 1 (purifying selection)
    - Class 1: omega = 1 (neutral)
    - Class 2: omega_2 > 1 (positive selection)

    Examples:

        \b
        # Simulate with strong positive selection
        crabml simulate m2a -t tree.nwk -o sim.fasta -l 1000 \\
            --p0 0.5 --p1 0.3 --omega0 0.1 --omega2 2.5

        \b
        # Multiple replicates
        crabml simulate m2a -t tree.nwk -o sim.fasta -l 500 \\
            --p0 0.6 --p1 0.2 --omega0 0.05 --omega2 3.0 -r 10
    """
    from ...simulate.codon import M2aSimulator
    from ...simulate.output import SimulationOutput

    if not quiet:
        typer.echo("crabML Sequence Simulator - M2a Model")
        typer.echo("=" * 50)

    # Load tree
    if not quiet:
        typer.echo(f"Loading tree from {tree}...")
    with open(tree) as f:
        tree_str = f.read()
    tree_obj = Tree.from_newick(tree_str)

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
        p2 = 1 - p0 - p1
        typer.echo(f"\nSimulation parameters:")
        typer.echo(f"  Model: M2a")
        typer.echo(f"  Sequence length: {length} codons")
        typer.echo(f"  Kappa (ts/tv): {kappa:.4f}")
        typer.echo(f"  Site classes:")
        typer.echo(f"    Class 0 (purifying): p={p0:.3f}, omega={omega0:.3f}")
        typer.echo(f"    Class 1 (neutral):   p={p1:.3f}, omega=1.000")
        typer.echo(f"    Class 2 (positive):  p={p2:.3f}, omega={omega2:.3f}")
        typer.echo(f"  Replicates: {replicates}")
        if seed is not None:
            typer.echo(f"  Seed: {seed}")

    try:
        simulator = M2aSimulator(
            tree=tree_obj,
            sequence_length=length,
            kappa=kappa,
            p0=p0,
            p1=p1,
            omega0=omega0,
            omega2=omega2,
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
        # Simulate
        try:
            sequences = simulator.simulate()
        except Exception as e:
            typer.echo(f"Error simulating replicate {rep+1}: {e}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if replicates == 1:
            out_path = output
        else:
            out_path = output.parent / f"{output.stem}_rep{rep+1}{output.suffix}"

        # Write sequences
        try:
            SimulationOutput.write_fasta(sequences, out_path, replicate_id=rep+1 if replicates > 1 else None)
        except Exception as e:
            typer.echo(f"Error writing output: {e}", err=True)
            raise typer.Exit(code=1)

        if not quiet:
            typer.echo(f"  Replicate {rep+1} -> {out_path}")

        # Write site classes
        if output_site_classes:
            site_classes_path = out_path.parent / f"{out_path.stem}.site_classes.txt"
            ps_sites_path = out_path.parent / f"{out_path.stem}.positive_sites.txt"

            try:
                site_info = simulator.get_site_classes()
                SimulationOutput.write_site_classes(
                    np.array(site_info['site_class_ids']),
                    np.array(site_info['site_class_omegas']),
                    site_classes_path
                )
                SimulationOutput.write_positively_selected_sites(
                    np.array(site_info['site_class_ids']),
                    np.array(site_info['site_class_omegas']),
                    ps_sites_path
                )
            except Exception as e:
                typer.echo(f"Warning: Could not write site classes: {e}", err=True)

    # Write parameters
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


@app.command(name="m1a")
def simulate_m1a(
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
    kappa: float = typer.Option(
        2.0,
        "--kappa",
        help="Transition/transversion ratio",
    ),
    p0: float = typer.Option(
        ...,
        "--p0",
        help="Proportion in purifying class (omega < 1)",
    ),
    omega0: float = typer.Option(
        ...,
        "--omega0",
        help="dN/dS for purifying class (must be < 1)",
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
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress messages",
    ),
):
    """
    Simulate under M1a model (nearly neutral).

    The M1a model has two site classes:
    - Class 0: omega_0 < 1 (purifying selection)
    - Class 1: omega = 1 (neutral)

    Examples:

        \\b
        # Simulate with 70% purifying, 30% neutral
        crabml simulate m1a -t tree.nwk -o sim.fasta -l 1000 --p0 0.7 --omega0 0.1

        \\b
        # Multiple replicates
        crabml simulate m1a -t tree.nwk -o sim.fasta -l 500 --p0 0.6 --omega0 0.05 -r 10
    """
    from ...simulate.codon import M1aSimulator
    from ...simulate.output import SimulationOutput

    if not quiet:
        typer.echo("crabML Sequence Simulator - M1a Model")
        typer.echo("=" * 50)

    # Load tree
    if not quiet:
        typer.echo(f"Loading tree from {tree}...")
    with open(tree) as f:
        tree_str = f.read()
    tree_obj = Tree.from_newick(tree_str)

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
        p1 = 1 - p0
        typer.echo(f"\nSimulation parameters:")
        typer.echo(f"  Model: M1a")
        typer.echo(f"  Sequence length: {length} codons")
        typer.echo(f"  Kappa (ts/tv): {kappa:.4f}")
        typer.echo(f"  Site classes:")
        typer.echo(f"    Class 0 (purifying): p={p0:.3f}, omega={omega0:.3f}")
        typer.echo(f"    Class 1 (neutral):   p={p1:.3f}, omega=1.000")
        typer.echo(f"  Replicates: {replicates}")
        if seed is not None:
            typer.echo(f"  Seed: {seed}")

    try:
        simulator = M1aSimulator(
            tree=tree_obj,
            sequence_length=length,
            kappa=kappa,
            p0=p0,
            omega0=omega0,
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
        # Simulate
        try:
            sequences = simulator.simulate()
        except Exception as e:
            typer.echo(f"Error simulating replicate {rep+1}: {e}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if replicates == 1:
            out_path = output
        else:
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


@app.command(name="m7")
def simulate_m7(
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
    kappa: float = typer.Option(
        2.0,
        "--kappa",
        help="Transition/transversion ratio",
    ),
    p: float = typer.Option(
        ...,
        "--p",
        help="Beta distribution parameter p (shape)",
    ),
    q: float = typer.Option(
        ...,
        "--q",
        help="Beta distribution parameter q (shape)",
    ),
    ncateg: int = typer.Option(
        10,
        "--ncateg",
        help="Number of discrete categories",
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
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress messages",
    ),
):
    """
    Simulate under M7 model (beta distribution).

    The M7 model uses a beta distribution to model variation in omega
    across sites, with omega constrained to (0,1).

    Examples:

        \\b
        # Simulate with beta(2,5) distribution (mean omega ~0.29)
        crabml simulate m7 -t tree.nwk -o sim.fasta -l 1000 --p 2 --q 5

        \\b
        # Use more categories for finer discretization
        crabml simulate m7 -t tree.nwk -o sim.fasta -l 1000 --p 1 --q 2 --ncateg 20
    """
    from ...simulate.codon import M7Simulator
    from ...simulate.output import SimulationOutput

    if not quiet:
        typer.echo("crabML Sequence Simulator - M7 Model")
        typer.echo("=" * 50)

    # Load tree
    if not quiet:
        typer.echo(f"Loading tree from {tree}...")
    with open(tree) as f:
        tree_str = f.read()
    tree_obj = Tree.from_newick(tree_str)

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
        from scipy.stats import beta as beta_dist
        mean_omega = beta_dist(p, q).mean()
        typer.echo(f"\nSimulation parameters:")
        typer.echo(f"  Model: M7")
        typer.echo(f"  Sequence length: {length} codons")
        typer.echo(f"  Kappa (ts/tv): {kappa:.4f}")
        typer.echo(f"  Beta distribution: p={p:.3f}, q={q:.3f}")
        typer.echo(f"  Mean omega: {mean_omega:.4f}")
        typer.echo(f"  Categories: {ncateg}")
        typer.echo(f"  Replicates: {replicates}")
        if seed is not None:
            typer.echo(f"  Seed: {seed}")

    try:
        simulator = M7Simulator(
            tree=tree_obj,
            sequence_length=length,
            kappa=kappa,
            p=p,
            q=q,
            n_categories=ncateg,
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
        # Simulate
        try:
            sequences = simulator.simulate()
        except Exception as e:
            typer.echo(f"Error simulating replicate {rep+1}: {e}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if replicates == 1:
            out_path = output
        else:
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


@app.command(name="m8")
def simulate_m8(
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
    kappa: float = typer.Option(
        2.0,
        "--kappa",
        help="Transition/transversion ratio",
    ),
    p0: float = typer.Option(
        ...,
        "--p0",
        help="Proportion in beta distribution classes",
    ),
    p: float = typer.Option(
        ...,
        "--p",
        help="Beta distribution parameter p (shape)",
    ),
    q: float = typer.Option(
        ...,
        "--q",
        help="Beta distribution parameter q (shape)",
    ),
    omega_s: float = typer.Option(
        ...,
        "--omega-s",
        help="dN/dS for positive selection class",
    ),
    ncateg: int = typer.Option(
        10,
        "--ncateg",
        help="Number of discrete categories for beta",
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
    output_site_classes: bool = typer.Option(
        True,
        "--output-site-classes/--no-output-site-classes",
        help="Write site class assignments to file",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress messages",
    ),
):
    """
    Simulate under M8 model (beta + positive selection).

    The M8 model combines a beta distribution for omega in (0,1)
    with an additional class for positive selection (omega > 1).

    Examples:

        \\b
        # Simulate with 80% beta-distributed, 20% positive selection
        crabml simulate m8 -t tree.nwk -o sim.fasta -l 1000 \\
            --p0 0.8 --p 2 --q 5 --omega-s 2.5

        \\b
        # Strong positive selection on small fraction of sites
        crabml simulate m8 -t tree.nwk -o sim.fasta -l 1000 \\
            --p0 0.95 --p 1 --q 2 --omega-s 5.0 -r 10
    """
    from ...simulate.codon import M8Simulator
    from ...simulate.output import SimulationOutput

    if not quiet:
        typer.echo("crabML Sequence Simulator - M8 Model")
        typer.echo("=" * 50)

    # Load tree
    if not quiet:
        typer.echo(f"Loading tree from {tree}...")
    with open(tree) as f:
        tree_str = f.read()
    tree_obj = Tree.from_newick(tree_str)

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
        from scipy.stats import beta as beta_dist
        mean_beta_omega = beta_dist(p, q).mean()
        p_s = 1 - p0
        typer.echo(f"\nSimulation parameters:")
        typer.echo(f"  Model: M8")
        typer.echo(f"  Sequence length: {length} codons")
        typer.echo(f"  Kappa (ts/tv): {kappa:.4f}")
        typer.echo(f"  Beta classes ({ncateg}): p0={p0:.3f}, beta({p:.2f},{q:.2f}), mean={mean_beta_omega:.4f}")
        typer.echo(f"  Selection class: p_s={p_s:.3f}, omega={omega_s:.3f}")
        typer.echo(f"  Replicates: {replicates}")
        if seed is not None:
            typer.echo(f"  Seed: {seed}")

    try:
        simulator = M8Simulator(
            tree=tree_obj,
            sequence_length=length,
            kappa=kappa,
            p0=p0,
            p=p,
            q=q,
            omega_s=omega_s,
            n_beta_categories=ncateg,
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
        # Simulate
        try:
            sequences = simulator.simulate()
        except Exception as e:
            typer.echo(f"Error simulating replicate {rep+1}: {e}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if replicates == 1:
            out_path = output
        else:
            out_path = output.parent / f"{output.stem}_rep{rep+1}{output.suffix}"

        # Write sequences
        try:
            SimulationOutput.write_fasta(sequences, out_path, replicate_id=rep+1 if replicates > 1 else None)
        except Exception as e:
            typer.echo(f"Error writing output: {e}", err=True)
            raise typer.Exit(code=1)

        if not quiet:
            typer.echo(f"  Replicate {rep+1} -> {out_path}")

        # Write site classes
        if output_site_classes:
            site_classes_path = out_path.parent / f"{out_path.stem}.site_classes.txt"
            ps_sites_path = out_path.parent / f"{out_path.stem}.positive_sites.txt"

            try:
                site_info = simulator.get_site_classes()
                SimulationOutput.write_site_classes(
                    np.array(site_info['site_class_ids']),
                    np.array(site_info['site_class_omegas']),
                    site_classes_path
                )
                SimulationOutput.write_positively_selected_sites(
                    np.array(site_info['site_class_ids']),
                    np.array(site_info['site_class_omegas']),
                    ps_sites_path
                )
            except Exception as e:
                typer.echo(f"Warning: Could not write site classes: {e}", err=True)

    # Write parameters
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


@app.command(name="m8a")
def simulate_m8a(
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
    kappa: float = typer.Option(
        2.0,
        "--kappa",
        help="Transition/transversion ratio",
    ),
    p0: float = typer.Option(
        ...,
        "--p0",
        help="Proportion in beta distribution classes",
    ),
    p: float = typer.Option(
        ...,
        "--p",
        help="Beta distribution parameter p (shape)",
    ),
    q: float = typer.Option(
        ...,
        "--q",
        help="Beta distribution parameter q (shape)",
    ),
    ncateg: int = typer.Option(
        10,
        "--ncateg",
        help="Number of discrete categories for beta",
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
    output_site_classes: bool = typer.Option(
        True,
        "--output-site-classes/--no-output-site-classes",
        help="Write site class assignments to file",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress messages",
    ),
):
    """
    Simulate under M8a model (beta + neutral).

    The M8a model is the null model for M8a vs M8 test. It combines
    a beta distribution for omega in (0,1) with a neutral class (omega=1).
    This is identical to M8 except omega_s is fixed to 1.0.

    Examples:

        \\b
        # Simulate with 80% beta-distributed, 20% neutral
        crabml simulate m8a -t tree.nwk -o sim.fasta -l 1000 \\
            --p0 0.8 --p 2 --q 5

        \\b
        # Mostly neutral with small fraction under purifying selection
        crabml simulate m8a -t tree.nwk -o sim.fasta -l 1000 \\
            --p0 0.9 --p 1 --q 2 -r 10
    """
    from ...simulate.codon import M8aSimulator
    from ...simulate.output import SimulationOutput

    if not quiet:
        typer.echo("crabML Sequence Simulator - M8a Model")
        typer.echo("=" * 50)

    # Load tree
    if not quiet:
        typer.echo(f"Loading tree from {tree}...")
    with open(tree) as f:
        tree_str = f.read()
    tree_obj = Tree.from_newick(tree_str)

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
        from scipy.stats import beta as beta_dist
        mean_beta_omega = beta_dist(p, q).mean()
        p_s = 1 - p0
        typer.echo(f"\nSimulation parameters:")
        typer.echo(f"  Model: M8a")
        typer.echo(f"  Sequence length: {length} codons")
        typer.echo(f"  Kappa (ts/tv): {kappa:.4f}")
        typer.echo(f"  Beta classes ({ncateg}): p0={p0:.3f}, beta({p:.2f},{q:.2f}), mean={mean_beta_omega:.4f}")
        typer.echo(f"  Neutral class: p_s={p_s:.3f}, omega=1.0 (fixed)")
        typer.echo(f"  Replicates: {replicates}")
        if seed is not None:
            typer.echo(f"  Seed: {seed}")

    try:
        simulator = M8aSimulator(
            tree=tree_obj,
            sequence_length=length,
            kappa=kappa,
            p0=p0,
            p=p,
            q=q,
            n_beta_categories=ncateg,
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
        # Simulate
        try:
            sequences = simulator.simulate()
        except Exception as e:
            typer.echo(f"Error simulating replicate {rep+1}: {e}", err=True)
            raise typer.Exit(code=1)

        # Determine output path
        if replicates == 1:
            out_path = output
        else:
            out_path = output.parent / f"{output.stem}_rep{rep+1}{output.suffix}"

        # Write sequences
        try:
            SimulationOutput.write_fasta(sequences, out_path, replicate_id=rep+1 if replicates > 1 else None)
        except Exception as e:
            typer.echo(f"Error writing output: {e}", err=True)
            raise typer.Exit(code=1)

        if not quiet:
            typer.echo(f"  Replicate {rep+1} -> {out_path}")

        # Write site classes
        if output_site_classes:
            site_classes_path = out_path.parent / f"{out_path.stem}.site_classes.txt"
            # Note: M8a has no positively selected sites (omega=1.0)

            try:
                site_info = simulator.get_site_classes()
                SimulationOutput.write_site_classes(
                    np.array(site_info['site_class_ids']),
                    np.array(site_info['site_class_omegas']),
                    site_classes_path
                )
            except Exception as e:
                typer.echo(f"Warning: Could not write site classes: {e}", err=True)

    # Write parameters
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
