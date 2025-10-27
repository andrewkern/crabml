#!/usr/bin/env python3
"""
crabML vs PAML Benchmark Orchestrator.

Run simulations, analyses, comparisons, and visualizations.
"""

import argparse
import shutil
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from lib import (
    BenchmarkSimulator,
    PAMLRunner,
    CrabMLRunner,
    BenchmarkComparator,
    BenchmarkVisualizer
)


def load_config(config_file: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_file is None:
        config_file = Path(__file__).parent / "config.yaml"

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def simulate_single(args):
    """Simulate a single dataset (for parallel execution)."""
    config, model, tree_newick, tree_id, params, seq_length, output_dir, rep_id = args

    simulator = BenchmarkSimulator(config)

    try:
        metadata = simulator.generate_dataset(
            model=model,
            replicate_id=rep_id,
            tree_newick=tree_newick,
            tree_id=tree_id,
            params=params,
            seq_length=seq_length,
            output_dir=output_dir
        )
        return (model, rep_id, metadata)
    except Exception as e:
        print(f"ERROR in {model} rep{rep_id:03d}: {e}")
        return (model, rep_id, None)


def phase_simulate(config: dict, models: list = None, parallel: bool = True):
    """Phase 1: Generate simulated datasets."""
    print("=" * 80)
    print("PHASE 1: SIMULATION")
    print("=" * 80)

    if models is None:
        models = config['models']

    data_dir = Path(__file__).parent / "data"
    simulator = BenchmarkSimulator(config)
    n_replicates = config['n_replicates']
    seq_lengths = config['simulation']['sequence_lengths']

    # Generate trees once
    trees = simulator.generate_trees()
    tree_ids = list(trees.keys())

    # Collect all simulation tasks
    tasks = []
    for model in models:
        model_dir = data_dir / model
        rep_id = 1
        for i in range(n_replicates):
            # Cycle through trees and sequence lengths
            tree_id = tree_ids[i % len(tree_ids)]
            seq_length = seq_lengths[i % len(seq_lengths)]
            tree_newick = trees[tree_id]

            # Sample parameters
            params = simulator.sample_parameters(model)

            tasks.append((
                config, model, tree_newick, tree_id, params,
                seq_length, model_dir, rep_id
            ))
            rep_id += 1

    print(f"Simulating {len(tasks)} datasets...")

    if parallel:
        n_jobs = config.get('simulation', {}).get('parallel_jobs', 30)
        print(f"Using {n_jobs} parallel jobs")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(simulate_single, task) for task in tasks]

            completed = 0
            success = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    model, rep_id, metadata = result
                    completed += 1
                    if metadata:
                        success += 1
                        print(f"  [{completed}/{len(tasks)}] {model} rep{rep_id:03d}: "
                              f"{metadata['tree_id']}, {metadata['sequence_length']} codons")
                    else:
                        print(f"  [{completed}/{len(tasks)}] {model} rep{rep_id:03d}: FAILED")

        print(f"\nGenerated {success}/{len(tasks)} datasets total.")
    else:
        # Sequential execution
        success = 0
        for i, task in enumerate(tasks, 1):
            result = simulate_single(task)
            if result:
                model, rep_id, metadata = result
                if metadata:
                    success += 1
                    print(f"  [{i}/{len(tasks)}] {model} rep{rep_id:03d}: "
                          f"{metadata['tree_id']}, {metadata['sequence_length']} codons")
                else:
                    print(f"  [{i}/{len(tasks)}] {model} rep{rep_id:03d}: FAILED")

        print(f"\nGenerated {success}/{len(tasks)} datasets total.")


def run_paml_single(args):
    """Run PAML on a single dataset (for parallel execution)."""
    config, model, data_dir, results_dir, rep_id = args

    runner = PAMLRunner(config)

    model_data_dir = data_dir / model
    seq_file = model_data_dir / f"rep{rep_id:03d}.fasta"
    tree_file = model_data_dir / f"rep{rep_id:03d}.nwk"

    if not seq_file.exists() or not tree_file.exists():
        return None

    model_results_dir = results_dir / "paml" / model

    try:
        result = runner.run_analysis(
            seq_file=seq_file,
            tree_file=tree_file,
            model=model,
            output_dir=model_results_dir,
            replicate_id=rep_id
        )
        return (model, rep_id, result)
    except Exception as e:
        print(f"ERROR in {model} rep{rep_id:03d}: {e}")
        return (model, rep_id, None)


def phase_run_paml(config: dict, models: list = None, parallel: bool = True, resume: bool = False):
    """Phase 2: Run PAML analyses."""
    print("=" * 80)
    print("PHASE 2: PAML ANALYSIS")
    print("=" * 80)

    if models is None:
        models = config['models']

    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    n_replicates = config['n_replicates']

    # Collect all tasks
    tasks = []
    for model in models:
        model_results_dir = results_dir / "paml" / model
        for rep_id in range(1, n_replicates + 1):
            # Check if already completed (for resume)
            if resume:
                results_file = model_results_dir / f"rep{rep_id:03d}_results.json"
                if results_file.exists():
                    continue

            tasks.append((config, model, data_dir, results_dir, rep_id))

    print(f"Running {len(tasks)} PAML analyses...")

    if parallel:
        n_jobs = config['paml'].get('parallel_jobs', 30)
        print(f"Using {n_jobs} parallel jobs")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(run_paml_single, task) for task in tasks]

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    model, rep_id, data = result
                    completed += 1
                    if data and data.get("converged"):
                        print(f"  [{completed}/{len(tasks)}] {model} rep{rep_id:03d}: "
                              f"lnL = {data['lnL']:.4f}")
                    else:
                        print(f"  [{completed}/{len(tasks)}] {model} rep{rep_id:03d}: FAILED")
    else:
        # Sequential execution
        for i, task in enumerate(tasks, 1):
            result = run_paml_single(task)
            if result:
                model, rep_id, data = result
                if data and data.get("converged"):
                    print(f"  [{i}/{len(tasks)}] {model} rep{rep_id:03d}: "
                          f"lnL = {data['lnL']:.4f}")
                else:
                    print(f"  [{i}/{len(tasks)}] {model} rep{rep_id:03d}: FAILED")

    print("\nPAML analysis complete.")


def phase_run_crabml(config: dict, models: list = None, resume: bool = False):
    """Phase 3: Run crabML analyses."""
    print("=" * 80)
    print("PHASE 3: crabML ANALYSIS")
    print("=" * 80)

    if models is None:
        models = config['models']

    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"
    n_replicates = config['n_replicates']

    runner = CrabMLRunner(config)

    total_tasks = len(models) * n_replicates
    completed = 0

    for model in models:
        print(f"\nRunning {model}...")
        model_data_dir = data_dir / model
        model_results_dir = results_dir / "crabml" / model

        for rep_id in range(1, n_replicates + 1):
            # Check if already completed (for resume)
            if resume:
                results_file = model_results_dir / f"rep{rep_id:03d}_crabml.json"
                if results_file.exists():
                    completed += 1
                    continue

            seq_file = model_data_dir / f"rep{rep_id:03d}.fasta"
            tree_file = model_data_dir / f"rep{rep_id:03d}.nwk"

            if not seq_file.exists() or not tree_file.exists():
                print(f"  rep{rep_id:03d}: Missing input files")
                completed += 1
                continue

            try:
                result = runner.run_analysis(
                    seq_file=seq_file,
                    tree_file=tree_file,
                    model=model,
                    output_dir=model_results_dir,
                    replicate_id=rep_id
                )
                completed += 1

                if result.get("converged"):
                    print(f"  [{completed}/{total_tasks}] rep{rep_id:03d}: "
                          f"lnL = {result['lnL']:.4f}, time = {result['runtime']:.2f}s")
                else:
                    print(f"  [{completed}/{total_tasks}] rep{rep_id:03d}: FAILED")

            except Exception as e:
                completed += 1
                print(f"  [{completed}/{total_tasks}] rep{rep_id:03d}: ERROR - {e}")

    print("\ncrabML analysis complete.")


def phase_compare(config: dict, models: list = None):
    """Phase 4: Compare results."""
    print("=" * 80)
    print("PHASE 4: COMPARISON")
    print("=" * 80)

    if models is None:
        models = config['models']

    data_dir = Path(__file__).parent / "data"
    paml_dir = Path(__file__).parent / "results" / "paml"
    crabml_dir = Path(__file__).parent / "results" / "crabml"
    output_dir = Path(__file__).parent / "results"

    comparator = BenchmarkComparator(config)
    df, summary_stats = comparator.compare_all(
        data_dir=data_dir,
        paml_dir=paml_dir,
        crabml_dir=crabml_dir,
        models=models,
        output_dir=output_dir
    )

    return df, summary_stats


def phase_visualize(config: dict, models: list = None):
    """Phase 5: Generate visualizations."""
    print("=" * 80)
    print("PHASE 5: VISUALIZATION")
    print("=" * 80)

    if models is None:
        models = config['models']

    # Load comparison results
    results_dir = Path(__file__).parent / "results"
    comparison_file = results_dir / "comparison.csv"
    summary_file = results_dir / "summary_stats.json"

    if not comparison_file.exists() or not summary_file.exists():
        print("ERROR: Comparison results not found. Run 'compare' phase first.")
        return

    import pandas as pd
    import json

    df = pd.read_csv(comparison_file)
    with open(summary_file, 'r') as f:
        summary_stats = json.load(f)

    plots_dir = Path(__file__).parent / "plots"
    visualizer = BenchmarkVisualizer(config)
    visualizer.plot_all(df, summary_stats, models, plots_dir)


def phase_clean():
    """Remove generated data and results."""
    print("=" * 80)
    print("CLEAN")
    print("=" * 80)

    base_dir = Path(__file__).parent

    dirs_to_remove = [
        base_dir / "data",
        base_dir / "results",
    ]

    for dir_path in dirs_to_remove:
        if dir_path.exists():
            print(f"Removing {dir_path}...")
            shutil.rmtree(dir_path)

    print("Clean complete.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="crabML vs PAML Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "command",
        choices=["simulate", "run-paml", "run-crabml", "compare", "visualize", "all", "clean"],
        help="Command to run"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to benchmark (default: all from config)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run simulation and PAML in parallel (default: False)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing results (skip completed runs)"
    )

    args = parser.parse_args()

    # Load configuration
    if args.command != "clean":
        config = load_config(args.config)
        models = args.models if args.models else config['models']
    else:
        config = None
        models = None

    # Execute command
    if args.command == "simulate":
        phase_simulate(config, models, parallel=args.parallel)

    elif args.command == "run-paml":
        phase_run_paml(config, models, parallel=args.parallel, resume=args.resume)

    elif args.command == "run-crabml":
        phase_run_crabml(config, models, resume=args.resume)

    elif args.command == "compare":
        phase_compare(config, models)

    elif args.command == "visualize":
        phase_visualize(config, models)

    elif args.command == "all":
        phase_simulate(config, models, parallel=args.parallel)
        phase_run_paml(config, models, parallel=args.parallel)
        phase_run_crabml(config, models)
        phase_compare(config, models)
        phase_visualize(config, models)

    elif args.command == "clean":
        phase_clean()

    print("\nDone!")


if __name__ == "__main__":
    main()
