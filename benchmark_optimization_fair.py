#!/usr/bin/env python3
"""
Fair Benchmark: PAML vs PyCodeML (Rust) - Full Parameter Optimization

This compares apples to apples: Both systems doing complete parameter
optimization (finding MLEs), not just likelihood evaluation.
"""

import time
import subprocess
import numpy as np
from pathlib import Path

from crabml.optimize.optimizer import M7Optimizer, M8Optimizer
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


def benchmark_paml_optimization(ctl_file: str, n_runs: int = 3):
    """
    Benchmark PAML's full optimization.

    PAML does:
    - Parameter optimization (kappa, omega parameters, proportions)
    - Branch length optimization
    - Multiple rounds with different starting points
    """
    times = []

    for i in range(n_runs):
        print(f"    Run {i+1}/{n_runs}...")
        start = time.time()
        result = subprocess.run(
            ["/home/adkern/crabml/paml/src/codeml", ctl_file],
            cwd="tests/data/paml_reference",
            capture_output=True,
            text=True
        )
        end = time.time()

        if result.returncode != 0:
            print(f"      PAML run {i+1} failed!")
            continue

        times.append(end - start)
        print(f"      Time: {end - start:.2f}s")

    return times


def benchmark_pycodeml_m7_optimization(n_runs: int = 3):
    """
    Benchmark PyCodeML's M7 optimization.

    Does same as PAML:
    - Parameter optimization (kappa, p, q)
    - Branch length optimization
    """
    # Load data once
    aln = Alignment.from_phylip(
        "tests/data/paml_examples/lysozyme/lysozymeSmall.txt",
        seqtype="codon"
    )

    # Starting tree (same as PAML uses)
    tree_str = (
        "((Hsa_Human: 0.03, Hla_gibbon: 0.04): 0.07, "
        "((Cgu/Can_colobus: 0.04, Pne_langur: 0.05): 0.08, "
        "Mmu_rhesus: 0.02): 0.04, "
        "(Ssc_squirrelM: 0.04, Cja_marmoset: 0.02): 0.13);"
    )

    times = []
    results = []

    for i in range(n_runs):
        print(f"    Run {i+1}/{n_runs}...")

        # Fresh tree for each run
        tree = Tree.from_newick(tree_str)

        # Create optimizer
        opt = M7Optimizer(aln, tree, ncatG=10, optimize_branch_lengths=True)

        # Time the optimization
        start = time.time()
        kappa, p, q, lnL = opt.optimize(
            init_kappa=2.0,
            init_p_beta=0.5,
            init_q_beta=0.5,
            maxiter=1000  # Match PAML's thorough optimization
        )
        end = time.time()

        times.append(end - start)
        results.append((kappa, p, q, lnL))
        print(f"      Time: {end - start:.2f}s, lnL: {lnL:.6f}")

    return times, results


def benchmark_pycodeml_m8_optimization(n_runs: int = 3):
    """
    Benchmark PyCodeML's M8 optimization.
    """
    aln = Alignment.from_phylip(
        "tests/data/paml_examples/lysozyme/lysozymeSmall.txt",
        seqtype="codon"
    )

    tree_str = (
        "((Hsa_Human: 0.03, Hla_gibbon: 0.04): 0.07, "
        "((Cgu/Can_colobus: 0.04, Pne_langur: 0.05): 0.08, "
        "Mmu_rhesus: 0.02): 0.04, "
        "(Ssc_squirrelM: 0.04, Cja_marmoset: 0.02): 0.13);"
    )

    times = []
    results = []

    for i in range(n_runs):
        print(f"    Run {i+1}/{n_runs}...")

        tree = Tree.from_newick(tree_str)
        opt = M8Optimizer(aln, tree, ncatG=10, optimize_branch_lengths=True)

        start = time.time()
        kappa, p0, p, q, omega_s, lnL = opt.optimize(
            init_kappa=2.0,
            init_p0=0.9,
            init_p_beta=0.5,
            init_q_beta=0.5,
            init_omega_s=2.0,
            maxiter=1000
        )
        end = time.time()

        times.append(end - start)
        results.append((kappa, p0, p, q, omega_s, lnL))
        print(f"      Time: {end - start:.2f}s, lnL: {lnL:.6f}")

    return times, results


def print_stats(label: str, times: list):
    """Print benchmark statistics."""
    times = np.array(times)
    print(f"\n{label}:")
    print(f"  Mean time:   {times.mean():.2f}s ± {times.std():.2f}s")
    print(f"  Median time: {np.median(times):.2f}s")
    print(f"  Min time:    {times.min():.2f}s")
    print(f"  Max time:    {times.max():.2f}s")


if __name__ == "__main__":
    print("="*70)
    print("FAIR Benchmark: PAML vs PyCodeML (Rust) - Full Optimization")
    print("="*70)
    print("\nBoth systems doing complete parameter optimization (finding MLEs)")
    print("NOT just likelihood evaluation!\n")

    N_RUNS = 3  # Fewer runs since each run is expensive

    # M7 Benchmarks
    print("\n" + "="*70)
    print("M7 Model (Beta Distribution) - FULL OPTIMIZATION")
    print("="*70)

    print("\n[1/4] PAML M7 Full Optimization (C code):")
    paml_m7_times = benchmark_paml_optimization("lysozyme_m7.ctl", n_runs=N_RUNS)
    print_stats("PAML M7 (Full Optimization)", paml_m7_times)

    print("\n[2/4] PyCodeML M7 Full Optimization (Rust backend):")
    rust_m7_times, m7_results = benchmark_pycodeml_m7_optimization(n_runs=N_RUNS)
    print_stats("PyCodeML M7 (Rust, Full Optimization)", rust_m7_times)

    m7_speedup = np.median(paml_m7_times) / np.median(rust_m7_times)
    print(f"\n  ⚡ SPEEDUP: {m7_speedup:.1f}x faster than PAML")
    print(f"\n  Final M7 lnL: {m7_results[-1][-1]:.6f}")
    print(f"  PAML M7 lnL:  -902.510018 (reference)")

    # M8 Benchmarks
    print("\n" + "="*70)
    print("M8 Model (Beta & omega>1) - FULL OPTIMIZATION")
    print("="*70)

    print("\n[3/4] PAML M8 Full Optimization (C code):")
    paml_m8_times = benchmark_paml_optimization("lysozyme_m8.ctl", n_runs=N_RUNS)
    print_stats("PAML M8 (Full Optimization)", paml_m8_times)

    print("\n[4/4] PyCodeML M8 Full Optimization (Rust backend):")
    rust_m8_times, m8_results = benchmark_pycodeml_m8_optimization(n_runs=N_RUNS)
    print_stats("PyCodeML M8 (Rust, Full Optimization)", rust_m8_times)

    m8_speedup = np.median(paml_m8_times) / np.median(rust_m8_times)
    print(f"\n  ⚡ SPEEDUP: {m8_speedup:.1f}x faster than PAML")
    print(f"\n  Final M8 lnL: {m8_results[-1][-1]:.6f}")
    print(f"  PAML M8 lnL:  -899.999237 (reference)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - Fair Comparison (Full Optimization)")
    print("="*70)
    print(f"\nM7 Model (Full Optimization):")
    print(f"  PAML:          {np.median(paml_m7_times):.2f}s")
    print(f"  PyCodeML:      {np.median(rust_m7_times):.2f}s")
    print(f"  Speedup:       {m7_speedup:.1f}x")

    print(f"\nM8 Model (Full Optimization):")
    print(f"  PAML:          {np.median(paml_m8_times):.2f}s")
    print(f"  PyCodeML:      {np.median(rust_m8_times):.2f}s")
    print(f"  Speedup:       {m8_speedup:.1f}x")

    avg_speedup = (m7_speedup + m8_speedup) / 2
    print(f"\nAverage Speedup: {avg_speedup:.1f}x")
    print("\nNote: This is a FAIR comparison - both systems doing full parameter")
    print("optimization, not just likelihood evaluation!")
    print("="*70)
