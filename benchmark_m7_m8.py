#!/usr/bin/env python3
"""
Benchmark M7 and M8 models: PyCodeML (Rust) vs PAML.

This script compares the performance of PyCodeML's Rust-accelerated
likelihood calculations against PAML's reference implementation.
"""

import time
import subprocess
import numpy as np
from pathlib import Path

from crabml.models.codon import (
    M7CodonModel,
    M8CodonModel,
    compute_codon_frequencies_f3x4,
)
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.core.likelihood_rust import RustLikelihoodCalculator


def benchmark_paml_m7(ctl_file: str, n_runs: int = 5):
    """Benchmark PAML M7 model."""
    times = []

    for i in range(n_runs):
        start = time.time()
        result = subprocess.run(
            ["/home/adkern/py-codeml/paml/src/codeml", ctl_file],
            cwd="tests/data/paml_reference",
            capture_output=True,
            text=True
        )
        end = time.time()

        if result.returncode != 0:
            print(f"PAML M7 run {i+1} failed!")
            continue

        times.append(end - start)

    return times


def benchmark_paml_m8(ctl_file: str, n_runs: int = 5):
    """Benchmark PAML M8 model."""
    times = []

    for i in range(n_runs):
        start = time.time()
        result = subprocess.run(
            ["/home/adkern/py-codeml/paml/src/codeml", ctl_file],
            cwd="tests/data/paml_reference",
            capture_output=True,
            text=True
        )
        end = time.time()

        if result.returncode != 0:
            print(f"PAML M8 run {i+1} failed!")
            continue

        times.append(end - start)

    return times


def benchmark_rust_m7(n_runs: int = 5):
    """Benchmark PyCodeML Rust M7 model."""
    # Load data once
    aln = Alignment.from_phylip(
        "tests/data/paml_examples/lysozyme/lysozymeSmall.txt",
        seqtype="codon"
    )
    pi = compute_codon_frequencies_f3x4(aln)

    tree_str = (
        "((Hsa_Human: 0.025610, Hla_gibbon: 0.039441): 0.069693, "
        "((Cgu/Can_colobus: 0.044220, Pne_langur: 0.052394): 0.077888, "
        "Mmu_rhesus: 0.021399): 0.044279, "
        "(Ssc_squirrelM: 0.041624, Cja_marmoset: 0.023739): 0.125873);"
    )
    tree = Tree.from_newick(tree_str)

    model = M7CodonModel(
        kappa=4.314921,
        p_beta=0.007506,
        q_beta=0.005000,
        ncatG=10,
        pi=pi
    )

    Q_matrices = model.get_Q_matrices()
    proportions = model.get_site_classes()[0]
    calc = RustLikelihoodCalculator(aln, tree)

    times = []
    for i in range(n_runs):
        start = time.time()
        lnL = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )
        end = time.time()
        times.append(end - start)

    return times, lnL


def benchmark_rust_m8(n_runs: int = 5):
    """Benchmark PyCodeML Rust M8 model."""
    # Load data once
    aln = Alignment.from_phylip(
        "tests/data/paml_examples/lysozyme/lysozymeSmall.txt",
        seqtype="codon"
    )
    pi = compute_codon_frequencies_f3x4(aln)

    tree_str = (
        "((Hsa_Human: 0.025174, Hla_gibbon: 0.041255): 0.077239, "
        "((Cgu/Can_colobus: 0.044374, Pne_langur: 0.053701): 0.084936, "
        "Mmu_rhesus: 0.019172): 0.046065, "
        "(Ssc_squirrelM: 0.042928, Cja_marmoset: 0.024782): 0.136634);"
    )
    tree = Tree.from_newick(tree_str)

    model = M8CodonModel(
        kappa=5.029752,
        p0=0.868486,
        p_beta=9.517504,
        q_beta=13.543157,
        omega_s=4.370617,
        ncatG=10,
        pi=pi
    )

    Q_matrices = model.get_Q_matrices()
    proportions = model.get_site_classes()[0]
    calc = RustLikelihoodCalculator(aln, tree)

    times = []
    for i in range(n_runs):
        start = time.time()
        lnL = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )
        end = time.time()
        times.append(end - start)

    return times, lnL


def print_stats(label: str, times: list, lnL: float = None):
    """Print benchmark statistics."""
    times = np.array(times)
    print(f"\n{label}:")
    print(f"  Mean time:   {times.mean():.4f}s ± {times.std():.4f}s")
    print(f"  Median time: {np.median(times):.4f}s")
    print(f"  Min time:    {times.min():.4f}s")
    print(f"  Max time:    {times.max():.4f}s")
    if lnL is not None:
        print(f"  lnL:         {lnL:.6f}")


if __name__ == "__main__":
    print("="*70)
    print("PyCodeML (Rust) vs PAML Benchmark: M7 and M8 Models")
    print("="*70)

    N_RUNS = 2

    # M7 Benchmarks
    print("\n" + "="*70)
    print("M7 Model (Beta Distribution)")
    print("="*70)

    print("\n[1/4] Benchmarking PAML M7 (C implementation)...")
    paml_m7_times = benchmark_paml_m7("lysozyme_m7.ctl", n_runs=N_RUNS)
    print_stats("PAML M7", paml_m7_times)

    print("\n[2/4] Benchmarking PyCodeML M7 (Rust implementation)...")
    rust_m7_times, rust_m7_lnL = benchmark_rust_m7(n_runs=N_RUNS)
    print_stats("PyCodeML M7 (Rust)", rust_m7_times, rust_m7_lnL)

    m7_speedup = np.median(paml_m7_times) / np.median(rust_m7_times)
    print(f"\n  ⚡ SPEEDUP: {m7_speedup:.1f}x faster than PAML")

    # M8 Benchmarks
    print("\n" + "="*70)
    print("M8 Model (Beta & omega>1)")
    print("="*70)

    print("\n[3/4] Benchmarking PAML M8 (C implementation)...")
    paml_m8_times = benchmark_paml_m8("lysozyme_m8.ctl", n_runs=N_RUNS)
    print_stats("PAML M8", paml_m8_times)

    print("\n[4/4] Benchmarking PyCodeML M8 (Rust implementation)...")
    rust_m8_times, rust_m8_lnL = benchmark_rust_m8(n_runs=N_RUNS)
    print_stats("PyCodeML M8 (Rust)", rust_m8_times, rust_m8_lnL)

    m8_speedup = np.median(paml_m8_times) / np.median(rust_m8_times)
    print(f"\n  ⚡ SPEEDUP: {m8_speedup:.1f}x faster than PAML")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nM7 Model:")
    print(f"  PAML:          {np.median(paml_m7_times):.4f}s")
    print(f"  PyCodeML:      {np.median(rust_m7_times):.4f}s")
    print(f"  Speedup:       {m7_speedup:.1f}x")

    print(f"\nM8 Model:")
    print(f"  PAML:          {np.median(paml_m8_times):.4f}s")
    print(f"  PyCodeML:      {np.median(rust_m8_times):.4f}s")
    print(f"  Speedup:       {m8_speedup:.1f}x")

    avg_speedup = (m7_speedup + m8_speedup) / 2
    print(f"\nAverage Speedup: {avg_speedup:.1f}x")
    print("\n" + "="*70)
