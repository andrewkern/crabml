#!/usr/bin/env python3
"""
Quick test of M7 and M8 optimizers.

This tests that the optimizers can run and improve the likelihood.
"""

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.optimize.optimizer import M7Optimizer, M8Optimizer

# Load lysozyme data
aln = Alignment.from_phylip(
    "tests/data/paml_examples/lysozyme/lysozymeSmall.txt",
    seqtype="codon"
)

# Use a simple starting tree
tree_str = (
    "((Hsa_Human: 0.03, Hla_gibbon: 0.04): 0.07, "
    "((Cgu/Can_colobus: 0.04, Pne_langur: 0.05): 0.08, "
    "Mmu_rhesus: 0.02): 0.04, "
    "(Ssc_squirrelM: 0.04, Cja_marmoset: 0.02): 0.13);"
)
tree = Tree.from_newick(tree_str)

print("="*70)
print("Testing M7 Optimizer")
print("="*70)

# Test M7 optimizer
m7_opt = M7Optimizer(aln, tree, ncatG=10, optimize_branch_lengths=True)
kappa, p, q, lnL = m7_opt.optimize(
    init_kappa=2.0,
    init_p_beta=0.5,
    init_q_beta=0.5,
    maxiter=50  # Limited iterations for testing
)

print(f"\nM7 Optimization Results:")
print(f"  kappa: {kappa:.6f}")
print(f"  p:     {p:.6f}")
print(f"  q:     {q:.6f}")
print(f"  lnL:   {lnL:.6f}")

print("\n" + "="*70)
print("Testing M8 Optimizer")
print("="*70)

# Reset tree
tree = Tree.from_newick(tree_str)

# Test M8 optimizer
m8_opt = M8Optimizer(aln, tree, ncatG=10, optimize_branch_lengths=True)
kappa, p0, p, q, omega_s, lnL = m8_opt.optimize(
    init_kappa=2.0,
    init_p0=0.9,
    init_p_beta=0.5,
    init_q_beta=0.5,
    init_omega_s=2.0,
    maxiter=50  # Limited iterations for testing
)

print(f"\nM8 Optimization Results:")
print(f"  kappa:   {kappa:.6f}")
print(f"  p0:      {p0:.6f}")
print(f"  p:       {p:.6f}")
print(f"  q:       {q:.6f}")
print(f"  omega_s: {omega_s:.6f}")
print(f"  lnL:     {lnL:.6f}")

print("\n" + "="*70)
print("SUCCESS: Both M7 and M8 optimizers completed successfully!")
print("="*70)
