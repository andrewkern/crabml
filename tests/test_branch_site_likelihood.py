"""
Test Branch-Site Model A likelihood calculation against PAML reference.

This test verifies that our Rust implementation produces exact numerical
agreement with PAML for the branch-site model A likelihood calculation.

PAML Reference (lysozyme dataset):
- lnL = -1035.533916
- kappa = 4.154927
- p0 = 0.326513
- p1 = 0.269308
- omega0 = 0.000001
- omega2 = 4.809765
"""
import pytest
import numpy as np
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.models.codon import build_codon_Q_matrix, compute_codon_frequencies_f3x4
from crabml.models.codon_branch_site import BranchSiteModelA
from crabml.core.likelihood_rust import RustLikelihoodCalculator
import crabml_rust


def test_branch_site_model_a_likelihood():
    """Test Branch-Site Model A likelihood matches PAML exactly."""

    # Load alignment
    alignment = Alignment.from_phylip(
        'tests/data/branch_site/lysozymeLarge_clean.nuc',
        seqtype='codon'
    )

    # Load tree
    with open('tests/data/branch_site/lysozymeLarge.trees') as f:
        tree_str = f.read()
    tree = Tree.from_newick(tree_str)

    # Update branch lengths to PAML's MLEs
    # These are the optimized values from PAML's output
    paml_branch_lengths = [
        0.042333, 0.075622, 0.026519, 0.008018, 0.007767, 0.000004, 0.023339,
        0.007530, 0.037779, 0.023589, 0.000004, 0.000004, 0.007956, 0.000004,
        0.007935, 0.016006, 0.008104, 0.016154, 0.007906, 0.016186, 0.008059,
        0.069643, 0.006258, 0.008004, 0.015965, 0.000004, 0.000004, 0.034475,
        0.135219, 0.027063, 0.000004, 0.032959, 0.040867
    ]

    branches = tree.get_branches()
    for i, (parent, child) in enumerate(branches):
        child.branch_length = paml_branch_lengths[i]

    # Validate tree has correct branch labels
    tree.validate_branch_site_labels()

    # PAML MLEs
    kappa = 4.154927
    p0 = 0.326513
    p1 = 0.269308
    omega0 = 0.000001
    omega2 = 4.809765

    # Compute codon frequencies using F3x4
    pi = compute_codon_frequencies_f3x4(alignment)

    # Create model
    model = BranchSiteModelA(pi, tree.get_branch_labels())

    # Compute site class frequencies
    site_class_freqs = model.compute_site_class_frequencies(p0, p1)

    # Verify site class frequencies match PAML
    expected_freqs = np.array([0.326513, 0.269308, 0.221492, 0.182687])
    np.testing.assert_allclose(site_class_freqs, expected_freqs, atol=1e-6)

    # Compute Qfactors
    qfactor_back, qfactor_fore = model.compute_qfactors(kappa, p0, p1, omega0, omega2)

    # Build Q matrices (no normalization - Qfactor applied to branch lengths)
    Q_omega0 = build_codon_Q_matrix(kappa=kappa, omega=omega0, pi=pi, normalization_factor=1.0)
    Q_omega1 = build_codon_Q_matrix(kappa=kappa, omega=1.0, pi=pi, normalization_factor=1.0)
    Q_omega2 = build_codon_Q_matrix(kappa=kappa, omega=omega2, pi=pi, normalization_factor=1.0)

    # Prepare tree structure for Rust
    calc = RustLikelihoodCalculator(alignment, tree)

    # Remap branch labels to renumbered tree structure
    branches_original = tree.get_branches()
    branch_labels_original = tree.get_branch_labels()
    branch_labels_renumbered = [0] * (len(calc.tree_structure) - 1)

    for i, (parent, child) in enumerate(branches_original):
        renumbered_child_id = calc.node_id_map[child.id]
        if renumbered_child_id >= len(branch_labels_renumbered):
            continue
        branch_labels_renumbered[renumbered_child_id] = branch_labels_original[i]

    branch_labels_u8 = [int(x) for x in branch_labels_renumbered]

    # Compute likelihood
    log_likelihood = crabml_rust.compute_branch_site_log_likelihood(
        Q_omega0,
        Q_omega1,
        Q_omega2,
        qfactor_back,
        qfactor_fore,
        omega0,
        omega2,
        site_class_freqs.tolist(),
        pi,
        calc.tree_structure,
        calc.branch_lengths_template,
        branch_labels_u8,
        calc.leaf_names_ordered,
        calc.sequences,
        calc.leaf_node_ids,
    )

    # PAML reference likelihood
    paml_lnL = -1035.533916

    # Verify exact match (within numerical precision)
    assert abs(log_likelihood - paml_lnL) < 1e-5, (
        f"Likelihood mismatch: our lnL = {log_likelihood:.6f}, "
        f"PAML lnL = {paml_lnL:.6f}, "
        f"difference = {abs(log_likelihood - paml_lnL):.6f}"
    )

    print(f"✓ Branch-Site Model A likelihood matches PAML exactly!")
    print(f"  Our lnL:  {log_likelihood:.6f}")
    print(f"  PAML lnL: {paml_lnL:.6f}")
    print(f"  Difference: {abs(log_likelihood - paml_lnL):.10f}")


if __name__ == "__main__":
    test_branch_site_model_a_likelihood()
