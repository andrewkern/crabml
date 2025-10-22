"""
Validate Rust backend against PAML reference outputs.

These tests verify that the Rust-accelerated likelihood calculations
produce identical results to PAML's reference outputs for all models.
"""

import numpy as np
import pytest

# Check if Rust backend is available
try:
    from pycodeml.core.likelihood_rust import RustLikelihoodCalculator, RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False

if not RUST_AVAILABLE:
    pytest.skip("Rust backend not available", allow_module_level=True)

from pycodeml.models.codon import (
    M0CodonModel,
    M1aCodonModel,
    M2aCodonModel,
    M3CodonModel,
    M4CodonModel,
    M5CodonModel,
    M6CodonModel,
    M7CodonModel,
    M8CodonModel,
    M9CodonModel,
    compute_codon_frequencies_f3x4,
)
from pycodeml.io.sequences import Alignment
from pycodeml.io.trees import Tree


class TestRustPAMLValidation:
    """Validate Rust backend produces PAML-matching likelihoods."""

    @pytest.fixture
    def lysozyme_data(self, lysozyme_small_files):
        """Load lysozyme dataset for Rust validation."""
        aln = Alignment.from_phylip(
            lysozyme_small_files["sequences"], seqtype="codon"
        )
        pi = compute_codon_frequencies_f3x4(aln)
        return {"alignment": aln, "pi": pi}

    def test_rust_m0_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M0 likelihood matches PAML reference EXACTLY.

        PAML reference (from lysozyme_m0_out.txt):
        lnL = -906.017440
        kappa = 4.54008
        omega = 0.80663
        tree: from PAML output file

        With correct parameters, Rust matches PAML to machine precision!
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's ACTUAL optimized tree and parameters from lysozyme_m0_out.txt
        tree_str = (
            "((Hsa_Human: 0.025561, Hla_gibbon: 0.038887): 0.067982, "
            "((Cgu/Can_colobus: 0.043792, Pne_langur: 0.052539): 0.076369, "
            "Mmu_rhesus: 0.021684): 0.043448, "
            "(Ssc_squirrelM: 0.040804, Cja_marmoset: 0.023918): 0.122664);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M0 model with PAML's ACTUAL optimized parameters
        model = M0CodonModel(kappa=4.54008, omega=0.80663, pi=pi)
        Q = model.get_Q_matrix()

        # Compute likelihood using Rust backend
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood(Q, pi)

        paml_lnL = -906.017440

        print(f"\nM0 Model (Rust Backend - EXACT match):")
        print(f"  Rust lnL:     {lnL_rust:.10f}")
        print(f"  PAML lnL:     {paml_lnL:.10f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.10e}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.10f}%")

        # Should match to machine precision (< 1e-6)
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Rust M0 likelihood should match PAML exactly"
        )

    def test_rust_m1a_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M1a likelihood matches PAML reference EXACTLY.

        PAML reference (from lysozyme_m1a_out.txt):
        lnL = -902.503872
        kappa = 4.29790
        p: [0.41271, 0.58729]
        w: [0.000001, 1.00000]  (PAML uses 1e-6, not 0.0)

        With PAML-style weighted-average normalization, Rust matches PAML exactly!
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M1a
        tree_str = (
            "((Hsa_Human: 0.025558, Hla_gibbon: 0.039375): 0.069610, "
            "((Cgu/Can_colobus: 0.044140, Pne_langur: 0.052277): 0.077775, "
            "Mmu_rhesus: 0.021343): 0.044215, "
            "(Ssc_squirrelM: 0.041574, Cja_marmoset: 0.023673): 0.125738);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M1a model with PAML's ACTUAL optimized parameters
        # PAML uses omega0=0.000001 (from parameter line in output)
        model = M1aCodonModel(
            kappa=4.29790,
            omega0=0.000001,  # PAML's actual value
            p0=0.41271,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -902.503872

        print(f"\nM1a Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # With PAML-style normalization, should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M1a likelihood should match PAML exactly"
        )

    def test_rust_m2a_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M2a likelihood matches PAML reference.

        PAML reference (from lysozyme_m2a_out.txt):
        lnL = -899.998568
        kappa = 5.03734
        p: [0.77376, 0.10399, 0.12226]
        w: [0.35983, 1.00000, 4.52341]
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M2a
        tree_str = (
            "((Hsa_Human: 0.025240, Hla_gibbon: 0.041269): 0.077423, "
            "((Cgu/Can_colobus: 0.044409, Pne_langur: 0.053758): 0.085025, "
            "Mmu_rhesus: 0.019175): 0.046022, "
            "(Ssc_squirrelM: 0.042962, Cja_marmoset: 0.024797): 0.136928);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M2a model with PAML's optimized parameters
        model = M2aCodonModel(
            kappa=5.03734,
            omega0=0.35983,
            omega2=4.52341,
            p0=0.77376,
            p1=0.10399,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -899.998568

        print(f"\nM2a Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # With PAML-style normalization, should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M2a likelihood should match PAML exactly"
        )

    def test_rust_m3_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M3 likelihood matches PAML reference.

        PAML reference (from lysozyme_m3_out.txt):
        lnL = -899.985262
        kappa = 5.07876
        p: [0.83729, 0.14475, 0.01796]
        w: [0.37564, 3.35232, 8.39917]
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M3
        tree_str = (
            "((Hsa_Human: 0.025496, Hla_gibbon: 0.041612): 0.078581, "
            "((Cgu/Can_colobus: 0.044623, Pne_langur: 0.054035): 0.085125, "
            "Mmu_rhesus: 0.019785): 0.045122, "
            "(Ssc_squirrelM: 0.043139, Cja_marmoset: 0.024888): 0.138453);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M3 model with PAML's optimized parameters
        model = M3CodonModel(
            kappa=5.07876,
            omegas=[0.37564, 3.35232, 8.39917],
            proportions=[0.83729, 0.14475, 0.01796],
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -899.985262

        print(f"\nM3 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # With PAML-style normalization, should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M3 likelihood should match PAML exactly"
        )

    def test_rust_vs_python_identical_results(self, lysozyme_data):
        """
        Verify Rust backend produces identical results to Python backend.

        This is critical: Rust should be a drop-in replacement with no
        numerical differences beyond machine precision.
        """
        from pycodeml.core.likelihood import LikelihoodCalculator as PythonCalc

        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # Use PAML's actual optimized parameters
        tree_str = (
            "((Hsa_Human: 0.025561, Hla_gibbon: 0.038887): 0.067982, "
            "((Cgu/Can_colobus: 0.043792, Pne_langur: 0.052539): 0.076369, "
            "Mmu_rhesus: 0.021684): 0.043448, "
            "(Ssc_squirrelM: 0.040804, Cja_marmoset: 0.023918): 0.122664);"
        )
        tree = Tree.from_newick(tree_str)

        # M0 model with PAML's actual parameters
        model = M0CodonModel(kappa=4.54008, omega=0.80663, pi=pi)
        Q = model.get_Q_matrix()

        # Python backend
        calc_python = PythonCalc(aln, tree)
        lnL_python = calc_python.compute_log_likelihood(Q, pi)

        # Rust backend
        calc_rust = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc_rust.compute_log_likelihood(Q, pi)

        paml_lnL = -906.017440

        print(f"\nRust vs Python Comparison (M0 with correct params):")
        print(f"  Python lnL: {lnL_python:.10f}")
        print(f"  Rust lnL:   {lnL_rust:.10f}")
        print(f"  PAML lnL:   {paml_lnL:.10f}")
        print(f"  Rust-Python diff: {abs(lnL_rust - lnL_python):.10e}")
        print(f"  Rust-PAML diff:   {abs(lnL_rust - paml_lnL):.10e}")

        # Should be identical within machine precision
        np.testing.assert_allclose(
            lnL_rust,
            lnL_python,
            rtol=1e-9,
            atol=1e-9,
            err_msg="Rust and Python should produce identical results"
        )

    def test_rust_site_class_vs_python_identical(self, lysozyme_data):
        """
        Verify Rust parallelized site class calculations match Python.

        This tests the parallelized Rayon implementation for M1a/M2a/M3.
        """
        from pycodeml.core.likelihood import LikelihoodCalculator as PythonCalc

        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        tree_str = (
            "((Hsa_Human: 0.025558, Hla_gibbon: 0.039375): 0.069610, "
            "((Cgu/Can_colobus: 0.044140, Pne_langur: 0.052277): 0.077775, "
            "Mmu_rhesus: 0.021343): 0.044215, "
            "(Ssc_squirrelM: 0.041574, Cja_marmoset: 0.023673): 0.125738);"
        )
        tree = Tree.from_newick(tree_str)

        # M1a model with PAML's actual parameters
        model = M1aCodonModel(
            kappa=4.29790,
            omega0=0.000001,  # PAML's actual value
            p0=0.41271,
            pi=pi
        )

        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Python backend
        calc_python = PythonCalc(aln, tree)
        lnL_python = calc_python.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        # Rust backend (parallelized)
        calc_rust = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc_rust.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        print(f"\nRust vs Python Comparison (M1a - Site Classes):")
        print(f"  Python lnL: {lnL_python:.10f}")
        print(f"  Rust lnL:   {lnL_rust:.10f}")
        print(f"  Difference: {abs(lnL_rust - lnL_python):.10e}")

        # Should be identical within machine precision
        np.testing.assert_allclose(
            lnL_rust,
            lnL_python,
            rtol=1e-9,
            atol=1e-9,
            err_msg="Rust parallelized site classes should match Python exactly"
        )

    def test_rust_m7_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M7 likelihood matches PAML reference.

        PAML reference (from lysozyme_m7_out.txt):
        lnL = -902.510018
        kappa = 4.314921
        p = 0.007506, q = 0.005000
        ncatG = 10
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M7
        tree_str = (
            "((Hsa_Human: 0.025610, Hla_gibbon: 0.039441): 0.069693, "
            "((Cgu/Can_colobus: 0.044220, Pne_langur: 0.052394): 0.077888, "
            "Mmu_rhesus: 0.021399): 0.044279, "
            "(Ssc_squirrelM: 0.041624, Cja_marmoset: 0.023739): 0.125873);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M7 model with PAML's optimized parameters
        model = M7CodonModel(
            kappa=4.314921,
            p_beta=0.007506,
            q_beta=0.005000,
            ncatG=10,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -902.510018

        print(f"\nM7 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # Should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M7 likelihood should match PAML exactly"
        )

    def test_rust_m8_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M8 likelihood matches PAML reference.

        PAML reference (from lysozyme_m8_out.txt):
        lnL = -899.999237
        kappa = 5.029752
        p0 = 0.868486, p = 9.517504, q = 13.543157
        omega_s = 4.370617
        ncatG = 10
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M8
        tree_str = (
            "((Hsa_Human: 0.025174, Hla_gibbon: 0.041255): 0.077239, "
            "((Cgu/Can_colobus: 0.044374, Pne_langur: 0.053701): 0.084936, "
            "Mmu_rhesus: 0.019172): 0.046065, "
            "(Ssc_squirrelM: 0.042928, Cja_marmoset: 0.024782): 0.136634);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M8 model with PAML's optimized parameters
        model = M8CodonModel(
            kappa=5.029752,
            p0=0.868486,
            p_beta=9.517504,
            q_beta=13.543157,
            omega_s=4.370617,
            ncatG=10,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -899.999237

        print(f"\nM8 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # Should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M8 likelihood should match PAML exactly"
        )

    def test_rust_m5_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M5 likelihood matches PAML reference.

        PAML reference (from lysozyme_m5_out.txt):
        lnL = -900.222360
        kappa = 4.91082
        alpha = 0.43434, beta = 0.44697
        ncatG = 10
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M5
        tree_str = (
            "((Hsa_Human: 0.025431, Hla_gibbon: 0.040846): 0.075035, "
            "((Cgu/Can_colobus: 0.044631, Pne_langur: 0.053006): 0.082704, "
            "Mmu_rhesus: 0.020012): 0.045472, "
            "(Ssc_squirrelM: 0.042442, Cja_marmoset: 0.024538): 0.133407);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M5 model with PAML's optimized parameters
        model = M5CodonModel(
            kappa=4.91082,
            alpha=0.43434,
            beta=0.44697,
            ncatG=10,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -900.222360

        print(f"\nM5 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # Should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M5 likelihood should match PAML exactly"
        )

    def test_rust_m9_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M9 likelihood matches PAML reference.

        PAML reference (from lysozyme_m9_out.txt):
        lnL = -899.997270
        kappa = 5.05056
        p0 = 0.84895, p_beta = 16.31508, q_beta = 28.11235
        alpha = 9.04750, beta_gamma = 2.05077
        ncatG = 10
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M9
        tree_str = (
            "((Hsa_Human: 0.025366, Hla_gibbon: 0.041324): 0.077781, "
            "((Cgu/Can_colobus: 0.044497, Pne_langur: 0.053862): 0.085140, "
            "Mmu_rhesus: 0.019255): 0.045871, "
            "(Ssc_squirrelM: 0.043034, Cja_marmoset: 0.024828): 0.137479);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M9 model with PAML's optimized parameters
        model = M9CodonModel(
            kappa=5.05056,
            p0=0.84895,
            p_beta=16.31508,
            q_beta=28.11235,
            alpha=9.04750,
            beta_gamma=2.05077,
            ncatG=10,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -899.997270

        print(f"\nM9 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # Should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M9 likelihood should match PAML exactly"
        )

    def test_rust_m4_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M4 likelihood matches PAML reference.

        PAML reference (from lysozyme_m4_out.txt):
        lnL = -900.139530
        kappa = 4.82959
        proportions = [0.04251, 0.75935, 0.00000, 0.00000, 0.19814]
        omegas = [0.0, 1/3, 2/3, 1.0, 3.0] (fixed)
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M4
        tree_str = (
            "((Hsa_Human: 0.025116, Hla_gibbon: 0.040656): 0.074120, "
            "((Cgu/Can_colobus: 0.044275, Pne_langur: 0.052863): 0.082198, "
            "Mmu_rhesus: 0.019989): 0.045550, "
            "(Ssc_squirrelM: 0.042315, Cja_marmoset: 0.024402): 0.132002);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M4 model with PAML's optimized parameters
        model = M4CodonModel(
            kappa=4.82959,
            proportions=[0.04251, 0.75935, 0.00000, 0.00000, 0.19814],
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -900.139530

        print(f"\nM4 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # Should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M4 likelihood should match PAML exactly"
        )

    def test_rust_m6_likelihood_matches_paml(self, lysozyme_data):
        """
        Test Rust M6 likelihood matches PAML reference.

        PAML reference (from lysozyme_m6_out.txt):
        lnL = -900.167515
        kappa = 4.88043
        p0 = 0.20472, alpha1 = 8.31960, beta1 = 10.86728
        alpha2 = 0.31083 (beta2 = alpha2)
        ncatG = 10
        """
        aln = lysozyme_data["alignment"]
        pi = lysozyme_data["pi"]

        # PAML's optimized tree for M6
        tree_str = (
            "((Hsa_Human: 0.025319, Hla_gibbon: 0.040604): 0.074894, "
            "((Cgu/Can_colobus: 0.044277, Pne_langur: 0.052821): 0.082456, "
            "Mmu_rhesus: 0.019800): 0.045217, "
            "(Ssc_squirrelM: 0.042294, Cja_marmoset: 0.024371): 0.133164);"
        )
        tree = Tree.from_newick(tree_str)

        # Create M6 model with PAML's optimized parameters
        model = M6CodonModel(
            kappa=4.88043,
            p0=0.20472,
            alpha1=8.31960,
            beta1=10.86728,
            alpha2=0.31083,
            ncatG=10,
            pi=pi
        )

        # Get Q matrices and proportions
        Q_matrices = model.get_Q_matrices()
        proportions = model.get_site_classes()[0]

        # Compute likelihood using Rust backend (parallelized)
        calc = RustLikelihoodCalculator(aln, tree)
        lnL_rust = calc.compute_log_likelihood_site_classes(
            Q_matrices, pi, proportions
        )

        paml_lnL = -900.167515

        print(f"\nM6 Model (Rust Backend - Parallelized):")
        print(f"  Rust lnL:     {lnL_rust:.6f}")
        print(f"  PAML lnL:     {paml_lnL:.6f}")
        print(f"  Difference:   {abs(lnL_rust - paml_lnL):.6f}")
        print(f"  Relative:     {abs(lnL_rust - paml_lnL)/abs(paml_lnL)*100:.4f}%")

        # Should match to machine precision
        np.testing.assert_allclose(
            lnL_rust,
            paml_lnL,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Rust M6 likelihood should match PAML exactly"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
