"""Tests for site-class codon simulators (M1a, M2a, M7, M8, M8a)."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from crabml.simulate.codon import M1aSimulator, M2aSimulator, M7Simulator, M8Simulator, M8aSimulator
from crabml.simulate.output import SimulationOutput
from crabml.io.trees import Tree


@pytest.fixture
def simple_tree():
    """Create a simple 4-taxon tree."""
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1):0.0;"
    return Tree.from_newick(newick)


@pytest.fixture
def codon_freqs():
    """Uniform codon frequencies."""
    return np.ones(61) / 61


class TestM1aSimulator:
    """Tests for M1a (nearly neutral) simulator."""

    def test_initialization(self, simple_tree, codon_freqs):
        """Test simulator can be initialized with valid parameters."""
        simulator = M1aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.7,
            omega0=0.1,
            codon_freqs=codon_freqs,
            seed=42
        )
        assert simulator.sequence_length == 100
        assert simulator.kappa == 2.0
        assert len(simulator.site_class_ids) == 100

    def test_invalid_parameters(self, simple_tree, codon_freqs):
        """Test that invalid parameters raise errors."""
        # omega0 must be < 1
        with pytest.raises(ValueError):
            M1aSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                p0=0.7,
                omega0=1.5,
                codon_freqs=codon_freqs
            )

        # p0 must be in (0, 1)
        with pytest.raises(ValueError):
            M1aSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                p0=1.5,
                omega0=0.1,
                codon_freqs=codon_freqs
            )

    def test_site_class_proportions(self, simple_tree, codon_freqs):
        """Test that site class proportions match expected values."""
        p0 = 0.7
        simulator = M1aSimulator(
            tree=simple_tree,
            sequence_length=10000,  # Large sample for accurate proportions
            kappa=2.0,
            p0=p0,
            omega0=0.1,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        site_classes = np.array(site_info['site_class_ids'])

        # Count sites in each class
        class0_count = np.sum(site_classes == 0)
        class1_count = np.sum(site_classes == 1)

        # Check proportions (with tolerance for randomness)
        assert abs(class0_count / 10000 - p0) < 0.02
        assert abs(class1_count / 10000 - (1 - p0)) < 0.02

    def test_simulate(self, simple_tree, codon_freqs):
        """Test that simulation produces valid output."""
        simulator = M1aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.7,
            omega0=0.1,
            codon_freqs=codon_freqs,
            seed=42
        )

        sequences = simulator.simulate()

        # Check we have sequences for all taxa
        assert len(sequences) == 4
        assert all(name in sequences for name in ['A', 'B', 'C', 'D'])

        # Check sequence lengths are correct (100 codons = 100 codon indices)
        # Sequences are numpy arrays of codon indices
        for seq in sequences.values():
            assert len(seq) == 100

    def test_reproducibility(self, simple_tree, codon_freqs):
        """Test that same seed produces same results."""
        sim1 = M1aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.7,
            omega0=0.1,
            codon_freqs=codon_freqs,
            seed=42
        )
        sequences1 = sim1.simulate()

        sim2 = M1aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.7,
            omega0=0.1,
            codon_freqs=codon_freqs,
            seed=42
        )
        sequences2 = sim2.simulate()

        # Check sequences are identical
        for taxon in sequences1:
            # Compare numpy arrays
            assert np.array_equal(sequences1[taxon], sequences2[taxon])


class TestM2aSimulator:
    """Tests for M2a (positive selection) simulator."""

    def test_initialization(self, simple_tree, codon_freqs):
        """Test simulator can be initialized with valid parameters."""
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.5,
            p1=0.3,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )
        assert simulator.sequence_length == 100
        assert len(simulator.site_class_ids) == 100

    def test_invalid_parameters(self, simple_tree, codon_freqs):
        """Test that invalid parameters raise errors."""
        # omega0 must be < 1
        with pytest.raises(ValueError):
            M2aSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                p0=0.5,
                p1=0.3,
                omega0=1.5,
                omega2=2.5,
                codon_freqs=codon_freqs
            )

        # omega2 must be > 1
        with pytest.raises(ValueError):
            M2aSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                p0=0.5,
                p1=0.3,
                omega0=0.1,
                omega2=0.5,
                codon_freqs=codon_freqs
            )

        # p0 + p1 must be < 1
        with pytest.raises(ValueError):
            M2aSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                p0=0.7,
                p1=0.5,
                omega0=0.1,
                omega2=2.5,
                codon_freqs=codon_freqs
            )

    def test_site_class_proportions(self, simple_tree, codon_freqs):
        """Test that site class proportions match expected values."""
        p0, p1 = 0.5, 0.3
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=10000,
            kappa=2.0,
            p0=p0,
            p1=p1,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        site_classes = np.array(site_info['site_class_ids'])

        # Count sites in each class
        class0_count = np.sum(site_classes == 0)
        class1_count = np.sum(site_classes == 1)
        class2_count = np.sum(site_classes == 2)

        # Check proportions
        assert abs(class0_count / 10000 - p0) < 0.02
        assert abs(class1_count / 10000 - p1) < 0.02
        assert abs(class2_count / 10000 - (1 - p0 - p1)) < 0.02

    def test_positive_selection_sites(self, simple_tree, codon_freqs):
        """Test that positively selected sites have omega > 1."""
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=10000,  # Larger sample for better proportion estimate
            kappa=2.0,
            p0=0.5,
            p1=0.3,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()

        # Use the precomputed positively_selected_sites
        ps_sites = site_info['positively_selected_sites']

        # Should be roughly 20% of sites (1 - p0 - p1)
        expected_fraction = 1 - 0.5 - 0.3
        assert abs(len(ps_sites) / 10000 - expected_fraction) < 0.02

    def test_simulate(self, simple_tree, codon_freqs):
        """Test that simulation produces valid output."""
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.5,
            p1=0.3,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )

        sequences = simulator.simulate()

        # Check we have sequences for all taxa
        assert len(sequences) == 4
        assert all(name in sequences for name in ['A', 'B', 'C', 'D'])

        # Check sequence lengths (100 codons = 100 codon indices)
        for seq in sequences.values():
            assert len(seq) == 100


class TestM7Simulator:
    """Tests for M7 (beta) simulator."""

    def test_initialization(self, simple_tree, codon_freqs):
        """Test simulator can be initialized with valid parameters."""
        simulator = M7Simulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p=2.0,
            q=5.0,
            n_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )
        assert simulator.sequence_length == 100
        assert len(simulator.site_class_ids) == 100

    def test_beta_discretization(self, simple_tree, codon_freqs):
        """Test that beta distribution is discretized correctly."""
        n_categories = 10
        simulator = M7Simulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p=2.0,
            q=5.0,
            n_categories=n_categories,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        omegas = np.unique(site_info['site_class_omegas'])

        # Should have exactly n_categories distinct omega values
        assert len(omegas) == n_categories

        # All omegas should be in (0, 1)
        assert np.all(omegas > 0.0)
        assert np.all(omegas < 1.0)

    def test_site_class_uniform_proportions(self, simple_tree, codon_freqs):
        """Test that site classes have uniform proportions."""
        n_categories = 10
        simulator = M7Simulator(
            tree=simple_tree,
            sequence_length=10000,
            kappa=2.0,
            p=2.0,
            q=5.0,
            n_categories=n_categories,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        site_classes = np.array(site_info['site_class_ids'])

        # Count sites in each class
        for class_id in range(n_categories):
            count = np.sum(site_classes == class_id)
            expected = 1.0 / n_categories
            assert abs(count / 10000 - expected) < 0.02

    def test_simulate(self, simple_tree, codon_freqs):
        """Test that simulation produces valid output."""
        simulator = M7Simulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p=2.0,
            q=5.0,
            n_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        sequences = simulator.simulate()

        # Check we have sequences for all taxa
        assert len(sequences) == 4
        assert all(name in sequences for name in ['A', 'B', 'C', 'D'])

        # Check sequence lengths (100 codons = 100 codon indices)
        for seq in sequences.values():
            assert len(seq) == 100


class TestM8Simulator:
    """Tests for M8 (beta + omega) simulator."""

    def test_initialization(self, simple_tree, codon_freqs):
        """Test simulator can be initialized with valid parameters."""
        simulator = M8Simulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            omega_s=2.5,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )
        assert simulator.sequence_length == 100
        assert len(simulator.site_class_ids) == 100

    def test_invalid_parameters(self, simple_tree, codon_freqs):
        """Test that invalid parameters raise errors."""
        # omega_s must be > 1
        with pytest.raises(ValueError):
            M8Simulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                p0=0.8,
                p=2.0,
                q=5.0,
                omega_s=0.5,
                n_beta_categories=10,
                codon_freqs=codon_freqs
            )

    def test_omega_classes(self, simple_tree, codon_freqs):
        """Test that omega classes include beta + selection class."""
        n_beta_categories = 10
        omega_s = 2.5
        simulator = M8Simulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            omega_s=omega_s,
            n_beta_categories=n_beta_categories,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        omegas = np.unique(site_info['site_class_omegas'])

        # Should have n_beta_categories + 1 distinct omega values (beta classes + selection)
        assert len(omegas) == n_beta_categories + 1

        # Check that one omega is the selection class (omega_s)
        assert np.any(np.isclose(omegas, omega_s))

        # Check that other omegas are in (0, 1)
        beta_omegas = omegas[~np.isclose(omegas, omega_s)]
        assert np.all(beta_omegas > 0.0)
        assert np.all(beta_omegas < 1.0)

    def test_selection_class_proportion(self, simple_tree, codon_freqs):
        """Test that selection class has expected proportion."""
        p0 = 0.8
        simulator = M8Simulator(
            tree=simple_tree,
            sequence_length=10000,
            kappa=2.0,
            p0=p0,
            p=2.0,
            q=5.0,
            omega_s=2.5,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()

        # Use the precomputed positively_selected_sites
        ps_sites = site_info['positively_selected_sites']
        selection_count = len(ps_sites)

        # Should be roughly (1 - p0)
        expected_fraction = 1 - p0
        assert abs(selection_count / 10000 - expected_fraction) < 0.02

    def test_simulate(self, simple_tree, codon_freqs):
        """Test that simulation produces valid output."""
        simulator = M8Simulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            omega_s=2.5,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        sequences = simulator.simulate()

        # Check we have sequences for all taxa
        assert len(sequences) == 4
        assert all(name in sequences for name in ['A', 'B', 'C', 'D'])

        # Check sequence lengths (100 codons = 100 codon indices)
        for seq in sequences.values():
            assert len(seq) == 100


class TestM8aSimulator:
    """Tests for M8a (beta + neutral) simulator."""

    def test_initialization(self, simple_tree, codon_freqs):
        """Test simulator can be initialized with valid parameters."""
        simulator = M8aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )
        assert simulator.sequence_length == 100
        assert len(simulator.site_class_ids) == 100
        assert simulator.omega_s == 1.0  # Always fixed to 1.0 for M8a

    def test_omega_s_is_fixed_to_one(self, simple_tree, codon_freqs):
        """Test that omega_s is always 1.0 (neutral) in M8a."""
        simulator = M8aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        # Check that omega_s is 1.0
        assert simulator.omega_s == 1.0

        # Check that parameters reflect this
        params = simulator.get_parameters()
        assert params['omega_s'] == 1.0
        assert params['model'] == 'M8a'

    def test_omega_classes(self, simple_tree, codon_freqs):
        """Test that omega classes include beta + neutral class."""
        n_beta_categories = 10
        simulator = M8aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=n_beta_categories,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        omegas = np.unique(site_info['site_class_omegas'])

        # Should have n_beta_categories + 1 distinct omega values (beta classes + neutral)
        assert len(omegas) == n_beta_categories + 1

        # Check that one omega is the neutral class (omega_s = 1.0)
        assert np.any(np.isclose(omegas, 1.0))

        # Check that other omegas are in (0, 1)
        beta_omegas = omegas[~np.isclose(omegas, 1.0)]
        assert np.all(beta_omegas > 0.0)
        assert np.all(beta_omegas < 1.0)

    def test_neutral_class_proportion(self, simple_tree, codon_freqs):
        """Test that neutral class has expected proportion."""
        p0 = 0.8
        simulator = M8aSimulator(
            tree=simple_tree,
            sequence_length=10000,
            kappa=2.0,
            p0=p0,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()
        site_class_ids = np.array(site_info['site_class_ids'])
        site_class_omegas = np.array(site_info['site_class_omegas'])

        # Get omega for each site using site_class_ids as index
        site_omegas = site_class_omegas[site_class_ids.astype(int)]

        # Count sites with omega = 1.0 (neutral class)
        neutral_count = np.sum(np.isclose(site_omegas, 1.0))

        # Should be roughly (1 - p0)
        expected_fraction = 1 - p0
        assert abs(neutral_count / 10000 - expected_fraction) < 0.02

    def test_no_positive_selection(self, simple_tree, codon_freqs):
        """Test that M8a has no positively selected sites (omega=1.0, not >1)."""
        simulator = M8aSimulator(
            tree=simple_tree,
            sequence_length=10000,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()

        # M8a should have no positively selected sites
        # (all omegas <= 1, with the extra class exactly = 1)
        ps_sites = site_info['positively_selected_sites']
        assert len(ps_sites) == 0

    def test_simulate(self, simple_tree, codon_freqs):
        """Test that simulation produces valid output."""
        simulator = M8aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )

        sequences = simulator.simulate()

        # Check we have sequences for all taxa
        assert len(sequences) == 4
        assert all(name in sequences for name in ['A', 'B', 'C', 'D'])

        # Check sequence lengths (100 codons = 100 codon indices)
        for seq in sequences.values():
            assert len(seq) == 100

    def test_reproducibility(self, simple_tree, codon_freqs):
        """Test that same seed produces same results."""
        sim1 = M8aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )
        sequences1 = sim1.simulate()

        sim2 = M8aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.8,
            p=2.0,
            q=5.0,
            n_beta_categories=10,
            codon_freqs=codon_freqs,
            seed=42
        )
        sequences2 = sim2.simulate()

        # Check sequences are identical
        for taxon in sequences1:
            assert np.array_equal(sequences1[taxon], sequences2[taxon])


class TestOutputFormats:
    """Tests for output file formats."""

    def test_site_classes_output(self, simple_tree, codon_freqs):
        """Test site class output format."""
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.5,
            p1=0.3,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "site_classes.txt"
            SimulationOutput.write_site_classes(
                np.array(site_info['site_class_ids']),
                np.array(site_info['site_class_omegas']),
                out_path
            )

            # Check file was created and has content
            assert out_path.exists()
            content = out_path.read_text()
            assert "site_id" in content
            assert "class_id" in content
            assert "omega" in content

    def test_positive_sites_output(self, simple_tree, codon_freqs):
        """Test positive selection sites output."""
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.5,
            p1=0.3,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )

        site_info = simulator.get_site_classes()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "positive_sites.txt"
            SimulationOutput.write_positively_selected_sites(
                np.array(site_info['site_class_ids']),
                np.array(site_info['site_class_omegas']),
                out_path
            )

            # Check file was created
            assert out_path.exists()
            content = out_path.read_text()

            # Check that only sites with omega > 1 are listed
            lines = [l for l in content.split('\n') if l.strip() and not l.startswith('#')]
            for line in lines:
                if '\t' in line:
                    parts = line.split('\t')
                    omega = float(parts[2])
                    assert omega > 1.0

    def test_parameters_output(self, simple_tree, codon_freqs):
        """Test parameters JSON output."""
        simulator = M2aSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            p0=0.5,
            p1=0.3,
            omega0=0.1,
            omega2=2.5,
            codon_freqs=codon_freqs,
            seed=42
        )

        params = simulator.get_parameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "params.json"
            SimulationOutput.write_parameters(params, out_path)

            # Check file was created
            assert out_path.exists()

            # Check JSON is valid and contains expected keys
            with open(out_path) as f:
                loaded_params = json.load(f)

            assert 'model' in loaded_params
            assert loaded_params['model'] == 'M2a'
            assert 'kappa' in loaded_params
            assert 'p0' in loaded_params
            assert 'p1' in loaded_params
