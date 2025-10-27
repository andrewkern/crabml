"""Tests for M0 codon simulator."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from crabml.simulate.codon import M0CodonSimulator
from crabml.simulate.output import SimulationOutput
from crabml.io.trees import Tree


class TestM0Simulator:
    """Test suite for M0CodonSimulator."""

    @pytest.fixture
    def simple_tree(self):
        """Create a simple tree for testing."""
        return Tree.from_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.05);")

    @pytest.fixture
    def uniform_freqs(self):
        """Uniform codon frequencies."""
        return np.ones(61) / 61

    def test_initialization(self, simple_tree, uniform_freqs):
        """Test simulator can be initialized."""
        sim = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        assert sim.sequence_length == 100
        assert sim.kappa == 2.0
        assert sim.omega == 0.5
        assert len(sim.codon_freqs) == 61

    def test_invalid_codon_freqs(self, simple_tree):
        """Test that invalid codon frequencies raise error."""
        # Wrong length
        with pytest.raises(ValueError, match="must have length 61"):
            M0CodonSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                omega=0.5,
                codon_freqs=np.ones(60) / 60,  # Wrong length
                seed=42
            )

        # Doesn't sum to 1
        with pytest.raises(ValueError, match="must sum to 1"):
            M0CodonSimulator(
                tree=simple_tree,
                sequence_length=100,
                kappa=2.0,
                omega=0.5,
                codon_freqs=np.ones(61) / 100,  # Doesn't sum to 1
                seed=42
            )

    def test_ancestral_sequence_frequencies(self, simple_tree):
        """Test that ancestral sequence follows equilibrium frequencies."""
        # Use large sequence for statistical test
        freqs = np.ones(61) / 61

        sim = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=10000,  # Large for good statistics
            kappa=2.0,
            omega=0.5,
            codon_freqs=freqs,
            seed=42
        )

        root_seq = sim._generate_ancestral_sequence()

        # Check frequencies match (approximately)
        observed_freqs = np.bincount(root_seq, minlength=61) / len(root_seq)

        # Should be close to uniform with large sample
        assert np.allclose(observed_freqs, freqs, atol=0.01)

    def test_simulate_returns_correct_species(self, simple_tree, uniform_freqs):
        """Test that simulate returns sequences for all tips."""
        sim = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        sequences = sim.simulate()

        # Should have 4 tip sequences (A, B, C, D)
        assert len(sequences) == 4
        assert 'A' in sequences
        assert 'B' in sequences
        assert 'C' in sequences
        assert 'D' in sequences

        # Each sequence should have correct length
        for seq in sequences.values():
            assert len(seq) == 100
            # All values should be valid codon indices (0-60)
            assert np.all(seq >= 0)
            assert np.all(seq < 61)

    def test_reproducibility(self, simple_tree, uniform_freqs):
        """Test that same seed produces identical results."""
        sim1 = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        sim2 = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        seqs1 = sim1.simulate()
        seqs2 = sim2.simulate()

        # Same seed should produce identical sequences
        for species in seqs1:
            assert np.array_equal(seqs1[species], seqs2[species])

    def test_different_seeds_differ(self, simple_tree, uniform_freqs):
        """Test that different seeds produce different results."""
        sim1 = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        sim2 = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=123
        )

        seqs1 = sim1.simulate()
        seqs2 = sim2.simulate()

        # Different seeds should produce different sequences
        # Check at least one sequence differs
        differs = False
        for species in seqs1:
            if not np.array_equal(seqs1[species], seqs2[species]):
                differs = True
                break
        assert differs

    def test_no_evolution_zero_branch(self, uniform_freqs):
        """Test that very short branches result in minimal evolution."""
        # Tree with very short branches
        tree = Tree.from_newick("(A:0.0001,B:0.0001);")

        sim = M0CodonSimulator(
            tree=tree,
            sequence_length=1000,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        seqs = sim.simulate()

        # With very short branches, sequences should be very similar
        identity = np.mean(seqs['A'] == seqs['B'])
        assert identity > 0.95  # Expect >95% identity

    def test_get_parameters(self, simple_tree, uniform_freqs):
        """Test that get_parameters returns correct info."""
        sim = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=500,
            kappa=2.5,
            omega=0.3,
            codon_freqs=uniform_freqs,
            seed=42
        )

        params = sim.get_parameters()

        assert params['model'] == 'M0'
        assert params['kappa'] == 2.5
        assert params['omega'] == 0.3
        assert params['sequence_length'] == 500
        assert params['genetic_code'] == 0
        assert 'tree_length' in params

    def test_output_fasta(self, simple_tree, uniform_freqs):
        """Test that FASTA output works."""
        sim = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=50,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        sequences = sim.simulate()

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            temp_path = Path(f.name)

        try:
            SimulationOutput.write_fasta(sequences, temp_path)

            # Check file exists and has content
            assert temp_path.exists()
            content = temp_path.read_text()

            # Should have 4 sequences (A, B, C, D)
            assert content.count('>') == 4
            assert '>A' in content
            assert '>B' in content
            assert '>C' in content
            assert '>D' in content

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_output_parameters(self, simple_tree, uniform_freqs):
        """Test that parameter output works."""
        sim = M0CodonSimulator(
            tree=simple_tree,
            sequence_length=100,
            kappa=2.0,
            omega=0.5,
            codon_freqs=uniform_freqs,
            seed=42
        )

        params = sim.get_parameters()

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            SimulationOutput.write_parameters(params, temp_path)

            # Check file exists
            assert temp_path.exists()

            # Read back and verify
            import json
            with open(temp_path) as f:
                loaded_params = json.load(f)

            assert loaded_params['model'] == 'M0'
            assert loaded_params['kappa'] == 2.0
            assert loaded_params['omega'] == 0.5

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


@pytest.mark.slow
class TestM0ParameterRecovery:
    """
    Integration tests for parameter recovery.

    These tests simulate sequences and verify that crabML can recover
    the original parameters. Marked as slow because they run full optimization.
    """

    def test_recovery_simple_case(self):
        """
        Test that crabML recovers known parameters from simulated data.

        This is a critical validation test!
        """
        from crabml import optimize_model

        # Known parameters
        true_omega = 0.5
        true_kappa = 2.5

        # Create tree
        tree = Tree.from_newick("((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.05);")

        # Simulate
        sim = M0CodonSimulator(
            tree=tree,
            sequence_length=1000,  # Long for good estimates
            kappa=true_kappa,
            omega=true_omega,
            codon_freqs=np.ones(61) / 61,
            seed=42
        )

        sequences = sim.simulate()

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            temp_path = Path(f.name)

        try:
            SimulationOutput.write_fasta(sequences, temp_path)

            # Fit with crabML
            result = optimize_model('M0', str(temp_path), tree)

            # Verify recovery (with tolerance for stochastic variation)
            assert abs(result.omega - true_omega) < 0.15, \
                f"omega: {result.omega} vs true {true_omega}"
            assert abs(result.kappa - true_kappa) < 0.8, \
                f"kappa: {result.kappa} vs true {true_kappa}"

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
