"""
Tests for high-level API (optimize_model and ModelResult).
"""

import pytest
from pathlib import Path
import tempfile
import json

from crabml import optimize_model, ModelResult
from crabml.io import Alignment, Tree


@pytest.fixture
def lysozyme_files():
    """Path to lysozyme test data."""
    data_dir = Path(__file__).parent / "data" / "paml_reference" / "lysozyme"
    return {
        'alignment': data_dir / "lysozymeSmall.txt",
        'tree_str': "((Hsa_Human, Hla_gibbon), ((Cgu/Can_colobus, Pne_langur), Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));"
    }


class TestOptimizeModel:
    """Test optimize_model() function."""

    def test_m0_with_file_paths(self, lysozyme_files):
        """Test M0 optimization with file paths."""
        result = optimize_model(
            "M0",
            lysozyme_files['alignment'],
            lysozyme_files['tree_str']
        )

        assert isinstance(result, ModelResult)
        assert result.model_name == "M0"
        assert result.lnL < -900  # Should be around -906
        assert result.lnL > -910
        assert result.kappa > 1.0
        assert result.omega > 0
        assert 'omega' in result.params
        assert result.n_params > 10  # kappa + omega + branch lengths

    def test_m0_with_objects(self, lysozyme_files):
        """Test M0 with Alignment and Tree objects."""
        align = Alignment.from_phylip(str(lysozyme_files['alignment']), seqtype='codon')
        tree = Tree.from_newick(lysozyme_files['tree_str'])

        result = optimize_model("M0", align, tree)

        assert isinstance(result, ModelResult)
        assert result.lnL < -900
        assert result.alignment is align
        assert result.tree is tree

    def test_model_name_case_insensitive(self, lysozyme_files):
        """Test that model names are case-insensitive."""
        r1 = optimize_model("m0", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        r2 = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert r1.model_name == r2.model_name == "M0"
        # Log-likelihoods should be very close (within numerical precision)
        assert abs(r1.lnL - r2.lnL) < 0.01

    def test_invalid_model_name(self, lysozyme_files):
        """Test error handling for invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            optimize_model("M99", lysozyme_files['alignment'], lysozyme_files['tree_str'])

    def test_m1a_optimization(self, lysozyme_files):
        """Test M1a model returns correct structure."""
        result = optimize_model("M1a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.model_name == "M1A"
        assert result.lnL < -900
        assert 'p0' in result.params
        assert 'omega0' in result.params
        assert result.omegas == [result.params['omega0'], 1.0]
        assert len(result.proportions) == 2
        assert abs(sum(result.proportions) - 1.0) < 0.001  # Proportions sum to 1

    def test_m2a_optimization(self, lysozyme_files):
        """Test M2a model."""
        result = optimize_model("M2a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.model_name == "M2A"
        assert len(result.omegas) == 3
        assert len(result.proportions) == 3
        assert 'omega2' in result.params
        assert abs(sum(result.proportions) - 1.0) < 0.001

    def test_m7_optimization(self, lysozyme_files):
        """Test M7 model."""
        result = optimize_model("M7", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.model_name == "M7"
        assert 'p' in result.params
        assert 'q' in result.params
        assert result.params['p'] > 0
        assert result.params['q'] > 0

    def test_m8_optimization(self, lysozyme_files):
        """Test M8 model."""
        result = optimize_model("M8", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.model_name == "M8"
        assert 'omega_s' in result.params
        assert 'p0' in result.params
        assert result.params['omega_s'] > 1  # Should allow omega > 1
        assert len(result.proportions) == 2

    def test_m8a_optimization(self, lysozyme_files):
        """Test M8a model."""
        result = optimize_model("M8a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.model_name == "M8A"
        assert 'p' in result.params
        assert 'q' in result.params
        assert len(result.proportions) == 2

    def test_m3_optimization(self, lysozyme_files):
        """Test M3 model."""
        result = optimize_model("M3", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.model_name == "M3"
        assert 'omegas' in result.params
        assert 'proportions' in result.params
        assert len(result.params['omegas']) == 3  # Default K=3
        assert len(result.params['proportions']) == 3

    def test_custom_optimizer_kwargs(self, lysozyme_files):
        """Test passing custom kwargs to optimizer."""
        result = optimize_model(
            "M0",
            lysozyme_files['alignment'],
            lysozyme_files['tree_str'],
            maxiter=100,
            init_kappa=3.0
        )

        assert isinstance(result, ModelResult)
        assert result.lnL < -900


class TestModelResult:
    """Test ModelResult class."""

    def test_summary_display(self, lysozyme_files):
        """Test summary() method."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        summary = result.summary()

        assert "MODEL: M0" in summary
        assert "Log-likelihood:" in summary
        assert "kappa" in summary
        assert "omega" in summary
        assert "TREE:" in summary

    def test_m1a_summary(self, lysozyme_files):
        """Test M1a summary shows site classes."""
        result = optimize_model("M1a", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        summary = result.summary()

        assert "MODEL: M1A" in summary
        assert "Site classes:" in summary
        assert "Class 0:" in summary
        assert "Class 1:" in summary

    def test_m7_summary(self, lysozyme_files):
        """Test M7 summary shows beta distribution."""
        result = optimize_model("M7", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        summary = result.summary()

        assert "Beta distribution:" in summary
        assert "p =" in summary
        assert "q =" in summary

    def test_to_dict_export(self, lysozyme_files):
        """Test dictionary export."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        d = result.to_dict()

        assert d['model_name'] == 'M0'
        assert 'lnL' in d
        assert 'kappa' in d
        assert 'params' in d
        assert isinstance(d['lnL'], float)
        assert isinstance(d['kappa'], float)
        assert isinstance(d['n_params'], int)

    def test_to_json_export_string(self, lysozyme_files):
        """Test JSON export to string."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        json_str = result.to_json()
        data = json.loads(json_str)

        assert data['model_name'] == 'M0'
        assert 'lnL' in data
        assert 'params' in data
        assert data['params']['omega'] > 0

    def test_to_json_export_file(self, lysozyme_files):
        """Test JSON export to file."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            result.to_json(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert data['model_name'] == 'M0'
            assert data['lnL'] < -900
        finally:
            Path(filepath).unlink()

    def test_repr_and_str(self, lysozyme_files):
        """Test string representations."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        repr_str = repr(result)
        assert "ModelResult" in repr_str
        assert "M0" in repr_str
        assert "lnL=" in repr_str

        str_output = str(result)
        assert "MODEL: M0" in str_output
        assert "Log-likelihood:" in str_output

    def test_omega_property_m0(self, lysozyme_files):
        """Test .omega property for M0."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.omega is not None
        assert result.omega > 0
        assert result.omega == result.params['omega']

    def test_omega_property_m1a(self, lysozyme_files):
        """Test .omega is None for M1a (use .omegas instead)."""
        result = optimize_model("M1a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.omega is None
        assert result.omegas is not None
        assert len(result.omegas) == 2

    def test_omegas_property(self, lysozyme_files):
        """Test .omegas property."""
        result = optimize_model("M2a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.omegas is not None
        assert len(result.omegas) == 3
        assert result.omegas[0] < 1  # omega0 < 1
        assert result.omegas[1] == 1.0  # omega1 = 1
        # omega2 can be > or < 1 depending on data

    def test_proportions_property(self, lysozyme_files):
        """Test .proportions property."""
        result = optimize_model("M2a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        assert result.proportions is not None
        assert len(result.proportions) == 3
        assert abs(sum(result.proportions) - 1.0) < 0.001

    def test_n_site_classes(self, lysozyme_files):
        """Test .n_site_classes property."""
        m0 = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        assert m0.n_site_classes == 1

        m1a = optimize_model("M1a", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        assert m1a.n_site_classes == 2

        m2a = optimize_model("M2a", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        assert m2a.n_site_classes == 3


class TestFileFormatDetection:
    """Test automatic file format detection."""

    def test_phylip_detection(self, lysozyme_files):
        """Test PHYLIP format loading (lysozyme is PHYLIP)."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        assert result.alignment is not None
        assert len(result.alignment.sequences) > 0

    def test_invalid_alignment_path(self, lysozyme_files):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            optimize_model("M0", "/nonexistent/file.fasta", lysozyme_files['tree_str'])

    def test_tree_from_string(self, lysozyme_files):
        """Test tree loading from Newick string."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])
        assert result.tree is not None
        assert result.tree.n_leaves > 0


class TestParameterCounting:
    """Test that n_params is correctly counted for each model."""

    def test_m0_param_count(self, lysozyme_files):
        """M0 should have kappa + omega + n_branches."""
        result = optimize_model("M0", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        # Count branches
        n_branches = sum(1 for node in result.tree.postorder() if node.parent is not None)

        expected_params = 2 + n_branches  # kappa + omega + branches
        assert result.n_params == expected_params

    def test_m1a_param_count(self, lysozyme_files):
        """M1a should have kappa + p0 + omega0 + n_branches."""
        result = optimize_model("M1a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        n_branches = sum(1 for node in result.tree.postorder() if node.parent is not None)

        expected_params = 3 + n_branches  # kappa + p0 + omega0 + branches
        assert result.n_params == expected_params

    def test_m2a_param_count(self, lysozyme_files):
        """M2a should have kappa + p0 + p1 + omega0 + omega2 + n_branches."""
        result = optimize_model("M2a", lysozyme_files['alignment'], lysozyme_files['tree_str'])

        n_branches = sum(1 for node in result.tree.postorder() if node.parent is not None)

        expected_params = 5 + n_branches  # kappa + p0 + p1 + omega0 + omega2 + branches
        assert result.n_params == expected_params


class TestBackwardsCompatibility:
    """Test that existing optimizer interface still works."""

    def test_direct_optimizer_use(self, lysozyme_files):
        """Test that direct optimizer access still works."""
        from crabml.optimize.optimizer import M0Optimizer

        align = Alignment.from_phylip(str(lysozyme_files['alignment']), seqtype='codon')
        tree = Tree.from_newick(lysozyme_files['tree_str'])

        optimizer = M0Optimizer(align, tree, use_f3x4=True)
        kappa, omega, lnL = optimizer.optimize()

        assert kappa > 0
        assert omega > 0
        assert lnL < -900


class TestBranchModels:
    """Test optimize_branch_model() function."""

    @pytest.fixture
    def branch_files(self):
        """Path to branch model test data."""
        data_dir = Path(__file__).parent / "data" / "paml_reference" / "branch_model"
        return {
            'alignment': data_dir / "lysozymeSmall.txt",
            'tree_str': "((Hsa_Human, Hla_gibbon) #1, ((Cgu/Can_colobus, Pne_langur), Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));",
        }

    def test_multi_ratio_optimization(self, branch_files):
        """Test multi-ratio branch model."""
        from crabml import optimize_branch_model, BranchModelResult

        result = optimize_branch_model(
            "multi-ratio",
            branch_files['alignment'],
            branch_files['tree_str']
        )

        assert isinstance(result, BranchModelResult)
        assert result.model_name == "Multi-ratio"
        assert result.lnL < -900
        assert result.kappa > 1.0

        # Should have omega0 (background) and omega1 (foreground)
        omega_dict = result.omega_dict
        assert 'omega0' in omega_dict
        assert 'omega1' in omega_dict
        assert result.background_omega == omega_dict['omega0']
        assert result.foreground_omega == omega_dict['omega1']

    def test_branch_model_result_summary(self, branch_files):
        """Test BranchModelResult summary display."""
        from crabml import optimize_branch_model

        result = optimize_branch_model("multi-ratio", branch_files['alignment'], branch_files['tree_str'])
        summary = result.summary()

        assert "Multi-ratio" in summary
        assert "Log-likelihood:" in summary
        assert "omega0" in summary
        assert "omega1" in summary

    def test_branch_model_export(self, branch_files):
        """Test BranchModelResult export methods."""
        from crabml import optimize_branch_model

        result = optimize_branch_model("multi-ratio", branch_files['alignment'], branch_files['tree_str'])

        # Test to_dict
        d = result.to_dict()
        assert 'omega_dict' in d['params']
        assert d['model_name'] == 'Multi-ratio'

        # Test to_json
        json_str = result.to_json()
        import json
        data = json.loads(json_str)
        assert data['model_name'] == 'Multi-ratio'


class TestBranchSiteModels:
    """Test optimize_branch_site_model() function."""

    @pytest.fixture
    def branch_site_files(self):
        """Path to branch-site model test data."""
        data_dir = Path(__file__).parent / "data" / "paml_reference" / "branch_model"
        return {
            'alignment': data_dir / "lysozymeSmall.txt",
            'tree_str': "((Hsa_Human, Hla_gibbon) #1, ((Cgu/Can_colobus, Pne_langur), Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));",
        }

    def test_branch_site_model_a(self, branch_site_files):
        """Test Branch-Site Model A."""
        from crabml import optimize_branch_site_model, BranchSiteModelResult

        result = optimize_branch_site_model(
            "model-a",
            branch_site_files['alignment'],
            branch_site_files['tree_str']
        )

        assert isinstance(result, BranchSiteModelResult)
        assert "Branch-Site Model A" in result.model_name
        assert result.lnL < -900
        assert result.kappa > 1.0

        # Check site-class parameters
        assert result.omega0 > 0
        assert result.omega0 < 1  # Conserved class
        assert result.omega2 >= 0  # Can be > or < 1

        # Check proportions
        props = result.proportions
        assert len(props) == 4  # p0, p1, p2a, p2b
        assert abs(sum(props) - 1.0) < 0.001

    def test_branch_site_null_model(self, branch_site_files):
        """Test Branch-Site Model A with omega2 fixed."""
        from crabml import optimize_branch_site_model

        result = optimize_branch_site_model(
            "model-a",
            branch_site_files['alignment'],
            branch_site_files['tree_str'],
            fix_omega=True
        )

        assert result.omega2 == 1.0
        assert "null" in result.model_name.lower()

    def test_branch_site_result_properties(self, branch_site_files):
        """Test BranchSiteModelResult properties."""
        from crabml import optimize_branch_site_model

        result = optimize_branch_site_model("model-a", branch_site_files['alignment'], branch_site_files['tree_str'])

        # Test foreground_positive_proportion
        fg_prop = result.foreground_positive_proportion
        assert fg_prop >= 0
        assert fg_prop <= 1
        assert abs(fg_prop - (result.proportions[2] + result.proportions[3])) < 0.001

    def test_branch_site_summary(self, branch_site_files):
        """Test BranchSiteModelResult summary display."""
        from crabml import optimize_branch_site_model

        result = optimize_branch_site_model("model-a", branch_site_files['alignment'], branch_site_files['tree_str'])
        summary = result.summary()

        assert "Branch-Site Model A" in summary
        assert "Site classes" in summary
        assert "ω₀" in summary or "omega0" in summary
        assert "ω₂" in summary or "omega2" in summary
        assert "Foreground positive selection" in summary

    def test_branch_site_export(self, branch_site_files):
        """Test BranchSiteModelResult export methods."""
        from crabml import optimize_branch_site_model

        result = optimize_branch_site_model("model-a", branch_site_files['alignment'], branch_site_files['tree_str'])

        # Test to_dict
        d = result.to_dict()
        assert 'omega0' in d['params']
        assert 'omega2' in d['params']
        assert 'p0' in d['params']
        assert 'p1' in d['params']

        # Test to_json
        json_str = result.to_json()
        import json
        data = json.loads(json_str)
        assert "Branch-Site Model A" in data['model_name']


class TestModelResultBackwardsCompatibility:
    """Test that ModelResult alias works correctly."""

    def test_modelresult_alias(self):
        """Test that ModelResult is an alias for SiteModelResult."""
        from crabml import ModelResult, SiteModelResult

        assert ModelResult is SiteModelResult

    def test_modelresult_import(self):
        """Test that ModelResult can still be imported and used."""
        from crabml import optimize_model, ModelResult

        lysozyme_dir = Path(__file__).parent / "data" / "paml_reference" / "lysozyme"
        result = optimize_model(
            "M0",
            lysozyme_dir / "lysozymeSmall.txt",
            "((Hsa_Human, Hla_gibbon), ((Cgu/Can_colobus, Pne_langur), Mmu_rhesus), (Ssc_squirrelM, Cja_marmoset));"
        )

        # Should be instance of both ModelResult and SiteModelResult
        assert isinstance(result, ModelResult)
        from crabml import SiteModelResult
        assert isinstance(result, SiteModelResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
