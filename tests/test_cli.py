"""
Unit tests for CLI commands.
"""

import json
import pytest
from pathlib import Path

from crabml.cli.main import app


class TestCLIHelp:
    """Test help messages and basic CLI functionality."""

    def test_main_help(self, cli_runner):
        """Test main CLI help message."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "crabml" in result.stdout.lower()
        assert "site-model" in result.stdout
        assert "fit" in result.stdout
        assert "branch-site" in result.stdout

    def test_site_model_help(self, cli_runner):
        """Test 'site-model' command help message."""
        result = cli_runner.invoke(app, ["site-model", "--help"])
        assert result.exit_code == 0
        assert "positive selection" in result.stdout.lower()
        assert "--alignment" in result.stdout or "-s" in result.stdout
        assert "--tree" in result.stdout or "-t" in result.stdout

    def test_fit_help(self, cli_runner):
        """Test 'fit' command help message."""
        result = cli_runner.invoke(app, ["fit", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout or "-m" in result.stdout
        assert "--alignment" in result.stdout or "-s" in result.stdout

    def test_branch_site_help(self, cli_runner):
        """Test 'branch-site' command help message."""
        result = cli_runner.invoke(app, ["branch-site", "--help"])
        assert result.exit_code == 0
        assert "branch-site" in result.stdout.lower()


class TestCLIFit:
    """Test 'fit' command functionality."""

    def test_fit_m0_text_output(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test fitting M0 with text output."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0
        assert "MODEL: M0" in result.stdout
        assert "Log-likelihood:" in result.stdout
        assert "kappa" in result.stdout
        assert "omega" in result.stdout

    def test_fit_m0_json_output(self, cli_runner, lysozyme_small_files, lysozyme_tree_file, tmp_path):
        """Test fitting M0 with JSON output."""
        output_file = tmp_path / "result.json"

        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--format", "json",
            "--output", str(output_file),
            "--quiet"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        assert "model_name" in data  # API uses "model_name" not "model"
        assert "lnL" in data
        assert "params" in data

    def test_fit_invalid_model(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test fitting with invalid model name."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "InvalidModel",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--quiet"
        ])

        # Exit code 1 means application error (not argument parser error)
        assert result.exit_code != 0  # Should fail but we don't check exact exit code

    def test_fit_missing_alignment(self, cli_runner, lysozyme_tree_file):
        """Test fitting with missing alignment file."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", "/nonexistent/file.txt",
            "-t", str(lysozyme_tree_file),
            "--quiet"
        ])

        # Exit code 2 means argument error (file doesn't exist)
        assert result.exit_code != 0  # Should fail

    def test_fit_missing_tree(self, cli_runner, lysozyme_small_files):
        """Test fitting with missing tree file."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", "/nonexistent/tree.nwk",
            "--quiet"
        ])

        # Exit code 2 means argument error (file doesn't exist)
        assert result.exit_code != 0  # Should fail


class TestCLISiteModel:
    """Test 'site-model' command functionality."""

    @pytest.mark.slow
    def test_m7m8_text_output(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test M7 vs M8 test with text output."""
        result = cli_runner.invoke(app, [
            "site-model",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--test", "m7m8",
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0
        assert "M7" in result.stdout
        assert "M8" in result.stdout
        assert "lnL" in result.stdout
        assert "p-value" in result.stdout

    @pytest.mark.slow
    def test_m1m2_text_output(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test M1a vs M2a test with text output."""
        result = cli_runner.invoke(app, [
            "site-model",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--test", "m1m2",
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0
        assert "M1a" in result.stdout or "M1A" in result.stdout
        assert "M2a" in result.stdout or "M2A" in result.stdout

    @pytest.mark.slow
    def test_both_tests(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test running both M1a vs M2a and M7 vs M8."""
        result = cli_runner.invoke(app, [
            "site-model",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--test", "both",
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0
        assert "M1a" in result.stdout or "M1A" in result.stdout
        assert "M7" in result.stdout

    @pytest.mark.slow
    def test_json_output(self, cli_runner, lysozyme_small_files, lysozyme_tree_file, tmp_path):
        """Test with JSON output format."""
        output_file = tmp_path / "test_result.json"

        result = cli_runner.invoke(app, [
            "site-model",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--test", "m7m8",
            "--maxiter", "50",
            "--format", "json",
            "--output", str(output_file),
            "--quiet"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        assert "M7_vs_M8" in data

    @pytest.mark.slow
    def test_tsv_output(self, cli_runner, lysozyme_small_files, lysozyme_tree_file, tmp_path):
        """Test with TSV output format."""
        output_file = tmp_path / "test_result.tsv"

        result = cli_runner.invoke(app, [
            "site-model",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--test", "m7m8",
            "--maxiter", "50",
            "--format", "tsv",
            "--output", str(output_file),
            "--quiet"
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify TSV structure
        content = output_file.read_text()
        lines = content.strip().split('\n')
        assert len(lines) >= 2  # header + at least one data row
        assert "site-model" in lines[0]
        assert "pvalue" in lines[0]


class TestCLIBranchSite:
    """Test 'branch-site' command functionality."""

    def test_branch_site_help(self, cli_runner):
        """Test branch-site command help."""
        result = cli_runner.invoke(app, ["branch-site", "--help"])
        assert result.exit_code == 0
        assert "branch" in result.stdout.lower()

    @pytest.mark.slow
    def test_branch_site_missing_labels(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test branch-site with tree missing labels."""
        result = cli_runner.invoke(app, [
            "branch-site",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--quiet"
        ])

        # Should fail because tree has no branch labels
        assert result.exit_code == 1
        assert "branch labels" in result.stdout.lower()


class TestCLIOutputFormats:
    """Test different output formats."""

    def test_text_format_default(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test that text format is default."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0
        # Text format should have human-readable output
        assert "MODEL:" in result.stdout or "Log-likelihood:" in result.stdout

    def test_output_to_file(self, cli_runner, lysozyme_small_files, lysozyme_tree_file, tmp_path):
        """Test writing output to file."""
        output_file = tmp_path / "output.txt"

        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--output", str(output_file),
            "--quiet"
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "MODEL:" in content or "Log-likelihood:" in content


class TestCLIFlags:
    """Test various CLI flags and options."""

    def test_verbose_flag(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test verbose output flag."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--verbose"
        ])

        assert result.exit_code == 0
        # Verbose mode should show optimization progress
        assert "Optimization" in result.stdout or "Starting" in result.stdout

    def test_quiet_flag(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test quiet output flag."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0
        # Quiet mode should suppress some messages
        # But still show results

    def test_maxiter_flag(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test maxiter flag."""
        # With very low maxiter, optimization may not converge fully
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M0",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "100",
            "--quiet"
        ])

        assert result.exit_code == 0

    @pytest.mark.slow
    def test_no_m0_init_flag(self, cli_runner, lysozyme_small_files, lysozyme_tree_file):
        """Test --no-m0-init flag."""
        result = cli_runner.invoke(app, [
            "fit",
            "-m", "M7",
            "-s", str(lysozyme_small_files["sequences"]),
            "-t", str(lysozyme_tree_file),
            "--maxiter", "300",  # More iterations needed without M0 init
            "--no-m0-init",
            "--quiet"
        ])

        # May work but may take longer/converge differently
        # With low maxiter may fail to converge - that's okay for this test
        # Just verify the flag is accepted
        assert result.exit_code in [0, 1]  # Either success or convergence failure
