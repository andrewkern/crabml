"""Branch model test command implementation."""

import sys
import json
from pathlib import Path
from typing import Optional

from crabml.io.sequences import Alignment
from crabml.io.trees import Tree
from crabml.analysis.branch_model import branch_model_test, free_ratio_test


def run_branch_model(
    alignment: Path,
    tree: Path,
    test_type: str,
    alpha: float,
    output: Optional[Path],
    format: str,
    verbose: bool,
    quiet: bool,
    maxiter: int,
):
    """Run branch model test."""
    # Load alignment
    try:
        aln = Alignment.from_fasta(str(alignment), seqtype='codon')
    except Exception:
        try:
            aln = Alignment.from_phylip(str(alignment), seqtype='codon')
        except Exception as e:
            print(f"Error: Could not load alignment from {alignment}", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)

    # Load tree
    try:
        with open(tree, 'r') as f:
            tree_str = f.read().strip()
        tree_obj = Tree.from_newick(tree_str)
    except Exception as e:
        print(f"Error: Could not load tree from {tree}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Print header unless quiet
    if not quiet:
        print("Testing for Lineage-Specific Selection", file=sys.stderr)
        print(f"Test: {test_type}", file=sys.stderr)
        print(f"Alignment: {alignment}", file=sys.stderr)
        print(f"Tree:      {tree}", file=sys.stderr)
        print(file=sys.stderr)

    # Run the appropriate test
    try:
        if test_type == "multi-ratio":
            result = branch_model_test(
                alignment=aln,
                tree=tree_obj,
                verbose=verbose,
            )
        elif test_type == "free-ratio":
            result = free_ratio_test(
                alignment=aln,
                tree=tree_obj,
                verbose=verbose,
            )
        else:
            print(f"Error: Unknown test type '{test_type}'", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: Analysis failed", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if format == "json":
        output_text = _format_json(result, alpha)
    elif format == "tsv":
        output_text = _format_tsv(result, alpha)
    else:  # text
        output_text = _format_text(result, alpha, quiet)

    # Write output
    if output:
        with open(output, 'w') as f:
            f.write(output_text)
    else:
        print(output_text)


def _format_text(result, alpha: float, quiet: bool) -> str:
    """Format result as human-readable text."""
    lines = []

    if not quiet:
        lines.append("=" * 80)
        lines.append("Branch Model Test Results")
        lines.append("=" * 80)
        lines.append("")

    # Test information
    lines.append(f"Test: {result.test_name}")
    lines.append("-" * 80)
    lines.append(f"Null ({result.null_model}):        lnL = {result.lnL_null:.3f}    "
                 f"parameters = {result.null_params}")
    lines.append(f"Alternative ({result.alt_model}):  lnL = {result.lnL_alt:.3f}    "
                 f"parameters = {result.alt_params}")
    lines.append("")
    lines.append("Likelihood Ratio Test:")
    lines.append(f"  2ΔlnL = {result.LRT:.2f}    "
                 f"df = {result.df}    "
                 f"p-value = {result.pvalue:.4f}")
    lines.append("")

    # Interpretation
    if result.significant(alpha):
        lines.append(f"Result: LINEAGE-SPECIFIC SELECTION DETECTED (p < {alpha})")

        # Extract omega values if available
        if 'omega0' in result.alt_params and 'omega1' in result.alt_params:
            omega_bg = result.alt_params['omega0']
            omega_fg = result.alt_params['omega1']
            lines.append(f"  Background ω = {omega_bg:.3f}")
            lines.append(f"  Foreground ω = {omega_fg:.3f}")

            if omega_fg > omega_bg:
                ratio = omega_fg / omega_bg if omega_bg > 0 else float('inf')
                lines.append(f"  Foreground is {ratio:.1f}x faster evolving")
        elif 'omega_dict' in result.alt_params:
            lines.append(f"  Branch-specific omegas: {len(result.alt_params['omega_dict'])} branches")
    else:
        lines.append(f"Result: No significant evidence for lineage-specific selection (p > {alpha})")

    if not quiet:
        lines.append("")
        lines.append("=" * 80)

    return "\n".join(lines)


def _format_json(result, alpha: float) -> str:
    """Format result as JSON."""
    output_dict = {
        "test_name": result.test_name,
        "null_model": result.null_model,
        "alt_model": result.alt_model,
        "lnL_null": result.lnL_null,
        "lnL_alt": result.lnL_alt,
        "LRT": result.LRT,
        "df": result.df,
        "pvalue": result.pvalue,
        "significant": result.significant(alpha),
        "alpha": alpha,
        "null_params": result.null_params,
        "alt_params": result.alt_params,
    }

    return json.dumps(output_dict, indent=2, default=str)


def _format_tsv(result, alpha: float) -> str:
    """Format result as TSV."""
    lines = []
    lines.append("\t".join([
        "test", "null_model", "alt_model", "null_lnL", "alt_lnL",
        "LRT", "df", "pvalue", "significant"
    ]))

    lines.append("\t".join([
        result.test_name,
        result.null_model,
        result.alt_model,
        f"{result.lnL_null:.6f}",
        f"{result.lnL_alt:.6f}",
        f"{result.LRT:.6f}",
        str(result.df),
        f"{result.pvalue:.6f}",
        "yes" if result.significant(alpha) else "no"
    ]))

    return "\n".join(lines)
