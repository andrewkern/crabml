"""Test command implementation."""

import sys
import json
from pathlib import Path
from typing import Optional
import numpy as np

from crabml.analysis import positive_selection
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_test(
    alignment: Path,
    tree: Path,
    test_type: str,
    alpha: float,
    output: Optional[Path],
    format: str,
    verbose: bool,
    quiet: bool,
    maxiter: int,
    init_with_m0: bool,
):
    """Run positive selection tests."""
    # Load data
    try:
        aln = Alignment.from_phylip(str(alignment), seqtype="codon")
    except Exception:
        try:
            aln = Alignment.from_fasta(str(alignment), seqtype="codon")
        except Exception as e:
            print(f"Error: Could not load alignment from {alignment}", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        with open(tree, 'r') as f:
            tree_str = f.read().strip()
        tree_obj = Tree.from_newick(tree_str)
    except Exception as e:
        print(f"Error: Could not load tree from {tree}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine which tests to run
    if test_type == "m1m2":
        test = "m1a_vs_m2a"
    elif test_type == "m7m8":
        test = "m7_vs_m8"
    elif test_type == "m8am8":
        test = "m8a_vs_m8"
    elif test_type == "both":
        test = "both"
    elif test_type == "all":
        test = "both"  # positive_selection() doesn't support "all", use "both"
    else:
        print(f"Error: Unknown test type '{test_type}'", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    if not quiet:
        print("Testing for Positive Selection", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"Alignment: {alignment}", file=sys.stderr)
        print(f"Tree:      {tree}", file=sys.stderr)
        print(file=sys.stderr)

    try:
        results = positive_selection(
            alignment=aln,
            tree=tree_obj,
            test=test,
            verbose=verbose,
        )
    except Exception as e:
        print(f"Error: Analysis failed", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    # Convert single result to dict for uniform handling
    if not isinstance(results, dict):
        # Single test result - wrap in dict with appropriate key
        if hasattr(results, 'test_name'):
            key = results.test_name.replace(' ', '_')
            results = {key: results}
        else:
            results = {'result': results}

    if format == "json":
        output_text = _format_json(results, alpha)
    elif format == "tsv":
        output_text = _format_tsv(results, alpha)
    else:  # text
        output_text = _format_text(results, alpha, quiet)

    # Write output
    if output:
        with open(output, 'w') as f:
            f.write(output_text)
        if not quiet:
            print(f"\nResults written to {output}", file=sys.stderr)
    else:
        print(output_text)


def _format_text(results: dict, alpha: float, quiet: bool) -> str:
    """Format results as human-readable text."""
    lines = []

    if not quiet:
        lines.append("")

    # M1a vs M2a
    if 'M1a_vs_M2a' in results:
        test_result = results['M1a_vs_M2a']
        lines.append("Test 1: M1a (Nearly Neutral) vs M2a (Positive Selection)")
        lines.append("-" * 80)
        lines.append(f"Null (M1a):          lnL = {test_result.lnL_null:.3f}    "
                     f"parameters = {test_result.null_params}")
        lines.append(f"Alternative (M2a):   lnL = {test_result.lnL_alt:.3f}    "
                     f"parameters = {test_result.alt_params}")
        lines.append("")
        lines.append("Likelihood Ratio Test:")
        lines.append(f"  2ΔlnL = {test_result.LRT:.2f}    "
                     f"df = {test_result.df}    "
                     f"p-value = {test_result.pvalue:.4f}")
        lines.append("")

        if test_result.significant(alpha):
            omega = test_result.alt_params.get('omega_2', 'N/A')
            p2 = test_result.alt_params.get('p2', 'N/A')
            lines.append(f"Result: POSITIVE SELECTION DETECTED (p < {alpha})")
            if omega != 'N/A':
                lines.append(f"  ω for positive selection = {omega:.2f}")
            if p2 != 'N/A':
                lines.append(f"  Proportion of sites under selection = {p2:.1%}")
        else:
            lines.append(f"Result: No significant evidence for positive selection (p > {alpha})")
        lines.append("")

    # M7 vs M8
    if 'M7_vs_M8' in results:
        test_result = results['M7_vs_M8']
        lines.append("Test 2: M7 (Beta) vs M8 (Beta + positive selection)")
        lines.append("-" * 80)
        lines.append(f"Null (M7):           lnL = {test_result.lnL_null:.3f}    "
                     f"parameters = {test_result.null_params}")
        lines.append(f"Alternative (M8):    lnL = {test_result.lnL_alt:.3f}    "
                     f"parameters = {test_result.alt_params}")
        lines.append("")
        lines.append("Likelihood Ratio Test:")
        lines.append(f"  2ΔlnL = {test_result.LRT:.2f}    "
                     f"df = {test_result.df}    "
                     f"p-value = {test_result.pvalue:.4f}")
        lines.append("")

        if test_result.significant(alpha):
            omega = test_result.alt_params.get('omega_s', 'N/A')
            p_sel = 1 - test_result.alt_params.get('p0', 0)
            lines.append(f"Result: POSITIVE SELECTION DETECTED (p < {alpha})")
            if omega != 'N/A':
                lines.append(f"  ω for positive selection = {omega:.2f}")
            lines.append(f"  Proportion of sites under selection = {p_sel:.1%}")
        else:
            lines.append(f"Result: No significant evidence for positive selection (p > {alpha})")
        lines.append("")

    # M8a vs M8
    if 'M8a_vs_M8' in results:
        test_result = results['M8a_vs_M8']
        lines.append("Test: M8a (Beta + neutral) vs M8 (Beta + positive selection)")
        lines.append("-" * 80)
        lines.append(f"Null (M8a):          lnL = {test_result.lnL_null:.3f}    "
                     f"parameters = {test_result.null_params}")
        lines.append(f"Alternative (M8):    lnL = {test_result.lnL_alt:.3f}    "
                     f"parameters = {test_result.alt_params}")
        lines.append("")
        lines.append("Likelihood Ratio Test (50:50 mixture chi-square):")
        lines.append(f"  2ΔlnL = {test_result.LRT:.2f}    "
                     f"df = {test_result.df}    "
                     f"p-value = {test_result.pvalue:.4f}")
        lines.append("")

        if test_result.significant(alpha):
            omega = test_result.alt_params.get('omega_s', 'N/A')
            p_sel = 1 - test_result.alt_params.get('p0', 0)
            lines.append(f"Result: POSITIVE SELECTION DETECTED (p < {alpha})")
            if omega != 'N/A':
                lines.append(f"  ω for positive selection = {omega:.2f}")
            lines.append(f"  Proportion of sites under selection = {p_sel:.1%}")
        else:
            lines.append(f"Result: No significant evidence for positive selection (p > {alpha})")
        lines.append("")
        lines.append("Note: This test uses a 50:50 mixture chi-square null distribution")
        lines.append("      because ω=1 is on the boundary of the parameter space.")
        lines.append("")

    # Summary
    if not quiet:
        lines.append("=" * 80)
        any_significant = any(
            results[key].significant(alpha)
            for key in results
            if hasattr(results[key], 'significant')
        )
        if any_significant:
            lines.append("SUMMARY: Evidence for positive selection detected")
        else:
            lines.append("SUMMARY: No evidence for positive selection in this dataset")
        lines.append("=" * 80)

    return "\n".join(lines)


def _format_json(results: dict, alpha: float) -> str:
    """Format results as JSON."""
    output_dict = {}

    for key, test_result in results.items():
        if hasattr(test_result, 'to_dict'):
            output_dict[key] = test_result.to_dict()
            output_dict[key]['significant'] = test_result.significant(alpha)
            output_dict[key]['alpha'] = alpha

    return json.dumps(output_dict, indent=2, cls=NumpyEncoder)


def _format_tsv(results: dict, alpha: float) -> str:
    """Format results as TSV."""
    lines = []
    lines.append("\t".join([
        "test", "null_model", "alt_model", "null_lnL", "alt_lnL",
        "LRT", "df", "pvalue", "significant"
    ]))

    for key, test_result in results.items():
        if hasattr(test_result, 'significant'):
            lines.append("\t".join([
                key,
                test_result.null_model if hasattr(test_result, 'null_model') else '',
                test_result.alt_model if hasattr(test_result, 'alt_model') else '',
                f"{test_result.lnL_null:.6f}",
                f"{test_result.lnL_alt:.6f}",
                f"{test_result.LRT:.6f}",
                str(test_result.df),
                f"{test_result.pvalue:.6f}",
                "yes" if test_result.significant(alpha) else "no"
            ]))

    return "\n".join(lines)
