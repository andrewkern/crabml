"""Branch-site command implementation."""

import sys
import json
from pathlib import Path
from typing import Optional

from crabml.analysis import branch_site_test
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


def run_branch_site(
    alignment: Path,
    tree: Path,
    alpha: float,
    output: Optional[Path],
    format: str,
    verbose: bool,
    quiet: bool,
    maxiter: int,
):
    """Run branch-site test for positive selection on specific lineages."""
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

    # Check if tree has branch labels
    has_labels = any(
        hasattr(node, 'label') and node.label is not None
        for node in tree_obj.postorder()
    )
    if not has_labels:
        print("Error: Tree must have branch labels (#0, #1, etc.) for branch-site analysis", file=sys.stderr)
        print("Example: ((human,chimp) #1, (mouse,rat) #0);", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    if not quiet:
        print("Branch-Site Test for Positive Selection", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"Alignment: {alignment}", file=sys.stderr)
        print(f"Tree:      {tree}", file=sys.stderr)
        print(file=sys.stderr)

    try:
        results = branch_site_test(
            alignment=aln,
            tree=tree_obj,
            use_f3x4=True,
            optimize_branch_lengths=True,
            maxiter=maxiter,
            verbose=verbose,
        )
    except Exception as e:
        print(f"Error: Branch-site analysis failed", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if format == "json":
        output_text = _format_json(results, alpha)
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

    lines.append("Branch-Site Model A Test")
    lines.append("-" * 80)
    lines.append(f"Null (ω₂ = 1):        lnL = {results['null_lnL']:.3f}")
    lines.append(f"Alternative (ω₂ free): lnL = {results['alt_lnL']:.3f}")
    lines.append("")
    lines.append("Likelihood Ratio Test:")
    lines.append(f"  2ΔlnL = {results['lrt_statistic']:.2f}    "
                 f"df = 1    "
                 f"p-value = {results['pvalue']:.4f}")
    lines.append("")

    if results['significant']:
        omega2 = results['alt_params'].get('omega2', 'N/A')
        p2 = results['alt_params'].get('p2', 'N/A')
        lines.append(f"Result: POSITIVE SELECTION DETECTED on foreground branches (p < {alpha})")
        if omega2 != 'N/A':
            lines.append(f"  ω₂ (selection on foreground) = {omega2:.2f}")
        if p2 != 'N/A':
            lines.append(f"  Proportion of sites under selection = {p2:.1%}")
    else:
        lines.append(f"Result: No significant evidence for positive selection (p > {alpha})")
    lines.append("")

    if not quiet:
        lines.append("=" * 80)

    return "\n".join(lines)


def _format_json(results: dict, alpha: float) -> str:
    """Format results as JSON."""
    output_dict = results.copy()
    output_dict['alpha'] = alpha
    return json.dumps(output_dict, indent=2)
