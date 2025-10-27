"""Fit command implementation."""

import sys
import json
from pathlib import Path
from typing import Optional

from crabml import optimize_model
from crabml.io.sequences import Alignment
from crabml.io.trees import Tree


def run_fit(
    model: str,
    alignment: Path,
    tree: Path,
    output: Optional[Path],
    format: str,
    verbose: bool,
    quiet: bool,
    maxiter: int,
    init_with_m0: bool,
):
    """Fit a single model."""
    # Validate model name (case-insensitive)
    valid_models = ['M0', 'M1a', 'M2a', 'M3', 'M7', 'M8', 'M8a']

    # Find matching model (case-insensitive)
    model_upper = None
    for valid in valid_models:
        if model.upper() == valid.upper():
            model_upper = valid
            break

    if model_upper is None:
        print(f"Error: Unknown model '{model}'", file=sys.stderr)
        print(f"Valid models: {', '.join(valid_models)}", file=sys.stderr)
        sys.exit(1)

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

    # Run model fitting
    if not quiet:
        print(f"Fitting Model: {model_upper}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"Alignment: {alignment}", file=sys.stderr)
        print(f"Tree:      {tree}", file=sys.stderr)
        print(file=sys.stderr)

    try:
        # Build kwargs - only pass init_with_m0 for models other than M0
        kwargs = {
            'model': model_upper,
            'alignment': aln,
            'tree': tree_obj,
            'use_f3x4': True,
            'optimize_branch_lengths': True,
            'maxiter': maxiter,
        }
        if model_upper != 'M0':
            kwargs['init_with_m0'] = init_with_m0

        result = optimize_model(**kwargs)
    except Exception as e:
        print(f"Error: Model fitting failed", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if format == "json":
        output_text = result.to_json()
    else:  # text
        output_text = result.summary()

    # Write output
    if output:
        if format == "json":
            result.to_json(str(output))
        else:
            with open(output, 'w') as f:
                f.write(output_text)
        if not quiet:
            print(f"\nResults written to {output}", file=sys.stderr)
    else:
        print(output_text)
