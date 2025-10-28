"""Benchmark dataset simulator."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import tempfile

from crabml.io.trees import Tree


class BenchmarkSimulator:
    """Generate simulated datasets for benchmarking."""

    def __init__(self, config: Dict):
        """
        Initialize simulator with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary from config.yaml
        """
        self.config = config
        self.rng = np.random.RandomState(config['simulation']['random_seed_start'])

    def generate_trees(self) -> Dict[str, str]:
        """
        Generate tree set with varied branch lengths.

        Returns
        -------
        dict
            Mapping from tree_id to scaled newick string
        """
        trees = {}
        tree_configs = self.config['simulation']['tree_topologies']

        for tree_id, base_newick in tree_configs.items():
            # Parse base tree
            tree = Tree.from_newick(base_newick)

            # Sample total tree length
            min_len = self.config['simulation']['tree_length_min']
            max_len = self.config['simulation']['tree_length_max']
            target_length = self.rng.uniform(min_len, max_len)

            # Calculate current total length
            current_length = sum(
                node.branch_length for node in tree.postorder()
                if node.parent is not None
            )

            # Scale all branch lengths
            scale_factor = target_length / current_length
            for node in tree.postorder():
                if node.parent is not None:
                    node.branch_length *= scale_factor

            # Convert back to newick
            trees[tree_id] = tree.to_newick()

        return trees

    def sample_parameters(self, model: str) -> Dict[str, float]:
        """
        Sample parameters for a model from configured ranges.

        Parameters
        ----------
        model : str
            Model name (M0, M1a, etc.)

        Returns
        -------
        dict
            Sampled parameters
        """
        ranges = self.config['parameter_ranges'][model]
        params = {}

        for param, value in ranges.items():
            if isinstance(value, list) and len(value) == 2:
                # Sample from uniform distribution
                params[param] = self.rng.uniform(value[0], value[1])
            else:
                # Fixed value
                params[param] = value

        return params

    def generate_dataset(
        self,
        model: str,
        replicate_id: int,
        tree_newick: str,
        tree_id: str,
        params: Dict[str, float],
        seq_length: int,
        output_dir: Path
    ) -> Dict[str, any]:
        """
        Generate a single simulated dataset.

        Parameters
        ----------
        model : str
            Model name
        replicate_id : int
            Replicate number
        tree_newick : str
            Tree in Newick format
        tree_id : str
            Tree identifier
        params : dict
            Simulation parameters
        seq_length : int
            Sequence length in codons
        output_dir : Path
            Output directory for this dataset

        Returns
        -------
        dict
            Dataset metadata
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tree file
        tree_file = output_dir / f"rep{replicate_id:03d}.nwk"
        with open(tree_file, 'w') as f:
            f.write(tree_newick)

        # Prepare simulation command
        fasta_file = output_dir / f"rep{replicate_id:03d}.fasta"
        params_file = output_dir / f"rep{replicate_id:03d}.params.json"

        # Build command based on model
        cmd = [
            "crabml", "simulate", model.lower(),
            "-t", str(tree_file),
            "-o", str(fasta_file),
            "-l", str(seq_length),
            "--seed", str(self.config['simulation']['random_seed_start'] + replicate_id),
            "-q"  # Quiet
        ]

        # Add model-specific parameters
        if model == "M0":
            cmd.extend(["--kappa", str(params['kappa'])])
            cmd.extend(["--omega", str(params['omega'])])

        elif model == "M1a":
            cmd.extend(["--kappa", str(params['kappa'])])
            cmd.extend(["--p0", str(params['p0'])])
            cmd.extend(["--omega0", str(params['omega0'])])

        elif model == "M2a":
            cmd.extend(["--kappa", str(params['kappa'])])
            cmd.extend(["--p0", str(params['p0'])])
            cmd.extend(["--p1", str(params['p1'])])
            cmd.extend(["--omega0", str(params['omega0'])])
            cmd.extend(["--omega2", str(params['omega2'])])

        elif model == "M7":
            cmd.extend(["--kappa", str(params['kappa'])])
            cmd.extend(["--p", str(params['p'])])
            cmd.extend(["--q", str(params['q'])])
            cmd.extend(["--ncateg", str(int(params['ncateg']))])

        elif model in ["M8", "M8a"]:
            cmd.extend(["--kappa", str(params['kappa'])])
            cmd.extend(["--p0", str(params['p0'])])
            cmd.extend(["--p", str(params['p'])])
            cmd.extend(["--q", str(params['q'])])
            cmd.extend(["--ncateg", str(int(params['ncateg']))])
            if model == "M8":
                cmd.extend(["--omega-s", str(params['omega_s'])])
            # M8a doesn't need omega_s (fixed to 1.0)

        # Run simulation
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Simulation failed: {result.stderr}")

        # Save metadata
        metadata = {
            "model": model,
            "replicate_id": replicate_id,
            "tree_id": tree_id,
            "sequence_length": seq_length,
            "true_parameters": params,
            "seed": self.config['simulation']['random_seed_start'] + replicate_id,
            "fasta_file": str(fasta_file),
            "tree_file": str(tree_file),
        }

        with open(params_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def generate_all_datasets(self, models: List[str], output_base: Path):
        """
        Generate all benchmark datasets.

        Parameters
        ----------
        models : list of str
            Models to simulate
        output_base : Path
            Base directory for all datasets

        Returns
        -------
        list of dict
            List of dataset metadata
        """
        datasets = []
        n_replicates = self.config['n_replicates']
        seq_lengths = self.config['simulation']['sequence_lengths']

        # Generate trees once
        trees = self.generate_trees()
        tree_ids = list(trees.keys())

        for model in models:
            print(f"\nGenerating datasets for {model}...")
            model_dir = output_base / model

            # Distribute replicates evenly across tree topologies and sequence lengths
            rep_id = 1
            for i in range(n_replicates):
                # Cycle through trees and sequence lengths
                tree_id = tree_ids[i % len(tree_ids)]
                seq_length = seq_lengths[i % len(seq_lengths)]
                tree_newick = trees[tree_id]

                # Sample parameters
                params = self.sample_parameters(model)

                # Generate dataset
                try:
                    metadata = self.generate_dataset(
                        model=model,
                        replicate_id=rep_id,
                        tree_newick=tree_newick,
                        tree_id=tree_id,
                        params=params,
                        seq_length=seq_length,
                        output_dir=model_dir
                    )
                    datasets.append(metadata)
                    print(f"  Generated replicate {rep_id:03d} "
                          f"({tree_id}, {seq_length} codons)")

                except Exception as e:
                    print(f"  ERROR in replicate {rep_id:03d}: {e}")

                rep_id += 1

        return datasets
