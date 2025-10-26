"""
Output formatting for simulated sequences.
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path
import json

from ..io.sequences import INDEX_TO_CODON


class SimulationOutput:
    """
    Handle output formatting for simulated sequences.

    Provides methods to write:
    - Sequences in FASTA format
    - Parameters in JSON format
    - Ancestral sequences
    - Site class assignments (for site models)
    """

    @staticmethod
    def indices_to_codons(seq_indices: np.ndarray) -> str:
        """
        Convert array of codon indices to string sequence.

        Parameters
        ----------
        seq_indices : np.ndarray
            Array of codon indices (0-60)

        Returns
        -------
        str
            String of concatenated codons (e.g., 'ATGCGATAA...')
        """
        return ''.join(INDEX_TO_CODON[int(i)] for i in seq_indices)

    @staticmethod
    def write_fasta(
        sequences: Dict[str, np.ndarray],
        output_path: Path,
        replicate_id: Optional[int] = None,
        line_width: int = 60  # 60 nucleotides = 20 codons
    ):
        """
        Write sequences to FASTA format.

        Parameters
        ----------
        sequences : dict
            Mapping from species name to sequence array (codon indices)
        output_path : Path
            Output file path
        replicate_id : int, optional
            Replicate number (added to header if provided)
        line_width : int
            Number of nucleotides per line (default 60)
        """
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            for species, seq_indices in sequences.items():
                # Convert indices to codon string
                codon_seq = SimulationOutput.indices_to_codons(seq_indices)

                # Write FASTA header
                header = f">{species}"
                if replicate_id is not None:
                    header += f" replicate={replicate_id}"
                f.write(header + '\n')

                # Write sequence (wrapped at line_width nucleotides)
                for i in range(0, len(codon_seq), line_width):
                    f.write(codon_seq[i:i+line_width] + '\n')

    @staticmethod
    def write_parameters(
        params: Dict,
        output_path: Path,
        indent: int = 2
    ):
        """
        Write simulation parameters to JSON file.

        Parameters
        ----------
        params : dict
            Simulation parameters
        output_path : Path
            Output file path
        indent : int
            JSON indentation level
        """
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=indent)

    @staticmethod
    def write_site_classes(
        site_class_ids: np.ndarray,
        site_class_omegas: np.ndarray,
        output_path: Path
    ):
        """
        Write site class assignments to file.

        Useful for site-class models (M1a, M2a, M7, M8) to track which
        sites have omega > 1 (positive selection).

        Parameters
        ----------
        site_class_ids : np.ndarray
            Site class ID for each site (0, 1, 2, ...)
        site_class_omegas : np.ndarray
            Omega value for each site class
        output_path : Path
            Output file path

        Output Format
        -------------
        site_id  class_id  omega   [marker]
        1        0         0.1000
        2        2         2.5000  *
        3        1         1.0000
        ...

        Sites with omega > 1 are marked with '*'
        """
        output_path = Path(output_path)

        with open(output_path, 'w') as f:
            # Write header
            f.write("# Site class assignments\n")
            f.write("# Sites marked with '*' have omega > 1 (positive selection)\n")
            f.write("site_id\tclass_id\tomega\n")

            # Write each site
            for site_idx, class_id in enumerate(site_class_ids):
                omega = site_class_omegas[int(class_id)]
                marker = "\t*" if omega > 1 else ""
                f.write(f"{site_idx+1}\t{class_id}\t{omega:.4f}{marker}\n")

    @staticmethod
    def write_positively_selected_sites(
        site_class_ids: np.ndarray,
        site_class_omegas: np.ndarray,
        output_path: Path
    ):
        """
        Write list of positively selected sites (omega > 1).

        Parameters
        ----------
        site_class_ids : np.ndarray
            Site class ID for each site
        site_class_omegas : np.ndarray
            Omega value for each site class
        output_path : Path
            Output file path

        Output Format
        -------------
        # Positively selected sites (omega > 1)
        # Total: N sites
        # Site IDs (1-indexed):
        1 5 12 23 45 67 89 ...
        """
        output_path = Path(output_path)

        # Find sites with omega > 1
        ps_sites = []
        for site_idx, class_id in enumerate(site_class_ids):
            omega = site_class_omegas[int(class_id)]
            if omega > 1:
                ps_sites.append(site_idx + 1)  # 1-indexed

        with open(output_path, 'w') as f:
            f.write("# Positively selected sites (omega > 1)\n")
            f.write(f"# Total: {len(ps_sites)} sites\n")
            f.write("# Site IDs (1-indexed):\n")

            # Write sites (15 per line for readability)
            for i in range(0, len(ps_sites), 15):
                sites = [str(s) for s in ps_sites[i:i+15]]
                f.write(" ".join(sites) + "\n")

    @staticmethod
    def write_ancestral_sequences(
        sequences: Dict[str, np.ndarray],
        tree,
        output_path: Path
    ):
        """
        Write ancestral sequences (internal nodes) to FASTA.

        Parameters
        ----------
        sequences : dict
            All node sequences (tips + internal)
        tree : Tree
            Phylogenetic tree
        output_path : Path
            Output file path
        """
        output_path = Path(output_path)

        # Filter to internal nodes only
        internal_seqs = {}
        for node in tree.postorder():
            # Internal nodes have children
            if node.children:
                name = node.name if node.name else f"node_{id(node)}"
                if node in sequences or id(node) in sequences:
                    seq = sequences.get(node, sequences.get(id(node)))
                    internal_seqs[name] = seq

        # Write to FASTA
        if internal_seqs:
            SimulationOutput.write_fasta(internal_seqs, output_path)
