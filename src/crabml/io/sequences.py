"""
Sequence file parsing and alignment handling.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Genetic code tables (standard code)
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Build codon to index mapping (excluding stop codons)
# IMPORTANT: Order must match PAML's FROM61 array!
# PAML uses nucleotide indices: T=0, C=1, A=2, G=3
# Codon i has index: n0*16 + n1*4 + n2
_INDEX_TO_NUCLEOTIDE_PAML = {0: 'T', 1: 'C', 2: 'A', 3: 'G'}
CODONS = []
for i in range(64):
    n0 = i // 16
    n1 = (i // 4) % 4
    n2 = i % 4
    codon = _INDEX_TO_NUCLEOTIDE_PAML[n0] + _INDEX_TO_NUCLEOTIDE_PAML[n1] + _INDEX_TO_NUCLEOTIDE_PAML[n2]
    if GENETIC_CODE[codon] != '*':  # Skip stop codons (TAA, TAG, TGA)
        CODONS.append(codon)

CODON_TO_INDEX = {codon: i for i, codon in enumerate(CODONS)}
INDEX_TO_CODON = {i: codon for i, codon in enumerate(CODONS)}

# Special codes for missing/ambiguous data
GAP_CODE = 64  # Gap codon (---)
UNKNOWN_CODE = -1  # Unknown/stop codon

# Nucleotide encoding
NUCLEOTIDE_TO_INDEX = {'T': 0, 'C': 1, 'A': 2, 'G': 3}
INDEX_TO_NUCLEOTIDE = {0: 'T', 1: 'C', 2: 'A', 3: 'G'}

# Amino acid encoding
AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
INDEX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}


@dataclass
class Alignment:
    """
    Multiple sequence alignment.

    Attributes
    ----------
    names : list[str]
        Sequence names/labels
    sequences : ndarray, shape (n_species, n_sites)
        Encoded sequences as integer arrays
    n_species : int
        Number of sequences
    n_sites : int
        Number of sites (alignment length)
    seqtype : str
        Sequence type ('codon', 'aa', 'dna')
    """

    names: list[str]
    sequences: np.ndarray
    n_species: int
    n_sites: int
    seqtype: str

    @classmethod
    def from_phylip(
        cls, filepath: Path | str, seqtype: str = "codon"
    ) -> "Alignment":
        """
        Parse PHYLIP format alignment file.

        Custom parser for PAML-style PHYLIP format (sequential).
        The first line contains n_sequences and sequence_length.
        Each sequence starts with a name line, followed by sequence data.

        Parameters
        ----------
        filepath : Path or str
            Path to PHYLIP format file
        seqtype : str
            Sequence type: 'codon', 'aa', or 'dna'

        Returns
        -------
        Alignment
            Parsed alignment

        Examples
        --------
        >>> aln = Alignment.from_phylip("lysozyme.txt", seqtype='codon')
        >>> aln.n_species
        7
        >>> aln.n_sites
        130
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]

        # Parse header
        header = lines[0].strip().split()
        n_species = int(header[0])
        n_chars = int(header[1])

        # Calculate n_sites
        if seqtype == 'codon':
            if n_chars % 3 != 0:
                raise ValueError(f"Codon sequence length {n_chars} not divisible by 3")
            n_sites = n_chars // 3
        else:
            n_sites = n_chars

        # Parse sequences
        names = []
        sequences_raw = []

        i = 1
        while i < len(lines) and len(names) < n_species:
            line = lines[i].strip()
            i += 1

            # Skip empty lines
            if not line:
                continue

            # This is a sequence name
            name = line
            names.append(name)

            # Collect sequence data until we have enough chars or hit next name
            seq_data = ""
            while i < len(lines):
                line = lines[i].strip()

                # Skip empty lines
                if not line:
                    i += 1
                    continue

                # Check if this is DNA/protein sequence data
                clean = re.sub(r'\s', '', line).upper()

                # If we have enough characters, we're done with this sequence
                if len(re.sub(r'\s', '', seq_data)) >= n_chars:
                    break

                # Add this line to sequence
                seq_data += clean
                i += 1

            sequences_raw.append(seq_data)

        if len(names) != n_species:
            raise ValueError(f"Expected {n_species} sequences, found {len(names)}")

        # Verify all sequences have correct length
        for i, seq in enumerate(sequences_raw):
            if len(seq) != n_chars:
                raise ValueError(
                    f"Sequence {names[i]} has length {len(seq)}, expected {n_chars}"
                )

        # Encode sequences
        if seqtype == 'codon':
            encoded = cls._encode_codons(sequences_raw)
        elif seqtype == 'dna':
            encoded = cls._encode_nucleotides(sequences_raw)
        elif seqtype == 'aa':
            encoded = cls._encode_amino_acids(sequences_raw)
        else:
            raise ValueError(f"Unknown seqtype: {seqtype}")

        return cls(
            names=names,
            sequences=encoded,
            n_species=n_species,
            n_sites=n_sites,
            seqtype=seqtype,
        )

    @classmethod
    def from_fasta(
        cls, filepath: Path | str, seqtype: str = "codon"
    ) -> "Alignment":
        """
        Parse FASTA format alignment file.

        Parameters
        ----------
        filepath : Path or str
            Path to FASTA format file
        seqtype : str
            Sequence type: 'codon', 'aa', or 'dna'

        Returns
        -------
        Alignment
            Parsed alignment

        Examples
        --------
        >>> aln = Alignment.from_fasta("alignment.fasta", seqtype='codon')
        """
        filepath = Path(filepath)

        names = []
        sequences_raw = []

        with open(filepath, 'r') as f:
            current_name = None
            current_seq = []

            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_name is not None:
                        names.append(current_name)
                        sequences_raw.append(''.join(current_seq))

                    # Start new sequence
                    current_name = line[1:].strip()
                    current_seq = []
                else:
                    # Add to current sequence
                    current_seq.append(line.upper())

            # Don't forget last sequence
            if current_name is not None:
                names.append(current_name)
                sequences_raw.append(''.join(current_seq))

        if not names:
            raise ValueError("No sequences found in FASTA file")

        # Remove spaces but KEEP gaps (PAML cleandata=0 behavior)
        sequences_clean = [re.sub(r'\s', '', seq) for seq in sequences_raw]

        # Check all sequences same length (WITH gaps intact)
        seq_lengths = [len(seq) for seq in sequences_clean]
        if len(set(seq_lengths)) > 1:
            raise ValueError(
                f"Sequences have different lengths: {set(seq_lengths)}"
            )

        n_species = len(names)
        n_chars = len(sequences_clean[0])

        # Calculate n_sites
        if seqtype == 'codon':
            if n_chars % 3 != 0:
                raise ValueError(f"Codon sequence length {n_chars} not divisible by 3")
            n_sites = n_chars // 3
        else:
            n_sites = n_chars

        # Encode sequences
        if seqtype == 'codon':
            encoded = cls._encode_codons(sequences_clean)
        elif seqtype == 'dna':
            encoded = cls._encode_nucleotides(sequences_clean)
        elif seqtype == 'aa':
            encoded = cls._encode_amino_acids(sequences_clean)
        else:
            raise ValueError(f"Unknown seqtype: {seqtype}")

        return cls(
            names=names,
            sequences=encoded,
            n_species=n_species,
            n_sites=n_sites,
            seqtype=seqtype,
        )

    @staticmethod
    def _encode_codons(sequences: list[str]) -> np.ndarray:
        """
        Encode codon sequences as integer arrays.

        Parameters
        ----------
        sequences : list[str]
            List of nucleotide sequences (length divisible by 3)

        Returns
        -------
        encoded : ndarray, shape (n_sequences, n_codons)
            Encoded sequences where each element is a codon index (0-60)
            or GAP_CODE (64) for gaps (---) or UNKNOWN_CODE (-1) for invalid
        """
        n_sequences = len(sequences)
        n_codons = len(sequences[0]) // 3

        encoded = np.zeros((n_sequences, n_codons), dtype=np.int8)

        for i, seq in enumerate(sequences):
            for j in range(n_codons):
                codon = seq[j * 3 : j * 3 + 3]
                if codon == '---':
                    # Gap codon (missing data)
                    encoded[i, j] = GAP_CODE
                elif codon in CODON_TO_INDEX:
                    # Valid codon
                    encoded[i, j] = CODON_TO_INDEX[codon]
                else:
                    # Stop codon or invalid - mark as unknown
                    encoded[i, j] = UNKNOWN_CODE

        return encoded

    @staticmethod
    def _encode_nucleotides(sequences: list[str]) -> np.ndarray:
        """Encode DNA sequences as integer arrays (0=T, 1=C, 2=A, 3=G)."""
        n_sequences = len(sequences)
        n_sites = len(sequences[0])

        encoded = np.zeros((n_sequences, n_sites), dtype=np.int8)

        for i, seq in enumerate(sequences):
            for j, nucleotide in enumerate(seq):
                if nucleotide in NUCLEOTIDE_TO_INDEX:
                    encoded[i, j] = NUCLEOTIDE_TO_INDEX[nucleotide]
                else:
                    encoded[i, j] = -1  # Unknown/ambiguous

        return encoded

    @staticmethod
    def _encode_amino_acids(sequences: list[str]) -> np.ndarray:
        """Encode amino acid sequences as integer arrays."""
        n_sequences = len(sequences)
        n_sites = len(sequences[0])

        encoded = np.zeros((n_sequences, n_sites), dtype=np.int8)

        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if aa in AA_TO_INDEX:
                    encoded[i, j] = AA_TO_INDEX[aa]
                else:
                    encoded[i, j] = -1  # Unknown

        return encoded

    def to_phylip(self, filepath: Path | str) -> None:
        """
        Write alignment to PHYLIP format file.

        Parameters
        ----------
        filepath : Path or str
            Output file path
        """
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            # Header
            n_chars = self.n_sites * (3 if self.seqtype == 'codon' else 1)
            f.write(f" {self.n_species}   {n_chars}\n\n")

            # Sequences
            for name, encoded_seq in zip(self.names, self.sequences):
                f.write(f"{name}\n")

                # Decode sequence
                if self.seqtype == 'codon':
                    seq = ''.join(INDEX_TO_CODON[idx] for idx in encoded_seq)
                elif self.seqtype == 'dna':
                    seq = ''.join(INDEX_TO_NUCLEOTIDE[idx] for idx in encoded_seq)
                elif self.seqtype == 'aa':
                    seq = ''.join(INDEX_TO_AA[idx] for idx in encoded_seq)

                # Write in blocks of 60
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + '\n')

                f.write('\n')

    def to_fasta(self, filepath: Path | str) -> None:
        """
        Write alignment to FASTA format file.

        Parameters
        ----------
        filepath : Path or str
            Output file path
        """
        filepath = Path(filepath)

        with open(filepath, 'w') as f:
            for name, encoded_seq in zip(self.names, self.sequences):
                f.write(f">{name}\n")

                # Decode sequence
                if self.seqtype == 'codon':
                    seq = ''.join(INDEX_TO_CODON[idx] for idx in encoded_seq)
                elif self.seqtype == 'dna':
                    seq = ''.join(INDEX_TO_NUCLEOTIDE[idx] for idx in encoded_seq)
                elif self.seqtype == 'aa':
                    seq = ''.join(INDEX_TO_AA[idx] for idx in encoded_seq)

                # Write in blocks of 60
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + '\n')

    def __repr__(self) -> str:
        return (
            f"Alignment(n_species={self.n_species}, n_sites={self.n_sites}, "
            f"seqtype='{self.seqtype}')"
        )
