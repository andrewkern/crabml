# Benchmark Data Directory

This directory contains simulated sequence datasets for validating crabML against PAML.

## Structure

Data is organized by model:
```
data/
├── M0/     # One-ratio model replicates
├── M1a/    # NearlyNeutral model replicates
├── M2a/    # PositiveSelection model replicates
├── M7/     # Beta model replicates
├── M8/     # Beta&omega model replicates
└── M8a/    # Beta&omega=1 model replicates
```

Each model directory contains 30 replicates:
```
M0/
├── rep001.fasta          # Codon alignment
├── rep001.nwk            # Newick tree file
├── rep001.params.json    # True simulation parameters
├── rep002.fasta
├── rep002.nwk
├── rep002.params.json
└── ...
```

## Generation

Data is generated using PAML's `evolver` tool:

```bash
uv run python run_benchmark.py generate --models M0 M1a M2a M7 M8 M8a
```

This creates 30 replicates per model with:
- Varying tree topologies (10-40 sequences)
- Random parameter values appropriate for each model
- Sufficient sequence length (300-900 codons) for parameter estimation

## File Formats

- **`.fasta`**: PHYLIP-formatted codon alignments
- **`.nwk`**: Newick tree with branch lengths
- **`.params.json`**: JSON with true parameter values used in simulation

## Note

These data files are gitignored to avoid repository bloat. Run the generation
command above to create them locally.
