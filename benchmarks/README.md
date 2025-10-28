# crabML Benchmarking Suite

This directory contains infrastructure for validating crabML against PAML using simulated data.

## Overview

The benchmarking suite:
- Generates simulated sequence alignments with known parameters using `evolver`
- Runs both PAML and crabML on identical datasets
- Compares likelihood values and parameter estimates
- Generates comprehensive visualizations

## Requirements

- PAML installed and `codeml` available in PATH
- Python dependencies (installed via `uv`)
- `evolver` from PAML package (for data generation)

## Quick Start

```bash
# Generate simulated data (30 replicates per model)
uv run python run_benchmark.py generate --models M0 M1a M2a M7 M8 M8a

# Run PAML analysis
uv run python run_benchmark.py run-paml --models M0 M1a M2a M7 M8 M8a

# Run crabML analysis
uv run python run_benchmark.py run-crabml --models M0 M1a M2a M7 M8 M8a

# Generate comparison statistics and visualizations
uv run python run_benchmark.py compare
uv run python run_benchmark.py visualize
```

## Output Structure

```
benchmarks/
├── data/           # Simulated datasets (gitignored)
│   ├── M0/        # Model-specific replicates
│   ├── M1a/
│   └── ...
├── paml_runs/     # PAML output files (gitignored)
├── crabml_runs/   # crabML output files (gitignored)
├── results/       # Comparison statistics (gitignored)
│   ├── comparison.csv
│   └── summary_stats.json
└── plots/         # Visualizations (gitignored)
    ├── M0_lnL.png
    ├── M0_params.png
    ├── runtime_comparison.png
    └── ...
```

## Key Results

From validation with 30 replicates per model (6 models × 30 = 180 total):

### Likelihood Accuracy
- All models show perfect correlation (r > 0.999) between PAML and crabML
- RMSE ranges from 0.0000 (M0) to 0.0588 (M8)
- Mean absolute difference < 0.03 lnL units across all models

### Performance
- M0: 7.11x speedup over PAML
- M8: 11.45x speedup over PAML
- M8a: 8.29x speedup over PAML

### Parameter Estimation
- Kappa (transition/transversion ratio): Excellent agreement across all models
- Omega (dN/dS) parameters: Agreement within numerical precision
- Known identifiability issues in M1a and M2a are replicated in both implementations (validates correctness)

## Models Validated

- **M0**: One-ratio model
- **M1a**: NearlyNeutral (2 classes: ω<1, ω=1)
- **M2a**: PositiveSelection (3 classes: ω<1, ω=1, ω>1)
- **M7**: Beta distribution for ω
- **M8**: Beta&ω (beta distribution + positive selection class)
- **M8a**: Beta&ω=1 (null model for M8)

## Implementation Notes

### Data Generation
- Uses PAML's `evolver` to simulate sequences under each model
- 30 replicates per model with varying parameters
- Tree topology varies across replicates

### Visualization
- Likelihood comparison: PAML vs crabML scatter plots
- Parameter estimation: Correlation plots for all estimated parameters
- Runtime comparison: Speedup analysis with box plots
- Aggregated analysis: All models on single plot with color-coding

### Known Identifiability Issues
- **M1a**: Flat likelihood surface when ω0 ≈ 1 causes parameter non-identifiability
- **M2a**: Label switching between neutral and positive selection classes
- Both implementations show identical issues, validating correctness

## Citation

If you use these benchmarking results, please cite both:
- crabML: [citation pending]
- PAML: Yang, Z. (2007) PAML 4: Phylogenetic Analysis by Maximum Likelihood
