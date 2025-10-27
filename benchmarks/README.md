# crabML vs PAML Benchmark Suite

Comprehensive validation of crabML implementation by comparing parameter estimates and likelihoods against PAML using simulated data with known parameters.

## Overview

This benchmark suite:
1. Simulates sequences under known parameters using crabML's simulation module
2. Analyzes each dataset with both PAML and crabML
3. Compares results to validate implementation correctness
4. Visualizes accuracy and performance metrics

## Directory Structure

```
benchmarks/
├── README.md                   # This file
├── config.yaml                 # Configuration (models, replicates, etc.)
├── run_benchmark.py           # Main orchestration script
├── lib/
│   ├── __init__.py
│   ├── simulator.py           # Generate simulated datasets
│   ├── paml_runner.py        # PAML execution and parsing
│   ├── crabml_runner.py      # crabML execution wrapper
│   ├── comparator.py         # Statistical comparison
│   └── visualizer.py         # Plotting functions
├── templates/
│   └── codeml.ctl.template   # PAML control file template
├── data/                      # Simulated sequences (gitignored)
│   ├── trees/                # Shared tree files
│   └── {model}/              # Per-model datasets
│       ├── rep001.fasta
│       ├── rep001.params.json
│       └── ...
├── results/                   # Analysis outputs (gitignored)
│   ├── paml/                 # PAML raw outputs
│   ├── crabml/               # crabML raw outputs
│   ├── comparison.csv        # Combined results table
│   └── summary_stats.json    # Summary statistics
└── plots/                     # Visualizations (committed)
    ├── {model}_lnL.png       # Per-model likelihood comparisons
    ├── {model}_params.png    # Per-model parameter comparisons
    └── summary.png           # Overall summary
```

## Simulation Design

### Fixed Parameters
- **Number of taxa**: 10 per tree
- **Tree topologies**: 3 fixed topologies, varied branch lengths
- **Branch length distribution**: Exponential
- **Total tree length**: Uniform(0.5, 5.0)
- **Sequence lengths**: Stratified: 200, 400, 600 codons
- **Genetic code**: 0 (universal)
- **Codon frequencies**: Uniform (1/61 per codon)
- **Random seeds**: Sequential (42, 43, 44, ...)

### Replicates Per Model
- 30 replicates per model
- Evenly distributed across:
  - 3 tree topologies × 10 replicates each
  - 3 sequence lengths (200/400/600) × 10 replicates each

### Parameter Ranges

**M0 (One-ratio)**:
```
kappa: uniform(1.5, 4.0)
omega: uniform(0.25, 5.0)  # Includes purifying, neutral, positive
```

**M1a (Nearly Neutral)**:
```
kappa: uniform(1.5, 4.0)
p0: uniform(0.6, 0.85)
omega0: uniform(0.05, 0.3)
```

**M2a (Positive Selection)**:
```
kappa: uniform(1.5, 4.0)
p0: uniform(0.4, 0.6)
p1: uniform(0.2, 0.35)
omega0: uniform(0.05, 0.3)
omega2: uniform(1.5, 4.0)
```

**M7 (Beta)**:
```
kappa: uniform(1.5, 4.0)
p: uniform(0.5, 3.0)
q: uniform(1.0, 8.0)
ncateg: 10 (fixed)
```

**M8 (Beta + Positive)**:
```
kappa: uniform(1.5, 4.0)
p0: uniform(0.7, 0.9)
p: uniform(0.5, 3.0)
q: uniform(1.0, 8.0)
omega_s: uniform(1.5, 4.0)
ncateg: 10 (fixed)
```

**M8a (Beta + Neutral)**:
```
kappa: uniform(1.5, 4.0)
p0: uniform(0.7, 0.9)
p: uniform(0.5, 3.0)
q: uniform(1.0, 8.0)
omega_s: 1.0 (fixed)
ncateg: 10 (fixed)
```

## Workflow

### Phase 1: Simulation
```bash
python run_benchmark.py simulate --models M0 M1a M2a M7 M8 M8a
```

For each model:
- Sample 30 parameter sets
- Generate sequences using `crabml simulate`
- Save: FASTA, params.json, tree files
- Record metadata (seed, true parameters)

### Phase 2: PAML Analysis
```bash
python run_benchmark.py run-paml --models M0 M1a M2a M7 M8 M8a --parallel
```

For each dataset:
- Generate PAML control file
- Run `codeml` with 60s timeout
- Parse output: lnL, parameters, convergence
- Save structured JSON
- Run in parallel (30 jobs simultaneously on 80-core machine)

### Phase 3: crabML Analysis
```bash
python run_benchmark.py run-crabml --models M0 M1a M2a M7 M8 M8a --sequential
```

For each dataset:
- Run `crabml fit -m {MODEL}`
- Record runtime, lnL, parameters
- Save structured JSON
- Run sequentially for accurate performance measurement
- Each run uses full multi-threading (Rayon default)

### Phase 4: Comparison
```bash
python run_benchmark.py compare
```

Statistical analysis:
- Match PAML/crabML results by dataset
- Calculate differences, correlations, RMSE
- Identify outliers (> 3σ)
- Export comparison.csv and summary_stats.json

### Phase 5: Visualization
```bash
python run_benchmark.py visualize
```

Generate plots:
- Per-model likelihood scatter plots (PAML vs crabML)
- Per-model parameter comparison grids
- Overall correlation heatmap
- Convergence rate comparison
- Runtime boxplots

## Parallelization Strategy

**PAML**:
- Single-threaded
- Run 30 instances in parallel via GNU parallel
- Utilizes 30/80 cores simultaneously

**crabML**:
- Multi-threaded via Rust/Rayon
- Run sequentially for accurate timing
- Each run uses full parallelism (~8-16 threads effective for M7/M8)
- Records single-dataset performance

## Success Criteria

✅ **Excellent likelihood agreement**: r > 0.999, mean |Δ| < 0.1 lnL units
✅ **Good parameter recovery**: r > 0.98 for all parameters
✅ **Equal or better convergence**: crabML converges ≥ PAML rate
✅ **Clear visualizations**: Obvious 1:1 correlation in scatter plots

## Output Format

**comparison.csv** (excerpt):
```csv
model,replicate,tree_id,seq_length,true_kappa,true_omega,paml_lnL,crabml_lnL,paml_kappa,crabml_kappa,paml_omega,crabml_omega,paml_converged,crabml_converged,paml_time,crabml_time
M0,001,tree1,300,2.34,0.45,-1234.56,-1234.58,2.35,2.36,0.44,0.45,True,True,5.2,0.8
```

**summary_stats.json** (excerpt):
```json
{
  "M0": {
    "n_replicates": 30,
    "n_converged_both": 29,
    "lnL_correlation": 0.9999,
    "lnL_rmse": 0.05,
    "lnL_mean_abs_diff": 0.03,
    "kappa_correlation": 0.998,
    "omega_correlation": 0.997,
    "paml_convergence_rate": 0.97,
    "crabml_convergence_rate": 0.97,
    "crabml_speedup_median": 6.5
  }
}
```

## Usage Examples

```bash
# Run complete benchmark for all models
python run_benchmark.py all

# Run specific phases
python run_benchmark.py simulate
python run_benchmark.py run-paml --parallel
python run_benchmark.py run-crabml
python run_benchmark.py compare
python run_benchmark.py visualize

# Run for specific models only
python run_benchmark.py all --models M0 M1a M2a

# Resume after failure
python run_benchmark.py run-paml --resume

# Clean and restart
python run_benchmark.py clean
python run_benchmark.py all
```

## Error Handling

- **PAML not found**: Check at startup, provide installation instructions
- **Convergence failures**: Log, mark in results, continue
- **Timeouts**: Kill after 60s, mark as failed
- **Invalid output**: Log parse error, continue
- **Resume**: Skip completed runs based on existing output files

## Requirements

- Python 3.11+
- crabML (installed in development mode)
- PAML 4.10.9+ (located in `paml/` directory)
- Python packages: numpy, pandas, matplotlib, seaborn, pyyaml
