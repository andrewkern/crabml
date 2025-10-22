# Reference Implementations

This directory contains pure Python reference implementations of performance-critical code that has been optimized in Rust.

## Purpose

These implementations are kept for:

1. **Development and prototyping** - Python is easier to modify and experiment with when implementing new models
2. **Educational reference** - Python code clearly shows the mathematical operations
3. **Validation** - Independent implementations help verify correctness
4. **Debugging** - Easier to trace through Python code than Rust when investigating issues

## Contents

### likelihood.py

Pure Python implementation of the Felsenstein pruning algorithm for phylogenetic likelihood calculation.

This is the reference implementation that was used to validate the Rust backend. The Rust version in `paml/src/` provides identical results with 3-10x speedup for full optimizations and 10,000x speedup for individual likelihood evaluations.

## Usage

These implementations are **not** used in production code. The main package requires the Rust backend. If you want to prototype a new model:

1. Implement the model logic in `src/pycodeml/models/`
2. Test likelihood calculations using this Python reference if needed
3. Once validated, the existing Rust backend will automatically work (it uses the Q matrices from Python models)

For models requiring new likelihood calculation patterns, prototype here first, then port to Rust.
