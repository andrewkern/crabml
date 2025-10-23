Changelog
=========

All notable changes to crabML will be documented in this file.

Version 0.2.1 (2025-10-23)
--------------------------

Added
~~~~~

* Specialized result classes for type safety

  * ``SiteModelResult`` for site-class models
  * ``BranchModelResult`` for branch models
  * ``BranchSiteModelResult`` for branch-site models

* New optimization functions

  * ``optimize_branch_model()`` for branch models
  * ``optimize_branch_site_model()`` for branch-site models

* Comprehensive Sphinx documentation

  * User guide with quickstart, models, hypothesis testing, and advanced topics
  * Full API reference
  * Installation instructions

Changed
~~~~~~~

* ``ModelResult`` is now an alias for ``SiteModelResult`` (backwards compatible)
* Improved docstrings across all modules
* Enhanced README with result class examples

Fixed
~~~~~

* Branch-site model parameter handling (``fix_omega`` instead of ``fix_omega2``)

Version 0.2.0 (2025-10-22)
--------------------------

Added
~~~~~

* **Simplified high-level API**

  * ``optimize_model()`` function with model name strings
  * ``ModelResult`` unified result class
  * Auto file format detection (FASTA/PHYLIP)

* **M0-first initialization**

  * Automatic initialization for all site-class models
  * Solves convergence issues on gapped data
  * Matches PAML's sequential optimization strategy

* **Comprehensive PAML validation**

  * 25 validation tests across 3 datasets
  * HIV NSsites: 9 tests (M0, M1a, M2a, M7, M8, M8a + LRTs)
  * lysin: 8 tests (same models + LRTs)
  * lysozyme: 8 tests (same models + LRTs)
  * All models match PAML within 0.01 log-likelihood units

* **Branch-site models**

  * Branch-Site Model A implementation
  * Null model with ω₂ = 1
  * Hypothesis test framework

* **Branch models**

  * Free-ratio model (independent ω per branch)
  * Multi-ratio model (ω per branch label)
  * Hypothesis tests

Changed
~~~~~~~

* Default behavior now uses M0 initialization for M1a, M2a, M7, M8, M8a
* Improved optimization convergence on complex datasets
* Better handling of gapped alignments

Fixed
~~~~~

* Gap handling in codon models
* Convergence issues with zero initial branch lengths

Version 0.1.0 (2025-10-15)
--------------------------

Initial release with core functionality.

Added
~~~~~

* **Site-class codon models**

  * M0 (one-ratio)
  * M1a (nearly neutral)
  * M2a (positive selection)
  * M3 (discrete)
  * M7 (beta)
  * M8 (beta & ω)
  * M8a (beta & ω=1)

* **Hypothesis testing framework**

  * M1a vs M2a test
  * M7 vs M8 test
  * M8a vs M8 test with 50:50 mixture
  * LRTResult class

* **High-performance Rust backend**

  * BLAS-accelerated matrix operations
  * Rayon parallelization
  * 300-500x faster than NumPy
  * 3-10x faster than PAML

* **Core features**

  * Sequence I/O (FASTA, PHYLIP)
  * Tree parsing (Newick)
  * F3x4 codon frequencies
  * Branch length optimization
  * Parameter optimization with scipy

* **Testing**

  * Comprehensive test suite
  * PAML validation on lysozyme dataset
  * Exact numerical agreement (< 1e-5 lnL difference)
