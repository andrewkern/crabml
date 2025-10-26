crabML Documentation
====================

**crabML** is a high-performance reimplementation of PAML's codeml for phylogenetic maximum likelihood analysis, powered by Rust.

Features
--------

* **Command-Line Interface**: Simple ``crabml`` command with four analysis modes

  - ``site-model``: Site-class model tests (M1a vs M2a, M7 vs M8)
  - ``branch-model``: Branch model tests (multi-ratio, free-ratio)
  - ``branch-site``: Branch-site model tests
  - ``fit``: Fit single models

* **Unified Python API**: Simple functions for all model types with specialized result classes
* **Site-class models**: M0, M1a, M2a, M3, M4, M5, M6, M7, M8, M8a, M9
* **Branch models**: Free-ratio and multi-ratio models for lineage-specific selection
* **Branch-site models**: Model A (test for positive selection on specific lineages)
* **Hypothesis testing**: Complete LRT framework for detecting positive selection
* **High-performance Rust backend**: 300-500x faster than NumPy, 3-10x faster than PAML
* **PAML validation**: All models produce exact numerical matches

Quick Start
-----------

Command-Line Interface:

.. code-block:: bash

   # Site-class model tests (positive selection)
   crabml site-model -s alignment.fasta -t tree.nwk --test both

   # Branch model tests (lineage-specific selection)
   crabml branch-model -s alignment.fasta -t labeled_tree.nwk --test multi-ratio

   # Branch-site model test (site + lineage selection)
   crabml branch-site -s alignment.fasta -t labeled_tree.nwk

   # Fit a single model
   crabml fit -m M0 -s alignment.fasta -t tree.nwk

Python API - Fit a single model:

.. code-block:: python

   from crabml import optimize_model

   result = optimize_model("M0", "alignment.fasta", "tree.nwk")
   print(result.summary())
   print(f"omega = {result.omega:.4f}")

Python API - Test for positive selection:

.. code-block:: python

   from crabml import positive_selection

   results = positive_selection("alignment.fasta", "tree.nwk", test="both")
   print(results['M1a_vs_M2a'].summary())

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/cli
   user_guide/models
   user_guide/hypothesis_testing
   user_guide/advanced

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/high_level
   api/analysis
   api/io
   api/optimize
   api/models
   api/core

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   contributing
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
