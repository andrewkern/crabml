Quick Start Guide
=================

This guide will get you up and running with crabML in minutes.

Your First Analysis
-------------------

Let's fit a simple M0 (one-ratio) model to an alignment:

.. code-block:: python

   from crabml import optimize_model

   # Fit M0 model
   result = optimize_model("M0", "alignment.fasta", "tree.nwk")

   # View results
   print(result.summary())

   # Access specific parameters
   print(f"omega (dN/dS) = {result.omega:.4f}")
   print(f"kappa (ts/tv) = {result.kappa:.4f}")
   print(f"log-likelihood = {result.lnL:.2f}")

The output will look like:

.. code-block:: text

   ======================================================================
   MODEL: M0
   ======================================================================

   Log-likelihood:       -906.017441
   Number of parameters: 13

   PARAMETERS:
     kappa (ts/tv) = 4.5402
     omega (dN/dS) = 0.8066

   TREE:
     7 sequences
     11 branches (optimized)

   ======================================================================

Testing for Positive Selection
-------------------------------

The most common analysis is testing for positive selection. crabML makes this easy:

.. code-block:: python

   from crabml import positive_selection

   # Run both standard tests
   results = positive_selection(
       alignment='alignment.fasta',
       tree='tree.nwk',
       test='both'  # Runs M1a vs M2a and M7 vs M8
   )

   # Check M1a vs M2a test
   m1a_m2a = results['M1a_vs_M2a']
   print(m1a_m2a.summary())

   if m1a_m2a.significant(0.05):
       print("Positive selection detected!")
       print(f"ω for positively selected sites: {m1a_m2a.omega_positive:.2f}")

Individual Tests
~~~~~~~~~~~~~~~~

You can also run individual tests:

.. code-block:: python

   from crabml import m1a_vs_m2a, m7_vs_m8

   # M1a (nearly neutral) vs M2a (positive selection)
   result = m1a_vs_m2a('alignment.fasta', 'tree.nwk')
   print(f"P-value: {result.pvalue:.6f}")

   # M7 (beta distribution) vs M8 (beta + omega > 1)
   result = m7_vs_m8('alignment.fasta', 'tree.nwk')
   print(f"LRT statistic: {result.LRT:.2f}")

Working with Different Model Types
-----------------------------------

Site-Class Models
~~~~~~~~~~~~~~~~~

Site-class models allow omega (dN/dS) to vary across sites:

.. code-block:: python

   from crabml import optimize_model

   # Simple models
   m0 = optimize_model("M0", align, tree)  # One omega for all sites

   # Models for testing positive selection
   m1a = optimize_model("M1a", align, tree)  # Nearly neutral
   m2a = optimize_model("M2a", align, tree)  # Positive selection

   # Beta distribution models
   m7 = optimize_model("M7", align, tree)   # Beta (omega < 1)
   m8 = optimize_model("M8", align, tree)   # Beta + omega > 1

   # Access site class information
   print(f"Site classes: {m2a.n_site_classes}")
   print(f"Omega values: {m2a.omegas}")
   print(f"Proportions: {m2a.proportions}")

Branch Models
~~~~~~~~~~~~~

Branch models allow omega to vary across lineages:

.. code-block:: python

   from crabml import optimize_branch_model

   # Tree with branch labels: #0 = background, #1 = foreground
   tree_str = "((human,chimp) #1, (mouse,rat) #0);"

   # Multi-ratio model (recommended)
   result = optimize_branch_model("multi-ratio", align, tree_str)
   print(f"Primate omega: {result.foreground_omega:.3f}")
   print(f"Rodent omega: {result.background_omega:.3f}")

   # Free-ratio model (exploratory)
   result = optimize_branch_model("free-ratio", align, tree)
   print(result.omega_dict)  # All branch-specific omegas

Branch-Site Models
~~~~~~~~~~~~~~~~~~

Branch-site models detect positive selection on specific sites and lineages:

.. code-block:: python

   from crabml import optimize_branch_site_model

   tree_str = "((human,chimp) #1, (mouse,rat) #0);"

   # Alternative model (omega2 free)
   alt = optimize_branch_site_model("model-a", align, tree_str)
   print(f"Positive selection omega: {alt.omega2:.3f}")
   print(f"Sites under selection: {alt.foreground_positive_proportion:.1%}")

   # Null model (omega2 = 1) for hypothesis testing
   null = optimize_branch_site_model("model-a", align, tree_str, fix_omega=True)

File Formats
------------

crabML automatically detects file formats:

**Alignments:**

* FASTA format (``.fa``, ``.fasta``)
* PHYLIP format (``.phy``)

**Trees:**

* Newick format in file (``.nwk``, ``.tree``)
* Newick string directly in code

Example:

.. code-block:: python

   # All of these work:
   result = optimize_model("M0", "data.fasta", "tree.nwk")
   result = optimize_model("M0", "data.phy", "tree.tree")
   result = optimize_model("M0", "data.fa", "((A,B),(C,D));")

Exporting Results
-----------------

Results can be exported to various formats:

.. code-block:: python

   result = optimize_model("M2a", align, tree)

   # Dictionary
   data = result.to_dict()

   # JSON file
   result.to_json("results.json")

   # Print summary
   print(result.summary())

What's Next?
------------

* :doc:`models` - Complete guide to all implemented models
* :doc:`hypothesis_testing` - Detailed guide to hypothesis testing
* :doc:`advanced` - Advanced features and customization
