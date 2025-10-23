Model Guide
===========

crabML implements 15 codon substitution models for detecting natural selection.
All models are fully validated against PAML reference outputs.

Model Categories
----------------

Models are organized into three categories based on how omega (dN/dS) varies:

1. **Site-class models**: omega varies across sites but not branches
2. **Branch models**: omega varies across branches but not sites
3. **Branch-site models**: omega varies across both sites and branches

Site-Class Models
-----------------

These models allow the ratio of non-synonymous to synonymous substitutions (ω = dN/dS)
to vary across codon sites.

M0: One-Ratio Model
~~~~~~~~~~~~~~~~~~~~

The simplest model with a single omega for all sites.

.. code-block:: python

   from crabml import optimize_model

   result = optimize_model("M0", "alignment.fasta", "tree.nwk")
   print(f"omega = {result.omega:.4f}")

**Use case**: Baseline model, average selection across gene

**Parameters**:
- kappa (transition/transversion ratio)
- omega (dN/dS)

M1a: Nearly Neutral
~~~~~~~~~~~~~~~~~~~

Two site classes: purifying selection (ω₀ < 1) and neutral evolution (ω₁ = 1).

.. code-block:: python

   result = optimize_model("M1a", "alignment.fasta", "tree.nwk")
   print(f"p0 = {result.params['p0']:.3f} (proportion ω < 1)")
   print(f"omega0 = {result.params['omega0']:.3f}")

**Use case**: Null model for M1a vs M2a test

**Parameters**:
- kappa
- p0 (proportion of sites with ω₀)
- omega0 (ω for purifying selection, 0 < ω₀ < 1)

M2a: Positive Selection
~~~~~~~~~~~~~~~~~~~~~~~~

Three site classes: purifying (ω₀ < 1), neutral (ω₁ = 1), and positive selection (ω₂ > 1).

.. code-block:: python

   result = optimize_model("M2a", "alignment.fasta", "tree.nwk")
   print(f"Omegas: {result.omegas}")  # [omega0, 1.0, omega2]
   print(f"Proportions: {result.proportions}")  # [p0, p1, p2]

   if result.params['omega2'] > 1:
       print(f"Positive selection detected: ω₂ = {result.params['omega2']:.2f}")

**Use case**: Detect positive selection (alternative to M1a)

**Parameters**:
- kappa
- p0, p1 (proportions, p2 = 1 - p0 - p1)
- omega0, omega2

M3: Discrete Model
~~~~~~~~~~~~~~~~~~

K discrete omega categories with estimated proportions.

.. code-block:: python

   result = optimize_model("M3", "alignment.fasta", "tree.nwk", K=3)
   print(f"Omegas: {result.omegas}")
   print(f"Proportions: {result.proportions}")

**Use case**: Exploratory analysis of omega variation

**Parameters**:
- kappa
- K omega values
- K-1 proportion parameters

M7: Beta Distribution
~~~~~~~~~~~~~~~~~~~~~

Beta distribution for omega constrained to (0, 1).

.. code-block:: python

   result = optimize_model("M7", "alignment.fasta", "tree.nwk")
   print(f"Beta parameters: p={result.params['p']:.2f}, q={result.params['q']:.2f}")

**Use case**: Null model for M7 vs M8 test (no positive selection)

**Parameters**:
- kappa
- p, q (beta distribution shape parameters)

M8: Beta + Omega
~~~~~~~~~~~~~~~~

Beta distribution (0 < ω < 1) plus an additional class allowing ω > 1.

.. code-block:: python

   result = optimize_model("M8", "alignment.fasta", "tree.nwk")

   if result.params['omega_s'] > 1:
       p_sel = result.params['p1']
       omega_sel = result.params['omega_s']
       print(f"{p_sel:.1%} of sites with ω = {omega_sel:.2f}")

**Use case**: Detect positive selection (alternative to M7)

**Parameters**:
- kappa
- p, q (beta distribution)
- p0 (proportion in beta)
- omega_s (omega for selection class)

M8a: Beta + Omega = 1
~~~~~~~~~~~~~~~~~~~~~

Like M8 but with the additional class fixed at ω = 1.

.. code-block:: python

   result = optimize_model("M8a", "alignment.fasta", "tree.nwk")

**Use case**: Null model for M8a vs M8 test

**Parameters**:
- kappa
- p, q (beta distribution)
- p0 (proportion in beta)

Additional Site Models
~~~~~~~~~~~~~~~~~~~~~~

**M4 (Frequencies)**: Five fixed omegas with variable proportions

**M5 (Gamma)**: Gamma distribution for omega

**M6 (2Gamma)**: Mixture of two gamma distributions

**M9 (Beta & Gamma)**: Mixture of beta and gamma distributions

.. code-block:: python

   m4 = optimize_model("M4", align, tree)
   m5 = optimize_model("M5", align, tree)
   m6 = optimize_model("M6", align, tree)
   m9 = optimize_model("M9", align, tree)

Branch Models
-------------

Branch models allow omega to vary across phylogenetic lineages.

Free-Ratio Model
~~~~~~~~~~~~~~~~

Estimates independent omega for each branch in the tree.

.. code-block:: python

   from crabml import optimize_branch_model

   result = optimize_branch_model("free-ratio", "alignment.fasta", "tree.nwk")

   # View all branch-specific omegas
   for branch, omega in result.omega_dict.items():
       print(f"{branch}: ω = {omega:.3f}")

**Use case**: Exploratory analysis of lineage-specific selection

**Warning**: Highly parameter-rich, prone to overfitting with small datasets

**Parameters**: One omega per branch (n-1 for n species)

Multi-Ratio Model
~~~~~~~~~~~~~~~~~

Different omega for labeled branch groups.

.. code-block:: python

   # Label branches in tree: #0 = background, #1 = foreground
   tree_str = "((human,chimp) #1, (mouse,rat) #0);"

   result = optimize_branch_model("multi-ratio", align, tree_str)

   print(f"Foreground ω (primates): {result.foreground_omega:.3f}")
   print(f"Background ω (rodents): {result.background_omega:.3f}")

**Use case**: Test for lineage-specific selection (recommended over free-ratio)

**Parameters**: One omega per unique branch label

Branch-Site Models
------------------

Branch-site models combine site variation with lineage-specific effects.

Branch-Site Model A
~~~~~~~~~~~~~~~~~~~

Four site classes with different omega on foreground vs background branches:

- **Class 0**: Conserved (ω₀ < 1) on all branches
- **Class 1**: Neutral (ω = 1) on all branches
- **Class 2a**: Conserved on background, positive selection (ω₂ > 1) on foreground
- **Class 2b**: Neutral on background, positive selection on foreground

.. code-block:: python

   from crabml import optimize_branch_site_model

   tree_str = "((human,chimp) #1, (mouse,rat) #0);"

   # Alternative model (ω₂ free)
   alt = optimize_branch_site_model("model-a", align, tree_str)

   print(f"ω₀ (conserved): {alt.omega0:.3f}")
   print(f"ω₂ (positive selection): {alt.omega2:.3f}")
   print(f"Sites under selection: {alt.foreground_positive_proportion:.1%}")
   print(f"Site class proportions: {alt.proportions}")

**Use case**: Detect positive selection on specific lineages

**Parameters**:
- kappa
- omega0, omega2
- p0, p1 (proportions)

Branch-Site Model A Null
~~~~~~~~~~~~~~~~~~~~~~~~~

Same as Model A but with ω₂ fixed to 1 for hypothesis testing.

.. code-block:: python

   null = optimize_branch_site_model("model-a", align, tree_str, fix_omega=True)

   assert null.omega2 == 1.0  # Fixed at neutral

**Use case**: Null model for branch-site test

Model Selection
---------------

Choosing the Right Model
~~~~~~~~~~~~~~~~~~~~~~~~

**For detecting positive selection:**

1. Start with standard tests:
   - M1a vs M2a (site-class test)
   - M7 vs M8 (more conservative)

2. If you suspect lineage-specific selection:
   - Branch-site Model A

3. For exploratory analysis:
   - M3 (discrete omegas)
   - Free-ratio (branch-specific, use with caution)

**For estimating average selection:**

- M0 provides overall omega across gene

**For detailed omega distribution:**

- M7/M8 model beta distribution
- M3 estimates discrete categories

Computational Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Speed** (fastest to slowest):

1. M0 (one class)
2. M1a, M7 (two classes / beta)
3. M2a, M8, M8a (three classes)
4. M3, M9 (K classes)
5. Branch models (many omegas)
6. Branch-site models (complex structure)

**Sample size requirements:**

- M0, M1a, M2a: Work with small datasets (< 10 sequences)
- M7, M8: Benefit from moderate datasets (10-50 sequences)
- Branch models: Need sufficient phylogenetic diversity
- Branch-site models: Need both sequence diversity and alignment length

Model Comparison
----------------

Comparing Results
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from crabml import compare_results

   m0 = optimize_model("M0", align, tree)
   m1a = optimize_model("M1a", align, tree)
   m2a = optimize_model("M2a", align, tree)

   # Compare models
   comparison = compare_results([m0, m1a, m2a])
   print(comparison)

This shows log-likelihoods, parameters, and omega estimates side-by-side.

PAML Compatibility
------------------

All crabML models are validated against PAML:

- Log-likelihoods match within 0.01 units
- Parameter estimates match within 1% relative error
- Same model parameterizations as PAML
- Tested on multiple diverse datasets

You can directly compare crabML results with PAML output files.
