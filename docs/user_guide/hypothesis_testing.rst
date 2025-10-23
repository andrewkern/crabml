Hypothesis Testing
==================

crabML provides publication-ready hypothesis tests for detecting natural selection
using likelihood ratio tests (LRT).

Quick Testing for Positive Selection
-------------------------------------

The simplest way to test for positive selection:

.. code-block:: python

   from crabml import positive_selection

   results = positive_selection(
       alignment='alignment.fasta',
       tree='tree.nwk',
       test='both'  # Runs M1a vs M2a and M7 vs M8
   )

   # Check M1a vs M2a
   if results['M1a_vs_M2a'].significant(0.05):
       print("Positive selection detected!")
       omega = results['M1a_vs_M2a'].omega_positive
       print(f"ω for positively selected sites: {omega:.2f}")

   # Check M7 vs M8 (more conservative)
   if results['M7_vs_M8'].significant(0.05):
       print("Positive selection confirmed by M7 vs M8")

Site-Class Model Tests
-----------------------

M1a vs M2a: Nearly Neutral vs Positive Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard test for positive selection.

**Null model (M1a)**: Two site classes
  - ω₀ < 1 (purifying selection)
  - ω₁ = 1 (neutral evolution)

**Alternative model (M2a)**: Three site classes
  - ω₀ < 1 (purifying)
  - ω₁ = 1 (neutral)
  - ω₂ > 1 (positive selection)

.. code-block:: python

   from crabml import m1a_vs_m2a

   result = m1a_vs_m2a('alignment.fasta', 'tree.nwk')

   # View formatted summary
   print(result.summary())

   # Access specific values
   print(f"LRT statistic: {result.LRT:.2f}")
   print(f"P-value: {result.pvalue:.6f}")
   print(f"Degrees of freedom: {result.df}")

   if result.significant(0.05):
       print(f"Proportion of sites under selection: {result.p_positive:.1%}")
       print(f"ω for positively selected sites: {result.omega_positive:.2f}")

**Output example:**

.. code-block:: text

   ================================================================================
   Likelihood Ratio Test for Positive Selection
   ================================================================================

   Test: M1a vs M2a

   NULL MODEL (M1a):
     Log-likelihood: -902.503872
     Parameters:
       p0 = 0.4923 (proportion ω < 1)
       ω0 = 0.0538
       κ  = 2.2945

   ALTERNATIVE MODEL (M2a):
     Log-likelihood: -899.998568
     Parameters:
       p2 = 0.2075 (proportion ω > 1)
       ω2 = 3.4472

   LIKELIHOOD RATIO TEST:
     LRT statistic: 5.0106
     Degrees of freedom: 2
     P-value: 0.0817

   CONCLUSION:
     No significant evidence for positive selection (α = 0.05)
   ================================================================================

M7 vs M8: Beta vs Beta + Omega
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more conservative test using beta distributions.

**Null model (M7)**: Beta distribution constrained to 0 < ω < 1

**Alternative model (M8)**: Beta distribution + additional class with ω > 1

.. code-block:: python

   from crabml import m7_vs_m8

   result = m7_vs_m8('alignment.fasta', 'tree.nwk')

   if result.significant(0.01):  # More stringent threshold
       print("Strong evidence for positive selection")

**Interpretation:**

- More conservative than M1a vs M2a
- Better for datasets with complex omega distributions
- Less prone to false positives

M8a vs M8: Null Test with 50:50 Mixture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative null model fixing the selection class at ω = 1.

**Null model (M8a)**: Beta + ω = 1 class

**Alternative model (M8)**: Beta + ω > 1 class

.. code-block:: python

   from crabml import m8a_vs_m8

   result = m8a_vs_m8('alignment.fasta', 'tree.nwk')

   # This test uses 50:50 mixture of chi-square distributions
   print(f"P-value (50:50 mixture): {result.pvalue:.6f}")

**Note**: This test accounts for the boundary constraint (ω = 1) using a 50:50
mixture of χ² distributions with df=0 and df=1.

Branch-Site Tests
-----------------

Branch-Site Model A Test
~~~~~~~~~~~~~~~~~~~~~~~~

Tests for positive selection on specific phylogenetic lineages.

.. code-block:: python

   from crabml import branch_site_test

   # Tree with branch labels: #0 = background, #1 = foreground
   tree_str = "((human,chimp) #1, (mouse,rat) #0);"

   result = branch_site_test(
       alignment='alignment.fasta',
       tree=tree_str
   )

   print(result.summary())

   if result.significant(0.05):
       omega2 = result.alt_params['omega2']
       p2 = result.foreground_positive_proportion
       print(f"Positive selection on foreground branches!")
       print(f"ω₂ = {omega2:.2f} (dN/dS for selected sites)")
       print(f"{p2:.1%} of sites under selection")

**Use cases:**

- Detecting adaptive evolution after gene duplication
- Testing for selection on primate-specific lineages
- Identifying sites under selection in specific clades

**Interpreting results:**

The test compares:

- **Null**: ω₂ = 1 on foreground branches
- **Alternative**: ω₂ free to vary on foreground branches

Significant result means some sites experience positive selection (ω > 1)
specifically on the foreground lineage.

Branch Model Tests
------------------

Multi-Ratio vs One-Ratio
~~~~~~~~~~~~~~~~~~~~~~~~~

Tests whether different lineages have different average selection pressures.

.. code-block:: python

   from crabml import branch_model_test

   tree_str = "((human,chimp) #1, (mouse,rat) #0);"

   result = branch_model_test(
       alignment='alignment.fasta',
       tree=tree_str
   )

   if result.significant(0.05):
       omega_fg = result.alt_params['omega1']
       omega_bg = result.alt_params['omega0']
       print(f"Different selection on foreground (ω={omega_fg:.3f}) "
             f"vs background (ω={omega_bg:.3f})")

Free-Ratio Test
~~~~~~~~~~~~~~~

Tests whether each branch has a different omega (exploratory).

.. code-block:: python

   from crabml import free_ratio_test

   result = free_ratio_test('alignment.fasta', 'tree.nwk')

   if result.significant(0.05):
       print("Significant omega variation across branches")
       # View branch-specific omegas
       print(result.alt_result.omega_dict)

**Warning**: This test is highly parameter-rich and prone to overfitting.
Use with caution and interpret conservatively.

Understanding the Results
-------------------------

LRTResult Object
~~~~~~~~~~~~~~~~

All hypothesis tests return an ``LRTResult`` object with:

.. code-block:: python

   result = m1a_vs_m2a(align, tree)

   # Test statistics
   result.LRT          # Likelihood ratio test statistic
   result.df           # Degrees of freedom
   result.pvalue       # P-value from chi-square distribution
   result.significant(alpha)  # Boolean, is p-value < alpha?

   # Model results
   result.null_result  # Full null model result
   result.alt_result   # Full alternative model result

   # Convenience properties (model-dependent)
   result.omega_positive      # ω for positive selection class
   result.p_positive          # Proportion under positive selection

   # Export
   result.to_dict()    # Dictionary representation
   result.to_json()    # JSON export
   result.summary()    # Formatted text summary

Statistical Considerations
--------------------------

Significance Thresholds
~~~~~~~~~~~~~~~~~~~~~~~

**Standard threshold**: α = 0.05

**More conservative** (for multiple testing): α = 0.01 or Bonferroni correction

.. code-block:: python

   # Multiple testing correction
   n_tests = 5
   alpha_bonferroni = 0.05 / n_tests

   results = [
       m1a_vs_m2a(align1, tree),
       m1a_vs_m2a(align2, tree),
       m1a_vs_m2a(align3, tree),
       m1a_vs_m2a(align4, tree),
       m1a_vs_m2a(align5, tree),
   ]

   significant = [r for r in results if r.significant(alpha_bonferroni)]
   print(f"{len(significant)}/{n_tests} genes with positive selection")

Degrees of Freedom
~~~~~~~~~~~~~~~~~~

Different tests have different degrees of freedom:

- **M1a vs M2a**: df = 2 (adds p2 and ω₂)
- **M7 vs M8**: df = 2 (adds p0 and ω_s)
- **M8a vs M8**: df = 1 (only ω changes from 1 to free)
- **Branch-site**: df = 1 (only ω₂ changes from 1 to free)
- **Multi-ratio vs M0**: df = k-1 (k = number of branch labels)
- **Free-ratio vs M0**: df = n-2 (n = number of branches)

Multiple Testing
~~~~~~~~~~~~~~~~

When testing multiple genes or multiple hypotheses:

1. **Bonferroni correction**: Divide α by number of tests
2. **FDR control**: Use Benjamini-Hochberg procedure
3. **Permutation tests**: For small sample sizes

.. code-block:: python

   from scipy.stats import false_discovery_control

   pvalues = [result.pvalue for result in results]
   significant_fdr = false_discovery_control(pvalues, alpha=0.05)

Power Considerations
~~~~~~~~~~~~~~~~~~~~

**Factors affecting power:**

- Alignment length (more sites = more power)
- Number of sequences (more phylogenetic signal)
- Strength of selection (larger ω easier to detect)
- Proportion of sites under selection

**Recommendations:**

- Minimum ~100 codons for reliable detection
- At least 8-10 sequences for good phylogenetic coverage
- Branch-site tests need sufficient foreground branch length

Best Practices
--------------

1. **Run multiple tests**: Both M1a vs M2a AND M7 vs M8
2. **Check convergence**: Ensure optimization converged
3. **Visualize results**: Plot omega distributions, site-specific posteriors
4. **Biological validation**: Check if selected sites make biological sense
5. **Multiple datasets**: Test across related genes/species

Publication Checklist
---------------------

When reporting hypothesis test results:

☐ Report both test names (e.g., "M1a vs M2a")

☐ Report LRT statistic, df, and p-value

☐ Report parameter estimates for both models

☐ State significance threshold used (α = ?)

☐ Report proportion of sites under selection

☐ Report ω estimate for selection class

☐ Describe biological interpretation

☐ Provide alignment statistics (n sequences, n codons)

Example for Methods Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   We tested for positive selection using crabML v0.2.0 (Kern, 2025),
   which implements the models of Yang et al. (2000). We performed
   likelihood ratio tests comparing M1a (nearly neutral) vs M2a
   (positive selection) and M7 (beta) vs M8 (beta + ω) models.
   Significance was assessed using a chi-square distribution with
   α = 0.05. For genes showing significant evidence of positive
   selection, we report the proportion of sites under selection and
   the estimated ω for the positively selected class.

Common Pitfalls
---------------

**1. Overfitting with small datasets**

   Don't use complex models (M8, branch-site) with < 50 codons

**2. Ignoring convergence warnings**

   Always check that optimization converged properly

**3. P-hacking**

   Don't test many models and only report significant ones

**4. Ignoring biological context**

   Statistical significance ≠ biological significance

**5. Wrong null model**

   Ensure you're using the appropriate null (M8a for M8, not M7)
