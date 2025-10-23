Advanced Usage
==============

This guide covers advanced features and customization options for power users.

Direct Optimizer Access
-----------------------

For maximum control, you can use optimizer classes directly.

Using Optimizer Classes
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from crabml.io.sequences import Alignment
   from crabml.io.trees import Tree
   from crabml.optimize.optimizer import M2aOptimizer

   # Load data
   alignment = Alignment.from_fasta("alignment.fasta", seqtype='codon')
   tree = Tree.from_newick("tree.nwk")

   # Create optimizer
   optimizer = M2aOptimizer(
       alignment=alignment,
       tree=tree,
       use_f3x4=True,
       optimize_branch_lengths=True
   )

   # Run optimization
   kappa, p0, p1, omega0, omega2, lnL = optimizer.optimize(
       maxiter=1000,
       ftol=1e-6
   )

   print(f"Log-likelihood: {lnL:.6f}")
   print(f"ω₂ (positive selection): {omega2:.4f}")

Available Optimizers
~~~~~~~~~~~~~~~~~~~~

Site-class models:

.. code-block:: python

   from crabml.optimize.optimizer import (
       M0Optimizer,
       M1aOptimizer,
       M2aOptimizer,
       M3Optimizer,
       M7Optimizer,
       M8Optimizer,
       M8aOptimizer,
   )

Branch models:

.. code-block:: python

   from crabml.optimize.branch import BranchModelOptimizer

Branch-site models:

.. code-block:: python

   from crabml.optimize.branch_site import BranchSiteModelAOptimizer

Customizing Optimization
-------------------------

Optimization Parameters
~~~~~~~~~~~~~~~~~~~~~~~

All optimizers accept these common parameters:

.. code-block:: python

   result = optimize_model(
       "M2a",
       align,
       tree,
       # Optimization settings
       maxiter=2000,          # Maximum iterations (default: 1000)
       ftol=1e-7,             # Function tolerance (default: 1e-6)
       init_with_m0=True,     # Initialize with M0 (default: True)

       # Model settings
       use_f3x4=True,         # Use F3x4 codon frequencies (default: True)
       optimize_branch_lengths=True,  # Optimize branches (default: True)
   )

M0-First Initialization
~~~~~~~~~~~~~~~~~~~~~~~

crabML automatically initializes complex models using M0 results:

.. code-block:: python

   # This happens automatically
   optimizer = M2aOptimizer(alignment, tree, init_with_m0=True)

   # Optimizer first runs M0 to get good starting values
   # Then optimizes M2a parameters

**Benefits:**

- Better convergence on complex datasets
- Avoids local optima
- Faster overall optimization

**Disable if needed:**

.. code-block:: python

   # Skip M0 initialization (not recommended)
   result = optimize_model("M2a", align, tree, init_with_m0=False)

Codon Frequency Models
~~~~~~~~~~~~~~~~~~~~~~

crabML supports different codon frequency models:

**F3x4** (recommended, default):

.. code-block:: python

   result = optimize_model("M0", align, tree, use_f3x4=True)

- Estimates nucleotide frequencies at each codon position
- 9 free parameters (3 positions × 3 nucleotides)
- More realistic, better fit

**F61** (equal frequencies):

.. code-block:: python

   result = optimize_model("M0", align, tree, use_f3x4=False)

- Equal frequencies for all 61 sense codons
- No free parameters
- Faster but less accurate

Branch Length Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Control whether branch lengths are optimized:

.. code-block:: python

   # Optimize branch lengths (default, recommended)
   result = optimize_model("M0", align, tree, optimize_branch_lengths=True)

   # Fix branch lengths (use tree branch lengths as-is)
   result = optimize_model("M0", align, tree, optimize_branch_lengths=False)

**When to fix branch lengths:**

- Comparing models on same tree
- Branch lengths pre-estimated with high confidence
- Debugging parameter estimates

Working with I/O Classes
-------------------------

Alignment Class
~~~~~~~~~~~~~~~

.. code-block:: python

   from crabml.io.sequences import Alignment

   # Load from FASTA
   align = Alignment.from_fasta("sequences.fasta", seqtype='codon')

   # Load from PHYLIP
   align = Alignment.from_phylip("sequences.phy", seqtype='codon')

   # Access sequences
   print(f"Number of sequences: {len(align.sequences)}")
   print(f"Sequence length: {len(align.sequences[0])}")
   print(f"Sequence names: {align.names}")

   # Get specific sequence
   seq = align.get_sequence("Human")

   # Count gaps
   gap_count = sum(1 for codon in align.sequences[0] if '-' in codon)

Tree Class
~~~~~~~~~~

.. code-block:: python

   from crabml.io.trees import Tree

   # Load from file
   tree = Tree.from_newick("tree.nwk")

   # Parse from string
   tree_str = "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);"
   tree = Tree.from_newick(tree_str)

   # Access tree structure
   print(f"Number of leaves: {len([n for n in tree.postorder() if n.is_leaf()])}")

   # Traverse tree
   for node in tree.postorder():
       if node.is_leaf():
           print(f"Leaf: {node.name}")
       else:
           print(f"Internal node with {len(node.children)} children")

   # Get/set branch lengths
   for node in tree.postorder():
       if node.parent is not None:
           print(f"{node.name}: length = {node.dist}")

Branch Labels for Branch-Site Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Label format: #0 = background, #1 = foreground, #2, #3, etc.

   tree_str = """
   (
       (human:0.1, chimp:0.1) #1 :0.2,    # Primate foreground
       (
           (mouse:0.15, rat:0.15) #0 :0.2,  # Rodent background
           dog:0.3 #0                        # Carnivore background
       ) :0.1
   ) ;
   """

   tree = Tree.from_newick(tree_str)
   result = optimize_branch_site_model("model-a", align, tree)

Result Object Details
---------------------

SiteModelResult
~~~~~~~~~~~~~~~

For site-class models (M0, M1a, M2a, M7, M8, M8a):

.. code-block:: python

   result = optimize_model("M2a", align, tree)

   # Basic attributes
   result.model_name     # "M2A"
   result.lnL            # Log-likelihood
   result.kappa          # Transition/transversion ratio
   result.n_params       # Number of parameters

   # Model-specific properties
   result.omega          # Single omega (M0 only)
   result.omegas         # List of omegas [ω₀, 1.0, ω₂]
   result.proportions    # Site class proportions [p₀, p₁, p₂]
   result.n_site_classes # Number of site classes

   # Raw parameters
   result.params         # Dict of all parameters

   # Optimized data
   result.tree          # Tree with optimized branch lengths
   result.alignment     # Original alignment

BranchModelResult
~~~~~~~~~~~~~~~~~

For branch models (free-ratio, multi-ratio):

.. code-block:: python

   result = optimize_branch_model("multi-ratio", align, tree)

   # Branch-specific omegas
   result.omega_dict           # {'omega0': 0.5, 'omega1': 2.3}
   result.foreground_omega     # omega1 (label #1)
   result.background_omega     # omega0 (label #0)

   # All omegas
   for name, omega in result.omega_dict.items():
       print(f"{name}: {omega:.3f}")

BranchSiteModelResult
~~~~~~~~~~~~~~~~~~~~~

For branch-site models (Model A):

.. code-block:: python

   result = optimize_branch_site_model("model-a", align, tree)

   # Site class omegas
   result.omega0                      # Conserved class
   result.omega2                      # Positive selection class

   # Proportions
   result.proportions                 # [p₀, p₁, p₂a, p₂b]
   result.foreground_positive_proportion  # p₂a + p₂b

   # Parameters
   result.params['p0']               # Proportion class 0
   result.params['p1']               # Proportion class 1

Performance Optimization
------------------------

Using Multiple Cores
~~~~~~~~~~~~~~~~~~~~

crabML automatically uses multiple cores for site-class models:

.. code-block:: python

   # Site classes are evaluated in parallel automatically
   result = optimize_model("M8", align, tree)

   # No configuration needed - Rust backend uses Rayon parallelism

For batch analysis of multiple genes:

.. code-block:: python

   from multiprocessing import Pool

   def analyze_gene(align_file):
       return optimize_model("M2a", align_file, tree)

   with Pool(8) as pool:
       results = pool.map(analyze_gene, alignment_files)

Memory Considerations
~~~~~~~~~~~~~~~~~~~~~

For large alignments:

.. code-block:: python

   # Memory usage ~ O(n_sequences × n_sites × n_site_classes)

   # For 100 sequences × 1000 codons × 10 site classes:
   # ~8 MB for likelihood arrays
   # ~1 MB for parameters
   # Total: ~10-20 MB per optimization

Debugging and Troubleshooting
------------------------------

Convergence Issues
~~~~~~~~~~~~~~~~~~

If optimization doesn't converge:

.. code-block:: python

   # Increase iterations
   result = optimize_model("M8", align, tree, maxiter=5000)

   # Relax tolerance
   result = optimize_model("M8", align, tree, ftol=1e-5)

   # Ensure M0 initialization is enabled
   result = optimize_model("M8", align, tree, init_with_m0=True)

Checking Convergence
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = optimize_model("M2a", align, tree)

   if result.convergence_info:
       print(f"Converged: {result.convergence_info.get('success')}")
       print(f"Message: {result.convergence_info.get('message')}")
       print(f"Iterations: {result.convergence_info.get('nit')}")

Comparing with PAML
~~~~~~~~~~~~~~~~~~~

To verify results match PAML:

.. code-block:: python

   # Run crabML
   result = optimize_model("M2a", "alignment.phy", "tree.nwk")

   print(f"crabML lnL: {result.lnL:.6f}")
   print(f"crabML κ: {result.kappa:.4f}")
   print(f"crabML ω₂: {result.params['omega2']:.4f}")

   # Compare with PAML mlc output
   # Differences should be < 0.01 log-likelihood units

Custom Analysis Workflows
--------------------------

Batch Processing Multiple Genes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import json

   # Process multiple alignment files
   align_dir = Path("alignments/")
   results = {}

   for align_file in align_dir.glob("*.fasta"):
       gene_name = align_file.stem

       try:
           result = optimize_model("M2a", str(align_file), "tree.nwk")
           results[gene_name] = {
               'lnL': result.lnL,
               'omega2': result.params.get('omega2'),
               'p2': result.proportions[2] if result.proportions else None
           }
       except Exception as e:
           print(f"Error processing {gene_name}: {e}")

   # Save results
   with open("results.json", "w") as f:
       json.dump(results, f, indent=2)

Comparing Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   models = ["M0", "M1a", "M2a", "M7", "M8"]
   results = {}

   for model_name in models:
       result = optimize_model(model_name, align, tree)
       results[model_name] = {
           'lnL': result.lnL,
           'n_params': result.n_params,
           'AIC': -2 * result.lnL + 2 * result.n_params,
           'BIC': -2 * result.lnL + result.n_params * np.log(n_sites)
       }

   # Find best model by AIC
   best_model = min(results.items(), key=lambda x: x[1]['AIC'])
   print(f"Best model: {best_model[0]}")

Parameter Scanning
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Scan omega values to understand likelihood surface
   omegas = np.linspace(0.1, 3.0, 30)
   lnLs = []

   for omega in omegas:
       # Would need custom optimizer to fix omega
       # This is an example of advanced usage
       pass

Low-Level Likelihood Calculation
---------------------------------

For expert users who need direct access to likelihood:

.. code-block:: python

   from crabml.core.likelihood import LikelihoodCalculator
   from crabml.models.codon import CodonModel

   # Create codon model
   codon_model = CodonModel(
       kappa=2.5,
       omega=0.8,
       codon_freqs=freqs
   )

   # Create likelihood calculator
   calc = LikelihoodCalculator(alignment, tree, codon_model)

   # Compute likelihood
   lnL = calc.compute_log_likelihood()

   print(f"Log-likelihood: {lnL:.6f}")

This is rarely needed but available for research and method development.

Tips and Best Practices
------------------------

1. **Always use M0 initialization** for complex models (enabled by default)

2. **Use F3x4 codon frequencies** for realistic models (enabled by default)

3. **Check convergence** before trusting results

4. **Run multiple tests** (M1a vs M2a AND M7 vs M8)

5. **Validate against PAML** for published analyses

6. **Monitor memory usage** for large datasets (100+ sequences)

7. **Use branch-site models** for lineage-specific questions

8. **Export results** to JSON for downstream analysis

9. **Keep alignments clean** - remove highly gapped regions

10. **Check biological plausibility** of omega estimates
