Sequence Simulation
===================

The simulation module in crabML allows you to generate synthetic codon sequence data under various evolutionary models. This is useful for:

* **Validation**: Verify that parameter estimation methods can recover known parameters
* **Power analysis**: Determine sample sizes needed to detect selection
* **Benchmarking**: Compare performance of different models and methods
* **Education**: Understand how different evolutionary processes generate sequence patterns

Overview
--------

crabML provides simulators for several codon models:

* **M0**: Single dN/dS ratio across all sites
* **M1a**: Nearly neutral model (purifying + neutral)
* **M2a**: Positive selection model (purifying + neutral + positive)
* **M7**: Beta distribution for omega in (0,1)
* **M8**: Beta distribution + positive selection class
* **M8a**: Beta distribution + neutral class (null for M8a vs M8 test)

All simulators use efficient eigendecomposition-based algorithms and support reproducible simulations with random seeds.

Command-Line Interface
----------------------

Basic Usage
~~~~~~~~~~~

The general syntax for simulation commands is:

.. code-block:: bash

   crabml simulate <model> -t <tree> -o <output> -l <length> [options]

Required parameters:

* ``-t, --tree``: Input tree file in Newick format with branch lengths
* ``-o, --output``: Output FASTA file
* ``-l, --length``: Sequence length in codons

Common options:

* ``--kappa``: Transition/transversion ratio (default: 2.0)
* ``-r, --replicates``: Number of replicates to simulate (default: 1)
* ``--seed``: Random seed for reproducibility
* ``-q, --quiet``: Suppress progress messages

M0 Model (Single Omega)
~~~~~~~~~~~~~~~~~~~~~~~~

The M0 model assumes a single dN/dS ratio for all sites.

.. code-block:: bash

   # Simulate 1000 codons with omega=0.3
   crabml simulate m0 -t tree.nwk -o sim.fasta -l 1000 --omega 0.3

   # Use custom kappa and reproducible seed
   crabml simulate m0 -t tree.nwk -o sim.fasta -l 1000 \
       --omega 0.5 --kappa 2.5 --seed 42

   # Simulate 10 replicates
   crabml simulate m0 -t tree.nwk -o sim.fasta -l 500 \
       --omega 0.3 -r 10

**Parameters:**

* ``--omega``: dN/dS ratio (required)

**Outputs:**

* FASTA file with simulated sequences
* ``<output>.params.json``: Parameters used for simulation

M1a Model (Nearly Neutral)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The M1a model has two site classes: purifying selection (omega < 1) and neutral (omega = 1).

.. code-block:: bash

   # 70% purifying (omega=0.1), 30% neutral
   crabml simulate m1a -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.7 --omega0 0.1

   # Multiple replicates
   crabml simulate m1a -t tree.nwk -o sim.fasta -l 500 \
       --p0 0.6 --omega0 0.05 -r 10

**Parameters:**

* ``--p0``: Proportion in purifying class (required)
* ``--omega0``: dN/dS for purifying class, must be < 1 (required)

The neutral class has proportion ``1 - p0`` and omega = 1.

**Outputs:**

* FASTA file with simulated sequences
* ``<output>.params.json``: Parameters used for simulation

M2a Model (Positive Selection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The M2a model has three site classes: purifying, neutral, and positive selection.

.. code-block:: bash

   # 50% purifying, 30% neutral, 20% positive
   crabml simulate m2a -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.5 --p1 0.3 --omega0 0.1 --omega2 2.5

   # Strong positive selection on few sites
   crabml simulate m2a -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.6 --p1 0.3 --omega0 0.05 --omega2 5.0 -r 10

**Parameters:**

* ``--p0``: Proportion in purifying class (required)
* ``--p1``: Proportion in neutral class (required)
* ``--omega0``: dN/dS for purifying class, must be < 1 (required)
* ``--omega2``: dN/dS for positive selection class, must be > 1 (required)

The positive selection class has proportion ``1 - p0 - p1``.

**Outputs:**

* FASTA file with simulated sequences
* ``<output>.params.json``: Parameters used for simulation
* ``<output>.site_classes.txt``: Site class assignments for each codon
* ``<output>.positive_sites.txt``: List of sites under positive selection

M7 Model (Beta Distribution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The M7 model uses a beta distribution to model continuous variation in omega across sites, constrained to (0,1).

.. code-block:: bash

   # Beta(2,5) distribution (mean omega ~0.29)
   crabml simulate m7 -t tree.nwk -o sim.fasta -l 1000 \
       --p 2 --q 5

   # Use more categories for finer discretization
   crabml simulate m7 -t tree.nwk -o sim.fasta -l 1000 \
       --p 1 --q 2 --ncateg 20

**Parameters:**

* ``--p``: Beta shape parameter p (alpha) (required)
* ``--q``: Beta shape parameter q (beta) (required)
* ``--ncateg``: Number of discrete categories (default: 10)

**Outputs:**

* FASTA file with simulated sequences
* ``<output>.params.json``: Parameters including mean omega

M8 Model (Beta + Positive Selection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The M8 model combines a beta distribution with an additional class for positive selection.

.. code-block:: bash

   # 80% beta-distributed, 20% positive selection
   crabml simulate m8 -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.8 --p 2 --q 5 --omega-s 2.5

   # Strong positive selection on small fraction
   crabml simulate m8 -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.95 --p 1 --q 2 --omega-s 5.0 -r 10

**Parameters:**

* ``--p0``: Proportion in beta distribution (required)
* ``--p``: Beta shape parameter p (required)
* ``--q``: Beta shape parameter q (required)
* ``--omega-s``: dN/dS for positive selection class, must be > 1 (required)
* ``--ncateg``: Number of discrete categories for beta (default: 10)

The positive selection class has proportion ``1 - p0``.

**Outputs:**

* FASTA file with simulated sequences
* ``<output>.params.json``: Parameters used for simulation
* ``<output>.site_classes.txt``: Site class assignments
* ``<output>.positive_sites.txt``: Sites under positive selection

M8a Model (Beta + Neutral)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The M8a model is the null model for M8a vs M8 test. It combines a beta distribution with a neutral class (omega=1.0). This is identical to M8 except omega_s is fixed to 1.0 instead of being > 1.

.. code-block:: bash

   # 80% beta-distributed, 20% neutral
   crabml simulate m8a -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.8 --p 2 --q 5

   # Mostly neutral with some purifying selection
   crabml simulate m8a -t tree.nwk -o sim.fasta -l 1000 \
       --p0 0.9 --p 1 --q 2 -r 10

**Parameters:**

* ``--p0``: Proportion in beta distribution (required)
* ``--p``: Beta shape parameter p (required)
* ``--q``: Beta shape parameter q (required)
* ``--ncateg``: Number of discrete categories for beta (default: 10)

The neutral class (omega=1.0) has proportion ``1 - p0``.

**Outputs:**

* FASTA file with simulated sequences
* ``<output>.params.json``: Parameters used for simulation (omega_s=1.0)
* ``<output>.site_classes.txt``: Site class assignments

**Note:** M8a has no positively selected sites since omega_s=1.0 (neutral), not > 1.

Python API
----------

You can also use the simulation module programmatically in Python.

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   from crabml.simulate.codon import M0CodonSimulator
   from crabml.io.trees import Tree
   import numpy as np

   # Load tree
   tree = Tree.from_newick("((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);")

   # Create simulator
   simulator = M0CodonSimulator(
       tree=tree,
       sequence_length=1000,
       kappa=2.0,
       omega=0.3,
       codon_freqs=np.ones(61) / 61,  # uniform frequencies
       seed=42
   )

   # Simulate sequences
   sequences = simulator.simulate()

   # Access sequences for each taxon
   for taxon, seq in sequences.items():
       print(f"{taxon}: {len(seq)} codons")

Positive Selection Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from crabml.simulate.codon import M2aSimulator
   from crabml.simulate.output import SimulationOutput

   # Create M2a simulator
   simulator = M2aSimulator(
       tree=tree,
       sequence_length=1000,
       kappa=2.0,
       p0=0.5,
       p1=0.3,
       omega0=0.1,
       omega2=2.5,
       codon_freqs=np.ones(61) / 61,
       seed=42
   )

   # Simulate
   sequences = simulator.simulate()

   # Get site class information
   site_info = simulator.get_site_classes()
   print(f"Positively selected sites: {site_info['positively_selected_sites']}")

   # Write outputs
   SimulationOutput.write_fasta(sequences, "output.fasta")
   SimulationOutput.write_parameters(simulator.get_parameters(), "params.json")

Beta Distribution Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from crabml.simulate.codon import M7Simulator, M8Simulator, M8aSimulator

   # M7: Beta distribution
   m7_simulator = M7Simulator(
       tree=tree,
       sequence_length=1000,
       kappa=2.0,
       p=2.0,
       q=5.0,
       n_categories=10,
       codon_freqs=np.ones(61) / 61,
       seed=42
   )

   # M8: Beta + positive selection
   m8_simulator = M8Simulator(
       tree=tree,
       sequence_length=1000,
       kappa=2.0,
       p0=0.8,
       p=2.0,
       q=5.0,
       omega_s=2.5,
       n_beta_categories=10,
       codon_freqs=np.ones(61) / 61,
       seed=42
   )

   # M8a: Beta + neutral (null for M8a vs M8 test)
   m8a_simulator = M8aSimulator(
       tree=tree,
       sequence_length=1000,
       kappa=2.0,
       p0=0.8,
       p=2.0,
       q=5.0,
       n_beta_categories=10,
       codon_freqs=np.ones(61) / 61,
       seed=42
   )

   sequences_m7 = m7_simulator.simulate()
   sequences_m8 = m8_simulator.simulate()
   sequences_m8a = m8a_simulator.simulate()

Validation Workflow
-------------------

A common workflow is to simulate data and then verify parameter recovery:

.. code-block:: python

   from crabml.simulate.codon import M0CodonSimulator
   from crabml import optimize_model
   import tempfile

   # 1. Simulate with known parameters
   true_omega = 0.3
   simulator = M0CodonSimulator(
       tree=tree,
       sequence_length=1000,
       kappa=2.0,
       omega=true_omega,
       codon_freqs=np.ones(61) / 61,
       seed=42
   )
   sequences = simulator.simulate()

   # 2. Write to temporary file
   with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
       SimulationOutput.write_fasta(sequences, f.name)
       fasta_path = f.name

   # 3. Fit model to simulated data
   result = optimize_model("M0", fasta_path, tree_string)

   # 4. Check parameter recovery
   print(f"True omega: {true_omega:.4f}")
   print(f"Estimated omega: {result.omega:.4f}")
   print(f"Difference: {abs(result.omega - true_omega):.4f}")

Power Analysis
--------------

Simulate multiple replicates to assess power to detect positive selection:

.. code-block:: bash

   # Simulate 100 replicates under M2a
   for i in {1..100}; do
       crabml simulate m2a -t tree.nwk -o sim_$i.fasta -l 500 \
           --p0 0.5 --p1 0.3 --omega0 0.1 --omega2 2.5 --seed $i
   done

   # Fit M1a vs M2a to each replicate
   for i in {1..100}; do
       crabml site-model -s sim_$i.fasta -t tree.nwk --test m1a_vs_m2a \
           -o results_$i.json
   done

   # Analyze power (fraction of significant results at alpha=0.05)

Or in Python:

.. code-block:: python

   from crabml.simulate.codon import M2aSimulator
   from crabml import positive_selection
   import numpy as np

   # Simulation parameters
   n_replicates = 100
   true_p_positive = 0.2  # 20% sites under positive selection

   significant_count = 0

   for i in range(n_replicates):
       # Simulate under M2a
       simulator = M2aSimulator(
           tree=tree,
           sequence_length=500,
           kappa=2.0,
           p0=0.5,
           p1=0.3,
           omega0=0.1,
           omega2=2.5,
           codon_freqs=np.ones(61) / 61,
           seed=i
       )
       sequences = simulator.simulate()

       # Write to file (or use in-memory)
       # ...

       # Test M1a vs M2a
       results = positive_selection(fasta_path, tree_string, test="m1a_vs_m2a")

       # Check if significant at alpha=0.05
       if results['M1a_vs_M2a'].p_value < 0.05:
           significant_count += 1

   power = significant_count / n_replicates
   print(f"Power to detect positive selection: {power:.2f}")

Tips and Best Practices
------------------------

1. **Tree branch lengths**: Ensure your tree has realistic branch lengths. Very short branches may cause numerical issues.

2. **Sequence length**: Longer sequences provide more statistical power but take longer to simulate and analyze. Start with 500-1000 codons.

3. **Reproducibility**: Always use ``--seed`` when you need reproducible results.

4. **Parameter constraints**:

   * M1a: omega0 must be < 1
   * M2a: omega0 < 1, omega2 > 1, p0 + p1 < 1
   * M8: omega_s > 1, p0 < 1

5. **Output organization**: Use meaningful output names and organize replicates in separate directories:

   .. code-block:: bash

      mkdir -p simulations/m2a_replicates
      crabml simulate m2a -t tree.nwk \
          -o simulations/m2a_replicates/rep.fasta \
          -l 1000 --p0 0.5 --p1 0.3 --omega0 0.1 --omega2 2.5 -r 100

6. **Validation**: Always check that simulated data looks reasonable:

   * Site class proportions match expected values
   * Omega values are in expected ranges
   * Parameter recovery tests succeed

Algorithm Details
-----------------

The simulation algorithm uses:

1. **Eigendecomposition**: The Q matrix is eigendecomposed once during initialization
2. **Matrix exponential**: Transition probabilities P(t) = exp(Qt) computed efficiently as P(t) = U @ diag(exp(λt)) @ V
3. **Root-to-tips evolution**: Sequences evolved recursively from root to tips
4. **Site classes**: For site-class models, each site is assigned to a class, then evolved with class-specific P matrices

This approach is O(n²) per branch, making it efficient for large trees.

Performance
-----------

Simulation speed depends on:

* Sequence length (linear)
* Number of taxa (linear in tree traversal)
* Number of site classes (linear)
* Tree structure (affects cache efficiency)

Typical performance:

* M0, 1000 codons, 10 taxa: ~0.1 seconds
* M2a, 1000 codons, 10 taxa: ~0.2 seconds
* M8, 1000 codons, 10 taxa: ~0.3 seconds

API Reference
-------------

For detailed API documentation, see:

* :class:`crabml.simulate.base.SequenceSimulator`
* :class:`crabml.simulate.codon.M0CodonSimulator`
* :class:`crabml.simulate.codon.M1aSimulator`
* :class:`crabml.simulate.codon.M2aSimulator`
* :class:`crabml.simulate.codon.M7Simulator`
* :class:`crabml.simulate.codon.M8Simulator`
* :class:`crabml.simulate.output.SimulationOutput`
