Command-Line Interface
======================

crabML provides a streamlined command-line interface for common analyses. The ``crabml`` command is automatically installed when you install the package.

Overview
--------

The CLI provides four main commands:

* ``crabml site-model``: Run site-class model tests (M1a vs M2a, M7 vs M8)
* ``crabml branch-model``: Run branch model tests (multi-ratio, free-ratio)
* ``crabml branch-site``: Run branch-site model tests
* ``crabml fit``: Fit single codon substitution models

All commands support multiple output formats (text, JSON, TSV) and can write results to files.

Getting Help
------------

Use ``--help`` to see available options for any command:

.. code-block:: bash

   crabml --help
   crabml site-model --help
   crabml branch-model --help
   crabml branch-site --help
   crabml fit --help

crabml site-model
-----------------

Run standard likelihood ratio tests for detecting positive selection.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Run M7 vs M8 test
   crabml site-model -s lysozyme.fasta -t lysozyme.nwk --test m7m8

   # Run M1a vs M2a test
   crabml site-model -s lysozyme.fasta -t lysozyme.nwk --test m1m2

   # Run both tests
   crabml site-model -s lysozyme.fasta -t lysozyme.nwk --test both

Output Formats
^^^^^^^^^^^^^^

.. code-block:: bash

   # Human-readable text (default)
   crabml site-model -s alignment.fasta -t tree.nwk --test both

   # JSON output (for parsing/pipelines)
   crabml site-model -s alignment.fasta -t tree.nwk --test both --format json -o results.json

   # TSV output (for Excel/R)
   crabml site-model -s alignment.fasta -t tree.nwk --test both --format tsv -o results.tsv

Options
^^^^^^^

* ``-s, --alignment PATH``: Path to alignment file (FASTA or PHYLIP) [required]
* ``-t, --tree PATH``: Path to tree file (Newick format) [required]
* ``--test TYPE``: Which test to run: ``m1m2``, ``m7m8``, ``both``, or ``all`` [default: both]
* ``--format FORMAT``: Output format: ``text``, ``json``, or ``tsv`` [default: text]
* ``--output, -o PATH``: Write output to file instead of stdout
* ``--maxiter INT``: Maximum optimization iterations [default: 500]
* ``--alpha FLOAT``: Significance threshold for tests [default: 0.05]
* ``--no-m0-init``: Skip M0 initialization (not recommended)
* ``--quiet``: Suppress progress output
* ``--verbose``: Show detailed optimization progress

Example Output (Text Format)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Test 2: M7 (Beta) vs M8 (Beta + positive selection)
   --------------------------------------------------------------------------------
   Null (M7):           lnL = -902.510    parameters = {...}
   Alternative (M8):    lnL = -899.999    parameters = {...}

   Likelihood Ratio Test:
     2ΔlnL = 5.02    df = 2    p-value = 0.0812

   Result: No significant evidence for positive selection (p > 0.05)

Example Output (JSON Format)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

   {
     "M7_vs_M8": {
       "test_name": "M7 vs M8",
       "lnL_null": -902.510,
       "lnL_alt": -899.999,
       "LRT": 5.022,
       "pvalue": 0.0812,
       "significant": false
     }
   }

crabml fit
----------

Fit a specific codon substitution model to your data.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Fit M0 model
   crabml fit -m M0 -s alignment.fasta -t tree.nwk

   # Fit M8 with custom settings
   crabml fit -m M8 -s alignment.fasta -t tree.nwk --maxiter 1000 --verbose

   # Output as JSON
   crabml fit -m M2a -s alignment.fasta -t tree.nwk --format json -o m2a_result.json

Supported Models
^^^^^^^^^^^^^^^^

* **M0**: One-ratio model (single ω for all sites)
* **M1a**: Nearly neutral model (purifying and neutral)
* **M2a**: Positive selection model (purifying, neutral, and positive)
* **M3**: Discrete model (K=3 discrete ω classes)
* **M7**: Beta distribution model (ω constrained to 0-1)
* **M8**: Beta + ω>1 model (positive selection)
* **M8a**: Beta + ω=1 model (null for M8)

Options
^^^^^^^

* ``-m, --model NAME``: Model name [required]
* ``-s, --alignment PATH``: Path to alignment file (FASTA or PHYLIP) [required]
* ``-t, --tree PATH``: Path to tree file (Newick format) [required]
* ``--format FORMAT``: Output format: ``text`` or ``json`` [default: text]
* ``--output, -o PATH``: Write output to file instead of stdout
* ``--maxiter INT``: Maximum optimization iterations [default: 500]
* ``--no-m0-init``: Skip M0 initialization (not recommended for complex models)
* ``--quiet``: Suppress progress output
* ``--verbose``: Show detailed optimization progress

Example Output
^^^^^^^^^^^^^^

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

crabml branch-model
-------------------

Test for lineage-specific selection using branch models.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Multi-ratio test (different omega for labeled branches)
   crabml branch-model -s alignment.fasta -t labeled_tree.nwk --test multi-ratio

   # Free-ratio test (independent omega for each branch)
   crabml branch-model -s alignment.fasta -t tree.nwk --test free-ratio

Supported Tests
^^^^^^^^^^^^^^^

* **multi-ratio**: Different ω for labeled branch groups (recommended)
  - Tests whether different phylogenetic lineages experience different selection pressures
  - Tree must have branch labels (#0, #1, etc.) to specify foreground/background
  - More statistically powerful than free-ratio with fewer parameters

* **free-ratio**: Independent ω for each branch (exploratory)
  - Estimates one ω per branch in the tree
  - Highly parameter-rich (n-1 omega parameters for n species)
  - Prone to overfitting with small datasets
  - Use with caution

Options
^^^^^^^

* ``-s, --alignment PATH``: Path to alignment file (FASTA or PHYLIP) [required]
* ``-t, --tree PATH``: Path to tree file (Newick format, with branch labels for multi-ratio) [required]
* ``--test TYPE``: Which test to run: ``multi-ratio`` or ``free-ratio`` [default: multi-ratio]
* ``--format FORMAT``: Output format: ``text``, ``json``, or ``tsv`` [default: text]
* ``--output, -o PATH``: Write output to file instead of stdout
* ``--maxiter INT``: Maximum optimization iterations [default: 1000]
* ``--alpha FLOAT``: Significance threshold for test [default: 0.05]
* ``--quiet``: Suppress progress output
* ``--verbose``: Show detailed optimization progress

Tree Format with Branch Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-ratio tests, the tree must have branch labels:

.. code-block:: text

   ((human,chimp) #1, (mouse,rat) #0);

* ``#0``: Background branches
* ``#1``: Foreground branches

Example Output
^^^^^^^^^^^^^^

.. code-block:: text

   ================================================================================
   Branch Model Test Results
   ================================================================================

   Test: Multi-ratio vs M0
   --------------------------------------------------------------------------------
   Null (M0):              lnL = -906.017    parameters = {'omega': 0.807}
   Alternative (Multi):    lnL = -903.245    parameters = {'omega0': 0.654, 'omega1': 1.234}

   Likelihood Ratio Test:
     2ΔlnL = 5.54    df = 1    p-value = 0.0186

   Result: LINEAGE-SPECIFIC SELECTION DETECTED (p < 0.05)
     Background ω = 0.654
     Foreground ω = 1.234
     Foreground is 1.9x faster evolving

crabml branch-site
------------------

Test for positive selection on specific lineages using branch-site Model A.

Basic Usage
^^^^^^^^^^^

.. code-block:: bash

   # Tree must have branch labels: #0 (background), #1 (foreground)
   crabml branch-site -s alignment.fasta -t labeled_tree.nwk

   # With custom settings
   crabml branch-site -s alignment.fasta -t labeled_tree.nwk --maxiter 1000 --alpha 0.01

   # Output as JSON
   crabml branch-site -s alignment.fasta -t labeled_tree.nwk --format json -o results.json

Tree Format with Branch Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tree must have branch labels to specify foreground and background branches:

.. code-block:: text

   ((human,chimp) #1, (mouse,rat) #0);

* ``#0``: Background branches (standard selection)
* ``#1``: Foreground branches (test for positive selection)

Options
^^^^^^^

* ``-s, --alignment PATH``: Path to alignment file (FASTA or PHYLIP) [required]
* ``-t, --tree PATH``: Path to tree file with branch labels (Newick format) [required]
* ``--format FORMAT``: Output format: ``text``, ``json``, or ``tsv`` [default: text]
* ``--output, -o PATH``: Write output to file instead of stdout
* ``--maxiter INT``: Maximum optimization iterations [default: 500]
* ``--alpha FLOAT``: Significance threshold for test [default: 0.05]
* ``--quiet``: Suppress progress output
* ``--verbose``: Show detailed optimization progress

Integration with Pipelines
---------------------------

The CLI is designed to work well in pipelines and scripts:

JSON Output for Parsing
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run test and parse with jq
   crabml site-model -s alignment.fasta -t tree.nwk --format json | jq '.M7_vs_M8.pvalue'

   # Save JSON for later analysis
   crabml fit -m M0 -s alignment.fasta -t tree.nwk --format json -o results.json

TSV Output for Spreadsheets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate TSV for multiple genes
   for gene in gene1 gene2 gene3; do
     crabml site-model -s ${gene}.fasta -t ${gene}.nwk --format tsv --quiet
   done > all_results.tsv

Exit Codes
^^^^^^^^^^

* ``0``: Success
* ``1``: Analysis error (e.g., optimization failed, invalid model)
* ``2``: Argument error (e.g., missing file, invalid options)

Batch Processing
^^^^^^^^^^^^^^^^

.. code-block:: bash

   #!/bin/bash
   # Process multiple alignments

   for alignment in *.fasta; do
     gene=$(basename $alignment .fasta)
     echo "Processing $gene..."

     crabml site-model \
       -s $alignment \
       -t ${gene}.nwk \
       --test both \
       --format json \
       -o ${gene}_results.json \
       --quiet

     if [ $? -eq 0 ]; then
       echo "  Success!"
     else
       echo "  Failed!"
     fi
   done

Tips and Best Practices
------------------------

1. **Use JSON for pipelines**: The JSON output format is ideal for parsing and integrating with other tools.

2. **Always specify output files**: Use ``-o`` to write results to files rather than relying on stdout redirection, especially in complex pipelines.

3. **Start with default settings**: The default settings (M0 initialization, 500 iterations) work well for most datasets.

4. **Use quiet mode for batch jobs**: Add ``--quiet`` when processing many files to reduce log output.

5. **Check exit codes**: In scripts, always check the exit code to detect failures.

6. **Increase maxiter for complex models**: Models like M8 on large datasets may need more iterations. Try ``--maxiter 1000`` if optimization doesn't converge.
