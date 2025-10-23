Installation
============

Requirements
------------

* Python 3.11 or higher
* Rust toolchain (install from https://rustup.rs)
* OpenBLAS or similar BLAS/LAPACK implementation

Install crabML
--------------

The recommended way to install crabML is using ``uv``:

.. code-block:: bash

   uv sync --all-extras --reinstall-package crabml-rust

This single command:

* Installs all Python dependencies
* Builds the Rust extension
* Ensures the Rust extension is rebuilt even if already installed

Alternative Installation
------------------------

If you prefer to use pip:

.. code-block:: bash

   pip install -e ".[dev]"
   cd rust
   maturin develop --release
   cd ..

Verify Installation
-------------------

To verify that crabML is installed correctly:

.. code-block:: python

   import crabml
   print(crabml.__version__)

   # Run a quick test
   from crabml import optimize_model
   # Should import without errors

Development Installation
------------------------

For development, install with all optional dependencies:

.. code-block:: bash

   uv sync --all-extras --reinstall-package crabml-rust --group dev --group docs

This installs:

* All runtime dependencies
* Testing tools (pytest, pytest-xdist)
* Documentation tools (Sphinx, themes)
* Development tools (ruff, ipython)

Troubleshooting
---------------

**Rust compiler not found**

If you see an error about the Rust compiler not being found:

1. Install Rust from https://rustup.rs
2. Restart your terminal
3. Verify with ``rustc --version``

**OpenBLAS not found**

On Ubuntu/Debian:

.. code-block:: bash

   sudo apt-get install libopenblas-dev

On macOS:

.. code-block:: bash

   brew install openblas

On Fedora/RHEL:

.. code-block:: bash

   sudo dnf install openblas-devel
