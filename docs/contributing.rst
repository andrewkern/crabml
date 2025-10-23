Contributing
============

We welcome contributions to crabML! This document provides guidelines for contributing.

Development Setup
-----------------

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/yourusername/crabml.git
   cd crabml

2. Install with development dependencies:

.. code-block:: bash

   uv sync --all-extras --reinstall-package crabml-rust --group dev --group docs

3. Verify installation:

.. code-block:: bash

   uv run pytest

Code Style
----------

Python Code
~~~~~~~~~~~

We use ``ruff`` for linting and formatting:

.. code-block:: bash

   # Format code
   uv run ruff format src/ tests/

   # Check for issues
   uv run ruff check src/ tests/

**Guidelines:**

* Follow PEP 8 style guide
* Use Google-style docstrings
* Maximum line length: 100 characters
* Type hints for function signatures

Rust Code
~~~~~~~~~

Rust code should follow standard Rust conventions:

.. code-block:: bash

   cd rust/
   cargo fmt
   cargo clippy

Running Tests
-------------

Run all tests:

.. code-block:: bash

   uv run pytest -n 8

Run specific test file:

.. code-block:: bash

   uv run pytest tests/test_api.py -v -n 8

Run tests with coverage:

.. code-block:: bash

   uv run pytest --cov=crabml --cov-report=html

PAML validation tests (slower):

.. code-block:: bash

   uv run pytest tests/test_paml_examples/ -v -n 4

Building Documentation
----------------------

Build Sphinx documentation:

.. code-block:: bash

   cd docs/
   make html

View documentation:

.. code-block:: bash

   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Contributing Guidelines
-----------------------

1. **Create an issue** first to discuss major changes

2. **Fork the repository** and create a feature branch:

.. code-block:: bash

   git checkout -b feature/your-feature-name

3. **Make your changes** following code style guidelines

4. **Add tests** for new functionality

5. **Update documentation** if needed

6. **Run tests** to ensure nothing breaks:

.. code-block:: bash

   uv run pytest -n 8

7. **Commit your changes** with clear messages:

.. code-block:: bash

   git add .
   git commit -m "Add feature: brief description

   More detailed explanation of changes, why they were needed,
   and any relevant context."

8. **Push to your fork** and create a pull request

Pull Request Process
--------------------

1. **Ensure all tests pass**
2. **Update CHANGELOG.md** with your changes
3. **Update documentation** if you changed the API
4. **Fill out the PR template** with:

   * Description of changes
   * Issue number (if applicable)
   * Testing performed
   * Screenshots (if UI changes)

5. **Wait for review** - maintainers will review and provide feedback

Code Review Checklist
---------------------

Reviewers will check for:

☐ All tests pass

☐ Code follows style guidelines

☐ Documentation updated

☐ CHANGELOG.md updated

☐ No unnecessary dependencies added

☐ PAML validation tests still pass (for model changes)

☐ Performance hasn't regressed

Reporting Bugs
--------------

Use GitHub Issues to report bugs. Include:

1. **Environment information:**

   * Python version
   * Rust version
   * Operating system
   * crabML version

2. **Steps to reproduce** the bug

3. **Expected behavior**

4. **Actual behavior**

5. **Minimal example** that demonstrates the issue

6. **Error messages** and stack traces

Example bug report:

.. code-block:: text

   **Bug**: M2a optimizer fails on large alignments

   **Environment:**
   - Python 3.11.5
   - Rust 1.73.0
   - Ubuntu 22.04
   - crabML 0.2.0

   **Steps to reproduce:**
   1. Load alignment with 200 sequences
   2. Run `optimize_model("M2a", align, tree)`

   **Expected:** Should complete successfully

   **Actual:** Crashes with "out of memory" error

   **Error message:**
   ```
   MemoryError: Unable to allocate array with shape (200, 1000, 10)
   ```

Feature Requests
----------------

We welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** in detail
3. **Explain the use case** and why it's needed
4. **Provide examples** of how it would be used

Adding New Models
-----------------

To add a new codon model:

1. **Implement optimizer class** in ``src/crabml/optimize/``

.. code-block:: python

   class MyModelOptimizer(BaseOptimizer):
       def __init__(self, alignment, tree, **kwargs):
           super().__init__(alignment, tree, **kwargs)

       def optimize(self, **kwargs):
           # Implement optimization logic
           pass

2. **Add parser function** to ``src/crabml/api.py``

.. code-block:: python

   def _parse_mymodel_result(result_tuple, ...):
       # Parse optimizer output into ModelResult
       pass

3. **Add to optimize_model()** function

.. code-block:: python

   OPTIMIZER_MAP = {
       ...
       "mymodel": MyModelOptimizer,
   }

   PARSER_MAP = {
       ...
       "mymodel": _parse_mymodel_result,
   }

4. **Add tests** in ``tests/test_api.py``

5. **Add PAML validation** in ``tests/test_paml_examples/``

6. **Update documentation** in ``docs/user_guide/models.rst``

Testing Guidelines
------------------

All code should be tested. We use pytest.

**Test structure:**

.. code-block:: python

   class TestMyFeature:
       def test_basic_functionality(self):
           """Test basic use case."""
           result = my_function()
           assert result == expected

       def test_edge_case(self):
           """Test edge case behavior."""
           # ...

       def test_error_handling(self):
           """Test that errors are raised appropriately."""
           with pytest.raises(ValueError):
               my_function(invalid_input)

**PAML validation tests:**

.. code-block:: python

   def test_mymodel_vs_paml():
       """Test MyModel against PAML reference."""
       result = optimize_model("MyModel", align, tree)
       paml_lnL = -1234.567890  # From PAML output

       assert abs(result.lnL - paml_lnL) < 0.01

Documentation Guidelines
------------------------

**Docstring format** (Google style):

.. code-block:: python

   def my_function(param1: str, param2: int = 5) -> bool:
       """
       Brief one-line description.

       Longer description with more details about what the function
       does, when to use it, and any important notes.

       Args:
           param1: Description of param1
           param2: Description of param2. Defaults to 5.

       Returns:
           Description of return value

       Raises:
           ValueError: When param1 is empty
           RuntimeError: When computation fails

       Examples:
           >>> my_function("test", 10)
           True

           >>> my_function("")
           ValueError: param1 cannot be empty
       """
       pass

Getting Help
------------

* **GitHub Issues**: For bugs and feature requests
* **GitHub Discussions**: For questions and general discussion
* **Email**: adkern@uoregon.edu for other inquiries

License
-------

By contributing, you agree that your contributions will be licensed under
the GPL-3.0-or-later license.
