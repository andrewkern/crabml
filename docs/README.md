# crabML Documentation

This directory contains the complete Sphinx documentation for crabML.

## Building Documentation Locally

To build the HTML documentation:

```bash
# Install documentation dependencies
uv sync --group docs

# Build HTML documentation
cd docs/
make html

# View the documentation
open _build/html/index.html  # macOS
xdg-open _build/html/index.html  # Linux
```

## Documentation Structure

```
docs/
├── index.rst                 # Main documentation index
├── conf.py                   # Sphinx configuration
├── Makefile                  # Build automation
├── requirements.txt          # Docs build requirements
│
├── user_guide/               # User-facing guides
│   ├── installation.rst      # Installation instructions
│   ├── quickstart.rst        # Quick start guide
│   ├── models.rst            # Complete model reference
│   ├── hypothesis_testing.rst# Hypothesis testing guide
│   └── advanced.rst          # Advanced features
│
├── api/                      # API reference
│   ├── high_level.rst        # High-level API (optimize_model, etc.)
│   ├── analysis.rst          # Hypothesis testing functions
│   ├── io.rst                # I/O classes (Alignment, Tree)
│   ├── optimize.rst          # Optimizer classes
│   ├── models.rst            # Model classes
│   └── core.rst              # Core likelihood calculation
│
├── changelog.rst             # Version history
├── contributing.rst          # Contribution guidelines
└── license.rst               # License information
```

## Read the Docs Deployment

This documentation is configured to build automatically on [Read the Docs](https://readthedocs.org/).

Configuration files:
- `.readthedocs.yaml` - RTD build configuration
- `docs/requirements.txt` - Python dependencies for doc build

## Writing Documentation

### reStructuredText (.rst) Format

We use reStructuredText (reST) format for documentation. Key syntax:

**Headings:**
```rst
Chapter Title
=============

Section Title
-------------

Subsection Title
~~~~~~~~~~~~~~~~
```

**Code blocks:**
```rst
.. code-block:: python

   from crabml import optimize_model
   result = optimize_model("M0", "align.fasta", "tree.nwk")
```

**Links:**
```rst
:doc:`models` - Link to another doc
:mod:`crabml.api` - Link to module
:func:`optimize_model` - Link to function
```

### Adding New Documentation

1. Create new `.rst` file in appropriate directory
2. Add to `toctree` in `index.rst` or parent file
3. Build and check: `make html`
4. Commit and push

### API Documentation

API documentation is auto-generated from docstrings using Sphinx autodoc.

**Docstring format (Google style):**
```python
def my_function(param1: str, param2: int = 5) -> bool:
    """
    Brief one-line description.

    Longer description with more details.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 5.

    Returns:
        Description of return value

    Examples:
        >>> my_function("test", 10)
        True
    """
```

## Useful Make Commands

```bash
make html       # Build HTML documentation
make clean      # Clean build files
make linkcheck  # Check for broken links
```

## Troubleshooting

**Import errors:**
- Ensure crabML is installed: `uv sync --all-extras`
- Check Python path in `conf.py`

**Missing autodoc:**
- Check module imports in API `.rst` files
- Ensure all classes/functions have docstrings

**Build warnings:**
- Review warnings in build output
- Fix broken cross-references
- Ensure all referenced sections exist
