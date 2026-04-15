.. _installation:

============
Installation
============

Requirements
------------

GenSec requires **Python 3.10** or later.

Core dependencies (installed automatically):

- ``numpy >= 1.24``
- ``scipy >= 1.10``
- ``matplotlib >= 3.6``
- ``pyyaml >= 6.0``
- ``shapely >= 2.0``
- ``triangle`` — constrained Delaunay triangulation for non-rectangular meshes.

Optional dependencies for building the documentation:

- ``sphinx``
- ``furo`` (HTML theme)
- ``sphinx-multiversion``
- ``sphinxcontrib-mermaid``


Install with uv (recommended)
------------------------------

`uv <https://docs.astral.sh/uv/>`_ is the recommended tool for managing
the virtual environment and dependencies:

.. code-block:: bash

   git clone https://github.com/Jagtree/gensec.git
   cd gensec
   uv sync

This creates a ``.venv`` in the project root, installs all dependencies
(including dev dependencies for testing), and makes the ``gensec`` CLI
available inside the virtual environment.


Install with pip
-----------------

.. code-block:: bash

   pip install -e .

For development (editable install with test dependencies):

.. code-block:: bash

   pip install -e ".[dev]"


Verify the installation
------------------------

.. code-block:: bash

   # Check version
   uv run gensec --version

   # Run the test suite
   uv run python -m pytest tests/ -v

All 106 tests should pass.


Building the documentation
---------------------------

.. code-block:: bash

   # Single-version build
   uv run sphinx-build docs docs/_build/html

   # Multi-version build (see :doc:`/ci_and_versioning`)
   uv run sphinx-multiversion docs docs/_build/multiversion

The HTML output is written to ``docs/_build/html/index.html``.
