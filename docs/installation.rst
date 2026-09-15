.. _installation:

============
Installation
============

Requirements
------------

GenSec requires **Python 3.10** or later (at the moment, 3.14 is not supported because of *triangle* package compatibility).

Core dependencies (installed automatically):

- ``numpy >= 1.24``
- ``scipy >= 1.10``
- ``matplotlib >= 3.6``
- ``pyyaml >= 6.0``
- ``shapely >= 2.0``
- ``triangle`` — constrained Delaunay triangulation for non-rectangular meshes.

Optional performance dependencies (at the moment, their impact is limited):

- ``numba >= 0.58`` — JIT compilation of material stress/tangent kernels.
  Provides a ~2–3× speed-up on large fiber arrays.  Install with
  ``pip install gensec[fast]`` or ``uv sync --all-extras``.

Optional dependencies for building the documentation:

- ``sphinx>=7.0,<9.0``
- ``furo`` (HTML theme),
- ``sphinx-autodoc-typehints``
- ``myst-parser``
- ``sphinx-multiversion``
- ``sphinxcontrib-mermaid>=1``

Optional dependencies for :ref:`developer` mode:

- ``pytest>=7.0``
- ``pytest-cov>=5.0``
- ``coverage[toml]>=7.0``
- ``pytest-timeout>=2.3``  
- ``httpx>=0.27``


Install with uv (recommended)
------------------------------

`uv <https://docs.astral.sh/uv/>`_ is the recommended tool for managing
the virtual environment and dependencies:

.. code-block:: bash

   git clone https://github.com/Jagtree/gensec.git
   cd gensec
   uv sync

To enable optional Numba acceleration:

.. code-block:: bash

   uv sync --all-groups --all-extras

This creates a ``.venv`` in the project root, installs all dependencies
(including dev dependencies for testing), and makes the ``gensec`` CLI
available inside the virtual environment.


Install with pip
-----------------

.. code-block:: bash

   pip install -e .

With Numba acceleration:

.. code-block:: bash

   pip install -e ".[fast]"

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

All 367 tests should pass.


Building the documentation
---------------------------

.. code-block:: bash

   # Single-version build
   uv run sphinx-build docs docs/_build/html

   # Multi-version build (see :doc:`/ci_and_versioning`)
   uv run sphinx-multiversion docs docs/_build/multiversion

The HTML output is written to ``docs/_build/html/index.html``.
