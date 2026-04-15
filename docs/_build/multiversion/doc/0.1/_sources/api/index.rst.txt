.. _api_reference:

=============
API Reference
=============

This section documents the public Python API of GenSec.  The
documentation is generated automatically from docstrings using
:mod:`sphinx.ext.autodoc`.

The package is organized into four subpackages and two top-level
modules:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Subpackage / Module
     - Responsibility
   * - :doc:`gensec.materials <gensec.materials>`
     - Constitutive laws: abstract base, concrete, steel, tabulated,
       EC2/EN 10025 property tables, bridge factories.
   * - :doc:`gensec.geometry <gensec.geometry>`
     - Section definition: point fibers, parametric primitives,
       generic polygon meshing, backward-compatible rectangular wrapper.
   * - :doc:`gensec.solver <gensec.solver>`
     - Fiber integration, Newton–Raphson equilibrium solver,
       N-M / N-Mx-My diagram generation, demand verification.
   * - :doc:`gensec.output <gensec.output>`
     - Plotting, CSV/JSON export, terminal reporting.
   * - :doc:`gensec.io_yaml <gensec.io_yaml>`
     - YAML input loader.
   * - :doc:`gensec.cli <gensec.cli>`
     - Command-line interface entry point.
