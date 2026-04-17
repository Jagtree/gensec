.. _api_cli:

=====================
:mod:`gensec.cli`
=====================

Command-line interface entry point.

Usage::

    uv run gensec input.yaml [--n-points 400] [--output-dir ./results]

Or equivalently::

    uv run python -m gensec input.yaml [--n-points 400] [--output-dir ./results]

The CLI reads a YAML input file (see :doc:`/user_guide/yaml_reference`),
runs the full analysis pipeline (section construction, N-M diagram
generation, demand verification, per-fiber post-processing), and writes
all outputs (plots, CSV, JSON) to the specified directory.

.. automodule:: gensec.cli
   :members:
   :show-inheritance:
