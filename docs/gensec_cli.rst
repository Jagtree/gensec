.. _api_cli:

=====================
:mod:`gensec.cli`
=====================

Command-line interface entry point.

GenSec provides three subcommands:

``gensec run``
   Full analysis pipeline: domain generation, verification, plots.

   .. code-block:: bash

      uv run gensec run input.yaml [--n-points 200] [--output-dir ./results]

``gensec analyze``
   Lightweight force decomposition without domain generation.

   .. code-block:: bash

      uv run gensec analyze input.yaml [--output-dir ./results] [--eta]

   The ``--eta`` flag enables on-demand :math:`\eta` computation via
   ray–bisection (see :ref:`architecture_solver`).

``gensec plot``
   Regenerate a plot from a previously exported JSON file.

   .. code-block:: bash

      uv run gensec plot data_file.json [--output plot.png] [--dpi 150]

For backward compatibility, passing a YAML file without a subcommand
is equivalent to ``gensec run``.

.. automodule:: gensec.cli
   :members:
   :show-inheritance:
