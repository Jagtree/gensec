.. _api_output:

=========================
:mod:`gensec.output`
=========================

Plotting, numerical export, and terminal reporting.

All output functions accept section and solver objects directly and
adapt to the section type (generic polygon or legacy rectangular).
Rebar layers are numbered sequentially (1-based), and this numbering
is consistent across terminal output, plots, and CSV exports.


Plotting
---------

.. automodule:: gensec.output.plots
   :members:
   :show-inheritance:


CSV and JSON export
--------------------

.. automodule:: gensec.output.export
   :members:
   :show-inheritance:


Terminal reporting
-------------------

.. automodule:: gensec.output.report
   :members:
   :show-inheritance:


Package-level exports
----------------------

.. automodule:: gensec.output
   :members:
   :show-inheritance:
