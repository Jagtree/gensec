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

Solver-related plots (M-:math:`\chi` curves, interaction diagrams,
3D capacity surfaces, stress/strain fibermaps).

.. automodule:: gensec.output.plots
   :members:
   :show-inheritance:


ideal_gross-section plot and geometric report
----------------------------------------

Dedicated plotting and textual reporting for the ideal_gross geometric
properties (see :doc:`ideal_gross_properties`).  Produces the figure with
centroid, principal axes :math:`\xi, \eta`, central inertia
ellipse, and kern, plus the matching text report.

.. automodule:: gensec.output.geometry_plot
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
