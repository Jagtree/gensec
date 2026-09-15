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
----------------------------------------------

Dedicated plotting and textual reporting for the ideal_gross geometric
properties (see :ref:`ideal_gross_properties`).  Produces the figure with
centroid, principal axes :math:`\xi, \eta`, central inertia
ellipse, and kern, plus the matching text report.

.. automodule:: gensec.output.geometry_plot
   :members:
   :show-inheritance:



.. _api_timeline_dump:

Machine-readable timeline dump
-------------------------------

The versioned JSON dump of a resolved construction timeline, and **the
single source of truth** for everything a timeline run produces.

The arrangement is deliberate and structural rather than conventional.
:func:`~gensec.output.timeline_dump.build_timeline_dump` returns a
``dict``; the CSV tables and ``timeline.png`` are then *pure functions
of that dict*, living in a module that imports nothing from
:mod:`gensec.solver` and therefore cannot reach a resolution even by
accident.  A figure and a table built in parallel from a resolution
agree only until somebody edits one of them, and the disagreement is
silent: two plausible renderings and no third thing to arbitrate.
Derived from one dict they cannot disagree, because there is only one
set of numbers.

Three properties hold, each measured rather than asserted:

**Versioned.**  Every dump declares ``schema`` and ``schema_version``.
A document with no version is *refused*, never read as the current
one -- assuming is how a format loses the ability to evolve.  The same
number gates the input document (D7): a model declaring
``zone_rheology:``, ``tolerances:`` or a provider ``declared:`` block
must state the schema it was written against, because all three carry
hand-written numbers that travel into the dump as provenance.

**Provenance-complete.**  Each provider's declared domain with its
citation; each :class:`~gensec.materials.base.DeclaredRule` with its
source; each zone's *declared* and *derived* age at every event; the
resolved tolerance set with the origin of every entry; and convergence
as a relative residual beside the tolerance it was judged against.

**Absence-explicit.**  A value that is not available is ``null`` *and*
a row in ``absences[]`` naming the cause.  There is no
``getattr(obj, "k", default)`` anywhere on this path: inside a
versioned contract a silent default is permanent, because a later
reader cannot tell it from a measurement.

The schema is specified in ``handoff/13_0-GENSEC_DUMP_SCHEMA.md``,
which governs; this module implements it.

.. automodule:: gensec.output.timeline_dump
   :members:
   :show-inheritance:


Projections of the dump
------------------------

CSV tables and the ``timeline.png`` figure.  Every function here takes
the dump dict and nothing else -- one positional parameter, checked by
a test with :mod:`inspect`, alongside an AST check that this module
holds no import of the solver.  A clean signature with a hidden import
would be parallelism with an alibi.

This module also owns the **casting palette**: the colours that encode
*when* a bulk zone was cast, used by both ``timeline.png`` and
:func:`~gensec.output.plots.plot_section`'s ``cast_steps`` argument, so
a reader moving between the two figures is reading one code.  The
palette is qualitative and there is no colorbar: a zone cast at
:math:`T_0` and one cast at :math:`T_1` are two events with nothing
between them, and a continuous ramp would invite an interpolation that
means nothing.

.. automodule:: gensec.output.timeline_projections
   :members:
   :show-inheritance:


CSV and JSON export
--------------------

.. automodule:: gensec.output.export
   :members:
   :show-inheritance:


Summary report
-------------------

.. automodule:: gensec.output.summary
   :members:
   :show-inheritance:

Terminal reporting
-------------------

.. automodule:: gensec.output.report
   :members:
   :show-inheritance:
