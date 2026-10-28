.. _prestress_scope:

======================================================
Prestressed sections: what is modelled, where it stops
======================================================

GenSec is a **cross-section** library.  Its prestress model is complete
for the quantities that are a function of the section and of its strain
plane.  Quantities that depend on a *length of member* -- the friction
along a cable profile, the strain of an unbonded tendon between its
anchorages, the spacing of cracks, the hyperstatic reactions of a
continuous beam -- are not computed, because a section does not contain
them.

This page states, for every physical effect a prestressed section
involves, which of five statuses applies.  The statuses are not a
ranking; they say who is answerable for the number.

**Computed**
    GenSec derives the quantity from the declared section, materials
    and history.

**Declared input**
    The quantity is a member-level or site-level fact.  GenSec consumes
    it as a number the input supplies and does not check it.

**Refused**
    The case is recognised and stopped with an exception that names
    the reason.  No number is produced.

**Approximated**
    The quantity is computed under a stated simplification.

**Absent**
    There is no input surface for it and nothing refuses it.  A run
    that completes says **nothing** about these effects; they must be
    verified elsewhere.

A sixth section lists the parameters that currently take a value when
the input is silent, and a seventh the known defects.


Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 46 18 36

   * - Effect
     - Status
     - Where
   * - Bonded tendon at ULS (strain compatibility, locked-in prestrain)
     - Computed
     - :class:`~gensec.geometry.fiber.Tendon`, fibre integrator
   * - Immediate elastic loss at release (pre-tensioning)
     - Computed
     - :func:`~gensec.solver.prestress_transfer.solve_pretension_transfer`
   * - Elastic-shortening debit per jack, staged injection, grouting
     - Computed
     - :func:`~gensec.solver.posttension.solve_posttension_sequence`,
       :func:`~gensec.solver.posttension.grout`
   * - Creep, shrinkage, relaxation (composite, multi-zone)
     - Computed / Approximated
     - :mod:`gensec.solver.losses`
   * - SLS stresses, cracking, decompression, per-zone limits
     - Computed
     - :func:`~gensec.solver.sls.verify_sls_staged`
   * - Friction, anchorage set
     - Declared input
     - net into the jacking stress
   * - Unbonded / external tendon, stress increment at ULS and SLS
     - Declared input
     - :meth:`~gensec.solver.section_state.PrestressAction.from_force_uls`,
       ``unbonded_sigma_p``
   * - Effective stress and strain state of an existing member
     - Declared input
     - :class:`~gensec.solver.posttension.PreexistingState`
   * - Hyperstatic (secondary) effects of prestress
     - Declared input
     - inside the demands
   * - Transmission length of pre-tensioned strands
     - Declared input
     - reduced ``eps_pe``
   * - Load between two jacks on an ungrouted tendon
     - Refused
     - :class:`~gensec.solver.posttension.LoadBetweenJacksNotModelled`
   * - Re-stressing a grouted tendon
     - Refused
     - :class:`NotImplementedError`
   * - Time-dependent loss of an unbonded tendon
     - Refused
     - :class:`NotImplementedError`
   * - Non-linear creep
     - Refused
     - :class:`ValueError`
   * - Shear, torsion, fatigue, fire, second order, anchorage zones,
       crack width, deflection
     - Absent
     - --


Computed
========

**Bonded tendons at the ultimate limit state.**  A bonded tendon is an
element of the section.  Its stress follows from sectional strain
compatibility on top of its locked-in prestrain,
:math:`\varepsilon_p = \varepsilon_{\mathrm{sec}} +
\varepsilon_{\mathrm{init}}`, through the prestressing-steel law
(horizontal or inclined top branch, EN 1992-1-1 §3.3.6).  The resistance
domain of a prestressed section is therefore built with the prestrain
in it, and so is every strain state the solver returns.

**Pre-tensioning transfer.**
:func:`~gensec.solver.prestress_transfer.solve_pretension_transfer`
releases the abutments by solving the self-equilibrium of the section.
The elastic loss is not a formula applied afterwards: it emerges from
that equilibrium, including the curvature an eccentric strand imposes.

**Post-tensioning sequences.**
:func:`~gensec.solver.posttension.solve_posttension_sequence` charges
each jack's elastic shortening to the tendons already anchored, against
the section *as it is* at that jack: which tendons are grouted is a
declared schedule (``bonded_before``), not an inference.  Grouting
(:func:`~gensec.solver.posttension.grout`) turns a demand-side tendon
into a section element with the locked-in strain it has at that moment.
Whether a tendon is pre- or post-tensioned is read from the order of the
``cast`` and ``stress`` events, never from a label (see
:doc:`construction_timeline`).

**Time-dependent losses.**  :mod:`gensec.solver.losses` solves a
sectional age-adjusted effective-modulus (AAEM) system on every
``interval`` event: creep and shrinkage per concrete zone, each zone
with its own age, curing age and provider, and the intrinsic relaxation
of every bonded tendon (EN 1992-1-1 §3.3.2, expressions (3.28)-(3.30),
evaluated by
:meth:`~gensec.materials.rheology.EC2RheologicalModel.relaxation`).
On a composite section the restraint one zone exerts on another is part
of the solve, not a correction.  The closed form EN 1992-1-1 (5.46) is
kept only as an independent cross-check
(:func:`~gensec.solver.losses.ec2_5106_closed_form`), because on a
composite section it is undefined.

**Serviceability stresses.**
:func:`~gensec.solver.sls.verify_sls_staged` accumulates stresses stage
by stage, each increment resisted by the section as it is when the
increment is applied.  The concrete enters through its service law
(for EN 1992-1-1, expression (3.14) in compression, released in tension
at :math:`f_{ct,\mathrm{eff}}`), so a cracked equilibrium is solved
directly.  The checks are concrete compression *per zone* against that
zone's own strength, reinforcement and tendon tension, and
decompression at a declared cover from each bonded tendon.  A YAML
combination reaches the engine through its ``sls`` block.


Declared input
==============

These are facts about a member, a site or a structure.  GenSec takes
them as numbers and is not able to check them.

**Friction and anchorage set.**  The jacking stress (or the jacking
strain in ``eps_pe``) must already be net of friction along the cable
profile and of anchorage draw-in.  Both depend on the cable geometry
over the member and on the draw-in length; neither has a home in a
section.

**Unbonded and external tendons.**  Between two contact points the
strain of an unbonded tendon is the *average* over its free length,
which depends on how the whole member deforms.  Such a tendon is
therefore not an element of the section but an action on it.

- At ULS its force is
  :math:`P_{\mathrm{ULS}} = P_{\mathrm{eff}} + \Delta\sigma_p A_p`, built
  by :meth:`~gensec.solver.section_state.PrestressAction.from_force_uls`.
  :math:`\Delta\sigma_p` is an input.
  :func:`~gensec.materials.prestress_properties.delta_sigma_p_uls`
  returns the EN 1992-1-1 §5.10.8(3) recommended 100 MPa, which that
  clause allows as a simplification; a member-level value can be passed
  instead.
- At SLS its stress is supplied per stage as ``unbonded_sigma_p``,
  checked against the tendon limit and tagged
  ``provenance='member_level'``.  It is never computed from the plane.

**An existing member (retrofit).**  The effective stress of each
existing tendon and the strain state of the existing concrete are
declared through :class:`~gensec.solver.posttension.PreexistingState`
or a ``retrofit`` event.  GenSec will not reconstruct them: that would
need decades of loading and humidity history nobody can validate.

**Hyperstatic (secondary) effects of prestress.**  In a statically
indeterminate structure, prestress produces support reactions and
therefore secondary moments and forces.  They come from a structural
analysis, and they must be **included in the demands** :math:`N, M_x,
M_y` handed to GenSec.  Nothing in the input distinguishes them, so
nothing can warn when they are forgotten.

**Transmission length of pre-tensioned strands.**  Over the
transmission length a strand is not yet fully anchored and carries less
than the full prestress.  GenSec applies the prestrain it is given; a
section inside the transmission length must be given the reduced
``eps_pe``.

**Non-isothermal curing.**  The rheological provider assumes a history
at 20 °C.  A temperature-adjusted (equivalent) age per
EN 1992-1-1 Annex B can be declared in its place.

**Confined concrete.**  No confined-concrete law is built in.  A
confined :math:`\sigma\text{-}\varepsilon` curve can be declared as a
tabulated material (``type: tabulated``).  The ``confinement`` argument
of :meth:`~gensec.solver.integrator.FiberSolver.sectional_state` selects
a *capacity factor* for codes that distinguish spirally confined
members; it does not change any constitutive law.

**Concrete strength at transfer.**  The SLS limits at transfer are
written against :math:`f_{ck}(t)` at the age of release, which no
28-day material knows.  It is declared in ``custom_zone_strengths``;
without it a ``kind: transfer`` limit set is refused.


Refused
=======

Each case below raises.  The exception says why and what to do instead.

**A load between two jacks while a tendon is stressed and not yet
grouted.**  The ungrouted tendon has no strain compatibility with the
section; its force change follows a member integral, and the local
value a section can evaluate is unconservative for :math:`\sigma_p` at
the critical section.  Raises
:class:`~gensec.solver.posttension.LoadBetweenJacksNotModelled`.

**Re-stressing a tendon after grouting it.**  The jack no longer pulls
against the anchorages alone.  Raises :class:`NotImplementedError`.
Stressing *another* tendon against an already grouted one is the
staged-injection case and is modelled.

**A time-dependent interval with an unbonded tendon active.**  Its loss
depends on the member-average strain.  Raises
:class:`NotImplementedError`; grout first, or compute the loss outside
GenSec.

**Concrete beyond its linear-creep limit during an interval.**  Above
the provider's linearity limit (EN 1992-1-1 §3.1.4(4):
:math:`0.45 f_{ck}(t_0)`) the creep compliance is stress-dependent and
the AAEM no longer applies.  Raises :class:`ValueError`.

**Existing tendons declared by index only.**  ``already_bonded=[0]``
names which tendons are grouted, not the state they are in.  Raises
:class:`~gensec.solver.posttension.PreexistingStateUndeclared`.

**Bulk deactivation (demolition)** raises :class:`NotImplementedError`.
**Two bodies with independent histories later made solidal** raise as
well: the case needs one strain plane per body before the connection,
which the single-plane walk does not have.

**A step-by-step integration whose strain increment per step is below
the domain-cache quantum.**  It would silently collapse onto one step.
Raises :class:`ValueError`.


Approximated
============

**Relaxation under a decaying strain.**  A tendon that shortens with the
concrete relaxes less than one held at constant length.  The container
multiplies the intrinsic relaxation by ``relaxation_reduction``, the
EN 1992-1-1 lump factor 0.8 unless declared otherwise.  The coupled
solution, with relaxation evaluated on the decaying stress path, is not
implemented.

**Ageing coefficient.**  :math:`\chi = 0.8` (``chi: lump``) unless
``chi: from_J`` is declared, in which case it is computed from the
provider's compliance.

**Creep on an uncracked section.**  Every active concrete fibre enters
the AAEM stiffness with :math:`\bar{E}_z`, and the creep eigenstrain is
formed from the linear stress :math:`E_{c,z}(t_0)\,\varepsilon_{\mathrm{mech}}`,
in tension as in compression.  A section cracked under the sustained
demand is not refused; its time-dependent losses are those of the
uncracked section.  (Read from the code, :mod:`gensec.solver.losses`
steps 6-7; not yet measured against a cracked reference.)

**Frozen demand over an interval.**  The external action is constant
within each interval (the AAEM premise).  A variable sustained load is
represented by splitting the history into intervals.

**Isothermal maturity** at 20 °C (see *Declared input*).

**Relaxation horizon.**  The relaxation law is not extrapolated beyond
500 000 h (EN 1992-1-1 §3.3.2(8)).

**Relaxation class per zone, not per tendon.**  Through the YAML route
and :func:`~gensec.materials.ec2_bridge.prestress_from_ec2` a tendon
takes the relaxation class and :math:`\rho_{1000}` of its zone's
``losses_models`` entry.  Two tendons of different class in the same
zone cannot be distinguished there; the API can, through the
``relaxation_class`` and ``rho_1000`` attributes of
:class:`~gensec.materials.prestress_properties.fbpren`.


Absent
======

There is no input surface for any of the following, so no run refuses
them and no run checks them.

- **Shear and torsion**, and their interaction with :math:`N` and
  :math:`M`: no shear resistance of a prestressed web, no principal
  tensile stress check, no torsion.
- **Second-order effects**: no slenderness, no :math:`P\text{-}\Delta`.
- **Fatigue** of tendons, reinforcement or concrete.
- **Fire**, and temperature fields as an action.
- **Anchorage zones**: bursting and spalling forces, local bearing
  under anchor plates.
- **Crack width and tension stiffening**: the service law returns the
  state *at* a crack; the member average between cracks depends on the
  crack spacing and is a member quantity (see
  :mod:`gensec.materials.service_concrete`).
- **Deflection and camber**: they are integrals of curvature over the
  member.


Values taken when the input is silent
=====================================

.. warning::

   The project rule is that no normative parameter has a default.  The
   following are currently **not** compliant with it: if the input does
   not declare them, a value is used and nothing is reported.
   Declare them explicitly.

   Measured on a ``losses_models`` entry declaring only
   ``provider: ec2`` and ``fck``, which is accepted:

   ==========================  =========================================
   Parameter                   Value used
   ==========================  =========================================
   ``cement_class``            ``N``
   ``RH``                      70 %
   ``relaxation_class``        2 (low relaxation), :math:`\rho_{1000}` = 2.5 %
   ``chi``                     ``lump`` (0.8)
   ``relaxation_reduction``    0.8
   ==========================  =========================================

   On the Python API, the same values are signature defaults of
   :class:`~gensec.materials.rheology.EC2RheologicalModel` and
   :class:`~gensec.solver.losses.LossModel`, and
   :meth:`~gensec.materials.codes.EC2Code.rheological_model` fills
   ``cement_class='N'`` when it is not passed;
   :func:`~gensec.materials.prestress_properties.delta_sigma_p_uls`
   defaults to ``NA='EC2'``, and
   :class:`~gensec.materials.prestress_properties.fbpren` to
   ``NA='EC2'``, ``ls='F'``, ``Ep=195000`` and ``eps_ud_factor=0.9``.
   The YAML ``prestressing_steel_ec2`` material does not reach those
   defaults: it requires ``ls``, ``NA``, ``eps_ud_factor`` and
   ``diagram``, and either a designation (``class``) or ``Ep`` with the
   characteristic values.


Known defects
=============

.. warning::

   **Retrofit clocks are not consumed by the time-dependent walk.**  The
   ages a ``retrofit`` event declares (concrete since casting, each
   tendon since jacking) are parsed and carried, but an ``interval``
   with ``losses`` after the retrofit does not read them: the
   relaxation of a pre-existing tendon restarts at zero, and the
   creep age of the base zone is taken from ``zone_rheology`` without
   being reconciled with the declared one.  Measured effect on the
   Phase-34 bench: 43 times the correct relaxation over the first
   year.  Details in :doc:`construction_timeline`.
