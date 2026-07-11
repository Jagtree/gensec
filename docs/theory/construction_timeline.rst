.. _construction_timeline:

======================================
Construction timeline (staged casting)
======================================

GenSec models staged construction — precast-plus-topping, timber–concrete
composites, steel-deck (*bac acier*) composite slabs, staged prestress —
as a single ordered *construction history*. A zone cast onto an already
loaded substrate is born at that substrate's deformation state and
carries only the loads applied *after* its birth. This page states the
architecture, the linear equivalence that makes it exact, the datum
convention, and the pre-/post-tensioning rule.

.. contents::
   :local:
   :depth: 1


Three layers
============

The machinery is layered so that each level is independently verifiable:

.. code-block:: text

   Layer 3  combinations        anchor at timeline points; factors applied
            (verification)      at compile time to symbolic history loads
                   |  compiler — emits stage lists
   Layer 2  timeline            single construction history: cast / stress /
            (resolution walk)   grout / interval / load events; produces
                   |            frozen datums + reconciled tendon strains
   Layer 1  engine ops          section_ops incl. activate_bulk(zone, plane);
                                SectionState.bulk_active + per-zone planes;
                                re-sliced views; capacity hash; invariants

Layer 1 is complete and independently exercisable through the API,
without Layers 2–3. Layer 2 compiles exclusively to Layer-1 ops. Layer 3
is the untouched demand-verification machinery, fed by the compiler.


Casting datums: per-zone locked-in planes
=========================================

Each bulk-zone activation carries a locked-in strain plane
:math:`(\varepsilon_{0,z},\ \chi_{x,z},\ \chi_{y,z})` — the negated strain
plane of the substrate at the casting instant, evaluated over the zone.
This is the zero-stress reference of the zone's constitutive law: the
fresh zone reads no stress from the substrate's pre-existing strain
field, only from loads applied after it casts.

The legacy scalar ``bulk_eps_init`` is the degenerate plane
:math:`(\varepsilon_0, 0, 0)` applied to every zone — one internal
mechanism, backward compatible. A per-zone *scalar* datum would drop the
curvature term :math:`\chi_0\,(y - y_z)` over the zone depth, which is
first order at SLS; the full plane is therefore mandatory.


Linear equivalence: one-shot with datum ≡ incremental staged analysis
=====================================================================

Let the substrate be in equilibrium under loads :math:`L_0` with strain
plane :math:`P_0`. Zone :math:`z` activates with locked-in datum
:math:`-P_0` (evaluated over the zone). The cumulative-demand walk
re-solves the **total** demand :math:`L_0 + L_1` on the composite
section. Writing the composite solution plane :math:`P = P_0 + \Delta P`,

.. math::

   \int_{\text{sub}} E_s\,P \;+\; \int_{z} E_z\,(P - P_0)
   \;=\; L_0 + L_1
   \;\Longrightarrow\;
   \underbrace{\int_{\text{sub}} E_s\,P_0}_{=\,L_0}
   \;+\; \int_{\text{comp}} E\,\Delta P \;=\; L_0 + L_1 .

Hence :math:`\Delta P` is exactly the incremental solution on the
composite section under :math:`L_1` alone. The substrate stress is
:math:`E_s\,(P_0 + \Delta P)` and the new zone's stress is
:math:`E_z\,\Delta P` — the new zone carries only the loads applied after
its birth, with the full linear strain distribution (the curvature term a
uniform scalar datum would lose).

This identity is the closed-form backbone of the staging validators,
assembled two independent ways: a hand-built 2-DOF incremental model
versus the GenSec one-shot solve with the datum. Beyond the linear range
the identity is not exact — as in any locked-in-strain staged formulation
— but the datum remains the correct zero-stress reference of the zone's
constitutive law.

At SLS this is the two-stage superposition: the substrate carries
:math:`L_0` on its own (pre-composite) section, the composite carries
:math:`\Delta L` on the composite section, and the accumulated fibre
stress is their sum. See :doc:`SLS verification <demand_verification>`
for the staged stress checks (extreme fibres, interface, decompression).


Datums are timeline properties (characteristic walk)
====================================================

Datums are computed **once**, by the timeline resolution walk under
**characteristic** permanent loads (:math:`\gamma = 1`) — the real
structure is built once — and then frozen. Explicit user-supplied datums
override the automatic value (``datum: auto`` versus an explicit plane).

``resolve_stages`` is pure with respect to demand: the datum is an
*input* to it, never computed inside it, mirroring the single-side
invariant of grouting. A zone never enters the domain without an explicit
reconciled plane.

.. note::

   **Normative datum convention.** When a ULS combination factors the
   construction history with :math:`\gamma_G`, the casting datums remain
   those of the *characteristic* walk. The built history is physical: the
   structure was cast once, under the loads that were actually present,
   irrespective of the partial factors a later verification applies. This
   is consistent with practice and is stated here so it is never merely
   implied. Providers for other codes inherit the same rule; it is a
   property of staged construction, not of any single normative document.


Pre- versus post-tensioning falls out of ordering
==================================================

There is no pre/post label on a tendon. The distinction is a pure
consequence of event ordering on the timeline:

* a ``stress`` event on a tendon whose parent zone is **not yet cast** is
  *pre-tensioning* — it has no sectional effect until the zone casts, at
  which point the tendon enters the domain bonded, with :math:`\varepsilon_{pe}`
  net of the explicit elastic-shortening sectional debit (the existing
  transfer machinery);
* a ``stress`` event on a tendon whose parent zone is **already cast** is
  *post-tensioning* — a demand-side ``PrestressAction``,
  later made bonded by a ``grout`` event.

Same event type; the physics is selected by *when* it occurs relative to
the parent zone's casting. A tendon's parent zone is the bulk zone that
geometrically contains it (derived at mesh time), with an optional
``parent:`` override for elements structurally attached to a zone they do
not sit in (legal only for non-embedded elements).
