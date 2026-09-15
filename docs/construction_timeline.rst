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


.. _staged-injection:

Staged injection: the section a jack is pulled against
======================================================

A post-tensioning sequence is usually described as though the section
resisting each jack were the same section throughout. It is not,
whenever the ducts are injected as the work proceeds rather than all at
once at the end. Once a tendon is grouted it is in strain compatibility
with the concrete around it, so it belongs to the **resisting** section
for every jacking operation that follows. The section a jack is pulled
against therefore grows by one tendon at every injection.

GenSec does not infer that schedule. Two declarations state the role of
**every** tendon at every step, and the driver refuses to guess when
either is missing:

``bonded_before = {tendon: step}``
   the tendon counts as bonded, and therefore as part of the resisting
   section, at and from that (0-based) step of ``order``. Because a
   tendon must be stressed before it can be grouted,
   ``order.index(k) < bonded_before[k]`` is required; the reverse is a
   :class:`ValueError` rather than a modelling choice.

``pending = [...]``
   the tendons **not yet stressed** at any point of this sequence.
   They are neither loads nor resistance elements. Required whenever
   ``order`` is a strict subset of the section's tendons; ``set(order)``
   and ``set(pending)`` must be disjoint and together cover every tendon
   exactly once.

Omitting ``bonded_before`` is a positive statement, not an absence: it
declares the deferred-single-grouting model, in which no tendon is
bonded during the sequence. That is the ordinary case for a
cast-in-place post-tensioned deck, and it is what the driver assumed
unconditionally before this schedule existed.

Why the schedule changes the answer
-----------------------------------

The elastic-shortening debit that jack :math:`j` imposes on the tendons
already in place is governed by the sectional stiffness the jack acts
against. With :math:`n` tendons already grouted, that section is the
homogenised one,

.. math::

   A_{i} \;=\; A_{c} + \sum_{k \in \mathcal{B}} (n_{p} - 1)\,A_{p,k},
   \qquad
   n_{p} \;=\; \frac{E_{p}}{E_{c}(t)},

with :math:`\mathcal{B}` the set of tendons bonded at that step, and the
second moment :math:`I_{i}` transformed the same way about the
homogenised centroid. Treating :math:`\mathcal{B}` as empty when it is
not understates :math:`A_{i}` and :math:`I_{i}`, overstates
:math:`\Delta\varepsilon_{c}` at the tendon level, and therefore
**overstates the loss**.

The direction matters more than the magnitude. Overstating the loss
understates the final :math:`\sigma_{p}`, which is the
**unconservative** direction for the transfer compression check of
EN 1992-1-1 §5.10.2.2 — the section is reported less compressed than it
is. Measured on a 400×1200 girder carrying three tendons of
:math:`A_{p} = 1500\ \mathrm{mm}^2` at :math:`y = -450, -400, -350`, the
tendon-free view overstates the elastic-shortening loss by **5.11 %**
and understates the final :math:`\sigma_{p}` by **0.46 %**; on a
1000×250 slab the same comparison gives 6.84 % and 0.88 %. Every
tolerance GenSec declares sits at :math:`10^{-9}` relative or below,
eight orders of magnitude further down.

What the declaration costs, and why it is required
--------------------------------------------------

Each distinct bonded set is a different resisting section, so a sequence
with :math:`n` injections solves against at most :math:`n+1` sectional
views instead of one. The views are cached per distinct bonded set and
reused across steps, so the cost is the number of *distinct* sets rather
than the number of steps.

The alternative to declaring is inferring, and inference here is unsafe
in a way that is worth stating. Reading ``order`` as a strict subset and
concluding that the remaining tendons must already be grouted is the
reading GenSec used before Phase 22, and on a timeline it is almost
always wrong: on that path a strict subset overwhelmingly means the
others are **not yet stressed**. An ordinary ``stress → grout → stress``
history therefore died at its *first* grout. The refusal named a real
piece of physics and named it correctly; what it got wrong was the set
of cases it fell on.

The retrofit: refused until it is declared
------------------------------------------

A tendon jacked into a section that **already carries grouted tendons in
service** — the strengthening case — is modelled from Phase 24, and only
when the state that comes with it is declared.

The stiffness algebra was never what was missing: it is the same algebra
the staged case above runs. What was missing is a set of facts no
property of the section carries and no computation can recover:

* the **effective stress** of each existing tendon — its jacking stress
  less the whole deferred history of creep, shrinkage and relaxation;
* the **strain state** of the existing concrete, and the point that
  state is declared about;
* the **two clocks** that history has already run down: the age of the
  concrete since casting, and the time since each cable was tensioned.
  They are decades apart on a member being retrofitted, and relaxation
  runs off the second while creep and shrinkage run off the first.

GenSec does not derive any of them, and — for the effective stress —
will not. That is a decision, not a gap. On an existing member the
stress is measured by lift-off or read from the original design;
reconstructing it would need forty years of casting, loading and
humidity history that nobody can validate and whose error nobody can
bound. Declaring it is the correct input.

So the declaration is
:class:`~gensec.solver.posttension.PreexistingState`, and every field of
it is required:

.. code-block:: python

    from gensec.solver.posttension import (PreexistingState,
                                           solve_posttension_sequence)

    existing = PreexistingState(
        sigma_pe={0: 1100.0},          # MPa, now, per tendon
        eps0=-3.0e-4, chi_x=1.5e-7, chi_y=0.0,
        x_ref=0.0, y_ref=0.0,          # the point the plane is read at
        concrete_age_days=14600.0,     # since casting
        tendon_age_days={0: 14200.0},  # since jacking -- a second clock
    )
    res = solve_posttension_sequence(
        section, sigma_p0=[float("nan"), 1300.0, 1300.0], Ep=195000.0,
        order=[1, 2], already_bonded=existing)

The ``nan`` is not decoration. Tendon 0 was jacked in a history this
call never saw, so its ``sigma_p0`` slot — the stress a jack of *this*
sequence applies — has no value. A finite number there would either
duplicate the declared effective stress or contradict it, and nothing in
the input would say which, so it is refused.

A bare index list, ``already_bonded=[0]``, still refuses. It names
*which* tendons are already grouted and not what state they are in, and
raises :class:`~gensec.solver.posttension.PreexistingStateUndeclared`
with ``reason='preexisting_undeclared'``. The refusal changed cause
between Phase 22 and Phase 24 — from unmodelled physics to undeclared
input — and the two carry different remedies, exactly as ``custom`` and
``undeclared`` do in the normative-code registry. The exception types
now say so: an undeclared input is a :class:`ValueError`,
:class:`~gensec.solver.posttension.LoadBetweenJacksNotModelled` remains
a :class:`NotImplementedError`, because there the physics really is out
of a sectional model's reach.

What each declared fact does
----------------------------

None of them is decoration; each changes a published number.

The tendon indices make the existing cables **bonded elements of the
resisting section from the first jack on**, and **anchored from the
first jack on** — so every new jack debits them by
:math:`\Delta\sigma_p = -E_p\,\Delta\varepsilon_c`. On the reference
400 x 1200 member of ``22_0 §1.1``, jacking two new cables costs the
existing one **104.27 MPa, 9.48 % of** :math:`\sigma_{pe}`, and its
stiffness moves the new cables' stress by **+0.14 %**.

The effective stress is the **baseline the debit is taken from**:
:math:`\sigma_{p,\text{after}} = \sigma_{pe} - \Delta\sigma`. Without
it there is no number to subtract from, which is precisely why the case
could not be answered before.

The declared plane is the **starting value of the accumulated concrete
strain**, hence of ``eps_ref_grout``, hence of the datum
:func:`~gensec.solver.posttension.grout` reconciles against. From
Phase 25 it also **enters the resisting view**, as a bulk pre-strain
offset: the concrete a new jack is pulled against is the concrete the
member is already in, and on any law but a straight line the tangent
depends on where that is. Under Phase 24 the plane only seeded the
accumulator, and two declarations differing by
:math:`-1\times 10^{-4}` against :math:`-9\times 10^{-4}` produced
**identical** losses on the new cables, which is right for a constant
modulus and wrong for concrete.

The pre-existing tendon enters that view at
:math:`\varepsilon_{\text{init}} = \sigma_{pe}/E_p`, so that it carries
the stress it was declared to carry. Not
:math:`\sigma_{pe}/E_p - \varepsilon_{c,\text{declared}}`: the bulk
offset reaches the concrete a tendon displaces but never the strand's
own law, so subtracting the declared plane there would count it twice.
Measured, the difference is invisible in the losses — they are
increments and the constant cancels — and 4.7 % of :math:`N` in
``declared_state_demand``.

The two clocks are **declared and carried, not consumed here** — the
deferred machinery is not reachable from this driver. They are required
all the same, because the first consumer that gets them wrong gets them
wrong in silence.

How the increments are formed
-----------------------------

The result declares it, in ``increment_form``. Since Phase 25 the value
is ``'incremental_on_state'``: each jack's demand increment is applied
to the section **as it exists at that step**, starting from the strain
state already accumulated,

.. math::

   \mathbf{R}\bigl(\varepsilon_{k}\bigr)
     = \mathbf{R}\bigl(\varepsilon_{k-1}\bigr) + \Delta\mathbf{D}_{k},

so that the tangent is taken where the section actually is. It is exact
for any constitutive law and for a bonded set that grows during the
sequence.

The form it replaced solved every jack alone from zero strain and summed
the planes. That is exact when the section is linear elastic and only
then — and the driver neither checked the premise nor said it was making
one. The retrofit is where it hurt most, because the existing concrete
is already deeply compressed: at
:math:`\varepsilon_c = -3\times 10^{-4}` the EN 1992-1-1 §3.1.5 tangent
is already 6.6 % below :math:`E_{cm}`, at :math:`-6\times 10^{-4}` some
19 %, and at :math:`-9\times 10^{-4}` some 32 %. Measured on the
400 x 1200 member of ``22_0 §1.1`` with three cables
:math:`A_p = 1500` under that law: the final :math:`\sigma_p` was off by
**2.1 %** and the elastic-shortening loss — the quantity this driver
exists to produce — by **20 to 33 %**.

.. note::

   A third form, *solve the cumulative action set and difference against
   the previous step*, is exact only while the resisting section is
   **constant**. As soon as a cable is injected mid-sequence it is wrong,
   and measurably so: 0.11 % against the closed form of
   :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`
   on a **linear** staged case, where that closed form is exact. A
   section that grows resists increments; only a constant one may be
   solved cumulatively. This is why ``increment_form`` names a state and
   not a sum.

What the declared state implies
-------------------------------

With the plane in the view and every existing cable at its declared
stress, the resultant of the declared state is something GenSec can
integrate, and it publishes it as ``declared_state_demand``:
:math:`(N, M_x, M_y)`, or ``None`` when nothing was declared — an absent
measurand, not a zero.

It is the base point of the sequence: the first jack is **not** solved
against zero, because a member in service is not at zero, and solving
against zero would move the section off the plane the caller declared
without saying so. Having been computed, it is also the only coherence
check against the original design that GenSec can offer: a member whose
declared state implies a sollecitazione far from the one its design
drawings record has a declaration, a drawing, or a history that is
wrong. GenSec does **not** refuse on it. It does not know which.

A sollecitazione interposed in the sequence
-------------------------------------------

``applied_at={step: (N, Mx, My)}`` applies a sollecitazione **before**
jack ``step``. Unlike ``base_N`` it is **not** debit-free: every tendon
anchored at that instant takes :math:`-E_p\,\Delta\varepsilon_c` from
it, exactly as it does from a jack.

That is why they are two parameters and not one reinterpreted by
context. A caller who wrote ``base_N`` for self-weight and later
declared a retrofit would otherwise find their self-weight silently
debiting a forty-year-old cable.

It is permitted **if and only if every tendon anchored at that step is
also bonded** — the same condition as :ref:`load-between-jacks`, because
it is the same physics. A grouted cable does not slide whenever it
arrived; an ungrouted one slides whenever it arrived. So *grouted
cables, then a load, then more cables* is modelled, and the refusal that
guards it names an actionable remedy: grout the open cable first with
``bonded_before``, and the same load becomes modelled.

``applied_at={0: ...}`` without a declared retrofit is refused: before
the first jack of a virgin section nothing is anchored, nothing can be
debited, and it is ``base_N`` under a second name.


.. _load-between-jacks:

A load between two jacks: the boundary GenSec declares
======================================================

A ``load`` event that falls **between two** ``stress`` **events**, at an
instant when at least one tendon is stressed but **not yet grouted**, is
refused. :class:`~gensec.solver.posttension.LoadBetweenJacksNotModelled`
is raised, naming the tendons, the load event and the two jacking events
it falls between.

This is not a complaint about the input. A load applied halfway through a
stressing sequence is an ordinary construction situation and a timeline
that describes one is well formed. It is GenSec stating the edge of what
a *sectional* model can answer, rather than answering anyway.

Why the edge is there
---------------------

A tendon that has been jacked but not yet grouted has **no strain
compatibility** with the concrete around it: it slides in its duct. Its
force therefore follows the change in *elongation* between the
anchorages, averaged over the free length,

.. math::

   \Delta\sigma_p \;=\; E_p \,\frac{1}{L}
       \int_{0}^{L} \Delta\varepsilon_c(x)\,\mathrm{d}x ,

which is a quantity of the **member**. A sectional model knows only the
local value :math:`\Delta\varepsilon_c(x^{*})` at the section it solves,
and at the critical section — which is exactly the section one analyses —
that local value is an *upper bound* on the mean along the member.
Debiting it would overstate the loss and understate :math:`\sigma_p`,
which is the **unconservative** direction for the serviceability stress
limits of EN 1992-1-1 §5.10.3. The refusal is therefore where the answer
would be wrong, not where it would merely be imprecise.

The elastic-shortening debit **between successive jacks** is kept, and
the asymmetry is deliberate: §5.10.5.1 prescribes that calculation and
has already folded the member-level averaging into its :math:`j`-factor.
No clause does the same for an arbitrary external load acting
mid-sequence, so there is nothing to stand behind an analogous debit.

What is *not* refused
---------------------

Three neighbouring cases are modelled, and each exclusion is a case that
is answered rather than an oversight:

* a load **before the first jack** is the documented meaning of
  ``base_N`` / ``base_Mx`` / ``base_My`` — the sollecitazione the section
  already carries when the sequence begins. It is applied once and
  debit-free, and it has always been the semantics of the driver;
* a load **after the last jack** is not interposed in a sequence: there
  is no subsequent jacking event whose debit it could contaminate;
* a load acting when every stressed tendon is **already grouted** is
  compatible in strain with the section and is resolved by the ordinary
  bonded path.

Where it is enforced, and where it is explained
-----------------------------------------------

Enforcement lives where the interleaving is visible — a timeline, not a
bare driver call — in
:meth:`~gensec.solver.timeline.ConstructionTimeline._check_load_between_jacks`,
called from ``compile_combination`` on the prefix being compiled. The
contract itself belongs to the driver and is stated in full in the Notes
of :func:`~gensec.solver.posttension.solve_posttension_sequence`; called
directly, that function cannot know that its ``base_*`` came from
mid-sequence, so passing one there is outside the contract and the caller
owns the consequence.

The way out, when it is built, will be a **declared** value — an
engineer-supplied :math:`\Delta\sigma_p` per tendon and per event,
computed at member level by something that is not GenSec, carried with
its provenance and marked in the dump as declared rather than calculated.
It is deliberately not implemented: a number GenSec did not compute must
never be indistinguishable from one it did.
