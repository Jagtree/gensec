# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.
#
# GenSec is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# GenSec is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GenSec.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
Sequential **bonded post-tensioning** driver (prestress Phase 4 / Step C).

This module is the fiber-method peer of
:mod:`gensec.solver.prestress_transfer` for *post-tensioning*.  Where
pre-tensioning releases all strands **simultaneously** (the loss
*emerges* from a single equilibrium solve), post-tensioning stresses
the tendons **sequentially**, each against the already-hardened
concrete; the elastic-shortening loss therefore **cannot emerge** from
one solve — it must be read and debited event by event.

The pre/post asymmetry, in one paragraph
----------------------------------------
A post-tensioned tendon being stressed at step *k* is **not** a section
element: it is a free cable, jacked to its target stress exactly, the
jack reacting on the section as the external action

.. math::

    (N_k,\,M_{x,k},\,M_{y,k})
        = \bigl(-P_k,\; -P_k\,(y_k - y_\text{ref}),\;
                +P_k\,(x_k - x_\text{ref})\bigr),
    \qquad P_k = \sigma_{p0,k}\,A_{p,k},

i.e. a :class:`~gensec.solver.section_state.PrestressAction`
(:meth:`PrestressAction.from_force`).  Consequently:

* the tendon stressed at step *k* suffers **no** loss from its own
  stressing (it is not strain-compatible with the concrete at that
  instant);
* every tendon *j* stressed **before** *k* and already anchored takes
  the concrete strain increment at its level and is debited

  .. math::

      \Delta\sigma_{p,j} = -E_p\,\Delta\varepsilon_{c,j},

  where :math:`\Delta\varepsilon_{c,j}` is read from the **solved
  strain plane** of the step-*k* event at the coordinate of tendon *j*.

Because the action is a *fixed* triple, the loss is **not emergent**: it
is computed explicitly from the plane.  This is the central distinction
from :func:`gensec.solver.prestress_transfer.solve_pretension_transfer`.

Relationship to the closed-form validator
------------------------------------------
For a **linear-elastic** transfer section the fiber solve reproduces, to
machine precision, the planar transformed-section algebra of
:func:`gensec.solver.prestress_transfer.sequential_posttension_loss`
(same linear system, two code paths).  That closed form is this
driver's **acceptance test** — see ``tests/test_posttension.py`` and
``run_posttension_sequence_validation.py``.  In production the bulk may
be nonlinear; the driver is material-agnostic (it only reads strain
planes), but only the linear case is checkable against the closed form.

Decision locked at the start of this driver (per the Step-C spec)
-----------------------------------------------------------------
**Elastic-shortening sequence fidelity** — the default is the *explicit
one-pass* debit (EC2 §5.10.5.1 / NTC 2018 §4.1.8.1.4 level, the
:math:`(n-1)/2n` mean-debit rule for *n* identical centroidal tendons).
The *self-consistent / coupled* scheme (``scheme="coupled"``) is
**opt-in**: it iterates because debiting a tendon reduces its action,
which reduces the shortening — a second-order feedback worth ~0.3 %
(quantified by the validator), the very reason EC2 stops at the explicit
form.  Both fidelity levels live behind one signature; Phase 5
(time-dependent losses) will later realise the *same* code-level
interface (state → effective prestrain) while staying a conceptually
separate, interval-triggered operator.

The single-side prestress invariant (grouting)
----------------------------------------------
A tendon is a load **xor** a resistance element, never both, never
neither.  At the grouting stage the rigorous transition is **atomic**
(:func:`grout`): the tendon's :class:`PrestressAction` is dropped from
the demand **and** the tendon enters the resistance domain
(:meth:`SectionState.with_grouted`: ``bonded`` flipped ``True`` and a
*reconciled* ``eps_init`` baked in) in one step.  The conservative
treatment simply never calls :func:`grout`: the
:class:`PrestressAction` stays a load for life.

References
----------
- EN 1992-1-1:2004 §5.10.5.1 (immediate losses — elastic deformation of
  concrete; sequential post-tensioning).
- NTC 2018 §4.1.8.1.4 (perdite istantanee per deformazione elastica).
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import (Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)

import numpy as np

from .integrator import FiberSolver
from .section_state import (
    PrestressAction,
    SectionState,
    materialize_view,
)


# ===========================================================================
#  Scope boundary
# ===========================================================================

@dataclass(frozen=True)
class PreexistingState:
    r"""
    The **declared** state of a section already in service — the retrofit.

    A new tendon jacked into an existing member is not a harder version
    of the staged-injection case; it is a case about facts GenSec does
    not have.  ``Tendon.bonded`` is the intrinsic property, ``True`` for
    a virgin cable and for a forty-year-old one alike, and nothing in
    the section says which is which, nor at what stress the old one
    sits, nor how strained the concrete around it already is.

    Phase 24 does not derive any of that.  It makes it **declarable**.
    The engineer who has those numbers — from a previous analysis, from
    site measurement, from an external calculation — states them here
    and gets an answer; the one who does not gets a refusal that names
    what is missing and why GenSec will not invent it.

    Every field is required.  There is no default effective stress and
    no default strain state, for the reason there is no default
    concrete class: a number nobody declared, silently supplied, is
    indistinguishable in the output from one that was measured.

    Parameters
    ----------
    sigma_pe : dict
        ``{tendon index: effective stress [MPa]}`` for **every** tendon
        already grouted when this sequence begins.  This is the stress
        *now* — the jacking stress minus the whole deferred history of
        creep, shrinkage and relaxation — and it is the baseline the
        elastic-shortening debit of the new jacks is subtracted from.
        Deriving it from that history is **not** in scope here: it is a
        time integration over a member's life (Phase-5 machinery driven
        from a timeline), registered as D1b-2.
    eps0 : float
        Strain of the existing concrete at ``(x_ref, y_ref)`` [-].
        Compression negative, as everywhere in GenSec.
    chi_x : float
        Curvature of the existing concrete about the *x* axis [1/mm].
    chi_y : float
        Curvature about the *y* axis [1/mm].
    x_ref, y_ref : float
        The point ``eps0`` is declared **at** [mm], in section
        coordinates.  A plane without its reference point is not a
        plane: the driver pins its own reference at the centroid of the
        tendon-free view, which the caller has no way to know, and a
        declaration read about the wrong point is wrong by
        :math:`\chi\,\Delta y` — at :math:`\chi_x = 1.5\times
        10^{-7}` a 100 mm mismatch is :math:`1.5\times 10^{-5}`, five
        per cent of a typical declared strain, silently.
    concrete_age_days : float
        Age of the existing concrete at the instant this sequence
        begins [days], counted from casting.  **Declared and carried,
        not consumed here**: the deferred machinery is not reachable
        from this driver.  It is required all the same, because the
        first consumer that gets it wrong gets it wrong in silence — a
        creep-and-shrinkage continuation restarted from a zero clock
        re-spends strains the member spent decades ago.
    tendon_age_days : dict
        ``{tendon index: days since jacking}``, one entry for every key
        of *sigma_pe*.  A **second, different clock**: relaxation runs
        from the day a cable was tensioned, not from the day its
        concrete was cast, and the two are decades apart on a member
        being retrofitted.  Also declared and carried, not consumed
        here.  A forty-year-old cable has spent nearly all of its
        relaxation; a continuation that restarted its clock would
        re-spend the whole of it, of order two to three per cent of
        :math:`\sigma_p`.

    Notes
    -----
    **What the declared plane is, and what it is not.**  The triple
    :math:`(\varepsilon_0, \chi_x, \chi_y)` is the strain field of the
    existing concrete, evaluated with the sign convention of
    :meth:`~gensec.solver.integrator.FiberSolver.strain_field`:

    .. math::

        \varepsilon_c(x, y) = \varepsilon_0
            + \chi_x\,(y - y_{\mathrm{ref}})
            - \chi_y\,(x - x_{\mathrm{ref}}).

    It is the **total** state the member is in: the existing prestress,
    the permanent loads, the deferred strains, everything, already
    combined.  GenSec does not decompose it and does not ask which part
    came from where — a decomposition it cannot verify would be a guess
    dressed as an input.

    That is also why a retrofit sequence takes **no base
    sollecitazione**.  The existing state is declared once, as this
    plane; a ``base_N`` alongside it would state part of the same fact
    a second time, and nothing would say whether the existing tendons
    should be debited for it.  The driver refuses the combination.

    **Where each declared fact is consumed.**  None of the three is
    decoration; each one changes a published number:

    ==============================  =====================================
    declared                        consumed as
    ==============================  =====================================
    the tendon indices              bonded elements of the resisting
                                    section from the first jack on, and
                                    anchored from the first jack on, so
                                    every new jack debits them
    :math:`\sigma_{pe}`             the baseline of
                                    :math:`\sigma_{p,\text{after}}
                                    = \sigma_{pe} - \Delta\sigma`
    the plane                       the initial value of the accumulated
                                    concrete strain, hence
                                    ``eps_ref_grout``, hence the datum
                                    :func:`grout` reconciles against
    ==============================  =====================================

    The plane deliberately does **not** enter the per-jack solves.
    Those are increments: each one applies a single jack's couple and
    reads back the response.  An imposed-strain offset present in every
    increment would contribute its own force to every one of them, and
    the accumulated shortening — hence the loss — would be wrong by a
    constant per jack.  A state belongs to the state, not to each
    increment of it.

    Raises
    ------
    ValueError
        If ``sigma_pe`` is empty, if an index is not an integer, or if
        an effective stress is not finite and strictly positive.

    See Also
    --------
    PreexistingStateUndeclared : the refusal raised without this object.
    solve_posttension_sequence : the driver that accepts it.
    """

    sigma_pe: Mapping[int, float]
    eps0: float
    chi_x: float
    chi_y: float
    x_ref: float
    y_ref: float
    concrete_age_days: float
    tendon_age_days: Mapping[int, float]

    def __post_init__(self):
        r"""
        Validate the declaration at the point of declaration.

        A refusal here names the field and the reason, which is worth
        more than the same refusal raised three call levels down where
        the caller can no longer see which number was wrong.
        """
        if not self.sigma_pe:
            raise ValueError(
                "PreexistingState: `sigma_pe` is empty. An empty "
                "declaration is not a declaration — it says nothing "
                "about a section already in service. If no tendon is "
                "grouted before this sequence, omit `already_bonded` "
                "entirely: that is a different statement about the "
                "section and it is the one you want.")
        clean = {}
        for key, value in self.sigma_pe.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"PreexistingState: `sigma_pe` key {key!r} is not a "
                    f"tendon index. Keys are 0-based indices into "
                    f"`section.tendons`.") from None
            val = float(value)
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(
                    f"PreexistingState: `sigma_pe[{idx}]`={value!r} is "
                    f"not a finite positive stress [MPa]. A tendon at "
                    f"zero or negative effective stress is not a "
                    f"prestressed tendon, and a non-finite one is not a "
                    f"number; either would run to completion and "
                    f"publish a capacity nobody could question. If the "
                    f"cable really is slack, it is not part of the "
                    f"resistance and does not belong in this "
                    f"declaration.")
            clean[idx] = val
        if set(self.tendon_age_days) != set(clean):
            raise ValueError(
                f"PreexistingState: `tendon_age_days` covers tendons "
                f"{sorted(self.tendon_age_days)} and `sigma_pe` covers "
                f"{sorted(clean)}. Every declared tendon needs both: a "
                f"stress without its clock cannot be continued in time, "
                f"and a clock without its stress describes nothing.")
        ages = {}
        for key, value in self.tendon_age_days.items():
            val = float(value)
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(
                    f"PreexistingState: `tendon_age_days[{int(key)}]`="
                    f"{value!r} is not a finite non-negative number of "
                    f"days since jacking.")
            ages[int(key)] = val
        object.__setattr__(self, "tendon_age_days",
                           dict(sorted(ages.items())))
        age = float(self.concrete_age_days)
        if not math.isfinite(age) or age < 0.0:
            raise ValueError(
                f"PreexistingState: `concrete_age_days`="
                f"{self.concrete_age_days!r} is not a finite "
                f"non-negative age in days.")
        object.__setattr__(self, "concrete_age_days", age)
        for name in ("eps0", "chi_x", "chi_y", "x_ref", "y_ref"):
            val = float(getattr(self, name))
            if not math.isfinite(val):
                raise ValueError(
                    f"PreexistingState: `{name}`="
                    f"{getattr(self, name)!r} is not finite. The strain "
                    f"state of the existing concrete is a declared "
                    f"fact; declare 0.0 if it really is zero. The "
                    f"reference point `x_ref`/`y_ref` is part of that "
                    f"fact: a plane is only a plane about a point.")
            object.__setattr__(self, name, val)
        object.__setattr__(self, "sigma_pe", dict(sorted(clean.items())))

    @property
    def tendons(self) -> Tuple[int, ...]:
        r"""Declared pre-existing tendon indices, ascending."""
        return tuple(sorted(self.sigma_pe))

    def plane_at(self, x, y):
        r"""
        The declared strain plane evaluated at points.

        Takes **no** reference point: the declaration carries its own,
        so the caller who wrote the plane and the driver that reads it
        cannot disagree about what the numbers mean.  The driver pins a
        reference of its own at the centroid of the tendon-free view;
        passing that one in here would silently re-anchor a plane
        declared about a different point.

        Parameters
        ----------
        x, y : numpy.ndarray
            Point coordinates [mm].

        Returns
        -------
        numpy.ndarray
            :math:`\varepsilon_0 + \chi_x (y - y_{\mathrm{ref}})
            - \chi_y (x - x_{\mathrm{ref}})`, about the **declared**
            reference point.
        """
        return (self.eps0
                + self.chi_x * (np.asarray(y, dtype=float) - self.y_ref)
                - self.chi_y * (np.asarray(x, dtype=float) - self.x_ref))

    def locked_in_strain(self, x, y, Ep):
        r"""
        The locked-in tendon strain implied by the declaration, **at the
        instant the declaration is made**.

        A bonded tendon in GenSec carries an initial strain
        :math:`\varepsilon_{\mathrm{init}}` such that the strand law,
        evaluated at *section strain plus initial strain*, returns the
        stress the tendon is actually at:

        .. math::

            \varepsilon_{\mathrm{init},j}
                = \varepsilon_{pe,\mathrm{eff},j}
                  - \varepsilon_{c,j},
            \qquad
            \varepsilon_{pe,\mathrm{eff},j} = \sigma_{pe,j} / E_{p,j},
            \qquad
            \varepsilon_{c,j} = \texttt{plane\_at}(x_j, y_j).

        This is the **same relation** :func:`grout` applies after a
        sequence, with the two terms taken from the declaration instead
        of from a solved sequence: for an existing cable the concrete
        strain at its level is the declared plane, and its effective
        strain is its declared effective stress over its modulus.  The
        quotient form of :math:`\varepsilon_{pe,\mathrm{eff}}` is not a
        choice made here — it is the driver's own definition
        (``eps_pe_eff = sigma_after / Ep``), read from the one place
        that owns it, so that a tendon seeded through this method and
        the same tendon reconciled after the first jack cannot mean two
        different things.

        Why the method exists at all.  A caller that begins a
        construction history on an existing member must place the
        pre-existing cables in the resisting section **before the first
        event**, and at that instant no sequence has run, so there is no
        :class:`PosttensionSequenceResult` to read a datum from.  The
        declaration is the only source, which is precisely what it is
        for.

        Parameters
        ----------
        x, y : numpy.ndarray
            Coordinates of the tendons [mm], in section coordinates,
            aligned with *Ep*.
        Ep : numpy.ndarray
            Prestressing-steel modulus of each tendon [MPa], aligned
            with *x* and *y*.  Passed in rather than read off a
            material here: which modulus a tendon has is the section's
            fact, not the declaration's.

        Returns
        -------
        numpy.ndarray
            :math:`\varepsilon_{\mathrm{init}}` for the tendons whose
            index appears in :attr:`sigma_pe`, in the order of
            ``sorted(self.tendons)``.

        Raises
        ------
        ValueError
            If a modulus is not finite and strictly positive.  A
            zero or negative modulus would turn the quotient into an
            infinity or a sign error, and the tendon would enter the
            section carrying it.

        See Also
        --------
        grout : forms the same quantity from a solved sequence.
        """
        idx = sorted(self.tendons)
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        eps_c = self.plane_at(xs, ys)
        out = np.zeros(len(idx), dtype=float)
        for k, j in enumerate(idx):
            E = float(np.asarray(Ep, dtype=float).ravel()[k])
            if not np.isfinite(E) or E <= 0.0:
                raise ValueError(
                    f"PreexistingState.locked_in_strain: tendon {j} has "
                    f"Ep={E!r}, which is not finite and strictly "
                    f"positive. The locked-in strain is "
                    f"sigma_pe/Ep - eps_c and cannot be formed.")
            out[k] = float(self.sigma_pe[j]) / E - float(
                np.asarray(eps_c, dtype=float).ravel()[k])
        return out


class PreexistingStateUndeclared(ValueError):
    r"""
    Tendons are declared **already bonded** without the state that goes
    with them.

    Until Phase 24 this refusal was called ``PreexistingBondedNotModelled``
    and it said something else: that GenSec did not model the retrofit
    at all.  That is no longer true.  What is refused now is an
    **undeclared input**, and the two carry different remedies — the
    same distinction the normative-code registry draws between
    ``custom`` (declared, and GenSec derives nothing) and ``undeclared``
    (nothing said, so nothing to work from).

    ``already_bonded`` given as a bare sequence of indices says *which*
    tendons are already grouted.  It does not say at what stress, and it
    does not say what state the concrete around them is in.  Those are
    the two facts an existing member arrives with and that no property
    of the section carries, so GenSec asks for them by name.

    What it will **not** do, ever, is *derive* the effective stress from
    the deferred history.  That is a decision and not a gap: on an
    existing member the stress is measured by lift-off or read from the
    original design, and reconstructing it would require the casting,
    loading and humidity history of the member's whole life — an input
    nobody can validate and whose error nobody can bound.  Declaring it
    is the correct input, not a fallback for a missing feature.

    Attributes
    ----------
    reason : str
        Stable machine-readable name of the refusal,
        ``'preexisting_undeclared'``.  Read by
        ``tools/numeric_fingerprint.py``: a refusal that changes cause
        moves no number at all, so the instrument records the *name*
        (the Phase-20 ``outer_reason`` rule, ``20_1 §3``).
    tendons : tuple of int
        The tendon indices declared already bonded.
    n_tendons : int
        Size of the section's tendon set, for the message.

    See Also
    --------
    PreexistingState : the declaration that lifts this refusal.
    solve_posttension_sequence : the driver that raises it.
    IntraSequenceUndeclared : the sibling refusal, for a tendon whose
        role in the sequence was never stated at all.
    """

    reason = "preexisting_undeclared"

    # A ``ValueError`` and not a ``NotImplementedError``: what is refused
    # is an input nobody declared, not a physics GenSec cannot do.  The
    # CLI reads the difference -- ``NotImplementedError`` prints "outside
    # GenSec's modelled scope", which would be a false statement about a
    # case that is modelled the moment the numbers arrive.  The sibling
    # ``LoadBetweenJacksNotModelled`` keeps ``NotImplementedError``,
    # because there the physics really is out of reach: an ungrouted
    # tendon follows a member-level integral a sectional model has no
    # access to.  Unmodelled physics and undeclared input are different
    # facts with different remedies, exactly as ``custom`` and
    # ``undeclared`` are in the normative-code registry.

    def __init__(self, tendons, n_tendons):
        self.tendons = tuple(int(t) for t in tendons)
        self.n_tendons = int(n_tendons)
        super().__init__(
            f"solve_posttension_sequence: `already_bonded`="
            f"{list(self.tendons)} names which tendons are already "
            f"grouted when this sequence begins, and not the state they "
            f"are in. A section already in service arrives with two "
            f"facts GenSec cannot read off it: the EFFECTIVE STRESS of "
            f"each existing tendon, and the STRAIN state of the "
            f"existing CONCRETE with the datum it is referred to. "
            f"Declare both — pass a `PreexistingState(sigma_pe="
            f"{{tendon: MPa}}, eps0=..., chi_x=..., chi_y=...)` instead "
            f"of a bare index list — and the retrofit is modelled. "
            f"GenSec will NOT derive the effective stress from the "
            f"deferred history of creep, shrinkage and relaxation: "
            f"that is a time integration over the member's life "
            f"(D1b-2), not a driver call. Interleaved stress/grout "
            f"*within* one sequence is a different case and is "
            f"modelled: `bonded_before={{tendon: step}}`.")


class IntraSequenceUndeclared(ValueError):
    r"""
    ``order`` covers only part of the section's tendons, and the rest
    have not been declared.

    A strict subset of the tendons being stressed has exactly two
    meanings, and they need opposite treatments:

    * the others are **not yet stressed** at this instant — they are
      neither loads nor elements, and the sequence is a prefix of a
      longer history.  Declare them in ``pending``;
    * the others are **already bonded** — the retrofit, modelled from
      Phase 24 when the state that comes with it is declared.  Pass a
      :class:`PreexistingState` in ``already_bonded``; a bare index
      list there raises :class:`PreexistingStateUndeclared`.

    Before Phase 22 the driver assumed the second and refused, so a
    perfectly ordinary staged-injection timeline — stress, grout, stress
    — was rejected with a message about a stiffness that had nothing to
    do with it.  The refusal is now the *absence of a declaration*, which
    is a different fact and has a different remedy.  **No default here**:
    guessing which of the two the caller meant is precisely the silent
    choice GenSec does not make.

    Attributes
    ----------
    reason : str
        ``'intra_sequence_undeclared'``.
    missing : tuple of int
        Tendon indices absent from both ``order`` and ``pending``.

    See Also
    --------
    solve_posttension_sequence
    """

    reason = "intra_sequence_undeclared"

    def __init__(self, missing, n_tendons):
        self.missing = tuple(int(t) for t in missing)
        self.n_tendons = int(n_tendons)
        super().__init__(
            f"solve_posttension_sequence: tendons {list(self.missing)} "
            f"are in neither `order` nor `pending`. A strict subset of "
            f"the section's {self.n_tendons} tendons being stressed has "
            f"two meanings and GenSec will not pick one: if they are "
            f"NOT YET STRESSED at this instant, list them in `pending` "
            f"(they are then neither loads nor elements); if they are "
            f"ALREADY GROUTED before this sequence, declare them in "
            f"`already_bonded` as a `PreexistingState`. "
            f"Grouting BETWEEN two jacks of this sequence is a third "
            f"case and is modelled: `bonded_before={{tendon: step}}`.")


class LoadBetweenJacksNotModelled(NotImplementedError):
    r"""
    An external load acts between two jacking events, on a tendon that is
    stressed but not yet grouted.

    This is **not** an error in the engineer's input: a load applied
    halfway through a stressing sequence is an ordinary construction
    situation, and a timeline that describes one is well formed.  It is
    GenSec declaring the edge of what a *sectional* model can answer.

    In one sentence: an ungrouted tendon has no strain compatibility with
    the section, so its force change follows the *mean* concrete strain
    over the free length between the anchorages — a **member** quantity —
    while a sectional model knows only the local value, which at the
    critical section is an upper bound on that mean and therefore
    unconservative for :math:`\sigma_p`.

    **The derivation is not repeated here.**  It is written once, in
    full, in the Notes of :func:`solve_posttension_sequence`, which owns
    the contract, and rendered for the reader in the narrative section
    *A load between two jacks: the boundary GenSec declares* of
    ``docs/construction_timeline.rst``.  One home for one fact: a second
    copy of a normative argument is a second copy that can drift.  The
    exception's own ``args[0]`` carries the argument in prose, because
    that is what an engineer actually reads when the refusal fires.

    Being a :class:`NotImplementedError` places it in the scope-boundary
    half of the exception taxonomy, next to
    :class:`~gensec.materials.base.OutOfDeclaredDomain`: the request is
    well formed and GenSec has no authority to answer it.  It is
    deliberately **not** a :class:`ValueError`; a caller that catches
    ``ValueError`` to mean "the engineer typed something wrong" must not
    catch this.

    Attributes
    ----------
    tendons : tuple of str
        The tendons that are stressed-but-not-grouted when the load acts.
        Named individually because the future way out is per tendon.
    load_index : int
        Index of the ``load`` event in the physical event stream.
    load_demand : str
        Name of the demand the ``load`` event references.
    prev_stress_index : int
        Index of the last ``stress`` event before the load.
    next_stress_index : int
        Index of the first ``stress`` event after the load.
    combo_name : str or None
        Combination being compiled, for the message.

    See Also
    --------
    solve_posttension_sequence : the driver whose contract this states.
    ConstructionTimeline._check_load_between_jacks : where it is raised.

    Notes
    -----
    The way out, when it is built, is a **declared** per-tendon,
    per-event value — ``declared_delta_sigma_p`` — supplied by the
    engineer from a member-level calculation GenSec did not do.  It is
    declared as an open item in ``handoff/18_0``; this class exists with
    per-tendon and per-event granularity so that adding it will not
    require reopening the driver to find out *which* tendon and *which*
    event were refused.
    """

    def __init__(self, tendons, load_index, load_demand,
                 prev_stress_index, next_stress_index, combo_name=None):
        self.tendons = tuple(str(t) for t in tendons)
        self.load_index = int(load_index)
        self.load_demand = str(load_demand)
        self.prev_stress_index = int(prev_stress_index)
        self.next_stress_index = int(next_stress_index)
        self.combo_name = None if combo_name is None else str(combo_name)
        where = ("" if self.combo_name is None
                 else f"combination '{self.combo_name}': ")
        super().__init__(
            f"{where}construction_history[{self.load_index}] applies load "
            f"'{self.load_demand}' between the jacking events "
            f"[{self.prev_stress_index}] and [{self.next_stress_index}], "
            f"while tendon(s) {list(self.tendons)} are stressed but not "
            f"yet grouted.  An ungrouted tendon has no strain "
            f"compatibility with the section: its force change is "
            f"governed by the mean concrete strain over the free length "
            f"between the anchorages, Dsigma_p = E_p*(1/L)*int(De_c dx), "
            f"which is a member quantity.  A sectional model knows only "
            f"the local value, which on the critical section is an upper "
            f"bound on that mean and therefore unconservative for the "
            f"sigma_p check of EN 1992-1-1 5.10.3.  GenSec does not model "
            f"it.  Ways out: grout the tendon(s) before the load, move "
            f"the load outside the stressing sequence (a load before the "
            f"first jack is the initial state, base_N/base_Mx/base_My), "
            f"or split the model."
        )


# ===========================================================================
#  Result containers
# ===========================================================================

@dataclass
class PosttensionStageResult:
    r"""
    Outcome of one stressing **stage** (one tendon jacked).

    Attributes
    ----------
    stage_index : int
        0-based position in the stressing sequence.
    stressed_tendon : int
        Index (into ``section.tendons``) of the tendon jacked at this
        stage.
    action : PrestressAction
        The **incremental** action applied at this stage (the jacked
        tendon's couple, at the effective jacking stress used by the
        active scheme).
    cumulative_demand : tuple of float
        Running prestress demand ``(N, Mx, My)`` after this stage
        [N, N·mm, N·mm] — the sum of the base sollecitazione and every
        action applied up to and including this stage.
    eps_c : numpy.ndarray
        Concrete section strain at **every** tendon location,
        accumulated through this stage [-] (shape ``(n_tendons,)``).
    sigma_p : numpy.ndarray
        Current effective strand stress at every tendon after this
        stage's debits [MPa] (shape ``(n_tendons,)``).
    converged : bool
        Whether the stage's equilibrium solve converged.
    report : dict
        Opaque, human-auditable payload for the stage ``report`` field
        (the values the engine deliberately surfaces rather than burying
        in the solver).
    """

    stage_index: int
    stressed_tendon: int
    action: PrestressAction
    cumulative_demand: Tuple[float, float, float]
    eps_c: np.ndarray
    sigma_p: np.ndarray
    converged: bool
    report: dict = field(default_factory=dict)


@dataclass
class PosttensionSequenceResult:
    r"""
    Aggregate outcome of a full sequential post-tensioning analysis.

    All per-tendon arrays are index-aligned with ``section.tendons``
    (declaration order), **not** with the stressing ``order``.

    Attributes
    ----------
    order : list of int
        Stressing order actually applied (tendon indices).
    scheme : {"one_pass", "coupled"}
        Elastic-shortening fidelity scheme used.
    sigma_p0 : numpy.ndarray
        Jacking stress per tendon [MPa] (the input, net of friction and
        anchorage slip).
    sigma_p_after : numpy.ndarray
        Effective strand stress after the whole sequence [MPa].
    loss : numpy.ndarray
        Cumulative immediate elastic-shortening loss per tendon
        :math:`\Delta\sigma_p = \sigma_{p0}-\sigma_{p,\text{after}}`
        [MPa] (positive = loss of tension; may be **negative** for a
        tendon whose stress is *raised* by an opposite-side eccentric
        event).
    eps_pe_eff : numpy.ndarray
        Effective jacking strain per tendon after losses,
        :math:`\sigma_{p,\text{after}}/E_p` [-] — the prestrain the
        grouting stage bakes into the resistance.
    eps_ref_grout : numpy.ndarray
        Concrete section strain at each tendon at the **end** of the
        sequence [-] — the datum the grouting reconciliation consumes
        so that :math:`\sigma_p(\varepsilon_{\text{sec,grout}}
        + \varepsilon_{\text{init}})` reproduces
        ``sigma_p_after``.
    Ep : numpy.ndarray
        Tendon modulus per tendon [MPa].
    Ap : numpy.ndarray
        Tendon area per tendon [mm²].
    x_ref, y_ref : float
        Reference point the actions and strains are taken about [mm].
    stages : list of PosttensionStageResult
        Per-stage history (length = number of stressed tendons).
    converged : bool
        Whether every stage's equilibrium solve converged.
    coupled_iterations : int
        Fixed-point sweeps used by ``scheme="coupled"`` (1 for
        ``one_pass``).
    preexisting : PreexistingState or None
        The declared state of the existing member when this sequence
        was a **retrofit**, ``None`` otherwise.  Carried on the result
        because :func:`grout` needs it: a tendon bonded before the
        sequence began is bonded after it ends, and a grouting
        transition that rebuilt the state from "every tendon an
        inactive load" would drop it out of the resistance domain in
        silence.  It also makes the distinction re-derivable from the
        result alone — ``sigma_p0`` of a pre-existing tendon is its
        declared effective stress, not a jacking stress, and nothing
        else on the object would say so.
    increment_form : str
        **How the increments were formed**, declared rather than
        assumed.  ``'incremental_on_state'``: each jack's demand
        increment is applied to the section *as it exists at that
        step*, starting from the strain state already accumulated, so
        that the tangent is taken where the section actually is.

        The form this replaced was ``'incremental_from_origin'``:
        every jack solved alone from zero strain and the planes summed.
        That is exact when the section is linear elastic and only then,
        and the driver neither checked the premise nor declared it
        (D1c).  Measured on 400 x 1200 with three tendons
        :math:`A_p = 1500` under the EN 1992-1-1 §3.1.5 service law:
        the final :math:`\sigma_p` was off by **2.2 %**, and the
        elastic-shortening loss, which is what this driver exists to
        produce, by far more.

        A third form — solve the **cumulative** action set and
        difference against the previous step — is *also* wrong as soon
        as the bonded set changes during the sequence, and by 0.11 %
        on a linear staged case where the closed form of
        :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`
        is exact.  A section that grows resists increments; only a
        constant one may be solved cumulatively.
    declared_state_demand : tuple of float or None
        :math:`(N, M_x, M_y)` that the **declared** pre-existing state
        implies, integrated on the view carrying it: the concrete at
        its declared plane, plus every pre-existing tendon at its
        declared effective stress.  ``None`` when nothing was declared
        — an absent measurand, not a zero.

        It is the base point of the sequence: the first jack is not
        solved against zero, because a member in service is not at
        zero, and solving against zero would move the section off the
        plane the caller declared without saying so.  Published because
        it is also the only coherence check against the original design
        that GenSec can offer.  GenSec does **not** refuse on it: it
        does not know what the right value is.
    """

    order: List[int]
    scheme: str
    sigma_p0: np.ndarray
    sigma_p_after: np.ndarray
    loss: np.ndarray
    eps_pe_eff: np.ndarray
    eps_ref_grout: np.ndarray
    Ep: np.ndarray
    Ap: np.ndarray
    x_ref: float
    y_ref: float
    stages: List[PosttensionStageResult]
    converged: bool
    coupled_iterations: int
    preexisting: Optional[PreexistingState]
    increment_form: str = "incremental_on_state"
    declared_state_demand: Optional[Tuple[float, float, float]] = None

    # -- derived, demand-side artefacts ------------------------------------

    def effective_actions(self, only=None) -> List[PrestressAction]:
        r"""
        Per-tendon life-long load actions at the **effective**
        (post-loss) strand stress.

        These are the demand-side loads of the **conservative**
        treatment (tendon never grouted): the section carries each
        tendon's couple at :math:`\sigma_{p,\text{after}}` for life.

        Parameters
        ----------
        only : sequence of int, optional
            Restrict the emission to these tendon indices.  ``None``
            (the default) means every tendon of the section, which is
            the all-loads treatment this method has always served.

            The parameter exists for a **retrofit**, and its meaning is
            narrow on purpose.  On a retrofit the pre-existing cables
            are bonded before the sequence begins: their force is
            already inside the declared concrete strain plane
            (:class:`PreexistingState`), so emitting it again as a
            demand-side action would count it twice.  A caller that
            walks a construction history needs the actions of the
            cables that are **still loads** — stressed and not yet
            grouted — and of no others.  Naming that set is not a way
            around the refusal below; it is the statement that the
            pre-existing cables are not being asked for.

        Returns
        -------
        list of PrestressAction
            One action per requested tendon, in index order.  With
            *only* omitted the list is aligned with
            ``section.tendons``.

        Raises
        ------
        ValueError
            If the emission would include a tendon declared
            pre-existing.  The refusal is on the **overlap**, not on
            the presence of a declaration: a retrofit result asked for
            the actions of its new cables only is a well-posed
            question, and the one this method now answers.

        See Also
        --------
        PreexistingState.locked_in_strain : the datum a pre-existing
            cable enters the resisting section with instead.
        """
        n = int(self.sigma_p_after.size)
        idx = (list(range(n)) if only is None
               else sorted({int(i) for i in only}))
        bad = [j for j in idx if not 0 <= j < n]
        if bad:
            raise ValueError(
                f"PosttensionSequenceResult.effective_actions: `only` "
                f"names tendon(s) {bad}, which are not indices of this "
                f"result (0..{n - 1}).")
        if self.preexisting is not None:
            overlap = sorted(set(idx) & set(self.preexisting.tendons))
            if overlap:
                raise ValueError(
                    f"PosttensionSequenceResult.effective_actions: this "
                    f"result comes from a retrofit, and tendons "
                    f"{overlap} were already in the "
                    f"resistance domain before the sequence began. The "
                    f"all-loads treatment this method serves -- every "
                    f"tendon a demand-side action for life -- cannot "
                    f"describe them: their force is already inside the "
                    f"declared concrete strain plane, so emitting it "
                    f"again would count it twice. Use `grout()`, which "
                    f"keeps them bonded and reconciles their datum, or "
                    f"pass `only=` naming the cables that are still "
                    f"loads.")
        acts = []
        for j in idx:
            P = float(self.sigma_p_after[j] * self.Ap[j])
            acts.append(PrestressAction.from_force(
                P, self._x[j], self._y[j],
                x_ref=self.x_ref, y_ref=self.y_ref,
                label=f"tendon{j}", origin="posttension_effective"))
        return acts

    def total_effective_action(self) -> PrestressAction:
        r"""Resultant of all :meth:`effective_actions` as one triple."""
        N = Mx = My = 0.0
        for a in self.effective_actions():
            N += a.N
            Mx += a.Mx
            My += a.My
        return PrestressAction(N, Mx, My, label="prestress_total",
                               origin="posttension_effective")

    # ``_x`` / ``_y`` are stamped by the driver (tendon coordinates).
    _x: np.ndarray = field(default=None, repr=False)
    _y: np.ndarray = field(default=None, repr=False)


@dataclass
class GroutResult:
    r"""
    Outcome of the rigorous (bonded) grouting transition.

    Attributes
    ----------
    state : SectionState
        The post-grouting section state: grouted tendons are
        ``active & bonded`` with their reconciled ``eps_init``; tendons
        left ungrouted stay inactive (their force remains a demand-side
        load).  Its capacity hash differs from the pre-grout state, so a
        :class:`~gensec.solver.section_state.StagedDomainManager` rebuilds
        the resistance domain automatically.
    grouted : list of int
        Tendon indices brought into the domain.
    dropped_actions : list of PrestressAction
        The loads removed from the demand at grouting (one per grouted
        tendon).  **These must be subtracted from the cumulative demand**
        in the same stage they enter the domain — that is the
        single-side invariant made explicit.
    residual_actions : list of PrestressAction
        The loads that **remain** in the demand (ungrouted tendons), at
        their effective post-loss stress.
    report : dict
        Per-grouted-tendon reconciliation record (visible correction):
        ``eps_ref_grout``, ``eps_pe_eff``, ``eps_init``,
        ``sigma_p_target`` and ``beyond_linear`` flag.
    """

    state: SectionState
    grouted: List[int]
    dropped_actions: List[PrestressAction]
    residual_actions: List[PrestressAction]
    report: dict


# ===========================================================================
#  Internal helpers
# ===========================================================================

def _tendon_geometry(section) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``(x, y, Ap)`` arrays for the section's tendons [mm, mm, mm²]."""
    tendons = getattr(section, "tendons", [])
    if not tendons:
        raise ValueError(
            "solve_posttension_sequence: section carries no tendons. "
            "Build the section with a non-empty `tendons` list "
            "(the post-tension cables, declared bonded=True).")
    x = np.asarray(section.x_tendons, dtype=float)
    y = np.asarray(section.y_tendons, dtype=float)
    Ap = np.asarray(section.A_tendons, dtype=float)
    return x, y, Ap


def _resolve_Ep(section, Ep, nt) -> np.ndarray:
    r"""
    Resolve the per-tendon modulus array.

    Order of precedence: explicit *Ep* argument (scalar broadcast or
    array), else each tendon material's ``Ep`` / ``E`` attribute.  Raises
    if a modulus cannot be resolved (no silent default).
    """
    if Ep is not None:
        arr = np.broadcast_to(np.asarray(Ep, dtype=float), (nt,)).astype(float)
        return arr.copy()
    out = np.empty(nt, dtype=float)
    for j, t in enumerate(section.tendons):
        mat = t.material
        E = getattr(mat, "Ep", None)
        if E is None:
            E = getattr(mat, "E", None)
        if E is None:
            raise ValueError(
                f"solve_posttension_sequence: tendon {j} material exposes "
                f"neither `Ep` nor `E`; pass an explicit `Ep=` argument.")
        out[j] = float(E)
    return out


def _stressing_view(section, base_state=None, bonded=(), *,
                    eps_init=None, bulk_plane=None):
    r"""
    Build the section **view** that resists during stressing: bulk +
    passive rebars, plus the tendons named in *bonded* and no others.

    Under the deferred-single-grouting model no tendon is bonded during
    the sequence, so the resisting section is exactly the hardened
    concrete plus passive reinforcement: the tendons are loads, not
    elements, and must not contribute stiffness.  That is *bonded* left
    empty, and it is the historical behaviour of this function.

    From Phase 22 the caller may name tendons already **grouted at this
    step** of the sequence (``bonded_before`` on
    :func:`solve_posttension_sequence`).  Those do contribute stiffness:
    a cable in a filled duct is strain-compatible with the concrete
    around it, so it transforms the section the *later* jacks are
    stressed against, exactly as the ``bonded_before`` hook of
    :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`
    does in closed form.  Measured cost of ignoring it, on a 400x1200
    with three tendons :math:`A_p = 1500\,\mathrm{mm^2}` at
    :math:`y = -450/-400/-350` and :math:`E_c = 35\,220.462`: the
    elastic-shortening loss comes out **5.1 %** too large and the final
    :math:`\sigma_p` **0.46 %** too low (``22_0 §1``).

    **What a bonded tendon enters at is the caller's fact, and this
    function will not invent it.**  Left unspecified, *eps_init* is
    zero for every element, which is right for a view solved as an
    increment from zero strain.  From Phase 25 the driver solves each
    increment **against the state already accumulated**, and there the
    right value is not zero and is not one value:

    * a tendon grouted **during** the sequence must resist only what
      happens after it joins, so it enters at minus the concrete strain
      at the instant it joined;
    * a tendon grouted **before** the sequence carries a declared
      effective stress that no solve produced, so it enters at
      :math:`\sigma_{pe}/E_p`.  Not
      :math:`\sigma_{pe}/E_p - \varepsilon_{c,\text{declared}}`: a
      *bulk_plane* reaches the concrete a tendon displaces but never
      the strand's own law (:meth:`FiberSolver._tendon_offset`), so the
      declared plane must not be subtracted a second time.

    Both are directly computable from what the caller declared — no
    fixed point — and both are the caller's to state.  A *base_state*
    that carries an ``eps_init`` override for a tendon (the timeline
    sets one at its ``grout`` event) is still ignored: *bonded* and
    *eps_init* together state which tendons resist and at what, and a
    second source for the same fact could disagree with the first.

    *Which* concrete is hardened is the caller's fact, not this
    function's.  Called bare, every bulk zone of the section resists ---
    right for a stand-alone single-bulk sequence, and wrong on a
    timeline, where a zone cast after the stressing would be carrying
    load before it exists.  Measured on the flagship: 48 fibres of a
    topping cast 90 days after the grout were stiffening the section the
    tendon was stressed against.  *base_state* is how a caller that
    knows the instant says so; the tendon-masking rule stays here, in
    its one home, either way.

    Parameters
    ----------
    section : GenericSection
    base_state : SectionState, optional
        The state whose bulk activity, casting datums and bulk
        prestrain the view inherits.  Its **tendon** flags and tendon
        ``eps_init`` are ignored: which tendons resist at this instant
        is stated by *bonded* and by nothing else, which is the whole
        point of this view.
    bonded : sequence of int, optional
        Tendon-local indices already grouted at this step.  Default:
        none — the deferred-single-grouting model.
    eps_init : dict, optional
        ``{tendon-local index: initial strain}`` for tendons in
        *bonded*.  Absent entries are zero.  Naming a tendon that is
        not bonded is a caller error and is refused: an initial strain
        on an element that does not resist is a statement with no
        consequence, and silently dropping it would hide the mistake.
    bulk_plane : sequence of float, optional
        ``(eps0, chi_x, chi_y)`` imposed on **every** bulk zone, about
        the reference point the caller will give the solver.  This is
        the strain the concrete is *already in*, and it is a strain
        that carries stress — unlike the creep/shrinkage eigenstrain
        that :mod:`gensec.solver.losses` writes into the same array
        with the opposite sign, which is why that sign flip stays in
        its own module and is not repeated here.

        Refused when *base_state* already carries a non-zero
        ``bulk_planes``: adding a declared total strain on top of an
        accumulated eigenstrain would count the creep twice, and the
        two cannot be told apart once summed.

    Returns
    -------
    GenericSection
        The materialized resisting view.

    Raises
    ------
    ValueError
        *bonded* not a subset of the tendons; *eps_init* naming a
        tendon that is not bonded; a malformed *bulk_plane*; a
        *bulk_plane* on a *base_state* that already carries one.
    """
    n_reb = int(section.x_rebars.size)
    n_ten = int(getattr(section, "x_tendons", np.empty(0)).size)
    n = n_reb + n_ten
    bonded_set = {int(b) for b in bonded}
    if bonded_set and not bonded_set.issubset(range(n_ten)):
        raise ValueError(
            f"_stressing_view: bonded={sorted(bonded_set)} is not a "
            f"subset of range({n_ten}).")
    active = np.ones(n, dtype=bool)
    bonded_mask = np.ones(n, dtype=bool)
    if n_ten:
        active[n_reb:] = False          # tendons are loads while stressing
        bonded_mask[n_reb:] = False
        for b in sorted(bonded_set):
            active[n_reb + b] = True    # ... except the grouted ones
            bonded_mask[n_reb + b] = True
    if base_state is not None:
        # Passive elements keep the caller's activity -- a rebar inside
        # an uncast zone does not exist either.
        active[:n_reb] = np.asarray(base_state.active, dtype=bool)[:n_reb]
    eps0_elem = np.zeros(n, dtype=float)
    for _k, _v in (eps_init or {}).items():
        _k = int(_k)
        if _k not in bonded_set:
            raise ValueError(
                f"_stressing_view: eps_init names tendon {_k}, which is "
                f"not in bonded={sorted(bonded_set)}. An initial strain "
                f"on an element that does not resist has no consequence; "
                f"either bond it or drop the entry.")
        eps0_elem[n_reb + _k] = float(_v)
    extra = {}
    if base_state is not None:
        for attr in ("bulk_active", "bulk_planes"):
            value = getattr(base_state, attr, None)
            if value is not None:
                extra[attr] = np.array(value, copy=True)
    if bulk_plane is not None:
        plane = np.asarray(bulk_plane, dtype=float).ravel()
        if plane.size != 3 or not np.all(np.isfinite(plane)):
            raise ValueError(
                f"_stressing_view: bulk_plane must be three finite "
                f"floats (eps0, chi_x, chi_y), got {bulk_plane!r}.")
        held = extra.get("bulk_planes")
        if held is not None and np.any(held):
            raise ValueError(
                "_stressing_view: the base state already carries a "
                "non-zero bulk_planes, and a declared existing strain "
                "would be added on top of it. The two are different "
                "quantities with the same storage — an accumulated "
                "creep/shrinkage eigenstrain and the total strain a "
                "member is declared to be in — and their sum is not "
                "either of them. Declare the existing state on a "
                "section whose bulk carries no locked-in planes, or "
                "state the plane net of the eigenstrain.")
        # Zone count as StagedDomainManager sizes it, widened to any
        # zone index the section's own fibre map actually uses.
        n_zones = 1 + len(getattr(section, "bulk_materials", []) or [])
        for _attr in ("mat_indices", "mat_indices_rebar",
                      "mat_indices_tendon"):
            _mi = getattr(section, _attr, None)
            if _mi is not None and np.size(_mi):
                n_zones = max(n_zones,
                              int(np.max(np.asarray(_mi, dtype=int))) + 1)
        planes = (np.zeros((n_zones, 3), dtype=float) if held is None
                  else held)
        planes = planes + plane[None, :]
        extra["bulk_planes"] = planes
        if "bulk_active" not in extra:
            extra["bulk_active"] = np.ones(n_zones, dtype=bool)
    state = SectionState(
        stage_index=0,
        active=active,
        bonded=bonded_mask,
        eps_init=eps0_elem,
        bulk_eps_init=float(
            getattr(base_state, "bulk_eps_init", None)
            if base_state is not None
            else getattr(section, "bulk_eps_init", 0.0)),
        label=("stressing" if not bonded_set
               else f"stressing+bonded{sorted(bonded_set)}"),
        **extra,
    )
    return materialize_view(section, state)


def _grouted_state(section, grouted_local, eps_init_local):
    r"""
    Build the post-grouting :class:`SectionState`.

    Constructs the base state (bulk + rebars active/bonded, every
    tendon an inactive load) and hands the grouted tendons to the
    atomic primitive :meth:`SectionState.with_grouted`, which flips
    ``active``/``bonded`` and sets the reconciled ``eps_init``
    together and enforces the single-side / no-re-grouting invariant.

    Parameters
    ----------
    section : GenericSection
    grouted_local : sequence of int
        Tendon-local indices to grout.
    eps_init_local : sequence of float
        Reconciled ``eps_init`` per grouted tendon (aligned with
        *grouted_local*); see :func:`reconcile_grouting`.

    Returns
    -------
    SectionState
    """
    n_reb = int(section.x_rebars.size)
    n_ten = int(section.x_tendons.size)
    n = n_reb + n_ten
    base = SectionState(
        stage_index=0,
        active=np.concatenate([np.ones(n_reb, bool), np.zeros(n_ten, bool)]),
        bonded=np.concatenate([np.ones(n_reb, bool), np.zeros(n_ten, bool)]),
        eps_init=np.zeros(n, dtype=float),
        bulk_eps_init=float(getattr(section, "bulk_eps_init", 0.0)),
        label="grout",
    )
    union_idx = [n_reb + int(j) for j in grouted_local]
    eps_map = {ui: float(e) for ui, e in zip(union_idx, eps_init_local)}
    return base.with_grouted(union_idx, eps_map)


def _bonded_state(section, base_state, grouted_local, eps_init_local):
    r"""
    The state the section is in **just after** grouting.

    Derived from *base_state* when there is one, so that everything the
    timeline has already established -- which zones are cast, which
    elements exist, the bulk datums, the bulk pre-strain -- survives.
    Rebuilding it from scratch would silently resurrect elements
    belonging to zones that have not been cast yet, which
    ``materialize_view`` correctly refuses.

    Parameters
    ----------
    section : GenericSection
    base_state : SectionState or None
        ``None`` for the stand-alone driver, which owns no staging
        history; the single-zone construction of :func:`_grouted_state`
        is then the right one.
    grouted_local : sequence of int
    eps_init_local : sequence of float

    Returns
    -------
    SectionState
    """
    if base_state is None:
        return _grouted_state(section, grouted_local, eps_init_local)
    n_reb = int(section.x_rebars.size)
    union_idx = [n_reb + int(j) for j in grouted_local]
    eps_map = {ui: float(e) for ui, e in zip(union_idx, eps_init_local)}
    return base_state.with_grouted(union_idx, eps_map)


def grout_selfconsistent(section, result, indices, *, demand_after,
                         base_state=None, modmap=None,
                         readback_rel=1.0e-9, max_iter=8,
                         eps_c_linear_limit=None):
    r"""
    Grout, then **close the loop** on the tendon force.

    :func:`grout` forms the reconciled datum

    .. math::

        \varepsilon_{\text{init},j}
            = \varepsilon_{pe,\text{eff},j}
              - \varepsilon_{\text{sec,grout},j}

    where :math:`\varepsilon_{\text{sec,grout},j}` is the concrete
    strain at the tendon **before** grouting -- read on a section in
    which the tendon is an external load and its area is *not* in the
    stiffness.  One instant later the same tendon is a bonded element
    whose area *is* in the stiffness, and the equilibrium that the rest
    of the analysis reads is solved on that transformed section.  The
    two strains are not the same, so the tendon reads back a stress
    that is not the one it was jacked to.

    Nothing mechanical happens at grouting: the duct is filled, the
    force in the strand does not change.  A read-back that differs from
    the jack is therefore a modelling residue, not physics, and this
    routine removes it by solving for the datum that the *bonded*
    configuration reproduces:

    .. math::

        \varepsilon_{\text{init},j}^{(k+1)}
            = \varepsilon_{pe,\text{eff},j}
              - \varepsilon_{\text{sec},j}
                \bigl(\varepsilon_{\text{init}}^{(k)}\bigr),

    a contraction with ratio :math:`\approx n\rho/(1+n\rho)`, which for
    ordinary post-tensioning is a few percent -- two or three
    iterations reach machine precision.

    Measured before the fix, on a 400x800 with :math:`A_p = 1500`
    jacked to 1250 MPa: read-back 1249.2282, a residue of -0.772 MPa.
    The residue is **independent of the applied load** (it grows, very
    slightly, when the load is removed) and scales as :math:`A_p^2`,
    which is the signature of a second-order :math:`n\rho` term and
    was how the cause was identified: an earlier hypothesis blamed the
    pre-grout load and the measurement falsified it.

    Parameters
    ----------
    section : GenericSection
    result : PostTensionResult
        From :func:`solve_posttension_sequence`.
    indices : sequence of int
        Local indices of the tendons grouted now.
    demand_after : tuple of float
        ``(N, Mx, My)`` [N, N·mm] acting **after** the grout: the base
        demand plus whatever ungrouted tendons still apply as loads.
        Grouted tendons must not appear here -- that is the
        double-count :func:`grout` exists to prevent.
    base_state : SectionState or None, optional
        The state the section is in (cast zones, bulk datums).  Its
        bulk fields are carried into the grouted state.
    modmap : MaterialMap or None, optional
        Service moduli.  ``None`` solves on the design laws, which is
        the F5 defect and is accepted only for a caller that has no
        modulus map to give.
    readback_rel : float, optional
        Relative tolerance on
        :math:`\sigma_{p,\text{read}}/\sigma_{p,\text{jack}} - 1`.
        Default ``1e-9``.
    max_iter : int, optional
        Iteration cap.  Reaching it raises: a contraction that does not
        contract is a finding.
    eps_c_linear_limit : float or None, optional
        Forwarded to :func:`grout`.

    Returns
    -------
    tuple
        ``(GroutResult, eps_init_by_local, n_iter, residual_rel)``.

    Raises
    ------
    ValueError
        If the iteration does not converge within *max_iter*.

    See Also
    --------
    grout : the single-pass form this corrects.
    """
    gres = grout(section, result, indices=indices,
                 eps_c_linear_limit=eps_c_linear_limit)
    grouted = sorted({int(j) for j in indices})
    eps_pe = np.asarray(result.eps_pe_eff, dtype=float)
    eps_init = {int(rec["tendon"]): float(rec["eps_init"])
                for rec in gres.report["reconciliation"]}

    n_reb = int(section.x_rebars.size)
    x = np.asarray(section.x_tendons, dtype=float)
    y = np.asarray(section.y_tendons, dtype=float)
    N, Mx, My = (float(v) for v in demand_after)

    resid = float("inf")
    for k in range(1, max_iter + 1):
        st = _bonded_state(section, base_state, grouted,
                           [eps_init[j] for j in grouted])
        view = materialize_view(section, st)
        if modmap is not None:
            from .sls import sls_view
            view = sls_view(view, modmap)
        solver = FiberSolver(view)
        if abs(N) < 1e-30 and abs(Mx) < 1e-30 and abs(My) < 1e-30:
            eps0 = chi_x = chi_y = 0.0
        else:
            sol = solver.solve_equilibrium(N, Mx, My)
            eps0 = float(sol["eps0"])
            chi_x = float(sol["chi_x"])
            chi_y = float(sol["chi_y"])
        xr, yr = solver.x_ref, solver.y_ref
        resid = 0.0
        for j in grouted:
            eps_sec = eps0 + chi_x * (y[j] - yr) - chi_y * (x[j] - xr)
            target = float(eps_pe[j])
            new = target - eps_sec
            denom = max(abs(target), 1e-30)
            resid = max(resid, abs(new - eps_init[j]) / denom)
            eps_init[j] = new
        if resid < readback_rel:
            break
    else:
        raise ValueError(
            f"grout: the self-consistent datum did not converge in "
            f"{max_iter} iterations (last relative move {resid:.3e}).  "
            f"The iteration is a contraction of ratio ~n*rho for a "
            f"linear section; a failure here means the view is not "
            f"linear or the tendon dominates the stiffness.")

    for rec in gres.report["reconciliation"]:
        rec["eps_init"] = eps_init[int(rec["tendon"])]
    # The datum is no longer the single-pass one: overwrite the marker
    # ``grout`` left, and record *against what* the fixed point was
    # accepted.  A bare "converged" would be the absolute-flag defect
    # of 12_12 §2 rebuilt one layer up; the residue and the tolerance
    # it was judged by are both here so a reader can re-decide.
    gres.report["closure"] = {
        "kind": "selfconsistent",
        "iterations": int(k),
        "residual_rel": float(resid),
        "tolerance_rel": float(readback_rel),
    }
    return gres, eps_init, k, resid


# ===========================================================================
#  Driver
# ===========================================================================

def solve_posttension_sequence(
    section,
    *,
    sigma_p0: Sequence[float],
    Ep=None,
    order: Optional[Sequence[int]] = None,
    bonded_before: Optional[Dict[int, int]] = None,
    pending: Optional[Sequence[int]] = None,
    already_bonded: Optional[Union[Sequence[int],
                                   "PreexistingState"]] = None,
    applied_at: Optional[Mapping[int, Sequence[float]]] = None,
    base_N: float = 0.0,
    base_Mx: float = 0.0,
    base_My: float = 0.0,
    base_state=None,
    modmap=None,
    x_ref: Optional[float] = None,
    y_ref: Optional[float] = None,
    scheme: str = "one_pass",
    coupled_tol: float = 1e-12,
    coupled_max_iter: int = 50,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> PosttensionSequenceResult:
    r"""
    Sequential bonded post-tensioning on a single existing bulk.

    Orchestrates the stressing sequence in the fiber model: for each
    event it applies the jacked tendon's couple as a
    :class:`~gensec.solver.section_state.PrestressAction`, solves the
    section, reads the concrete strain increment at every
    already-anchored tendon, and debits

    .. math::

        \Delta\sigma_{p,j} = -E_{p,j}\,\Delta\varepsilon_{c,j}.

    The resisting section while stressing is the hardened bulk plus
    passive reinforcement; the tendons are loads, **not** elements, until
    grouting (see :func:`grout`).

    Parameters
    ----------
    section : GenericSection
        A meshed section whose ``tendons`` carry the post-tension cables
        (declared ``bonded=True``; their not-yet-grouted state is the
        per-stage mask handled here).
    sigma_p0 : sequence of float
        Jacking stress per tendon [MPa], **already net of friction and
        anchorage slip** (member-level effects, injected as input;
        out of scope for this sectional driver).  Aligned with
        ``section.tendons``.
    Ep : float or sequence of float, optional
        Tendon modulus [MPa].  Default: each tendon material's ``Ep``
        (or ``E``) attribute.
    order : sequence of int, optional
        Stressing order, as tendon indices.  Default: declaration order
        ``range(n_tendons)``.
    bonded_before : dict, optional
        ``{tendon index: step index}`` — the tendon counts as **bonded
        in the resisting section at and from that step** of *order*
        (0-based).  This is the staged-injection schedule: "stress
        tendon 0, inject it, then stress tendon 1 against a section
        that now contains tendon 0".  Exact parallel of the
        per-tendon ``bonded_before`` of
        :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`,
        deliberately the same name for the same fact.  A tendon must be
        stressed before it is grouted, so
        ``order.index(k) < bonded_before[k]``; the reverse is a
        :class:`ValueError`, not a modelling choice.  Default ``None``:
        no tendon is bonded during the sequence, which is the
        deferred-single-grouting model and the overwhelming majority of
        cast-in-place post-tensioned decks.
    pending : sequence of int, optional
        Tendon indices **not yet stressed** at any point of this
        sequence: neither loads nor resistance elements.  Required
        whenever *order* is a strict subset of the section's tendons —
        see :class:`IntraSequenceUndeclared` for why GenSec refuses to
        guess.  ``set(order)`` and ``set(pending)`` must be disjoint and
        together cover every tendon exactly once.
    applied_at : dict, optional
        ``{step: (N, Mx, My)}`` — a sollecitazione applied **before**
        jack ``step``, with ``step`` a 0-based index into *order*.
        ``{0: ...}`` acts before the first jack.

        Unlike *base_N* this is **not** debit-free: every tendon
        anchored at that instant is strain-compatible with the concrete
        and takes :math:`\Delta\sigma_p = -E_p\,\Delta\varepsilon_c`
        from it, exactly as it does from a jack.  That is the whole
        difference between the two parameters, and it is why they are
        two parameters and not one reinterpreted by context: a caller
        who wrote ``base_N`` for self-weight and later declared a
        retrofit would otherwise find their self-weight silently
        debiting a forty-year-old cable.

        Permitted **if and only if** every tendon anchored at that step
        is also bonded.  A tendon stressed and not yet grouted slides
        in its duct: its force change follows the mean elongation over
        the free length, a member-level integral a sectional model
        cannot evaluate, and the local value it *can* evaluate is
        unconservative for :math:`\sigma_p`.  Otherwise
        :class:`LoadBetweenJacksNotModelled` is raised naming the open
        tendons — and the remedy is actionable: grout them first with
        *bonded_before*, and the same load becomes modelled.

        ``{0: ...}`` without a declared retrofit is refused: at that
        instant nothing is anchored, so it is debit-free, so it is
        *base_N* under a second name.  One fact, one name.
    already_bonded : PreexistingState or sequence of int, optional
        The tendons already grouted **before this sequence begins** —
        the retrofit.  Modelled from Phase 24, and only when the state
        that comes with them is declared: pass a
        :class:`PreexistingState`, which carries the effective stress
        of every existing tendon and the strain plane of the existing
        concrete.  A bare sequence of indices names *which* tendons
        without saying what state they are in and raises
        :class:`PreexistingStateUndeclared`.

        A declared pre-existing tendon is a bonded element of the
        resisting section from the first jack onwards **and** an
        anchored tendon from the first jack onwards, so every new jack
        debits it by :math:`-E_p\,\Delta\varepsilon_c`.  Its
        ``sigma_p0`` entry must be ``nan``: it was jacked in a history
        this call did not see, and a finite value there would either
        duplicate or contradict the declared effective stress.

        This is *not* the staged-injection case, which is modelled
        separately: see *bonded_before*.
        ``6_8-WARNING_intra_sequence_bonded.md`` §2 distinguishes the
        two.
    base_state : SectionState, optional
        The construction state at this instant: which bulk zones exist,
        with which casting datums.  Default: every zone of the section
        resists, which is right for a stand-alone sequence on a single
        bulk and wrong inside a timeline.
    modmap : MaterialMap, optional
        Resolved service moduli.  When given, the resisting view is
        linearised at them
        (:func:`~gensec.solver.sls.sls_view`) instead of using the
        declared constitutive laws.  Required for a datum: see the
        note in the body.
    base_N, base_Mx, base_My : float, optional
        The sollecitazione **at the start of the sequence** — the state
        the section is already in when the first jack is pulled (e.g.
        self-weight) [N, N·mm, N·mm].  It is *not* "a load, applied at
        some point during the sequence": see the contract below.
        Applied once, **debit-free**: it contributes to
        ``eps_ref_grout`` but never to the loss (only the *shortening*
        from a stressing event is a loss; load-induced strain is not).
        GenSec takes sollecitazioni, never loads — pass the resultant.
        **Must be zero on a retrofit**: a declared
        :class:`PreexistingState` already states the whole strain state
        of the existing member as a plane, and a base sollecitazione
        would state part of it a second time.  The combination is
        refused rather than resolved by a rule nobody chose.
    x_ref, y_ref : float, optional
        Reference point for actions and strains [mm].  Default: section
        centroid.
    scheme : {"one_pass", "coupled"}, optional
        Elastic-shortening fidelity.  ``"one_pass"`` (default): each
        event debits once (EC2 §5.10.5.1 level).  ``"coupled"``:
        fixed-point iteration over ``sigma_eff = sigma_p0 - loss``,
        capturing the second-order feedback.
    coupled_tol, coupled_max_iter : float, int, optional
        Convergence control for ``scheme="coupled"`` (max change in
        per-tendon stress between sweeps [MPa]).
    tol, max_iter : float, int, optional
        Force tolerance [N] and iteration cap for each equilibrium solve.

    Returns
    -------
    PosttensionSequenceResult

    Raises
    ------
    ValueError
        If the section carries no tendons, if a tendon modulus cannot be
        resolved, or if ``scheme`` is unknown.

    Notes
    -----
    **The contract on** ``base_N`` / ``base_Mx`` / ``base_My``\\ **, and
    the scope boundary that follows from it.**  These three numbers are
    the sollecitazione the section already carries when the sequence
    *begins*.  They are applied once, before the first jack, and they are
    debit-free.  There is deliberately no argument for "a load that
    arrives at step *k* of the sequence", and the omission is a modelling
    limit, not an oversight:

    A tendon that has been jacked but not yet grouted is **not
    compatible in strain** with the section around it.  It slides in its
    duct, so a change in its force follows the change in *elongation*
    between the anchorages,

    .. math::

        \Delta\sigma_p \;=\; E_p\,\frac{1}{L}
            \int_{0}^{L}\Delta\varepsilon_c(x)\,\mathrm{d}x ,

    which is a quantity of the **member**.  This function solves one
    section: the only thing it can see is the local
    :math:`\Delta\varepsilon_c(x^{*})`.  At the critical section — the
    section one actually analyses — that local value is an *upper bound*
    on the mean along the member, so debiting it would overstate the loss
    and understate :math:`\sigma_p`.  That is the unconservative
    direction for the serviceability stress limits of EN 1992-1-1
    §5.10.3, i.e. wrong in the place where the number is used.

    The elastic-shortening debit **between successive jacks** is kept,
    and the asymmetry is the point: §5.10.5.1 prescribes that
    calculation, having already folded the member-level averaging into
    its :math:`j`-factor.  No clause does the same for an arbitrary
    external load acting mid-sequence, so there is nothing to stand
    behind an analogous debit and GenSec does not invent one.

    Enforcement lives where the interleaving is visible — a timeline, not
    a bare driver call — in
    :meth:`~gensec.solver.timeline.ConstructionTimeline._check_load_between_jacks`,
    which raises :class:`LoadBetweenJacksNotModelled` naming the tendons
    and the two jacking events.  Called directly, this function has no
    way to know that its ``base_*`` came from mid-sequence: passing one
    there is outside the contract, and the caller owns the consequence.

    **Precondition on a stressing stage (the check that stops).** A
    stressing event varies the prestress; the base sollecitazione is
    applied separately and once.  This driver therefore never mixes a
    prestress increment with an external-load increment within one
    event, so the read :math:`\Delta\varepsilon_{c,j}` is pure
    shortening.  The YAML stage layer (deferred — see the Step-C status
    note) must enforce the same rule: a stage that varies **both**
    prestress and external sollecitazione must raise rather than report a
    polluted loss.

    **Why incremental solves reproduce the closed form.** For a
    linear-elastic transfer section, applying tendon *k*'s action alone
    and reading its plane increment is, by superposition, identical to
    applying the cumulative action set and differencing against the
    previous step.  The incremental form is implemented because the loss
    is defined *per event* and it is the exact algebra of
    :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`.
    """
    x, y, Ap = _tendon_geometry(section)
    nt = x.size
    sp0 = np.asarray(sigma_p0, dtype=float)
    if sp0.shape != (nt,):
        raise ValueError(
            f"solve_posttension_sequence: sigma_p0 has shape {sp0.shape}, "
            f"expected ({nt},) — one jacking stress per tendon.")
    Epv = _resolve_Ep(section, Ep, nt)
    if order is None:
        order = list(range(nt))
    order = [int(o) for o in order]
    # --- ordering guards -------------------------------------------------
    order_set = set(order)
    if len(order) != len(order_set) or not order_set.issubset(range(nt)):
        # Genuine input error: duplicates or out-of-range indices.
        raise ValueError(
            "solve_posttension_sequence: `order` must list each tendon index "
            f"in range({nt}) exactly once; got {order}.")
    # --- the retrofit refusal, taken first --------------------------------
    # The section carries no signal distinguishing a tendon grouted years ago
    # from one about to be jacked (``Tendon.bonded`` is the intrinsic
    # property, True for both). ``already_bonded`` is the explicit input by
    # which a caller declares the pre-existing set, and it is refused for the
    # reasons the exception names -- none of which is the stiffness.  Taken
    # BEFORE the coverage check, so that a caller who HAS declared the
    # retrofit reads the refusal about the retrofit rather than one about a
    # declaration it did make.
    preexisting: Optional[PreexistingState] = None
    pre_set: set = set()
    if already_bonded is not None and (
            isinstance(already_bonded, PreexistingState)
            or len(already_bonded) > 0):
        if not isinstance(already_bonded, PreexistingState):
            # A bare index list: WHICH, without the state that goes
            # with it.  Refused by name, with both missing facts and
            # the remedy -- and with the boundary of what GenSec will
            # not derive, so a reader does not expect D1b-2 here.
            bad = sorted(set(int(i) for i in already_bonded))
            if not set(bad).issubset(range(nt)):
                raise ValueError(
                    f"solve_posttension_sequence: `already_bonded` indices "
                    f"{bad} are not a subset of range({nt}).")
            raise PreexistingStateUndeclared(bad, nt)
        preexisting = already_bonded
        pre_set = set(preexisting.tendons)
        if not pre_set.issubset(range(nt)):
            raise ValueError(
                f"solve_posttension_sequence: `already_bonded` declares "
                f"tendons {sorted(pre_set)}, which are not a subset of "
                f"range({nt}).")
        clash = sorted(pre_set & order_set)
        if clash:
            raise ValueError(
                f"solve_posttension_sequence: tendons {clash} are both "
                f"declared already grouted and listed in `order`. A tendon "
                f"was jacked before this sequence or it is jacked during "
                f"it; it cannot be both, and GenSec will not decide which "
                f"declaration to believe.")
        # The jacking-stress slot of an existing tendon must be empty.
        # Unlike a `pending` tendon -- which WILL be jacked, so a
        # forward-declared jacking stress there is legitimate -- an
        # existing tendon was already jacked, in a history this call
        # never saw.  A finite number in its slot is either a duplicate
        # of the declared effective stress or a contradiction of it,
        # and nothing in the input says which.
        contradicted = sorted(k for k in pre_set
                              if not math.isnan(float(sp0[k])))
        if contradicted:
            raise ValueError(
                f"solve_posttension_sequence: tendons {contradicted} are "
                f"declared already grouted, at the effective stresses "
                f"given in `already_bonded.sigma_pe`, and ALSO carry a "
                f"finite `sigma_p0` entry "
                f"({[float(sp0[k]) for k in contradicted]}). That slot is "
                f"the stress a jack of THIS sequence applies, and an "
                f"existing tendon has none. Pass float('nan') there: the "
                f"declared effective stress is the only baseline, and two "
                f"numbers for one fact is how a wrong one survives.")
        if (base_N, base_Mx, base_My) != (0.0, 0.0, 0.0):
            raise ValueError(
                f"solve_posttension_sequence: a retrofit was declared and "
                f"base_(N, Mx, My)=({base_N}, {base_Mx}, {base_My}) is not "
                f"zero. The existing state is declared ONCE, as the "
                f"concrete strain plane (`eps0`, `chi_x`, `chi_y`) of "
                f"`PreexistingState` -- which is the total state of the "
                f"member, existing prestress and permanent loads "
                f"included. A base sollecitazione would state part of it "
                f"a second time, and nothing would say whether the "
                f"existing tendons should be debited for it. Fold the "
                f"sollecitazione into the declared plane, or model the "
                f"load as a separate stage.")

    # --- every tendon accounted for, exactly once, explicitly -------------
    # Three declared roles and no fourth: stressed here (``order``), not yet
    # stressed (``pending``), already bonded (``already_bonded``, refused
    # above).  A tendon in none of them is an UNDECLARED tendon, and the two
    # meanings a strict subset can carry need opposite treatments -- so
    # GenSec asks instead of picking (``IntraSequenceUndeclared``).  Before
    # Phase 22 this branch assumed the second meaning and refused, which
    # rejected an ordinary staged-injection timeline with a message about a
    # stiffness that had nothing to do with it.
    pending_set = {int(p) for p in (pending or ())}
    if not pending_set.issubset(range(nt)):
        raise ValueError(
            f"solve_posttension_sequence: `pending` indices "
            f"{sorted(pending_set)} are not a subset of range({nt}).")
    both = sorted(pending_set & order_set)
    if both:
        raise ValueError(
            f"solve_posttension_sequence: tendons {both} are in both `order` "
            f"and `pending`. A tendon is stressed in this sequence or it is "
            f"not; it cannot be both.")
    both_pre = sorted(pre_set & pending_set)
    if both_pre:
        raise ValueError(
            f"solve_posttension_sequence: tendons {both_pre} are in both "
            f"`pending` and `already_bonded`. `pending` means NOT YET "
            f"stressed; `already_bonded` means stressed and grouted "
            f"before this sequence. One tendon, one role.")
    undeclared = sorted(set(range(nt)) - order_set - pending_set - pre_set)
    if undeclared:
        raise IntraSequenceUndeclared(undeclared, nt)

    # --- the grouting schedule inside the sequence ------------------------
    # ``bonded_before[k] = s`` means: from step ``s`` of ``order`` onwards,
    # tendon k is a bonded element of the resisting section.  Validated, not
    # trusted: a tendon cannot be grouted before it has been stressed, and a
    # step outside the sequence is a caller error rather than a no-op -- a
    # silently ignored schedule entry would publish the pre-Phase-22 numbers
    # under a Phase-22 declaration, which is the worst of the two.
    schedule: Dict[int, int] = {}
    for _key, _step in (bonded_before or {}).items():
        k, s = int(_key), int(_step)
        if k not in order_set:
            raise ValueError(
                f"solve_posttension_sequence: `bonded_before` names tendon "
                f"{k}, which is not in `order`. A tendon grouted during this "
                f"sequence must be stressed during it; one bonded before the "
                f"sequence starts is `already_bonded`.")
        if not 0 <= s <= len(order):
            raise ValueError(
                f"solve_posttension_sequence: `bonded_before[{k}]`={s} is "
                f"outside the sequence, which has {len(order)} step(s).")
        if s <= order.index(k):
            raise ValueError(
                f"solve_posttension_sequence: `bonded_before[{k}]`={s} would "
                f"grout tendon {k} at or before its own jack (step "
                f"{order.index(k)}). A tendon is stressed, then grouted.")
        schedule[k] = s

    def _build_view(bonded_now, eps_init_now=None, bulk_plane_now=None):
        r"""The resisting view for a given bonded set, linearised if asked."""
        v = _stressing_view(section, base_state, bonded=bonded_now,
                            eps_init=eps_init_now,
                            bulk_plane=bulk_plane_now)
        if modmap is not None:
            # A datum is a physical fact -- "this tendon is unstressed at
            # this instant" -- so the elastic shortening that produces it
            # must be computed on the section's real, service-level
            # response.  On a design law the same sequence reads a concrete
            # 30-41 % too soft, and the whole difference lands in the
            # locked-in strain as E_p (eps_sec,service - eps_sec,design)
            # (12_5 §7, measured at +7.08 MPa on the flagship).
            from .sls import sls_view
            v = sls_view(v, modmap)
        return v

    # The tendon-free view is built first and unconditionally, and it is the
    # ONLY view whose centroid is allowed to define the reference point.
    # Bonding a tendon moves the homogenised centroid, so a per-step solver
    # left to default its own reference would report each step's plane about
    # a DIFFERENT origin, and the accumulation ``eps_c += d_eps`` would be a
    # sum of quantities that are not the same quantity.  Pinning (xr, yr)
    # here is what makes the per-step views comparable; it is not an
    # optimisation.
    view = _build_view(())
    solver = FiberSolver(view, x_ref=x_ref, y_ref=y_ref)
    xr, yr = solver.x_ref, solver.y_ref
    ly = y - yr
    lx = x - xr

    # The solver cache lives inside ``_run_sequence`` from Phase 25: a view
    # now carries the ``eps_init`` of the tendons bonded at that step, and a
    # tendon injected mid-sequence enters at minus the concrete strain of the
    # instant it joined -- a quantity that changes between the sweeps of
    # ``scheme='coupled'``.  A cache that outlived the sweep would hand the
    # second sweep the first sweep's initial strains, silently.

    def _plane_at_tendons(eps0, chi_x, chi_y):
        r"""Section strain at every tendon for a solved plane [-].

        Off-view evaluation of the same formula as
        :meth:`FiberSolver._tendon_section_strain`:

        .. math::

            \varepsilon_{\text{sec},i} = \varepsilon_0
                + \chi_x\,(y_i - y_\text{ref})
                - \chi_y\,(x_i - x_\text{ref}).
        """
        return eps0 + chi_x * ly - chi_y * lx

    def _advance(on, acc, triple):
        r"""Apply one demand **increment** to an accumulated state.

        The Phase-25 form.  *acc* is the strain plane already reached,
        about the pinned reference; *triple* is the increment of
        demand; *on* is the solver of the configuration that resists
        from here on.  The target is the resultant that
        configuration already carries at *acc*, plus the increment:

        .. math::

            \mathbf{R}\bigl(\varepsilon_{k}\bigr)
              = \mathbf{R}\bigl(\varepsilon_{k-1}\bigr)
                + \Delta\mathbf{D}_{k},

        so the tangent is taken where the section actually is, and the
        answer is exact for any constitutive law.  On a linear law
        :math:`\mathbf{R}` is affine and this reduces identically to
        solving the increment alone from zero, which is what the form
        before Phase 25 did and why it was exact there and only there.

        Reading the base off *on* rather than off the previous step's
        solver is deliberate: when a tendon joins, the configuration
        changes at constant strain, and the increment must be measured
        against the configuration that resists it.

        Returns
        -------
        tuple
            ``(eps0, chi_x, chi_y, converged)``.
        """
        N, Mx, My = (float(triple[0]), float(triple[1]), float(triple[2]))
        if abs(N) < 1e-30 and abs(Mx) < 1e-30 and abs(My) < 1e-30:
            return acc[0], acc[1], acc[2], True
        held = on.integrate(acc[0], acc[1], acc[2])
        kw = {}
        if acc[0] or acc[1] or acc[2]:
            kw = dict(eps0_init=float(acc[0]), chi_x_init=float(acc[1]),
                      chi_y_init=float(acc[2]))
        sol = on.solve_equilibrium(float(held[0]) + N,
                                   float(held[1]) + Mx,
                                   float(held[2]) + My,
                                   tol=tol, max_iter=max_iter, **kw)
        return (sol["eps0"], sol["chi_x"], sol["chi_y"],
                bool(sol["converged"]))

    # The declared existing state, resolved once against the pinned
    # reference point.  ``pre_bonded`` is the set that is a resisting
    # element at EVERY step -- unlike a ``bonded_before`` entry, which
    # joins partway through -- and ``eps_c_init`` is the strain the
    # concrete is already in, which is a STATE and therefore the
    # starting value of the accumulation rather than a term inside each
    # increment (see PreexistingState.Notes).
    # ``applied_at``: resolved against `order` here, gated inside the
    # sweep where the bond state at each step is known.
    applied: Dict[int, Tuple[float, float, float]] = {}
    if applied_at:
        for key, triple in applied_at.items():
            step = int(key)
            if not 0 <= step < len(order):
                raise ValueError(
                    f"solve_posttension_sequence: `applied_at` step "
                    f"{step} is not a jack index of this sequence "
                    f"(0..{len(order) - 1}). A sollecitazione after the "
                    f"last jack is not interposed in the sequence and "
                    f"has no debit to contaminate: apply it downstream, "
                    f"on the finished member.")
            vals = tuple(float(v) for v in triple)
            if len(vals) != 3:
                raise ValueError(
                    f"solve_posttension_sequence: `applied_at[{step}]`="
                    f"{triple!r} is not an (N, Mx, My) triple.")
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(
                    f"solve_posttension_sequence: `applied_at[{step}]`="
                    f"{triple!r} is not finite.")
            if step == 0 and preexisting is None:
                raise ValueError(
                    f"solve_posttension_sequence: `applied_at[0]`="
                    f"{triple!r} on a section with no declared "
                    f"pre-existing tendons. Before the first jack "
                    f"nothing is anchored, so nothing can be debited, "
                    f"so this is `base_N`/`base_Mx`/`base_My` under a "
                    f"second name. Use those. `applied_at` earns its "
                    f"own name only where a tendon is already anchored "
                    f"and the sollecitazione therefore costs it stress.")
            applied[step] = vals

    pre_bonded = tuple(sorted(pre_set))
    # The declared plane re-expressed about the PINNED reference, which
    # is the point the views and the solvers use.  Read through
    # ``plane_at`` rather than re-derived here: the transport of a plane
    # from one reference to another is the declaration object's fact and
    # has one home.
    bulk_plane = None
    if preexisting is not None:
        bulk_plane = (
            float(np.asarray(preexisting.plane_at(
                np.array([xr]), np.array([yr])), dtype=float).ravel()[0]),
            float(preexisting.chi_x),
            float(preexisting.chi_y),
        )
    if preexisting is None:
        eps_c_init = np.zeros(nt)
    else:
        eps_c_init = np.asarray(
            preexisting.plane_at(x, y), dtype=float)
        # The declared effective stress becomes the baseline in the
        # working array, so that `loss = sigma_p0 - sigma_p_after`
        # stays one relation with one meaning across both cases: for a
        # tendon jacked here the baseline is its jacking stress, for an
        # existing one it is the stress it was already at.  Copied
        # first -- `np.asarray` may alias the caller's array, and a
        # driver that wrote into its own input would be a defect of a
        # kind no fingerprint would catch.
        sp0 = sp0.copy()
        for _t, _v in preexisting.sigma_pe.items():
            sp0[_t] = float(_v)

    def _run_sequence(sigma_eff):
        r"""
        One forward sweep in the **incremental-on-state** form.

        Returns
        -------
        tuple
            ``(debit, eps_c, stages, ok, declared_demand)``.
        """
        debit = np.zeros(nt)
        anchored = np.zeros(nt, dtype=bool)
        # A tendon grouted before the sequence began is strain-compatible
        # with the concrete before the first new jack, so it is debited
        # by every one of them.  A tendon jacked here becomes anchored
        # only after its own jack.
        for t in pre_bonded:
            anchored[t] = True
        eps_c = eps_c_init.copy()
        stages: List[PosttensionStageResult] = []
        ok = True

        # What every bonded tendon enters its view at.  Both values are
        # arithmetic on declared quantities -- there is no fixed point
        # here and no iteration.
        #
        #   pre-existing : eps_init = sigma_pe/Ep, NOT
        #       sigma_pe/Ep - eps_c,declared.  MEASURED, not assumed: the
        #       fibre kernel evaluates a tendon at ``e_sec + eps_init``
        #       with ``e_sec`` the SOLVED plane alone, and applies the
        #       bulk offset only to the concrete it displaces
        #       (``FiberSolver._tendon_offset``).  The declared plane
        #       therefore never reaches the strand's own law, and
        #       subtracting it here would leave the cable at
        #       ``sigma_pe - Ep eps_c,declared`` at the declared state.
        #       It costs nothing in the losses -- they are increments,
        #       and the constant cancels -- and 4.7 % of N and 7.9 % of
        #       Mx in ``declared_state_demand``, which is precisely the
        #       quantity nothing else would have caught.
        #       ``sp0`` and not ``sigma_eff``: the declaration is a fact
        #       about the past and does not move between the sweeps of
        #       ``scheme='coupled'``.
        #   injected here : filled in at its own step, below.
        eps_init_map: Dict[int, float] = {
            int(t): float(sp0[t] / Epv[t]) for t in pre_bonded}

        solvers: Dict[frozenset, FiberSolver] = {}

        def _solver_for(bonded_now):
            key = frozenset(int(b) for b in bonded_now)
            got = solvers.get(key)
            if got is None:
                got = FiberSolver(
                    _build_view(sorted(key),
                                {t: eps_init_map[t] for t in key},
                                bulk_plane),
                    x_ref=xr, y_ref=yr)
                solvers[key] = got
            return got

        # The accumulated SOLVED plane, on top of the declared one.  It
        # starts at zero, which -- with the declared plane in the view --
        # is the declared state itself and not the unstrained section.
        acc = (0.0, 0.0, 0.0)
        first = _solver_for(pre_bonded)
        held0 = first.integrate(0.0, 0.0, 0.0)
        declared_demand = (None if preexisting is None
                           else (float(held0[0]), float(held0[1]),
                                 float(held0[2])))

        # Base sollecitazione: applied once, debit-free.  Refused above
        # when a retrofit is declared, so on that path this increment is
        # the trivial zero one and the existing tendons take no debit.
        e0, cx, cy, c_ok = _advance(first, acc,
                                    (base_N, base_Mx, base_My))
        ok &= c_ok
        eps_c += (_plane_at_tendons(e0, cx, cy)
                  - _plane_at_tendons(*acc))
        acc = (e0, cx, cy)
        cum = np.array([base_N, base_Mx, base_My], dtype=float)

        for k, j in enumerate(order):
            if k in applied:
                # A sollecitazione interposed in the sequence.  Gated on
                # the BOND state, not on the timing: the refusal belongs
                # to a tendon that slides, and a tendon that is grouted
                # does not slide whenever it arrived.
                bonded_here = set(pre_bonded) | {
                    t for t, st in schedule.items() if st <= k}
                open_t = sorted(set(np.flatnonzero(anchored).tolist())
                                - bonded_here)
                if open_t:
                    raise LoadBetweenJacksNotModelled(
                        tendons=open_t,
                        load_index=k,
                        load_demand=applied[k],
                        prev_stress_index=(order[k - 1] if k else None),
                        next_stress_index=j,
                        combo_name=None)
                for t, st in schedule.items():
                    if st == k and t not in eps_init_map:
                        eps_init_map[t] = float(
                            -_plane_at_tendons(*acc)[t])
                aN, aMx, aMy = applied[k]
                sol_here = _solver_for(sorted(bonded_here))
                e0a, cxa, cya, a_ok = _advance(sol_here, acc,
                                               (aN, aMx, aMy))
                ok &= a_ok
                d_eps_a = (_plane_at_tendons(e0a, cxa, cya)
                           - _plane_at_tendons(*acc))
                acc = (e0a, cxa, cya)
                eps_c += d_eps_a
                # Every anchored tendon pays for it; by the gate above
                # every anchored tendon is bonded, so the local strain
                # is the right quantity for all of them.
                debit[anchored] += -Epv[anchored] * d_eps_a[anchored]
                cum += np.array([aN, aMx, aMy], dtype=float)

            # Which tendons are bonded when THIS jack is pulled.  A tendon
            # with ``bonded_before[t] = s`` is in the resisting section from
            # step s onwards; with an empty schedule the set is empty at
            # every step and ``_solver_for`` returns the tendon-free one.
            bonded_now = sorted(
                set(pre_bonded)
                | {t for t, s in schedule.items() if s <= k})
            # A tendon that joins at this step enters carrying nothing
            # new: its jacking force is already in the accumulated
            # demand, and it must resist only what happens after it
            # joined.  Minus the concrete strain of that instant is
            # exactly that, and it is read off the state, not solved for.
            for t, st in schedule.items():
                if st == k and t not in eps_init_map:
                    # The SOLVED plane at that tendon, not the total:
                    # the declared plane does not reach a strand's law,
                    # so subtracting it here would double-count it.
                    eps_init_map[t] = float(-_plane_at_tendons(*acc)[t])
            step_solver = _solver_for(bonded_now)
            P = float(sigma_eff[j] * Ap[j])
            act = PrestressAction.from_force(
                P, float(x[j]), float(y[j]), x_ref=xr, y_ref=yr,
                label=f"jack_tendon{j}", origin="posttension")
            e0, cx, cy, c_ok = _advance(step_solver, acc, act.triple())
            ok &= c_ok
            d_eps = (_plane_at_tendons(e0, cx, cy)
                     - _plane_at_tendons(*acc))
            acc = (e0, cx, cy)
            eps_c += d_eps

            # Debit every already-anchored tendon (never the one being
            # stressed: it is not strain-compatible at this instant).
            mask = anchored.copy()
            mask[j] = False
            debit[mask] += -Epv[mask] * d_eps[mask]

            anchored[j] = True
            cum += np.array(act.triple(), dtype=float)

            sigma_now = sigma_eff - debit
            stages.append(PosttensionStageResult(
                stage_index=k,
                stressed_tendon=j,
                action=act,
                cumulative_demand=(float(cum[0]), float(cum[1]),
                                   float(cum[2])),
                eps_c=eps_c.copy(),
                sigma_p=sigma_now.copy(),
                converged=c_ok,
                report={
                    "stressed_tendon": j,
                    "sigma_eff_applied": float(sigma_eff[j]),
                    "d_eps_c_at_tendons": d_eps.copy(),
                    # The resisting section this jack was pulled against,
                    # as an INTEGER count and as the set itself.  Recorded
                    # because a count has no precision to hide behind: the
                    # whole D1 repair moves the sixth decimal of a stress
                    # and nothing else, so a report that carried only
                    # stresses could not tell a per-step view from a
                    # constant one (20_1 §3, the same lesson as the axial
                    # ``reason``).
                    "bonded_at_jack": tuple(bonded_now),
                    "n_bonded_at_jack": len(bonded_now),
                    # Which of those were there BEFORE the sequence.
                    # Recorded separately from the schedule-driven set
                    # because the two have different provenance: one is
                    # declared by the caller, the other is derived from
                    # `bonded_before`, and a report that merged them
                    # could not tell a retrofit from a staged injection.
                    "preexisting": tuple(pre_bonded),
                    # The accumulated SOLVED plane after this jack,
                    # about the pinned reference.  It is what the next
                    # increment is measured from, so a sequence that
                    # lost it could not be audited step by step.
                    "plane_after": (float(e0), float(cx), float(cy)),
                },
            ))
        return debit, eps_c, stages, ok, declared_demand

    if scheme == "one_pass":
        debit, eps_c, stages, ok, declared_demand = _run_sequence(
            sp0.copy())
        coupled_iters = 1
    elif scheme == "coupled":
        sigma_eff = sp0.copy()
        prev = None
        coupled_iters = 0
        for _ in range(coupled_max_iter):
            coupled_iters += 1
            debit, eps_c, stages, ok, declared_demand = _run_sequence(
                sigma_eff)
            sigma_eff = sp0 - debit
            if prev is not None and np.max(np.abs(sigma_eff - prev)) < coupled_tol:
                break
            prev = sigma_eff.copy()
    else:
        raise ValueError(
            f"solve_posttension_sequence: unknown scheme {scheme!r}; "
            f"use 'one_pass' or 'coupled'.")

    sigma_after = sp0 - debit
    eps_pe_eff = sigma_after / Epv

    res = PosttensionSequenceResult(
        order=order,
        scheme=scheme,
        sigma_p0=sp0.copy(),
        sigma_p_after=sigma_after,
        loss=debit,
        eps_pe_eff=eps_pe_eff,
        eps_ref_grout=eps_c,
        Ep=Epv,
        Ap=Ap.copy(),
        x_ref=float(xr),
        y_ref=float(yr),
        stages=stages,
        converged=bool(ok),
        coupled_iterations=int(coupled_iters),
        preexisting=preexisting,
        increment_form="incremental_on_state",
        declared_state_demand=declared_demand,
    )
    res._x = x.copy()
    res._y = y.copy()
    return res


# ===========================================================================
#  Grouting (rigorous transition) — the single-side invariant, atomic
# ===========================================================================

def _effective_action(result, j: int) -> PrestressAction:
    r"""
    One tendon's life-long demand action at its post-loss stress.

    The per-tendon core of
    :meth:`PosttensionSequenceResult.effective_actions`, factored out
    so that :func:`grout` can form the actions it needs on a **retrofit**
    result, where the aggregate method refuses: a pre-existing tendon
    has no demand-side action at all, and returning a list with a hole
    in it would be worse than refusing.

    Parameters
    ----------
    result : PosttensionSequenceResult
    j : int
        Tendon index.

    Returns
    -------
    PrestressAction
    """
    P = float(result.sigma_p_after[j] * result.Ap[j])
    return PrestressAction.from_force(
        P, result._x[j], result._y[j], x_ref=result.x_ref,
        y_ref=result.y_ref, label=f"tendon{j}",
        origin="posttension_effective")


def grout(
    section,
    result: PosttensionSequenceResult,
    *,
    indices: Optional[Sequence[int]] = None,
    eps_c_linear_limit: Optional[float] = None,
) -> GroutResult:
    r"""
    Rigorous (bonded) grouting transition for selected tendons.

    Atomically (i) drops each grouted tendon's
    :class:`~gensec.solver.section_state.PrestressAction` from the demand
    and (ii) brings the tendon into the resistance domain via
    :meth:`SectionState.with_grouted` — ``bonded`` flipped ``True`` with
    a **reconciled** locked-in strain

    .. math::

        \varepsilon_{\text{init},j}
            = \varepsilon_{pe,\text{eff},j}
              - \varepsilon_{\text{sec,grout},j},

    so that the resistance evaluates
    :math:`\sigma_p(\varepsilon_{\text{sec,grout},j}
    + \varepsilon_{\text{init},j})
    = \sigma_p(\varepsilon_{pe,\text{eff},j})`, i.e. reproduces the
    effective post-loss stress at the grouting strain datum.  Here
    :math:`\varepsilon_{\text{sec,grout},j}` is ``eps_ref_grout[j]`` and
    :math:`\varepsilon_{pe,\text{eff},j}` is ``eps_pe_eff[j]``.

    The post-grout state's capacity hash differs from the pre-grout
    state, so a
    :class:`~gensec.solver.section_state.StagedDomainManager` recomputes
    the resistance domain automatically — there is no user flag.

    Parameters
    ----------
    section : GenericSection
        The same meshed section passed to
        :func:`solve_posttension_sequence`.
    result : PosttensionSequenceResult
        The sequence result whose effective prestrain is baked in.
    indices : sequence of int, optional
        Tendon indices to grout.  Default: **all** tendons (the usual
        "stress all, then grout all" lifecycle).
    eps_c_linear_limit : float, optional
        If given, the absolute concrete strain at a grouted tendon
        (:math:`|\varepsilon_{\text{sec,grout},j}|`) exceeding this value
        **raises**: the reconciliation premise (concrete on its linear
        branch at transfer) is violated and a silent reconciliation would
        be wrong.  If ``None``, the reconciliation proceeds and the
        maximum :math:`|\varepsilon_{\text{sec,grout}}|` is surfaced in
        the report (visible, not buried) for the engineer to judge.

    Returns
    -------
    GroutResult

    Raises
    ------
    ValueError
        If *indices* is not a subset of the tendon range, or if a grouted
        tendon's concrete strain exceeds *eps_c_linear_limit*.
    RuntimeError
        If the single-side invariant is violated (a defensive
        construction-time check; should be unreachable).

    See Also
    --------
    grout_selfconsistent : the fixed-point wrapper around this function.

    Notes
    -----
    **This function is a single pass, and that is a modelling choice, not
    a deprecation.**  It reconciles :math:`\\varepsilon_{init}` once,
    against the strain plane of the demand handed to it.  But grouting a
    tendon *changes the section*: the strand's area enters the bonded
    stiffness, so the plane the reconciliation was computed on is no
    longer the plane the grouted section produces under the same demand.
    The residue is second order — the :math:`n\\rho` term — and on the
    flagship it read back as **-7.08 MPa** until Phase 12 measured it.

    :func:`grout_selfconsistent` closes that loop by iterating to a
    fixed point, and it **calls this function as its first iterate**.
    Neither replaces the other: use this one when the section stiffness
    is not materially changed by the grouting (a small :math:`A_p`
    relative to the transformed area), and the self-consistent form when
    the read-back residue matters.  ``13_0`` makes the choice visible
    in the dump: a single-pass grout datum raises
    :class:`~gensec.output.timeline_dump.DumpContractViolation` rather
    than being recorded as if it were converged.

    The **conservative** treatment is simply *not calling this function*:
    ``result.effective_actions()`` then remain demand-side loads for
    life.  Mixing the two — grouting a tendon *and* keeping its load — is
    the §F double-count this routine exists to prevent.
    """
    nt = result.sigma_p_after.size
    if indices is None:
        indices = list(range(nt))
    indices = [int(i) for i in indices]
    if not set(indices).issubset(range(nt)):
        raise ValueError(
            f"grout: indices {indices} not a subset of range({nt}).")
    # A tendon grouted BEFORE the sequence began is bonded already and
    # stays bonded: it is added to the set that enters the resistance
    # domain, and removed from the set whose demand-side load is
    # dropped, because it never had one.  Without this, a grouting
    # transition rebuilt from "every tendon an inactive load" would
    # take a physically present cable out of the section, and the
    # single-side invariant below would then fail on a section that is
    # perfectly well posed.
    pre = tuple(getattr(result.preexisting, "tendons", ()) or ())
    grouted_here = sorted(set(indices) - set(pre))
    grouted = sorted(set(grouted_here) | set(pre))
    ungrouted = [j for j in range(nt) if j not in grouted]

    eps_ref = result.eps_ref_grout
    eps_pe = result.eps_pe_eff

    # Reconciliation + linear-branch guard.
    eps_init = []
    max_abs_eps = 0.0
    for j in grouted:
        sj = float(eps_ref[j])
        max_abs_eps = max(max_abs_eps, abs(sj))
        if eps_c_linear_limit is not None and abs(sj) > eps_c_linear_limit:
            raise ValueError(
                f"grout: tendon {j} concrete strain at grouting "
                f"{sj:.3e} exceeds the linear-branch limit "
                f"{eps_c_linear_limit:.3e}; reconciliation premise "
                f"(linear concrete at transfer) is violated.")
        eps_init.append(float(eps_pe[j]) - sj)

    state = _grouted_state(section, grouted, eps_init)

    # Demand-side bookkeeping: drop the loads of the tendons grouted
    # HERE, keep the ungrouted ones.  The pre-existing tendons appear in
    # neither list: their force was applied in a history this sequence
    # did not see and was never a demand of it.  The actions are formed
    # here rather than through ``effective_actions()``, which refuses on
    # a retrofit result for exactly that reason.
    dropped = [_effective_action(result, j) for j in grouted_here]
    residual = [_effective_action(result, j) for j in ungrouted]

    # --- single-side invariant, made explicit ---------------------------
    # Every tendon is in EXACTLY one camp.
    n_reb = int(section.x_rebars.size)
    bonded_local = {int(ui) - n_reb for ui in np.nonzero(state.bonded)[0]
                    if int(ui) >= n_reb}
    load_local = set(ungrouted)
    if not bonded_local.isdisjoint(load_local):
        raise RuntimeError(
            "grout: single-side invariant violated — a tendon is both in "
            "the resistance domain and a demand load (the §F double-count).")
    if bonded_local | load_local != set(range(nt)):
        raise RuntimeError(
            "grout: single-side invariant violated — a tendon is neither "
            "grouted nor a load (prestress lost).")

    report = {
        "grouted_tendons": grouted,
        "ungrouted_tendons": ungrouted,
        "reconciliation": [
            {
                "tendon": j,
                "eps_ref_grout": float(eps_ref[j]),
                "eps_pe_eff": float(eps_pe[j]),
                "eps_init": float(eps_pe[j]) - float(eps_ref[j]),
                "sigma_p_target": float(result.sigma_p_after[j]),
                "beyond_linear": (None if eps_c_linear_limit is None
                                  else abs(float(eps_ref[j]))
                                  > eps_c_linear_limit),
            }
            for j in grouted
        ],
        "max_abs_eps_sec_grout": max_abs_eps,
        "linear_limit": eps_c_linear_limit,
        # Which routine produced the datum above.  ``grout`` is a
        # primitive with no solver context: it forms eps_init against
        # the *unbonded* stiffness and stops, leaving the O(n*rho)
        # residue Phase 12 measured at -0.772 MPa.  Saying so in a
        # sentence would be a claim; saying so in a field is a
        # contract, and the machine-readable dump refuses a datum
        # carrying this value (13_0 §8.1).
        "closure": {"kind": "single_pass"},
    }

    return GroutResult(
        state=state,
        grouted=grouted,
        dropped_actions=dropped,
        residual_actions=residual,
        report=report,
    )
# GENSEC-COMPLETE phase=34 utc=2026-09-06T10:33:16Z sha256=3a319cbe0ffa97019c8c64204184d323d8861e4ae0ec10cae24b0850ba67ee39 body_lines=2495
