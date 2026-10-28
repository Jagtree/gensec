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
SLS stress verification on the evolving section (Phase 7).

Overview
--------
This module verifies serviceability stresses — decompression and
stress limits at transfer and in service — on a section that evolves
through construction stages (:class:`~gensec.solver.section_state.SectionState`),
per the Phase-7 decision sheet:

- **D1** the stress engine is the *fiber solver on a substituted
  view* (the *SLS view*: every material replaced by its service
  material, see :func:`sls_view` and D4), not a closed-form
  transformed-section path.  Every subtlety already encoded and
  validated in :class:`~gensec.solver.integrator.FiberSolver` —
  per-tendon prestrain offsets, embedded-area subtraction, bulk
  imposed strain, multi-material zones — is reused verbatim.  The
  transformed section (:func:`sls_transformed_properties`) is the
  reporting / validation layer, never the engine.
- **D2** stresses are accumulated **incrementally** over the stage
  history (staged-construction superposition, Ghali–Favre–Elbadry
  consistent): each demand increment is resisted by the stiffness of
  the section *as it is* when the increment is applied.  See
  *Incremental algebra* below.
- **D3** SLS moduli are **never derived silently**.  The engine
  consumes an explicit modulus map; normative, time-dependent values
  (:math:`E_{cm}(t)`, :math:`E_{c,\mathrm{eff}}`) are computed by the
  caller through the appropriate bridge and passed in as numbers
  (:func:`resolve_sls_moduli`).
- **D4** was the violation of the uncracked linear basis, reported
  as a flag while the SLS view was linear-elastic in tension.  Phase
  27 removed the mechanism rather than the symptom: the service law
  now enters the view, the fibre releases at
  :math:`f_{ct,\mathrm{eff}}`, and the solve returns the cracked
  equilibrium directly.  There is no basis left to violate, so there
  is no flag.
- **D5** was the API-only surface.  Since Phase 33 a combination
  reaches this engine from YAML through its ``sls`` block
  (:func:`gensec.io_yaml._parse_sls_block`, driven by
  :func:`gensec.solver.timeline_run.run_sls`); a combination without
  the block is reported as skipped, not silently omitted.
- **D6** limits and decompression geometry are normative-agnostic
  primitives (:class:`~gensec.materials.verification_limits.StressLimits`,
  cover distance ``c_dec``).  The coefficients come from the provider
  of the **declared** code
  (:func:`~gensec.materials.verification_limits.ec2_stress_limits`
  for EN 1992-1-1); no code is assumed when none is declared.

Incremental algebra (D2)
------------------------
Let :math:`\sigma(\mathcal{V}, F)` denote the fiber stress state from
the linear equilibrium solve on SLS view :math:`\mathcal{V}` under
cumulative demand :math:`F = (N, M_x, M_y)`.  With per-stage views
:math:`\mathcal{V}_k` (stage state + stage moduli) and cumulative
demands :math:`F_k`, the accumulated stress is

.. math::

    S_0 &= \sigma(\mathcal{V}_0, F_0), \\
    S_k &= S_{k-1}
        \;+\; \underbrace{\sigma(\mathcal{V}_k, F_k)
              - \sigma(\mathcal{V}_k, F_{k-1})}_{\text{demand term}}
        \;+\; \underbrace{\sigma(\mathcal{V}_k, F_{k-1})
              - \sigma(\mathcal{V}_k^{(k-1)}, F_{k-1})}_{\text{state
              term (loss stages only)}},

where :math:`\mathcal{V}_k^{(k-1)}` is the stage-\ *k* view carrying
the *previous* stage's initial strains (``eps_init`` /
``bulk_eps_init``) — same masks, same moduli.  Properties of this
scheme:

- the **demand term** applies each demand increment to the *current*
  stiffness (correct staged-construction attribution; per-element
  initial strains cancel in the difference, so prestress is never
  double-counted);
- the **state term** captures the elastic redistribution produced by
  an initial-strain change at constant demand (time-dependent losses
  via ``eps_override`` / ``bulk_eps``); it is evaluated only on
  eps-only transitions and vanishes otherwise;
- when the state and moduli never change, the telescoping sum
  collapses to the total solve
  :math:`S_k = \sigma(\mathcal{V}, F_k)` — the scheme is exactly
  backward-compatible with the single-hash case;
- **grouting is stress-neutral by construction**: the reconciled
  ``eps_init`` baked by
  :func:`~gensec.solver.posttension.reconcile_grouting` makes the
  entering tendon's plane read reproduce :math:`\sigma_{p,\mathrm{
  after}}` at the grouting demand, so an entering element is simply
  initialised from the stage's total read (see *Transition
  taxonomy*).

Transition taxonomy
-------------------
Between consecutive stages, comparing the resistance masks
(``active & bonded``) and the initial strains:

=====================  =============================================
transition              handling
=====================  =============================================
identical state         demand term only (telescopes to total solve)
eps-only change         demand term + state term (loss redistribution)
elements entering       demand term for persisting elements; entering
(grout / activation)    elements initialised from the stage's total
                        read :math:`\sigma(\mathcal{V}_k, F_k)` —
                        physical for grouted tendons (reconciliation),
                        and the engine's own strain-compatible-from-
                        entry convention for activated bars
elements leaving        **NotImplementedError** — the compensating-
                        release demand semantics at SLS is deferred
compound (entering +    **ValueError** — split into two stages
eps change on
persisting elements)
moduli change           allowed on any transition (aging modulus per
                        stage); it changes stiffness for *subsequent*
                        increments only and produces no spurious
                        stress at constant demand
=====================  =============================================

Concrete stress field and decompression
---------------------------------------
On a linear-elastic view, every per-stage stress increment in a bulk
zone of modulus :math:`E_z` is an **affine** function of the section
coordinates,

.. math::

    \Delta\sigma_z(x, y) = E_z\!\left[\Delta\varepsilon_0
        + \Delta\chi_x\,(y - y_{\mathrm{ref}})
        - \Delta\chi_y\,(x - x_{\mathrm{ref}})
        + \Delta\varepsilon_{b}\right],

so the *accumulated* field is itself affine per zone,
:math:`S_z(x, y) = c_{0,z} + c_{x,z}\,x + c_{y,z}\,y`, with
coefficients accumulated alongside the fiber arrays.  This gives the
decompression check in closed form: the most tensile point of the
disc of radius :math:`c_{\mathrm{dec}}` around a tendon lies along
the (constant) stress gradient :math:`\nabla S_z = (c_{x,z},
c_{y,z})`, so the probe point is

.. math::

    \mathbf{p} = (x_p, y_p)
        + c_{\mathrm{dec}}\,\frac{\nabla S_z}{\lVert\nabla S_z\rVert},

and the check is :math:`S_z(\mathbf{p}) \le 0` (compression negative).
The probe point is evaluated on the affine field even if it falls
outside the concrete outline (conservative).  With a uniform field
(:math:`\lVert\nabla S_z\rVert \approx 0`) the probe degenerates to
the tendon location itself.

Unbonded tendons
----------------
An active-but-unbonded tendon is not in the view (its force is a
demand-side :class:`~gensec.solver.section_state.PrestressAction`)
and its SLS stress is a **member-level** quantity — same caveat as
ULS.  The caller supplies it per stage (``"unbonded_sigma_p"``); the
engine checks it against ``sigma_p_max`` and tags the result
``provenance='member_level'``.  It is never computed from the plane.

Out of scope (deferred, explicit)
---------------------------------
Concrete stress redistribution by creep beyond what the losses model
routes through ``eps_init`` / ``bulk_eps`` (full AAEM on the stress
field); stages with leaving elements; per-zone transformed properties
for multi-zone bulk (mirrors the existing ``compute_ideal_properties``
restriction).  Crack width and tension stiffening are not computed:
the service law releases a fibre at :math:`f_{ct,\mathrm{eff}}` and
returns the state *at* the crack, which is a sectional quantity; the
member average between cracks is not (see
:mod:`gensec.materials.service_concrete`).
"""

import copy
import math
import dataclasses
from abc import ABC, abstractmethod
from dataclasses import field as _dc_field
from dataclasses import replace as _dc_replace

import numpy as np
from shapely.geometry import LineString, Point

from ..materials.elastic import LinearElastic
from ..materials.verification_limits import StressLimits
from .integrator import FiberSolver
from .section_state import SectionState, materialize_view


__all__ = [
    "DatumPolicy",
    "MaterialMap",
    "ModulusPolicy",
    "ModulusSlot",
    "SlsPolicy",
    "covered_zone_slots",
    "resolve_moduli",
    "resolve_sls_moduli",
    "sls_view",
    "sls_transformed_properties",
    "verify_sls_staged",
]


# Numerical tolerance on the decompression verdict [MPa]: a probe
# stress up to this magnitude on the tensile side still passes, to
# absorb solver round-off at the exact decompression boundary.
_DECOMPRESSION_TOL = 1.0e-9


#: Placeholder modulus [MPa] handed to a bulk zone that does **not
#: exist** at the instant being resolved -- not yet cast, or being cast
#: right now.  Its fibres are masked out of the stage view, so the value
#: cannot reach any result; the invariant is asserted by perturbing it
#: six orders of magnitude and checking the answer does not move.
#:
#: It exists only because :func:`resolve_moduli` enumerates every slot of
#: the *base* section, cast or not -- which is why it lives here, beside
#: the mechanism that requires it, and not in the callers.  Written into
#: the absent zone's **own** slot, it cannot reach a zone that does exist
#: (12_0d D-2).
_UNCAST_PLACEHOLDER_E: float = 1.0


# ==================================================================
#  Modulus resolution (D3: never silent -- 12_0d S1: slot-keyed)
# ==================================================================


@dataclasses.dataclass(frozen=True)
class ModulusSlot:
    r"""
    One place a modulus can be attached to.

    A slot is *where* a modulus lives, and that is never a material.
    Two bulk zones may share a single material **object** — declaring
    the same material name for the base bulk and for a
    ``material_zones`` entry hands ``io_yaml`` the identical instance
    to both — and still carry different moduli, because
    :math:`E_c(t)` is a property of the pair *(zone, instant)*:

    .. math::

        E_{c,z} = E_{c}\bigl(t - t_{\mathrm{cast},z}\bigr),

    where only :math:`t_{\mathrm{cast},z}` distinguishes a topping
    from the substrate it was cast on.  Keying a modulus map by
    material collapses that pair; keying it by slot cannot
    (12_0d D-1/D-2).

    Parameters
    ----------
    kind : {'zone', 'rebar', 'tendon'}
        Population the slot belongs to.
    index : int
        Zone index for ``kind='zone'`` (``0`` is the base bulk);
        **union** index for the two element kinds — the canonical
        ``rebars + tendons`` order, which is the index every
        :class:`~gensec.solver.section_state.SectionState` array is
        already defined over.
    local_index : int
        Position within the slot's own population: the zone index,
        the rebar index, or the tendon index.  Equal to ``index`` for
        zones and rebars, ``index - n_rebars`` for tendons.
    name : str
        Declared name of the slot when it has one
        (:attr:`~gensec.geometry.fiber.RebarLayer.name`,
        :attr:`~gensec.geometry.fiber.Tendon.name`, or the entry of
        ``section.zone_names``), otherwise a synthetic label
        (``'rebar_3'``, ``'tendon_1'``).  Used in messages and as the
        map's label; a *synthetic* label is never accepted as an
        override key.
    material : Material
        Constitutive law attached to the slot.  Carried so that a
        policy can read intrinsic properties (``Es``, ``Ep``) and so
        that the material-name sugar can be expanded, never so that
        it can be used as a key.
    declared : bool
        Whether :attr:`name` was declared by the engineer.  Only a
        declared name is accepted as an override reference — same
        rule and same scope as :func:`gensec.io_yaml._union_name_map`,
        the project's one element-reference mechanism.  A synthetic
        label is for reading, not for addressing.
    """

    kind: str
    index: int
    local_index: int
    name: str
    material: object
    declared: bool = True


class ModulusPolicy(ABC):
    r"""
    Normative content of a modulus resolution.

    :func:`resolve_moduli` owns the *mechanism* — slot enumeration,
    override expansion, dead-override detection, validation, and the
    shape of the result.  A policy owns the *content*: what a slot
    with no explicit override is worth.  Splitting them this way is
    what keeps a second resolver from being written for the datum
    (12_0d §6/S1): a new normative rule is a new policy, not a new
    resolver, and a user supplying an ACI or AASHTO policy needs no
    library change.

    A policy that needs more than the section — a construction state,
    per-zone ages, rheological providers — binds them at
    **construction time**, not through the resolver signature.  The
    resolver stays ``(section, policy, explicit)`` for every policy
    that will ever exist, and the bindings a given normative rule
    needs are declared where that rule lives.

    Both hooks return ``None`` to mean *"I do not supply this slot"*,
    which is not an error: an explicit override may still cover it,
    and only a slot that reaches the end of the resolution order with
    no value raises.

    See Also
    --------
    SlsPolicy : the Phase-7 D3 policy, shipped here.
    """

    def explain(self, slot):
        r"""
        Optional note appended when *slot* cannot be resolved.

        A policy that knows *why* a slot has no value can say so here.
        :func:`resolve_moduli` appends the note to the unresolved-slot
        message; returning ``None`` (the default) adds nothing.

        The note must keep two things apart that are easy to conflate:
        what is **mechanically true** under any code, and what is a
        **GenSec architectural rule**.  A message that presents D3 --
        never derive a normative modulus silently -- as though EN
        1992-1-1 required it would be attributing GenSec's own choice
        to a document that does not make it, which is exactly the leak
        the normative-agnostic design exists to prevent.

        Parameters
        ----------
        slot : ModulusSlot
            The slot that could not be resolved.

        Returns
        -------
        str or None

        See Also
        --------
        resolve_moduli : appends the note to its error message.
        """
        return None

    @abstractmethod
    def bulk_material(self, slot):
        r"""
        Service material of a bulk-zone slot, or ``None``.

        Returning a **material** rather than a modulus is the Phase-27
        change, and the reason is physical: a non-linear law ages in
        more than one constant.  Between three and twenty-eight days a
        Eurocode concrete moves its stiffness, its mean strength, its
        strain at peak and its tensile strength, all independently; a
        single float can carry the first and silently drops the rest,
        and what is dropped is exactly what decides whether a fibre has
        cracked.

        A number is still accepted from a caller and means
        :class:`~gensec.materials.elastic.LinearElastic` at that
        modulus -- declaring a modulus and nothing else *is* declaring
        a straight line.  What no longer happens is the reverse: a
        declared material replaced by a line the model never wrote
        down.

        Parameters
        ----------
        slot : ModulusSlot
            A slot with ``kind == 'zone'``.

        Returns
        -------
        Material or float or None
        """

    @abstractmethod
    def elem_material(self, slot):
        r"""
        Service material of a union-element slot, or ``None``.

        Parameters
        ----------
        slot : ModulusSlot
            A slot with ``kind`` in ``{'rebar', 'tendon'}``.

        Returns
        -------
        Material or float or None
        """


def _intrinsic_modulus(material):
    r"""
    The material's own elastic modulus, when it *is* an intrinsic
    constant of the law object.

    ``Es`` (reinforcing steel) and ``Ep`` (prestressing steel) are
    declared constants of the constitutive law and carry no time or
    normative choice.  A concrete SLS modulus does — :math:`E_{cm}`,
    :math:`E_{cm}(t)` at transfer, or the effective modulus

    .. math::

        E_{c,\mathrm{eff}} = \frac{E_{cm}}{1 + \varphi\,\psi}

    under quasi-permanent load — so it is never read from the object
    (Phase-7 D3).

    Parameters
    ----------
    material : Material

    Returns
    -------
    float or None
        ``None`` when the material declares neither ``Es`` nor ``Ep``.
    """
    for attr in ("Es", "Ep"):
        value = getattr(material, attr, None)
        if value is not None:
            return float(value)
    return None


class SlsPolicy(ModulusPolicy):
    r"""
    Phase-7 D3 policy: intrinsic steel moduli, everything else
    explicit.

    Resolution supplied by this policy, after the explicit overrides
    :func:`resolve_moduli` applies first:

    1. the slot material's intrinsic ``Es`` / ``Ep``;
    2. nothing — concrete, tabulated laws and unknown materials
       return ``None`` and therefore raise, listing the slots.

    That second line is the D3 rule itself: the engine never reaches
    into a normative bridge on its own.  The caller computes
    :math:`E_{cm}(t)` through the appropriate bridge — e.g.
    ``concrete.ec2.ecm`` from
    :class:`~gensec.materials.ec2_properties.fben2` with its ``time``
    argument — and passes it in as a number.
    """

    def bulk_material(self, slot):
        r"""
        The slot's **own** material, when it is already a service law.

        A material that declares an intrinsic modulus is one whose
        stiffness is a constant of its constitutive law and carries no
        time and no partial factor -- a declared straight line is the
        example.  Such a material is returned unchanged: it needs no
        substitution and never did.

        Anything else -- a design envelope, a tabulated law -- returns
        ``None`` and :func:`resolve_moduli` refuses, which is Phase-7
        D3 unchanged: the engine never reaches into a normative bridge
        on its own.  What changed in Phase 27 is only what the caller
        supplies in answer: a **material at a declared age**, or a
        number, which declares a line.

        Parameters
        ----------
        slot : ModulusSlot

        Returns
        -------
        Material or None
        """
        if _intrinsic_modulus(slot.material) is None:
            return None
        return slot.material

    def elem_material(self, slot):
        r"""
        The element's **own** material, when it declares its stiffness.

        Reinforcement and strand are returned unchanged rather than
        replaced by a line at ``Es`` / ``Ep``.  Below yield the two
        answer identically, so nothing moves where the substitution was
        harmless; above it they do not, and a bar that yields in service
        is an event a verification should see rather than one the view
        smooths away.

        Parameters
        ----------
        slot : ModulusSlot

        Returns
        -------
        Material or None
        """
        if _intrinsic_modulus(slot.material) is None:
            return None
        return slot.material


class DatumPolicy(ModulusPolicy):
    r"""
    Service modulus of every zone at one instant of the timeline.

    The Phase-12 policy: the content of the resolution that draws a
    **datum** -- the plane a zone calls its own zero at the instant it
    is cast, and the plane a tendon calls its own zero at the instant
    it is grouted.

    A datum is a physical fact, *"this element is unstressed at this
    instant"*, so it must be drawn on the section's real response at
    that instant.  That response is elastic and service-level.  It is
    **not** the design law: a parabola-rectangle at
    :math:`\gamma_c = 1.5` has an initial tangent

    .. math::

        E_{\mathrm{tan}} = \frac{n\,f_{cd}}{\varepsilon_{c2}}

    which for C45 is about :math:`25\,500` MPa against
    :math:`E_{cm}(28\,\mathrm{d}) \simeq 36\,283` MPa -- a concrete
    some 30 % too soft, and 41 % too soft for a four-day-old topping.
    Drawing a datum on it was measured as the whole of the F4/F5
    residue (12_5 §7).

    Per zone, in order:

    1. the zone **does not exist** at this instant -- not yet cast, or
       being cast right now: :attr:`placeholder`, written into that
       zone's own slot, where it cannot reach a zone that does exist;
    2. the zone declares a **rheological provider**:
       :math:`E_{c,z}(t_z)`, the very modulus the creep eigenstrain
       :math:`\varphi\,\sigma_c / E_c` is later written against, so
       the datum and its read-back are self-consistent by
       construction;
    3. the model declares **no time axis at all** and a
       :attr:`reference` rule is supplied: the code's conventional
       modulus at its reference age -- :math:`E_{cm}(28\,\mathrm{d})`
       under EN 1992-1-1, another number under another code.  This
       branch is reached only when *no* zone of the model declares a
       provider: there is then no age to get wrong, because the
       analysis names one instant.  A model that *has* a time axis and
       a zone that does not declare its part of it is an incomplete
       declaration, and falls to case 4 (finding F-A);
    4. otherwise ``None`` -- *"I do not supply this slot"*.

    Case 4 is not a refusal: an explicit override may still cover the
    slot, and :func:`resolve_moduli` alone decides.  Writing a
    ``raise`` here would put the same message in two houses and make
    the explicit override unreachable, which is the failure mode
    :meth:`explain` exists to describe rather than cause.

    Parameters
    ----------
    state : SectionState
        The state the datum is drawn at.  ``bulk_active`` says which
        zones exist.
    providers : dict
        ``{zone index: RheologicalModel}``, already bound to the
        zone's drying geometry (``with_geometry(A_c, u)``) by the
        caller -- the only party that knows it.
    ages : dict
        ``{zone index: age of the zone [days]}`` at this instant.
    placeholder : float, optional
        Modulus of a zone that does not exist here.  Defaults to
        :data:`_UNCAST_PLACEHOLDER_E`.
    casting_zone : int or None, optional
        The zone being cast, when the datum is a casting datum.  It is
        held absent for its own solve: a zone cannot restrain the
        substrate it is about to be poured onto.
    reference : ReferenceModulusRule or None, optional
        The code's conventional modulus rule, consulted only for a
        zone with no provider.  The **caller** decides whether to pass
        one: it is legitimate exactly when the model declares no time
        axis, and the policy is not the party that can tell.  ``None``
        (the default) means the policy declines instead, which is what
        every time-dependent analysis wants.

    Raises
    ------
    ValueError
        A zone has a provider but no age.  :math:`E_c(t)` is a
        function of age; a provider without one cannot be evaluated,
        and assuming 28 days at a 3-day transfer is exactly the silent
        normative choice this policy exists to remove.

    See Also
    --------
    resolve_moduli : the single resolver this supplies.
    SlsPolicy : the Phase-7 D3 policy for the verification domain.

    Examples
    --------
    >>> modmap = resolve_moduli(
    ...     section,
    ...     DatumPolicy(state=state, providers={0: prov},
    ...                 ages={0: 28.0}, casting_zone=1),
    ...     explicit=dict(moduli or {}))
    """

    def __init__(self, *, state, providers, ages,
                 placeholder=_UNCAST_PLACEHOLDER_E, casting_zone=None,
                 reference=None):
        self.state = state
        self.providers = dict(providers or {})
        self.ages = dict(ages or {})
        self.placeholder = float(placeholder)
        self.casting_zone = (None if casting_zone is None
                             else int(casting_zone))
        self.reference = reference

    def bulk_material(self, slot):
        r"""
        Service **material** of a bulk-zone slot at this instant, or
        ``None``.

        Parameters
        ----------
        slot : ModulusSlot
            A slot with ``kind == 'zone'``.

        Returns
        -------
        float or None

        Raises
        ------
        ValueError
            The zone has a provider but no declared age.
        """
        z = int(slot.index)
        active = np.asarray(self.state.bulk_active, dtype=bool)
        if z == self.casting_zone or z >= active.size or not bool(active[z]):
            # The placeholder is a number, and a number declares a line.
            # It stands for a zone that is not there yet, so what it
            # answers a strain with is irrelevant as long as it is
            # negligible and explicit.
            return self.placeholder
        prov = self.providers.get(z)
        if prov is None:
            # No provider.  Either the model has no time axis -- and
            # the caller says so by supplying a reference rule -- or it
            # has one and this zone is missing from it, which is not a
            # material question but an incomplete declaration.
            if self.reference is not None:
                return self.reference.E_c(slot.material)
            return None
        t = self.ages.get(z)
        if t is None:
            raise ValueError(
                f"datum: zone {z} ('{slot.name}') declares a rheological "
                f"model but the timeline gives it no age at this instant. "
                f"The service law is a function of age -- a provider "
                f"without one cannot be evaluated, and assuming 28 days "
                f"at a 3-day transfer is the silent normative choice this "
                f"resolution exists to remove."
            )
        # The provider's own code writes this law.  A Eurocode provider
        # returns EN 1992-1-1 (3.14) with f_ctm at the age; an ACI
        # provider would return ACI's, and a compliance table refuses
        # because it nominates no strength.  Rheology is normative, and
        # so is the law it hands back.
        return prov.service_law(float(t))

    def elem_material(self, slot):
        r"""
        The element's own material, or ``None``.

        ``Es`` and ``Ep`` are declared constants of the constitutive
        law and carry no time and no normative choice, so an element
        that declares one is already service-ready and is handed back
        unchanged; anything else declines.

        Parameters
        ----------
        slot : ModulusSlot

        Returns
        -------
        Material or None
        """
        if _intrinsic_modulus(slot.material) is None:
            return None
        return slot.material

    def explain(self, slot):
        r"""
        Why a zone with no rheological model still needs a modulus.

        Appended by :func:`resolve_moduli` to the unresolved-slot
        message.  The note keeps two things apart that are easy to
        conflate: the **mechanics**, which hold under any code, and
        the **GenSec rule**, which is this project's own and is not a
        requirement of EN 1992-1-1 or of any other document.

        Parameters
        ----------
        slot : ModulusSlot

        Returns
        -------
        str or None
            The note for a bulk slot; ``None`` for an element slot,
            whose failure has a different cause.
        """
        if slot.kind != "zone":
            return None
        return (
            "The zone has no declared rheological model.  It does not "
            "creep -- but it keeps its stiffness, and therefore restrains "
            "the zones that do, which is frequently the governing effect. "
            "That is mechanics, not code: it holds under EN 1992-1-1, NTC "
            "2018, ACI 209 or any other. That GenSec refuses instead of "
            "deriving a modulus is GenSec's own rule (D3, normative "
            "agnosticism), not a requirement of EN 1992-1-1.  Declare a "
            "loss model for the zone, or pass its modulus in 'moduli'."
        )


@dataclasses.dataclass(frozen=True, eq=False)
class MaterialMap:
    r"""
    Resolved moduli of a section, indexed like the section's state.

    Two arrays that mirror
    :class:`~gensec.solver.section_state.SectionState`'s own
    indexing — zone-indexed like ``bulk_active`` / ``bulk_planes``,
    union-indexed like ``active`` / ``bonded`` / ``eps_init``.  That
    the modulus map takes the same shape as the state is the argument
    for this design: the project already indexes every other per-zone
    and per-element quantity that way, and an index cannot collapse
    two zones the way a shared material object can.

    Parameters
    ----------
    bulk : sequence of Material
        One service material per bulk zone, stored as a tuple.
        ``bulk[0]`` is the base bulk (``section.bulk_material``);
        ``bulk[z]`` for ``z >= 1`` is ``section.bulk_materials[z - 1]``.
        Since Phase 27 an entry is a material, not a number: a bare
        modulus is accepted upstream and means
        :class:`~gensec.materials.elastic.LinearElastic` at that value.
    elem : sequence of Material
        One material per point element, in the canonical
        ``rebars + tendons`` order.
    labels_bulk : tuple of str
        Zone names, index-aligned with *bulk*.
    labels_elem : tuple of str
        Element names, index-aligned with *elem*.

    Attributes
    ----------
    E_bulk, E_elem : numpy.ndarray
        Read-only, derived from each material's
        ``reference_modulus()`` [MPa]: the number a homogenisation
        ratio is written on (the secant :math:`E_{cm}` for
        :class:`~gensec.materials.service_concrete.ServiceConcrete`,
        not its initial tangent).

    Raises
    ------
    ValueError
        If an entry is not a material, if a label tuple does not match
        its sequence, if the bulk axis is empty, or if any reference
        modulus is not finite and strictly positive.  These run
        **always** (rule 1), not under
        ``assert``: a shape check that ``python -O`` removes is not a
        check.  The shape check is what makes 12_0d's D-1 (zone /
        material axis collapse) *unrepresentable* rather than merely
        fixed — a map with one entry for two zones cannot be built.

    Notes
    -----
    The container is frozen **and** its arrays are made read-only, so
    "frozen" is not a promise the object breaks through a field it
    happens to hold by reference.

    Equality is identity (``eq=False``).  Generated field-by-field
    equality would compare arrays and raise on the ambiguous truth
    value of the result; a modulus map is a computed artefact that is
    consumed positionally and never compared.

    Examples
    --------
    >>> mm = resolve_moduli(section, SlsPolicy(),
    ...                     explicit={("zone", "topping"): 22000.0,
    ...                               ("zone", 0): 36283.0})
    >>> float(mm.bulk[0]), float(mm.bulk[1])
    (36283.0, 22000.0)
    """

    bulk: tuple
    elem: tuple
    labels_bulk: tuple
    labels_elem: tuple
    E_bulk: np.ndarray = _dc_field(init=False, default=None)
    E_elem: np.ndarray = _dc_field(init=False, default=None)

    def __post_init__(self):
        r"""
        Coerce, freeze and validate both axes (always-on).

        The materials are stored as tuples, which are immutable without
        further ceremony; the derived modulus arrays are made read-only
        for the same reason the material tuples are frozen -- "frozen"
        must not be a promise the object breaks through a field it
        happens to hold by reference.
        """
        for field in ("bulk", "elem"):
            seq = tuple(getattr(self, field) or ())
            for pos, mat in enumerate(seq):
                if not (hasattr(mat, "stress_array")
                        and hasattr(mat, "tangent_array")):
                    raise ValueError(
                        f"MaterialMap.{field}[{pos}] is not a material: "
                        f"got {type(mat).__name__}.  Since Phase 27 the "
                        f"map carries the material a zone answers a "
                        f"strain with, not a single number -- a "
                        f"non-linear law ages in a stiffness, a strength "
                        f"and a strain at peak, and a scalar can carry "
                        f"only the first.  A caller that means a "
                        f"straight line declares LinearElastic(E)."
                    )
            object.__setattr__(self, field, seq)

        for field, labels in (("bulk", self.labels_bulk),
                              ("elem", self.labels_elem)):
            lab = tuple(str(x) for x in (labels or ()))
            seq = getattr(self, field)
            if len(lab) != len(seq):
                raise ValueError(
                    f"MaterialMap: {len(seq)} material(s) in '{field}' "
                    f"but {len(lab)} label(s).  Labels are index-aligned "
                    f"with the values; a mismatch means the map was "
                    f"assembled against a different section."
                )
            object.__setattr__(self, f"labels_{field}", lab)

        if len(self.bulk) < 1:
            raise ValueError(
                "MaterialMap.bulk is empty: every section has at least "
                "the base bulk zone (index 0)."
            )

        # The reference modulus is *derived*, never declared twice.  It
        # is what a homogenisation ratio and a creep basis are written
        # on, and EN 1992-1-1 writes both on the secant -- which is why
        # ServiceConcrete.reference_modulus returns E_cm and not its own
        # initial tangent.  Consumers that need a number read here;
        # consumers that need behaviour read the material.
        for field in ("bulk", "elem"):
            seq = getattr(self, field)
            vals = np.empty(len(seq), dtype=float)
            for pos, mat in enumerate(seq):
                try:
                    vals[pos] = float(mat.reference_modulus().value)
                except Exception as exc:        # noqa: BLE001
                    labels = getattr(self, f"labels_{field}")
                    raise ValueError(
                        f"MaterialMap: {field}[{pos}] "
                        f"('{labels[pos]}') declares no reference "
                        f"modulus ({type(exc).__name__}: {exc}).  A "
                        f"material that cannot say what modulus a "
                        f"homogenisation should be written on cannot "
                        f"enter a map that is read that way."
                    ) from exc
            bad = np.nonzero(~(np.isfinite(vals) & (vals > 0.0)))[0]
            if bad.size:
                labels = getattr(self, f"labels_{field}")
                shown = ", ".join(
                    f"{field}[{int(i)}] ('{labels[int(i)]}') = "
                    f"{float(vals[int(i)])!r}" for i in bad[:6]
                )
                raise ValueError(
                    f"MaterialMap: every reference modulus must be "
                    f"finite and strictly positive; {bad.size} entr"
                    f"{'y' if bad.size == 1 else 'ies'} violate(s) it: "
                    f"{shown}."
                )
            vals.setflags(write=False)
            object.__setattr__(self, f"E_{field}", vals)

    @property
    def n_zones(self):
        r"""Number of bulk zones covered by the map (``int``)."""
        return int(len(self.bulk))

    @property
    def n_union(self):
        r"""Number of union elements covered by the map (``int``)."""
        return int(len(self.elem))

    def require_cover(self, n_zones, n_union, *, where=""):
        r"""
        Fail loud when the map does not cover a consumer's slots.

        Parameters
        ----------
        n_zones : int
            Bulk zones the consumer will read.  Must match exactly:
            the zone axis is a partition of the section, not a
            subset, and a view never re-slices ``bulk_materials``.
        n_union : int
            One past the largest union index the consumer will read.
            May be **smaller** than :attr:`n_union`, because a stage
            view holds only the active-and-bonded elements while the
            map is resolved on the base section.
        where : str, optional
            Context prefix for the message.

        Raises
        ------
        ValueError
        """
        prefix = f"{where}: " if where else ""
        if int(n_zones) != self.n_zones:
            raise ValueError(
                f"{prefix}modulus map covers {self.n_zones} bulk "
                f"zone(s) but the section has {int(n_zones)}.  Resolve "
                f"the map on the same section the view came from."
            )
        if int(n_union) > self.n_union:
            raise ValueError(
                f"{prefix}modulus map covers {self.n_union} union "
                f"element(s) but index {int(n_union) - 1} is read.  "
                f"Resolve the map on the *base* section, not on a "
                f"stage view."
            )


# ------------------------------------------------------------------
#  Slot inventory and override expansion
# ------------------------------------------------------------------

#: Canonical override key prefixes, mapped to the population they
#: address.  ``'elem'`` addresses the union set directly, so it is the
#: only form that can reach an element whose kind the caller does not
#: want to spell out.
_SLOT_KINDS = ("zone", "rebar", "tendon", "elem")


def _slot_inventory(section):
    r"""
    Enumerate the bulk-zone and union-element slots of a section.

    Parameters
    ----------
    section : GenericSection
        Base section or :func:`~gensec.solver.section_state.materialize_view`
        output.  Enumerating a *view* yields that view's own union
        set, which is smaller; consumers resolve on the base section
        and index into the result through ``view._union_index``.

    Returns
    -------
    slots_bulk : list of ModulusSlot
    slots_elem : list of ModulusSlot

    Raises
    ------
    ValueError
        If ``zone_names`` is present but not index-aligned with the
        zone materials (a broken section invariant), or if two
        elements declare the same name (an ambiguous reference —
        same rule and same scope as
        :func:`gensec.io_yaml._union_name_map`).
    """
    if hasattr(section, "get_all_bulk_materials"):
        zone_mats = list(section.get_all_bulk_materials())
    else:
        zone_mats = [section.bulk_material]
    zone_names = list(getattr(section, "zone_names", None) or [])
    if not zone_names:
        zone_names = ["base"] + [f"zone_{z}"
                                 for z in range(1, len(zone_mats))]
    if len(zone_names) != len(zone_mats):
        raise ValueError(
            f"section: {len(zone_mats)} bulk zone(s) but "
            f"{len(zone_names)} zone name(s); zone_names is "
            f"index-aligned with the zone materials by contract."
        )
    slots_bulk = [
        ModulusSlot("zone", z, z, str(zone_names[z]), mat, True)
        for z, mat in enumerate(zone_mats)
    ]

    rebars = list(getattr(section, "rebars", None) or [])
    tendons = list(getattr(section, "tendons", None) or [])
    slots_elem = []
    seen_names = {}
    for k, el in enumerate(rebars + tendons):
        kind = "rebar" if k < len(rebars) else "tendon"
        local = k if kind == "rebar" else k - len(rebars)
        raw = getattr(el, "name", None)
        if raw is not None:
            nm = str(raw)
            if nm in seen_names:
                raise ValueError(
                    f"section: duplicate element name '{nm}' (union "
                    f"indices {seen_names[nm]} and {k}).  A name is a "
                    f"reference; a duplicate makes it ambiguous."
                )
            seen_names[nm] = k
            is_declared = True
        else:
            nm = f"{kind}_{local}"
            is_declared = False
        slots_elem.append(
            ModulusSlot(kind, k, local, nm, el.material, is_declared))
    return slots_bulk, slots_elem


def _key_error(key):
    r"""
    The message a rejected override key gets.

    Split out because it is the only place the retired key forms are
    named, and because a caller hitting it is a caller whose mental
    model is the pre-12_0d one: the message has to point at the
    successor, not merely refuse.

    Parameters
    ----------
    key : object

    Returns
    -------
    ValueError
    """
    if isinstance(key, int) and not isinstance(key, bool):
        return ValueError(
            f"modulus override key {key!r}: a bare integer is not a "
            f"slot.  It is ambiguous between a bulk-zone index and a "
            f"union-element index, and the ``id(material)`` form it "
            f"used to carry (Phase-5 finding F3) has no successor — a "
            f"modulus belongs to a zone, not to a material object "
            f"(12_0d).  Write ('zone', {key}) or ('elem', {key})."
        )
    return ValueError(
        f"modulus override key {key!r}: expected a material name "
        f"(str, broadcast to every slot carrying that name) or a slot "
        f"key ('zone', <name|index>), ('rebar', <name|index>), "
        f"('tendon', <name|index>), ('elem', <union index>).  A "
        f"material object is not a key: the modulus of a bulk zone is "
        f"a property of the pair (zone, instant), and two zones may "
        f"legitimately share one material object (12_0d)."
    )


def _expand_key(key, slots_bulk, slots_elem):
    r"""
    Expand one override key into the slots it addresses.

    Parameters
    ----------
    key : str or tuple
        Material name (sugar) or canonical slot key.
    slots_bulk, slots_elem : list of ModulusSlot

    Returns
    -------
    canonical : bool
        ``True`` for a slot key, ``False`` for the name sugar.  A slot
        key wins over the sugar on the slots they share; the sugar is
        a broadcast, and a broadcast never overrides a value written
        for one place in particular.
    bulk_idx : list of int
    elem_idx : list of int

    Raises
    ------
    ValueError
        Unknown kind, unknown name, index out of range, boolean
        reference, or a key form that is not accepted at all.
    """
    if isinstance(key, str):
        bulk_idx = [s.index for s in slots_bulk
                    if getattr(s.material, "name", "") == key]
        elem_idx = [s.index for s in slots_elem
                    if getattr(s.material, "name", "") == key]
        return False, bulk_idx, elem_idx

    if not isinstance(key, tuple) or len(key) != 2:
        raise _key_error(key)
    kind, ref = key
    if kind not in _SLOT_KINDS:
        raise ValueError(
            f"modulus override key {key!r}: unknown slot kind "
            f"'{kind}'.  Known: {list(_SLOT_KINDS)}."
        )
    if isinstance(ref, bool):
        raise ValueError(
            f"modulus override key {key!r}: a boolean is not a slot "
            f"reference (YAML 'true' is a bool, which is an int "
            f"subclass — accepting it as index 1 would mask an input "
            f"error)."
        )

    pool = slots_bulk if kind == "zone" else [
        s for s in slots_elem
        if kind == "elem" or s.kind == kind
    ]
    if isinstance(ref, str):
        # A synthetic label is not a reference (``ModulusSlot.declared``).
        hits = [s for s in pool if s.declared and s.name == ref]
        if not hits:
            known = sorted({s.name for s in pool if s.declared})
            raise ValueError(
                f"modulus override key {key!r}: no {kind} named "
                f"'{ref}'.  Known: {known}."
            )
        idx = [s.index for s in hits]
    elif isinstance(ref, (int, np.integer)):
        ref = int(ref)
        if kind == "elem":
            match = [s for s in pool if s.index == ref]
        else:
            match = [s for s in pool if s.local_index == ref]
        if not match:
            raise ValueError(
                f"modulus override key {key!r}: {kind} index {ref} is "
                f"out of range (the section has {len(pool)} "
                f"{kind if kind != 'elem' else 'union element'}(s))."
            )
        idx = [s.index for s in match]
    else:
        raise ValueError(
            f"modulus override key {key!r}: a slot reference is a name "
            f"(str) or an index (int), got {type(ref).__name__}."
        )

    if kind == "zone":
        return True, idx, []
    return True, [], idx


def _expand_overrides(explicit, slots_bulk, slots_elem):
    r"""
    Expand every override key into ``{slot index: (value, key)}``.

    Parameters
    ----------
    explicit : dict or None
        Override mapping.
    slots_bulk, slots_elem : list of ModulusSlot

    Returns
    -------
    bulk_over : dict
        ``{zone index: (E, key)}``.
    elem_over : dict
        ``{union index: (E, key)}``.
    expanded : dict
        ``{key: (n_bulk_hits, n_elem_hits)}``, for the dead-override
        report.

    Raises
    ------
    ValueError
        On an unusable key form, or when two *canonical* keys address
        the same slot — whichever won would be a silent choice.
    """
    bulk_over, elem_over, expanded = {}, {}, {}
    canon_bulk, canon_elem = {}, {}
    parsed = []
    for key, val in (explicit or {}).items():
        is_canon, b_idx, e_idx = _expand_key(key, slots_bulk, slots_elem)
        parsed.append((key, val, is_canon, b_idx, e_idx))
        expanded[key] = (len(b_idx), len(e_idx))
    for canonical in (False, True):       # sugar first, slot keys win
        for key, val, is_canon, b_idx, e_idx in parsed:
            if is_canon != canonical:
                continue
            for store, owner, idx, labels in (
                (bulk_over, canon_bulk, b_idx,
                 [s.name for s in slots_bulk]),
                (elem_over, canon_elem, e_idx,
                 [s.name for s in slots_elem]),
            ):
                for i in idx:
                    if canonical and i in owner and owner[i] != key:
                        raise ValueError(
                            f"modulus overrides: keys {owner[i]!r} and "
                            f"{key!r} both address slot {i} "
                            f"('{labels[i]}').  Two slot keys for one "
                            f"slot leave the winner to dictionary "
                            f"order; keep one."
                        )
                    store[i] = (val, key)
                    if canonical:
                        owner[i] = key
    return bulk_over, elem_over, expanded


def covered_zone_slots(section, explicit):
    r"""
    Zone slots an override mapping already supplies.

    The companion of :func:`resolve_moduli` for callers that
    *complete* a partial override map — they must know which zones
    the engineer has already spoken for, so as not to write a second
    key for the same slot.  Both time-dependent modulus loops use it
    (:func:`gensec.solver.losses.interval_losses`,
    :meth:`gensec.solver.timeline.ConstructionTimeline._service_datum`).

    Parameters
    ----------
    section : GenericSection
    explicit : dict or None
        Override mapping, in any accepted key form.

    Returns
    -------
    set of int
        Zone indices covered.

    Raises
    ------
    ValueError
        Propagated from key expansion (an unusable or dead-ended key
        is an error here too, and earlier is better).
    """
    slots_bulk, slots_elem = _slot_inventory(section)
    bulk_over, _elem_over, _exp = _expand_overrides(
        explicit, slots_bulk, slots_elem)
    return set(bulk_over)


def _as_material(got, slot, key, policy, cache=None):
    r"""
    Coerce one resolved entry to a material, or refuse.

    A caller may hand in either a material or a **number**.  A number is
    not a shortcut and is not coerced silently into somebody else's law:
    declaring a modulus and nothing else *is* declaring a straight line,
    so it becomes :class:`~gensec.materials.elastic.LinearElastic` at
    that modulus and the model gets exactly what it asked for.  What no
    longer happens is the reverse -- a declared concrete being replaced
    by a line nobody wrote down.

    Parameters
    ----------
    got : Material or float
    slot : ModulusSlot
    key : hashable or None
        The override key that supplied *got*, or ``None`` if the policy
        did.  Used only to name the source in a refusal.
    policy : ModulusPolicy

    Returns
    -------
    Material

    Raises
    ------
    ValueError
        If *got* is a number that is not finite and strictly positive,
        or is neither a material nor a number.
    """
    from ..materials.elastic import LinearElastic

    src = (f"override {key!r}" if key is not None
           else f"policy {type(policy).__name__}")
    if hasattr(got, "stress_array") and hasattr(got, "tangent_array"):
        return got
    try:
        E = float(got)
    except (TypeError, ValueError):
        raise ValueError(
            f"{slot.kind} slot {slot.index} ('{slot.name}'): {src} gave "
            f"{type(got).__name__}, which is neither a material nor a "
            f"modulus.  Since Phase 27 the map carries the material a "
            f"zone answers a strain with; a bare number is accepted and "
            f"means LinearElastic at that modulus."
        ) from None
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError(
            f"modulus of {slot.kind} slot {slot.index} "
            f"('{slot.name}') must be finite and strictly "
            f"positive; {src} gave {E!r}."
        )
    # One law per (declared name, modulus), not one per slot.  Two
    # zones of the same material at the same modulus must share a
    # single object, because the solver groups fibres by material
    # identity: handing it two equal-but-distinct laws would split one
    # group into two and change how the integration is batched.
    base_name = getattr(slot.material, "name", "") or slot.name
    ck = (base_name, E)
    if cache is not None and ck in cache:
        return cache[ck]
    law = LinearElastic(E=E)
    law.name = base_name
    if cache is not None:
        cache[ck] = law
    return law


def resolve_moduli(section, policy, *, explicit=None):
    r"""
    Resolve the elastic modulus of every slot of a section.

    The single resolver (12_0d §6/S1).  Resolution order, per slot:

    1. an **explicit** override addressing the slot;
    2. the **policy**'s value for the slot;
    3. otherwise a :class:`ValueError` listing every unresolved slot.

    Override keys
    -------------

    ======================================  ==============================
    key                                     meaning
    ======================================  ==============================
    ``("zone", "topping")``                 bulk-zone slot, by name
    ``("zone", 1)``                         bulk-zone slot, by index
    ``("rebar", "A1")``                     rebar slot, by declared name
    ``("tendon", "P1")``                    tendon slot, by declared name
    ``("rebar", 2)`` / ``("tendon", 0)``    element slot, by index
                                            within its own population
    ``("elem", 3)``                         element slot, by union index
    ``"C35"``                               material **name**: sugar,
                                            broadcast to every slot whose
                                            material carries that name
    ======================================  ==============================

    A slot key wins over the name sugar on the slots they share, so
    ``{"C35": 33000.0, ("zone", 1): 22000.0}`` reads "C35 everywhere,
    except the topping".  Two slot keys addressing one slot raise.

    A bare integer is **rejected**: it is ambiguous between a zone
    index and a union index, and the ``id(material)`` form it carried
    (Phase-5 finding F3) has no successor.  A material object is not
    a key either — the library's materials are unhashable by contract,
    so ``{material: E}`` raises :class:`TypeError` at construction
    before it can reach here, and the hashable outliers are refused
    with the same message.

    Parameters
    ----------
    section : GenericSection
        Base section or view; its slots are enumerated.
    policy : ModulusPolicy
        Normative content of the resolution.
    explicit : dict, optional
        Override mapping, keyed as above.  Values in MPa.

    Returns
    -------
    MaterialMap

    Raises
    ------
    ValueError
        If any slot cannot be resolved; if a value is not finite and
        strictly positive; if an override key matches no slot (a dead
        override is a silent-error trap: the caller believes a
        modulus is in force when it is not); or if a key is
        unusable.

    Notes
    -----
    The dead-override rule counts a key as *used* only when it
    actually supplies a slot's value.  A material-name key whose every
    slot is also addressed by a slot key therefore raises: it reads
    as being in force and is not.

    Examples
    --------
    >>> mm = resolve_moduli(section, SlsPolicy(),
    ...                     explicit={"C45": 36283.0,
    ...                               ("zone", "topping"): 22000.0})
    >>> mm.labels_bulk
    ('base', 'topping')
    """
    slots_bulk, slots_elem = _slot_inventory(section)
    bulk_over, elem_over, expanded = _expand_overrides(
        explicit, slots_bulk, slots_elem)

    values = {"zone": [None] * len(slots_bulk),
              "elem": [None] * len(slots_elem)}
    cache = {}
    used = set()
    missing = []
    for slots, over, out, hook in (
        (slots_bulk, bulk_over, values["zone"], policy.bulk_material),
        (slots_elem, elem_over, values["elem"], policy.elem_material),
    ):
        for pos, slot in enumerate(slots):
            key = None
            if slot.index in over:
                got, key = over[slot.index]
                used.add(key)
            else:
                got = hook(slot)
            if got is None:
                missing.append(slot)
                continue
            out[pos] = _as_material(got, slot, key, policy, cache)

    if missing:
        shown = ", ".join(
            f"{s.kind} {s.index} ('{s.name}', "
            f"{getattr(s.material, 'name', '') or type(s.material).__name__})"
            for s in missing
        )
        notes = []
        for s in missing:
            note = policy.explain(s)
            if note:
                text = f"{s.kind} {s.index} ('{s.name}'): {note}"
                if text not in notes:
                    notes.append(text)
        tail = ("\n\n" + "\n\n".join(notes)) if notes else ""
        raise ValueError(
            f"No modulus for {shown}: supply it explicitly through "
            f"the 'moduli' mapping, keyed by slot — ('zone', name or "
            f"index) for a bulk zone, ('rebar'/'tendon'/'elem', name "
            f"or index) for an element — or by material name.  "
            f"Concrete and tabulated SLS moduli are normative, "
            f"time-dependent choices and are never derived silently "
            f"(Phase-7 D3)." + tail
        )

    dead = [k for k, hits in expanded.items()
            if k not in used]
    if dead:
        detail = ", ".join(
            f"{k!r}" + ("" if sum(expanded[k]) else " (matches nothing)")
            for k in dead
        )
        raise ValueError(
            f"modulus overrides that supply no slot: {detail}.  A dead "
            f"override is a silent-error trap — the caller believes a "
            f"modulus is in force when it is not.  Remove it, or fix "
            f"the key."
        )

    return MaterialMap(
        bulk=tuple(values["zone"]),
        elem=tuple(values["elem"]),
        labels_bulk=tuple(s.name for s in slots_bulk),
        labels_elem=tuple(s.name for s in slots_elem),
    )


def resolve_sls_moduli(section, moduli=None):
    r"""
    Resolve the SLS moduli of a section under the Phase-7 D3 rule.

    Thin binding of :func:`resolve_moduli` to :class:`SlsPolicy` —
    the SLS engine's entry point, kept because it is the name every
    caller already uses.  It is not a second resolver: it supplies a
    policy and forwards.

    Parameters
    ----------
    section : GenericSection
        Base section or view; its slots are enumerated.
    moduli : dict, optional
        Override mapping.  See :func:`resolve_moduli` for the key
        contract: slot keys ``("zone", ...)``, ``("rebar", ...)``,
        ``("tendon", ...)``, ``("elem", ...)``, or a material name as
        broadcast sugar.

    Returns
    -------
    MaterialMap
        Zone-indexed and union-indexed moduli [MPa].

    Raises
    ------
    ValueError
        Unresolved slot, non-positive value, dead override, or an
        unusable key.  See :func:`resolve_moduli`.

    See Also
    --------
    resolve_moduli : the resolver, with the full key contract.
    SlsPolicy : the D3 policy this binds.
    """
    return resolve_moduli(section, SlsPolicy(), explicit=moduli)


# ==================================================================
#  SLS view (D1: constitutive substitution on the materialized view)
# ==================================================================


def sls_view(view, modmap):
    r"""
    Build the SLS view: the section view with every material replaced
    by the **service material** the map resolved for its slot.

    Since Phase 27 the map carries materials, not moduli.  A slot
    resolved from an explicit number holds
    :class:`~gensec.materials.elastic.LinearElastic` at that modulus; a
    concrete zone resolved from its rheological provider holds the
    provider's ``service_law(t)`` -- written by the provider's own
    code, e.g.
    :class:`~gensec.materials.service_concrete.ServiceConcrete`
    (non-linear in compression, released in tension at
    :math:`f_{ct,\mathrm{eff}}`) -- so the view is **not** linear in
    general and a cracked equilibrium is solved directly.

    The substitution is a **shallow re-wrap**: geometry arrays,
    initial-strain arrays and zone indices are shared by reference
    with the input view; only the material references change.  Point
    elements are re-created with :func:`dataclasses.replace` so the
    base section's :class:`~gensec.geometry.fiber.RebarLayer` /
    :class:`~gensec.geometry.fiber.Tendon` objects are **never
    mutated**.

    The map is consumed **positionally**, through the index maps the
    view already publishes (12_0d M13):

    - bulk zone :math:`z` takes ``modmap.bulk[z]`` — a view never
      re-slices ``bulk_materials``, so its zone axis *is* the base
      section's, whichever zones the state masks out of the mesh;
    - view element :math:`j` takes
      ``modmap.elem[view._union_index[j]]``, which is how a map
      resolved on the base section reaches a stage view that holds
      only the active-and-bonded elements.

    Nothing is looked up by material.  Two bulk zones sharing one
    material object therefore keep their own moduli — the collapse
    registered as 12_0d D-1 is not fixed here, it is *unrepresentable*
    here.

    One ``LinearElastic`` instance is created per distinct
    ``(original material name, E)`` pair.  This replaces the former
    invariant *"one instance per unique original material object"*
    with an explicit one: the solver's identity-based grouping
    (:meth:`FiberSolver._build_rebar_groups` and siblings) still sees
    one object per group and the per-material breakdown of
    :mod:`gensec.solver.analysis` stays separable, but two zones that
    differ only by :math:`E` now get two laws — which is the point.

    Parameters
    ----------
    view : GenericSection
        Output of :func:`~gensec.solver.section_state.materialize_view`
        (or a base section, for non-staged use, in which case the
        union index map is the identity).
    modmap : MaterialMap
        From :func:`resolve_sls_moduli` or :func:`resolve_moduli`,
        resolved on the **base** section this view came from.

    Returns
    -------
    GenericSection
        The substituted view, ready for ``FiberSolver``.

    Raises
    ------
    ValueError
        If the map does not cover the view's zones or elements
        (resolve on the same section the view came from), or if the
        view's ``_union_index`` is inconsistent with its own element
        lists.
    """
    reb = list(getattr(view, "rebars", None) or [])
    ten = list(getattr(view, "tendons", None) or [])
    n_zones = 1 + len(getattr(view, "bulk_materials", None) or [])

    uidx = getattr(view, "_union_index", None)
    if uidx is None:
        uidx = list(range(len(reb) + len(ten)))
    uidx = [int(i) for i in uidx]
    if len(uidx) != len(reb) + len(ten):
        raise ValueError(
            f"sls_view: the view publishes {len(uidx)} union index(es) "
            f"for {len(reb)} rebar(s) + {len(ten)} tendon(s).  "
            f"_union_index is index-aligned with the view's own "
            f"element lists, in that order."
        )
    modmap.require_cover(n_zones, (max(uidx) + 1) if uidx else 0,
                         where="sls_view")

    vw = copy.copy(view)
    vw.bulk_material = modmap.bulk[0]
    zones = getattr(view, "bulk_materials", None)
    if zones:
        vw.bulk_materials = [
            (poly, modmap.bulk[z + 1])
            for z, (poly, _m) in enumerate(zones)
        ]
    vw.rebars = [
        _dc_replace(r, material=modmap.elem[uidx[j]])
        for j, r in enumerate(reb)
    ]
    if ten:
        n_r = len(reb)
        vw.tendons = [
            _dc_replace(
                t, material=modmap.elem[uidx[n_r + k]])
            for k, t in enumerate(ten)
        ]
    # Per-view lazy caches must not leak across substitutions.
    vw._ideal_gross_props_cache = None
    return vw


# ==================================================================
#  Per-stage transformed properties (reporting / validation layer)
# ==================================================================


def sls_transformed_properties(base_section, state, moduli=None, *,
                               E_ref=None, compute_plastic=False):
    r"""
    Homogenized (transformed) elastic properties of a stage.

    Computes :class:`~gensec.geometry.properties.SectionProperties`
    on the **stage view** — active & bonded elements only, per the
    state — with every point element contributing its differential
    transformed area :math:`(n_i - n_{\mathrm{bulk}})\,A_i`,
    **bonded tendons included**.  Moduli come from the same explicit
    map as the solver (:func:`resolve_sls_moduli`), so the closed-form
    :math:`P/A \pm P e/W \pm M/W` built on these properties is exactly
    consistent with the fiber SLS solve (up to mesh quadrature).

    This function is the *reporting / validation* layer of Phase 7 —
    the stress engine is the fiber solve (D1).  It also resolves, in
    the per-stage direction, the tendon exclusion documented in
    :meth:`GenericSection.compute_ideal_properties` (roadmap §5.3):
    tendons stay out of the *base* ideal properties (where the fiber
    path would double-count the prestrain), and enter the *elastic
    transformed* properties here, where they belong.

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState
    moduli : dict, optional
        Explicit modulus overrides, as in :func:`resolve_sls_moduli`.
    E_ref : float or None, optional
        Reference modulus for homogenization.  ``None`` (default)
        uses the bulk SLS modulus (:math:`n_{\mathrm{bulk}} = 1`).
    compute_plastic : bool, optional
        Forwarded to
        :func:`~gensec.geometry.properties.compute_section_properties`.

    Returns
    -------
    gensec.geometry.properties.SectionProperties

    Raises
    ------
    NotImplementedError
        For multi-zone bulk sections (mirrors the existing
        ``compute_ideal_properties`` restriction).
    ValueError
        Propagated from :func:`resolve_sls_moduli` (missing moduli).
    """
    from ..geometry.properties import (
        compute_section_properties, HomogenizedRebar,
    )

    if len(getattr(base_section, "bulk_materials", [])) > 0:
        raise NotImplementedError(
            "sls_transformed_properties currently supports a single "
            "bulk zone (mirrors compute_ideal_properties)."
        )

    view = materialize_view(base_section, state)
    modmap = resolve_sls_moduli(base_section, moduli)
    E_bulk = float(modmap.E_bulk[0])

    # Positional consumption, as in sls_view: the map is resolved on
    # the base section, the view holds a subset of its elements, and
    # ``_union_index`` is the bridge (12_0d M13).
    n_r = len(view.rebars)
    uidx = [int(i) for i in getattr(
        view, "_union_index",
        range(n_r + len(getattr(view, "tendons", None) or [])))]
    homog = [
        HomogenizedRebar(r.x, r.y, r.As, float(modmap.E_elem[uidx[j]]))
        for j, r in enumerate(view.rebars)
        if r.embedded and r.x is not None
    ]
    homog += [
        HomogenizedRebar(float(x), float(y), float(A),
                         float(modmap.E_elem[uidx[n_r + k]]))
        for k, (t, x, y, A) in enumerate(zip(
            getattr(view, "tendons", []),
            getattr(view, "x_tendons", np.empty(0)),
            getattr(view, "y_tendons", np.empty(0)),
            getattr(view, "A_tendons", np.empty(0)),
        ))
        if t.embedded
    ]
    return compute_section_properties(
        base_section.polygon, rebars=homog, E_bulk=E_bulk,
        E_ref=E_ref, compute_plastic=compute_plastic,
    )


# ==================================================================
#  Internal helpers
# ==================================================================


def _solve_linear(solver, F, tol, max_iter, ctx):
    r"""
    Equilibrium solve on an SLS view, fail-loud on non-convergence.

    This is :meth:`FiberSolver.solve_equilibrium` with a stricter
    message; the name is Phase-7's and is kept because the call site
    has not changed.  On a view that declares straight lines the Newton
    solve is exact after one tangent step.  On a view that declares a
    real constitutive law it is a genuine Newton iteration, and
    non-convergence can mean an ill-posed input **or** a section that
    has no equilibrium at the demand asked of it — an unreinforced zone
    past its cracking moment has none, and that is physics, not a bug.

    Parameters
    ----------
    solver : FiberSolver
    F : tuple of float
        ``(N, Mx, My)`` [N, N·mm, N·mm].
    tol, max_iter :
        Forwarded to :meth:`FiberSolver.solve_equilibrium`.
    ctx : str
        Context string for the error message.

    Returns
    -------
    dict
        The solver result (``eps0``, ``chi_x``, ``chi_y``, …).

    Raises
    ------
    RuntimeError
        If the solve does not converge.
    """
    sol = solver.solve_equilibrium(F[0], F[1], F[2],
                                   tol=tol, max_iter=max_iter)
    if not sol["converged"]:
        raise RuntimeError(
            f"SLS linear equilibrium solve failed to converge "
            f"({ctx}; target N={F[0]:.6g} N, Mx={F[1]:.6g} N·mm, "
            f"My={F[2]:.6g} N·mm).  Either the view is degenerate, or "
            f"the section carries no equilibrium at this demand under "
            f"the declared laws."
        )
    return sol


def _element_states(solver, sol, view):
    r"""
    Union-indexed element **strains** and laws from one solved plane.

    Phase 28 reads strains here where Phase 7 read stresses.  The
    reason is the defect this module was rewritten for: stresses do not
    superpose unless the law is a straight line, while strains always
    do, because the kinematics of a plane section are linear whatever
    the material is.  The accumulator therefore carries strain, and the
    stress is formed once, at the end, by asking the element's own law.

    The strain returned is the one the element's law is evaluated at —
    the rebar's section strain, the tendon's ``eps_tot`` including its
    locked-in prestrain — not the net-of-displaced-bulk quantity used
    internally for integration.

    Parameters
    ----------
    solver : FiberSolver
    sol : dict
        Solver result (``eps0``, ``chi_x``, ``chi_y``).
    view : GenericSection
        The (substituted) view the solver was built on; must carry
        ``_union_index``.

    Returns
    -------
    elem_union_idx : numpy.ndarray of int
        Union indices of the view's point elements, rebars first.
    elem_eps : numpy.ndarray
        Element strains aligned with ``elem_union_idx`` [-].
    elem_materials : list of Material
        The law of each element, in the same order.
    plane : tuple of float
        ``(eps0, chi_x, chi_y)``.
    """
    fr = solver.get_fiber_results(sol["eps0"], sol["chi_x"],
                                  sol["chi_y"])
    eps_r = np.asarray(fr["rebars"]["eps"], dtype=float)
    mats_r = [r.material for r in (view.rebars or [])]
    if "tendons" in fr:
        eps_t = np.asarray(fr["tendons"]["eps_tot"], dtype=float)
        mats_t = [t.material for t in (view.tendons or [])]
    else:
        eps_t = np.empty(0, dtype=float)
        mats_t = []
    elem_eps = np.concatenate([eps_r, eps_t])
    elem_materials = mats_r + mats_t
    elem_union_idx = np.asarray(view._union_index, dtype=int)
    if elem_union_idx.size != elem_eps.size:
        raise RuntimeError(
            "View element count does not match _union_index — the "
            "view was not produced by materialize_view."
        )
    if len(elem_materials) != elem_eps.size:
        raise RuntimeError(
            "View element count does not match its material list — "
            "the view was not produced by materialize_view."
        )
    plane = (float(sol["eps0"]), float(sol["chi_x"]),
             float(sol["chi_y"]))
    return elem_union_idx, elem_eps, elem_materials, plane


def _sigma_of(material, eps):
    r"""
    One stress from one law at one strain [MPa].

    Parameters
    ----------
    material : Material
    eps : float

    Returns
    -------
    float
    """
    return float(np.asarray(
        material.stress_array(np.asarray([eps], dtype=float)))[0])


def _strain_entering_set(strains, plane_hi, state_planes,
                         zone_entering, x_ref, y_ref, d_bulk_eps=0.0):
    r"""
    Initialise the per-zone affine **strain** field of the zones that
    enter the resistance set at the current stage.

    An entering zone comes into existence carrying its locked-in datum
    plane :math:`(\varepsilon_0^d, \chi_x^d, \chi_y^d)_z`, which the
    fibre kernel *adds* to the section plane (F7); the coefficients are
    therefore SET, not incremented, from the effective plane
    :math:`\mathbf{p}_{hi} + \mathbf{p}^d_z`:

    .. math::

        e_{0,z} &= \varepsilon_0 - \chi_x\, y_{\mathrm{ref}}
            + \chi_y\, x_{\mathrm{ref}} + \varepsilon_b, \\
        e_{x,z} &= -\chi_y, \qquad e_{y,z} = \chi_x ,

    so that the material strain anywhere in the zone is
    :math:`\varepsilon_z(x,y) = e_{0,z} + e_{x,z}x + e_{y,z}y`.

    This is the Phase-7 expression with the zone modulus taken out of
    it.  Taking it out is the repair: the modulus belonged there only
    while the field carried stress, and a stress field is affine only
    on a straight line.

    Parameters
    ----------
    strains : dict
        ``{"e0", "ex", "ey"}`` per-zone arrays, mutated in place.
    plane_hi : tuple of float
        Solved section plane ``(eps0, chi_x, chi_y)``.
    state_planes : numpy.ndarray or None
        Per-zone locked-in datum planes, shape ``(n_zones, 3)``; ``None``
        means a zero datum for every zone.
    zone_entering : numpy.ndarray of bool
        Zones to initialise this call.
    x_ref, y_ref : float
        Solver reference point.
    d_bulk_eps : float, optional
        Legacy uniform bulk imposed-strain offset.
    """
    ent = np.nonzero(np.asarray(zone_entering, dtype=bool))[0]
    if ent.size == 0:
        return
    n_zones = int(strains["e0"].size)
    planes = (np.zeros((n_zones, 3), dtype=float)
              if state_planes is None
              else np.asarray(state_planes, dtype=float))
    for z in ent:
        e0 = plane_hi[0] + float(planes[z, 0])
        cx = plane_hi[1] + float(planes[z, 1])
        cy = plane_hi[2] + float(planes[z, 2])
        strains["e0"][z] = e0 - cx * y_ref + cy * x_ref + d_bulk_eps
        strains["ex"][z] = -cy
        strains["ey"][z] = cx


def _active_zone_labels(base_section, zone_active, n_zones):
    r"""
    Human-readable labels of the bulk zones active at a stage.

    Parameters
    ----------
    base_section : GenericSection
    zone_active : numpy.ndarray of bool or None
        The stage's bulk-active mask; ``None`` means every zone active.
    n_zones : int

    Returns
    -------
    list of str
        Labels (``zone_names`` when available, else the zone index) of
        the active zones, in index order.
    """
    names = getattr(base_section, "zone_names", None)
    if zone_active is None:
        active = range(n_zones)
    else:
        za = np.asarray(zone_active, dtype=bool)
        active = [int(z) for z in np.nonzero(za)[0]]
    out = []
    for z in active:
        if names is not None and z < len(names):
            out.append(str(names[z]))
        else:
            out.append(str(z))
    return out


def _strain_increment(strains, plane_hi, plane_lo,
                      x_ref, y_ref, d_bulk_eps=0.0, zone_mask=None):
    r"""
    Accumulate one plane difference into the per-zone affine strain
    field.

    For a plane increment
    :math:`(\Delta\varepsilon_0, \Delta\chi_x, \Delta\chi_y)` and an
    optional bulk imposed-strain increment
    :math:`\Delta\varepsilon_b`:

    .. math::

        \Delta e_0 &= \Delta\varepsilon_0
            - \Delta\chi_x\, y_{\mathrm{ref}}
            + \Delta\chi_y\, x_{\mathrm{ref}}
            + \Delta\varepsilon_b, \\
        \Delta e_x &= -\Delta\chi_y, \qquad
        \Delta e_y = \Delta\chi_x .

    Why this is exact and its Phase-7 predecessor was not
    -----------------------------------------------------
    The increment structure is not an approximation: it exists because
    a zone cast at stage :math:`k` must not be handed the strain the
    rest of the section carried before it existed, and differencing two
    solves on the *same* view is what removes that.  What was an
    approximation was doing it in stress.  Superposing
    :math:`\Delta\sigma = E_z\Delta\varepsilon` assumes one secant
    modulus governs every increment, which is true of a straight line
    and of nothing else; superposing strain assumes only that plane
    sections stay plane.

    Parameters
    ----------
    strains : dict
        ``{"e0": ndarray, "ex": ndarray, "ey": ndarray}`` per zone,
        mutated in place.
    plane_hi, plane_lo : tuple of float
        ``(eps0, chi_x, chi_y)`` of the two solves.
    x_ref, y_ref : float
        Solver reference point.
    d_bulk_eps : float, optional
        Bulk imposed-strain increment carried by this pair.
    zone_mask : array_like of bool, optional
        Restrict the increment to a subset of zones (persisting zones —
        entering zones are SET, not incremented).
    """
    de0 = plane_hi[0] - plane_lo[0]
    dcx = plane_hi[1] - plane_lo[1]
    dcy = plane_hi[2] - plane_lo[2]
    n_zones = int(strains["e0"].size)
    d_e0 = np.full(n_zones,
                   de0 - dcx * y_ref + dcy * x_ref + d_bulk_eps,
                   dtype=float)
    d_ex = np.full(n_zones, -dcy, dtype=float)
    d_ey = np.full(n_zones, dcx, dtype=float)
    if zone_mask is not None:
        m = np.asarray(zone_mask, dtype=bool)
        d_e0 = np.where(m, d_e0, 0.0)
        d_ex = np.where(m, d_ex, 0.0)
        d_ey = np.where(m, d_ey, 0.0)
    strains["e0"] += d_e0
    strains["ex"] += d_ex
    strains["ey"] += d_ey


def _decompression_probe(x_t, y_t, zone, strains, material, e_ref,
                         c_dec):
    r"""
    Decompression probe for one bonded tendon (closed form).

    The accumulated **strain** field in the tendon's zone is affine,
    :math:`\varepsilon(x,y) = e_0 + e_x x + e_y y`, so the most tensile
    point of the disc of radius :math:`c_{\mathrm{dec}}` centred on the
    tendon lies along the gradient :math:`(e_x, e_y)`.

    The verdict is taken **on strain**, not on stress, and that is the
    repair.  Decompression asks whether the concrete at the probe point
    is still in compression.  A cracked point returns
    :math:`\sigma = 0` from the service law, which would satisfy
    :math:`\sigma \le 0` and pass — a section that has cracked through
    would be certified decompressed.  Strain has no such blind spot.

    The tolerance is the declared stress tolerance divided by the
    zone's reference modulus, so that on a straight line the criterion
    is bit-for-bit the one Phase 7 applied and the linear discriminator
    does not move.

    Parameters
    ----------
    x_t, y_t : float
        Tendon coordinates [mm].
    zone : int
        Bulk-zone index of the tendon.
    strains : dict
        Per-zone affine strain coefficients.
    material : Material
        The zone's constitutive law, used to report the stress at the
        probe point.
    e_ref : float
        The zone's reference modulus [MPa], used only to express the
        declared stress tolerance as a strain tolerance.
    c_dec : float
        Cover distance [mm].

    Returns
    -------
    dict
        ``x_probe``, ``y_probe``, ``eps_probe`` [-],
        ``passed`` (bool).

    Notes
    -----
    The stress at the probe point is deliberately **not** returned.
    It used to be, and the record published it beside the strain the
    verdict is actually taken on; on a cracking law the two give
    opposite answers the moment the probe point cracks, because the
    law returns zero past :math:`\varepsilon_{ct}`.  Measured at
    :math:`M = -90` kNm on the Phase-7 reference: strain
    :math:`1.6766\times10^{-4}` (fails), stress :math:`0.000000`
    (would pass).  Two numbers for one question is the shape that
    produced F6, so the second one is not computed at all.
    """
    e0 = float(strains["e0"][zone])
    ex = float(strains["ex"][zone])
    ey = float(strains["ey"][zone])
    e_ref = float(e_ref)
    g = float(np.hypot(ex, ey))
    if g * e_ref < 1.0e-15:
        # Uniform field: every point of the disc carries the same
        # strain; probe at the tendon itself.
        xp, yp = float(x_t), float(y_t)
    else:
        xp = float(x_t) + c_dec * ex / g
        yp = float(y_t) + c_dec * ey / g
    ep = e0 + ex * xp + ey * yp
    return {
        "x_probe": xp, "y_probe": yp,
        "eps_probe": ep,
        "passed": bool(ep <= _DECOMPRESSION_TOL / e_ref),
    }


def _resist_mask(state, n_union):
    """``active & bonded`` over the union set, validated length."""
    m = np.asarray(state.active, dtype=bool) & \
        np.asarray(state.bonded, dtype=bool)
    if m.size != n_union:
        raise ValueError(
            f"SectionState arrays have length {m.size}, expected the "
            f"union length {n_union} of the base section."
        )
    return m


def _check_entry(name, value, limit, *, extra=None):
    r"""
    Build one check record.

    .. math::

        \eta = \frac{|v|}{v_{\lim}},
        \qquad
        \texttt{passed} \iff |v| \le v_{\lim},

    with both quantities positive magnitudes.

    One arithmetic, not two
    -----------------------
    Until Phase 30 the three published numbers were rounded to six
    decimals while the verdict was taken on the unrounded comparison.
    That is two arithmetics, and near a limit they disagree.  There were
    two artefacts with two different widths:

    - ``value_MPa`` collapsed onto ``limit_MPa`` within a constant band
      of :math:`5\times10^{-7}` MPa;
    - ``eta`` collapsed onto :math:`1.0` within a band of
      :math:`5\times10^{-7}\,v_{\lim}`, which **scales with the limit**
      -- 1.062e-05 MPa at 21 MPa, 9.449e-04 MPa at 1860 MPa, the latter
      1833 times the band on the value.

    On a tendon limit the record could read ``value_MPa = 1860.0009``,
    ``limit_MPa = 1860.0``, ``eta = 1.0``, ``passed = False``: three
    assertions of which two contradict the third.  A reader auditing the
    record cannot tell which one the engine acted on.

    The record now publishes what the verdict was taken on.  Rounding is
    a presentation concern and belongs to whatever renders the record,
    where it can be applied without anything downstream mistaking a
    rounded number for the number.

    Parameters
    ----------
    name : str
        Name of the check, used verbatim in the record and in the
        refusal below.
    value : float
        Positive magnitude of the acting quantity [MPa].
    limit : float
        Positive magnitude of the threshold [MPa].
    extra : dict, optional
        Additional payload merged into the record.

    Returns
    -------
    dict
        Keys ``name``, ``value_MPa``, ``limit_MPa``, ``eta``,
        ``passed``, plus anything in ``extra``.

    Raises
    ------
    ValueError
        If ``limit`` is not a finite positive magnitude.  A zero limit
        used to surface as ``ZeroDivisionError`` from the arithmetic,
        which names nothing a caller can act on; a negative limit did
        not raise at all and produced a negative utilisation, which is
        worse because it is silent.  Both are malformed declarations,
        and the refusal belongs where the division is rather than in
        whatever built the limit set: this function is reachable from
        every check and from callers that build a ``StressLimits`` by
        hand.
    """
    lim = float(limit)
    if not np.isfinite(lim) or lim <= 0.0:
        raise ValueError(
            f"check {name!r}: the limit must be a finite positive "
            f"magnitude [MPa], got {limit!r}.  A utilisation "
            f"eta = value / limit has no meaning otherwise, and "
            f"reporting one would be a number with no content.  If the "
            f"intent is that this check does not apply, leave its "
            f"threshold undeclared (None) and it will be recorded as "
            f"skipped."
        )
    val = float(value)
    rec = {
        "name": name,
        "value_MPa": val,
        "limit_MPa": lim,
        "eta": val / lim,
        "passed": bool(val <= lim),
    }
    if extra:
        rec.update(extra)
    return rec


def _skipped_entry(name, reason):
    """Explicit record for a check that could not run (no silence)."""
    return {"name": name, "skipped": True, "reason": reason}


# ==================================================================
#  Public: staged SLS verification
# ==================================================================


def _zone_regions(base_section):
    r"""
    The **material region** of every bulk zone, as a Shapely geometry.

    A zone declared in ``bulk_materials`` is an *overlay*: it occupies its
    own polygon, and the base zone occupies whatever is left.  So the base
    region is the section polygon **minus** the union of the overlays --
    not the section polygon.

    Returns
    -------
    list of shapely geometry
        Index-aligned with the zone indices (0 = base).
    """
    from shapely.ops import unary_union
    overlays = [p for (p, _m) in (base_section.bulk_materials or [])]
    base = base_section.polygon
    if overlays:
        regions = [base.difference(unary_union(overlays))]
    else:
        regions = [base]
    regions += [p.intersection(base) for p in overlays]
    return regions


def _zone_declared_materials(base_section):
    r"""
    The **declared** material of every bulk zone, index-aligned.

    Not the same objects as the ``zone_materials`` the SLS walk uses.
    Those are the *service view* -- a
    :class:`~gensec.materials.service_concrete.ServiceConcrete` or a
    bare modulus, resolved by :func:`resolve_sls_moduli`, which carries
    a stress-strain law and no characteristic strength.  ``f_ck`` is a
    property of the concrete the engineer declared, and it lives on the
    section.

    Measured, and the reason this function exists rather than a
    ``getattr(zone_materials[z], "fck", None)``: that expression
    returns ``None`` for **every** zone of every model, because no
    service view carries the attribute.  The per-zone limit would then
    have been skipped on every section in the tree -- a silent no-op
    wearing the shape of a repair.

    Returns
    -------
    list
        Index-aligned with the zone indices (0 = base).
    """
    return ([base_section.bulk_material]
            + [m for (_p, m) in (base_section.bulk_materials or [])])


def _region_vertices(geom):
    r"""Every boundary vertex of a (possibly multi-part, possibly holed)
    region, as ``(x, y)`` pairs."""
    pts = []
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for g in parts:
        if g.is_empty:
            continue
        pts.extend(g.exterior.coords)
        for ring in g.interiors:
            pts.extend(ring.coords)
    return pts


def _snap_to_isostrain(region, e0, ex, ey, e_b, interpolated):
    r"""
    Move a breakpoint coordinate onto material, without moving its value.

    The locus of a given strain in an affine field is a straight line,

    .. math::

        \mathcal{L}_{\varepsilon_b}
        = \{(x,y) : e_0 + e_x x + e_y y = \varepsilon_b\},

    so **every** point of :math:`\mathcal{L}_{\varepsilon_b}\cap R`
    carries exactly :math:`\varepsilon_b` and therefore exactly the same
    stress.  Relocating the reported coordinate along that line is free:
    it changes the label and cannot change the value.  That is the whole
    reason a snap is admissible where a discard is not -- discarding the
    candidate instead was measured to move the reported extreme stress
    by up to 28.2743 MPa on ``example_custom``, roughly 65 % of the peak
    of a C35 service law.

    The coordinate is a choice, and the docstring must not pretend
    otherwise
    -----------------------------------------------------------------
    The line meets a re-entrant or holed region in more than one
    disjoint segment for 66.7 % of strain directions on
    ``example_annulus`` and ``example_custom``, and 14.4 % on
    ``example_tee``.  Each of them is an equally correct answer for the
    value.  What is published is therefore **a** point at which the
    extreme is attained, not **the** point, and the choice made here --
    the admissible point nearest the interpolated one, ties broken
    lexicographically -- exists so that two identical runs produce two
    identical fingerprints, not because that point is more true than
    its neighbours.

    Parameters
    ----------
    region : shapely geometry
        The zone's material region, possibly non-convex or holed, and
        possibly a ``MultiPolygon`` of disjoint parts.
    e0, ex, ey : float
        Affine strain coefficients of the zone.
    e_b : float
        The breakpoint strain to be located.
    interpolated : tuple of float
        The coordinate obtained by interpolating along the segment
        joining the two extreme-strain vertices, used as the point to
        stay nearest to.

    Returns
    -------
    tuple of float or None
        A point of the region carrying strain ``e_b``, or ``None`` when
        the region carries that strain nowhere.  ``None`` is not a
        silent loss: on a zone made of disjoint parts a strain lying
        between the global extremes may be attained in neither part, and
        then no material turns the law over at ``e_b``.  Reporting a
        stress that no material carries would be the defect, not the
        remedy.
    """
    grad = math.hypot(ex, ey)
    if grad <= 0.0:
        # A uniform strain field has no iso-strain line: either the
        # whole region is at e_b or none of it is, and the vertex
        # candidates already say which.
        return None
    minx, miny, maxx, maxy = region.bounds
    reach = 2.0 * math.hypot(maxx - minx, maxy - miny) + 1.0
    # A point on the line, and the unit direction along it.
    s = (e_b - e0) / (grad * grad)
    px, py = s * ex, s * ey
    dx, dy = -ey / grad, ex / grad
    line = LineString([(px - reach * dx, py - reach * dy),
                       (px + reach * dx, py + reach * dy)])
    inter = line.intersection(region)
    if inter.is_empty:
        return None
    target = Point(interpolated)
    parts = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
    best = None
    for part in parts:
        if part.is_empty:
            continue
        near = part.interpolate(part.project(target)) \
            if part.geom_type != "Point" else part
        key = (target.distance(near), near.x, near.y)
        if best is None or key < best[0]:
            best = (key, (near.x, near.y))
    return None if best is None else best[1]


def _zone_candidates(strains, z, region, material):
    r"""
    Every point of one zone that can carry that zone's extreme stress.

    The strain field is affine over the region, so its own extremes sit
    at **vertices**.  The stress is the law evaluated on that field, and
    a law is not monotone: EN 1992-1-1 (3.14) peaks at
    :math:`-\varepsilon_{c1}` and the tension branch drops to zero past
    :math:`\varepsilon_{ct}`.  Wherever the law turns over *inside* the
    strain interval the region spans, that turning point is a candidate
    the vertices cannot see.

    The material is asked for its own turning points
    (:meth:`~gensec.materials.base.Material.stress_breakpoints`) rather
    than sampled: sampling would put back the discretisation error that
    the closed-form extreme-fibre evaluation exists to remove (F6), and
    where a law turns over is a fact belonging to the law.

    The coordinate reported for a turning point is obtained by linear
    interpolation along the segment joining the two extreme-strain
    vertices.  The strain there is exact — the field is affine along any
    segment — and for a **convex** region the point lies inside it.

    On a re-entrant or holed region that segment leaves the material,
    and until Phase 30 the record then named a coordinate with no
    material under it: measured over 360 strain directions, every
    direction on ``example_annulus`` (worst excursion 366.335 mm), 359
    of 360 on ``example_custom`` (108.788 mm), and 121 of 360 on
    ``example_tee`` (17.144 mm).  The tee is worth naming because
    ``29_0`` §10 recorded it as unaffected: it is unaffected only for
    the single bending direction that measurement used, and GenSec
    supports biaxial bending.

    The **value** was never in question.  A connected region under an
    affine field attains, by the intermediate-value theorem, every
    strain between its own extremes; only the label was wrong.  Points
    that fall outside are therefore relocated onto the iso-strain line
    by :func:`_snap_to_isostrain`, which cannot move the strain and so
    cannot move the stress.  Points already inside are returned
    untouched, so convex zones — the great majority — are bit-for-bit
    unchanged.

    Parameters
    ----------
    strains : dict
        Per-zone affine strain coefficients.
    z : int
    region : shapely geometry
    material : Material

    Returns
    -------
    list of tuple
        ``(eps, (x, y))`` pairs.
    """
    e0 = float(strains["e0"][z])
    ex = float(strains["ex"][z])
    ey = float(strains["ey"][z])
    cand = []
    for (x, y) in _region_vertices(region):
        cand.append((e0 + ex * float(x) + ey * float(y),
                     (float(x), float(y))))
    if not cand:
        return cand
    lo = min(cand, key=lambda t: t[0])
    hi = max(cand, key=lambda t: t[0])
    span = hi[0] - lo[0]
    if span <= 0.0:
        return cand
    for e_b in material.stress_breakpoints():
        e_b = float(e_b)
        if not (lo[0] < e_b < hi[0]):
            continue
        t = (e_b - lo[0]) / span
        point = (lo[1][0] + t * (hi[1][0] - lo[1][0]),
                 lo[1][1] + t * (hi[1][1] - lo[1][1]))
        if not region.covers(Point(point)):
            point = _snap_to_isostrain(region, e0, ex, ey, e_b, point)
            if point is None:
                # No material in this zone carries e_b: the law does
                # not turn over anywhere real, so there is no candidate
                # to report.  See _snap_to_isostrain for why this is a
                # correction and not a dropped extreme.
                continue
        cand.append((e_b, (float(point[0]), float(point[1]))))
    return cand


def _extreme_fibre(base_section, strains, zone_materials,
                   present_zones):
    r"""
    The **true extreme-fibre** concrete stresses, in closed form.

    The accumulated strain field is exactly affine over each zone,
    :math:`\varepsilon_z(x,y) = e_{0,z} + e_{x,z}x + e_{y,z}y`, because
    plane sections stay plane whatever the material does.  The stress is
    the zone's own law evaluated on that field, and the extremes are
    attained at a vertex of the zone's material region or at a strain
    where the law itself turns over — a finite candidate set, so this
    stays a closed form and not a search.

    .. warning::

        Taking ``argmin``/``argmax`` over the *fibres* reports the
        stress at the outermost fibre's **centroid**, inset half a fibre
        from the section face.  On a 800 mm web meshed at 50 mm that
        under-reports the extreme-fibre stress by
        :math:`1 - 50/800 = 6.25\,\%` of the bending part, in the
        **non-conservative** direction: EN 1992-1-1 §7.2 limits
        :math:`\sigma_c` at the *extreme fibre*, not at the centroid of
        the last row of fibres.  A fine mesh hides it, which is why it
        survived (Phase-5 finding F6).  The record produced by
        :func:`verify_sls_staged` reports this closed form; the
        ``concrete_compression`` **check** still reads the fibre value,
        which is F6 applied to one path and not the other and is a
        separate open defect (``28_0`` §2).

    Parameters
    ----------
    base_section : GenericSection
    strains : dict
        ``{'e0': (n_zones,), 'ex': ..., 'ey': ...}`` -- the per-zone
        affine strain field already assembled by the caller.
    zone_materials : sequence of Material
        The law of each zone, index-aligned with *strains*.
    present_zones : iterable of int
        Zones that have entered the history (a not-yet-cast zone must not
        spoof the extremes with its zero).

    Returns
    -------
    tuple
        ``(sigma_min, (x, y), sigma_max, (x, y))``.
    """
    regions = _zone_regions(base_section)
    s_min, p_min = float("inf"), None
    s_max, p_max = float("-inf"), None
    for z in present_zones:
        z = int(z)
        if z >= len(regions):
            continue
        mat = zone_materials[z]
        for eps, pt in _zone_candidates(strains, z, regions[z], mat):
            s = _sigma_of(mat, eps)
            if s < s_min:
                s_min, p_min = s, pt
            if s > s_max:
                s_max, p_max = s, pt
    if p_min is None:
        raise RuntimeError(
            "extreme-fibre stress: no present zone has a region.")
    return s_min, p_min, s_max, p_max


def _extreme_fibre_by_zone(base_section, strains, zone_materials,
                           present_zones):
    r"""
    The extreme-fibre stresses of **each** present zone, separately.

    The global extremes are not enough on a composite: the maximum
    compressive stress and the maximum tensile stress may sit in
    *different* zones, each zone has its own law and its own
    :math:`f_{ck}`, and a check that reads only the global extreme is
    checking one zone against another zone's limit.

    Parameters
    ----------
    base_section : GenericSection
    strains : dict
    zone_materials : sequence of Material
    present_zones : iterable of int

    Returns
    -------
    dict
        ``{zone_name: {'sigma_min_MPa', 'at_min', 'sigma_max_MPa',
        'at_max'}}``.
    """
    regions = _zone_regions(base_section)
    out = {}
    for z in present_zones:
        z = int(z)
        if z >= len(regions):
            continue
        mat = zone_materials[z]
        s_min, p_min = float("inf"), None
        s_max, p_max = float("-inf"), None
        for eps, pt in _zone_candidates(strains, z, regions[z], mat):
            s = _sigma_of(mat, eps)
            if s < s_min:
                s_min, p_min = s, pt
            if s > s_max:
                s_max, p_max = s, pt
        if p_min is None:
            continue
        name = (base_section.zone_names[z]
                if z < len(base_section.zone_names) else f"zone[{z}]")
        # NOT rounded (Phase 32).  Since this phase the per-zone
        # concrete check takes its verdict on exactly this number, and
        # Phase 30 settled the rule in ``_check_entry``: the record
        # publishes what the verdict was taken on, because two
        # arithmetics disagree near a limit and a reader auditing the
        # record cannot tell which one the engine acted on.  Rounding
        # is a presentation concern and belongs to whatever renders
        # the record.
        out[name] = {"sigma_min_MPa": s_min, "at_min": p_min,
                     "sigma_max_MPa": s_max, "at_max": p_max}
    return out


def verify_sls_staged(base_section, stages, *, moduli=None,
                      x_ref=None, y_ref=None, tol=1.0e-3,
                      max_iter=50):
    r"""
    SLS stress verification over a stage history (Phase 7 engine,
    Phase 28 accumulation).

    Runs the incremental accumulation described in the module docstring
    over the given stages, then evaluates the per-stage
    :class:`~gensec.materials.verification_limits.StressLimits`
    checks on the accumulated state.

    What is accumulated, and why it is strain
    -----------------------------------------
    Phase 7 accumulated **stress**, as
    :math:`\Delta\sigma = E_z\,\Delta\varepsilon` with one secant
    modulus per zone (decision D2).  That is superposition, and
    superposition of stresses is exact only while the law is a straight
    line.  Phase 27 put the real service law in the view; the scheme
    kept running and stopped being correct.

    This engine accumulates **strain** instead, in one per-zone affine
    field, and forms every stress by asking the zone's own law at the
    end of the stage.  Strain superposes exactly, because the
    kinematics of a plane section are linear whatever the material
    does.  The increment structure is unchanged and still necessary: a
    zone cast at stage :math:`k` must not be handed the strain the rest
    of the section carried before it existed, and differencing two
    solves on the same view is what prevents that.

    A consequence worth stating: the per-fibre array and the per-zone
    field are now **the same object**, since the strain at a fibre is
    the field evaluated at the fibre.  The Phase-7 cross-check between
    the two bookkeepings (``debug_check_affine``) is therefore removed
    rather than adapted — it existed to compare two accumulators, and
    there is one.

    Parameters
    ----------
    base_section : GenericSection
        The immutable base section (union element set).
    stages : list of dict
        One dict per stage, keys:

        ``"name"`` : str, required
            Stage identifier.
        ``"state"`` : SectionState, required
            The stage's section state.
        ``"increment"`` : tuple of float, optional
            External demand increment ``(dN, dMx, dMy)`` [N, N·mm,
            N·mm] about ``(x_ref, y_ref)``.  Default ``(0, 0, 0)``.
        ``"prestress_actions"`` : list of PrestressAction, optional
            Demand-side loads summed into the stage increment (their
            triples must have been built about the **same**
            ``(x_ref, y_ref)``).
        ``"time"`` : float, optional
            Time [days], echoed to the report.
        ``"limits"`` : StressLimits, optional
            Checks to evaluate on this stage's accumulated state.
            Stages without limits accumulate but produce no verdict.
        ``"unbonded_sigma_p"`` : dict, optional
            ``{label: sigma_p [MPa]}`` member-level stresses of
            active-unbonded tendons, checked against
            ``sigma_p_max`` with ``provenance='member_level'``.

        A ``"moduli"`` key is **refused**; see the raises section.

    moduli : dict, optional
        Global SLS material overrides (see
        :func:`resolve_sls_moduli`).  A bare number means
        :class:`~gensec.materials.elastic.LinearElastic` at that
        modulus, because declaring a modulus and nothing else *is*
        declaring a straight line.
    x_ref, y_ref : float, optional
        Demand reference point.  Default: the base-section centroid.
        Fixed once for the whole history — demand triples and
        :class:`PrestressAction` couples must share it.
    tol : float, optional
        Force tolerance forwarded to the equilibrium solves [N].
    max_iter : int, optional
        Newton iteration cap.

    Returns
    -------
    dict
        ``{"type": "sls_staged", "stages": [...], "verified": bool,
        "governing": {...} or None}``.  Each stage record carries the
        cumulative demand, the solved plane of the stage, concrete
        stress extremes with locations, the per-element stress table
        (union-indexed), the check records, and ``"fiber_field"`` --
        the same accumulated bulk/rebar strain-stress arrays
        repackaged in the shape
        :meth:`~gensec.solver.integrator.FiberSolver.get_fiber_results`
        returns, together with the stage's materialised (active-only)
        geometry under ``"section_view"``, for
        :func:`gensec.output.sls_state_plots.export_sls_state_plots`.

    Raises
    ------
    NotImplementedError
        A stage removes elements from the resistance set (leaving
        transitions are deferred), or the locked-in datum of a
        persisting bulk zone changes mid-history.
    ValueError
        Compound transition (elements entering while initial strains
        change on persisting elements — split into two stages), a
        per-stage ``"moduli"`` override, or malformed inputs.
    RuntimeError
        An equilibrium solve failed to converge.

    Notes
    -----
    The per-element table reports the **material** stress of each
    active-and-bonded element (its own law at its own accumulated
    strain, prestrain included) — the quantity the normative limits
    address.

    **Single-compatibility-plane invariant (C4).** This walk models
    staged construction under a *single* strain-compatibility plane
    per stage: every active bulk zone shares one solved plane
    :math:`(\varepsilon_0, \chi_x, \chi_y)`.  An entering zone is
    assumed to be cast onto the current member and to share its plane
    from that stage on; its locked-in datum is frozen at cast and is
    constant thereafter.  Two situations are therefore **out of
    scope** and raise: (i) a persisting zone whose datum changes
    mid-history (re-datuming of cured concrete), and (ii) two
    structurally independent bodies each carrying their own
    pre-connection load history on distinct planes, later made
    solidal — that requires per-body equilibria and a connection
    event (see the composability handoff in ``10_6``).
    """
    if x_ref is None:
        x_ref = float(base_section.x_centroid)
    if y_ref is None:
        y_ref = float(base_section.y_centroid)

    n_reb = int(base_section.x_rebars.size)
    n_ten = int(getattr(base_section, "x_tendons",
                        np.empty(0)).size)
    n_union = n_reb + n_ten
    n_fib = int(base_section.n_fibers)

    # Zone bookkeeping for the strain field.
    n_zones = 1 + len(getattr(base_section, "bulk_materials", []))
    mat_indices = getattr(base_section, "mat_indices", None)
    if mat_indices is None:
        mat_indices = np.zeros(n_fib, dtype=int)
    mat_idx_t = getattr(base_section, "mat_indices_tendon", None)
    if mat_idx_t is None:
        mat_idx_t = np.zeros(n_ten, dtype=int)

    # ---- Accumulators -------------------------------------------
    # One accumulator, not two: the per-zone affine strain field IS
    # the per-fibre strain, evaluated at the fibre.
    strains = {
        "e0": np.zeros(n_zones, dtype=float),
        "ex": np.zeros(n_zones, dtype=float),
        "ey": np.zeros(n_zones, dtype=float),
    }
    # Which zones have entered the resistance history.  A not-yet-cast
    # zone stays False so its zero cannot spoof the stress extremes.
    zone_present = np.zeros(n_zones, dtype=bool)
    eps_union = np.full(n_union, np.nan, dtype=float)
    mat_union = [None] * n_union
    present = np.zeros(n_union, dtype=bool)

    prev_state = None
    prev_resist = None
    F_prev = (0.0, 0.0, 0.0)
    stage_results = []
    all_etas = []          # (eta, stage_name, check_name)
    failed_checks = []     # (eta or None, stage_name, check_name)
    all_verified = True

    for k, stage in enumerate(stages):
        if "name" not in stage or "state" not in stage:
            raise ValueError(
                f"Stage {k}: every stage dict requires 'name' and "
                f"'state'."
            )
        name = stage["name"]
        state = stage["state"]
        if not isinstance(state, SectionState):
            raise ValueError(
                f"Stage '{name}': 'state' must be a SectionState, "
                f"got {type(state).__name__}."
            )
        if "moduli" in stage:
            # Phase 7 (D3) offered this as the ageing hook.  Under
            # strain accumulation it is demonstrably wrong: under
            # sustained load with a stiffening modulus and no creep the
            # strain does not move, so the accumulated strain does not
            # move, and reading a new law at the old strain would
            # report a stress that changed while nothing physical did.
            # Ageing is not a modulus swap; it is imposed strain plus a
            # law at the age, which is what the rheological provider
            # produces.  Removed and not deprecated: measured across
            # the repository, no caller ever passed the key.
            raise ValueError(
                f"Stage '{name}': a per-stage 'moduli' override is no "
                f"longer accepted.  It expressed ageing as a modulus "
                f"swap, which the strain accumulation of Phase 28 "
                f"cannot honour: at constant load a stiffer modulus "
                f"would move the reported stress while no strain "
                f"moved.  Declare the age on the rheological provider "
                f"and let it hand back service_law(t) together with "
                f"the imposed strain, then pass the resulting "
                f"materials in the global 'moduli' mapping."
            )
        # ---- Composite bulk staging (C4) ----------------------
        zone_active = getattr(state, "bulk_active", None)
        zone_active_prev = (None if prev_state is None
                            else getattr(prev_state, "bulk_active",
                                         None))
        state_planes = getattr(state, "bulk_planes", None)
        resist = _resist_mask(state, n_union)

        # ---- Demand walk ----------------------------------------
        dN, dMx, dMy = (float(v) for v in
                        stage.get("increment", (0.0, 0.0, 0.0)))
        for act in stage.get("prestress_actions", []):
            aN, aMx, aMy = act.triple()
            dN += aN
            dMx += aMx
            dMy += aMy
        F_cum = (F_prev[0] + dN, F_prev[1] + dMx, F_prev[2] + dMy)

        # ---- Transition classification --------------------------
        if k == 0:
            entering = np.nonzero(resist)[0]
            eps_changed = False
        else:
            leaving = prev_resist & ~resist
            if leaving.any():
                raise NotImplementedError(
                    f"Stage '{name}': elements leave the resistance "
                    f"set (union indices "
                    f"{np.nonzero(leaving)[0].tolist()}).  SLS "
                    f"accumulation across deactivation/de-bonding is "
                    f"deferred (compensating-release demand "
                    f"semantics)."
                )
            persisting = prev_resist & resist
            entering = np.nonzero(resist & ~prev_resist)[0]
            d_eps = (np.asarray(state.eps_init, dtype=float)
                     - np.asarray(prev_state.eps_init, dtype=float))
            eps_changed = bool(
                np.any(d_eps[persisting] != 0.0)
                or float(state.bulk_eps_init)
                != float(prev_state.bulk_eps_init)
            )
            if eps_changed and entering.size:
                raise ValueError(
                    f"Stage '{name}': compound transition — elements "
                    f"enter the resistance set while initial strains "
                    f"change on persisting elements.  Split into two "
                    f"stages (Phase-7 transition taxonomy)."
                )

        # ---- Views, solver, materials ---------------------------
        modmap = resolve_sls_moduli(base_section, dict(moduli or {}))
        # Cheap always-on cover check (rule 1): the map is resolved on
        # the base section, so a mismatch means the section changed
        # under the walk.
        modmap.require_cover(
            n_zones, n_union,
            where=f"verify_sls_staged, stage '{name}'")
        zone_materials = list(modmap.bulk)
        zone_E = np.array(modmap.E_bulk, dtype=float)

        view_k = materialize_view(base_section, state)
        slsv_k = sls_view(view_k, modmap)
        solver_k = FiberSolver(slsv_k, x_ref=x_ref, y_ref=y_ref)

        sol_hi = _solve_linear(solver_k, F_cum, tol, max_iter,
                               f"stage '{name}', cumulative demand")
        (u_idx, e_eps_hi, e_mats,
         plane_hi) = _element_states(solver_k, sol_hi, slsv_k)

        if k == 0:
            zone_entering = (np.ones(n_zones, dtype=bool)
                             if zone_active is None else
                             np.asarray(zone_active, dtype=bool))
        elif zone_active is None:
            zone_entering = np.zeros(n_zones, dtype=bool)
        else:
            za_prev = (np.ones(n_zones, dtype=bool)
                       if zone_active_prev is None
                       else np.asarray(zone_active_prev, dtype=bool))
            zone_entering = np.asarray(zone_active, dtype=bool) \
                & ~za_prev

        if k == 0:
            # Total read: every present zone comes into existence
            # carrying its initial strains; no prior history.
            _strain_entering_set(
                strains, plane_hi, state_planes,
                zone_entering, x_ref, y_ref,
                d_bulk_eps=float(state.bulk_eps_init))
            eps_union[u_idx] = e_eps_hi
            present[u_idx] = True
            for j, u in enumerate(u_idx):
                mat_union[int(u)] = e_mats[j]
        else:
            sol_lo = _solve_linear(
                solver_k, F_prev, tol, max_iter,
                f"stage '{name}', carried demand")
            (u_idx_lo, e_eps_lo, _e_mats_lo,
             plane_lo) = _element_states(solver_k, sol_lo, slsv_k)

            # Demand term (initial strains cancel; applies to every
            # element of the current view).
            d_elem = e_eps_hi - e_eps_lo
            # Persisting zones accumulate the plane increment; the
            # per-zone datum cancels here (it is constant once cast).
            zone_persist = ~zone_entering
            # Invariant (C4): a cast zone's datum is frozen.  A change
            # on a persisting zone is a re-datuming or a multi-body
            # merge — neither is modelled here.  Fail loud.
            _prev_planes = getattr(prev_state, "bulk_planes", None)
            if state_planes is not None and _prev_planes is not None:
                _sp = np.asarray(state_planes, dtype=float)
                _pp = np.asarray(_prev_planes, dtype=float)
                _pers = np.nonzero(zone_persist)[0]
                _bad = [int(z) for z in _pers
                        if not np.allclose(_sp[z], _pp[z],
                                           rtol=0.0, atol=1e-12)]
                if _bad:
                    raise NotImplementedError(
                        f"Stage '{name}': the locked-in datum of "
                        f"persisting bulk zone(s) {_bad} changed since "
                        f"the previous stage.  A cast zone's datum is "
                        f"frozen; a mid-history change is either "
                        f"re-datuming of cured concrete or a "
                        f"multi-body merge, neither of which the "
                        f"single-compatibility-plane SLS walk models "
                        f"(see composability handoff, 10_6)."
                    )
            _strain_increment(strains, plane_hi, plane_lo,
                              x_ref, y_ref, zone_mask=zone_persist)

            # State term (loss redistribution) — eps-only
            # transitions, evaluated on the *current* view.
            if eps_changed:
                view_pe = materialize_view(base_section, prev_state)
                slsv_pe = sls_view(view_pe, modmap)
                solver_pe = FiberSolver(slsv_pe, x_ref=x_ref,
                                        y_ref=y_ref)
                sol_pe = _solve_linear(
                    solver_pe, F_prev, tol, max_iter,
                    f"stage '{name}', previous-eps reference")
                (u_idx_pe, e_eps_pe, _m_pe,
                 plane_pe) = _element_states(solver_pe, sol_pe,
                                             slsv_pe)
                if not np.array_equal(u_idx_lo, u_idx_pe):
                    raise RuntimeError(
                        "eps-only transition produced mismatched "
                        "element sets — masks were expected equal."
                    )
                d_elem += e_eps_lo - e_eps_pe
                _strain_increment(
                    strains, plane_lo, plane_pe,
                    x_ref, y_ref,
                    d_bulk_eps=(float(state.bulk_eps_init)
                                - float(prev_state.bulk_eps_init)),
                    zone_mask=zone_persist)

            # Entering zones initialise from the stage's total read
            # (their datum is already carried by plane_hi through
            # bulk_planes_active).
            _strain_entering_set(
                strains, plane_hi, state_planes,
                zone_entering, x_ref, y_ref,
                d_bulk_eps=float(state.bulk_eps_init))
            # Persisting elements accumulate the increment; entering
            # elements initialise from the stage's total read.
            enter_set = set(entering.tolist())
            for j, u in enumerate(u_idx):
                u = int(u)
                if u in enter_set:
                    eps_union[u] = e_eps_hi[j]
                    present[u] = True
                else:
                    eps_union[u] += d_elem[j]
                mat_union[u] = e_mats[j]

        za_now = (np.ones(n_zones, dtype=bool) if zone_active is None
                  else np.asarray(zone_active, dtype=bool))
        zone_present |= za_now

        # ---- Stage record ---------------------------------------
        present_zones = [int(z) for z in np.nonzero(zone_present)[0]]
        if not present_zones:
            raise RuntimeError(
                f"Stage '{name}': no active bulk fiber in the SLS "
                f"history (empty composite state)."
            )
        # Fibre-level stresses, read from the one accumulator through
        # each zone's own law.
        eps_fib = (strains["e0"][mat_indices]
                   + strains["ex"][mat_indices] * base_section.x_fibers
                   + strains["ey"][mat_indices] * base_section.y_fibers)
        S_bulk = np.zeros(n_fib, dtype=float)
        for z in present_zones:
            sel = (mat_indices == z)
            if np.any(sel):
                S_bulk[sel] = np.asarray(
                    zone_materials[z].stress_array(eps_fib[sel]),
                    dtype=float)
        fib_present = zone_present[mat_indices]
        idx_present = np.nonzero(fib_present)[0]
        if idx_present.size == 0:
            raise RuntimeError(
                f"Stage '{name}': no active bulk fiber in the SLS "
                f"history (empty composite state)."
            )
        i_min = int(idx_present[np.argmin(S_bulk[idx_present])])
        i_max = int(idx_present[np.argmax(S_bulk[idx_present])])
        # F6: the reported extremes are the TRUE extreme-fibre stresses,
        # from the closed form on the strain field, not the outermost
        # fibre's centroid value (which is inset half a fibre and
        # under-reports, non-conservatively, the very stress EN 1992-1-1
        # 7.2 limits).
        (_s_min, _at_min,
         _s_max, _at_max) = _extreme_fibre(base_section, strains,
                                           zone_materials,
                                           present_zones)
        _by_zone = _extreme_fibre_by_zone(base_section, strains,
                                          zone_materials,
                                          present_zones)
        S_union = np.full(n_union, np.nan, dtype=float)
        elements = []
        for u in np.nonzero(present)[0]:
            u = int(u)
            S_union[u] = _sigma_of(mat_union[u], float(eps_union[u]))
            if u < n_reb:
                kind, x_e, y_e = ("rebar",
                                  float(base_section.x_rebars[u]),
                                  float(base_section.y_rebars[u]))
            else:
                i_t = u - n_reb
                kind = "tendon"
                x_e = float(base_section.x_tendons[i_t])
                y_e = float(base_section.y_tendons[i_t])
            elements.append({
                "union_index": u, "kind": kind,
                "x": x_e, "y": y_e,
                "eps": float(eps_union[u]),
                "sigma_MPa": round(float(S_union[u]), 6),
            })

        sr = {
            "name": name,
            "time": stage.get("time"),
            "increment": {"N_kN": round(dN / 1e3, 4),
                          "Mx_kNm": round(dMx / 1e6, 4),
                          "My_kNm": round(dMy / 1e6, 4)},
            "cumulative": {"N_kN": round(F_cum[0] / 1e3, 4),
                           "Mx_kNm": round(F_cum[1] / 1e6, 4),
                           "My_kNm": round(F_cum[2] / 1e6, 4)},
            "plane": {"eps0": plane_hi[0], "chi_x": plane_hi[1],
                      "chi_y": plane_hi[2]},
            "concrete": {
                # Unrounded for the same reason as ``by_zone`` above:
                # on a single-zone section these two records describe
                # the same quantity, and publishing one rounded and
                # one not would make them disagree in the last digits
                # for no reason a reader could reconstruct.
                "sigma_min_MPa": _s_min,
                "at_min": _at_min,
                "sigma_max_MPa": _s_max,
                "at_max": _at_max,
                # The global extremes are not enough on a composite: each
                # zone has its own law and its own f_ck, and the governing
                # compression may sit in one zone while the governing
                # tension sits in another.  GenSec's job is to hand the
                # engineer every number it has, not only the one it acted
                # on.
                "by_zone": _by_zone,
            },
            "elements": elements,
            # Plot-ready repackaging of the arrays already computed
            # above (S_bulk/eps_fib over idx_present, elements over
            # rebar union indices) in the shape
            # ``FiberSolver.get_fiber_results`` returns, plus the
            # stage's own materialised (active-only) geometry.  No
            # physics happens here: one home for one fact — the strain
            # field is the accumulator above, not a fresh solve — see
            # gensec.output.sls_state_plots.
            "fiber_field": {
                "bulk": {
                    "x": base_section.x_fibers[idx_present].copy(),
                    "y": base_section.y_fibers[idx_present].copy(),
                    "eps": eps_fib[idx_present].copy(),
                    "sigma": S_bulk[idx_present].copy(),
                    "dA": base_section.A_fibers[idx_present].copy(),
                },
                "rebars": {
                    "x": np.array([e["x"] for e in elements
                                  if e["kind"] == "rebar"], dtype=float),
                    "y": np.array([e["y"] for e in elements
                                  if e["kind"] == "rebar"], dtype=float),
                    "eps": np.array([e["eps"] for e in elements
                                     if e["kind"] == "rebar"],
                                    dtype=float),
                    "sigma": np.array([e["sigma_MPa"] for e in elements
                                       if e["kind"] == "rebar"],
                                      dtype=float),
                    "A": base_section.A_rebars[
                        [e["union_index"] for e in elements
                         if e["kind"] == "rebar"]].copy(),
                },
                "section_view": view_k,
            },
            "active_zones": _active_zone_labels(base_section,
                                                zone_active, n_zones),
        }

        # ---- Checks ---------------------------------------------
        limits = stage.get("limits")
        if limits is not None:
            if not isinstance(limits, StressLimits):
                raise ValueError(
                    f"Stage '{name}': 'limits' must be a "
                    f"StressLimits, got {type(limits).__name__}."
                )
            checks = []
            # F6, closed on the check path (Phase 29) and carried onto
            # the zone axis (Phase 32).  The value verified is the one
            # the record reports and the one the label belongs to: the
            # closed form over the zone's polygon, which is the extreme
            # fibre of the MEMBER.  It used to be S_bulk[i_min], the
            # stress at the centroid of the outermost fibre -- a point
            # of GenSec's discretisation and not of the member, inset
            # half a fibre and therefore non-conservative by an amount
            # the user controls with n_fibers_y.
            #
            # The section-wide ``sig_min`` is gone with Phase 32: with
            # one check per zone there is no quantity a single global
            # extreme could be the subject of.  ``sig_max`` went in
            # Phase 29, a residue of the tensile path Phase 27 removed.

            # The uncracked-basis flag is gone (Phase 27).  It existed
            # because the view was a straight line with no tensile
            # ceiling, so the solve could report a concrete carrying
            # more tension than any concrete carries, and the flag was
            # the only thing that said so.  With the service law in the
            # view that state is unreachable: the fibre releases at
            # f_ct,eff and the solve returns the cracked equilibrium
            # directly.
            #

            # Concrete compression, ONE CHECK PER PRESENT ZONE.
            #
            # Until Phase 32 this was a single check of the section's
            # global compressive extreme against a single section-wide
            # limit.  On a composite that is not an imprecise limit but
            # a false pass: the zones carry different concrete, hence
            # different f_ck, and one figure is checked against the
            # material of whichever zone happened to supply it.
            # Measured on the two-zone bench (C45 base, C30 topping,
            # Mx = -350 kNm): the topping reached 22.5753 MPa, passed
            # against the base's 27.0, and the stage was reported
            # verified while that zone stood at 125 % of what its own
            # material prescribes.
            #
            # The per-zone extremes are NOT recomputed here.  They are
            # ``_by_zone``, already assembled above by
            # ``_extreme_fibre_by_zone`` and already published in the
            # record: one home for one fact, and it is what makes the
            # record and the verdict provably the same number.
            zone_declared = _zone_declared_materials(base_section)
            for z in present_zones:
                z = int(z)
                zname = (base_section.zone_names[z]
                         if z < len(base_section.zone_names)
                         else f"zone[{z}]")
                cname = f"concrete_compression[{zname}]"
                zrec = _by_zone.get(zname)
                if zrec is None:
                    # A present zone with no region of its own -- fully
                    # covered by overlays.  It owes a record all the
                    # same: obligation B4 says every branch leaves one.
                    checks.append(_skipped_entry(
                        cname,
                        "the zone has no material region in this "
                        "section (its polygon is entirely covered by "
                        "overlaying zones), so it has no extreme fibre "
                        "to check"))
                    continue
                if limits.k_c is None:
                    checks.append(_skipped_entry(
                        cname,
                        "no k_c coefficient declared; the concrete "
                        "limit is k_c * f_ck(zone) and the coefficient "
                        "is the combination's, which GenSec cannot "
                        "choose"))
                    continue
                mat_f_ck = (getattr(zone_declared[z], "fck", None)
                            if z < len(zone_declared) else None)
                f_ck, provenance = limits.strength_for(zname, mat_f_ck)
                if f_ck is None:
                    checks.append(_skipped_entry(
                        cname,
                        f"the material of zone '{zname}' declares no "
                        f"f_ck and no strength is declared for it in "
                        f"'custom_zone_strengths'; k_c has nothing to "
                        f"multiply"))
                    continue
                # The coefficient multiplies the strength exactly once,
                # whichever branch produced it.  There is no path on
                # which a product enters and is multiplied again.
                checks.append(_check_entry(
                    cname,
                    max(0.0, -float(zrec["sigma_min_MPa"])),
                    float(limits.k_c) * f_ck,
                    extra={"at": zrec["at_min"],
                           "zone": zname,
                           # Who produced the strength, not whether it
                           # complies: GenSec cannot tell an f_ck(t)
                           # computed under EN 1992-1-1 3.1.2(6) from an
                           # invented number, so the label claims only
                           # what can be known.
                           "provenance": provenance,
                           "f_ck_MPa": f_ck,
                           "k_c": float(limits.k_c)}))

            # Reinforcing steel (tension).
            # A declared limit always leaves a record.  Before Phase 30
            # the whole block below sat inside ``if reb_present.any()``
            # with no ``else``, so a stage with no rebar present lost
            # the check and gained nothing: an unrun check and a passed
            # check were indistinguishable in the output.  The absence
            # of steel is a perfectly ordinary state -- a precast stage
            # before the rebar cage is active, a strand-only slab --
            # and it is exactly the state a reader most needs told.
            reb_present = present[:n_reb]
            if limits.sigma_s_tens is None:
                checks.append(_skipped_entry(
                    "steel_tension", "no sigma_s_tens limit"))
            elif not reb_present.any():
                checks.append(_skipped_entry(
                    "steel_tension",
                    "no reinforcing steel present at this stage"))
            else:
                val = max(0.0,
                          float(np.max(S_union[:n_reb]
                                       [reb_present])))
                checks.append(_check_entry(
                    "steel_tension", val, limits.sigma_s_tens))

            # Bonded tendons.  Same obligation, same reason: the
            # decompression check a few lines below already emitted a
            # skipped entry when it found no bonded tendon to work on,
            # so two checks in one block were treating the same absence
            # in two different ways.  An all-unbonded stage is the
            # common case that lost its record here; its member-level
            # stress, when the caller declares one, is checked in the
            # loop that follows under ``tendon_stress[label]``.
            ten_present = present[n_reb:]
            if limits.sigma_p_max is None:
                checks.append(_skipped_entry(
                    "tendon_stress", "no sigma_p_max limit"))
            elif not ten_present.any():
                checks.append(_skipped_entry(
                    "tendon_stress",
                    "no bonded prestressing steel present at this "
                    "stage; declare 'unbonded_sigma_p' to have a "
                    "member-level tendon stress checked instead"))
            else:
                val = float(np.max(S_union[n_reb:][ten_present]))
                checks.append(_check_entry(
                    "tendon_stress", max(0.0, val),
                    limits.sigma_p_max))

            # Unbonded tendons: member-level values from the caller.
            for label, sp in (stage.get("unbonded_sigma_p")
                              or {}).items():
                if limits.sigma_p_max is not None:
                    checks.append(_check_entry(
                        f"tendon_stress[{label}]",
                        max(0.0, float(sp)), limits.sigma_p_max,
                        extra={"provenance": "member_level"}))
                else:
                    checks.append(_skipped_entry(
                        f"tendon_stress[{label}]",
                        "no sigma_p_max limit"))

            # Decompression (per bonded tendon of this stage).
            if limits.decompression:
                ten_union = np.nonzero(resist[n_reb:])[0]
                if ten_union.size == 0:
                    checks.append(_skipped_entry(
                        "decompression",
                        "no bonded tendons at this stage"))
                for i_t in ten_union:
                    z_t = int(mat_idx_t[i_t])
                    probe = _decompression_probe(
                        base_section.x_tendons[i_t],
                        base_section.y_tendons[i_t],
                        z_t, strains, zone_materials[z_t],
                        zone_E[z_t], limits.c_dec)
                    # One verdict, one number, and it is the number
                    # the verdict was taken on (Phase 29, Q3-E').
                    # The record used to publish sigma_probe_MPa
                    # beside eps_probe: on a cracking law the two
                    # disagree the moment the probe point cracks,
                    # because the law returns zero past eps_ct, so
                    # a reader taking sigma <= 0 as the criterion
                    # reads PASS where the engine reads FAIL.
                    #
                    # value and limit are strains, and there is no
                    # eta: decompression is the one check with no
                    # capacity to divide by.  _DECOMPRESSION_TOL is
                    # solver round-off, not a resistance, and a
                    # ratio against it is a division by zero in
                    # disguise -- measured at 5.4e+09 on this very
                    # probe.  The governing selection below is what
                    # was repaired instead.
                    eps_tol = _DECOMPRESSION_TOL / zone_E[z_t]
                    checks.append({
                        "name": f"decompression[tendon {i_t}]",
                        "value": probe["eps_probe"],
                        "limit": eps_tol,
                        "eps_probe": probe["eps_probe"],
                        "probe_at": (round(probe["x_probe"], 3),
                                     round(probe["y_probe"], 3)),
                        "c_dec_mm": limits.c_dec,
                        "passed": probe["passed"],
                    })

            # Verdict.  A check passes or it does not; there is no
            # third state in which a number is reported and
            # simultaneously disowned.
            stage_ok = True
            for c in checks:
                if c.get("skipped"):
                    continue
                if not c["passed"]:
                    stage_ok = False
                if "eta" in c:
                    all_etas.append((c["eta"], name, c["name"]))
                if not c["passed"]:
                    failed_checks.append(
                        (c.get("eta"), name, c["name"]))
            sr["checks"] = checks
            sr["limits"] = limits.name
            sr["verified"] = stage_ok
            all_verified = all_verified and stage_ok

        stage_results.append(sr)
        prev_state = state
        prev_resist = resist
        F_prev = F_cum

    # The governing check is one that actually governed.  Before
    # Phase 29 the maximum was taken over the utilisations alone,
    # so a check carrying no eta -- decompression is the only one,
    # because it is the only one with no capacity to divide by --
    # could never be named.  A run that FAILED decompression then
    # reported concrete_compression, a check that PASSED, at eta
    # 0.353984, as governing.
    #
    # A failed check governs over a passing one whatever their
    # utilisations; among failed checks the largest eta wins, and
    # a failed check without an eta still outranks every check
    # that passed.  eta is None where the check has none, rather
    # than a number invented so the field is never empty: a field
    # filled for completeness is what produced F6.
    governing = None
    if failed_checks:
        eta, s_name, c_name = max(
            failed_checks,
            key=lambda t: (t[0] is not None, t[0] or 0.0))
        governing = {"eta": eta, "stage": s_name,
                     "check": c_name, "failed": True}
    elif all_etas:
        eta, s_name, c_name = max(all_etas, key=lambda t: t[0])
        governing = {"eta": eta, "stage": s_name,
                     "check": c_name, "failed": False}

    return {
        "type": "sls_staged",
        "x_ref": x_ref, "y_ref": y_ref,
        "stages": stage_results,
        "verified": bool(all_verified),
        "governing": governing,
    }
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=5d969340828858108817edc7884654600ac5ecdff2d288995473d8b47850351d body_lines=3281
