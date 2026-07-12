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
Time-dependent prestress losses — the age-adjusted (AAEM) container.

This module is **mechanics**; it holds no normative content.  Every
code-specific formula lives behind
:class:`~gensec.materials.base.RheologicalModel`
(:mod:`gensec.materials.rheology`); the container consumes only the four
abstract functions and derives the ageing coefficient :math:`\chi`
itself, from the provider's compliance.

The engine
----------
A **sectional** AAEM: one 3×3 linear system on the age-adjusted
transformed section, per interval.  Deliberately *not* the closed form
of EN 1992-1-1 (5.46): that expression is undefined on a composite
section — *which* :math:`A_c`, *which* :math:`I_c`, *which*
:math:`\varphi`, with two concrete zones of different age, creep and
shrinkage? — and a composite deck is precisely what Phase 8 exists to
model.  (5.46) is implemented here as an **independent** cross-check
(:func:`ec2_5106_closed_form`); on a single-zone, single-tendon section
the two agree to 0.04 %.

Over an interval :math:`[t_0, t]` the external demand is **frozen**
(the AAEM premise: a sustained quasi-permanent action).  The
constitutive law of the increment is, per concrete fibre of zone *z*,

.. math::

    \Delta\sigma_z = \bar{E}_z\,
        \bigl(\Delta\varepsilon - \varepsilon_{\mathrm{imp},z}\bigr),
    \qquad
    \bar{E}_z = \frac{E_{c,z}(t_0)}{1 + \chi_z \varphi_z} ,

.. math::

    \varepsilon_{\mathrm{imp},z}(x,y) =
        \underbrace{\varphi_z\,
        \varepsilon_{\mathrm{mech},z}(x,y;t_0)}_{\text{creep — a plane}}
        + \underbrace{\varepsilon_{\mathrm{sh},z}}_{\text{shrinkage —
        uniform}} ,

with every point element (bonded tendon, embedded rebar) entering
*differentially* against the bulk it displaces, and the tendons carrying
their intrinsic relaxation :math:`\Delta\sigma_{pr}` (reduced by the
container: :attr:`LossModel.relaxation_reduction`).  Writing the strain
increment as :math:`\Delta\varepsilon_i = \mathbf{b}_i\cdot\mathbf{u}`
with :math:`\mathbf{b}_i = [1,\;\ell_{y,i},\;-\ell_{x,i}]` and
:math:`\mathbf{u} = (\Delta\varepsilon_0,\Delta\chi_x,\Delta\chi_y)` —
the *same* basis in which the integrator forms
:math:`(N,M_x,M_y) = \sum_i \sigma_i A_i \mathbf{b}_i` — and enforcing
:math:`\Delta N = \Delta M_x = \Delta M_y = 0` gives a symmetric,
work-conjugate system:

.. math::

    \mathbf{K}\mathbf{u} = \mathbf{f},
    \qquad
    \mathbf{K} = \sum_i \tilde{E}_i A_i\,
                 (\mathbf{b}_i \otimes \mathbf{b}_i),

.. math::

    \mathbf{f} = \sum_{\mathrm{fib}} A_i \mathbf{b}_i\,
                    \bar{E}_z \varepsilon_{\mathrm{imp},z}
               - \sum_{\mathrm{pt}} A_j \mathbf{b}_j
                    \bigl(\Delta\sigma_{pr,j}
                          + \bar{E}_z \varepsilon_{\mathrm{imp},z}\bigr),

:math:`\tilde{E}` being :math:`\bar{E}_z` on a concrete fibre and
:math:`(E_{\mathrm{pt}} - \bar{E}_z)` on a point element.

The emission theorem
--------------------
The container emits a **strain state**, reconciled at the
end-of-interval plane exactly as
:meth:`~gensec.solver.timeline.ConstructionTimeline._reconcile_grout`
does:

.. math::

    \varepsilon_{\mathrm{init},j}(t)
        = \frac{\sigma_{p,j}(t)}{E_{p,j}}
          - \varepsilon_{\mathrm{sec},j}(t)
        = \varepsilon_{\mathrm{init},j}(t_0)
          + \frac{\Delta\sigma_{pr,j}}{E_{p,j}} .

**The tendon's own prestrain changes only by its relaxation.**  Not an
approximation — an identity: substituting
:math:`\Delta\sigma_{p,j} = E_{p,j}\,\mathbf{b}_j\!\cdot\!\mathbf{u}
+ \Delta\sigma_{pr,j}` and
:math:`\varepsilon_{\mathrm{sec},j}(t) =
\varepsilon_{\mathrm{sec},j}(t_0) + \mathbf{b}_j\!\cdot\!\mathbf{u}`
cancels the plane term.  Creep and shrinkage are properties of the
*concrete*; they reach the tendon through the section plane, which moves
because the concrete carries an eigenstrain — emitted as a per-zone
**plane increment** (all three components: creep under a linear stress
field is a linear imposed strain):

.. math::

    \boldsymbol{\beta}_z = -\,\varepsilon_{\mathrm{imp},z} .

The minus sign is not a convention.  The fiber kernel evaluates the bulk
law at :math:`\mathrm{law}(\varepsilon_{\mathrm{sec}} +
\varepsilon_{\mathrm{bulk}})` while the physics is
:math:`E(\varepsilon_{\mathrm{tot}} - \varepsilon_{\mathrm{imp}})`.  Its
falsifiable consequence: a fully restrained shrinking member develops
**tension** — which is why composite toppings crack.

Fail-loud boundaries
--------------------
- **Bonded tendons only.**  An unbonded or external tendon's stress
  change is governed by the *member-average* strain over its free
  length: a beam-level quantity a section library does not have.  The
  container raises rather than silently substitute the local value.
- **Linear viscoelasticity only.**  :math:`J`, :math:`\chi` and the
  whole AAEM presume a stress-independent compliance.  The container
  asks the provider for its
  :meth:`~gensec.materials.base.RheologicalModel.linearity_limit` and
  raises beyond it (EN 1992-1-1 §3.1.4(4): :math:`0.45 f_{ck}`).
- **Sub-quantum steps.**  A step-by-step integration whose per-step
  strain increment falls below
  :data:`~gensec.solver.section_state.QUANT_EPS` would be silently
  collapsed onto one domain build by the state hash.  Detected; raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..materials.base import RheologicalModel, aging_coefficient
from .integrator import FiberSolver
from .section_state import QUANT_EPS, materialize_view
from .sls import resolve_sls_moduli, sls_view

__all__ = [
    "EC2_LUMP_CHI",
    "EC2_LUMP_RELAXATION_REDUCTION",
    "LossModel",
    "IntervalLosses",
    "compute_interval_losses",
    "expand_losses",
    "ec2_5106_closed_form",
]

#: EN 1992-1-1 (5.46) writes 0.8 twice, and both are **container** knobs,
#: not provider constants.  The denominator's is the AAEM ageing
#: coefficient :math:`\chi` — which
#: :func:`~gensec.materials.base.aging_coefficient` shows is what EC2's
#: own compliance produces (0.76–0.86 over the usual range, centred on
#: 0.80).  The numerator's is the reduction of the intrinsic relaxation
#: caused by the tendon shortening with the concrete.
EC2_LUMP_CHI: float = 0.8
EC2_LUMP_RELAXATION_REDUCTION: float = 0.8

#: Placeholder modulus [MPa] handed to a concrete zone that has **not yet
#: been cast** at an interval.  Its fibres are masked out of the stage
#: view, so the value cannot reach any result -- an invariant the
#: validator asserts by perturbing it six orders of magnitude and
#: checking the answer does not move.  It exists only because
#: :func:`~gensec.solver.sls.resolve_sls_moduli` enumerates every
#: material of the *base* section, cast or not.
_UNCAST_PLACEHOLDER_E: float = 1.0


# ==================================================================
#  Specification
# ==================================================================

@dataclass(frozen=True)
class LossModel:
    r"""
    A rheological provider plus the container's mechanics knobs.

    Parameters
    ----------
    provider : RheologicalModel
        The normative provider, carrying **material** parameters only.
        Drying geometry is bound per *zone* and the prestressing steel
        per *tendon*, by the container.
    chi : {'lump', 'from_J'} or float, optional
        The AAEM ageing coefficient.

        - ``'lump'`` (default) — :math:`\chi = 0.8`, the value
          EN 1992-1-1 (5.46) assumes.  With the EC2 provider this
          default is Eurocode-conforming.
        - ``'from_J'`` — computed per zone and per interval from the
          provider's own compliance
          (:func:`~gensec.materials.base.aging_coefficient`).  The
          rigorous AAEM.
        - a float — an explicit value.

        On the EC2 provider the two differ by <0.1 % of the loss: the
        lump value is an excellent approximation *because it is what
        EC2's compliance produces*.  That agreement is the free
        consistency check of the architecture, not a coincidence.
    relaxation_reduction : float, optional
        Factor applied to the provider's **intrinsic**
        :math:`\Delta\sigma_{pr}`: a tendon that shortens with the
        concrete relaxes less than one held at constant strain.  Default
        :data:`EC2_LUMP_RELAXATION_REDUCTION` = 0.8 — EC2's own lump
        value, held **here**, in the container, so the provider never
        sees it and an ACI or AASHTO user sets their own.  ``1.0``
        disables the reduction.

        .. note::

            A fully mechanical reduction (iterating the coupled
            equilibrium so the relaxation is evaluated on the decaying
            stress path) is a refinement of this combinator, explicitly
            deferred by ``7_0-GENSEC_PHASE5_PRIMER.md`` §13.  The knob
            is the honest interim: a number the engineer sets, not a
            constant buried inside a formula.
    n_steps : int, optional
        Log-grid resolution of the Volterra inversion when
        ``chi='from_J'``.  Default 100 (:math:`\chi` converged to
        :math:`1.4\times 10^{-4}`).
    steps : sequence of float, optional
        Opt-in step-by-step integration: intermediate **ages** [days]
        subdividing the interval, each sub-step its own AAEM solve on
        the updated state.  Default ``None`` — one AAEM step over the
        whole interval, the classical (and for a monotone loss history
        accurate) treatment.  Opens the sub-quantum trap; see
        :func:`_check_subquantum`.
    name : str, optional
        Identifier, used in reports and error messages.

    Raises
    ------
    ValueError
        Malformed ``chi`` or ``relaxation_reduction``; non-monotone
        ``steps``.
    """

    provider: RheologicalModel
    chi: Union[str, float] = "lump"
    relaxation_reduction: float = EC2_LUMP_RELAXATION_REDUCTION
    n_steps: int = 100
    steps: Optional[Sequence[float]] = None
    name: str = ""

    def __post_init__(self):
        if not isinstance(self.provider, RheologicalModel):
            raise ValueError(
                f"LossModel '{self.name}': provider must be a "
                f"RheologicalModel, got {type(self.provider).__name__}."
            )
        if isinstance(self.chi, str):
            if self.chi not in ("lump", "from_J"):
                raise ValueError(
                    f"LossModel '{self.name}': chi must be 'lump' (0.8, "
                    f"EN 1992-1-1 (5.46)), 'from_J' (computed from the "
                    f"provider's compliance), or an explicit float; got "
                    f"{self.chi!r}."
                )
        elif not 0.0 < float(self.chi) <= 1.5:
            raise ValueError(
                f"LossModel '{self.name}': an explicit chi must lie in "
                f"(0, 1.5]; got {self.chi!r}."
            )
        if not 0.0 <= float(self.relaxation_reduction) <= 1.0:
            raise ValueError(
                f"LossModel '{self.name}': relaxation_reduction must lie "
                f"in [0, 1] (1.0 = the full intrinsic relaxation, 0.8 = "
                f"the EN 1992-1-1 lump); got "
                f"{self.relaxation_reduction!r}."
            )
        if self.steps is not None:
            s = [float(v) for v in self.steps]
            if any(b <= a for a, b in zip(s, s[1:])):
                raise ValueError(
                    f"LossModel '{self.name}': 'steps' must be strictly "
                    f"increasing ages [days]; got {list(self.steps)!r}."
                )

    def chi_at(self, provider, t, t0) -> float:
        r"""
        Resolve :attr:`chi` for one zone and one interval.

        Parameters
        ----------
        provider : RheologicalModel
            The **geometry-bound** provider of the zone.
        t, t0 : float
            Interval ends, as ages **of that zone** [days].

        Returns
        -------
        float
        """
        if self.chi == "lump":
            return EC2_LUMP_CHI
        if self.chi == "from_J":
            return aging_coefficient(provider, t, t0, n_steps=self.n_steps)
        return float(self.chi)


# ==================================================================
#  Result
# ==================================================================

@dataclass
class IntervalLosses:
    r"""
    The AAEM end-of-interval state, and the section ops that encode it.

    Attributes
    ----------
    u : numpy.ndarray
        Strain-plane increment
        :math:`(\Delta\varepsilon_0, \Delta\chi_x, \Delta\chi_y)`
        [-, 1/mm, 1/mm].
    sigma_p0, sigma_p : dict
        ``{tendon_name: stress [MPa]}`` at the start and end.
    d_sigma_p : dict
        ``{tendon_name: total loss [MPa]}`` — signed, negative.
    d_sigma_pr : dict
        ``{tendon_name: applied (reduced) relaxation [MPa]}`` — the part
        intrinsic to the steel, and the *only* part that moves the
        tendon's prestrain.
    eps_override : dict
        ``{union_index: eps_init}`` — absolute post-loss prestrain of
        every bonded tendon.
    bulk_plane_delta : numpy.ndarray
        Shape ``(n_zones, 3)``: the per-zone eigenstrain plane increment
        :math:`\boldsymbol{\beta}_z`, to be **added** to
        ``SectionState.bulk_planes``.
    phi, chi : dict
        ``{zone: value}`` actually used.  Reported, never hidden — the
        engineer is entitled to see which creep coefficient and which
        ageing coefficient produced the number.
    sigma_c_tendon : dict
        ``{tendon_name: sigma_c [MPa]}``, the concrete stress at the
        tendon level at the start of the interval — the
        :math:`\sigma_{c,QP}` of EN 1992-1-1 (5.46), exposed for the
        closed-form cross-check.
    """

    u: np.ndarray
    sigma_p0: Dict[str, float] = field(default_factory=dict)
    sigma_p: Dict[str, float] = field(default_factory=dict)
    d_sigma_p: Dict[str, float] = field(default_factory=dict)
    d_sigma_pr: Dict[str, float] = field(default_factory=dict)
    eps_override: Dict[int, float] = field(default_factory=dict)
    bulk_plane_delta: Optional[np.ndarray] = None
    phi: Dict[int, float] = field(default_factory=dict)
    chi: Dict[int, float] = field(default_factory=dict)
    sigma_c_tendon: Dict[str, float] = field(default_factory=dict)
    d_sigma_zone: Optional[np.ndarray] = None
    eps_imp_zone: Optional[np.ndarray] = None
    sigma_zone0: Optional[np.ndarray] = None

    @property
    def loss_fraction(self) -> Dict[str, float]:
        r"""``{tendon_name: |Δσ_p| / σ_p0}`` [-]."""
        return {k: (abs(self.d_sigma_p[k] / v) if v else 0.0)
                for k, v in self.sigma_p0.items()}


# ==================================================================
#  Guards
# ==================================================================

def _check_linearity(provider, zone_name, t0_zone, sigma_c_min):
    r"""
    Fail loud if the concrete is stressed outside the provider's
    linear-viscoelastic range.

    Parameters
    ----------
    provider : RheologicalModel
        Geometry-bound provider of the zone.
    zone_name : str
        For the message.
    t0_zone : float
        Age of the zone at the start of the interval [days].
    sigma_c_min : float
        Most compressive fibre stress in the zone [MPa, signed].

    Raises
    ------
    ValueError
        If ``|sigma_c_min|`` exceeds the provider's
        :meth:`~gensec.materials.base.RheologicalModel.linearity_limit`.
    """
    lim = provider.linearity_limit(t0_zone)
    if lim is None:
        return
    if abs(min(0.0, float(sigma_c_min))) > float(lim):
        raise ValueError(
            f"losses: zone '{zone_name}' is compressed to "
            f"{abs(sigma_c_min):.2f} MPa at the start of the interval, "
            f"beyond the {float(lim):.2f} MPa linear-viscoelasticity "
            f"limit its rheological model declares (EN 1992-1-1 "
            f"§3.1.4(4): 0.45 f_ck(t0)).  Above it the creep compliance "
            f"is no longer stress-independent, so J, chi and the whole "
            f"age-adjusted formulation are out of range.  Non-linear "
            f"creep is not modelled: reduce the prestress, or supply a "
            f"provider whose linearity_limit covers the stress."
        )


def _check_subquantum(d_eps, label):
    r"""
    Fail loud on a step whose strain increment the domain cache would
    swallow.

    The staged domain cache buckets ``eps_init`` and the bulk planes on
    :data:`~gensec.solver.section_state.QUANT_EPS`.  A step whose largest
    strain change is below that quantum hashes to the **same** state as
    the one before it: the domain is reused, the step silently does
    nothing, and a 40-step integration quietly becomes a 1-step one —
    while still returning a plausible-looking number.

    Parameters
    ----------
    d_eps : array_like
        Every strain increment the step emits (tendon prestrains, and
        the :math:`\varepsilon_0` component of each bulk-plane delta).
    label : str
        Step identifier, for the message.

    Raises
    ------
    ValueError
        If ``max |d_eps| < QUANT_EPS``.
    """
    d = np.abs(np.asarray(list(d_eps), dtype=float))
    if d.size == 0:
        return
    peak = float(d.max())
    if peak < QUANT_EPS:
        raise ValueError(
            f"losses: step '{label}' emits a largest strain increment of "
            f"{peak:.3e}, below the domain-cache quantum QUANT_EPS = "
            f"{QUANT_EPS:.1e}.  The staged state hash would bucket this "
            f"step onto the previous one: the domain would be reused, "
            f"the step would silently do nothing, and a fine-grained "
            f"integration would collapse into a coarse one *while still "
            f"returning a plausible number*.  Either use fewer, coarser "
            f"steps (a single AAEM interval is the classical treatment, "
            f"and for a monotone loss history an accurate one), or lower "
            f"QUANT_EPS and accept the domain-cache miss rate that "
            f"follows."
        )


# ==================================================================
#  Section-side helpers
# ==================================================================

def _zone_material(base_section, z):
    r"""The bulk material of zone *z* (``0`` = ``bulk_material``)."""
    if int(z) == 0:
        return base_section.bulk_material
    return base_section.bulk_materials[int(z) - 1][1]


def _zone_drying_geometry(base_section, z) -> Tuple[float, float]:
    r"""
    :math:`(A_c, u)` of zone *z*: area [mm²] and the perimeter exposed
    to drying [mm].

    Taken as the zone polygon's area and full boundary length.  A zone
    whose faces are **not** all exposed — a topping cast onto a precast
    web dries only through its top and sides — has a smaller effective
    perimeter, hence a *larger* notional size and *less* shrinkage.
    Overriding it is the engineer's call: pass an explicitly bound
    provider (``provider.with_geometry(A_c, u)``) in the loss model.

    Parameters
    ----------
    base_section : GenericSection
    z : int
        Zone index.

    Returns
    -------
    tuple of float
        ``(A_c, u)``.
    """
    z = int(z)
    poly = (base_section.polygon if z == 0
            else base_section.bulk_materials[z - 1][0])
    return float(poly.area), float(poly.length)


def _tendon_steel(tendon):
    r"""
    Probe a tendon's material for :math:`f_{pk}` and the relaxation
    parameters.

    Uses the ``getattr`` degradation idiom: the constitutive
    :class:`~gensec.materials.steel.PrestressingSteel` carries only the
    design branch, while :func:`~gensec.materials.ec2_bridge.
    prestress_from_ec2` attaches the full
    :class:`~gensec.materials.prestress_properties.fbpren` as ``.ec2``.
    A tendon built without that bridge simply has no ``f_pk``, and the
    caller must say so — an intrinsic relaxation is a function of
    :math:`\mu = \sigma_{pi}/f_{pk}` and cannot be guessed.

    Parameters
    ----------
    tendon : Tendon

    Returns
    -------
    dict
        ``{'f_pk': float or None, 'relaxation_class': int or None,
        'rho_1000': float or None, 'Ep': float}``.
    """
    mat = tendon.material
    props = getattr(mat, "ec2", None)
    Ep = float(getattr(mat, "Ep", 0.0) or getattr(mat, "E", 0.0) or 0.0)
    return {
        "f_pk": getattr(props, "f_pk", None),
        "relaxation_class": getattr(props, "relaxation_class", None),
        "rho_1000": getattr(props, "rho_1000", None),
        "Ep": Ep,
    }


def _fpk_by_name(base_section, name):
    r"""
    Characteristic strength :math:`f_{pk}` [MPa] of the named tendon, or
    ``None``.

    Used to freeze :math:`\mu_0 = \sigma_{pi}/f_{pk}` at the first
    sub-step of a step-by-step integration: the relaxation law is
    parameterised by the stress ratio **at stressing**, and re-reading it
    from the decaying stress at every step would let the relaxation feed
    on its own decay.
    """
    for j, t in enumerate(getattr(base_section, "tendons", [])):
        if (getattr(t, "name", None) or f"tendon[{j}]") == name:
            return _tendon_steel(t)["f_pk"]
    return None


def _point_elements(base_section, state, solver, models, zone_provider,
                    tendon_ages, relax_from, p_sec, p_mech, E0_z, n_reb,
                    tendons):
    r"""
    Assemble every **active** point element (bonded tendon, embedded
    rebar) for the AAEM system.

    Each enters *differentially* against the bulk it displaces — the
    same net-of-displaced-bulk convention the fiber integrator uses
    (:math:`F = [\sigma_s(\varepsilon) - \sigma_b(\varepsilon)]\,A_s`).
    Rebars carry no eigenstrain of their own, but they **restrain** the
    concrete's, and that restraint is real: it is why heavily reinforced
    members crack under restrained shrinkage.

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState
    solver : FiberSolver
        Only its reference point is used.
    models : dict
        ``{zone: LossModel}``.
    zone_provider : dict
        ``{zone: geometry-bound RheologicalModel}``.
    tendon_ages : dict
        ``{tendon_name: duration under load at the interval end [days]}``.
    relax_from : dict
        ``{tendon_name: (T_start [days], mu_0)}``.  The relaxation of the
        step is the **increment** ``rho(T_end, mu_0) - rho(T_start,
        mu_0)``: relaxation is a function of the *total* time under load,
        so a cumulative value added to an ``eps_init`` that already
        carries it would double-count.
    p_sec : numpy.ndarray
        Section strain plane at :math:`t_0`, ``(eps0, chi_x, chi_y)``.
    p_mech : numpy.ndarray
        Shape ``(n_zones, 3)``: the mechanical (stress-producing) strain
        plane of every zone at :math:`t_0`.
    E0_z : numpy.ndarray
        Per-zone concrete modulus at :math:`t_0` [MPa].
    n_reb : int
        Number of rebars (the union-index offset of the tendons).
    tendons : list of Tendon

    Returns
    -------
    dict of numpy.ndarray
        ``A, lx, ly, zone, E, dsigma_pr, eps_mech, sigma_c, sigma_p0``,
        plus ``tendon_j`` (``-1`` for a rebar) and ``name``.

    Raises
    ------
    ValueError
        A tendon with no usable modulus, or with a loss model but no
        :math:`f_{pk}`.
    """
    xr, yr = float(solver.x_ref), float(solver.y_ref)
    A, lx, ly, zone, E = [], [], [], [], []
    dspr, e_mech, s_c, s_p0, tj, nm = [], [], [], [], [], []

    def _at(plane, x, y):
        return float(plane[0] + plane[1] * (y - yr) - plane[2] * (x - xr))

    for i, r in enumerate(base_section.rebars):
        if not state.active[i] or not r.embedded or r.x is None:
            continue
        z = int(base_section._material_index(r.x, r.y))
        A.append(float(r.As))
        lx.append(float(r.x) - xr)
        ly.append(float(r.y) - yr)
        zone.append(z)
        E.append(float(getattr(r.material, "E", 0.0) or 0.0))
        dspr.append(0.0)
        e_mech.append(_at(p_mech[z], r.x, r.y))
        s_c.append(float(E0_z[z]) * _at(p_mech[z], r.x, r.y))
        s_p0.append(0.0)
        tj.append(-1)
        nm.append(getattr(r, "name", None) or f"rebar[{i}]")

    for j, t in enumerate(tendons):
        ui = n_reb + j
        if not state.active[ui]:
            continue
        z = int(base_section._material_index(t.x, t.y))
        steel = _tendon_steel(t)
        Ep = steel["Ep"]
        name = getattr(t, "name", None) or f"tendon[{j}]"
        if Ep <= 0.0:
            raise ValueError(
                f"losses: tendon '{name}' has no usable modulus (neither "
                f"'Ep' nor 'E' on its material).  The AAEM equation is "
                f"written on E_p."
            )
        # sigma_p0 = E_p * (section strain at the tendon + its prestrain).
        # The AAEM is a linear-viscoelastic theory; at service level the
        # tendon sits far below f_p0.1k and its law is linear.
        eps_sec_t = _at(p_sec, t.x, t.y)
        sp0 = Ep * (eps_sec_t + float(state.eps_init[ui]))

        d_pr = 0.0
        if z in models and name in tendon_ages:
            f_pk = steel["f_pk"]
            if not f_pk:
                raise ValueError(
                    f"losses: tendon '{name}' carries no characteristic "
                    f"strength f_pk (its material has no '.ec2' "
                    f"properties bundle).  The intrinsic relaxation is a "
                    f"function of mu = sigma_pi / f_pk and cannot be "
                    f"guessed: build the tendon material through "
                    f"gensec.materials.ec2_bridge.prestress_from_ec2 (or "
                    f"prestress_from_class), which attaches it."
                )
            lm = models[z]
            prov = zone_provider[z].with_steel(
                float(f_pk),
                relaxation_class=steel["relaxation_class"],
                rho_1000=steel["rho_1000"],
            )
            T_a, mu0 = (relax_from or {}).get(
                name, (0.0, abs(sp0) / float(f_pk)))
            T_b = float(tendon_ages[name])
            d_pr = lm.relaxation_reduction * (
                prov.relaxation(T_b, float(mu0))
                - prov.relaxation(float(T_a), float(mu0))
            )

        A.append(float(t.Ap))
        lx.append(float(t.x) - xr)
        ly.append(float(t.y) - yr)
        zone.append(z)
        E.append(Ep)
        dspr.append(float(d_pr))
        e_mech.append(_at(p_mech[z], t.x, t.y))
        s_c.append(float(E0_z[z]) * _at(p_mech[z], t.x, t.y))
        s_p0.append(float(sp0))
        tj.append(j)
        nm.append(name)

    return {
        "A": np.array(A, dtype=float),
        "lx": np.array(lx, dtype=float),
        "ly": np.array(ly, dtype=float),
        "zone": np.array(zone, dtype=int),
        "E": np.array(E, dtype=float),
        "dsigma_pr": np.array(dspr, dtype=float),
        "eps_mech": np.array(e_mech, dtype=float),
        "sigma_c": np.array(s_c, dtype=float),
        "sigma_p0": np.array(s_p0, dtype=float),
        "tendon_j": np.array(tj, dtype=int),
        "name": nm,
    }


# ==================================================================
#  The engine
# ==================================================================

def compute_interval_losses(base_section, state, *, models, demand,
                            zone_ages_t0, zone_ages_t, zone_curing_ages,
                            tendon_ages, history=None, relax_from=None,
                            service_datums=None, moduli=None, tol=1e-6,
                            max_iter=100):
    r"""
    One AAEM interval on the section: solve, and emit the section ops.

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState
        The state at the **start** of the interval (the resolution walk
        hands the frozen prefix state).
    models : dict
        ``{zone_index: LossModel}``, one per creeping concrete zone.  A
        zone with **no** model is inert: it does not creep and does not
        shrink, but it keeps its instantaneous stiffness and therefore
        *restrains* the zones that do — a real, and often governing,
        effect (an old, fully-crept substrate under a fresh topping).
    demand : tuple of float
        The frozen ``(N, Mx, My)`` [N, N·mm] sustained over the
        interval — the quasi-permanent action of EN 1992-1-1 (5.46).
        The timeline hands its cumulative construction demand, which is
        the same quantity ``_auto_datum`` and ``_reconcile_grout``
        already use.
    zone_ages_t0, zone_ages_t : dict
        ``{zone_index: age [days]}`` at the two ends.  In a composite
        section these differ *per zone*: that is the entire point.
    zone_curing_ages : dict
        ``{zone_index: t_s [days]}``, the age at the end of curing — the
        origin of drying shrinkage.
    tendon_ages : dict
        ``{tendon_name: duration under load at the interval end
        [days]}``.  Relaxation is measured from **stressing**, not from
        casting.
    history : dict, optional
        ``{zone: [(tau_j, sigma_plane_j), ...]}`` — the concrete stress
        **history** of every zone: each entry a stress *increment*
        (a plane ``(sigma_0, sigma_x, sigma_y)`` [MPa, MPa/mm, MPa/mm]
        in the integrator's ``b = [1, ly, -lx]`` basis) and the age at
        which it was applied.

        This is **not** an optimisation, it is the physics.  Creep obeys
        Boltzmann superposition,

        .. math::

            \varepsilon_{\mathrm{cr}} = \sum_j \Delta\sigma_j
                \bigl[J(t_b, \tau_j) - J(t_a, \tau_j)\bigr] ,

        so a stress that has been in place for years creeps *far less*
        over the next step than one just applied.  Treating the current
        stress as if it were all applied at the start of the step —
        the naive cumulative sum ``7_0`` §7 warns about — over-counts
        badly: on a 7-step integration of a 70-year interval it inflates
        the loss by 82 %.

        The state cannot supply the history: once an eigenstrain is
        emitted, reading a stress back with the *instantaneous* modulus
        gives the wrong number (the age-adjusted modulus governed the
        increment).  The walk must carry it.

        ``None`` (default) means *first loading*: the whole current
        stress is taken as established at :math:`t_0`.  Seeding the
        history with ``[(t0, E_c(t0) * p_mech)]`` reproduces that case
        identically, so the single-interval path is unchanged.
    relax_from : dict, optional
        ``{tendon_name: (T_start, mu_0)}`` — the duration under load at
        the **start** of the interval and the stress ratio at stressing.
        The relaxation of the step is then the *increment*

        .. math::

            \Delta\sigma_{pr} = \rho(T_{\mathrm{end}}, \mu_0)
                                - \rho(T_{\mathrm{start}}, \mu_0) ,

        not the cumulative value — which, added to an ``eps_init`` that
        already carries it, would double-count.  ``None`` means
        ``T_start = 0`` and ``mu_0`` read from the current stress: the
        first interval after stressing.
    moduli : dict, optional
        Explicit SLS moduli for the service solve at :math:`t_0`.  When
        omitted each concrete zone takes the modulus of **its own**
        rheological provider, :math:`E_{c,z}(t_{0,z})` — the
        self-consistent choice, and the one that makes the creep
        eigenstrain :math:`\varphi\,\sigma_c / E_c` exact.
    tol, max_iter : float, int, optional
        Forwarded to the linear service solve.

    Returns
    -------
    IntervalLosses

    Raises
    ------
    NotImplementedError
        An unbonded tendon is active (see the module docstring).
    ValueError
        Concrete outside the provider's linear range; a modelled zone
        with no age; a non-converging service solve.
    """
    planes0 = np.asarray(state.bulk_planes, dtype=float)
    n_zones = int(planes0.shape[0])
    n_reb = int(base_section.x_rebars.size)
    tendons = list(getattr(base_section, "tendons", []))

    unbonded = [
        (getattr(t, "name", None) or f"tendon[{j}]")
        for j, t in enumerate(tendons)
        if state.active[n_reb + j] and not state.bonded[n_reb + j]
    ]
    if unbonded:
        raise NotImplementedError(
            f"losses: tendon(s) {unbonded} are unbonded at this interval. "
            f"An unbonded (or external) tendon's stress change over time "
            f"is governed by the *member-average* concrete strain along "
            f"its free length — a beam-level quantity a cross-section "
            f"library does not have.  Substituting the local sectional "
            f"strain would be a silent mismodel.  Either grout the tendon "
            f"before the interval, or compute its time-dependent loss "
            f"externally and declare the reduced sigma_p directly in the "
            f"stage's 'prestress_actions'."
        )

    # -- 1. geometry-bound providers, and the service moduli at t0 ----
    zone_provider: Dict[int, RheologicalModel] = {}
    for z, lm in models.items():
        if z not in zone_ages_t0:
            raise ValueError(
                f"losses: zone {z} ('{base_section.zone_names[z]}') "
                f"carries a loss model but the timeline gives it no age "
                f"at the start of the interval.  A zone that creeps must "
                f"have been cast."
            )
        A_c, u_dry = _zone_drying_geometry(base_section, z)
        p = lm.provider
        zone_provider[z] = (p if p.A_c is not None
                            else p.with_geometry(A_c, u_dry))

    # resolve_sls_moduli keys its *input* by material instance (or name);
    # only its output map is keyed by id().  Phase-7 D3 forbids deriving an
    # SLS modulus silently — here the derivation is explicit and auditable:
    # it is the loss model's own E_c(t0), the very modulus whose creep the
    # eigenstrain phi*sigma_c/E_c is written against.
    # resolve_sls_moduli keys its *input* by material instance, name, or
    # id() (the last added by Phase-5 finding F3: a Concrete is an
    # unfrozen dataclass, hence unhashable, and its name is empty unless
    # declared).  Phase-7 D3 forbids deriving an SLS modulus silently --
    # here every derivation is explicit and auditable:
    #
    #   * a zone WITH a loss model      -> its own provider's E_c(t0), the
    #     very modulus the creep eigenstrain phi*sigma_c/E_c is written
    #     against.  Self-consistent by construction.
    #   * a zone NOT YET CAST           -> a placeholder.  The state masks
    #     its fibres out of the view, so the value cannot reach any
    #     result; the asserted invariant is that the answer is invariant
    #     to it (the validator perturbs it by 10^6 and checks).
    #   * an ACTIVE zone with no model  -> raise.  Such a zone does not
    #     creep, but it keeps its stiffness and therefore *restrains* the
    #     zones that do -- often the governing effect.  Guessing its
    #     modulus would be exactly the silent normative choice D3 forbids.
    mod = dict(moduli or {})
    n_zones_sec = 1 + len(getattr(base_section, "bulk_materials", []))
    for z in range(n_zones_sec):
        mat = _zone_material(base_section, z)
        if id(mat) in mod or getattr(mat, "name", "") in mod:
            continue
        if z in models:
            mod[id(mat)] = zone_provider[z].E_c(zone_ages_t0[z])
        elif not bool(np.asarray(state.bulk_active)[z]):
            mod[id(mat)] = _UNCAST_PLACEHOLDER_E
        else:
            raise ValueError(
                f"losses: zone {z} "
                f"('{base_section.zone_names[z]}') is active during this "
                f"interval but carries neither a loss model nor an "
                f"explicit SLS modulus.  It does not creep -- but it "
                f"keeps its stiffness, and therefore restrains the zones "
                f"that do, which is frequently the governing effect.  "
                f"Give it a loss model, or pass its modulus in 'moduli'.  "
                f"Deriving it silently is what Phase-7 D3 forbids."
            )

    # -- 2. the linear (service) state at t0 --------------------------
    view = materialize_view(base_section, state)
    modmap = resolve_sls_moduli(base_section, mod)
    solver = FiberSolver(sls_view(view, modmap))
    N, Mx, My = (float(v) for v in demand)
    sol = solver.solve_equilibrium(N, Mx, My, tol=tol, max_iter=max_iter)
    if not sol["converged"]:
        raise ValueError(
            f"losses: the linear service solve at the start of the "
            f"interval did not converge under the frozen demand "
            f"(N={N / 1e3:.1f} kN, Mx={Mx / 1e6:.1f} kN·m, "
            f"My={My / 1e6:.1f} kN·m).  A linear-elastic view should "
            f"always converge — the stage view is likely degenerate."
        )
    p_sec = np.array([sol["eps0"], sol["chi_x"], sol["chi_y"]], dtype=float)

    # -- 3. the mechanical-strain plane of every zone at t0 -----------
    # eps_mech,z = eps_sec + bulk_eps_init + datum_z, every term linear in
    # the same basis b = [1, ly, -lx] -> a plane, in closed form.  No
    # fitting.
    #
    # The datum is the *service* one, not ``state.bulk_planes``.  Those
    # two are different objects and conflating them is a real error
    # (Phase-5 finding F4): ``bulk_planes`` carries the datum written at
    # casting by ``ConstructionTimeline._auto_datum``, which solves the
    # substrate with the **ULS** view (design laws, non-linear).  The
    # losses need the plane of the **service** view (linear, at E_c(t)).
    # The two differ by ~1e-4 of strain, so negating one against the other
    # leaves a freshly-cast zone reading a few MPa of spurious stress --
    # when "a zone is unstressed at its own casting" is a statement of
    # physics, not an artefact of whichever solver drew the plane.
    datums = planes0 if service_datums is None else np.array(
        [service_datums.get(z, planes0[z]) for z in range(n_zones)],
        dtype=float)
    p_mech = datums + p_sec[None, :]
    p_mech[:, 0] += float(state.bulk_eps_init)

    # -- 4. rheology, per zone ----------------------------------------
    phi_z = np.zeros(n_zones)
    chi_z = np.zeros(n_zones)
    E0_z = np.zeros(n_zones)
    Ebar_z = np.zeros(n_zones)
    esh_z = np.zeros(n_zones)
    for z in range(n_zones):
        if z not in models:
            E0_z[z] = float(modmap[id(_zone_material(base_section, z))][1])
            Ebar_z[z] = E0_z[z]
            continue
        lm, p = models[z], zone_provider[z]
        a0, a1 = float(zone_ages_t0[z]), float(zone_ages_t[z])
        ts = float(zone_curing_ages.get(z, a0))
        E0_z[z] = p.E_c(a0)
        phi_z[z] = p.phi(a1, a0)
        chi_z[z] = lm.chi_at(p, a1, a0)
        Ebar_z[z] = E0_z[z] / (1.0 + chi_z[z] * phi_z[z])
        esh_z[z] = p.eps_imposed(a1, ts) - p.eps_imposed(a0, ts)

    # -- 5. fibre arrays, linearity guard -----------------------------
    mi = np.asarray(base_section.mat_indices, dtype=int)
    ly_f = np.asarray(base_section.y_fibers, float) - float(solver.y_ref)
    lx_f = np.asarray(base_section.x_fibers, float) - float(solver.x_ref)
    eps_mech_f = (p_mech[mi, 0] + p_mech[mi, 1] * ly_f
                  - p_mech[mi, 2] * lx_f)
    live = np.asarray(state.bulk_active, dtype=bool)[mi]
    for z in models:
        sel = live & (mi == z)
        if sel.any():
            _check_linearity(zone_provider[z], base_section.zone_names[z],
                             float(zone_ages_t0[z]),
                             float((E0_z[z] * eps_mech_f[sel]).min()))

    # -- 6. the eigenstrain of every zone, by Boltzmann superposition --
    # eps_imp,z = sum_j d_sigma_j [J(t_b, tau_j) - J(t_a, tau_j)] + d_eps_sh
    # Every d_sigma_j is a *plane*, J a scalar -> the eigenstrain is a
    # plane, in closed form.  With no history the single increment
    # (t0, E_c(t0) * p_mech) reproduces phi * eps_mech exactly.
    hist = dict(history or {})
    eps_imp_p = np.zeros((n_zones, 3), dtype=float)
    for z in models:
        p = zone_provider[z]
        a0, a1 = float(zone_ages_t0[z]), float(zone_ages_t[z])
        entries = hist.get(z)
        if not entries:
            entries = [(a0, E0_z[z] * p_mech[z])]
        acc = np.zeros(3, dtype=float)
        for tau, d_sig in entries:
            tau = float(tau)
            if tau > a0 + 1e-12:
                raise ValueError(
                    f"losses: zone {z} carries a stress increment applied "
                    f"at age {tau:g} d, later than the start of the "
                    f"interval ({a0:g} d).  The history must be causal."
                )
            dJ = p.J(a1, tau) - p.J(a0, tau)
            acc += np.asarray(d_sig, dtype=float) * dJ
        acc[0] += esh_z[z]
        eps_imp_p[z] = acc

    # -- 7. the AAEM system --------------------------------------------
    A_f = np.asarray(base_section.A_fibers, float)[live]
    ly_f, lx_f, mi_f = ly_f[live], lx_f[live], mi[live]
    eps_imp_f = (eps_imp_p[mi_f, 0] + eps_imp_p[mi_f, 1] * ly_f
                 - eps_imp_p[mi_f, 2] * lx_f)
    b_f = np.stack([np.ones_like(ly_f), ly_f, -lx_f], axis=1)
    K = np.einsum('i,ij,ik->jk', Ebar_z[mi_f] * A_f, b_f, b_f)
    f = np.einsum('i,ij->j', A_f * Ebar_z[mi_f] * eps_imp_f, b_f)

    pt = _point_elements(base_section, state, solver, models, zone_provider,
                         tendon_ages, relax_from, p_sec, p_mech, E0_z, n_reb,
                         tendons)
    b_p = np.zeros((0, 3))
    if pt["A"].size:
        zp = pt["zone"]
        eps_imp_pt = (eps_imp_p[zp, 0] + eps_imp_p[zp, 1] * pt["ly"]
                      - eps_imp_p[zp, 2] * pt["lx"])
        b_p = np.stack([np.ones_like(pt["ly"]), pt["ly"], -pt["lx"]], axis=1)
        K += np.einsum('i,ij,ik->jk',
                       (pt["E"] - Ebar_z[zp]) * pt["A"], b_p, b_p)
        f -= np.einsum('i,ij->j',
                       pt["A"] * (pt["dsigma_pr"]
                                  + Ebar_z[zp] * eps_imp_pt), b_p)

    u = np.linalg.solve(K, f)

    # -- 7. emission ---------------------------------------------------
    out = IntervalLosses(
        u=u,
        phi={int(z): float(phi_z[z]) for z in models},
        chi={int(z): float(chi_z[z]) for z in models},
    )
    # The emitted bulk offset is the *negative* of the eigenstrain: the
    # kernel ADDS the offset, the physics SUBTRACTS the eigenstrain.
    beta = np.zeros((n_zones, 3), dtype=float)
    for z in models:
        beta[z] = -eps_imp_p[z]
    out.bulk_plane_delta = beta
    # The stress increment this step produced, per zone, as a plane --
    # what the caller must append to the history at the step's effective
    # age so the NEXT step superposes instead of restarting the clock.
    d_sigma_z = np.zeros((n_zones, 3), dtype=float)
    for z in models:
        d_sigma_z[z] = Ebar_z[z] * (u - eps_imp_p[z])
    out.d_sigma_zone = d_sigma_z
    out.eps_imp_zone = eps_imp_p
    # The stress standing in each zone at the start of the step, as a
    # plane -- the seed of the Boltzmann history (sigma = E_c(t0) *
    # eps_mech is the true stress only while no eigenstrain has yet been
    # emitted onto the state, i.e. exactly at the first loading).
    out.sigma_zone0 = np.array(
        [E0_z[z] * p_mech[z] for z in range(n_zones)], dtype=float)

    for k in range(pt["A"].size):
        j = int(pt["tendon_j"][k])
        if j < 0:
            continue
        name = pt["name"][k]
        d_sig = float(pt["E"][k] * (b_p[k] @ u) + pt["dsigma_pr"][k])
        out.sigma_p0[name] = float(pt["sigma_p0"][k])
        out.sigma_p[name] = float(pt["sigma_p0"][k]) + d_sig
        out.d_sigma_p[name] = d_sig
        out.d_sigma_pr[name] = float(pt["dsigma_pr"][k])
        out.sigma_c_tendon[name] = float(pt["sigma_c"][k])
        # The emission theorem: only the relaxation moves the tendon's
        # own prestrain.  Creep and shrinkage reach it through the plane.
        out.eps_override[n_reb + j] = float(
            state.eps_init[n_reb + j] + pt["dsigma_pr"][k] / pt["E"][k]
        )
    return out


# ==================================================================
#  Expander:  loss spec  ->  section_ops
# ==================================================================

def expand_losses(base_section, state, *, models, demand, zone_ages_t0,
                  zone_ages_t, zone_curing_ages, tendon_ages_t0,
                  interval_days, steps=None, history=None, relax_from=None,
                  service_datums=None, moduli=None, label="interval"):
    r"""
    Lower one ``interval`` with a ``losses`` reference into
    ``section_ops``.

    A single AAEM step over the whole interval is the default and the
    classical treatment.  When ``steps`` is given, the interval is
    integrated step by step: each sub-step is its own AAEM solve on the
    **updated** state, and the emitted ops are the *cumulative* result —
    so the timeline still sees exactly one op-set per interval.

    .. warning::

        Step-by-step integration is **not** a free refinement.  Each
        sub-step is checked against
        :data:`~gensec.solver.section_state.QUANT_EPS`: a step whose
        strain increment falls below the domain-cache quantum would be
        silently absorbed by the state hash.  See
        :func:`_check_subquantum`.

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState
        State at the start of the interval.
    models : dict
        ``{zone_index: LossModel}``.
    demand : tuple of float
        The frozen ``(N, Mx, My)``.
    zone_ages_t0, zone_ages_t, zone_curing_ages : dict
        Per-zone ages [days], as in :func:`compute_interval_losses`.
    tendon_ages_t0 : dict
        ``{tendon_name: duration under load at the *start* [days]}``;
        the interval adds to it.
    interval_days : float
        Length of the interval [days].
    steps : sequence of float, optional
        Intermediate **elapsed times** [days] within the interval,
        strictly increasing and in ``(0, interval_days)``.  Overrides
        each model's own ``steps``.
    moduli : dict, optional
        Explicit SLS moduli for the service solves.
    label : str, optional
        For error messages.

    history : dict, optional
        ``{zone: [(tau, sigma_plane)]}`` carried in from the **previous**
        intervals of the walk (Boltzmann superposition does not restart
        at an interval boundary any more than it does at a sub-step
        boundary).  ``None`` on the first interval.

    relax_from : dict, optional
        ``{tendon: (T_start [days], mu_0)}`` carried in from the previous
        intervals.  ``mu_0`` is the stress ratio **at stressing** and
        stays frozen: re-reading it from the decayed stress would let the
        relaxation feed on its own decay.

    Returns
    -------
    tuple
        ``(section_ops, [IntervalLosses, ...], history, relax_from)`` — the ops dict
        carries ``eps_override`` (``{union_index: eps}``) and
        ``bulk_plane_delta`` (``{zone_index: [d_eps0, d_chi_x,
        d_chi_y]}``); the list is the per-step trace (one entry when
        ``steps`` is ``None``); ``history`` is the updated stress history,
        to be handed to the next interval.

    Raises
    ------
    ValueError
        Malformed ``steps``; a sub-quantum step; anything
        :func:`compute_interval_losses` raises.
    """
    T = float(interval_days)
    if steps is None:
        for lm in models.values():
            if lm.steps is not None:
                steps = lm.steps
                break
    if steps is None:
        cuts = [T]
    else:
        s = [float(v) for v in steps]
        if any(v <= 0.0 or v >= T for v in s):
            raise ValueError(
                f"losses ({label}): every entry of 'steps' must be an "
                f"elapsed time strictly inside (0, {T:g}) days; got {s!r}."
            )
        cuts = s + [T]

    n_zones = int(np.asarray(state.bulk_planes).shape[0])
    beta_tot = np.zeros((n_zones, 3), dtype=float)
    eps_tot: Dict[int, float] = {}
    trace: List[IntervalLosses] = []

    st = state
    history = {int(z): list(v) for z, v in (history or {}).items()}
    # Relaxation is a function of the TOTAL time under load, so it too must
    # be carried across intervals: (T_start, mu_0) per tendon, with mu_0
    # frozen at stressing.  Rebuilding it per interval would charge the
    # cumulative relaxation again at every interval -- three intervals,
    # three full relaxations.
    relax_from = {k: tuple(v) for k, v in (relax_from or {}).items()}
    elapsed_prev = 0.0
    for k, dt in enumerate(cuts):
        a0 = {z: zone_ages_t0[z] + elapsed_prev for z in zone_ages_t0}
        a1 = {z: zone_ages_t0[z] + dt for z in zone_ages_t0}
        tend = {n: tendon_ages_t0[n] + dt for n in tendon_ages_t0}
        res = compute_interval_losses(
            base_section, st, models=models, demand=demand,
            zone_ages_t0=a0, zone_ages_t=a1,
            zone_curing_ages=zone_curing_ages, tendon_ages=tend,
            history=(history or None), relax_from=(relax_from or None),
            service_datums=service_datums, moduli=moduli,
        )
        if len(cuts) > 1:
            probe = [res.eps_override[i] - st.eps_init[i]
                     for i in res.eps_override]
            probe += list(res.bulk_plane_delta[:, 0])
            _check_subquantum(probe, f"{label}[{k + 1}/{len(cuts)}]")

        beta_tot += res.bulk_plane_delta
        eps_tot.update(res.eps_override)
        trace.append(res)

        # The history must close over the *last* sub-step too: the next
        # interval superposes on everything this one did.
        for z in models:
            h = history.setdefault(z, [])
            if not h:
                h.append((a0[z], res.sigma_zone0[z].copy()))
            h.append((float(np.sqrt(a0[z] * a1[z])),
                      res.d_sigma_zone[z].copy()))
        for name, sp0 in res.sigma_p0.items():
            if name not in relax_from:
                fpk = _fpk_by_name(base_section, name)
                relax_from[name] = (tendon_ages_t0[name],
                                    abs(sp0) / fpk if fpk else 0.0)
            relax_from[name] = (tend[name], relax_from[name][1])

        if k < len(cuts) - 1:
            # Carry the *stress history* forward: Boltzmann superposition
            # needs to know when each stress increment was applied, and
            # the emitted state cannot say (reading a stress back with the
            # instantaneous modulus gives the wrong number).  The first
            # step seeds the history with the stress standing at t0; every
            # step then appends its own redistribution, recorded at the
            # geometric mid-age of the step -- the same mid-step device the
            # Volterra kernel of ``relaxation_function`` uses.
            st = st.with_eps_override(res.eps_override)
            st = _apply_bulk_delta(st, res.bulk_plane_delta)
        elapsed_prev = dt

    ops = {}
    if eps_tot:
        ops["eps_override"] = eps_tot
    nz = np.flatnonzero(np.abs(beta_tot).sum(axis=1) > 0.0)
    if nz.size:
        ops["bulk_plane_delta"] = {int(z): beta_tot[z].tolist() for z in nz}
    return ops, trace, history, relax_from


def _apply_bulk_delta(state, delta):
    r"""
    Add a per-zone plane increment to a state's ``bulk_planes``.

    The in-container twin of the ``bulk_plane_delta`` section op that
    :meth:`~gensec.solver.section_state.StagedDomainManager.
    resolve_stages` applies on the capacity side.  Used *between* the
    sub-steps of a step-by-step integration, where the next AAEM solve
    must see the eigenstrain the previous one accrued.

    Parameters
    ----------
    state : SectionState
    delta : numpy.ndarray
        Shape ``(n_zones, 3)``.  Zones with an all-zero row are skipped,
        so an inert zone is never touched (and never trips the
        not-active guard).

    Returns
    -------
    SectionState
        A copy.
    """
    d = np.asarray(delta, dtype=float)
    nz = np.flatnonzero(np.abs(d).sum(axis=1) > 0.0)
    if not nz.size:
        return state
    return state.with_bulk_plane_delta({int(z): d[z] for z in nz})


# ==================================================================
#  The independent closed form — EN 1992-1-1 (5.46)
# ==================================================================

def ec2_5106_closed_form(*, Ep, Ec, Ap, Ac, Ic, z_cp, sigma_c0, eps_sh,
                         d_sigma_pr, phi, chi=EC2_LUMP_CHI,
                         k_r=EC2_LUMP_RELAXATION_REDUCTION):
    r"""
    EN 1992-1-1 expression (5.46), in GenSec's **signed** convention.

    .. math::

        \Delta\sigma_{p,c+s+r} =
        \frac{E_p\,\varepsilon_{cs}
              + \frac{E_p}{E_{cm}}\varphi(t,t_0)\,\sigma_{c,QP}
              + k_r\,\Delta\sigma_{pr}}
             {1 + \frac{E_p}{E_{cm}}\,\Omega\,\bigl(1 + \chi\,\varphi\bigr)}
        ,
        \qquad
        \Omega = A_p\left(\frac{1}{A_c}
                          + \frac{z_{cp}^{2}}{I_c}\right) .

    EC2 writes the denominator's bracket as
    :math:`\frac{A_p}{A_c}\bigl(1 + \frac{A_c z_{cp}^2}{I_c}\bigr)`,
    which is algebraically :math:`\Omega`; and its two 0.8's are exactly
    :math:`\chi` (the AAEM ageing coefficient) and :math:`k_r` (the
    relaxation reduction).  Compression, shrinkage and relaxation all
    enter **negative**, so the returned value is negative — a loss.

    This function is **not** the engine: it is the independent
    assembly against which the sectional AAEM
    (:func:`compute_interval_losses`) is checked on a single-zone,
    single-tendon section, where the two agree to 0.04 %.  It is
    undefined on a composite section, which is why it cannot be the
    engine.

    Parameters
    ----------
    Ep, Ec : float
        Tendon and concrete moduli [MPa].
    Ap, Ac, Ic : float
        Tendon area [mm²], concrete area [mm²], concrete second moment
        [mm⁴].
    z_cp : float
        Tendon eccentricity from the concrete centroid [mm].
    sigma_c0 : float
        Concrete stress at the tendon level under the quasi-permanent
        action [MPa, **negative** in compression].
    eps_sh : float
        Shrinkage over the interval [-, **negative**].
    d_sigma_pr : float
        **Intrinsic** relaxation [MPa, **negative**].
    phi : float
        Creep coefficient over the interval [-].
    chi, k_r : float, optional
        Ageing coefficient and relaxation reduction.  Default 0.8 each
        — EC2's lump values.

    Returns
    -------
    float
        :math:`\Delta\sigma_p` [MPa], negative.
    """
    omega = float(Ap) * (1.0 / float(Ac) + float(z_cp) ** 2 / float(Ic))
    num = (float(Ep) * float(eps_sh)
           + (float(Ep) / float(Ec)) * float(phi) * float(sigma_c0)
           + float(k_r) * float(d_sigma_pr))
    den = 1.0 + (float(Ep) / float(Ec)) * omega * (1.0 + float(chi)
                                                   * float(phi))
    return num / den
