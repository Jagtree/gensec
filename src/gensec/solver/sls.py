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

- **D1** the stress engine is the *fiber solver on a linear-elastic
  substituted view* (the *SLS view*), not a closed-form
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
- **D4** violation of the uncracked linear basis
  (:math:`\max \sigma_{ct} > f_{ct,\mathrm{eff}}`) is a reported
  **flag** (``uncracked_basis_violated``): stress values remain in
  the output as informative, basis-dependent checks are marked
  ``basis_valid = False`` and the stage is not verified.  Cracked
  re-analysis is a later phase.
- **D5** the surface is API-only; YAML wiring follows the
  ``prestress_sequence`` compiler (§4.3).
- **D6** limits and decompression geometry are normative-agnostic
  primitives (:class:`~gensec.materials.verification_limits.StressLimits`,
  cover distance ``c_dec``); EC2 values are a provider default.

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
Cracked-section SLS re-analysis (the natural extension: swap the
concrete law and iterate — flagged by ``uncracked_basis_violated``);
concrete stress redistribution by creep beyond what the losses model
routes through ``eps_init`` / ``bulk_eps`` (full AAEM on the stress
field); stages with leaving elements; YAML wiring (D5); per-zone
transformed properties for multi-zone bulk (mirrors the existing
``compute_ideal_properties`` restriction).
"""

import copy
import dataclasses
from dataclasses import replace as _dc_replace

import numpy as np

from ..materials.elastic import LinearElastic
from ..materials.verification_limits import StressLimits
from .integrator import FiberSolver
from .section_state import SectionState, materialize_view


__all__ = [
    "resolve_sls_moduli",
    "sls_view",
    "sls_transformed_properties",
    "verify_sls_staged",
]


# Numerical tolerance on the decompression verdict [MPa]: a probe
# stress up to this magnitude on the tensile side still passes, to
# absorb solver round-off at the exact decompression boundary.
_DECOMPRESSION_TOL = 1.0e-9


# ==================================================================
#  Modulus resolution (D3: never silent)
# ==================================================================


def _unique_materials(section):
    r"""
    Unique material instances of a section (or view), by identity.

    Order: primary bulk, zone bulks, rebar materials, tendon
    materials — first occurrence wins.

    Parameters
    ----------
    section : GenericSection
        Base section or :func:`materialize_view` output.

    Returns
    -------
    list of Material
    """
    mats = []
    seen = set()

    def _add(m):
        if m is not None and id(m) not in seen:
            seen.add(id(m))
            mats.append(m)

    if hasattr(section, "get_all_bulk_materials"):
        for m in section.get_all_bulk_materials():
            _add(m)
    else:
        _add(section.bulk_material)
    for r in section.rebars:
        _add(r.material)
    for t in getattr(section, "tendons", []):
        _add(t.material)
    return mats


def resolve_sls_moduli(section, moduli=None):
    r"""
    Resolve the SLS elastic modulus of every material of a section.

    Resolution order, per unique material instance:

    1. explicit override by **instance** (``moduli`` key is the
       material object itself);
    2. explicit override by **name** (``moduli`` key equals
       ``material.name``, when non-empty);
    3. the material's own *intrinsic* elastic-modulus field:
       ``Es`` (reinforcing steel) or ``Ep`` (prestressing steel).

    Anything else — concrete, tabulated laws, unknown materials —
    **must** be overridden explicitly, or a :class:`ValueError` is
    raised listing every unresolved material.  This is the D3 rule:
    a steel's Young modulus is an intrinsic constant of the law
    object, but a concrete SLS modulus (:math:`E_{cm}`, possibly
    :math:`E_{cm}(t)` at transfer, or the effective modulus
    :math:`E_{c,\mathrm{eff}} = E_{cm}/(1 + \varphi\,\psi)` under
    quasi-permanent load) is a normative, time-dependent choice that
    the caller computes through the appropriate bridge — e.g.
    ``concrete.ec2.ecm`` from
    :class:`~gensec.materials.ec2_properties.fben2` with its ``time``
    argument — and passes in as a number.  The engine never reaches
    into a normative bridge on its own.

    Parameters
    ----------
    section : GenericSection
        Base section or view; its unique materials are enumerated.
    moduli : dict, optional
        ``{material_instance_or_name: E [MPa]}`` overrides.

    Returns
    -------
    dict
        ``{id(material): (material, E)}`` covering **every** unique
        material of the section.

    Raises
    ------
    ValueError
        If any material cannot be resolved, if an override value is
        not finite and strictly positive, or if an override key
        matches no material of the section (a dead override is a
        silent-error trap: the caller believes a modulus is in force
        when it is not).
    """
    moduli = dict(moduli or {})

    # Split overrides into by-instance and by-name maps.
    by_id = {}
    by_name = {}
    for key, val in moduli.items():
        if isinstance(key, str):
            by_name[key] = val
        elif isinstance(key, int):
            # An ``id(material)`` handed directly.  Needed because a
            # ``Concrete`` is an unfrozen dataclass -- hence *unhashable*
            # -- and carries an empty ``name`` unless one is declared, so
            # neither the by-instance nor the by-name path is reachable
            # for a programmatically built section.  Without this key the
            # explicit-moduli contract of D3 has a dead end.  (Phase-5
            # finding F3.)
            by_id[key] = val
        else:
            by_id[id(key)] = val

    mats = _unique_materials(section)

    resolved = {}
    missing = []
    used_names = set()
    used_ids = set()
    for m in mats:
        E = None
        if id(m) in by_id:
            E = by_id[id(m)]
            used_ids.add(id(m))
        elif getattr(m, "name", "") and m.name in by_name:
            E = by_name[m.name]
            used_names.add(m.name)
        elif hasattr(m, "Es"):
            E = m.Es
        elif hasattr(m, "Ep"):
            E = m.Ep
        if E is None:
            missing.append(m)
            continue
        E = float(E)
        if not np.isfinite(E) or E <= 0.0:
            raise ValueError(
                f"SLS modulus for material "
                f"{getattr(m, 'name', '') or type(m).__name__!s} "
                f"must be finite and strictly positive; got {E!r}."
            )
        resolved[id(m)] = (m, E)

    if missing:
        names = [
            (getattr(m, "name", "") or type(m).__name__) for m in missing
        ]
        raise ValueError(
            f"No SLS modulus for material(s) {names}: supply it "
            f"explicitly via the 'moduli' mapping (by instance or by "
            f"name).  Concrete/tabulated SLS moduli are normative and "
            f"time-dependent choices and are never derived silently "
            f"(Phase-7 D3)."
        )

    # Dead overrides: fail loud.
    dead_names = set(by_name) - used_names
    dead_ids = set(by_id) - used_ids
    if dead_names or dead_ids:
        raise ValueError(
            f"SLS moduli overrides matched no material of the "
            f"section: names {sorted(dead_names)}, "
            f"{len(dead_ids)} instance key(s).  A dead override "
            f"is a silent-error trap; remove it or fix the key."
        )
    return resolved


# ==================================================================
#  SLS view (D1: constitutive substitution on the materialized view)
# ==================================================================


def sls_view(view, modmap):
    r"""
    Build the SLS view: the section view with every material replaced
    by its :class:`~gensec.materials.elastic.LinearElastic`
    counterpart.

    The substitution is a **shallow re-wrap**: geometry arrays,
    initial-strain arrays and zone indices are shared by reference
    with the input view; only the material references change.  Point
    elements are re-created with :func:`dataclasses.replace` so the
    base section's :class:`~gensec.geometry.fiber.RebarLayer` /
    :class:`~gensec.geometry.fiber.Tendon` objects are **never
    mutated**.  One ``LinearElastic`` instance is created per unique
    original material, so the solver's identity-based grouping
    (:meth:`FiberSolver._build_rebar_groups` and siblings) is
    preserved exactly.

    Parameters
    ----------
    view : GenericSection
        Output of :func:`~gensec.solver.section_state.materialize_view`
        (or a base section, for non-staged use).
    modmap : dict
        ``{id(material): (material, E)}`` from
        :func:`resolve_sls_moduli`, covering every material of the
        view.

    Returns
    -------
    GenericSection
        The substituted view, ready for ``FiberSolver``.

    Raises
    ------
    KeyError
        If a material of the view is not covered by ``modmap``
        (resolve on the same section the view came from).
    """
    def _lin(mat):
        try:
            _, E = modmap[id(mat)]
        except KeyError:
            raise KeyError(
                f"Material "
                f"{getattr(mat, 'name', '') or type(mat).__name__} "
                f"is not covered by the SLS modulus map; call "
                f"resolve_sls_moduli on the same section."
            ) from None
        key = id(mat)
        if key not in _lin.cache:
            base_name = getattr(mat, "name", "") or type(mat).__name__
            _lin.cache[key] = LinearElastic(
                E=E, name=f"SLS({base_name})")
        return _lin.cache[key]

    _lin.cache = {}

    vw = copy.copy(view)
    vw.bulk_material = _lin(view.bulk_material)
    zones = getattr(view, "bulk_materials", None)
    if zones:
        vw.bulk_materials = [(poly, _lin(m)) for poly, m in zones]
    vw.rebars = [_dc_replace(r, material=_lin(r.material))
                 for r in view.rebars]
    if getattr(view, "tendons", None):
        vw.tendons = [_dc_replace(t, material=_lin(t.material))
                      for t in view.tendons]
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
    E_bulk = modmap[id(base_section.bulk_material)][1]

    homog = [
        HomogenizedRebar(r.x, r.y, r.As, modmap[id(r.material)][1])
        for r in view.rebars if r.embedded and r.x is not None
    ]
    homog += [
        HomogenizedRebar(float(x), float(y), float(A),
                         modmap[id(t.material)][1])
        for t, x, y, A in zip(
            getattr(view, "tendons", []),
            getattr(view, "x_tendons", np.empty(0)),
            getattr(view, "y_tendons", np.empty(0)),
            getattr(view, "A_tendons", np.empty(0)),
        )
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

    On a linear-elastic view the Newton solve is exact after one
    tangent step; non-convergence indicates an ill-posed input (a
    degenerate view, a zero-stiffness direction) and must surface.

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
            f"My={F[2]:.6g} N·mm).  A linear-elastic view should "
            f"always converge — the view is likely degenerate."
        )
    return sol


def _element_sigmas(solver, sol, view):
    r"""
    Union-indexed element stresses from one solved plane.

    Reads :meth:`FiberSolver.get_fiber_results` at the solved plane
    and maps the view-local rebar/tendon stresses back to the union
    index space through ``view._union_index``.  The reported element
    stress is the **material** stress (the law evaluated at the
    element's total strain), not the net-of-displaced-bulk value used
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
    bulk_sigma : numpy.ndarray
        Stress at every bulk fiber [MPa] (full mesh length).
    elem_union_idx : numpy.ndarray of int
        Union indices of the view's point elements, rebars first.
    elem_sigma : numpy.ndarray
        Element stresses aligned with ``elem_union_idx`` [MPa].
    plane : tuple of float
        ``(eps0, chi_x, chi_y)``.
    """
    fr = solver.get_fiber_results(sol["eps0"], sol["chi_x"],
                                  sol["chi_y"])
    bulk_sigma = np.asarray(fr["bulk"]["sigma"], dtype=float)

    sig_r = np.asarray(fr["rebars"]["sigma"], dtype=float)
    if "tendons" in fr:
        sig_t = np.asarray(fr["tendons"]["sigma"], dtype=float)
    else:
        sig_t = np.empty(0, dtype=float)
    elem_sigma = np.concatenate([sig_r, sig_t])
    elem_union_idx = np.asarray(view._union_index, dtype=int)
    if elem_union_idx.size != elem_sigma.size:
        raise RuntimeError(
            "View element count does not match _union_index — the "
            "view was not produced by materialize_view."
        )
    plane = (float(sol["eps0"]), float(sol["chi_x"]),
             float(sol["chi_y"]))
    return bulk_sigma, elem_union_idx, elem_sigma, plane


def _affine_entering_set(coeffs, zone_E, plane_hi, state_planes,
                         zone_entering, x_ref, y_ref, d_bulk_eps=0.0):
    r"""
    Initialise the per-zone affine concrete-stress coefficients of the
    zones that enter the resistance set at the current stage.

    An entering zone comes into existence carrying its locked-in datum
    plane :math:`(\varepsilon_0^d, \chi_x^d, \chi_y^d)_z`; the solver
    has already subtracted it in the fibre stresses, so the coefficients
    are SET (not incremented) from the effective plane
    :math:`\mathbf{p}_{hi} - \mathbf{p}^d_z`:

    .. math::

        c_{0,z} &= E_z\left(\Delta\varepsilon_0
            - \Delta\chi_x\, y_{\mathrm{ref}}
            + \Delta\chi_y\, x_{\mathrm{ref}}
            + \Delta\varepsilon_b\right), \\
        c_{x,z} &= -E_z\,\Delta\chi_y, \qquad
        c_{y,z} = E_z\,\Delta\chi_x ,

    with :math:`\Delta\mathbf{p} = \mathbf{p}_{hi} - \mathbf{p}^d_z`
    and :math:`\Delta\varepsilon_b` the legacy uniform bulk pre-strain.

    Parameters
    ----------
    coeffs : dict
        ``{"c0", "cx", "cy"}`` per-zone arrays, mutated in place.
    zone_E : numpy.ndarray
        Zone moduli, index-aligned with the coefficient arrays.
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
    planes = (np.zeros((zone_E.size, 3), dtype=float)
              if state_planes is None
              else np.asarray(state_planes, dtype=float))
    for z in ent:
        # F7: the fibre kernel ADDS the datum -- integrator.py evaluates the
        # bulk law at  eps_sec + [eps_b0 + eps_0,z + chi_x,z (y-y_ref)
        # - chi_y,z (x-x_ref)] -- which is exactly why
        # ConstructionTimeline._auto_datum stores the NEGATED plane.  This
        # function used to SUBTRACT it, on the (false) premise recorded in
        # its own docstring that "the solver has already subtracted it".  It
        # has not.  The affine field was therefore wrong by 2x the datum on
        # every entering zone, and debug_check_affine -- which is off by
        # default -- was the only thing that could see it.
        e0 = plane_hi[0] + float(planes[z, 0])
        cx = plane_hi[1] + float(planes[z, 1])
        cy = plane_hi[2] + float(planes[z, 2])
        Ez = float(zone_E[z])
        coeffs["c0"][z] = Ez * (e0 - cx * y_ref + cy * x_ref
                                + d_bulk_eps)
        coeffs["cx"][z] = -Ez * cy
        coeffs["cy"][z] = Ez * cx


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


def _affine_increment(coeffs, zone_E, plane_hi, plane_lo,
                      x_ref, y_ref, d_bulk_eps=0.0, zone_mask=None):
    r"""
    Accumulate one plane difference into the per-zone affine
    coefficients of the concrete stress field.

    For zone modulus :math:`E_z` and plane increment
    :math:`(\Delta\varepsilon_0, \Delta\chi_x, \Delta\chi_y)` (plus
    an optional bulk imposed-strain increment
    :math:`\Delta\varepsilon_b`):

    .. math::

        \Delta c_0 &= E_z\left(\Delta\varepsilon_0
            - \Delta\chi_x\, y_{\mathrm{ref}}
            + \Delta\chi_y\, x_{\mathrm{ref}}
            + \Delta\varepsilon_b\right), \\
        \Delta c_x &= -E_z\,\Delta\chi_y, \qquad
        \Delta c_y = E_z\,\Delta\chi_x .

    Parameters
    ----------
    coeffs : dict
        ``{"c0": ndarray, "cx": ndarray, "cy": ndarray}`` per zone,
        mutated in place.
    zone_E : numpy.ndarray
        Zone moduli, index-aligned with the coefficient arrays.
    plane_hi, plane_lo : tuple of float
        ``(eps0, chi_x, chi_y)`` of the two solves.
    x_ref, y_ref : float
        Solver reference point.
    d_bulk_eps : float, optional
        Bulk imposed-strain increment carried by this pair.
    """
    de0 = plane_hi[0] - plane_lo[0]
    dcx = plane_hi[1] - plane_lo[1]
    dcy = plane_hi[2] - plane_lo[2]
    d_c0 = zone_E * (de0 - dcx * y_ref + dcy * x_ref + d_bulk_eps)
    d_cx = -zone_E * dcy
    d_cy = zone_E * dcx
    if zone_mask is not None:
        # C4: restrict the increment to a subset of zones (persisting
        # zones — entering zones are SET, not incremented).
        m = np.asarray(zone_mask, dtype=bool)
        d_c0 = np.where(m, d_c0, 0.0)
        d_cx = np.where(m, d_cx, 0.0)
        d_cy = np.where(m, d_cy, 0.0)
    coeffs["c0"] += d_c0
    coeffs["cx"] += d_cx
    coeffs["cy"] += d_cy


def _decompression_probe(x_t, y_t, zone, coeffs, c_dec):
    r"""
    Decompression probe for one bonded tendon (closed form).

    The accumulated concrete stress field in the tendon's zone is
    affine, :math:`S(x,y) = c_0 + c_x x + c_y y`; the most tensile
    point of the disc of radius :math:`c_{\mathrm{dec}}` centred on
    the tendon lies along the gradient :math:`(c_x, c_y)`.

    Parameters
    ----------
    x_t, y_t : float
        Tendon coordinates [mm].
    zone : int
        Bulk-zone index of the tendon.
    coeffs : dict
        Per-zone affine coefficients.
    c_dec : float
        Cover distance [mm].

    Returns
    -------
    dict
        ``x_probe``, ``y_probe``, ``sigma_probe`` [MPa],
        ``passed`` (bool).
    """
    c0 = float(coeffs["c0"][zone])
    cx = float(coeffs["cx"][zone])
    cy = float(coeffs["cy"][zone])
    g = float(np.hypot(cx, cy))
    if g < 1.0e-15:
        # Uniform field: every point of the disc carries the same
        # stress; probe at the tendon itself.
        xp, yp = float(x_t), float(y_t)
    else:
        xp = float(x_t) + c_dec * cx / g
        yp = float(y_t) + c_dec * cy / g
    sp = c0 + cx * xp + cy * yp
    return {
        "x_probe": xp, "y_probe": yp,
        "sigma_probe": sp,
        "passed": bool(sp <= _DECOMPRESSION_TOL),
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


def _check_entry(name, value, limit, *, basis_dependent,
                 extra=None):
    r"""
    Build one check record.

    ``eta = value / limit`` (both positive magnitudes);
    ``passed = value <= limit``.

    Parameters
    ----------
    name : str
    value : float
        Positive magnitude of the acting quantity.
    limit : float
        Positive magnitude of the threshold.
    basis_dependent : bool
        Whether the check reads the uncracked linear basis.
    extra : dict, optional
        Additional payload merged into the record.

    Returns
    -------
    dict
    """
    rec = {
        "name": name,
        "value_MPa": round(float(value), 6),
        "limit_MPa": round(float(limit), 6),
        "eta": round(float(value) / float(limit), 6),
        "passed": bool(value <= limit),
        "basis_dependent": bool(basis_dependent),
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


def _extreme_fibre(base_section, coeffs, present_zones):
    r"""
    The **true extreme-fibre** concrete stresses, in closed form.

    The SLS stress field is *exactly affine* over each zone --
    :math:`\sigma_z(x,y) = c_{0,z} + c_{x,z}\,x + c_{y,z}\,y`, which
    ``debug_check_affine`` asserts against the fiber accumulator to
    :math:`10^{-8}` MPa.  The extremum of an affine function over a
    polygon is attained at a **vertex**.  So the extremes are a closed
    form, and there is no reason to accept a discretisation error where
    one exists.

    .. warning::

        Taking ``argmin``/``argmax`` over the *fibres* -- which is what
        this function replaces -- reports the stress at the outermost
        fibre's **centroid**, inset half a fibre from the section face.
        On a 800 mm web meshed at 50 mm that under-reports the
        extreme-fibre stress by :math:`1 - 50/800 = 6.25\,\%` of the
        bending part, in the **non-conservative** direction: EN 1992-1-1
        §7.2 limits :math:`\sigma_c` at the *extreme fibre*, not at the
        centroid of the last row of fibres.  A fine mesh hides it, which
        is why it survived (Phase-5 finding F6).

    Parameters
    ----------
    base_section : GenericSection
    coeffs : dict
        ``{'c0': (n_zones,), 'cx': ..., 'cy': ...}`` -- the per-zone
        affine stress field already assembled by the caller.
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
        for (x, y) in _region_vertices(regions[z]):
            s = float(coeffs["c0"][z] + coeffs["cx"][z] * x
                      + coeffs["cy"][z] * y)
            if s < s_min:
                s_min, p_min = s, (float(x), float(y))
            if s > s_max:
                s_max, p_max = s, (float(x), float(y))
    if p_min is None:
        raise RuntimeError(
            "extreme-fibre stress: no present zone has a region.")
    return s_min, p_min, s_max, p_max


def _extreme_fibre_by_zone(base_section, coeffs, present_zones):
    r"""
    The extreme-fibre stresses of **each** present zone, separately.

    The global extremes are not enough on a composite: the maximum
    compressive stress and the maximum tensile stress may sit in
    *different* zones, and each zone has its own limit (its own
    :math:`f_{ck}`).  A check that reads only the global extreme is
    checking one zone against another zone's limit.

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
        s_min, p_min = float("inf"), None
        s_max, p_max = float("-inf"), None
        for (x, y) in _region_vertices(regions[z]):
            s = float(coeffs["c0"][z] + coeffs["cx"][z] * x
                      + coeffs["cy"][z] * y)
            if s < s_min:
                s_min, p_min = s, (float(x), float(y))
            if s > s_max:
                s_max, p_max = s, (float(x), float(y))
        if p_min is None:
            continue
        name = (base_section.zone_names[z]
                if z < len(base_section.zone_names) else f"zone[{z}]")
        out[name] = {"sigma_min_MPa": round(s_min, 6), "at_min": p_min,
                     "sigma_max_MPa": round(s_max, 6), "at_max": p_max}
    return out


def verify_sls_staged(base_section, stages, *, moduli=None,
                      x_ref=None, y_ref=None, tol=1.0e-3,
                      max_iter=50, debug_check_affine=True):
    r"""
    SLS stress verification over a stage history (Phase 7 engine).

    Runs the incremental accumulation scheme described in the module
    docstring over the given stages, then evaluates the per-stage
    :class:`~gensec.materials.verification_limits.StressLimits`
    checks on the accumulated stresses.

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
        ``"moduli"`` : dict, optional
            Per-stage modulus overrides merged over the global
            ``moduli`` (aging modulus, e.g. :math:`E_{cm}(t)` at
            transfer).
        ``"limits"`` : StressLimits, optional
            Checks to evaluate on this stage's accumulated state.
            Stages without limits accumulate stresses but produce no
            verdict.
        ``"unbonded_sigma_p"`` : dict, optional
            ``{label: sigma_p [MPa]}`` member-level stresses of
            active-unbonded tendons, checked against
            ``sigma_p_max`` with ``provenance='member_level'``.

    moduli : dict, optional
        Global SLS modulus overrides (see
        :func:`resolve_sls_moduli`).
    x_ref, y_ref : float, optional
        Demand reference point.  Default: the base-section centroid.
        Fixed once for the whole history — demand triples and
        :class:`PrestressAction` couples must share it.
    tol : float, optional
        Force tolerance forwarded to the equilibrium solves [N].
    max_iter : int, optional
        Newton iteration cap (linear views converge immediately).
    debug_check_affine : bool, optional
        After every stage, assert that the per-zone affine field
        reproduces the accumulated per-fiber bulk stresses (internal
        consistency of the two bookkeeping paths).  Default
        ``False``.

    Returns
    -------
    dict
        ``{"type": "sls_staged", "stages": [...], "verified": bool,
        "governing": {...} or None}``.  Each stage record carries the
        cumulative demand, the solved plane of the stage, concrete
        stress extremes with locations, the per-element stress table
        (union-indexed), the ``uncracked_basis_violated`` flag (D4)
        and the check records.

    Raises
    ------
    NotImplementedError
        A stage removes elements from the resistance set (leaving
        transitions are deferred).
    ValueError
        Compound transition (elements entering while initial strains
        change on persisting elements — split into two stages), or
        malformed inputs.
    RuntimeError
        A linear solve failed to converge (degenerate view).

    Notes
    -----
    The per-element table reports the **material** stress of each
    active-and-bonded element (the constitutive law at the element's
    total strain, prestrain included) — the quantity the normative
    limits address.

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
    event (see the composability handoff in ``10_6``).  The fibre
    stresses stay correct in every supported case because the solver
    consumes ``bulk_planes_active``; the guard protects the affine
    decompression bookkeeping and the physical meaning of the walk.
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

    # Zone bookkeeping for the affine concrete field.
    n_zones = 1 + len(getattr(base_section, "bulk_materials", []))
    mat_indices = getattr(base_section, "mat_indices", None)
    if mat_indices is None:
        mat_indices = np.zeros(n_fib, dtype=int)
    mat_idx_t = getattr(base_section, "mat_indices_tendon", None)
    if mat_idx_t is None:
        mat_idx_t = np.zeros(n_ten, dtype=int)

    if hasattr(base_section, "get_all_bulk_materials"):
        zone_mats = list(base_section.get_all_bulk_materials())
    else:
        zone_mats = [base_section.bulk_material]
    if len(zone_mats) != n_zones:
        raise RuntimeError(
            "Zone material list is inconsistent with bulk_materials."
        )

    # ---- Accumulators -------------------------------------------
    # --- GENSEC T3 C4 composite-SLS (idempotency sentinel) ---
    S_bulk = np.zeros(n_fib, dtype=float)
    # C4: which base fibers have entered the resistance history yet.
    # Not-yet-cast zones stay False so their zero entries cannot spoof
    # the concrete stress extremes.
    present_bulk = np.zeros(n_fib, dtype=bool)
    fiber_zone = (mat_indices if mat_indices is not None
                  else np.zeros(n_fib, dtype=int))
    S_union = np.full(n_union, np.nan, dtype=float)
    present = np.zeros(n_union, dtype=bool)
    coeffs = {
        "c0": np.zeros(n_zones, dtype=float),
        "cx": np.zeros(n_zones, dtype=float),
        "cy": np.zeros(n_zones, dtype=float),
    }

    prev_state = None
    prev_resist = None
    F_prev = (0.0, 0.0, 0.0)
    stage_results = []
    all_etas = []          # (eta, stage_name, check_name)
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
        # ---- Composite bulk staging (C4) ----------------------
        # Inactive zones and per-zone locked-in planes are now
        # supported: the solver consumes bulk_planes_active, so b_hi
        # is the true per-zone stress; entering zones initialise from
        # that total read, persisting zones accumulate the increment.
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

        # ---- Views, solver, moduli ------------------------------
        stage_overrides = dict(moduli or {})
        stage_overrides.update(stage.get("moduli", {}) or {})
        modmap = resolve_sls_moduli(base_section, stage_overrides)
        zone_E = np.array([modmap[id(m)][1] for m in zone_mats],
                          dtype=float)

        view_k = materialize_view(base_section, state)
        slsv_k = sls_view(view_k, modmap)
        solver_k = FiberSolver(slsv_k, x_ref=x_ref, y_ref=y_ref)

        sol_hi = _solve_linear(solver_k, F_cum, tol, max_iter,
                               f"stage '{name}', cumulative demand")
        (b_hi, u_idx, e_hi,
         plane_hi) = _element_sigmas(solver_k, sol_hi, slsv_k)

        # C4: view-order bulk stresses scatter back to base indices;
        # per-fiber entering mask follows the fiber's zone.
        keep = np.asarray(getattr(view_k, "_bulk_keep",
                                  np.arange(b_hi.size)), dtype=int)
        base_zone_view = fiber_zone[keep]
        if k == 0:
            zone_entering = (np.ones(n_zones, dtype=bool)
                             if zone_active is None else zone_active)
        elif zone_active is None:
            zone_entering = np.zeros(n_zones, dtype=bool)
        else:
            za_prev = (np.ones(n_zones, dtype=bool)
                       if zone_active_prev is None
                       else np.asarray(zone_active_prev, dtype=bool))
            zone_entering = np.asarray(zone_active, dtype=bool) \
                & ~za_prev
        fiber_enter_view = zone_entering[base_zone_view]

        if k == 0:
            # Total read: every present fiber comes into existence
            # carrying its initial strains; no prior stress history.
            S_bulk[keep] = b_hi
            present_bulk[keep] = True
            _affine_entering_set(
                coeffs, zone_E, plane_hi, state_planes,
                zone_entering, x_ref, y_ref,
                d_bulk_eps=float(state.bulk_eps_init))
            S_union[u_idx] = e_hi
            present[u_idx] = True
        else:
            sol_lo = _solve_linear(
                solver_k, F_prev, tol, max_iter,
                f"stage '{name}', carried demand")
            (b_lo, u_idx_lo, e_lo,
             plane_lo) = _element_sigmas(solver_k, sol_lo, slsv_k)

            # Demand term (initial strains cancel; applies to every
            # element of the current view).
            d_bulk = b_hi - b_lo
            d_elem = e_hi - e_lo
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
            _affine_increment(coeffs, zone_E, plane_hi, plane_lo,
                              x_ref, y_ref, zone_mask=zone_persist)

            # State term (loss redistribution) — eps-only
            # transitions, evaluated on the *current* moduli.
            if eps_changed:
                view_pe = materialize_view(base_section, prev_state)
                slsv_pe = sls_view(view_pe, modmap)
                solver_pe = FiberSolver(slsv_pe, x_ref=x_ref,
                                        y_ref=y_ref)
                sol_pe = _solve_linear(
                    solver_pe, F_prev, tol, max_iter,
                    f"stage '{name}', previous-eps reference")
                (b_pe, u_idx_pe, e_pe,
                 plane_pe) = _element_sigmas(solver_pe, sol_pe,
                                             slsv_pe)
                if not np.array_equal(u_idx_lo, u_idx_pe):
                    raise RuntimeError(
                        "eps-only transition produced mismatched "
                        "element sets — masks were expected equal."
                    )
                d_bulk += b_lo - b_pe
                d_elem += e_lo - e_pe
                _affine_increment(
                    coeffs, zone_E, plane_lo, plane_pe,
                    x_ref, y_ref,
                    d_bulk_eps=(float(state.bulk_eps_init)
                                - float(prev_state.bulk_eps_init)),
                    zone_mask=zone_persist)

            # Persisting fibers accumulate the increment; entering
            # fibers initialise from the stage's total read (datum
            # already carried by b_hi through bulk_planes_active).
            persist_view = ~fiber_enter_view
            S_bulk[keep[persist_view]] += d_bulk[persist_view]
            S_bulk[keep[fiber_enter_view]] = b_hi[fiber_enter_view]
            present_bulk[keep] = True
            _affine_entering_set(
                coeffs, zone_E, plane_hi, state_planes,
                zone_entering, x_ref, y_ref,
                d_bulk_eps=float(state.bulk_eps_init))
            # Persisting elements accumulate the increment; entering
            # elements initialise from the stage's total read.
            enter_set = set(entering.tolist())
            for j, u in enumerate(u_idx):
                if u in enter_set:
                    S_union[u] = e_hi[j]
                    present[u] = True
                else:
                    S_union[u] += d_elem[j]

        # ---- Optional internal consistency check ----------------
        if debug_check_affine:
            recon = (coeffs["c0"][mat_indices]
                     + coeffs["cx"][mat_indices]
                     * base_section.x_fibers
                     + coeffs["cy"][mat_indices]
                     * base_section.y_fibers)
            if not np.allclose(recon, S_bulk, rtol=1.0e-8,
                               atol=1.0e-8):
                raise AssertionError(
                    f"Stage '{name}': affine field / fiber "
                    f"accumulator mismatch (max "
                    f"{np.max(np.abs(recon - S_bulk)):.3e} MPa)."
                )

        # ---- Stage record ---------------------------------------
        # C4: extremes over fibers that have entered the history; a
        # not-yet-cast zone must not spoof min/max with its zero.
        idx_present = np.nonzero(present_bulk)[0]
        if idx_present.size == 0:
            raise RuntimeError(
                f"Stage '{name}': no active bulk fiber in the SLS "
                f"history (empty composite state)."
            )
        i_min = int(idx_present[np.argmin(S_bulk[idx_present])])
        i_max = int(idx_present[np.argmax(S_bulk[idx_present])])
        # F6: the reported extremes are the TRUE extreme-fibre stresses,
        # from the closed form on the affine field, not the outermost
        # fibre's centroid value (which is inset half a fibre and
        # under-reports, non-conservatively, the very stress EN 1992-1-1
        # 7.2 limits).  The fibre argmin/argmax above is kept only to
        # name the present zones.
        _present_zones = sorted(set(int(v) for v in mat_indices[idx_present]))
        (_s_min, _at_min,
         _s_max, _at_max) = _extreme_fibre(base_section, coeffs,
                                           _present_zones)
        _by_zone = _extreme_fibre_by_zone(base_section, coeffs,
                                          _present_zones)
        elements = []
        for u in np.nonzero(present)[0]:
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
                "union_index": int(u), "kind": kind,
                "x": x_e, "y": y_e,
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
                "sigma_min_MPa": round(_s_min, 6),
                "at_min": _at_min,
                "sigma_max_MPa": round(_s_max, 6),
                "at_max": _at_max,
                # The global extremes are not enough on a composite: each
                # zone has its own f_ck, and the governing compression may
                # sit in one zone while the governing tension sits in
                # another.  GenSec's job is to hand the engineer every
                # number it has, not only the one it acted on.
                "by_zone": _by_zone,
            },
            "elements": elements,
            "active_zones": _active_zone_labels(base_section,
                                                zone_active, n_zones),
        }

        # ---- Checks (D4 semantics) ------------------------------
        limits = stage.get("limits")
        if limits is not None:
            if not isinstance(limits, StressLimits):
                raise ValueError(
                    f"Stage '{name}': 'limits' must be a "
                    f"StressLimits, got {type(limits).__name__}."
                )
            checks = []
            sig_min = float(S_bulk[i_min])
            sig_max = float(S_bulk[i_max])

            # Uncracked-basis validity.
            if limits.fct_eff is not None:
                basis_checked = True
                basis_violated = bool(sig_max > limits.fct_eff)
            else:
                basis_checked = False
                basis_violated = False
            sr["basis_checked"] = basis_checked
            sr["uncracked_basis_violated"] = basis_violated

            # Concrete compression.
            if limits.sigma_c_comp is not None:
                checks.append(_check_entry(
                    "concrete_compression",
                    max(0.0, -sig_min), limits.sigma_c_comp,
                    basis_dependent=True,
                    extra={"at": sr["concrete"]["at_min"]}))
            else:
                checks.append(_skipped_entry(
                    "concrete_compression", "no sigma_c_comp limit"))

            # Reinforcing steel (tension).
            reb_present = present[:n_reb]
            if reb_present.any():
                if limits.sigma_s_tens is not None:
                    val = max(0.0,
                              float(np.max(S_union[:n_reb]
                                           [reb_present])))
                    checks.append(_check_entry(
                        "steel_tension", val, limits.sigma_s_tens,
                        basis_dependent=True))
                else:
                    checks.append(_skipped_entry(
                        "steel_tension", "no sigma_s_tens limit"))

            # Bonded tendons.
            ten_present = present[n_reb:]
            if ten_present.any():
                if limits.sigma_p_max is not None:
                    val = float(np.max(S_union[n_reb:]
                                       [ten_present]))
                    checks.append(_check_entry(
                        "tendon_stress", max(0.0, val),
                        limits.sigma_p_max, basis_dependent=True))
                else:
                    checks.append(_skipped_entry(
                        "tendon_stress", "no sigma_p_max limit"))

            # Unbonded tendons: member-level values from the caller.
            for label, sp in (stage.get("unbonded_sigma_p")
                              or {}).items():
                if limits.sigma_p_max is not None:
                    checks.append(_check_entry(
                        f"tendon_stress[{label}]",
                        max(0.0, float(sp)), limits.sigma_p_max,
                        basis_dependent=False,
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
                    probe = _decompression_probe(
                        base_section.x_tendons[i_t],
                        base_section.y_tendons[i_t],
                        int(mat_idx_t[i_t]), coeffs, limits.c_dec)
                    checks.append({
                        "name": f"decompression[tendon {i_t}]",
                        "sigma_probe_MPa":
                            round(probe["sigma_probe"], 6),
                        "probe_at": (round(probe["x_probe"], 3),
                                     round(probe["y_probe"], 3)),
                        "c_dec_mm": limits.c_dec,
                        "passed": probe["passed"],
                        "basis_dependent": True,
                    })

            # Verdict (D4): a violated basis invalidates every
            # basis-dependent check and the stage verdict, while the
            # numeric values stay in the record as informative.
            stage_ok = True
            for c in checks:
                if c.get("skipped"):
                    continue
                if basis_violated and c["basis_dependent"]:
                    c["basis_valid"] = False
                if not c["passed"]:
                    stage_ok = False
                if "eta" in c:
                    all_etas.append((c["eta"], name, c["name"]))
            if basis_violated:
                stage_ok = False
            sr["checks"] = checks
            sr["limits"] = limits.name
            sr["verified"] = stage_ok
            all_verified = all_verified and stage_ok

        stage_results.append(sr)
        prev_state = state
        prev_resist = resist
        F_prev = F_cum

    governing = None
    if all_etas:
        eta, s_name, c_name = max(all_etas, key=lambda t: t[0])
        governing = {"eta": eta, "stage": s_name, "check": c_name}

    return {
        "type": "sls_staged",
        "x_ref": x_ref, "y_ref": y_ref,
        "stages": stage_results,
        "verified": bool(all_verified),
        "governing": governing,
    }
