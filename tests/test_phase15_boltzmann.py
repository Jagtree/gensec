# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
Phase 15 B — the constitutive memory must date the loads it receives.

Creep obeys Boltzmann superposition: the strain at time :math:`t` is the
sum, over every stress increment :math:`\Delta\sigma_j` applied at age
:math:`\tau_j`, of

.. math::

    \varepsilon_{\mathrm{imp}}(t)
        = \sum_j \Delta\sigma_j
          \left[ J(t, \tau_j) - J(t_a, \tau_j) \right] ,

so a *when* is as much a part of the input as a *how much*.  The walk
carries the history across intervals precisely because the emitted state
cannot say when a stress was applied — reading it back with the
instantaneous modulus gives the wrong number.

Before this phase the history was seeded once and thereafter received
only the viscous redistribution of each interval.  A ``load`` standing
**between** two intervals moved the mechanical strain plane and never
entered the history, so it flowed **zero days**: the section crept as if
the load had been applied at the very end of the walk.  Measured in
``handoff/12_10-GENSEC_LEDGER_RECAP.md`` section 7, re-measured in
``handoff/15_0`` on this tree.

Both tests below fail on the pre-Phase-15 tree, and the second one names
the missing term in closed form rather than pinning whatever the
implementation happens to produce.
"""
import os

import numpy as np
import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.losses import (_UNCAST_PLACEHOLDER_E,
                                  _zone_drying_geometry,
                                  covered_zone_slots)
from gensec.solver.section_state import materialize_view
from gensec.solver.sls import resolve_sls_moduli, sls_view
from gensec.solver.timeline import ConstructionTimeline

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLAGSHIP = os.path.join(_HERE, os.pardir, "examples",
                         "example_composite_losses.yaml")


def _demand_db(demands):
    r"""Build the SI demand database the timeline resolve expects."""
    return {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                        "My": d.get("My", 0.0)} for d in demands}


def _resolve(history, data):
    r"""
    Resolve one construction history against the flagship section.

    Parameters
    ----------
    history : list of dict
        The ``construction_history`` block to walk.
    data : dict
        The parsed flagship model, for section and rheology.

    Returns
    -------
    TimelineResolution
    """
    tl = ConstructionTimeline.from_block(
        history,
        losses_models=data.get("losses_models"),
        zone_rheology=data.get("zone_rheology"))
    return tl.resolve(data["section"], _demand_db(data["demands"]))


def _last_emission(resolution):
    r"""
    Per-zone eigenstrain plane emitted by the **last** interval.

    Returns
    -------
    dict
        ``{zone_index: numpy.ndarray of shape (3,)}``.
    """
    key = max(resolution.losses_ops)
    delta = resolution.losses_ops[key].get("bulk_plane_delta", {})
    return {int(z): np.asarray(v, dtype=float) for z, v in delta.items()}


def _move_load_to_the_end(history, demand_name):
    r"""
    Return a copy of *history* with one ``load`` moved past the last
    ``interval``.

    The load is applied to the same section, in the same order relative
    to every ``cast`` and ``grout``, and carries the same demand — the
    only thing that changes is how long it stands before the walk ends.
    """
    out = [dict(ev) for ev in history]
    hit = [i for i, ev in enumerate(out)
           if "load" in ev and ev["load"].get("demand") == demand_name]
    assert len(hit) == 1, f"expected exactly one load of {demand_name!r}"
    moved = out.pop(hit[0])
    last_interval = max(i for i, ev in enumerate(out) if "interval" in ev)
    out.insert(last_interval + 1, moved)
    return out


def _service_view_at_last_interval(data):
    r"""
    Rebuild the service view of the last interval, and its per-zone
    instantaneous moduli and providers.

    This is the same construction :func:`~gensec.solver.losses.
    compute_interval_losses` performs — repeated here on purpose, so the
    reference below is built from the *providers and the section*, not
    from the quantity under test.

    Returns
    -------
    tuple
        ``(solver, providers, ages_t0, ages_t, demand_now,
        demand_before)``.
    """
    from gensec.solver import losses as losses_mod

    seen = []
    original = losses_mod.compute_interval_losses

    def spy(base_section, state, **kwargs):
        result = original(base_section, state, **kwargs)
        seen.append((base_section, state, kwargs, result))
        return result

    losses_mod.compute_interval_losses = spy
    try:
        _resolve(data["construction_history"], data)
    finally:
        losses_mod.compute_interval_losses = original

    section, state, kwargs, _res = seen[-1]
    _s_prev, _st_prev, kw_prev, _r_prev = seen[-2]
    models = kwargs["models"]
    ages_t0 = kwargs["zone_ages_t0"]

    providers = {}
    for z, lm in models.items():
        A_c, u_dry = _zone_drying_geometry(section, z)
        p = lm.provider
        providers[z] = p if p.A_c is not None else p.with_geometry(A_c, u_dry)

    mod = {}
    covered = covered_zone_slots(section, mod)
    for z in range(1 + len(section.bulk_materials)):
        if z in covered:
            continue
        if z in models:
            # The same law the engine resolves, not a number.  Asking
            # the provider for E_c here while losses.py asks it for
            # service_law gave this reference a straight line and the
            # measurement a curve, and the two planes then differed by
            # ~37 % for reasons that had nothing to do with creep.
            mod[("zone", z)] = providers[z].service_law(ages_t0[z])
        elif not bool(np.asarray(state.bulk_active)[z]):
            mod[("zone", z)] = _UNCAST_PLACEHOLDER_E

    solver = FiberSolver(
        sls_view(materialize_view(section, state),
                 resolve_sls_moduli(section, mod)))
    return (solver, providers, ages_t0, kwargs["zone_ages_t"],
            tuple(kwargs["demand"]), tuple(kw_prev["demand"]))


# ======================================================================
#  1.  The load must not flow zero days
# ======================================================================

def test_load_between_intervals_creeps_for_its_own_duration():
    r"""
    Moving a ``load`` past the last interval must change the answer.

    The flagship applies ``G2`` after the 3-day interval and before the
    25457-day one, so ``G2`` stands under sustained load for the whole
    final interval.  Moving it *after* that interval leaves it zero days
    to creep.  Two histories that differ by 25457 days of sustained
    bending cannot emit the same viscous eigenstrain — and if they do,
    the load never reached the superposition integral.
    """
    data = load_yaml(_FLAGSHIP)
    history = data["construction_history"]

    emitted_between = _last_emission(_resolve(history, data))
    emitted_after = _last_emission(
        _resolve(_move_load_to_the_end(history, "G2"), data))

    assert set(emitted_between) == set(emitted_after)
    worst = max(
        float(np.abs(emitted_between[z] - emitted_after[z]).max())
        for z in emitted_between)
    assert worst > 1e-9, (
        "the viscous eigenstrain of the last interval is identical "
        "whether the load stands for 25457 days or for none: the load "
        "is not in the Boltzmann history at all"
    )


# ======================================================================
#  2.  The missing term, named in closed form
# ======================================================================

def test_load_contributes_its_closed_form_boltzmann_term():
    r"""
    The difference between the two histories is
    :math:`\sigma_{G2}\,[J(t_1,\tau)-J(t_0,\tau)]`, per zone.

    The reference is built from the rheological provider's own
    compliance and from an *incremental* service solve — the elastic
    response the section gives to the load alone, isolated as the
    difference of two solves on one view so that the self-equilibrated
    prestrain response cancels.  Neither quantity is read back from the
    emission under test.

    ``load`` does not advance the clock, so the age of application is
    exactly the age at the start of the following interval and
    :math:`E_c` is the modulus that same interval already uses.
    """
    data = load_yaml(_FLAGSHIP)
    history = data["construction_history"]

    (solver, providers, ages_t0, ages_t,
     demand_now, demand_before) = _service_view_at_last_interval(data)

    now = solver.solve_equilibrium(*demand_now, tol=1e-6, max_iter=100)
    before = solver.solve_equilibrium(*demand_before, tol=1e-6, max_iter=100)
    assert now["converged"] and before["converged"]
    d_plane = (np.array([now["eps0"], now["chi_x"], now["chi_y"]])
               - np.array([before["eps0"], before["chi_x"],
                           before["chi_y"]]))

    emitted_between = _last_emission(_resolve(history, data))
    emitted_after = _last_emission(
        _resolve(_move_load_to_the_end(history, "G2"), data))

    for z, provider in providers.items():
        a0, a1 = float(ages_t0[z]), float(ages_t[z])
        # The emission is the NEGATIVE of the eigenstrain (the kernel
        # adds the offset, the physics subtracts the eigenstrain), so
        # the expected difference carries the same minus sign.
        expected = -(provider.E_c(a0) * d_plane) * (
            provider.J(a1, a0) - provider.J(a0, a0))
        measured = emitted_between[z] - emitted_after[z]
        scale = max(float(np.abs(expected).max()), 1e-12)
        assert np.allclose(measured, expected, rtol=5e-3,
                           atol=1e-3 * scale), (
            f"zone {z}: the load's own creep term is "
            f"{expected} but the walk emits a difference of {measured}"
        )


if __name__ == "__main__":         # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
