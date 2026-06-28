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
Pre-tensioning transfer and immediate elastic-shortening loss.

This module implements **Phase 2** of the prestress feature: the
two-state transfer of a *pre-tensioned* member, where the strands are
stressed against external abutments **before** the concrete is cast,
and then released onto the hardened concrete.

The whole computation rides on the Phase-1 locked-in-prestrain core in
:class:`~gensec.solver.integrator.FiberSolver`: a bonded
:class:`~gensec.geometry.fiber.Tendon` evaluates its constitutive law at
the **offset total strain**

.. math::

    \varepsilon_{\text{tot},i}
        = \varepsilon_{\text{sec},i} + \varepsilon_{\text{init},i}

while the displaced bulk it occupies is evaluated at the **section**
strain :math:`\varepsilon_{\text{sec},i}` alone.  No new solver physics
is introduced here; the elastic-shortening loss is **not** injected from
a formula — it *emerges* from section equilibrium.

Two reference states, one section
---------------------------------
The same :class:`~gensec.geometry.section.GenericSection` is read at two
strain planes, with the tendon prestrain referenced to the
**pre-transfer** state:

**State 0 — jacking (pre-transfer).**
    The strand is stretched to the jacking strain
    :math:`\varepsilon_{p0} = \sigma_{p0}/E_p` against the abutments;
    this is the value carried in ``Tendon.eps_pe``.  The concrete is
    unstrained, so the section strain plane is the trivial zero plane
    :math:`(\varepsilon_0,\chi_x,\chi_y)=(0,0,0)`.  The force the
    abutments react is the prestress resultant

    .. math::

        N_{\text{ab}} = \sum_i \sigma_p(\varepsilon_{p0,i})\,A_{p,i},
        \quad
        M_{x,\text{ab}} = \sum_i \sigma_p(\varepsilon_{p0,i})\,A_{p,i}
                           (y_i - y_{\text{ref}}),
        \;\dots

**State 1 — release (post-transfer).**
    The abutments are cut.  With no external restraint the section must
    carry the prestress internally, so it self-equilibrates against the
    external action present at transfer (zero by default, or self-weight
    :math:`(N_\text{ext}, M_{x,\text{ext}}, M_{y,\text{ext}})`).  The
    solver returns the post-transfer strain plane; the section strain at
    each tendon is the *elastic-shortening strain*, and the strand
    stress drops by

    .. math::

        \Delta\sigma_{p,i}
            = \sigma_p(\varepsilon_{p0,i})
            - \sigma_p(\varepsilon_{\text{sec},i} + \varepsilon_{p0,i})
            \;\overset{\text{elastic}}{=}\; -E_p\,\varepsilon_{\text{sec},i}.

Equilibrium, not a formula
--------------------------
For an **eccentric** tendon the released state has a non-zero curvature:
the moment of the net tendon force about the section reference must be
balanced by the bulk, so :math:`\chi \neq 0`.  This is why the transfer
solve must use the full uniaxial/biaxial equilibrium and **not** the
pure-axial fast path (see the note in
:func:`solve_pretension_transfer`).

Scope
-----
This phase models the *immediate elastic* loss only.  Anchorage set,
friction, relaxation and time-dependent (creep/shrinkage) losses are
deferred; the constitutive law and prestrain offset are the only inputs,
so a later phase can fold an additional prestrain decrement into
``eps_pe`` without touching this driver.

References
----------
- EN 1992-1-1:2004 §5.10.5.1 (immediate losses — elastic deformation
  of concrete).
- NTC 2018 §4.1.8.1.4 (perdite istantanee per deformazione elastica).
"""

from dataclasses import dataclass, field

import numpy as np

from .integrator import FiberSolver


@dataclass
class PretensionTransferResult:
    r"""
    Outcome of a single pre-tensioning transfer solve.

    All per-tendon arrays are index-aligned with
    ``section.tendons`` (and with the solver's tendon arrays).

    Attributes
    ----------
    eps0, chi_x, chi_y : float
        Post-transfer (released) strain plane at the section reference
        point :math:`(x_{\text{ref}}, y_{\text{ref}})`.
    converged : bool
        Whether the release equilibrium solve converged.
    iterations : int
        Newton/bisection iterations used by the release solve.
    N_ext, Mx_ext, My_ext : float
        External action assumed present at transfer [N, N·mm].
    N_abutment, Mx_abutment, My_abutment : float
        Prestress resultant reacted by the abutments in the pre-transfer
        (jacking) state [N, N·mm].
    sigma_p0 : numpy.ndarray
        Pre-transfer (jacking) strand stress
        :math:`\sigma_p(\varepsilon_{p0,i})` [MPa].
    sigma_p : numpy.ndarray
        Post-transfer strand stress
        :math:`\sigma_p(\varepsilon_{\text{sec},i}+\varepsilon_{p0,i})`
        [MPa].
    loss : numpy.ndarray
        Immediate elastic-shortening loss
        :math:`\Delta\sigma_{p,i} = \sigma_{p0,i} - \sigma_{p,i}`
        [MPa] (positive = loss of tension).
    eps_sec : numpy.ndarray
        Section strain at each tendon in the released state
        (the elastic-shortening strain, negative in compression).
    eps_tot : numpy.ndarray
        Total strain seen by each strand's law,
        :math:`\varepsilon_{\text{sec},i}+\varepsilon_{p0,i}`.
    Ap : numpy.ndarray
        Strand areas [mm²].

    Properties
    ----------
    P0 : float
        Total prestress force before transfer
        :math:`\sum_i \sigma_{p0,i} A_{p,i}` [N].
    P_after : float
        Total prestress force after transfer [N].
    loss_fraction : numpy.ndarray
        Per-tendon relative loss :math:`\Delta\sigma_{p,i}/\sigma_{p0,i}`.
    """

    eps0: float
    chi_x: float
    chi_y: float
    converged: bool
    iterations: int
    N_ext: float
    Mx_ext: float
    My_ext: float
    N_abutment: float
    Mx_abutment: float
    My_abutment: float
    sigma_p0: np.ndarray
    sigma_p: np.ndarray
    loss: np.ndarray
    eps_sec: np.ndarray
    eps_tot: np.ndarray
    Ap: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def P0(self):
        r"""Total prestress force before transfer [N]."""
        return float(np.sum(self.sigma_p0 * self.Ap))

    @property
    def P_after(self):
        r"""Total prestress force after transfer [N]."""
        return float(np.sum(self.sigma_p * self.Ap))

    @property
    def loss_fraction(self):
        r"""Per-tendon relative loss :math:`\Delta\sigma_p/\sigma_{p0}`."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.sigma_p0 != 0.0,
                            self.loss / self.sigma_p0, 0.0)


def solve_pretension_transfer(section, *, N_ext=0.0, Mx_ext=0.0,
                              My_ext=0.0, x_ref=None, y_ref=None,
                              tol=1e-6, max_iter=100):
    r"""
    Solve the pre-tensioning transfer and immediate elastic-shortening
    loss for ``section``.

    The section's tendons must already carry their **jacking** prestrain
    in ``Tendon.eps_pe`` (i.e. :math:`\varepsilon_{p0}=\sigma_{p0}/E_p`
    after seating/friction, which are out of scope here).  The driver
    reads the jacking state, releases the abutments by solving the
    self-equilibrium of the section, and reports the loss that *emerges*
    from that equilibrium.

    Parameters
    ----------
    section : GenericSection
        A meshed section whose ``tendons`` carry ``eps_pe`` set to the
        jacking strain.
    N_ext, Mx_ext, My_ext : float, optional
        External action assumed present at transfer (e.g. self-weight)
        [N, N·mm].  Default ``0`` — pure prestress transfer.
    x_ref, y_ref : float or None, optional
        Reference point for the strain plane.  Default the section
        centroid (forwarded to :class:`FiberSolver`).
    tol : float, optional
        Force tolerance for the release equilibrium solve [N].
        Default ``1e-6``.
    max_iter : int, optional
        Maximum solver iterations.  Default ``100``.

    Returns
    -------
    PretensionTransferResult

    Raises
    ------
    ValueError
        If the section carries no tendons.

    Notes
    -----
    **Eccentric transfer requires a moment-aware equilibrium solve.**
    With :math:`N_\text{ext}=M_\text{ext}=0` and an eccentric tendon, the
    target forces are all zero, yet the *released* state has
    :math:`\chi\neq 0`: the moment of the net tendon force about the
    reference must be balanced by the bulk.  A pure-axial solve
    (:math:`\chi=0`) that only checks the axial residual would report a
    spurious "converged" state.  :func:`FiberSolver.solve_equilibrium`
    must therefore reject a pure-axial solution whose moment residual
    exceeds tolerance and fall through to the uniaxial/biaxial solve;
    this driver relies on that guard.

    Examples
    --------
    >>> # section built with a tendon whose eps_pe = sigma_p0 / Ep
    >>> res = solve_pretension_transfer(section)        # doctest: +SKIP
    >>> res.loss                                        # doctest: +SKIP
    array([39.72])
    """
    tendons = getattr(section, "tendons", [])
    if not tendons:
        raise ValueError(
            "solve_pretension_transfer: section carries no tendons. "
            "Build the section with a non-empty `tendons` list whose "
            "Tendon.eps_pe holds the jacking strain."
        )

    solver = FiberSolver(section, x_ref=x_ref, y_ref=y_ref)

    # ---- State 0: jacking (pre-transfer), trivial zero strain plane ----
    # The section strain is identically zero; the only internal force is
    # the strand resultant the abutments react.
    N_ab, Mx_ab, My_ab = solver.integrate(0.0, 0.0, 0.0)
    res0 = solver.get_fiber_results(0.0, 0.0, 0.0)
    sigma_p0 = res0["tendons"]["sigma"].copy()

    # ---- State 1: release (post-transfer), self-equilibrium ----
    sol = solver.solve_equilibrium(
        N_ext, Mx_ext, My_ext, tol=tol, max_iter=max_iter)
    res1 = solver.get_fiber_results(
        sol["eps0"], sol["chi_x"], sol["chi_y"])
    t = res1["tendons"]
    sigma_p = t["sigma"].copy()

    return PretensionTransferResult(
        eps0=sol["eps0"], chi_x=sol["chi_x"], chi_y=sol["chi_y"],
        converged=sol["converged"], iterations=sol["iterations"],
        N_ext=N_ext, Mx_ext=Mx_ext, My_ext=My_ext,
        N_abutment=N_ab, Mx_abutment=Mx_ab, My_abutment=My_ab,
        sigma_p0=sigma_p0, sigma_p=sigma_p,
        loss=sigma_p0 - sigma_p,
        eps_sec=t["eps_sec"].copy(), eps_tot=t["eps_tot"].copy(),
        Ap=t["A"].copy(),
    )


# ---------------------------------------------------------------------------
#  Closed-form validator (single-tendon transformed-section elasticity)
# ---------------------------------------------------------------------------

def elastic_shortening_loss(Ec, Ep, Ac, Ic, Ap, e, sigma_p0):
    r"""
    Exact immediate elastic-shortening loss for a single embedded tendon.

    Closed-form solution of the released-section equilibrium under
    **linear-elastic** concrete and tendon, for one bonded tendon at
    eccentricity :math:`e` from the gross (= reference) centroid, no
    passive reinforcement, and no external action.  This is the value a
    linear-elastic fiber model reproduces *exactly* (it is the same
    linear algebra), so it is the right reference for a regression test.

    Equilibrium of the released section (tension positive), with the
    embedded tendon's net force
    :math:`F_p = \bigl[E_p(\varepsilon_{p0}+s)-E_c\,s\bigr]A_p`
    where :math:`s=\varepsilon_0+\chi e` is the concrete strain at the
    tendon, gives

    .. math::

        E_c A_c\,\varepsilon_0 + F_p = 0,
        \qquad
        E_c I_c\,\chi + F_p\,e = 0,

    whence, with
    :math:`k = \dfrac{1}{E_c}\!\left(\dfrac{1}{A_c}+\dfrac{e^2}{I_c}\right)`,

    .. math::

        F_p = \frac{E_p\,\varepsilon_{p0}\,A_p}
                   {1 + (E_p - E_c)\,A_p\,k},
        \qquad
        s = -F_p\,k,
        \qquad
        \Delta\sigma_p = -E_p\,s = E_p\,F_p\,k .

    The denominator coupling term :math:`(E_p-E_c)A_p k` is the tendon
    *transformed-area* contribution.  The familiar engineering
    approximation :math:`\Delta\sigma_p \approx (E_p/E_c)\,\sigma_c`
    drops it (computing :math:`\sigma_c` on the bare concrete section),
    which slightly **overestimates** the loss; the form above is exact.

    Parameters
    ----------
    Ec : float
        Concrete modulus at transfer [MPa].
    Ep : float
        Tendon modulus [MPa].
    Ac : float
        Gross concrete area about the reference centroid [mm²].
    Ic : float
        Gross concrete second moment of area about the reference
        centroid [mm⁴].  Ignored when ``e == 0``.
    Ap : float
        Tendon area [mm²].
    e : float
        Tendon eccentricity from the reference centroid [mm].
    sigma_p0 : float
        Jacking (pre-transfer) strand stress [MPa].

    Returns
    -------
    dict
        Keys: ``loss`` (Δσ_p [MPa]), ``sigma_p_after`` [MPa],
        ``eps_sec`` (concrete strain at tendon), ``eps0``, ``chi``,
        ``Fp_net`` (post-transfer net tendon force [N]).

    Notes
    -----
    For a regression test against the meshed fiber model, pass the
    section's **meshed** ``Ac`` and ``Ic`` (summed from the fiber
    arrays) rather than the continuum values, to isolate the solver
    from the :math:`\mathcal{O}(1/n_y^2)` strip-inertia discretization.
    """
    eps_p0 = sigma_p0 / Ep
    P0 = Ep * eps_p0 * Ap
    k = (1.0 / Ac + (e * e / Ic if e != 0.0 else 0.0)) / Ec
    Fp_net = P0 / (1.0 + (Ep - Ec) * Ap * k)
    s = -Fp_net * k
    return {
        "loss": -Ep * s,
        "sigma_p_after": Ep * (eps_p0 + s),
        "eps_sec": s,
        "eps0": -Fp_net / (Ec * Ac),
        "chi": (-Fp_net * e / (Ec * Ic)) if e != 0.0 else 0.0,
        "Fp_net": Fp_net,
    }


# ---------------------------------------------------------------------------
#  Closed-form validator — sequential post-tension elastic shortening
#  (general: N tendons, arbitrary eccentricities, arbitrary order)
# ---------------------------------------------------------------------------


def sequential_posttension_loss(
    Ec, Es, Ep, *,
    Ac, Sc, Ic,
    tendons,
    rebars=(),
    order=None,
    base_N=0.0, base_M=0.0,
    y_ref=0.0,
    scheme="one_pass",
    coupled_tol=1e-12, coupled_max_iter=50,
):
    r"""
    Exact immediate elastic-shortening losses for **sequential**
    bonded post-tensioning, in closed transformed-section form.

    Independent reference for the Phase-4 fiber-method driver
    (``solve_posttension_sequence``): pure planar transformed-section
    elasticity, no fiber model.  It is the multi-tendon, sequential
    generalisation of :func:`elastic_shortening_loss`, and like that
    function it is the value a **linear-elastic** fiber model
    reproduces exactly (same linear algebra), hence the right
    regression anchor.

    Physics encoded (the post-tension / pre-tension asymmetry)
    ----------------------------------------------------------
    A post-tensioned tendon is **not** a section element while it is
    being stressed — it is a free cable, so it is jacked to its target
    stress exactly and the jack reacts against the section as the
    external action :math:`(N_j, M_j) = (-P_j,\,-P_j e_j)`.  Therefore:

    - the tendon being stressed at step *j* suffers **no** loss from
      its own stressing (it is not strain-compatible with the
      concrete at that instant);
    - every tendon *i* stressed **before** *j* and already anchored
      takes the concrete strain increment at its level produced by
      stressing *j*, and is debited
      :math:`\Delta\sigma_i = E_p\,\Delta\varepsilon_{c,i}`.

    Contrast with pre-tensioning, where all tendons are strain-
    compatible from the start and lose **simultaneously** at release
    (the loss *emerges* from one equilibrium solve — the coupling term
    :math:`(E_p-E_c)A_p k` in :func:`elastic_shortening_loss`).  Here
    the loss cannot emerge; it must be read and debited step by step.

    Transformed section per step
    ----------------------------
    The strain-plane increment from stressing tendon *j* is found on
    the section **homogenised to the current state** — concrete +
    passive rebars + tendons **already grouted/bonded**, but **not**
    the free tendons (incl. *j* itself).  With axial/coupling/flexural
    rigidities :math:`(EA, ES, EI)` about ``y_ref`` (tension positive,
    :math:`M` about the reference, sagging convention inherited from
    the caller):

    .. math::

        \begin{bmatrix} EA & ES \\ ES & EI \end{bmatrix}
        \begin{bmatrix} \Delta\varepsilon_0 \\ \Delta\chi \end{bmatrix}
        = \begin{bmatrix} N_j \\ M_j \end{bmatrix},
        \qquad
        \Delta\varepsilon_{c,i} = \Delta\varepsilon_0 + \Delta\chi\,
        (y_i - y_{\mathrm{ref}}) .

    ``ES`` is non-zero in general (the homogenised centroid differs
    from ``y_ref`` and shifts as tendons are bonded); the 2×2 solve is
    exact for arbitrary eccentricities, so no symmetry is assumed.

    In this **post-tension** model grouting is deferred to the
    *grouting stage* (the engine's action→element transition), so by
    default no tendon is bonded during stressing and the homogenised
    section is concrete + rebars throughout.  The ``bonded_before``
    hook on each tendon entry lets a caller model "stress, grout, then
    stress the next" — a freshly grouted tendon then participates in
    the transformed section for subsequent steps.

    Parameters
    ----------
    Ec, Es, Ep : float
        Concrete, passive-steel, prestressing-steel moduli [MPa].
    Ac : float
        Gross concrete area [mm²].
    Sc : float
        Gross concrete first moment of area about ``y_ref``
        (:math:`\int (y-y_{\mathrm{ref}})\,dA`) [mm³].  Zero iff
        ``y_ref`` is the concrete centroid.
    Ic : float
        Gross concrete second moment of area about ``y_ref`` [mm⁴].
    tendons : sequence of dict
        One entry per tendon: ``{"y": float, "Ap": float,
        "sigma_p0": float, "bonded_before": int or None}``.  ``y`` is
        the tendon level [mm]; ``sigma_p0`` the jacking stress at this
        tendon [MPa] (already net of friction / anchorage draw-in —
        member-level losses are inputs); ``bonded_before`` optionally
        names the step index at and after which this tendon counts as
        bonded in the transformed section (default: never during the
        sequence — grouting is a later stage).
    rebars : sequence of dict, optional
        Passive bars ``{"y": float, "As": float}`` [mm, mm²]; always
        part of the transformed section.
    order : sequence of int or None, optional
        Stressing order as indices into ``tendons``.  Default
        ``range(len(tendons))`` (declaration order).
    base_N, base_M : float, optional
        Sollecitazione present at transfer (e.g. self-weight) [N, N·mm
        about ``y_ref``].  Applied once, before the sequence, on the
        bare transformed section; it does **not** debit any tendon
        (no tendon is anchored yet) but it sets the concrete strain
        datum used by the grouting-reference computation.
    y_ref : float, optional
        Reference level for moments and eccentricities [mm].
        Default 0.
    scheme : {"one_pass", "coupled"}, optional
        ``"one_pass"`` (default): each step debits once, no
        re-iteration (EC2 §5.10.5.1 level).  ``"coupled"``:
        fixed-point iteration capturing the second-order feedback
        (a debit reduces the action, which reduces the shortening).
    coupled_tol, coupled_max_iter : float, int, optional
        Convergence control for ``scheme="coupled"`` (max ‖Δσ‖
        change between sweeps [MPa]).

    Returns
    -------
    dict
        ``loss`` : ndarray, per-tendon cumulative Δσ [MPa] (debit > 0
        means a stress reduction);
        ``sigma_p_after`` : ndarray, effective stress after the whole
        sequence [MPa];
        ``eps_ref_grout`` : ndarray, concrete strain at each tendon at
        the END of the sequence (the datum the engine's grouting
        reconciliation must use);
        ``order`` : the order actually applied;
        ``scheme`` : the scheme used.

    Notes
    -----
    Sign convention is the caller's: pass ``Sc, Ic, y, base_M`` in one
    consistent frame about ``y_ref``.  For a check against the meshed
    fiber model, pass the **meshed** ``Ac, Sc, Ic`` (summed from the
    fiber arrays) to remove the :math:`\mathcal{O}(1/n_y^2)` strip
    discretization from the comparison, exactly as advised for
    :func:`elastic_shortening_loss`.
    """
    tendons = [dict(t) for t in tendons]
    nt = len(tendons)
    if order is None:
        order = list(range(nt))
    order = list(order)

    yv = np.array([t["y"] for t in tendons], dtype=float)
    Ap = np.array([t["Ap"] for t in tendons], dtype=float)
    sp0 = np.array([t["sigma_p0"] for t in tendons], dtype=float)
    bonded_before = [t.get("bonded_before", None) for t in tendons]

    rb_y = np.array([r["y"] for r in rebars], dtype=float)
    rb_A = np.array([r["As"] for r in rebars], dtype=float)

    def _transformed(bonded_mask):
        r"""(EA, ES, EI) about y_ref for the current bonded set."""
        EA = Ec * Ac
        ES = Ec * Sc
        EI = Ec * Ic
        # passive rebars (always present)
        for yi, Ai in zip(rb_y, rb_A):
            d = yi - y_ref
            EA += Es * Ai
            ES += Es * Ai * d
            EI += Es * Ai * d * d
        # bonded tendons (transformed with (Ep - Ec): they displace
        # concrete already counted in Ac/Sc/Ic)
        for k in range(nt):
            if not bonded_mask[k]:
                continue
            d = yv[k] - y_ref
            dE = Ep - Ec
            EA += dE * Ap[k]
            ES += dE * Ap[k] * d
            EI += dE * Ap[k] * d * d
        return EA, ES, EI

    def _solve_plane(EA, ES, EI, N, M):
        det = EA * EI - ES * ES
        de0 = (EI * N - ES * M) / det
        dchi = (-ES * N + EA * M) / det
        return de0, dchi

    def _run_sequence(sigma_eff):
        r"""
        One forward sweep.  ``sigma_eff`` is the current best estimate
        of each tendon's effective stress (used only by the coupled
        scheme to reduce the action by the accrued debit).  Returns
        the per-tendon debit array and the final concrete strain at
        each tendon.
        """
        debit = np.zeros(nt)
        anchored = np.zeros(nt, dtype=bool)   # stressed & locked
        bonded = np.zeros(nt, dtype=bool)     # grouted -> in section
        # concrete strain at each tendon level, accumulated
        eps_c = np.zeros(nt)

        # base sollecitazione on the bare transformed section
        EA, ES, EI = _transformed(bonded)
        if base_N != 0.0 or base_M != 0.0:
            de0, dchi = _solve_plane(EA, ES, EI, base_N, base_M)
            eps_c += de0 + dchi * (yv - y_ref)

        for step, j in enumerate(order):
            # bond any tendon whose grouting precedes this step
            for k in range(nt):
                if (bonded_before[k] is not None
                        and bonded_before[k] <= step):
                    bonded[k] = True
            EA, ES, EI = _transformed(bonded)

            # action of stressing tendon j (reduced by its own accrued
            # debit only in the coupled scheme; one_pass uses sp0)
            Pj = sigma_eff[j] * Ap[j]
            Nj = -Pj
            Mj = -Pj * (yv[j] - y_ref)
            de0, dchi = _solve_plane(EA, ES, EI, Nj, Mj)
            d_eps = de0 + dchi * (yv - y_ref)
            eps_c += d_eps

            # debit every already-anchored tendon (NOT j itself)
            for i in range(nt):
                if anchored[i] and i != j:
                    debit[i] += -Ep * d_eps[i]   # shortening<0 -> debit>0

            anchored[j] = True

        return debit, eps_c

    if scheme == "one_pass":
        debit, eps_c = _run_sequence(sp0.copy())
    elif scheme == "coupled":
        sigma_eff = sp0.copy()
        prev = None
        for _ in range(coupled_max_iter):
            debit, eps_c = _run_sequence(sigma_eff)
            sigma_eff = sp0 - debit
            if prev is not None and np.max(np.abs(sigma_eff - prev)) < coupled_tol:
                break
            prev = sigma_eff.copy()
    else:
        raise ValueError(
            f"sequential_posttension_loss: unknown scheme {scheme!r}; "
            f"use 'one_pass' or 'coupled'."
        )

    return {
        "loss": debit,
        "sigma_p_after": sp0 - debit,
        "eps_ref_grout": eps_c,
        "order": order,
        "scheme": scheme,
    }
