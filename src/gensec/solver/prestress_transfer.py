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
