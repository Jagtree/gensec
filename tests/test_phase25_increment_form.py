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
Phase 25 — the increment form of the post-tensioning driver (D1c).

What is under test
------------------
:func:`~gensec.solver.posttension.solve_posttension_sequence` used to
solve **each jack's action alone** and sum the planes it read back.
That is exact when — and only when — the section is linear elastic, and
the driver neither verified the premise nor declared it.  The exact
form for any constitutive law is the other one, which the driver's own
docstring already named: solve the **cumulative** action set and
difference against the previous step.

The defect is not that the linear case is wrong.  There the two forms
are the same algebra, and Phase 22 checked the driver against
:func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`
in closed form.  The defect is that the premise is assumed silently,
and that on a nonlinear law the answer is wrong by an amount measured
at the opening of this phase: on a 400 x 1200 girder with three tendons
:math:`A_p = 1500` and the EN 1992-1-1 §3.1.5 service law, the final
:math:`\sigma_p` moves **5.9 %** and the elastic-shortening loss
**79 %**; at the compression a realistic member reaches
(:math:`\varepsilon_c` between :math:`-3\cdot10^{-4}` and
:math:`-6\cdot10^{-4}`), 0.19 ÷ 0.70 % and 9.6 ÷ 18.8 %.

How these fixtures are written
------------------------------
Nothing Phase 25 introduces is imported at module scope.  New names are
fetched with :func:`getattr` and their absence turned into an
**assertion**, so a red says "this behaviour is missing" and never
:class:`ImportError` or :class:`AttributeError` — a red that says "this
name does not exist" demonstrates that a name is missing, not that a
defect is present (``21_0 §5``).

**The reference is not the driver compared with itself.**  Two
independent references are written out here:

:func:`_closed_form_linear`
    Transformed-section algebra: assemble :math:`EA, ES, EI` once, with
    every bonded tendon transformed by :math:`(E_p - E_c)A_p`, invert
    the 2x2 per jack.  Valid only for a constant modulus, and used only
    there.

:func:`_cumulative_reference`
    A fibre integration written in this file, with its own Newton loop
    over :math:`(\varepsilon_0, \chi_x)` on the raw ``A_fibers`` /
    ``y_fibers`` arrays.  It does not call
    :class:`~gensec.solver.integrator.FiberSolver` and it does not call
    the driver.  At every step it applies the **cumulative** action set,
    reads the total plane and differences it against the previous
    total; a tendon that becomes bonded at step *s* enters the
    integration carrying :math:`-\varepsilon_c^{(s-1)}` as its initial
    strain, so that it resists only what happened after it joined, and
    a tendon bonded before the sequence began enters at
    :math:`\sigma_{pe}/E_p - \varepsilon_{c,\text{declared}}`, so that
    it carries the stress it already had.

:func:`_cumulative_reference` is checked against
:func:`_closed_form_linear` on a constant modulus by
:func:`test_reference_agrees_with_closed_form_on_a_linear_law`, which
is green on both trees.  Without that fixture the nonlinear numbers
below would rest on an unverified reference.
"""

import numpy as np
import pytest

from gensec.geometry.fiber import Tendon
from gensec.geometry.section import RectSection
import gensec.solver.posttension as pt


# ---------------------------------------------------------------------------
#  Declared geometry and materials — 22_0 §1.1
# ---------------------------------------------------------------------------

B, H, N_Y = 400.0, 1200.0, 240
AP, SP0, EP = 1500.0, 1300.0, 195000.0
Y_TENDONS = (150.0, 200.0, 250.0)

FCK = 35.0                              # C35/45
ECM = 34077.14619918933                 # fben2(35).ecm, pinned

SIGMA_PE = 1100.0
EPS0_EX, CHIX_EX = -3.0e-4, 1.5e-7
X_REF_DECL, Y_REF_DECL = 0.0, 0.0
AGE_C, AGE_P = 14600.0, 14200.0
NAN = float("nan")


class _LinearElastic:
    r"""Constant-modulus law :math:`\sigma = E\varepsilon`."""

    def __init__(self, E, eps_lim=1.0):
        self.E = float(E)
        self._lim = float(eps_lim)

    @property
    def eps_min(self):
        return -self._lim

    @property
    def eps_max(self):
        return self._lim

    def stress(self, eps):
        return self.E * eps

    def stress_array(self, eps):
        return self.E * np.asarray(eps, dtype=float)

    def tangent(self, eps):
        return self.E

    def tangent_array(self, eps):
        return np.full_like(np.asarray(eps, dtype=float), self.E)

    def content_token(self):
        from gensec.materials.base import digest_bytes, encode_value
        return digest_bytes(
            encode_value("tests.test_phase25._LinearElastic"),
            encode_value(self.E), encode_value(self._lim))


class _EC2ServiceLaw:
    r"""
    EN 1992-1-1 §3.1.5 expression (3.14), **through its one home**.

    The formula is not written here.  It lives in
    :func:`~gensec.materials.ec2_properties.en2sargin`, and the
    constants come from :class:`~gensec.materials.ec2_properties.fben2`,
    which is the module that owns the normative document.  This class
    is an adapter and nothing else: it converts sign convention, adds
    the derivative the clause does not state, and presents the
    ``stress_array`` / ``tangent_array`` surface the fiber solver
    consumes.

    Sign
    ----
    ``en2sargin`` takes compression **positive** and returns
    compression **positive**; GenSec takes compression negative.  So

    .. math::

        \sigma_{\text{GenSec}}(\varepsilon) = -S(-\varepsilon),
        \qquad
        \frac{d\sigma_{\text{GenSec}}}{d\varepsilon}(\varepsilon)
            = S'(-\varepsilon),

    with :math:`S` the Eurocode function.

    The derivative
    --------------
    EN 1992-1-1 states :math:`\sigma(\varepsilon)` and not
    :math:`d\sigma/d\varepsilon`, so the tangent has no normative home
    to be taken from.  It is differentiated here in closed form,

    .. math::

        S'(e) = \frac{f_{cm}}{\varepsilon_{c1}}\,
            \frac{(k - 2\eta)\bigl[1 + (k-2)\eta\bigr]
                  - (k\eta - \eta^{2})(k-2)}
                 {\bigl[1 + (k-2)\eta\bigr]^{2}},
        \qquad \eta = e/\varepsilon_{c1},

    and it is **checked against a central difference of**
    ``en2sargin`` itself, so that the derivative cannot drift away from
    the function it differentiates without a red.

    Outside :math:`|\varepsilon| > \varepsilon_{cu1}` the Eurocode
    function returns zero; the tangent returns zero there too.  A law
    whose stress is flat while its tangent is not is not a law, and a
    Newton loop on it does not converge — measured, in the first
    version of this probe.

    Consequence, stated rather than hidden
    --------------------------------------
    Going through the normative surface is the opposite of the rule
    :func:`_retrofit_laws` follows (``24_0 §3``): these keys will move
    if an EN 1992-1-1 constant moves, which is a different fact from
    the one they watch.  The mitigation is that
    ``d1c.law.tangent_ratio[...]`` is recorded first and separately: if
    a normative constant moves, those five keys move with it and say
    so, and the rest of the ``d1c.*`` battery is read in that light.

    Parameters
    ----------
    fck : float, optional
        Characteristic cylinder strength [MPa].  Default 35, the
        C35/45 of every published measurement of this defect family.
    ls, loadtype, TypeConc : str, optional
        Passed through to :class:`fben2`.
    """

    def __init__(self, fck=35.0, ls="F", loadtype="LT", TypeConc="N"):
        from gensec.materials.ec2_properties import fben2
        self.props = fben2(float(fck), ls, loadtype, TypeConc)
        self.fcm = float(self.props.fcm)
        self.Ecm = float(self.props.ecm)
        self.eps_c1 = float(self.props.eps_c1)
        self.eps_cu1 = float(self.props.eps_cu1)
        # Read off expression (3.14) as ``en2sargin`` forms it; the
        # finite-difference check is what keeps the two tied.
        #
        # F-P26-0: this line mirrored the law's own defect, dividing by
        # the 28-day mean strength while the stress scales with
        # f_cm(t).  Both sides were wrong in the same way, so the
        # finite-difference check agreed and proved nothing away from
        # 28 days.  The divisor is now the age-matched f_cm, as in
        # EN 1992-1-1 (3.14); at the 28-day age this class is built at,
        # the two are the same float and no measurement moves.
        self.k = (1.05 * self.Ecm * abs(self.eps_c1)
                  / float(self.props.fcm))

    @property
    def eps_min(self):
        return -self.eps_cu1

    @property
    def eps_max(self):
        return self.eps_cu1

    def _S(self, e):
        r"""``en2sargin`` in its own sign convention, vectorised."""
        from gensec.materials.ec2_properties import en2sargin
        e = np.asarray(e, dtype=float)
        flat = e.ravel()
        out = np.array([en2sargin(float(v), self.props) for v in flat])
        return out.reshape(e.shape)

    def _dS(self, e):
        r"""Closed-form derivative of ``en2sargin``, same convention."""
        e = np.asarray(e, dtype=float)
        out = np.zeros_like(e)
        inside = np.abs(e) <= abs(self.eps_cu1)
        if np.any(inside):
            eta = e[inside] / self.eps_c1
            den = 1.0 + (self.k - 2.0) * eta
            num = self.k * eta - eta * eta
            d_num = (self.k - 2.0 * eta) / self.eps_c1
            d_den = (self.k - 2.0) / self.eps_c1
            out[inside] = self.fcm * (d_num * den - num * d_den) / (den * den)
        return out

    def stress(self, eps):
        return float(self.stress_array(np.asarray([eps], dtype=float))[0])

    def stress_array(self, eps):
        return -self._S(-np.asarray(eps, dtype=float))

    def tangent(self, eps):
        return float(self.tangent_array(np.asarray([eps], dtype=float))[0])

    def tangent_array(self, eps):
        return self._dS(-np.asarray(eps, dtype=float))

    def content_token(self):
        r"""Declared content: the constants the law was built from."""
        from gensec.materials.base import digest_bytes, encode_value
        return digest_bytes(
            encode_value("ec2_properties.en2sargin"),
            encode_value(self.fcm), encode_value(self.Ecm),
            encode_value(self.eps_c1), encode_value(self.eps_cu1))

def _section(law):
    r"""400 x 1200 rectangle, three tendons at y = 150 / 200 / 250."""
    strand = _LinearElastic(EP)
    tendons = [Tendon(y=float(yt), x=B / 2.0, material=strand, Ap=AP,
                      eps_pe=0.0) for yt in Y_TENDONS]
    return RectSection(B=B, H=H, bulk_material=law, rebars=[],
                       n_fibers_y=N_Y, tendons=tendons)


def _declaration_cls():
    r"""The declaration object, or ``None`` on a tree without it."""
    return getattr(pt, "PreexistingState", None)


def _declare(sigma_pe, eps0=EPS0_EX, chi_x=CHIX_EX):
    cls = _declaration_cls()
    assert cls is not None, (
        "gensec.solver.posttension carries no PreexistingState.")
    return cls(sigma_pe=dict(sigma_pe), eps0=float(eps0),
               chi_x=float(chi_x), chi_y=0.0,
               x_ref=X_REF_DECL, y_ref=Y_REF_DECL,
               concrete_age_days=AGE_C,
               tendon_age_days={int(k): AGE_P for k in sigma_pe})


# ---------------------------------------------------------------------------
#  Reference 1 — transformed section, valid only for a constant modulus
# ---------------------------------------------------------------------------

def _closed_form_linear(section, Ec, order, sigma_p0, *, preexisting=(),
                        sigma_pe=None, eps_plane=(0.0, 0.0),
                        bonded_before=None):
    r"""
    Transformed-section reference for a **linear** section.

    .. math::

        \begin{bmatrix} EA & ES \\ ES & EI \end{bmatrix}
        \begin{bmatrix} \Delta\varepsilon_0 \\ \Delta\chi \end{bmatrix}
        = \begin{bmatrix} N_j \\ M_j \end{bmatrix},
        \qquad \Delta\sigma_{p,i} = -E_p\,\Delta\varepsilon_{c,i}.

    The transformed properties are rebuilt at every step from the set
    of tendons bonded at that step, so an intra-sequence injection
    (*bonded_before*) is represented.  Superposition is used freely:
    that is legitimate here and only here, because the law is linear.

    Returns
    -------
    dict
        ``sigma_p_after``, ``loss``, ``eps_c``.
    """
    y_ref = section.y_centroid
    A = np.asarray(section.A_fibers, dtype=float)
    yf = np.asarray(section.y_fibers, dtype=float) - y_ref
    Ac, Sc, Ic = (float(np.sum(A)), float(np.sum(A * yf)),
                  float(np.sum(A * yf * yf)))
    yv = np.asarray(section.y_tendons, dtype=float)
    Ap = np.asarray(section.A_tendons, dtype=float)
    d = yv - y_ref
    nt = yv.size
    pre = {int(p) for p in preexisting}
    sched = {int(k): int(v) for k, v in (bonded_before or {}).items()}

    def _props(bonded):
        EA, ES, EI = Ec * Ac, Ec * Sc, Ec * Ic
        for k in sorted(bonded):
            dE = EP - Ec
            EA += dE * Ap[k]
            ES += dE * Ap[k] * d[k]
            EI += dE * Ap[k] * d[k] * d[k]
        return EA, ES, EI

    sig = np.array(sigma_p0, dtype=float)
    for k, v in (sigma_pe or {}).items():
        sig[int(k)] = float(v)

    e0_ex, chix_ex = eps_plane
    eps_c = (e0_ex + chix_ex * (yv - Y_REF_DECL)) if pre else np.zeros(nt)
    eps_c = np.asarray(eps_c, dtype=float).copy()

    debit = np.zeros(nt)
    anchored = np.zeros(nt, dtype=bool)
    for k in pre:
        anchored[k] = True

    for step, j in enumerate(order):
        bonded = set(pre) | {t for t, s in sched.items() if s <= step}
        EA, ES, EI = _props(bonded)
        P = float(sig[j] * Ap[j])
        Nj, Mj = -P, -P * d[j]
        det = EA * EI - ES * ES
        d_e0 = (EI * Nj - ES * Mj) / det
        d_chi = (EA * Mj - ES * Nj) / det
        d_eps = d_e0 + d_chi * d
        eps_c += d_eps
        mask = anchored.copy()
        mask[j] = False
        debit[mask] += -EP * d_eps[mask]
        anchored[j] = True

    return {"sigma_p_after": sig - debit, "loss": debit, "eps_c": eps_c}


# ---------------------------------------------------------------------------
#  Reference 2 — fibre integration with its own Newton loop, any law
# ---------------------------------------------------------------------------

def _cumulative_reference(section, law, order, sigma_p0, *,
                          preexisting=(), sigma_pe=None,
                          eps_plane=(0.0, 0.0), bonded_before=None,
                          tol=1e-11, max_iter=200):
    r"""
    The **cumulative** increment form, integrated here from scratch.

    Independent of :class:`~gensec.solver.integrator.FiberSolver` and of
    the driver: the bulk is the raw ``A_fibers`` / ``y_fibers`` arrays,
    the residual is assembled by hand and driven to zero by a Newton
    loop on :math:`(\varepsilon_0, \chi_x)`.

    At step *k* the residual is

    .. math::

        \mathbf{r} =
        \begin{bmatrix}
          \int \sigma_c(\varepsilon)\,dA
            + \sum_{i\in\mathcal{B}} \bigl[\sigma_p(\varepsilon_i
              + \varepsilon_{\text{init},i})
              - \sigma_c(\varepsilon_i)\bigr] A_{p,i} - N_k \\[4pt]
          \int \sigma_c(\varepsilon)\,y'\,dA
            + \sum_{i\in\mathcal{B}} [\cdots] A_{p,i}\,y'_i - M_k
        \end{bmatrix},

    with :math:`\varepsilon = \varepsilon_0 + \chi_x y' +
    \varepsilon_{\text{bulk}}`, :math:`\varepsilon_{\text{bulk}}` the
    declared plane of the existing concrete, :math:`\mathcal{B}` the
    tendons bonded at that step, and :math:`(N_k, M_k)` the
    **cumulative** action set plus the resultant the declared state
    itself carries.  The subtraction of :math:`\sigma_c` at the tendon
    is the displaced-bulk correction the fibre model makes.

    The initial strains, all directly computable from the declaration:

    * a tendon bonded before the sequence: :math:`\varepsilon_{\text{init}}
      = \sigma_{pe}/E_p - \varepsilon_{c,\text{declared}}`, so that its
      stress at the declared plane is :math:`\sigma_{pe}`;
    * a tendon injected at step *s*: :math:`\varepsilon_{\text{init}} =
      -\varepsilon_c^{(s-1)}`, so that it resists only what happens
      after it joins.  Its jacking force is already in the cumulative
      action set and must not be counted twice.

    Returns
    -------
    dict
        ``sigma_p_after``, ``loss``, ``eps_c``, ``base_demand``.
    """
    y_ref = section.y_centroid
    A = np.asarray(section.A_fibers, dtype=float)
    yb = np.asarray(section.y_fibers, dtype=float) - y_ref
    yv = np.asarray(section.y_tendons, dtype=float)
    Ap = np.asarray(section.A_tendons, dtype=float)
    d = yv - y_ref
    nt = yv.size

    pre = sorted({int(p) for p in preexisting})
    sched = {int(k): int(v) for k, v in (bonded_before or {}).items()}
    sig = np.array(sigma_p0, dtype=float)
    for k, v in (sigma_pe or {}).items():
        sig[int(k)] = float(v)

    e0_ex, chix_ex = eps_plane
    # The declared plane, re-expressed about the reference used here.
    eps_bulk = e0_ex + chix_ex * ((yb + y_ref) - Y_REF_DECL) if pre \
        else np.zeros_like(yb)
    eps_bulk_t = e0_ex + chix_ex * (yv - Y_REF_DECL) if pre \
        else np.zeros(nt)

    eps_init = np.zeros(nt)
    for k in pre:
        eps_init[k] = sig[k] / EP - eps_bulk_t[k]

    def _resultant(e0, chi, bonded):
        e = e0 + chi * yb + eps_bulk
        s = law.stress_array(e)
        N = float(np.sum(s * A))
        M = float(np.sum(s * A * yb))
        t = law.tangent_array(e)
        kaa = float(np.sum(t * A))
        kab = float(np.sum(t * A * yb))
        kbb = float(np.sum(t * A * yb * yb))
        for i in bonded:
            e_sec = e0 + chi * d[i] + eps_bulk_t[i]
            s_net = (EP * (e_sec + eps_init[i])
                     - float(law.stress(e_sec))) * Ap[i]
            N += s_net
            M += s_net * d[i]
            t_net = (EP - float(law.tangent(e_sec))) * Ap[i]
            kaa += t_net
            kab += t_net * d[i]
            kbb += t_net * d[i] * d[i]
        return (np.array([N, M]),
                np.array([[kaa, kab], [kab, kbb]]))

    def _solve(N, M, bonded, guess):
        r"""
        Damped Newton on :math:`(\varepsilon_0, \chi_x)`.

        The step is halved until the residual norm decreases, because
        an undamped step overshoots into the clipped branch of the
        service law, where the tangent is constant and the iteration
        stops being contractive.
        """
        e0, chi = float(guess[0]), float(guess[1])
        scale = np.array([max(abs(N), 1.0),
                          max(abs(M), 1.0)])

        def _res(a, b):
            r, K = _resultant(a, b, bonded)
            return (r - np.array([N, M])), K

        res, K = _res(e0, chi)
        for _ in range(max_iter):
            if np.max(np.abs(res / scale)) < tol:
                return e0, chi
            step = np.linalg.solve(K, -res)
            lam = 1.0
            for _ in range(60):
                cand = _res(e0 + lam * float(step[0]),
                            chi + lam * float(step[1]))
                if np.linalg.norm(cand[0] / scale) \
                        < np.linalg.norm(res / scale):
                    break
                lam *= 0.5
            else:                               # pragma: no cover
                raise AssertionError(
                    "reference Newton: no descent direction")
            e0 += lam * float(step[0])
            chi += lam * float(step[1])
            res, K = cand
        raise AssertionError(                   # pragma: no cover
            "reference Newton did not converge")

    # The base point of EVERY step is the accumulated state, not the
    # origin.  Measured on a linear staged case: differencing two
    # solves of the CUMULATIVE action set against a section whose
    # bonded set changed between them is wrong by 0.11 % against the
    # closed form, and the closed form is right there.  A changing
    # section resists increments; only a constant one may be solved
    # cumulatively.  The form below reduces to the cumulative
    # difference when the bonded set never changes, and to the
    # pre-Phase-25 driver when the law is linear.
    base, _ = _resultant(0.0, 0.0, pre)
    debit = np.zeros(nt)
    anchored = np.zeros(nt, dtype=bool)
    for k in pre:
        anchored[k] = True
    eps_prev = eps_bulk_t.copy()
    e0_acc, chi_acc = 0.0, 0.0

    for step, j in enumerate(order):
        bonded = sorted(set(pre) | {t for t, s in sched.items()
                                    if s <= step})
        # A tendon that joins here enters carrying nothing new: its
        # jacking force is already accounted, and it must resist only
        # what happens after it joins.
        for t, s in sched.items():
            if s == step:
                eps_init[t] = -(e0_acc + chi_acc * d[t] + eps_bulk_t[t])
        # The resultant the accumulated state already carries, on the
        # section that resists from here on.
        held, _ = _resultant(e0_acc, chi_acc, bonded)
        P = float(sig[j] * Ap[j])
        target = held + np.array([-P, -P * d[j]])
        e0_acc, chi_acc = _solve(target[0], target[1], bonded,
                                 np.array([e0_acc, chi_acc]))
        eps_tot = e0_acc + chi_acc * d + eps_bulk_t
        d_eps = eps_tot - eps_prev
        eps_prev = eps_tot.copy()
        mask = anchored.copy()
        mask[j] = False
        debit[mask] += -EP * d_eps[mask]
        anchored[j] = True

    return {"sigma_p_after": sig - debit, "loss": debit,
            "eps_c": eps_tot, "base_demand": base}


def _run(law, **kw):
    r"""The driver, with this phase's geometry."""
    return pt.solve_posttension_sequence(_section(law), Ep=EP, **kw)


def _rel(a, b):
    r"""Max relative difference between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = np.where(np.abs(a) > 1e-30, np.abs(a), 1.0)
    return float(np.max(np.abs(b - a) / den))


# ===========================================================================
#  A — the reference is checked before it is used
# ===========================================================================

def test_reference_agrees_with_closed_form_on_a_linear_law():
    r"""
    Green on both trees, and the load-bearing fixture of this file.

    The fibre reference and the transformed-section reference are two
    different computations of one linear problem.  If they disagree the
    fibre reference is wrong and every nonlinear number below is
    worthless.
    """
    law = _LinearElastic(ECM)
    sec = _section(law)
    ref = _cumulative_reference(sec, law, [0, 1, 2], [SP0] * 3)
    cf = _closed_form_linear(sec, ECM, [0, 1, 2], [SP0] * 3)
    assert _rel(cf["sigma_p_after"], ref["sigma_p_after"]) < 1e-9
    assert _rel(cf["eps_c"], ref["eps_c"]) < 1e-9


def test_reference_agrees_with_closed_form_on_a_staged_linear_law():
    r"""Green on both trees: the ``bonded_before`` bookkeeping too."""
    law = _LinearElastic(ECM)
    sec = _section(law)
    sched = {0: 1, 1: 2}
    ref = _cumulative_reference(sec, law, [0, 1, 2], [SP0] * 3,
                                bonded_before=sched)
    cf = _closed_form_linear(sec, ECM, [0, 1, 2], [SP0] * 3,
                             bonded_before=sched)
    assert _rel(cf["sigma_p_after"], ref["sigma_p_after"]) < 1e-9


def test_reference_agrees_with_closed_form_on_a_linear_retrofit():
    r"""Green on both trees: the declared pre-existing tendon too."""
    law = _LinearElastic(ECM)
    sec = _section(law)
    ref = _cumulative_reference(
        sec, law, [1, 2], [NAN, SP0, SP0], preexisting=[0],
        sigma_pe={0: SIGMA_PE}, eps_plane=(EPS0_EX, CHIX_EX))
    cf = _closed_form_linear(
        sec, ECM, [1, 2], [NAN, SP0, SP0], preexisting=[0],
        sigma_pe={0: SIGMA_PE}, eps_plane=(EPS0_EX, CHIX_EX))
    assert _rel(cf["sigma_p_after"][1:], ref["sigma_p_after"][1:]) < 1e-9
    assert _rel(cf["loss"][:1], ref["loss"][:1]) < 1e-8


# ===========================================================================
#  B — the linear control: these must not move by a bit
# ===========================================================================

def test_linear_virgin_matches_the_closed_form():
    r"""Green on both trees.  The cumulative form must reduce to this."""
    law = _LinearElastic(ECM)
    got = _run(law, sigma_p0=[SP0] * 3, order=[0, 1, 2])
    cf = _closed_form_linear(_section(law), ECM, [0, 1, 2], [SP0] * 3)
    assert _rel(cf["sigma_p_after"], got.sigma_p_after) < 1e-9
    assert _rel(cf["eps_c"], got.eps_ref_grout) < 1e-9


def test_linear_staged_matches_the_closed_form():
    r"""Green on both trees: intra-sequence injection, Phase 22."""
    law = _LinearElastic(ECM)
    got = _run(law, sigma_p0=[SP0] * 3, order=[0, 1, 2],
               bonded_before={0: 1, 1: 2})
    cf = _closed_form_linear(_section(law), ECM, [0, 1, 2], [SP0] * 3,
                             bonded_before={0: 1, 1: 2})
    assert _rel(cf["sigma_p_after"], got.sigma_p_after) < 1e-9


def test_linear_retrofit_matches_the_closed_form():
    r"""Green on both trees: the declared retrofit, Phase 24."""
    law = _LinearElastic(ECM)
    got = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
               already_bonded=_declare({0: SIGMA_PE}))
    cf = _closed_form_linear(
        _section(law), ECM, [1, 2], [NAN, SP0, SP0], preexisting=[0],
        sigma_pe={0: SIGMA_PE}, eps_plane=(EPS0_EX, CHIX_EX))
    assert _rel(cf["sigma_p_after"][1:], got.sigma_p_after[1:]) < 1e-9
    assert _rel(cf["loss"][:1], got.loss[:1]) < 1e-8


def test_linear_declared_plane_does_not_change_the_answer_of_the_new_cables():
    r"""
    Green on both trees, and the companion of the nonlinear fixture.

    On a constant modulus the strain the concrete is already in cannot
    change the stiffness it offers, so two declarations differing only
    in the plane must give the **same** loss on the newly jacked
    tendons.  This is what makes the nonlinear fixture below a
    measurement of the tangent and not of something else.
    """
    law = _LinearElastic(ECM)
    a = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
             already_bonded=_declare({0: SIGMA_PE}, eps0=-1.0e-4))
    b = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
             already_bonded=_declare({0: SIGMA_PE}, eps0=-9.0e-4))
    assert _rel(a.loss[1:], b.loss[1:]) < 1e-12


# ===========================================================================
#  C — the increment form is declared
# ===========================================================================

def test_result_declares_its_increment_form():
    r"""
    The result says which increment form produced it.

    D1c is not that the driver is wrong on a linear section — there it
    is exact.  It is that it does not say it is assuming one.
    """
    got = _run(_LinearElastic(ECM), sigma_p0=[SP0] * 3, order=[0, 1, 2])
    form = getattr(got, "increment_form", None)
    assert form is not None, (
        "PosttensionSequenceResult publishes no `increment_form`: the "
        "driver assumes superposition without declaring the premise.")


def test_declared_increment_form_is_cumulative():
    r"""And what it says is the exact form, not the assuming one."""
    got = _run(_LinearElastic(ECM), sigma_p0=[SP0] * 3, order=[0, 1, 2])
    form = getattr(got, "increment_form", None)
    assert form == "incremental_on_state", (
        f"increment_form={form!r}; the exact form for any constitutive "
        f"law applies each increment to the state already reached.")


# ===========================================================================
#  D — the nonlinear case: the driver must agree with the reference
# ===========================================================================

def test_nonlinear_virgin_matches_the_cumulative_reference():
    r"""
    Red on the Phase-24 tree: 5.9 % on the final stress.

    The incremental form sums the responses to three separate actions,
    each evaluated on a tangent taken at zero strain.  The section is
    softer than that by the time the third jack is pulled.
    """
    law = _EC2ServiceLaw(FCK)
    got = _run(law, sigma_p0=[SP0] * 3, order=[0, 1, 2])
    ref = _cumulative_reference(_section(law), law, [0, 1, 2], [SP0] * 3)
    assert _rel(ref["sigma_p_after"], got.sigma_p_after) < 1e-6
    assert _rel(ref["eps_c"], got.eps_ref_grout) < 1e-6


def test_nonlinear_virgin_loss_matches_the_cumulative_reference():
    r"""The loss is the quantity the driver exists to produce."""
    law = _EC2ServiceLaw(FCK)
    got = _run(law, sigma_p0=[SP0] * 3, order=[0, 1, 2])
    ref = _cumulative_reference(_section(law), law, [0, 1, 2], [SP0] * 3)
    assert _rel(ref["loss"][:2], got.loss[:2]) < 1e-6


def test_nonlinear_staged_matches_the_cumulative_reference():
    r"""
    Red on the Phase-24 tree.

    A tendon injected mid-sequence must resist only what happens after
    it joins.  In the cumulative form that is an initial strain of
    :math:`-\varepsilon_c^{(s-1)}`, not zero.
    """
    law = _EC2ServiceLaw(FCK)
    sched = {0: 1, 1: 2}
    got = _run(law, sigma_p0=[SP0] * 3, order=[0, 1, 2],
               bonded_before=sched)
    ref = _cumulative_reference(_section(law), law, [0, 1, 2], [SP0] * 3,
                                bonded_before=sched)
    assert _rel(ref["sigma_p_after"], got.sigma_p_after) < 1e-6


def test_nonlinear_retrofit_matches_the_cumulative_reference():
    r"""
    Red on the Phase-24 tree, and the case the phase is about.

    The retrofit is precisely where the concrete is already deeply
    compressed, so it is where the tangent is furthest from
    :math:`E_{cm}`.
    """
    law = _EC2ServiceLaw(FCK)
    got = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
               already_bonded=_declare({0: SIGMA_PE}))
    ref = _cumulative_reference(
        _section(law), law, [1, 2], [NAN, SP0, SP0], preexisting=[0],
        sigma_pe={0: SIGMA_PE}, eps_plane=(EPS0_EX, CHIX_EX))
    assert _rel(ref["sigma_p_after"][1:], got.sigma_p_after[1:]) < 1e-6


def test_nonlinear_declared_plane_shifts_the_tangent():
    r"""
    Red on the Phase-24 tree, and behavioural rather than numerical.

    Today the declared plane only seeds the accumulator
    ``eps_ref_grout`` and never enters the resisting view, so two
    declarations differing only in how compressed the existing concrete
    is give **identical** losses on the new tendons.  They must not:
    concrete at :math:`-9\cdot10^{-4}` is 31 % softer in tangent than
    concrete at :math:`-1\cdot10^{-4}`, and the new jack is pulled
    against that softer section.

    The companion linear fixture above asserts the opposite on a
    constant modulus, which is what makes this one a measurement of the
    tangent and not of a bookkeeping accident.
    """
    law = _EC2ServiceLaw(FCK)
    a = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
             already_bonded=_declare({0: SIGMA_PE}, eps0=-1.0e-4))
    b = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
             already_bonded=_declare({0: SIGMA_PE}, eps0=-9.0e-4))
    moved = _rel(a.loss[1:], b.loss[1:])
    assert moved > 1e-3, (
        f"the declared plane moves the new tendons' loss by {moved:.3e}: "
        f"it is not in the resisting view, so it cannot shift the "
        f"tangent the next jack is pulled against.")


# ===========================================================================
#  E — the sollecitazione the declared state implies
# ===========================================================================

def test_declared_state_demand_is_published():
    r"""
    Red on the Phase-24 tree.

    With the declared plane and the pre-existing tendon's real initial
    strain both in the view, the resultant of the declared state is a
    quantity GenSec can integrate.  It is the base point of the
    cumulative ray — the first solve is not against zero — and once
    computed it is also the coherence check against the original design
    that does not exist today.  GenSec cannot refuse on it: it does not
    know what is right.  It publishes it.
    """
    got = _run(_EC2ServiceLaw(FCK), sigma_p0=[NAN, SP0, SP0], order=[1, 2],
               pending=[], already_bonded=_declare({0: SIGMA_PE}))
    dem = getattr(got, "declared_state_demand", None)
    assert dem is not None, (
        "the result publishes no `declared_state_demand`: the "
        "sollecitazione the declared state implies is not available, "
        "so the declaration cannot be checked against anything.")
    assert len(tuple(dem)) == 3


def test_declared_state_demand_is_none_without_a_declaration():
    r"""No declaration, no measurand — and not a zero triple."""
    got = _run(_EC2ServiceLaw(FCK), sigma_p0=[SP0] * 3, order=[0, 1, 2])
    assert hasattr(got, "declared_state_demand"), (
        "the attribute does not exist, so its absence cannot be "
        "distinguished from a declared zero.")
    assert got.declared_state_demand is None


def test_declared_state_demand_matches_the_reference():
    r"""And the number is the one an independent integration gives."""
    law = _EC2ServiceLaw(FCK)
    got = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
               already_bonded=_declare({0: SIGMA_PE}))
    ref = _cumulative_reference(
        _section(law), law, [1, 2], [NAN, SP0, SP0], preexisting=[0],
        sigma_pe={0: SIGMA_PE}, eps_plane=(EPS0_EX, CHIX_EX))
    dem = getattr(got, "declared_state_demand", None)
    assert dem is not None, "no `declared_state_demand` to compare."
    assert _rel(ref["base_demand"], np.asarray(dem, float)[:2]) < 1e-6


def test_declared_plane_remains_the_accumulation_baseline():
    r"""
    **Green on both trees, and it does not discriminate.**  Said here
    rather than left to be discovered.

    It was written expecting a red, on the argument that solving the
    first jack against zero would move the section off the declared
    plane.  Measured: it passes on the Phase-24 tree, because there the
    declared plane never enters a solve at all — it only seeds the
    accumulator, so the identity it asserts holds by construction.  A
    fixture that holds by construction on both trees proves nothing
    about the repair.

    It is kept, and re-labelled, as an **immobility** check: the
    cumulative form must not change what ``eps_ref_grout`` means.  The
    discriminating fixture for the base point is
    :func:`test_declared_state_demand_matches_the_reference`.
    """
    law = _EC2ServiceLaw(FCK)
    got = _run(law, sigma_p0=[NAN, SP0, SP0], order=[1, 2], pending=[],
               already_bonded=_declare({0: SIGMA_PE}))
    stages = list(got.stages)
    assert stages, "no stages recorded."
    declared_at_0 = EPS0_EX + CHIX_EX * (Y_TENDONS[0] - Y_REF_DECL)
    d_eps = np.asarray(stages[0].report["d_eps_c_at_tendons"], float)
    reached = float(got.stages[0].eps_c[0])
    assert abs(reached - (declared_at_0 + float(d_eps[0]))) < 1e-12, (
        "the accumulated strain after the first jack is not the "
        "declared plane plus the increment of that jack.")


# ===========================================================================
#  F — the refusals of earlier phases keep their names
# ===========================================================================

def test_bare_already_bonded_is_still_refused_with_its_reason():
    r"""Green on both trees: a set of indices is not a declaration."""
    with pytest.raises(ValueError) as exc:
        _run(_EC2ServiceLaw(FCK), sigma_p0=[SP0] * 3, order=[1, 2], pending=[],
             already_bonded=[0])
    assert getattr(exc.value, "reason", None) is not None


def test_undeclared_tendon_is_still_refused():
    r"""Green on both trees: every tendon declares its role."""
    with pytest.raises(ValueError):
        _run(_EC2ServiceLaw(FCK), sigma_p0=[SP0] * 3, order=[0, 1])


def test_service_law_is_actually_nonlinear():
    r"""
    Green on both trees, and the ruler of this file.

    If the service law's tangent stopped varying, every red above would
    turn green for a reason that has nothing to do with the driver.
    The figures are EN 1992-1-1 §3.1.5 softening against
    :math:`E_{cm}`, from ``fben2(35)``.
    """
    law = _EC2ServiceLaw(FCK)
    assert abs(law.Ecm - ECM) < 1e-9
    assert law.tangent(-1.0e-4) / ECM > 1.0
    assert law.tangent(-3.0e-4) / ECM < 0.94
    assert law.tangent(-6.0e-4) / ECM < 0.82
    assert law.tangent(-9.0e-4) / ECM < 0.69


def test_analytic_tangent_matches_a_difference_of_en2sargin():
    r"""
    Green on both trees.

    The Eurocode states the stress and not its derivative, so the
    tangent is differentiated in the adapter.  This fixture ties it
    back to the function it differentiates: a drift between the two
    would otherwise be invisible, and every nonlinear number in this
    file is solved with that tangent.
    """
    law = _EC2ServiceLaw(FCK)
    h = 1e-8
    for eps in (-1.0e-4, -3.0e-4, -6.0e-4, -9.0e-4, -1.2e-3, 5.0e-5):
        fd = (law.stress(eps + h) - law.stress(eps - h)) / (2.0 * h)
        assert abs(law.tangent(eps) - fd) / abs(fd) < 1e-8
