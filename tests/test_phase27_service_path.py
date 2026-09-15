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
Phase 27 — the service law reaches the solver, and the straight line
stops being substituted for it.

Physics first, the clause afterwards
------------------------------------
Every assertion in this file is a statement about what a section *does*,
not about which function it calls.  That is deliberate and it is the
closure rule of the phase (``27_P`` §7.3): a fixture that goes red with
``ImportError`` or ``AttributeError`` records that a name moved, which
is not evidence that anything was repaired.  A fixture that goes red on
a **measured value** records that the tree answered a physical question
wrongly, and that is the only red worth anything.

The two families in this file
-----------------------------
**Repair fixtures**, marked ``RED-BEFORE`` in their docstring: measured
red on the tree rebuilt from ``gensec_phase26.bundle``, green after.
Each one carries the value the pristine tree produced, so the red is
reproducible rather than remembered.

**Guards**, marked ``GUARD``: green on both trees by construction and
declared as such, never counted as evidence of repair.  They exist
because Phase 26 measured that the cut law and the uncut one agree to
machine zero below cracking (``26_0`` §3.3), and a phase that changes
the law above cracking must show it changed nothing below it.  ``27_P``
§7.4 forbids relabelling a two-tree-green fixture as a repair fixture;
it does not forbid a guard that says what it is.
"""

import math
import unittest

import numpy as np

from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.section import RectSection
from gensec.materials.concrete import Concrete
from gensec.materials.ec2_properties import fben2
from gensec.materials.elastic import LinearElastic
from gensec.materials.rheology import (EC2RheologicalModel,
                                       TabulatedRheologicalModel)
from gensec.solver.section_state import materialize_view
from gensec.solver.integrator import FiberSolver
from gensec.solver.sls import DatumPolicy, resolve_moduli, sls_view
from gensec.materials.steel import Steel, PrestressingSteel
from gensec.materials.verification_limits import StressLimits
from gensec.solver.section_state import SectionState
from gensec.solver.sls import verify_sls_staged

#: The 400 x 1200 rectangle every published measurement of this defect
#: family is written on (``22_0`` §1.1, ``26_0`` §3.3).
B, H, N_FIB_Y = 400.0, 1200.0, 240
AS_LAYER, Y_BOT, Y_TOP = 2000.0, 60.0, 1140.0
FCK, CEM = 35.0, "N"

#: Values read from the tree rebuilt from ``gensec_phase26.bundle``.
#: They are written down so that the red is a reproducible measurement
#: and not a memory of one.
PRISTINE_SIG_C_MAX_M900 = 6.531641          # MPa, tension
PRISTINE_CHI_X_M900 = 4.173735040e-07       # 1/mm
PRISTINE_SIG_C_MIN_M900 = -10.535836        # MPa, compression
PRISTINE_SIG_C_MAX_M300 = 0.842482          # MPa, tension
PRISTINE_CHI_X_M300 = 1.39124501346e-07     # 1/mm


def _ec2(t):
    """EN 1992-1-1 Table 3.1 at age *t* [d]."""
    return fben2(fck=FCK, ls="S", loadtype="slow", TypeConc=CEM, time=t)


def _section():
    """The bare reinforced rectangle, concrete declared as a design law."""
    conc = Concrete(fck=FCK, gamma_c=1.5, long_term_reduction_factor=0.85)
    conc.name = "C35svc"
    steel = Steel(fyk=450.0)
    steel.name = "B450C"
    rebars = [RebarLayer(y=Y_BOT, As=AS_LAYER, material=steel, x=B / 2.0),
              RebarLayer(y=Y_TOP, As=AS_LAYER, material=steel, x=B / 2.0)]
    return RectSection(B, H, conc, rebars, n_fibers_y=N_FIB_Y)


def _state(idx=0, n=2, label="svc"):
    """A state with every element active and bonded."""
    return SectionState(idx,
                        np.ones(n, dtype=bool),
                        np.ones(n, dtype=bool),
                        np.zeros(n, dtype=float),
                        bulk_eps_init=0.0, label=label)


def _datum_solve(moment, *, t=28.0):
    r"""
    Solve one demand through the **datum** path.

    This is the route where nothing declares a straight line.  The
    model declares a concrete and a rheological provider; the age comes
    from the timeline; :class:`~gensec.solver.sls.DatumPolicy` resolves
    the zone, and whatever it resolves is what the fibres answer with.
    A caller who writes ``moduli={"C35": 34077.0}`` has *declared* a
    line and correctly gets one -- that is not the defect and never
    was.  The defect is here, where the line was never written down by
    anybody.

    Returns
    -------
    dict
        ``{"map": MaterialMap, "view": ..., "eps0": float,
        "chi_x": float, "sig_bot": float, "sig_top": float}``.
    """
    section = _section()
    state = SectionState(0, np.ones(2, dtype=bool), np.ones(2, dtype=bool),
                         np.zeros(2, dtype=float), bulk_eps_init=0.0,
                         bulk_active=np.ones(1, dtype=bool), label="datum")
    policy = DatumPolicy(state=state,
                         providers={0: EC2RheologicalModel(fck=FCK,
                                                           cement_class=CEM)},
                         ages={0: float(t)}, casting_zone=None,
                         placeholder=1.0, reference=None)
    modmap = resolve_moduli(section, policy)
    view = sls_view(materialize_view(section, state), modmap)
    sol = FiberSolver(view).solve_equilibrium(-1000.0e3, float(moment), 0.0)
    eps0, chi = float(sol["eps0"]), float(sol["chi_x"])
    edge = np.array([0.0, H], dtype=float)
    sig = view.bulk_material.stress_array(eps0 - chi * (edge - H / 2.0))
    return {"map": modmap, "view": view, "eps0": eps0, "chi_x": chi,
            "sig_bot": float(sig[0]), "sig_top": float(sig[1])}


def _run(moment, *, limits=None, t=28.0):
    """One SLS stage at ``N = -1000 kN`` and the given ``Mx``."""
    p = _ec2(t)
    return verify_sls_staged(
        _section(),
        [{"name": "svc", "state": _state(),
          "increment": (-1000.0e3, float(moment), 0.0),
          "limits": limits}],
        moduli={"C35svc": float(p.ecm)},
        x_ref=B / 2.0, y_ref=H / 2.0)


class TestConcreteCarriesNoImpossibleTension(unittest.TestCase):
    r"""
    RED-BEFORE.  The verification must not report a stress the material
    cannot reach.
    """

    def test_max_concrete_tension_is_capped(self):
        r"""
        RED-BEFORE — pristine value +6.531641 MPa against
        :math:`f_{ctm} = 3.209962` MPa, measured on the datum path.

        At :math:`N = -1000` kN and :math:`M_x = 900` kNm the section is
        well past its cracking moment.  A C35/45 does not carry 6.53 MPa
        of tension; no clause makes it, and EN 1992-1-1 §7.1(2)
        prescribing an uncracked analysis with an *a posteriori* check
        does not license the intermediate state as a physical one
        (``26_0`` §0).

        The demand goes through the **datum** path on purpose.  There
        the model declares a concrete and a provider and never declares
        a line, so the straight line that produced 6.53 MPa was one
        nobody wrote down -- which is the difference between a
        substitution and a declaration, and the only version of this
        assertion that can go green.
        """
        fct_eff = float(_ec2(28.0).fctm)
        got = _datum_solve(900.0e6)

        self.assertGreater(
            PRISTINE_SIG_C_MAX_M900, fct_eff,
            "the recorded pristine value must exceed f_ct,eff, or this "
            "fixture cannot exhibit the defect it is written for")
        self.assertLessEqual(
            got["sig_bot"], fct_eff * (1.0 + 1.0e-9),
            f"concrete reported at {got['sig_bot']:.6f} MPa of tension "
            f"against f_ct,eff = {fct_eff:.6f} MPa")
        self.assertLess(
            got["sig_top"], PRISTINE_SIG_C_MIN_M900,
            "the compressed edge did not take up the load the cracked "
            "tension zone stopped carrying")

    def test_cracked_section_is_softer(self):
        r"""
        RED-BEFORE — pristine curvature 4.173735040e-07 1/mm on the
        datum path.

        Capping the tension branch is not a cosmetic clamp on a reported
        stress: the concrete above the crack stops carrying, the section
        redistributes onto the steel, and the curvature grows.  Phase 26
        measured +279.84 % on its own geometry (``26_0`` §3.3).  The
        threshold here is deliberately loose -- a factor of 1.5 --
        because the fixture asserts that the section cracked, not by how
        much.
        """
        got = _datum_solve(900.0e6)

        self.assertGreater(
            got["chi_x"], 1.5 * PRISTINE_CHI_X_M900,
            f"curvature {got['chi_x']:.6e} is not that of a cracked "
            f"section; the substituted answer was "
            f"{PRISTINE_CHI_X_M900:.6e}")

    def test_the_datum_zone_carries_a_law_not_a_number(self):
        r"""
        RED-BEFORE — the pristine map held ``float64`` 34077.146 and the
        view's bulk material was ``LinearElastic``.

        Stated as physics rather than as a type: the material the datum
        zone answers with must be **non-linear**, i.e. its tangent at a
        service compressive strain must differ from its tangent at the
        origin.  A straight line cannot fail this test by accident, and
        a law cannot pass it by accident either.
        """
        got = _datum_solve(900.0e6)
        mat = got["view"].bulk_material

        e_service = -1.2e-3
        t0 = float(mat.tangent_array(np.zeros(1))[0])
        ts = float(mat.tangent_array(np.array([e_service]))[0])
        self.assertGreater(
            abs(ts - t0) / t0, 0.10,
            f"the datum zone's tangent moved by "
            f"{abs(ts - t0) / t0:.4e} between zero and "
            f"{e_service:g} strain: it is a straight line")

class TestNoFlagWithoutALever(unittest.TestCase):
    r"""
    RED-BEFORE.  ``uncracked_basis_violated`` must be gone, not
    redefined (``27_P`` §5.4).
    """

    def test_uncracked_basis_flag_is_absent(self):
        r"""
        RED-BEFORE — the key is present with value ``True`` on the
        pristine tree.

        The flag told a user that the answer had been computed on a
        basis the answer itself contradicted, and left the values in
        place as informative (Phase 7, D4).  With the cut law the
        contradiction cannot arise: the solution *is* the cracked one by
        the time the flag would fire.  A warning with no correction
        lever left is worse than no warning, because it trains a reader
        to ignore it.
        """
        # Phase 30: fct_eff is no longer a stress limit.  The tensile
        # strength that matters here is the one on the concrete's own
        # service law, which is what the view already evaluates.
        limits = StressLimits(k_c=0.6,
                              sigma_s_tens=0.8 * 450.0)
        res = _run(900.0e6, limits=limits)
        record = res["stages"][0]

        self.assertNotIn(
            "uncracked_basis_violated", record,
            "the flag is still reported; §5.4 asks for removal, not "
            "redefinition")
        self.assertNotIn("basis_checked", record)


class TestTheJackPullsAgainstATangent(unittest.TestCase):
    r"""
    RED-BEFORE.  The three consumers that are **not** stress
    verifications ask a different question and need a different modulus.
    """

    def test_shortening_differs_from_the_secant_answer(self):
        r"""
        RED-BEFORE — measured 0.000e+00 relative on the pristine tree.

        How much a section shortens under a jack depends on the
        **tangent**; :math:`E_{cm}` is the secant at :math:`0.4 f_{cm}`.
        Used as a constant tangent it errs by +1.19 % at
        :math:`\varepsilon = -10^{-4}` and -45.58 % at
        :math:`-1.2\cdot10^{-3}`, which is the transfer range itself
        (``27_P`` §4).  A driver that substitutes the secant answers the
        secant's question whatever material the section declares, so the
        difference against an explicitly declared straight line at the
        same :math:`E_{cm}` is exactly zero.

        **A modmap must be passed, and the first version of this fixture
        did not pass one.**  The substitution at
        ``posttension.py:1280`` and ``:1733`` is guarded by
        ``if modmap is not None`` -- it is *not* unconditional there, as
        it is at ``losses.py:983`` and ``timeline.py:1416``.  Without a
        map the driver used the declared **design** law, the two answers
        differed by 3.586 relative, and the fixture passed on the
        pristine tree while claiming to detect the substitution.  It
        could not exhibit the defect, so it was replaced rather than
        relabelled (``27_P`` §7.4).

        That discarded number is worth keeping: 3.586 is what removing
        the substitution would cost if nothing put a **service** law in
        its place, because the design parabola-rectangle's initial
        tangent is 0.58 :math:`E_{cm}` at C35/45.  It is the measured
        reason the material map of §5.2 has to land before the removal
        of §5.3.
        """
        from gensec.solver.posttension import solve_posttension_sequence

        ecm = float(_ec2(28.0).ecm)
        ps = PrestressingSteel(f_p01d=1391.3, Ep=195000.0)
        ps.name = "Y1860"

        def _built(bulk):
            tendons = [Tendon(y=y, x=B / 2.0, material=ps, Ap=1500.0,
                              eps_pe=0.0)
                       for y in (150.0, 200.0, 250.0)]
            return RectSection(B, H, bulk, [], n_fibers_y=N_FIB_Y,
                               tendons=tendons)

        declared = Concrete(fck=FCK, gamma_c=1.5,
                            long_term_reduction_factor=0.85)
        declared.name = "C35svc"
        straight = LinearElastic(E=ecm)
        straight.name = "C35svc"

        kw = dict(sigma_p0=[1300.0] * 3, Ep=195000.0, order=[0, 1, 2])
        sec_declared = _built(declared)
        n_t = 3
        st = SectionState(0, np.ones(n_t, dtype=bool),
                          np.ones(n_t, dtype=bool),
                          np.zeros(n_t, dtype=float), bulk_eps_init=0.0,
                          bulk_active=np.ones(1, dtype=bool), label="jack")
        policy = DatumPolicy(
            state=st,
            providers={0: EC2RheologicalModel(fck=FCK, cement_class=CEM)},
            ages={0: 28.0}, casting_zone=None, placeholder=1.0,
            reference=None)
        got = solve_posttension_sequence(
            sec_declared, modmap=resolve_moduli(sec_declared, policy), **kw)
        ref = solve_posttension_sequence(_built(straight), **kw)

        spread = max(abs(float(got.loss[k]) - float(ref.loss[k]))
                     for k in range(3))
        scale = max(abs(float(ref.loss[k])) for k in range(3))
        self.assertGreater(scale, 0.0, "the reference lost nothing; the "
                                       "fixture has no measurand")
        self.assertGreater(
            spread / scale, 1.0e-3,
            f"the declared concrete and an explicit straight line at "
            f"E_cm gave the same shortening to {spread / scale:.3e} "
            f"relative: the secant is still being substituted")


class TestRheologyIsNormative(unittest.TestCase):
    r"""
    GUARD.  A provider hands back the law **its own** code writes.
    """

    def test_ec2_provider_ages_three_constants(self):
        r"""
        GUARD — green on both trees only after ``service_law`` exists;
        recorded as a guard and not as evidence of the D26-7 repair.

        A non-linear law ages in more than a modulus.  Stiffness,
        strength and the strain at peak all move with :math:`t`, which
        is the whole argument for a map that carries a material instead
        of a scalar.
        """
        prov = EC2RheologicalModel(fck=FCK, cement_class=CEM)
        laws = {t: prov.service_law(t) for t in (3.0, 7.0, 28.0)}

        self.assertLess(laws[3.0].fcm, laws[7.0].fcm)
        self.assertLess(laws[7.0].fcm, laws[28.0].fcm)
        self.assertLess(laws[3.0].Ecm, laws[28.0].Ecm)
        self.assertLess(laws[3.0].eps_c1, laws[28.0].eps_c1)
        self.assertLess(laws[3.0].fct_eff, laws[28.0].fct_eff)

        for t, law in laws.items():
            self.assertAlmostEqual(
                law.initial_tangent / prov.E_c(t), 1.05, places=12,
                msg=f"the initial tangent drifted from 1.05 E_cm at {t} d")

    def test_ec2_provider_refuses_below_its_own_domain(self):
        r"""
        GUARD.  Below three days the refusal is a scope boundary, not a
        bad input.

        ``fben2`` applies EN 1992-1-1 §3.1.2(5) to two of its four
        constants: it zeroes :math:`f_{cm}` and
        :math:`\varepsilon_{c1}` while still reporting :math:`E_{cm}`
        and :math:`f_{ctm}`.  A material stiff at 24 590 MPa, resisting
        1.04 MPa of tension and **zero** compression does not exist,
        and nothing detected it for twenty-six phases because every
        consumer substituted a straight line, which has no strength to
        contradict.

        The refusal is a ``NotImplementedError`` and is raised from
        :class:`~gensec.materials.ec2_properties.EC2EarlyAgeConcrete` --
        inside the Eurocode module, because the three-day limit is the
        Eurocode's statement about the Eurocode's own expression.  A
        two-day age is a perfectly ordinary thing for a model to
        declare and is not a typo, so it is not a ``ValueError``: what
        is missing is a law, not a correct input.
        """
        prov = EC2RheologicalModel(fck=FCK, cement_class=CEM)
        with self.assertRaises(NotImplementedError):
            prov.service_law(1.0)
        with self.assertRaises(NotImplementedError):
            prov.service_law(2.9)
        self.assertIsNotNone(prov.service_law(3.0))

    def test_a_table_does_not_borrow_a_code(self):
        r"""
        GUARD.  A compliance table nominates no strength, so it cannot
        produce a constitutive law and must refuse rather than reach for
        EN 1992-1-1 Table 3.1 because it happens to be importable.

        ``NotImplementedError`` and not ``ValueError``: the input is
        complete and well formed, and what is missing is a model.
        """
        grid = [1.0, 10.0, 100.0]
        table = [[1.0 / 30000.0] * 3 for _ in grid]
        prov = TabulatedRheologicalModel(t_prime_grid=grid, t_grid=grid,
                                         J_table=table)
        with self.assertRaises(NotImplementedError):
            prov.service_law(28.0)


class TestBelowCrackingNothingMoves(unittest.TestCase):
    r"""
    GUARD.  Declared green on both trees, and green on purpose.
    """

    def test_uncracked_answer_is_unchanged(self):
        r"""
        GUARD — green before and after, by construction.

        Phase 26 measured the cut law and the uncut one identical to
        :math:`0.000\mathrm{e}{+}00` % below the cracking moment
        (``26_0`` §3.3): the cut costs nothing where it does not bite.
        A phase that changes the answer above cracking has to show it
        changed nothing below, and the only way to show that is to pin
        the pristine numbers here.

        This fixture is **not** evidence that anything was repaired and
        must never be counted as such.
        """
        res = _run(300.0e6)
        sig_max = float(res["stages"][0]["concrete"]["sigma_max_MPa"])
        chi = float(res["stages"][0]["plane"]["chi_x"])
        fct_eff = float(_ec2(28.0).fctm)

        self.assertLess(sig_max, fct_eff,
                        "the guard is mis-sited: this demand cracks the "
                        "section, so it cannot witness the uncracked case")
        self.assertTrue(
            math.isclose(sig_max, PRISTINE_SIG_C_MAX_M300, rel_tol=1e-6),
            f"{sig_max:.6f} against pristine {PRISTINE_SIG_C_MAX_M300:.6f}")
        self.assertTrue(
            math.isclose(chi, PRISTINE_CHI_X_M300, rel_tol=1e-6),
            f"{chi:.6e} against pristine {PRISTINE_CHI_X_M300:.6e}")


if __name__ == "__main__":
    unittest.main()
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=de5b1ca81b8bfb1bafe5352a80f7394dd21e9fcf22083c593129e581c05ddd68 body_lines=475
