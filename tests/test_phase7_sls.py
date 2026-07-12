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
Phase-7 test suite: SLS stress verification on the evolving section.

Mirrors ``run_phase7_sls_validation.py`` in ``unittest`` form: every
engine result is compared against an independent analytic 2-DOF model

.. math::

    \mathbf{K}\,\mathbf{u} = \mathbf{F}_{\mathrm{ext}} - \mathbf{F}_0

assembled from the section's own fiber arrays (solver quadrature —
tight tolerances) and, where noted, from exact continuum rectangle
formulas (mesh tolerance).  Regression anchors are explicit
full-precision values captured from the validated run.

Run with either::

    python -m unittest test_phase7_sls.py
    pytest test_phase7_sls.py
"""

import unittest

import numpy as np

from gensec.materials.concrete import Concrete
from gensec.materials.elastic import LinearElastic
from gensec.materials.steel import PrestressingSteel, Steel
from gensec.materials.verification_limits import (
    StressLimits, ec2_stress_limits,
)
from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.section import RectSection
from gensec.solver.section_state import SectionState, materialize_view
from gensec.solver.sls import (
    resolve_sls_moduli, sls_transformed_properties, sls_view,
    verify_sls_staged,
)


# ==================================================================
#  Shared model data
# ==================================================================

B, H = 300.0, 500.0
XR, YR = 150.0, 250.0
EC, ES, EP = 32000.0, 200000.0, 195000.0
AS1 = AS2 = 628.0
Y_S1, Y_S2 = 50.0, 450.0
AP, Y_T = 1500.0, 100.0
EPS_PE = 0.006
N_FIB_Y = 200

# Regression anchors (full precision, captured from the validated
# closed-form run of run_phase7_sls_validation.py on this exact
# model — transfer stage M = 40 kNm, losses stage eps_pe -> 0.0055
# after +80 kNm service moment):
# F6 (Phase-5): re-anchored to the EXTREME-FIBRE stresses.  The previous
# values (-30.490098 / 10.261473 / -33.620496) were the outermost FIBRE
# CENTROID's, inset half a fibre from the section face -- a systematic,
# NON-CONSERVATIVE under-report of the very stress EN 1992-1-1 7.2 limits.
REG_TRANSFER_SIG_MIN = -30.592489      # [MPa]
REG_TRANSFER_SIG_MAX = 10.363864       # [MPa]
REG_TRANSFER_SIGMA_P = 1033.492574     # [MPa]
REG_TRANSFER_SIGMA_S0 = -165.605338    # [MPa]
REG_LOSSES_SIG_MIN = -33.743676        # [MPa]
REG_LOSSES_SIGMA_P = 926.924518        # [MPa]


class _SLSCaseBase(unittest.TestCase):
    """Shared fixtures and analytic helpers."""

    @classmethod
    def setUpClass(cls):
        cls.sec = cls._build_section()
        cls.moduli = {"C35": EC}

    # ---------------- fixtures ----------------

    @staticmethod
    def _build_section():
        conc = Concrete(fck=35.0)
        conc.name = "C35"
        st = Steel(fyk=450.0)
        st.name = "B450C"
        ps = PrestressingSteel(f_p01d=1391.3, Ep=EP)
        ps.name = "Y1860"
        reb = [RebarLayer(y=Y_S1, As=AS1, material=st, x=XR),
               RebarLayer(y=Y_S2, As=AS2, material=st, x=XR)]
        ten = [Tendon(y=Y_T, Ap=AP, material=ps, x=XR,
                      eps_pe=EPS_PE, name="T1")]
        return RectSection(B, H, conc, reb, n_fibers_y=N_FIB_Y,
                           tendons=ten)

    @staticmethod
    def _state(active, bonded, eps_t, idx=0, beps=0.0):
        return SectionState(
            idx, np.asarray(active, dtype=bool),
            np.asarray(bonded, dtype=bool),
            np.array([0.0, 0.0, eps_t], dtype=float),
            bulk_eps_init=beps)

    # ---------------- analytic 2-DOF model ----------------

    @staticmethod
    def _assemble(sec, tendon_in, eps_pe, beps=0.0, discrete=True):
        r"""
        :math:`\mathbf{K}` (2x2) and internal constant load
        :math:`\mathbf{F}_0` about ``(XR, YR)``; differential element
        stiffness :math:`(E_e - E_c)A_e` (embedded net), tendon
        constant :math:`E_p A_p \varepsilon_{pe} [1, y_t - y_R]^T`,
        bulk offset on the net concrete area.
        """
        if discrete:
            Af = sec.A_fibers
            yf = sec.y_fibers - YR
            A_c = Af.sum()
            S_c = (Af * yf).sum()
            I_c = (Af * yf ** 2).sum()
        else:
            A_c, S_c, I_c = B * H, 0.0, B * H ** 3 / 12.0
        elems = [(ES, AS1, Y_S1 - YR), (ES, AS2, Y_S2 - YR)]
        if tendon_in:
            elems.append((EP, AP, Y_T - YR))
        EA = EC * A_c + sum((E - EC) * A for E, A, _ in elems)
        ESm = EC * S_c + sum((E - EC) * A * d for E, A, d in elems)
        EI = EC * I_c + sum((E - EC) * A * d ** 2
                            for E, A, d in elems)
        K = np.array([[EA, ESm], [ESm, EI]])
        F0 = np.zeros(2)
        if tendon_in:
            F0 += EP * AP * eps_pe * np.array([1.0, Y_T - YR])
        if beps:
            A_net = A_c - sum(A for _, A, _ in elems)
            S_net = S_c - sum(A * d for _, A, d in elems)
            F0 += EC * beps * np.array([A_net, S_net])
        return K, F0

    @classmethod
    def _solve_u(cls, sec, F_ext, tendon_in, eps_pe, beps=0.0,
                 discrete=True):
        K, F0 = cls._assemble(sec, tendon_in, eps_pe, beps, discrete)
        return np.linalg.solve(K, np.asarray(F_ext, float) - F0)

    @staticmethod
    def _sig_c(u, y, beps=0.0):
        return EC * (u[0] + u[1] * (y - YR) + beps)

    @staticmethod
    def _sig_el(u, y, E, eps0_el=0.0):
        return E * (u[0] + u[1] * (y - YR) + eps0_el)

    @staticmethod
    def _elem_sigma(res, k, union_index):
        for e in res["stages"][k]["elements"]:
            if e["union_index"] == union_index:
                return e["sigma_MPa"]
        raise KeyError(union_index)

    def _run(self, stages, **kw):
        kw.setdefault("moduli", self.moduli)
        kw.setdefault("x_ref", XR)
        kw.setdefault("y_ref", YR)
        kw.setdefault("debug_check_affine", True)
        return verify_sls_staged(self.sec, stages, **kw)


# ==================================================================
#  Transfer + regression anchors
# ==================================================================

class TestTransfer(_SLSCaseBase):

    M0 = 40.0e6

    def _transfer_result(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        return self._run([{"name": "t", "state": s0,
                           "increment": (0.0, self.M0, 0.0)}])

    def test_transfer_against_discrete_analytic(self):
        res = self._transfer_result()
        u = self._solve_u(self.sec, (0.0, self.M0), True, EPS_PE)
        c = res["stages"][0]["concrete"]
        y_bot = 0.0
        y_top = H
        self.assertAlmostEqual(c["sigma_min_MPa"],
                               self._sig_c(u, y_bot), delta=1e-3)
        self.assertAlmostEqual(c["sigma_max_MPa"],
                               self._sig_c(u, y_top), delta=1e-3)
        self.assertAlmostEqual(self._elem_sigma(res, 0, 2),
                               self._sig_el(u, Y_T, EP, EPS_PE),
                               delta=1e-3)
        self.assertAlmostEqual(self._elem_sigma(res, 0, 0),
                               self._sig_el(u, Y_S1, ES), delta=1e-3)

    def test_transfer_against_continuum_analytic(self):
        res = self._transfer_result()
        u = self._solve_u(self.sec, (0.0, self.M0), True, EPS_PE,
                          discrete=False)
        c = res["stages"][0]["concrete"]
        ref = self._sig_c(u, 0.0)
        self.assertAlmostEqual(c["sigma_min_MPa"], ref,
                               delta=5e-3 * abs(ref))

    def test_transfer_regression_anchors(self):
        res = self._transfer_result()
        c = res["stages"][0]["concrete"]
        self.assertAlmostEqual(c["sigma_min_MPa"],
                               REG_TRANSFER_SIG_MIN, delta=1e-4)
        self.assertAlmostEqual(c["sigma_max_MPa"],
                               REG_TRANSFER_SIG_MAX, delta=1e-4)
        self.assertAlmostEqual(self._elem_sigma(res, 0, 2),
                               REG_TRANSFER_SIGMA_P, delta=1e-4)
        self.assertAlmostEqual(self._elem_sigma(res, 0, 0),
                               REG_TRANSFER_SIGMA_S0, delta=1e-4)

    def test_elastic_shortening_emerges(self):
        res = self._transfer_result()
        sp = self._elem_sigma(res, 0, 2)
        self.assertLess(sp, EP * EPS_PE)
        u = self._solve_u(self.sec, (0.0, self.M0), True, EPS_PE)
        loss = EP * EPS_PE - sp
        self.assertAlmostEqual(
            loss, -EP * (u[0] + u[1] * (Y_T - YR)), delta=1e-3)


# ==================================================================
#  Staged accumulation: increments, losses, telescoping
# ==================================================================

class TestStagedAccumulation(_SLSCaseBase):

    M0, DM = 40.0e6, 80.0e6
    EPS_PE2 = 0.0055

    def _three_stage_result(self):
        s = [self._state([1, 1, 1], [1, 1, 1], e, i)
             for i, e in enumerate((EPS_PE, EPS_PE, self.EPS_PE2))]
        return self._run(
            [{"name": "t", "state": s[0],
              "increment": (0, self.M0, 0)},
             {"name": "svc", "state": s[1],
              "increment": (0, self.DM, 0)},
             {"name": "loss", "state": s[2]}])

    def test_same_state_increment_telescopes_to_total(self):
        res = self._three_stage_result()
        u = self._solve_u(self.sec, (0.0, self.M0 + self.DM),
                          True, EPS_PE)
        c = res["stages"][1]["concrete"]
        self.assertAlmostEqual(
            c["sigma_min_MPa"],
            self._sig_c(u, 0.0),
            delta=1e-3)
        self.assertAlmostEqual(self._elem_sigma(res, 1, 2),
                               self._sig_el(u, Y_T, EP, EPS_PE),
                               delta=1e-3)

    def test_loss_stage_telescopes_to_total_on_final_state(self):
        res = self._three_stage_result()
        u = self._solve_u(self.sec, (0.0, self.M0 + self.DM),
                          True, self.EPS_PE2)
        c = res["stages"][2]["concrete"]
        self.assertAlmostEqual(
            c["sigma_min_MPa"],
            self._sig_c(u, 0.0),
            delta=1e-3)
        self.assertAlmostEqual(self._elem_sigma(res, 2, 2),
                               self._sig_el(u, Y_T, EP,
                                            self.EPS_PE2),
                               delta=1e-3)

    def test_loss_redistribution_bounded_by_free_drop(self):
        res = self._three_stage_result()
        dsp = (self._elem_sigma(res, 2, 2)
               - self._elem_sigma(res, 1, 2))
        self.assertGreater(-dsp, 0.0)
        self.assertLess(-dsp, EP * (EPS_PE - self.EPS_PE2))

    def test_losses_regression_anchors(self):
        res = self._three_stage_result()
        self.assertAlmostEqual(
            res["stages"][2]["concrete"]["sigma_min_MPa"],
            REG_LOSSES_SIG_MIN, delta=1e-4)
        self.assertAlmostEqual(self._elem_sigma(res, 2, 2),
                               REG_LOSSES_SIGMA_P, delta=1e-4)

    def test_superposition_two_halves_equal_one(self):
        M = 120.0e6
        mk = lambda i: self._state([1, 1, 1], [1, 1, 1],  # noqa: E731
                                   EPS_PE, i)
        r1 = self._run([{"name": "a", "state": mk(0),
                         "increment": (0, M, 0)}])
        r2 = self._run([{"name": "a", "state": mk(0),
                         "increment": (0, M / 2, 0)},
                        {"name": "b", "state": mk(1),
                         "increment": (0, M / 2, 0)}])
        self.assertAlmostEqual(
            r1["stages"][0]["concrete"]["sigma_min_MPa"],
            r2["stages"][1]["concrete"]["sigma_min_MPa"], delta=1e-6)
        self.assertAlmostEqual(self._elem_sigma(r1, 0, 2),
                               self._elem_sigma(r2, 1, 2),
                               delta=1e-6)


# ==================================================================
#  Entering elements
# ==================================================================

class TestEnteringElements(_SLSCaseBase):

    M0, M1 = 40.0e6, 120.0e6

    def _entering_result(self):
        s0 = self._state([1, 1, 0], [1, 1, 1], EPS_PE, 0)
        s1 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 1)
        return self._run(
            [{"name": "a", "state": s0,
              "increment": (0, self.M0, 0)},
             {"name": "b", "state": s1,
              "increment": (0, self.M1 - self.M0, 0)}])

    def test_entering_tendon_initialises_from_total_read(self):
        res = self._entering_result()
        u1 = self._solve_u(self.sec, (0.0, self.M1), True, EPS_PE)
        self.assertAlmostEqual(self._elem_sigma(res, 1, 2),
                               self._sig_el(u1, Y_T, EP, EPS_PE),
                               delta=1e-3)

    def test_persisting_rebar_uses_staged_attribution(self):
        res = self._entering_result()
        u0 = self._solve_u(self.sec, (0.0, self.M0), False, 0.0)
        u_hi = self._solve_u(self.sec, (0.0, self.M1), True, EPS_PE)
        u_lo = self._solve_u(self.sec, (0.0, self.M0), True, EPS_PE)
        ref = (self._sig_el(u0, Y_S1, ES)
               + self._sig_el(u_hi, Y_S1, ES)
               - self._sig_el(u_lo, Y_S1, ES))
        self.assertAlmostEqual(self._elem_sigma(res, 1, 0), ref,
                               delta=1e-3)
        # And staged attribution is NOT the total on the final view.
        self.assertGreater(
            abs(ref - self._sig_el(u_hi, Y_S1, ES)), 1.0)


# ==================================================================
#  Decompression
# ==================================================================

class TestDecompression(_SLSCaseBase):

    def test_verdict_flips_across_analytic_boundary(self):
        c_dec = 25.0
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)

        def chi(M):
            return self._solve_u(self.sec, (0.0, M), True,
                                 EPS_PE)[1]

        def sig_at(y, M):
            u = self._solve_u(self.sec, (0.0, M), True, EPS_PE)
            return self._sig_c(u, y)

        M_a, M_b = 100.0e6, 300.0e6
        found = None
        for s in (+1.0, -1.0):
            y_s = Y_T + s * c_dec
            sa, sb = sig_at(y_s, M_a), sig_at(y_s, M_b)
            sl = (sb - sa) / (M_b - M_a)
            root = M_a - sa / sl
            if np.sign(chi(root)) == s:
                found = (root, y_s, sl)
                break
        self.assertIsNotNone(found, "no probe-side fixed point")
        M_star, y_probe, slope = found

        lim = StressLimits(name="dec", decompression=True,
                           c_dec=c_dec)
        verdicts = {}
        for tag, M in (("below", M_star - 2.0e6),
                       ("above", M_star + 2.0e6)):
            res = self._run([{"name": "t", "state": s0,
                              "increment": (0.0, M, 0.0),
                              "limits": lim}])
            verdicts[tag] = [
                c for c in res["stages"][0]["checks"]
                if c["name"].startswith("decompression")][0]
        pass_side, fail_side = (("above", "below") if slope < 0
                                else ("below", "above"))
        self.assertTrue(verdicts[pass_side]["passed"])
        self.assertFalse(verdicts[fail_side]["passed"])
        self.assertEqual(verdicts[fail_side]["probe_at"],
                         (XR, y_probe))
        self.assertLess(
            abs(verdicts[fail_side]["sigma_probe_MPa"]), 0.2)


# ==================================================================
#  Basis flag (D4) and check semantics
# ==================================================================

class TestBasisFlagAndChecks(_SLSCaseBase):

    def test_basis_violation_flags_and_keeps_values(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        lim = ec2_stress_limits("characteristic", fck=35.0,
                                fyk=450.0, fpk=1860.0, fct_eff=3.2)
        res = self._run([{"name": "big", "state": s0,
                          "increment": (0.0, 600.0e6, 0.0),
                          "limits": lim}])
        st = res["stages"][0]
        self.assertTrue(st["uncracked_basis_violated"])
        self.assertFalse(st["verified"])
        dep = [c for c in st["checks"] if not c.get("skipped")
               and c["basis_dependent"]]
        self.assertTrue(dep)
        for c in dep:
            self.assertIs(c.get("basis_valid"), False)
            self.assertTrue("value_MPa" in c
                            or "sigma_probe_MPa" in c)

    def test_unbonded_member_level_check(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        lim = ec2_stress_limits("characteristic", fck=35.0,
                                fpk=1860.0)
        res = self._run([{"name": "u", "state": s0,
                          "limits": lim,
                          "unbonded_sigma_p": {"T_ext": 1200.0}}])
        ub = [c for c in res["stages"][0]["checks"]
              if c["name"] == "tendon_stress[T_ext]"][0]
        self.assertEqual(ub["provenance"], "member_level")
        self.assertFalse(ub["basis_dependent"])
        self.assertTrue(ub["passed"])

    def test_missing_limits_reported_as_skipped(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        lim = ec2_stress_limits("transfer", fck=25.0)
        res = self._run([{"name": "t", "state": s0,
                          "limits": lim}])
        names = {c["name"]: c for c in res["stages"][0]["checks"]}
        self.assertTrue(names["steel_tension"].get("skipped"))
        self.assertTrue(names["tendon_stress"].get("skipped"))


# ==================================================================
#  Fail-loud guards
# ==================================================================

class TestGuards(_SLSCaseBase):

    def test_missing_concrete_modulus_raises(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        with self.assertRaises(ValueError):
            verify_sls_staged(self.sec,
                              [{"name": "s", "state": s0}],
                              moduli={})

    def test_dead_override_raises(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        with self.assertRaises(ValueError):
            verify_sls_staged(
                self.sec, [{"name": "s", "state": s0}],
                moduli={**self.moduli, "Ghost": 1.0e4})

    def test_leaving_element_raises(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        s1 = self._state([1, 1, 0], [1, 1, 1], EPS_PE, 1)
        with self.assertRaises(NotImplementedError):
            self._run([{"name": "a", "state": s0},
                       {"name": "b", "state": s1}])

    def test_compound_transition_raises(self):
        s0 = self._state([1, 1, 0], [1, 1, 1], EPS_PE, 0)
        s1 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 1,
                         beps=-1.0e-4)
        with self.assertRaises(ValueError):
            self._run([{"name": "a", "state": s0},
                       {"name": "b", "state": s1}])

    def test_linear_elastic_validation(self):
        with self.assertRaises(ValueError):
            LinearElastic(E=0.0)
        with self.assertRaises(ValueError):
            LinearElastic(E=-1.0)
        with self.assertRaises(ValueError):
            LinearElastic(E=1.0, eps_lim=0.0)

    def test_stress_limits_validation(self):
        with self.assertRaises(ValueError):
            StressLimits(sigma_c_comp=-3.0)
        with self.assertRaises(ValueError):
            StressLimits(c_dec=-1.0)
        with self.assertRaises(ValueError):
            ec2_stress_limits("transfer")
        with self.assertRaises(ValueError):
            ec2_stress_limits("transfer", fck=30.0, k9=1.0)
        with self.assertRaises(ValueError):
            ec2_stress_limits("nope", fck=30.0)


# ==================================================================
#  View substitution and transformed properties
# ==================================================================

class TestViewAndProperties(_SLSCaseBase):

    def test_sls_view_does_not_mutate_base(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        modmap = resolve_sls_moduli(self.sec, self.moduli)
        vw = sls_view(materialize_view(self.sec, s0), modmap)
        self.assertIsInstance(self.sec.bulk_material, Concrete)
        self.assertIsInstance(vw.bulk_material, LinearElastic)
        self.assertIsNot(self.sec.rebars[0].material,
                         vw.rebars[0].material)
        self.assertIsInstance(vw.tendons[0].material, LinearElastic)
        # Identity grouping preserved: both rebar layers share one
        # substituted instance.
        self.assertIs(vw.rebars[0].material, vw.rebars[1].material)

    def test_transformed_properties_include_tendon(self):
        s0 = self._state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
        p = sls_transformed_properties(self.sec, s0,
                                       moduli=self.moduli)
        A_ref = (B * H + (ES / EC - 1.0) * (AS1 + AS2)
                 + (EP / EC - 1.0) * AP)
        self.assertAlmostEqual(p.area, A_ref,
                               delta=1e-6 * A_ref)

    def test_transformed_properties_follow_state(self):
        s = self._state([1, 1, 0], [1, 1, 1], EPS_PE, 0)
        p = sls_transformed_properties(self.sec, s,
                                       moduli=self.moduli)
        A_ref = B * H + (ES / EC - 1.0) * (AS1 + AS2)
        self.assertAlmostEqual(p.area, A_ref, delta=1e-6 * A_ref)

    def test_intrinsic_steel_moduli_autoresolve(self):
        modmap = resolve_sls_moduli(self.sec, self.moduli)
        st_mat = self.sec.rebars[0].material
        ps_mat = self.sec.tendons[0].material
        self.assertEqual(modmap[id(st_mat)][1], ES)
        self.assertEqual(modmap[id(ps_mat)][1], EP)


# ==================================================================
#  Phase-7 hygiene edits (post-patcher: Steel.E / Concrete.E)
# ==================================================================

class TestHygieneEdits(unittest.TestCase):
    """Require apply_phase7_sls.py to have been applied."""

    def test_steel_E_alias(self):
        s = Steel(fyk=450.0)
        self.assertEqual(s.E, s.Es)

    def test_concrete_E_fail_loud(self):
        # AttributeError by design: direct access fails loudly, but
        # hasattr()/getattr(mat, "E", default) probing idioms (used
        # e.g. in posttension.py) degrade gracefully instead of
        # exploding — ValueError would propagate through them.
        c = Concrete(fck=35.0)
        with self.assertRaises(AttributeError):
            _ = c.E
        self.assertFalse(hasattr(c, "E"))
        self.assertIsNone(getattr(c, "E", None))
        c2 = Concrete(fck=35.0, fct=3.2, Ec=34000.0)
        self.assertEqual(c2.E, 34000.0)

    def test_compute_ideal_properties_repaired(self):
        # Finding F0: this call used to die with AttributeError for
        # Steel + Concrete sections.
        c = Concrete(fck=35.0, fct=3.2, Ec=34000.0)
        s = Steel(fyk=450.0)
        sec = RectSection(
            300.0, 500.0, c,
            [RebarLayer(y=50.0, As=628.0, material=s, x=150.0)],
            n_fibers_y=50)
        p = sec.compute_ideal_properties()
        A_ref = 300.0 * 500.0 + (200000.0 / 34000.0 - 1.0) * 628.0
        self.assertAlmostEqual(p.area, A_ref, delta=1e-6 * A_ref)


if __name__ == "__main__":
    unittest.main()
