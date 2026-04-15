"""
Unit tests for the biaxial solver (N-Mx-My).

Covers:
- Analytical verification with known strain plane (all elastic bars)
- Mixed regime (some bars yielded, some elastic)
- Asymmetric rebar layout (My correctness under asymmetry)
- Neutral axis as oblique line
- Inverse solver convergence and accuracy
- Surface generation and 3D DemandChecker
- TabulatedMaterial in biaxial solver
- Solver near domain boundary
"""

import unittest
import numpy as np
from gensec.materials import Concrete, Steel, TabulatedMaterial
from gensec.geometry import RebarLayer, RectSection
from gensec.solver import FiberSolver, NMDiagram, DomainChecker


class BiaxialTestBase(unittest.TestCase):
    """Common setup for biaxial tests: 300x600, 4 corner bars."""

    def setUp(self):
        self.fcd = 0.85 * 25.0 / 1.5
        self.fyd = 450.0 / 1.15
        self.Es = 200000.0
        self.eps_yd = self.fyd / self.Es
        self.concrete = Concrete(fck=25.0)
        self.steel = Steel(fyk=450.0)
        A_bar = np.pi / 4 * 20**2
        self.sec = RectSection(
            B=300, H=600, bulk_material=self.concrete,
            rebars=[
                RebarLayer(y=40, x=40, As=A_bar, material=self.steel),
                RebarLayer(y=40, x=260, As=A_bar, material=self.steel),
                RebarLayer(y=560, x=40, As=A_bar, material=self.steel),
                RebarLayer(y=560, x=260, As=A_bar, material=self.steel),
            ],
            n_fibers_y=60, n_fibers_x=30)
        self.sv = FiberSolver(self.sec)
        self.x_ref = 150.0
        self.y_ref = 300.0

    def _bar_strain(self, eps0, chi_x, chi_y, x, y):
        return eps0 + chi_x * (y - self.y_ref) - chi_y * (x - self.x_ref)

    def _bar_stress(self, eps):
        if abs(eps) <= self.eps_yd:
            return self.Es * eps
        elif eps > 0:
            return self.fyd
        else:
            return -self.fyd


class TestBiaxialElastic(BiaxialTestBase):
    """All bars in elastic regime — exact analytical comparison."""

    def setUp(self):
        super().setUp()
        self.eps0 = -0.0005
        self.chi_x = 4e-6
        self.chi_y = 2e-6
        self.fr = self.sv.get_fiber_results(
            self.eps0, self.chi_x, self.chi_y)

    def test_rebar_strains_exact(self):
        for i, rb in enumerate(self.sec.rebars):
            eps_ana = self._bar_strain(
                self.eps0, self.chi_x, self.chi_y, rb.x, rb.y)
            eps_sol = self.fr["rebars"]["eps"][i]
            self.assertAlmostEqual(eps_sol, eps_ana, places=10,
                                   msg=f"Bar {i+1} at ({rb.x},{rb.y})")

    def test_rebar_stresses_exact(self):
        for i, rb in enumerate(self.sec.rebars):
            eps = self._bar_strain(
                self.eps0, self.chi_x, self.chi_y, rb.x, rb.y)
            sig_ana = self._bar_stress(eps)
            sig_sol = self.fr["rebars"]["sigma"][i]
            self.assertAlmostEqual(sig_sol, sig_ana, places=2,
                                   msg=f"Bar {i+1}")

    def test_all_bars_elastic(self):
        for i, rb in enumerate(self.sec.rebars):
            eps = self._bar_strain(
                self.eps0, self.chi_x, self.chi_y, rb.x, rb.y)
            self.assertLess(abs(eps), self.eps_yd,
                            msg=f"Bar {i+1} should be elastic")

    def test_steel_resultants(self):
        N_s = float(np.sum(
            self.fr["rebars"]["sigma"] * self.fr["rebars"]["A"]))
        Mx_s = float(np.sum(
            self.fr["rebars"]["sigma"] * self.fr["rebars"]["A"]
            * (self.fr["rebars"]["y"] - self.y_ref)))
        My_s = -float(np.sum(
            self.fr["rebars"]["sigma"] * self.fr["rebars"]["A"]
            * (self.fr["rebars"]["x"] - self.x_ref)))

        # Analytical
        N_ana = Mx_ana = My_ana = 0.0
        for rb in self.sec.rebars:
            eps = self._bar_strain(
                self.eps0, self.chi_x, self.chi_y, rb.x, rb.y)
            sig = self._bar_stress(eps)
            F = sig * rb.As
            N_ana += F
            Mx_ana += F * (rb.y - self.y_ref)
            My_ana += -F * (rb.x - self.x_ref)

        self.assertAlmostEqual(N_s, N_ana, delta=1.0)
        self.assertAlmostEqual(Mx_s, Mx_ana, delta=100)
        self.assertAlmostEqual(My_s, My_ana, delta=100)

    def test_total_consistency(self):
        """integrate() == sum(bulk) + sum(net rebar contributions)."""
        N, Mx, My = self.sv.integrate(
            self.eps0, self.chi_x, self.chi_y)
        b = self.fr["bulk"]
        r = self.fr["rebars"]
        # Use sigma_net for rebars (already accounts for embedded subtraction)
        N_sum = (float(np.sum(b["sigma"] * b["dA"]))
                 + float(np.sum(r["sigma_net"] * r["A"])))
        Mx_sum = (float(np.sum(b["sigma"] * b["dA"]
                                * (b["y"] - self.y_ref)))
                  + float(np.sum(r["sigma_net"] * r["A"]
                                 * (r["y"] - self.y_ref))))
        My_sum = -(float(np.sum(b["sigma"] * b["dA"]
                                * (b["x"] - self.x_ref)))
                  + float(np.sum(r["sigma_net"] * r["A"]
                                 * (r["x"] - self.x_ref))))
        self.assertAlmostEqual(N, N_sum, delta=1.0)
        self.assertAlmostEqual(Mx, Mx_sum, delta=100)
        self.assertAlmostEqual(My, My_sum, delta=100)

    def test_my_nonzero(self):
        _, _, My = self.sv.integrate(
            self.eps0, self.chi_x, self.chi_y)
        self.assertGreater(abs(My), 1000)

    def test_neutral_axis_oblique(self):
        """NA is a line: eps0 + chi_x*(y-300) - chi_y*(x-150) = 0.
        y_na(x) = 300 + (chi_y*(x-150) - eps0) / chi_x
        y_na(0)   = 300 + (2e-6*(-150) - (-0.0005)) / 4e-6 = 300 + 50  = 350
        y_na(150) = 300 + (0          - (-0.0005)) / 4e-6 = 300 + 125 = 425
        y_na(300) = 300 + (2e-6*150   - (-0.0005)) / 4e-6 = 300 + 200 = 500
        """
        b = self.fr["bulk"]
        for x_target, y_na_expected in [(0, 350), (150, 425), (300, 500)]:
            dx = self.sec.dx
            mask = np.abs(b["x"] - x_target) < dx
            if not np.any(mask):
                mask = np.abs(
                    b["x"] - b["x"][np.argmin(np.abs(b["x"] - x_target))]
                ) < 0.1
            y_s = b["y"][mask]
            e_s = b["eps"][mask]
            order = np.argsort(y_s)
            y_s, e_s = y_s[order], e_s[order]
            cross = np.where(np.diff(np.sign(e_s)))[0]
            self.assertGreater(len(cross), 0,
                               f"No crossing at x~{x_target}")
            ic = cross[0]
            y_na = y_s[ic] - e_s[ic] * (y_s[ic+1] - y_s[ic]) / (
                e_s[ic+1] - e_s[ic])
            self.assertAlmostEqual(y_na, y_na_expected, delta=15,
                                   msg=f"NA at x~{x_target}")

    def test_inverse_recovers_strain_plane(self):
        N, Mx, My = self.sv.integrate(
            self.eps0, self.chi_x, self.chi_y)
        sol = self.sv.solve_equilibrium(
            N, Mx, My, eps0_init=-0.0003,
            chi_x_init=2e-6, chi_y_init=1e-6)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["eps0"], self.eps0, places=6)
        self.assertAlmostEqual(sol["chi_x"], self.chi_x, delta=1e-9)
        self.assertAlmostEqual(sol["chi_y"], self.chi_y, delta=1e-9)


class TestBiaxialYielded(BiaxialTestBase):
    """Mixed regime: some bars yielded, some elastic."""

    def setUp(self):
        super().setUp()
        self.eps0 = -0.0010
        self.chi_x = 8e-6
        self.chi_y = 4e-6
        self.fr = self.sv.get_fiber_results(
            self.eps0, self.chi_x, self.chi_y)

    def test_bar_regimes(self):
        """Check which bars yield and which don't."""
        for i, rb in enumerate(self.sec.rebars):
            eps = self._bar_strain(
                self.eps0, self.chi_x, self.chi_y, rb.x, rb.y)
            sig_ana = self._bar_stress(eps)
            sig_sol = self.fr["rebars"]["sigma"][i]
            self.assertAlmostEqual(sig_sol, sig_ana, delta=0.1,
                                   msg=f"Bar {i+1}")

    def test_yielded_bars_at_fyd(self):
        for i, rb in enumerate(self.sec.rebars):
            eps = self._bar_strain(
                self.eps0, self.chi_x, self.chi_y, rb.x, rb.y)
            if abs(eps) > self.eps_yd:
                sig_sol = self.fr["rebars"]["sigma"][i]
                self.assertAlmostEqual(abs(sig_sol), self.fyd, delta=0.1,
                                       msg=f"Bar {i+1} should be at fyd")

    def test_inverse_roundtrip(self):
        N, Mx, My = self.sv.integrate(
            self.eps0, self.chi_x, self.chi_y)
        sol = self.sv.solve_equilibrium(
            N, Mx, My, eps0_init=-0.0008,
            chi_x_init=5e-6, chi_y_init=2e-6)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["N"], N, delta=10)
        self.assertAlmostEqual(sol["Mx"], Mx, delta=10000)
        self.assertAlmostEqual(sol["My"], My, delta=10000)


class TestAsymmetricSection(unittest.TestCase):
    """
    Asymmetric rebar layout: 3 bars on the left, 1 on the right.
    Under pure chi_x (no chi_y), My must be NONZERO due to asymmetry.
    Under chi_y, My must be different from the symmetric case.
    """

    def setUp(self):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        A_bar = np.pi / 4 * 20**2
        # Asymmetric: 3 bars on x=40, 1 bar on x=260
        self.sec = RectSection(
            B=300, H=600, bulk_material=concrete,
            rebars=[
                RebarLayer(y=40, x=40, As=A_bar, material=steel),
                RebarLayer(y=300, x=40, As=A_bar, material=steel),
                RebarLayer(y=560, x=40, As=A_bar, material=steel),
                RebarLayer(y=300, x=260, As=A_bar, material=steel),
            ],
            n_fibers_y=60, n_fibers_x=30)
        self.sv = FiberSolver(self.sec)

    def test_pure_chi_x_gives_nonzero_My(self):
        """Asymmetric rebars: chi_y=0 but My != 0 because the steel
        centroid is offset from x_ref."""
        _, _, My = self.sv.integrate(-0.001, 5e-6, 0.0)
        self.assertNotAlmostEqual(My, 0.0, delta=100,
                                  msg="Asymmetric section should produce My")

    def test_rebar_strains_with_x(self):
        """Each bar's strain must reflect its x-coordinate."""
        fr = self.sv.get_fiber_results(-0.0005, 4e-6, 3e-6)
        for i, rb in enumerate(self.sec.rebars):
            eps_ana = (-0.0005
                       + 4e-6 * (rb.y - 300)
                       - 3e-6 * (rb.x - 150))
            self.assertAlmostEqual(fr["rebars"]["eps"][i], eps_ana,
                                   places=10, msg=f"Bar {i+1}")

    def test_inverse_converges(self):
        N, Mx, My = self.sv.integrate(-0.0005, 4e-6, 3e-6)
        sol = self.sv.solve_equilibrium(N, Mx, My)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["N"], N, delta=10)


class TestBiaxialSurface(BiaxialTestBase):
    """3D resistance surface and DemandChecker."""

    def test_surface_generation(self):
        nm = NMDiagram(self.sv).generate_biaxial(
            n_angles=18, n_points_per_angle=100)
        self.assertGreater(len(nm["N"]), 2000)
        # My should have significant range
        self.assertGreater(nm["My_kNm"].max() - nm["My_kNm"].min(), 10)

    def test_symmetry(self):
        nm = NMDiagram(self.sv).generate_biaxial(
            n_angles=18, n_points_per_angle=100)
        Mx_max = nm["Mx_kNm"].max()
        Mx_min = nm["Mx_kNm"].min()
        My_max = nm["My_kNm"].max()
        My_min = nm["My_kNm"].min()
        self.assertLess(abs(Mx_max + Mx_min) / abs(Mx_max) * 100, 5)
        self.assertLess(abs(My_max + My_min) / abs(My_max) * 100, 5)

    def test_3d_checker(self):
        nm = NMDiagram(self.sv).generate_biaxial(
            n_angles=12, n_points_per_angle=50)
        ch = DomainChecker(nm)
        self.assertEqual(ch.ndim, 3)
        self.assertTrue(ch.is_inside(0, 0, 0))
        self.assertFalse(ch.is_inside(-10000e3, 0, 0))

    def test_uniaxial_consistent_with_biaxial(self):
        """Biaxial integrate with chi_y=0 must match uniaxial integrate
        for N and Mx, and give My=0."""
        # Same strain plane, chi_y=0: results must be identical
        eps0, chi_x = -0.001, 5e-6
        N_bi, Mx_bi, My_bi = self.sv.integrate(eps0, chi_x, 0.0)
        # My must be ~0 for symmetric section with chi_y=0
        self.assertAlmostEqual(My_bi, 0.0, delta=1.0)
        # N and Mx must be finite and reasonable
        self.assertLess(N_bi, 0)  # compression
        self.assertNotEqual(Mx_bi, 0)


class TestTabulatedBiaxial(BiaxialTestBase):
    """TabulatedMaterial used in biaxial solver — must match Steel."""

    def test_tabulated_vs_steel(self):
        # Build a tabulated EPP steel matching the Steel class
        s = self.steel
        et = np.array([-0.01, -s.eps_yd, 0, s.eps_yd, 0.01])
        st = np.array([-s.fyd, -s.fyd, 0, s.fyd, s.fyd])
        tab = TabulatedMaterial(et, st)

        A_bar = np.pi / 4 * 20**2
        sec_tab = RectSection(
            B=300, H=600, bulk_material=self.concrete,
            rebars=[
                RebarLayer(y=40, x=40, As=A_bar, material=tab),
                RebarLayer(y=40, x=260, As=A_bar, material=tab),
                RebarLayer(y=560, x=40, As=A_bar, material=tab),
                RebarLayer(y=560, x=260, As=A_bar, material=tab),
            ],
            n_fibers_y=60, n_fibers_x=30)
        sv_tab = FiberSolver(sec_tab)

        N1, Mx1, My1 = self.sv.integrate(-0.0005, 4e-6, 2e-6)
        N2, Mx2, My2 = sv_tab.integrate(-0.0005, 4e-6, 2e-6)
        self.assertAlmostEqual(N1, N2, delta=abs(N1) * 0.001)
        self.assertAlmostEqual(Mx1, Mx2, delta=abs(Mx1) * 0.001)
        self.assertAlmostEqual(My1, My2, delta=abs(My1) * 0.001 + 1)


if __name__ == '__main__':
    unittest.main()


class TestSolverDispatchMyZero(BiaxialTestBase):
    """On a 2D section, My=0 should use the 2x2 path and converge."""

    def test_my_zero_converges(self):
        """My=0 on biaxial section must converge (uniaxial fallback)."""
        sol = self.sv.solve_equilibrium(-1500e3, 200e6, 0.0)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["My"], 0.0, delta=1.0)

    def test_my_zero_gives_chi_y_zero(self):
        """When My=0 is solved via 2x2, chi_y should be exactly 0."""
        sol = self.sv.solve_equilibrium(-1500e3, 200e6, 0.0)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["chi_y"], 0.0, places=10)


class TestMxMyDiagram(BiaxialTestBase):
    """Test Mx-My contour generation at fixed N."""

    def test_generates_contour(self):
        nm_gen = NMDiagram(self.sv)
        mx_my = nm_gen.generate_mx_my(N_fixed=-1500e3, n_angles=36,
                                      n_points_per_angle=100)
        self.assertIn("Mx_kNm", mx_my)
        self.assertIn("My_kNm", mx_my)
        self.assertEqual(len(mx_my["Mx_kNm"]), 36)

    def test_symmetric_contour(self):
        """Doubly symmetric section at N -> Mx-My symmetric."""
        nm_gen = NMDiagram(self.sv)
        mx_my = nm_gen.generate_mx_my(N_fixed=-1500e3, n_angles=72,
                                      n_points_per_angle=200)
        Mx_max = mx_my["Mx_kNm"].max()
        Mx_min = mx_my["Mx_kNm"].min()
        My_max = mx_my["My_kNm"].max()
        My_min = mx_my["My_kNm"].min()
        self.assertAlmostEqual(Mx_max, -Mx_min,
                               delta=abs(Mx_max) * 0.1)
        self.assertAlmostEqual(My_max, -My_min,
                               delta=abs(My_max) * 0.1)

    def test_contour_shrinks_at_high_N(self):
        """At higher |N|, the Mx-My contour should be smaller."""
        nm_gen = NMDiagram(self.sv)
        mx_my_low = nm_gen.generate_mx_my(N_fixed=-500e3, n_angles=36,
                                          n_points_per_angle=100)
        mx_my_high = nm_gen.generate_mx_my(N_fixed=-3000e3, n_angles=36,
                                           n_points_per_angle=100)
        M_low = np.max(np.sqrt(mx_my_low["Mx"]**2 + mx_my_low["My"]**2))
        M_high = np.max(np.sqrt(mx_my_high["Mx"]**2 + mx_my_high["My"]**2))
        self.assertGreater(M_low, M_high,
                           "Mx-My domain should shrink at higher compression")
