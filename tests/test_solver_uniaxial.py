"""
Unit tests for the uniaxial solver (N-Mx, chi_y=0).

Covers: pure compression/tension, bending N=0, round-trip, symmetry,
detailed fiber state (neutral axis, yielded/not yielded rebars,
concrete stress profile).
"""

import unittest
import numpy as np
from gensec.materials import Concrete, Steel
from gensec.geometry import RebarLayer, RectSection
from gensec.solver import FiberSolver, NMDiagram


class TestPureAxial(unittest.TestCase):
    """Pure compression and pure tension against analytical values."""

    def setUp(self):
        self.fcd = 0.85 * 25.0 / 1.5
        self.fyd = 450.0 / 1.15
        As = 3 * np.pi / 4 * 16**2
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        rb = [RebarLayer(y=40, As=As, material=steel, n_bars=3, diameter=16),
              RebarLayer(y=460, As=As, material=steel, n_bars=3, diameter=16)]
        self.sec = RectSection(B=300, H=500, bulk_material=concrete,
                               rebars=rb, n_fibers_y=200, n_fibers_x=1)
        self.nm = NMDiagram(FiberSolver(self.sec)).generate(500)
        self.As_tot = 2 * As

    def test_compression(self):
        # Correct formula with embedded bar subtraction:
        # N_Rd = fcd * B * H + (fyd - fcd) * As_tot
        N_ana = -(self.fcd * 300 * 500 + (self.fyd - self.fcd) * self.As_tot)
        N_sol = self.nm["N"].min()
        self.assertAlmostEqual(N_sol, N_ana, delta=abs(N_ana) * 0.001)

    def test_tension(self):
        # In pure tension, concrete stress is zero everywhere.
        # No bulk to subtract: N = fyd * As_tot
        N_ana = self.fyd * self.As_tot
        N_sol = self.nm["N"].max()
        self.assertAlmostEqual(N_sol, N_ana, delta=N_ana * 0.001)


class TestSimpleBending(unittest.TestCase):
    """Simple bending at N=0 against EC2 stress-block formula."""

    def _check_bending(self, n_bars, diameter, d, tol_pct=3.0):
        As = n_bars * np.pi / 4 * diameter**2
        fcd = 0.85 * 25.0 / 1.5
        fyd = 450.0 / 1.15
        B, H = 300.0, 500.0

        # Analytical (stress block)
        x = As * fyd / (0.8 * B * fcd)
        Mrd_ana = As * fyd * (d - 0.8 * x / 2)

        concrete = Concrete(fck=25.0)
        steel_t = Steel(fyk=450.0, works_in_compression=False)
        rb = [RebarLayer(y=H - d, As=As, material=steel_t,
                         n_bars=n_bars, diameter=diameter)]
        sec = RectSection(B=B, H=H, bulk_material=concrete,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        nm = NMDiagram(FiberSolver(sec)).generate(500)
        m = np.abs(nm["N"]) < 5000
        M_sol = np.max(np.abs(nm["M"][m])) if np.any(m) else 0
        err = abs(M_sol - Mrd_ana) / Mrd_ana * 100
        self.assertLess(err, tol_pct,
                        f"M_sol={M_sol/1e6:.2f}, M_ana={Mrd_ana/1e6:.2f},"
                        f" err={err:.2f}%")

    def test_low_rho(self):
        self._check_bending(4, 20, 460)

    def test_higher_rho(self):
        self._check_bending(6, 20, 450)


class TestRoundTrip(unittest.TestCase):
    """Forward integrate then inverse solve: must recover same state."""

    def setUp(self):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        rb = [RebarLayer(y=40, As=942.5, material=steel),
              RebarLayer(y=560, As=942.5, material=steel)]
        self.sec = RectSection(B=300, H=600, bulk_material=concrete,
                               rebars=rb, n_fibers_y=200, n_fibers_x=1)
        self.sv = FiberSolver(self.sec)

    def test_roundtrip(self):
        Nf, Mxf, _ = self.sv.integrate(-0.0008, 3e-6, 0.0)
        sol = self.sv.solve_equilibrium(Nf, Mxf, 0.0)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["N"], Nf, delta=1.0)
        self.assertAlmostEqual(sol["Mx"], Mxf, delta=1000.0)


class TestSymmetry(unittest.TestCase):
    """Doubly symmetric section: N-M diagram must be symmetric in M."""

    def test_symmetric_diagram(self):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        rb = [RebarLayer(y=40, As=942.5, material=steel),
              RebarLayer(y=560, As=942.5, material=steel)]
        sec = RectSection(B=300, H=600, bulk_material=concrete,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        nm = NMDiagram(FiberSolver(sec)).generate(500)
        M_max = nm["M_kNm"].max()
        M_min = nm["M_kNm"].min()
        sym_err = abs(M_max + M_min) / max(abs(M_max), abs(M_min)) * 100
        self.assertLess(sym_err, 1.0)


class TestDetailedFiberState(unittest.TestCase):
    """
    Known strain plane: verify neutral axis, rebar states, concrete
    stress profile — all analytically.
    """

    def setUp(self):
        concrete = Concrete(fck=25.0)
        self.fcd = 0.85 * 25.0 / 1.5
        self.fyd = 450.0 / 1.15
        self.Es = 200000.0
        self.eps_yd = self.fyd / self.Es
        steel = Steel(fyk=450.0)
        rb = [RebarLayer(y=40, As=942.5, material=steel),
              RebarLayer(y=560, As=942.5, material=steel)]
        self.sec = RectSection(B=300, H=600, bulk_material=concrete,
                               rebars=rb, n_fibers_y=200, n_fibers_x=1)
        self.sv = FiberSolver(self.sec)

        # Strain plane: eps(y) = -0.001 + 8e-6*(y-300)
        self.eps0 = -0.001
        self.chi_x = 8e-6
        self.fr = self.sv.get_fiber_results(self.eps0, self.chi_x, 0.0)

    def test_neutral_axis(self):
        """NA at y = 300 + 0.001/8e-6 = 425 mm."""
        by = self.fr["bulk"]["y"]
        be = self.fr["bulk"]["eps"]
        idx = np.where(np.diff(np.sign(be)))[0]
        self.assertGreater(len(idx), 0)
        i = idx[0]
        y_na = by[i] - be[i] * (by[i+1] - by[i]) / (be[i+1] - be[i])
        self.assertAlmostEqual(y_na, 425.0, delta=5.0)

    def test_rebar_bottom_yielded(self):
        """eps(40) = -0.001 + 8e-6*(-260) = -0.00308 -> yielded."""
        eps_ana = self.eps0 + self.chi_x * (40 - 300)
        self.assertLess(eps_ana, -self.eps_yd)  # past yield
        sig = self.fr["rebars"]["sigma"][0]
        self.assertAlmostEqual(sig, -self.fyd, places=1)

    def test_rebar_top_elastic(self):
        """eps(560) = -0.001 + 8e-6*(260) = +0.00108 -> elastic."""
        eps_ana = self.eps0 + self.chi_x * (560 - 300)
        self.assertLess(abs(eps_ana), self.eps_yd)  # elastic
        sig = self.fr["rebars"]["sigma"][1]
        self.assertAlmostEqual(sig, self.Es * eps_ana, places=1)

    def test_concrete_tension_zero(self):
        """All fibers with eps > 0 must have sigma = 0."""
        mask = self.fr["bulk"]["eps"] > 1e-10
        if np.any(mask):
            self.assertAlmostEqual(
                np.max(np.abs(self.fr["bulk"]["sigma"][mask])), 0.0,
                places=3)

    def test_concrete_compression_negative(self):
        """All fibers with eps < 0 must have sigma < 0."""
        mask = self.fr["bulk"]["eps"] < -1e-10
        self.assertTrue(np.all(self.fr["bulk"]["sigma"][mask] < 0))

    def test_concrete_plateau(self):
        """Near y=100: eps ~ -0.0026 (between eps_c2 and eps_cu2)
        -> sigma = -fcd."""
        by = self.fr["bulk"]["y"]
        idx = np.argmin(np.abs(by - 100))
        eps = self.fr["bulk"]["eps"][idx]
        sig = self.fr["bulk"]["sigma"][idx]
        self.assertLess(eps, -0.002)
        self.assertAlmostEqual(sig, -self.fcd, delta=0.1)


class TestHighStrengthConcreteSolver(unittest.TestCase):
    """Run the solver with fck=70 (n!=2, different eps_c2, eps_cu2)."""

    def test_nm_diagram_generates(self):
        from gensec.materials import concrete_from_ec2
        c70 = concrete_from_ec2(fck=70, ls='F')
        steel = Steel(fyk=500.0, gamma_s=1.15)
        As = 4 * np.pi / 4 * 25**2
        rb = [RebarLayer(y=50, As=As, material=steel),
              RebarLayer(y=550, As=As, material=steel)]
        sec = RectSection(B=400, H=600, bulk_material=c70,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        nm = NMDiagram(FiberSolver(sec)).generate(400)
        # Must produce valid N range
        self.assertLess(nm["N"].min(), -3000e3)
        self.assertGreater(nm["N"].max(), 0)
        # M must have both signs (symmetric section)
        self.assertGreater(nm["M"].max(), 0)
        self.assertLess(nm["M"].min(), 0)

    def test_inverse_converges(self):
        from gensec.materials import concrete_from_ec2
        c70 = concrete_from_ec2(fck=70, ls='F')
        steel = Steel(fyk=500.0, gamma_s=1.15)
        As = 4 * np.pi / 4 * 25**2
        rb = [RebarLayer(y=50, As=As, material=steel),
              RebarLayer(y=550, As=As, material=steel)]
        sec = RectSection(B=400, H=600, bulk_material=c70,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        sv = FiberSolver(sec)
        N, Mx, _ = sv.integrate(-0.0008, 3e-6)
        sol = sv.solve_equilibrium(N, Mx)
        self.assertTrue(sol["converged"])


if __name__ == '__main__':
    unittest.main()


class TestSolverRobustness(unittest.TestCase):
    """
    Systematic convergence test: the solver must converge for every
    point inside the resistance domain.
    """

    def setUp(self):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        A20 = np.pi / 4 * 20**2
        rb = [RebarLayer(y=40, As=3 * A20, material=steel),
              RebarLayer(y=560, As=3 * A20, material=steel)]
        self.sec = RectSection(B=300, H=600, bulk_material=concrete,
                               rebars=rb, n_fibers_y=200, n_fibers_x=1)
        self.sv = FiberSolver(self.sec)
        from gensec.solver import DomainChecker
        nm = NMDiagram(self.sv).generate(500)
        self.checker = DomainChecker(nm)

    def test_grid_convergence(self):
        """Every grid point inside the domain must converge."""
        failed = []
        for N_kN in np.arange(-3000, 800, 300):
            for M_kNm in np.arange(-350, 360, 70):
                if not self.checker.is_inside(N_kN * 1e3, M_kNm * 1e6):
                    continue
                sol = self.sv.solve_equilibrium(N_kN * 1e3, M_kNm * 1e6)
                if not sol["converged"]:
                    failed.append((N_kN, M_kNm))
        self.assertEqual(len(failed), 0,
                         f"Failed to converge at: {failed}")

    def test_outside_domain_does_not_crash(self):
        """Points outside the domain may not converge but must not crash."""
        for N_kN, M_kNm in [(-5000, 0), (0, 500), (-1500, 500)]:
            sol = self.sv.solve_equilibrium(N_kN * 1e3, M_kNm * 1e6)
            # Just check it returns a valid dict, regardless of convergence
            self.assertIn("converged", sol)
            self.assertIn("N", sol)


class TestEmbeddedBarSubtraction(unittest.TestCase):
    """
    Verify that embedded bars correctly subtract the displaced bulk
    material, and that non-embedded bars do not.
    """

    def setUp(self):
        self.concrete = Concrete(fck=25.0)
        self.fcd = 0.85 * 25.0 / 1.5
        self.fyd = 450.0 / 1.15
        self.steel = Steel(fyk=450.0)

    def test_pure_compression_embedded(self):
        """N_Rd = fcd*B*H + (fyd-fcd)*As for embedded bars."""
        As = 1000.0  # large area to make difference visible
        rb = [RebarLayer(y=250, As=As, material=self.steel,
                         embedded=True)]
        sec = RectSection(B=300, H=500, bulk_material=self.concrete,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        nm = NMDiagram(FiberSolver(sec)).generate(500)
        N_ana = -(self.fcd * 300 * 500 + (self.fyd - self.fcd) * As)
        N_sol = nm["N"].min()
        self.assertAlmostEqual(N_sol, N_ana, delta=abs(N_ana) * 0.002)

    def test_pure_compression_not_embedded(self):
        """N_Rd = fcd*B*H + fyd*As for non-embedded bars."""
        As = 1000.0
        rb = [RebarLayer(y=250, As=As, material=self.steel,
                         embedded=False)]
        sec = RectSection(B=300, H=500, bulk_material=self.concrete,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        nm = NMDiagram(FiberSolver(sec)).generate(500)
        N_ana = -(self.fcd * 300 * 500 + self.fyd * As)
        N_sol = nm["N"].min()
        self.assertAlmostEqual(N_sol, N_ana, delta=abs(N_ana) * 0.002)

    def test_embedded_vs_not_embedded_difference(self):
        """The difference between embedded and not-embedded must be
        exactly fcd * As at pure compression."""
        As = 2000.0
        rb_emb = [RebarLayer(y=250, As=As, material=self.steel,
                             embedded=True)]
        rb_ext = [RebarLayer(y=250, As=As, material=self.steel,
                             embedded=False)]
        sec_emb = RectSection(B=300, H=500, bulk_material=self.concrete,
                              rebars=rb_emb, n_fibers_y=200,
                              n_fibers_x=1)
        sec_ext = RectSection(B=300, H=500, bulk_material=self.concrete,
                              rebars=rb_ext, n_fibers_y=200,
                              n_fibers_x=1)
        nm_emb = NMDiagram(FiberSolver(sec_emb)).generate(500)
        nm_ext = NMDiagram(FiberSolver(sec_ext)).generate(500)
        # At pure compression, difference = fcd * As
        diff = abs(nm_ext["N"].min()) - abs(nm_emb["N"].min())
        expected_diff = self.fcd * As
        self.assertAlmostEqual(diff, expected_diff,
                               delta=expected_diff * 0.02)

    def test_tension_no_difference(self):
        """In pure tension, bulk stress is zero -> no subtraction
        regardless of embedded flag."""
        As = 2000.0
        rb_emb = [RebarLayer(y=250, As=As, material=self.steel,
                             embedded=True)]
        rb_ext = [RebarLayer(y=250, As=As, material=self.steel,
                             embedded=False)]
        sec_emb = RectSection(B=300, H=500, bulk_material=self.concrete,
                              rebars=rb_emb, n_fibers_y=200,
                              n_fibers_x=1)
        sec_ext = RectSection(B=300, H=500, bulk_material=self.concrete,
                              rebars=rb_ext, n_fibers_y=200,
                              n_fibers_x=1)
        nm_emb = NMDiagram(FiberSolver(sec_emb)).generate(500)
        nm_ext = NMDiagram(FiberSolver(sec_ext)).generate(500)
        # In tension, concrete contributes zero -> no difference
        diff = abs(nm_ext["N"].max() - nm_emb["N"].max())
        self.assertAlmostEqual(diff, 0.0, delta=100)  # < 0.1 kN


class TestSignConventions(unittest.TestCase):
    """
    Verify that the sign conventions are physically correct and
    internally consistent.

    Convention:
      - eps > 0: tension, eps < 0: compression
      - N > 0: tension, N < 0: compression
      - Mx = sum(sigma * A * (y - yref))
        -> Mx > 0 when bottom is compressed (sagging)
        -> Mx < 0 when top is compressed (hogging)
    """

    def setUp(self):
        self.c = Concrete(fck=25.0)
        self.s = Steel(fyk=450.0)
        A20 = np.pi / 4 * 20**2
        rb = [RebarLayer(y=40, As=3 * A20, material=self.s),
              RebarLayer(y=560, As=3 * A20, material=self.s)]
        self.sec = RectSection(B=300, H=600, bulk_material=self.c,
                               rebars=rb, n_fibers_y=100, n_fibers_x=1)
        self.sv = FiberSolver(self.sec)

    def test_pure_compression_N_negative(self):
        N, Mx, My = self.sv.integrate(-0.002, 0.0, 0.0)
        self.assertLess(N, 0)

    def test_pure_tension_N_positive(self):
        N, Mx, My = self.sv.integrate(0.005, 0.0, 0.0)
        self.assertGreater(N, 0)

    def test_uniform_strain_zero_moment(self):
        """Uniform strain -> zero curvature -> Mx = 0."""
        N, Mx, My = self.sv.integrate(-0.001, 0.0, 0.0)
        self.assertAlmostEqual(Mx, 0.0, delta=1.0)

    def test_top_compressed_Mx_negative(self):
        """chi_x < 0 -> top more negative (compressed) -> Mx < 0."""
        N, Mx, My = self.sv.integrate(0.0, -1e-5, 0.0)
        self.assertLess(Mx, 0)

    def test_bottom_compressed_Mx_positive(self):
        """chi_x > 0 -> bottom more negative (compressed) -> Mx > 0."""
        N, Mx, My = self.sv.integrate(0.0, 1e-5, 0.0)
        self.assertGreater(Mx, 0)

    def test_rebar_tension_when_top_compressed(self):
        """When top is compressed (chi_x<0), bottom rebar is in tension."""
        fr = self.sv.get_fiber_results(0.0, -1e-5, 0.0)
        # Bottom rebar at y=40: eps > 0, sigma > 0
        self.assertGreater(fr["rebars"]["eps"][0], 0)
        self.assertGreater(fr["rebars"]["sigma"][0], 0)
        # Top rebar at y=560: eps < 0, sigma < 0
        self.assertLess(fr["rebars"]["eps"][1], 0)
        self.assertLess(fr["rebars"]["sigma"][1], 0)


class TestMomentCurvature(unittest.TestCase):
    """Test moment-curvature diagram generation."""

    def setUp(self):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        A20 = np.pi / 4 * 20**2
        rb = [RebarLayer(y=40, As=3 * A20, material=steel),
              RebarLayer(y=560, As=3 * A20, material=steel)]
        self.sec = RectSection(B=300, H=600, bulk_material=concrete,
                               rebars=rb, n_fibers_y=100, n_fibers_x=1)
        self.sv = FiberSolver(self.sec)
        self.nm_gen = NMDiagram(self.sv)

    def test_generates_data(self):
        mc = self.nm_gen.generate_moment_curvature(N_fixed=-1000e3)
        self.assertIn("chi", mc)
        self.assertIn("M_kNm", mc)
        self.assertGreater(len(mc["chi"]), 100)

    def test_m_increases_with_chi(self):
        """Moment should increase with curvature (at least initially)."""
        mc = self.nm_gen.generate_moment_curvature(N_fixed=-1000e3)
        # Positive curvature side
        pos_mask = mc["chi"] > 0
        chi_pos = mc["chi"][pos_mask]
        M_pos = mc["M"][pos_mask]
        # First 10% of points: M should be increasing
        n10 = max(1, len(chi_pos) // 10)
        self.assertGreater(M_pos[n10], M_pos[0])

    def test_yield_detected(self):
        mc = self.nm_gen.generate_moment_curvature(N_fixed=-500e3)
        self.assertIsNotNone(mc["yield_chi_pos"])
        self.assertIsNotNone(mc["yield_M_pos"])
        self.assertGreater(abs(mc["yield_chi_pos"]), 0)

    def test_ultimate_detected(self):
        mc = self.nm_gen.generate_moment_curvature(N_fixed=-500e3)
        self.assertIsNotNone(mc["ultimate_chi_pos"])
        self.assertGreater(abs(mc["ultimate_chi_pos"]),
                           abs(mc["yield_chi_pos"]))

    def test_symmetric_section_symmetric_mchi(self):
        """Doubly symmetric section -> M-chi symmetric."""
        mc = self.nm_gen.generate_moment_curvature(N_fixed=-1000e3)
        M_max = mc["M_kNm"].max()
        M_min = mc["M_kNm"].min()
        self.assertAlmostEqual(M_max, -M_min, delta=abs(M_max) * 0.05)
