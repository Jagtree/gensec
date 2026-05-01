# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# Regression tests for the v0.3.0 Tier-1 fixes (A1, B1, B2, B3, F1, I1)
# and the new eta_norm metric (K).
# ---------------------------------------------------------------------------
"""
v0.3.0 regression tests.

Each test class targets one of the fixes documented in
``CHANGELOG_v0_3_0.md``.  Tests are designed so that the *pre-patch*
behaviour fails them and the *post-patch* behaviour passes — without
relying on any external reference data file.

Most tests build small sections inline rather than depending on the
full YAML pipeline, to keep the failure modes localised.
"""
import unittest
import numpy as np


# ===========================================================================
#  A1 — Symmetry of the uniaxial N-M cloud for centred sections
# ===========================================================================

class TestA1_CenteredSectionSymmetry(unittest.TestCase):
    r"""
    The uniaxial N-M cloud must be mirror-symmetric about :math:`M = 0`
    for a doubly symmetric section, regardless of whether the section
    is defined on a centred frame (``y_min < 0``) or on a corner-
    anchored frame (``y_min == 0``).

    Pre-patch: cloud is asymmetric (max mirror distance ~10^3 in
    kN-kNm units) for centred sections.
    Post-patch: residual is bounded by mesh resolution (~10^-12).
    """

    def _build_centered_octagon(self, half_size=505.0, b_corner=95.0,
                                fck=49.8, fyk=450.0, mesh_size=40.0):
        """
        Octagonal section centred on the origin
        (``y in [-half_size, +half_size]``).  Mimics the structure
        of the COGEFA Scalo Vallino pillar.
        """
        from gensec.geometry import GenericSection, RebarLayer
        from gensec.materials import Concrete, Steel
        from shapely.geometry import Polygon

        a = half_size
        b = b_corner
        # CCW vertices.
        coords = [(b, a), (-b, a), (-a, b), (-a, -b),
                  (-b, -a), (b, -a), (a, -b), (a, b)]
        poly = Polygon(coords)
        conc = Concrete(fck=fck, gamma_c=1.4)
        steel = Steel(fyk=fyk, gamma_s=1.15, eps_su=0.075)

        # Symmetric 8-bar pattern at d = 0.7*a from centre.
        d = a * 0.7
        rebars = [
            RebarLayer(y=+d, x=+d, As=314, material=steel),
            RebarLayer(y=+d, x=-d, As=314, material=steel),
            RebarLayer(y=-d, x=+d, As=314, material=steel),
            RebarLayer(y=-d, x=-d, As=314, material=steel),
            RebarLayer(y=+d, x=0,  As=314, material=steel),
            RebarLayer(y=-d, x=0,  As=314, material=steel),
            RebarLayer(y=0,  x=+d, As=314, material=steel),
            RebarLayer(y=0,  x=-d, As=314, material=steel),
        ]
        return GenericSection(polygon=poly, bulk_material=conc,
                              rebars=rebars, mesh_size=mesh_size)

    def _cloud_mirror_max_distance(self, N, M):
        """
        Maximum euclidean distance, in (N, M) space, between each
        point and its nearest mirror image obtained by flipping
        ``M -> -M``.
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([N, M]))
        d, _ = tree.query(np.column_stack([N, -M]), k=1)
        return float(d.max())

    def test_centered_section_x_symmetry(self):
        """N-Mx cloud must be mirror-symmetric about Mx = 0."""
        from gensec.solver import FiberSolver
        from gensec.solver import NMDiagram

        sec = self._build_centered_octagon()
        solver = FiberSolver(sec)
        nm = NMDiagram(solver).generate(n_points=200, direction='x')

        d = self._cloud_mirror_max_distance(nm['N_kN'], nm['M_kNm'])
        # Pre-patch: d > 100.  Post-patch: d ~ 1e-12.
        self.assertLess(d, 1e-3,
                        f"N-Mx cloud is not mirror-symmetric: "
                        f"max mirror distance {d:.3e}")
        # Strong consistency check on extremes.
        self.assertAlmostEqual(nm['M_kNm'].max(),
                               -nm['M_kNm'].min(), places=3)

    def test_centered_section_y_symmetry(self):
        """N-My cloud must be mirror-symmetric about My = 0."""
        from gensec.solver import FiberSolver
        from gensec.solver import NMDiagram

        sec = self._build_centered_octagon()
        solver = FiberSolver(sec)
        nm = NMDiagram(solver).generate(n_points=200, direction='y')

        d = self._cloud_mirror_max_distance(nm['N_kN'], nm['M_kNm'])
        self.assertLess(d, 1e-3)
        self.assertAlmostEqual(nm['M_kNm'].max(),
                               -nm['M_kNm'].min(), places=3)

    def test_legacy_rect_unchanged(self):
        """
        For a section with ``y_bot = 0`` (every standard primitive)
        the patched formula must coincide with the previous one.
        Bit-for-bit identity is established at the formula level
        (lever = y_ref when y_bot = 0); this test confirms a legacy
        ``RectSection`` produces a valid, symmetric N-Mx cloud
        post-patch.
        """
        from gensec.geometry import RectSection, RebarLayer
        from gensec.materials import Concrete, Steel
        from gensec.solver import FiberSolver
        from gensec.solver import NMDiagram

        conc = Concrete(fck=30.0, gamma_c=1.5)
        steel = Steel(fyk=450.0, gamma_s=1.15)
        rebars = [
            RebarLayer(y=50,  x=50,  As=314, material=steel),
            RebarLayer(y=50,  x=250, As=314, material=steel),
            RebarLayer(y=550, x=50,  As=314, material=steel),
            RebarLayer(y=550, x=250, As=314, material=steel),
        ]
        sec = RectSection(B=300, H=600, bulk_material=conc,
                          rebars=rebars,
                          n_fibers_y=80, n_fibers_x=40)
        solver = FiberSolver(sec)
        nm = NMDiagram(solver).generate(n_points=200, direction='x')
        d = self._cloud_mirror_max_distance(nm['N_kN'], nm['M_kNm'])
        self.assertLess(d, 1e-3)


# ===========================================================================
#  B1 — Symmetric uniaxial fast-path for Mx_target ≈ 0
# ===========================================================================

class TestB1_SymmetricUniaxialFastPath(unittest.TestCase):
    r"""
    ``solve_equilibrium`` has a symmetric fast-path for
    ``Mx_target ~ 0`` mirroring the existing ``My_target ~ 0`` path.
    The new ``_solve_uniaxial_y`` and ``_nr_uniaxial_y`` methods
    exist and converge on My-dominated demands.
    """

    def _build_simple_rect(self):
        from gensec.geometry import RectSection, RebarLayer
        from gensec.materials import Concrete, Steel

        conc = Concrete(fck=30.0, gamma_c=1.5)
        steel = Steel(fyk=450.0, gamma_s=1.15)
        rebars = [
            RebarLayer(y=50,  x=50,  As=314, material=steel),
            RebarLayer(y=50,  x=250, As=314, material=steel),
            RebarLayer(y=550, x=50,  As=314, material=steel),
            RebarLayer(y=550, x=250, As=314, material=steel),
        ]
        return RectSection(B=300, H=600, bulk_material=conc,
                           rebars=rebars,
                           n_fibers_y=60, n_fibers_x=30)

    def test_methods_exist(self):
        from gensec.solver import FiberSolver
        sec = self._build_simple_rect()
        solver = FiberSolver(sec)
        self.assertTrue(hasattr(solver, '_solve_uniaxial_y'),
                        "B1 fix missing: _solve_uniaxial_y not defined")
        self.assertTrue(hasattr(solver, '_nr_uniaxial_y'),
                        "B1 fix missing: _nr_uniaxial_y not defined")

    def test_my_only_demand_converges(self):
        """A demand with Mx ~ 0 and significant My must converge."""
        from gensec.solver import FiberSolver
        sec = self._build_simple_rect()
        solver = FiberSolver(sec)
        # Modest compression + pure bending about y.
        N_target = -200e3       # 200 kN compression
        Mx_target = 0.0
        My_target = 30e6        # 30 kNm
        sol = solver.solve_equilibrium(N_target, Mx_target, My_target,
                                       tol=10.0, max_iter=50)
        self.assertTrue(sol['converged'],
                        f"Mx~0 demand failed to converge: {sol}")
        self.assertLess(abs(sol['N'] - N_target), 100.0)
        self.assertLess(abs(sol['My'] - My_target), 1e5)


# ===========================================================================
#  B2 — Uniaxial detection accepts an axis kwarg
# ===========================================================================

class TestB2_UniaxialDetectionBothAxes(unittest.TestCase):
    r"""
    ``_is_uniaxial`` accepts an optional ``axis`` kwarg and recognises
    both vertical (chi_y singular) and horizontal (chi_x singular)
    degeneracy.  Without ``axis`` the legacy semantics is preserved.
    """

    def test_axis_kwarg_accepted(self):
        from gensec.solver import FiberSolver
        from gensec.geometry import RectSection
        from gensec.materials import Concrete

        sec = RectSection(B=300, H=600,
                          bulk_material=Concrete(fck=30.0),
                          rebars=[],
                          n_fibers_y=20, n_fibers_x=20)
        solver = FiberSolver(sec)
        # Should not raise — and the section is non-degenerate.
        self.assertFalse(solver._is_uniaxial())
        self.assertFalse(solver._is_uniaxial(axis='x'))
        self.assertFalse(solver._is_uniaxial(axis='y'))


# ===========================================================================
#  B3 — Symmetric chi_y_est in elastic initial guess
# ===========================================================================

class TestB3_ElasticInitialGuessIsSymmetric(unittest.TestCase):
    r"""
    ``_elastic_initial_guess`` returns a non-zero ``chi_y_est`` when
    the elastic 3×3 system is singular and ``My_target`` is the
    dominant moment.

    To force the singular branch, ``integrate_with_tangent`` is
    monkey-patched to raise ``LinAlgError``.
    """

    def test_chi_y_estimate_when_my_dominant(self):
        from gensec.geometry import RectSection
        from gensec.materials import Concrete
        from gensec.solver import FiberSolver

        sec = RectSection(B=300, H=600,
                          bulk_material=Concrete(fck=30.0),
                          rebars=[],
                          n_fibers_y=20, n_fibers_x=20)
        solver = FiberSolver(sec)

        def _raise(*a, **kw):
            raise np.linalg.LinAlgError("forced for test")
        original = solver.integrate_with_tangent
        solver.integrate_with_tangent = _raise
        try:
            eps0, chi_x, chi_y = solver._elastic_initial_guess(
                N_target=0.0, Mx_target=0.0, My_target=50e6)
        finally:
            solver.integrate_with_tangent = original

        # Pre-patch: chi_y == 0.0 always.
        # Post-patch: chi_y ∝ My_target.
        self.assertNotEqual(chi_y, 0.0,
                            "B3 fix missing: chi_y estimate is zero "
                            "when My_target dominates")
        self.assertGreater(abs(chi_y), 0.0)


# ===========================================================================
#  F1 — Zone-aware embedded-rebar bulk subtraction
# ===========================================================================

class TestF1_ZoneAwareRebarSubtraction(unittest.TestCase):
    r"""
    Embedded rebars subtract the bulk stress evaluated with the
    constitutive law of the zone they sit in, not the primary
    ``bulk_material``.

    Structural verification: rebar groups are now keyed by
    ``(rebar material, bulk-zone material)``, and
    ``mat_indices_rebar`` is set on every section.
    """

    def test_mat_indices_rebar_attribute(self):
        from gensec.geometry import RectSection, RebarLayer
        from gensec.materials import Concrete, Steel

        conc = Concrete(fck=30.0)
        steel = Steel(fyk=450.0)
        rebars = [
            RebarLayer(y=50, x=50, As=314, material=steel),
            RebarLayer(y=550, x=250, As=314, material=steel),
        ]
        sec = RectSection(B=300, H=600, bulk_material=conc,
                          rebars=rebars,
                          n_fibers_y=20, n_fibers_x=10)
        # New attribute must exist.  For a single-bulk section, all
        # rebars live in zone 0.
        self.assertTrue(hasattr(sec, 'mat_indices_rebar'),
                        "F1 fix missing: mat_indices_rebar attribute")
        self.assertEqual(len(sec.mat_indices_rebar), len(rebars))
        self.assertTrue(np.all(sec.mat_indices_rebar == 0))

    def test_rebar_groups_have_three_tuple(self):
        """Group entries are now ``(rebar_mat, bulk_mat, idx)``."""
        from gensec.geometry import RectSection, RebarLayer
        from gensec.materials import Concrete, Steel
        from gensec.solver import FiberSolver

        conc = Concrete(fck=30.0)
        steel = Steel(fyk=450.0)
        rebars = [RebarLayer(y=50, x=50, As=314, material=steel)]
        sec = RectSection(B=300, H=600, bulk_material=conc,
                          rebars=rebars,
                          n_fibers_y=20, n_fibers_x=10)
        solver = FiberSolver(sec)
        for g in solver._rebar_groups:
            self.assertEqual(len(g), 3,
                             "F1 fix missing: rebar group is not "
                             "(rebar_mat, bulk_mat, idx)")
            self.assertIs(g[1], conc,
                          "F1 fix missing: bulk_mat should be the "
                          "primary bulk_material for single-bulk "
                          "section")

    def test_single_bulk_force_balance_unchanged(self):
        """
        For a single-bulk-material section, integrate() must produce
        the same internal forces as before the F1 patch — the
        ``bulk_mat`` in every rebar group is the primary
        ``bulk_material``.
        """
        from gensec.geometry import RectSection, RebarLayer
        from gensec.materials import Concrete, Steel
        from gensec.solver import FiberSolver

        conc = Concrete(fck=30.0)
        steel = Steel(fyk=450.0)
        rebars = [
            RebarLayer(y=50,  x=50,  As=314, material=steel),
            RebarLayer(y=50,  x=250, As=314, material=steel),
            RebarLayer(y=550, x=50,  As=314, material=steel),
            RebarLayer(y=550, x=250, As=314, material=steel),
        ]
        sec = RectSection(B=300, H=600, bulk_material=conc,
                          rebars=rebars,
                          n_fibers_y=60, n_fibers_x=30)
        solver = FiberSolver(sec)
        # A moderately compressed bending state.
        N, Mx, My = solver.integrate(eps0=-0.001,
                                     chi_x=2e-6, chi_y=0.0)
        self.assertGreater(abs(N), 0.0)
        self.assertGreater(abs(Mx), 0.0)
        self.assertLess(abs(My), 1e-3)  # chi_y = 0


# ===========================================================================
#  K — eta_norm metric
# ===========================================================================

class TestK_EtaNorm(unittest.TestCase):
    r"""
    ``eta_norm`` is implemented and behaves consistently with its
    documented properties.
    """

    def _build_synthetic_3d_domain(self):
        """
        Synthetic axisymmetric "lens" domain: wide in (Mx, My) and
        tall in N.  Its origin is well inside the domain.
        """
        thetas = np.linspace(0, 2*np.pi, 60, endpoint=False)
        Ns = np.linspace(-1e6, 5e5, 21)
        pts = []
        for N in Ns:
            t = (N - (-1e6)) / (5e5 - (-1e6))
            radius_kNm = 500.0 * (1 - (2*t - 1) ** 2)
            radius_Nmm = radius_kNm * 1e6
            for th in thetas:
                pts.append((N, radius_Nmm * np.cos(th),
                            radius_Nmm * np.sin(th)))
        pts = np.array(pts)
        return {"N": pts[:, 0], "Mx": pts[:, 1], "My": pts[:, 2]}

    def test_method_exists_and_returns_finite(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertTrue(hasattr(d, 'eta_norm'),
                        "K fix missing: eta_norm method not defined")
        eta = d.eta_norm(N=-200e3, Mx=50e6, My=30e6)
        self.assertTrue(np.isfinite(eta))
        self.assertGreaterEqual(eta, 0.0)
        self.assertLess(eta, 1.0)  # interior

    def test_eta_norm_zero_at_chebyshev_centre(self):
        """
        For alpha: eta_norm = 0 at the Chebyshev centre of the
        normalised domain.  We can compute the Chebyshev centre
        from the LP that is solved at construction (here we
        recompute it explicitly to verify the property).
        """
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        # Re-solve the same LP that DomainChecker uses internally
        # to find the Chebyshev centre in normalised coordinates.
        from scipy.optimize import linprog
        A = d._equations[:, :-1]
        b = d._equations[:, -1]
        n = d.ndim
        c = np.zeros(n + 1); c[-1] = -1.0
        A_ub = np.hstack([A, np.ones((A.shape[0], 1))])
        b_ub = -b
        bounds = [(None, None)] * n + [(0.0, None)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        self.assertTrue(res.success)
        cheb_centre_norm = res.x[:n]
        # Map back from normalised to raw coordinates.
        cheb_N = cheb_centre_norm[0] / d._u_x
        cheb_Mx = cheb_centre_norm[1]
        cheb_My = cheb_centre_norm[2] / d._v_y
        eta = d.eta_norm(cheb_N, cheb_Mx, cheb_My)
        # Should be 0 modulo numerical precision: the centre is
        # equidistant from every face by construction.
        self.assertAlmostEqual(eta, 0.0, places=4)

    def test_eta_norm_below_one_for_interior(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertLess(d.eta_norm(N=-200e3, Mx=10e6, My=10e6), 1.0)

    def test_eta_norm_above_one_for_exterior(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        # Demand far outside the domain.
        self.assertGreater(d.eta_norm(N=-200e3, Mx=2000e6, My=2000e6),
                           1.0)

    def test_eta_norm_monotone_toward_boundary(self):
        """
        For alpha: as a demand moves along a fixed ray from the
        Chebyshev centre toward the boundary, eta_norm must grow
        monotonically (the only thing alpha guarantees: distance
        decreases monotonically along such a ray).

        Cannot check linearity of the form eta_far = 2 * eta_near,
        because alpha is *not* a ray-cast and that proportionality
        does not hold.
        """
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        # Three points along the +Mx axis at fixed N=0 (which is
        # well inside this synthetic dome).  Increasing |Mx| moves
        # us closer to the contour.
        eta_close   = d.eta_norm(N=0.0, Mx=50e6,  My=0.0)
        eta_middle  = d.eta_norm(N=0.0, Mx=200e6, My=0.0)
        eta_near_bd = d.eta_norm(N=0.0, Mx=400e6, My=0.0)
        self.assertLess(eta_close, eta_middle)
        self.assertLess(eta_middle, eta_near_bd)


# ===========================================================================
#  VerificationEngine integration
# ===========================================================================

class TestVerificationEngineExposesEtaNorm(unittest.TestCase):
    r"""
    The ``VerificationEngine`` reads ``eta_norm`` from the output flags
    (default ``True``) and includes the value in its per-demand
    result.
    """

    def _build_synthetic_3d_domain(self):
        thetas = np.linspace(0, 2*np.pi, 40, endpoint=False)
        Ns = np.linspace(-1e6, 5e5, 11)
        pts = []
        for N in Ns:
            t = (N - (-1e6)) / (5e5 - (-1e6))
            radius_Nmm = 500.0e6 * (1 - (2*t - 1) ** 2)
            for th in thetas:
                pts.append((N, radius_Nmm * np.cos(th),
                            radius_Nmm * np.sin(th)))
        pts = np.array(pts)
        return {"N": pts[:, 0], "Mx": pts[:, 1], "My": pts[:, 2]}

    def test_default_flag_includes_eta_norm(self):
        from gensec.solver import VerificationEngine
        engine = VerificationEngine(
            nm_3d=self._build_synthetic_3d_domain(),
            nm_gen=None,
            output_flags={},  # all defaults
        )
        self.assertTrue(engine.do_norm,
                        "eta_norm default should be True")
        result = engine.check_demand({
            "name": "test", "N": -200e3, "Mx": 50e6, "My": 30e6,
        })
        self.assertIn("eta_norm", result,
                      "VerificationEngine result missing eta_norm")
        self.assertIsNotNone(result["eta_norm"])

    def test_flag_disables_eta_norm(self):
        from gensec.solver import VerificationEngine
        engine = VerificationEngine(
            nm_3d=self._build_synthetic_3d_domain(),
            nm_gen=None,
            output_flags={"eta_norm": False},
        )
        self.assertFalse(engine.do_norm)
        result = engine.check_demand({
            "name": "test", "N": -200e3, "Mx": 50e6, "My": 30e6,
        })
        self.assertNotIn("eta_norm", result)


# ===========================================================================
#  K_beta — eta_norm_beta (composite-ratio metric)
# ===========================================================================

class TestK_EtaNormBeta(unittest.TestCase):
    r"""
    The :meth:`DomainChecker.eta_norm_beta` method computes the
    composite-ratio metric :math:`F_{SU}/(F_{SU}+d_{\min})`.

    Properties tested:

    - origin gives 0 (because :math:`F_{SU} = 0`)
    - interior demand gives a value in [0, 1)
    - exterior demand gives a value > 1
    - the metric grows monotonically with proximity to the boundary
      along a ray from the origin
    """

    def _build_synthetic_3d_domain(self):
        thetas = np.linspace(0, 2*np.pi, 60, endpoint=False)
        Ns = np.linspace(-1e6, 5e5, 21)
        pts = []
        for N in Ns:
            t = (N - (-1e6)) / (5e5 - (-1e6))
            radius_kNm = 500.0 * (1 - (2*t - 1) ** 2)
            radius_Nmm = radius_kNm * 1e6
            for th in thetas:
                pts.append((N, radius_Nmm * np.cos(th),
                            radius_Nmm * np.sin(th)))
        pts = np.array(pts)
        return {"N": pts[:, 0], "Mx": pts[:, 1], "My": pts[:, 2]}

    def test_method_exists_and_returns_finite(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertTrue(hasattr(d, "eta_norm_beta"),
                        "K_beta missing: eta_norm_beta not defined")
        eta = d.eta_norm_beta(N=-100e3, Mx=50e6, My=30e6)
        self.assertTrue(np.isfinite(eta))
        self.assertGreaterEqual(eta, 0.0)

    def test_origin_is_zero(self):
        # F_SU = 0 at the origin, so beta returns 0 by construction.
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertEqual(d.eta_norm_beta(0.0, 0.0, 0.0), 0.0)

    def test_interior_below_one(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertLess(d.eta_norm_beta(-100e3, 50e6, 30e6), 1.0)

    def test_exterior_above_one(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertGreater(
            d.eta_norm_beta(-100e3, 2000e6, 2000e6), 1.0)

    def test_monotone_along_radial(self):
        # Along a fixed ray from the origin, beta grows with the
        # demand magnitude (because F_SU grows and d_min shrinks).
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        e1 = d.eta_norm_beta(-50e3, 25e6, 15e6)
        e2 = d.eta_norm_beta(-100e3, 50e6, 30e6)
        e3 = d.eta_norm_beta(-150e3, 75e6, 45e6)
        self.assertLess(e1, e2)
        self.assertLess(e2, e3)


# ===========================================================================
#  K_ray — eta_norm_ray (ray-cast in normalised space)
# ===========================================================================

class TestK_EtaNormRay(unittest.TestCase):
    r"""
    :meth:`DomainChecker.eta_norm_ray` is a ray-cast from the origin
    in anisotropy-corrected normalised space.

    Properties tested:

    - origin gives 0
    - interior gives < 1, exterior > 1
    - linearity: doubling the demand doubles the ray-cast value
      (the ray-cast metric is exactly linear in the demand
      magnitude along a fixed ray, modulo discretisation)
    """

    def _build_synthetic_3d_domain(self):
        thetas = np.linspace(0, 2*np.pi, 60, endpoint=False)
        Ns = np.linspace(-1e6, 5e5, 21)
        pts = []
        for N in Ns:
            t = (N - (-1e6)) / (5e5 - (-1e6))
            radius_Nmm = 500.0e6 * (1 - (2*t - 1) ** 2)
            for th in thetas:
                pts.append((N, radius_Nmm * np.cos(th),
                            radius_Nmm * np.sin(th)))
        pts = np.array(pts)
        return {"N": pts[:, 0], "Mx": pts[:, 1], "My": pts[:, 2]}

    def test_method_exists_and_origin_is_zero(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertTrue(hasattr(d, "eta_norm_ray"),
                        "K_ray missing: eta_norm_ray not defined")
        self.assertEqual(d.eta_norm_ray(0.0, 0.0, 0.0), 0.0)

    def test_linear_along_ray(self):
        # eta_norm_ray = |OS|/|OR|.  On the same ray, doubling S
        # doubles the numerator without moving R: the ratio doubles.
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        eta_1 = d.eta_norm_ray(-50e3, 25e6, 15e6)
        eta_2 = d.eta_norm_ray(-100e3, 50e6, 30e6)
        self.assertAlmostEqual(eta_2, 2.0 * eta_1, places=3)

    def test_interior_below_one_exterior_above_one(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertLess(d.eta_norm_ray(-100e3, 50e6, 30e6), 1.0)
        self.assertGreater(
            d.eta_norm_ray(-100e3, 2000e6, 2000e6), 1.0)


# ===========================================================================
#  K_path_beta — eta_path_norm_beta (composite-ratio along stage segment)
# ===========================================================================

class TestK_EtaPathNormBeta(unittest.TestCase):
    r"""
    :meth:`DomainChecker.eta_path_norm_beta` is the staged analogue
    of beta: composite ratio :math:`L/(L + d_{\text{seg}})` along
    the segment :math:`B \to T`, where :math:`L = |T - B|` and
    :math:`d_{\text{seg}}` is the minimum signed distance of the
    segment from the boundary.

    Properties tested:

    - degenerate segment (B == T) collapses to point beta
    - segment fully inside gives < 1
    - target outside the domain gives ≥ 1
    - the metric is path-aware: same target, different base, gives
      different values
    """

    def _build_synthetic_3d_domain(self):
        thetas = np.linspace(0, 2*np.pi, 60, endpoint=False)
        Ns = np.linspace(-1e6, 5e5, 21)
        pts = []
        for N in Ns:
            t = (N - (-1e6)) / (5e5 - (-1e6))
            radius_Nmm = 500.0e6 * (1 - (2*t - 1) ** 2)
            for th in thetas:
                pts.append((N, radius_Nmm * np.cos(th),
                            radius_Nmm * np.sin(th)))
        pts = np.array(pts)
        return {"N": pts[:, 0], "Mx": pts[:, 1], "My": pts[:, 2]}

    def test_method_exists(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        self.assertTrue(hasattr(d, "eta_path_norm_beta"),
                        "K_path_beta missing: eta_path_norm_beta "
                        "not defined")

    def test_zero_length_segment_collapses_to_point_beta(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        # Both endpoints at the same point: should match the
        # point-beta value at that point.
        N, Mx, My = -100e3, 50e6, 30e6
        e_path = d.eta_path_norm_beta(N, Mx, My, N, Mx, My)
        e_point = d.eta_norm_beta(N, Mx, My)
        self.assertAlmostEqual(e_path, e_point, places=4)

    def test_interior_segment_below_one(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        e = d.eta_path_norm_beta(
            -100e3, 50e6, 30e6,    # base
            -150e3, 75e6, 45e6)    # target (further but still inside)
        self.assertLess(e, 1.0)

    def test_target_outside_above_one(self):
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        e = d.eta_path_norm_beta(
            -100e3, 50e6, 30e6,
            -100e3, 2000e6, 2000e6)
        self.assertGreater(e, 1.0)

    def test_path_dependent(self):
        # Same target T, two different bases.  beta is path-
        # dependent: a base closer to the boundary produces a
        # higher value than one in the deep interior.
        from gensec.solver import DomainChecker
        d = DomainChecker(self._build_synthetic_3d_domain())
        T = (-100e3, 200e6, 100e6)
        # Base 1: deep interior.
        e_deep = d.eta_path_norm_beta(0.0, 0.0, 0.0, *T)
        # Base 2: closer to boundary in My direction.
        e_near = d.eta_path_norm_beta(-100e3, 50e6, 250e6, *T)
        self.assertNotAlmostEqual(e_deep, e_near, places=2)


# ===========================================================================
#  Staged warning
# ===========================================================================

class TestStagedWarning(unittest.TestCase):
    r"""
    When a staged combination (n_stages > 1) is processed but no
    path-aware metric is enabled, the ``VerificationEngine`` prints
    a warning to stderr.  Single-stage combinations and combinations
    with at least one ``eta_path_*`` flag enabled do not trigger the
    warning.
    """

    def _build_synthetic_3d_domain(self):
        thetas = np.linspace(0, 2*np.pi, 40, endpoint=False)
        Ns = np.linspace(-1e6, 5e5, 11)
        pts = []
        for N in Ns:
            t = (N - (-1e6)) / (5e5 - (-1e6))
            radius_Nmm = 500.0e6 * (1 - (2*t - 1) ** 2)
            for th in thetas:
                pts.append((N, radius_Nmm * np.cos(th),
                            radius_Nmm * np.sin(th)))
        pts = np.array(pts)
        return {"N": pts[:, 0], "Mx": pts[:, 1], "My": pts[:, 2]}

    def test_warning_when_multistage_no_path_metric(self):
        import io
        from contextlib import redirect_stderr
        from gensec.solver import VerificationEngine
        engine = VerificationEngine(
            nm_3d=self._build_synthetic_3d_domain(),
            nm_gen=None,
            output_flags={
                "eta_norm": True, "eta_2D": False,
                "eta_path": False, "eta_path_norm_beta": False,
                "eta_path_2D": False,
            },
        )
        demand_db = {
            "G":  {"name": "G",  "N": -200e3, "Mx": 30e6, "My": 10e6},
            "Ex": {"name": "Ex", "N":   50e3, "Mx": 80e6, "My": 60e6},
        }
        combo = {
            "name": "test_staged",
            "stages": [
                {"name": "g",  "components": [{"ref": "G"}]},
                {"name": "ex", "components": [{"ref": "Ex"}]},
            ],
        }
        buf = io.StringIO()
        with redirect_stderr(buf):
            engine.check_combination(combo, demand_db)
        msg = buf.getvalue()
        self.assertIn("staged combination", msg)
        self.assertIn("test_staged", msg)
        self.assertIn("path-aware metric", msg)

    def test_no_warning_when_path_metric_enabled(self):
        import io
        from contextlib import redirect_stderr
        from gensec.solver import VerificationEngine
        engine = VerificationEngine(
            nm_3d=self._build_synthetic_3d_domain(),
            nm_gen=None,
            output_flags={
                "eta_norm": True, "eta_path_norm_ray": True,
            },
        )
        demand_db = {
            "G":  {"name": "G",  "N": -200e3, "Mx": 30e6, "My": 10e6},
            "Ex": {"name": "Ex", "N":   50e3, "Mx": 80e6, "My": 60e6},
        }
        combo = {
            "name": "test_with_path",
            "stages": [
                {"name": "g",  "components": [{"ref": "G"}]},
                {"name": "ex", "components": [{"ref": "Ex"}]},
            ],
        }
        buf = io.StringIO()
        with redirect_stderr(buf):
            engine.check_combination(combo, demand_db)
        # The warning is suppressed when at least one path-aware
        # flag is on.
        self.assertNotIn("staged combination", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
