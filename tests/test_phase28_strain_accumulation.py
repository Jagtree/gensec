# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
# This file is part of GenSec.  AGPL-3.0-or-later.
# ---------------------------------------------------------------------------
r"""
Phase-28 fixtures for defect **D27-1**, the affine accumulation.

What the defect is
------------------
:func:`gensec.solver.sls.verify_sls_staged` did not merely *use* a linear
view — it was written on linearity.  Two bookkeepings ran in parallel:

* a per-fibre **stress** accumulator, summed as
  :math:`\Delta\sigma = E_z\,\Delta\varepsilon` with one scalar modulus
  per zone;
* a per-zone **affine stress field**
  :math:`\sigma_z(x,y) = c_{0,z} + c_{x,z}x + c_{y,z}y`, accumulated the
  same way, from which the reported extreme-fibre stresses and the
  decompression probe were read.

Both are superposition, and superposition of *stresses* is exact only
when the law is a straight line.  Phase 27 removed the straight line
(``27_0`` §4): the machinery did not stop running, it stopped being
correct, and no fixture reached it, so its green said nothing.

What is repaired
----------------
The kinematics of a plane section are linear whatever the material is,
so **strains** superpose exactly.  The engine therefore accumulates one
per-zone affine *strain* field and reads :math:`\sigma` from the zone's
own constitutive law wherever a stress is wanted.  The per-fibre array
and the per-zone field become the same object, which is why the
cross-check between them (``debug_check_affine``) is removed rather than
adapted: it existed to compare two accumulators, and there is now one.

What is deliberately **not** repaired here
------------------------------------------
``concrete_compression`` verifies the stress at the **centroid of the
outermost fibre** while the record reports the stress at the section
**face** and labels the check with the face's coordinates.  That is
hotfix F6 applied to the report and not to the check, it is visible on a
straight line, and it is therefore a defect independent of D27-1.
Andrea's decision for this phase is that it stays open
(``28_0`` §2), so the guards below **pin** the disagreement instead of
removing it.  A fixture that silently repaired it would move the linear
discriminator and hide the fact that the two numbers still differ.

Red on the pristine tree
------------------------
Every value asserted below was measured on a tree rebuilt from
``gensec_phase27.bundle`` and is written into the file, so the red is
reproducible rather than remembered.  The five repair fixtures fail with
``AssertionError`` there — none with ``ImportError`` or
``AttributeError``, which is the red the closure gate excludes.
"""
import unittest

import numpy as np
from shapely.geometry import box

from gensec.geometry.fiber import RebarLayer
from gensec.geometry.geometry import GenericSection
from gensec.geometry.primitives import rect_poly
from gensec.geometry.section import RectSection
from gensec.materials.concrete import Concrete
from gensec.materials.ec2_properties import fben2
from gensec.materials.elastic import LinearElastic
from gensec.materials.service_concrete import ServiceConcrete
from gensec.materials.steel import Steel
from gensec.materials.verification_limits import StressLimits
from gensec.solver.section_state import SectionState
from gensec.solver.sls import verify_sls_staged

# ----------------------------------------------------------------------
#  The reference section: the one the Phase-28 numeric probe drives.
# ----------------------------------------------------------------------
B, H, NFY = 400.0, 1200.0, 240
AS, Y1, Y2 = 2000.0, 60.0, 1140.0
N_AX = -1000.0e3
AGE = 28.0
EPS_CU1 = 0.0035

#: Values measured on the tree rebuilt from ``gensec_phase27.bundle``.
#: They are the red these fixtures reproduce, written down rather than
#: recalled (``28_0`` §4).
PRISTINE = {
    "svc.report.sig_max": 37.007148,
    "svc.law.sig_at_max": 0.0,
    "svc.report.sig_min": -17.305557,
    "svc.law.sig_at_min": -16.4598428658,
    "svc.outcome": "AssertionError",
    "lin.report.sig_max": 6.531641,
    "lin.report.sig_min": -10.535836,
    "lin.check.value": 10.500279,
    "lin.check.eta": 0.500013,
}


def _ec2_at(fck, age):
    r"""
    The EN 1992-1-1 constants of one concrete at one age.

    Parameters
    ----------
    fck : float
        Characteristic cylinder strength [MPa].
    age : float
        Age [days].

    Returns
    -------
    fben2
        The property bundle, carrying ``fcm``, ``ecm``, ``eps_c1`` and
        ``fctm`` at *age*.
    """
    return fben2(fck=fck, ls="S", loadtype="slow", TypeConc="N",
                 time=age)


def _service_law(fck, age, name):
    r"""
    The EN 1992-1-1 (3.14) service law of one concrete at one age.

    Parameters
    ----------
    fck, age : float
    name : str

    Returns
    -------
    ServiceConcrete
    """
    p = _ec2_at(fck, age)
    law = ServiceConcrete(fcm=float(p.fcm), Ecm=float(p.ecm),
                          eps_c1=abs(float(p.eps_c1)),
                          eps_cu1=EPS_CU1, fct_eff=float(p.fctm),
                          reference_age=age, name=name)
    return law


def _reference_section():
    r"""
    The 400x1200 C35/45 reference section, two rebar layers.

    Returns
    -------
    tuple
        ``(section, x_ref, y_ref, limits, service_law, Ecm, fctm)``.
    """
    p = _ec2_at(35.0, AGE)
    law = _service_law(35.0, AGE, "C35svc")
    conc = Concrete(fck=35.0, gamma_c=1.5,
                    long_term_reduction_factor=0.85)
    conc.name = "C35svc"
    steel = Steel(fyk=450.0)
    steel.name = "B450C"
    xr, yr = B / 2.0, H / 2.0
    rebars = [RebarLayer(y=Y1, As=AS, material=steel, x=xr),
              RebarLayer(y=Y2, As=AS, material=steel, x=xr)]
    section = RectSection(B, H, conc, rebars, n_fibers_y=NFY)
    # Phase 30: fct_eff removed from StressLimits.  The law built
    # above already carries fct_eff=p.fctm, which is where the engine
    # reads it.
    limits = StressLimits(k_c=0.6,
                          sigma_s_tens=0.8 * 450.0)
    return (section, xr, yr, limits, law, float(p.ecm),
            float(p.fctm))


def _state(idx, label, n_union=2):
    r"""
    A fully active, fully bonded state with no initial strains.

    Parameters
    ----------
    idx : int
    label : str
    n_union : int, optional

    Returns
    -------
    SectionState
    """
    return SectionState(idx,
                        np.ones(n_union, dtype=bool),
                        np.ones(n_union, dtype=bool),
                        np.zeros(n_union, dtype=float),
                        bulk_eps_init=0.0, label=label)


def _histories(limits):
    r"""
    The single-stage and two-stage histories that reach the same demand.

    The pair matters: the single-stage run tests the *reading* of the
    stress field, the two-stage run tests its *accumulation*, and the
    two must agree because no element enters and no modulus changes
    between them.

    Parameters
    ----------
    limits : StressLimits

    Returns
    -------
    dict of list
    """
    return {
        "single": [{"name": "one", "state": _state(0, "one"),
                    "increment": (N_AX, 900.0e6, 0.0),
                    "limits": limits}],
        "staged": [{"name": "a", "state": _state(0, "a"),
                    "increment": (N_AX, 300.0e6, 0.0),
                    "limits": limits},
                   {"name": "b", "state": _state(1, "b"),
                    "increment": (0.0, 600.0e6, 0.0),
                    "limits": limits}],
    }


def _run_quiet(section, stages, moduli, xr, yr):
    r"""
    Run the verification without whatever internal cross-check the
    engine publishes, so that the *numbers* can be compared on both
    trees.

    The pristine engine aborts on its own consistency assertion when the
    declared law is not a straight line; a fixture that only observed
    the abort would test the assertion, not the stresses.  The keyword
    is looked up in the signature rather than assumed, so this helper
    keeps working across the rename it is written to survive.

    Parameters
    ----------
    section : GenericSection
    stages : list of dict
    moduli : dict
    xr, yr : float

    Returns
    -------
    dict
        The ``verify_sls_staged`` result.
    """
    import inspect
    params = inspect.signature(verify_sls_staged).parameters
    kw = {}
    for flag in ("debug_check_strain_field", "debug_check_affine"):
        if flag in params:
            kw[flag] = False
            break
    return verify_sls_staged(section, stages, moduli=moduli,
                             x_ref=xr, y_ref=yr, **kw)


def _eps_at(plane, x, y, xr, yr):
    r"""
    The fibre kernel's strain at one point of a solved plane.

    .. math::

        \varepsilon(x,y) = \varepsilon_0
            + \chi_x\,(y - y_{\mathrm{ref}})
            - \chi_y\,(x - x_{\mathrm{ref}})

    Parameters
    ----------
    plane : dict
        ``eps0``, ``chi_x``, ``chi_y``.
    x, y, xr, yr : float

    Returns
    -------
    float
    """
    return (float(plane["eps0"])
            + float(plane["chi_x"]) * (float(y) - yr)
            - float(plane["chi_y"]) * (float(x) - xr))


def _sigma(material, eps):
    r"""
    One stress from one material at one strain [MPa].

    Parameters
    ----------
    material : Material
    eps : float

    Returns
    -------
    float
    """
    return float(np.asarray(
        material.stress_array(np.array([eps], dtype=float)))[0])


# ======================================================================
#  Repair fixtures — AssertionError-red on gensec_phase27.bundle
# ======================================================================


class TestReportedStressIsAPointOfTheLaw(unittest.TestCase):
    r"""
    The extremes the engine reports must be stresses the declared
    concrete can be in.

    A reported stress that is not a point of the constitutive law is the
    defect stated without reference to how the bookkeeping is organised
    internally: it is checkable from the outside, from the solved plane
    and the material alone.
    """

    def setUp(self):
        (self.sec, self.xr, self.yr, self.limits,
         self.law, self.ecm, self.fctm) = _reference_section()

    def _gaps(self, material):
        r"""
        Report versus law, at both reported extremes, per history.

        The law returned by a cracking material is **discontinuous**: it
        reaches :math:`f_{ct,\mathrm{eff}}` at :math:`\varepsilon_{ct}`
        and returns zero immediately beyond.  The maximum of such a
        function over a region is attained *at* the jump, so the point
        the engine names sits exactly on it, and re-deriving that
        point's strain from the solved plane — a different arithmetic
        path from the one the engine used — can land a few ulps on the
        far side and read zero.  The law is therefore evaluated on both
        sides and the nearer value is the one compared, which tests the
        stress and not the last bit of a coordinate.

        Returns
        -------
        dict
            ``{(history, tag): (reported, law)}``.
        """
        out = {}
        for hist, stages in _histories(self.limits).items():
            res = _run_quiet(self.sec, stages, {"C35svc": material},
                             self.xr, self.yr)
            sr = res["stages"][-1]
            plane, conc = sr["plane"], sr["concrete"]
            for tag, skey, akey in (("min", "sigma_min_MPa", "at_min"),
                                    ("max", "sigma_max_MPa", "at_max")):
                x_a, y_a = conc[akey]
                eps = _eps_at(plane, x_a, y_a, self.xr, self.yr)
                rep = float(conc[skey])
                cand = [_sigma(material, eps),
                        _sigma(material, eps * (1.0 - 1.0e-12)),
                        _sigma(material, eps * (1.0 + 1.0e-12))]
                out[(hist, tag)] = (
                    rep, min(cand, key=lambda s: abs(s - rep)))
        return out

    def test_tensile_extreme_is_a_point_of_the_law(self):
        r"""
        Pristine: the engine reports **+37.007148 MPa** where the law
        returns **0.000000 MPa** — the point is past the cracking strain
        and carries nothing.
        """
        gaps = self._gaps(self.law)
        for hist in ("single", "staged"):
            rep, law_sigma = gaps[(hist, "max")]
            self.assertAlmostEqual(
                rep, law_sigma, places=5,
                msg=(f"{hist}: reported sigma_max {rep:.6f} MPa is not "
                     f"the law's stress {law_sigma:.6f} MPa at the "
                     f"point the report names (pristine: "
                     f"{PRISTINE['svc.report.sig_max']} vs "
                     f"{PRISTINE['svc.law.sig_at_max']})."))

    def test_compressed_extreme_is_a_point_of_the_law(self):
        r"""
        Pristine: **-17.305557 MPa** reported against **-16.459843 MPa**
        from the law — the affine field extrapolates the secant past the
        curvature of (3.14).
        """
        gaps = self._gaps(self.law)
        for hist in ("single", "staged"):
            rep, law_sigma = gaps[(hist, "min")]
            self.assertAlmostEqual(
                rep, law_sigma, places=5,
                msg=(f"{hist}: reported sigma_min {rep:.6f} MPa is not "
                     f"the law's stress {law_sigma:.6f} MPa at the "
                     f"point the report names (pristine: "
                     f"{PRISTINE['svc.report.sig_min']} vs "
                     f"{PRISTINE['svc.law.sig_at_min']})."))

    def test_no_tension_beyond_fct_eff_is_reported(self):
        r"""
        Nothing in the section may be reported carrying more tension
        than the declared :math:`f_{ct,\mathrm{eff}}`.

        Pristine: **+37.007148 MPa** on a concrete whose
        :math:`f_{ctm}` is **3.209962 MPa**, a factor 11.5.
        """
        for hist, stages in _histories(self.limits).items():
            res = _run_quiet(self.sec, stages, {"C35svc": self.law},
                             self.xr, self.yr)
            for sr in res["stages"]:
                s_max = float(sr["concrete"]["sigma_max_MPa"])
                self.assertLessEqual(
                    s_max, self.fctm + 1.0e-6,
                    msg=(f"{hist}/{sr['name']}: reported sigma_max "
                         f"{s_max:.6f} MPa exceeds the declared "
                         f"fct_eff {self.fctm:.6f} MPa."))


class TestTheEngineCompletesOnADeclaredServiceLaw(unittest.TestCase):
    r"""
    The function must run to the end on the law Phase 27 delivered.

    Pristine outcome: ``AssertionError`` from the affine cross-check on
    the **first** stage — the field and the fibre accumulator disagree by
    1.119e-01 MPa, because they are two bookkeepings of a quantity that
    does not superpose.  The exception is caught and turned into this
    fixture's own assertion so that the red is the fixture's, not the
    engine's.
    """

    def setUp(self):
        (self.sec, self.xr, self.yr, self.limits,
         self.law, self.ecm, self.fctm) = _reference_section()

    def test_default_call_does_not_raise(self):
        for hist, stages in _histories(self.limits).items():
            try:
                verify_sls_staged(self.sec, stages,
                                  moduli={"C35svc": self.law},
                                  x_ref=self.xr, y_ref=self.yr)
            except Exception as exc:        # noqa: BLE001 — failure is data
                raise AssertionError(
                    f"{hist}: verify_sls_staged raised "
                    f"{type(exc).__name__} on a declared ServiceConcrete "
                    f"at its own defaults ({exc}).  Pristine outcome: "
                    f"{PRISTINE['svc.outcome']}."
                ) from None


class TestStageModuliOverrideIsRefused(unittest.TestCase):
    r"""
    A per-stage modulus override must be refused by name.

    Phase 7 decision D3 offered ``stage["moduli"]`` as the ageing hook.
    Under strain accumulation it is demonstrably wrong: under sustained
    load with a stiffening modulus and no creep the strain does not
    change, so the accumulated strain does not change, and reading the
    new law at the old strain would report a stress that moved while
    nothing physical did.  Ageing belongs to the rheological provider,
    which expresses it as imposed strain and a law at the age.

    Nothing in the repository ever passed the key — measured, not
    assumed — so it is removed rather than deprecated, and the refusal
    names the lever.
    """

    def setUp(self):
        (self.sec, self.xr, self.yr, self.limits,
         self.law, self.ecm, self.fctm) = _reference_section()

    def test_per_stage_moduli_raises_value_error(self):
        stages = [{"name": "a", "state": _state(0, "a"),
                   "increment": (N_AX, 300.0e6, 0.0),
                   "moduli": {"C35svc": self.ecm}}]
        raised = None
        try:
            _run_quiet(self.sec, stages, {"C35svc": self.law},
                       self.xr, self.yr)
        except ValueError as exc:
            raised = exc
        self.assertIsNotNone(
            raised,
            msg=("a per-stage 'moduli' override was accepted; it must "
                 "be a ValueError naming service_law(t) as the lever "
                 "(pristine: accepted silently)."))
        self.assertIn("service_law", str(raised))


class TestCompositeEnteringZone(unittest.TestCase):
    r"""
    The same requirement on the one history where the accumulation is
    not a telescoping sum.

    With a zone entering mid-history the accumulated strain is genuinely
    **not** the current solved plane: the topping must not be handed the
    strain the precast carried before it was cast.  This is the case the
    increment structure exists for, and it survives the repair — what
    changes is that the increments are strains.
    """

    B2, H1, H2 = 400.0, 800.0, 200.0

    def setUp(self):
        self.H = self.H1 + self.H2
        self.xr, self.yr = self.B2 / 2.0, self.H / 2.0
        self.pre = _service_law(45.0, AGE, "precast")
        self.top = _service_law(30.0, AGE, "topping")
        self.e_pre = float(_ec2_at(45.0, AGE).ecm)
        self.e_top = float(_ec2_at(30.0, AGE).ecm)
        steel = Steel(fyk=450.0)
        steel.name = "B450C"
        rebars = [RebarLayer(y=60.0, As=2400.0, material=steel,
                             x=self.xr),
                  RebarLayer(y=740.0, As=800.0, material=steel,
                             x=self.xr)]
        self.sec = GenericSection(
            polygon=rect_poly(self.B2, self.H),
            bulk_material=LinearElastic(E=self.e_pre, name="precast"),
            rebars=rebars,
            bulk_materials=[(box(0.0, self.H1, self.B2, self.H),
                             LinearElastic(E=self.e_top,
                                           name="topping"),
                             "topping")],
            mesh_size=50.0, n_grid_x=8, n_grid_y=20)

    def _state_c(self, idx, active, planes, label):
        return SectionState(
            idx, np.ones(2, dtype=bool), np.ones(2, dtype=bool),
            np.zeros(2, dtype=float),
            bulk_active=np.asarray(active, dtype=bool),
            bulk_planes=np.asarray(planes, dtype=float), label=label)

    def _run(self, moduli):
        m1, m2 = -2.0e8, -3.0e8
        first = [{"name": "a",
                  "state": self._state_c(0, [True, False],
                                         [[0, 0, 0], [0, 0, 0]], "a"),
                  "increment": (0.0, m1, 0.0)}]
        r0 = _run_quiet(self.sec, first, moduli, self.xr, self.yr)
        pl = r0["stages"][0]["plane"]
        datum = [[0.0, 0.0, 0.0],
                 [-pl["eps0"], -pl["chi_x"], -pl["chi_y"]]]
        both = [first[0],
                {"name": "b",
                 "state": self._state_c(1, [True, True], datum, "b"),
                 "increment": (0.0, m2, 0.0)}]
        return _run_quiet(self.sec, both, moduli, self.xr, self.yr)

    def test_each_zone_reports_stresses_of_its_own_law(self):
        r"""
        Pristine: the precast zone is reported at **+50.635741 MPa** of
        tension, on a C45/55 whose :math:`f_{ctm}` is 3.799770 MPa.
        """
        mats = {"precast": self.pre, "topping": self.top}
        res = self._run(mats)
        sr = res["stages"][-1]
        for zone_name, rec in sr["concrete"]["by_zone"].items():
            law = self.pre if zone_name != "topping" else self.top
            for skey, akey in (("sigma_min_MPa", "at_min"),
                               ("sigma_max_MPa", "at_max")):
                x_a, y_a = rec[akey]
                dat = (0.0 if zone_name != "topping" else None)
                if dat is None:
                    # The topping's locked-in datum is carried by the
                    # accumulator; compare against the cap instead of
                    # re-deriving it here.
                    self.assertLessEqual(
                        float(rec["sigma_max_MPa"]),
                        law.fct_eff + 1.0e-6,
                        msg=(f"{zone_name}: reported "
                             f"{rec['sigma_max_MPa']:.6f} MPa exceeds "
                             f"its own fct_eff {law.fct_eff:.6f} MPa."))
                    break
                eps = _eps_at(sr["plane"], x_a, y_a, self.xr, self.yr)
                self.assertAlmostEqual(
                    float(rec[skey]), _sigma(law, eps), places=5,
                    msg=(f"{zone_name}/{skey}: {rec[skey]} MPa is not "
                         f"a point of that zone's law (pristine "
                         f"precast sigma_max: +50.635741 MPa)."))


# ======================================================================
#  Declared guards — green on BOTH trees, by design.
#  They are no evidence of repair (closure gate 4).
# ======================================================================


class TestLinearDeclarationIsTheDiscriminator(unittest.TestCase):
    r"""
    **Declared guard.  Green on both trees on purpose; no evidence of
    repair.**

    A declaration that hands in a bare modulus is a declaration of a
    straight line (``27_0`` §4), and on a straight line the old stress
    superposition and the new strain superposition are the same
    arithmetic.  These values must therefore be identical before and
    after: if they move, the rewrite has an implementation error and not
    a modelling one.
    """

    def setUp(self):
        (self.sec, self.xr, self.yr, self.limits,
         self.law, self.ecm, self.fctm) = _reference_section()
        self.lin = LinearElastic(E=self.ecm)

    def test_reported_extremes_are_pinned(self):
        for hist, stages in _histories(self.limits).items():
            res = _run_quiet(self.sec, stages, {"C35svc": self.lin},
                             self.xr, self.yr)
            conc = res["stages"][-1]["concrete"]
            self.assertAlmostEqual(float(conc["sigma_min_MPa"]),
                                   PRISTINE["lin.report.sig_min"],
                                   places=6, msg=hist)
            self.assertAlmostEqual(float(conc["sigma_max_MPa"]),
                                   PRISTINE["lin.report.sig_max"],
                                   places=6, msg=hist)

    def test_f6_is_closed_on_the_check_path(self):
        r"""
        **Repaired in Phase 29** — was a guard pinning an open defect.

        Phase 28 left this fixture asserting that the check verified
        ``S_bulk[i_min]``, the stress at the centroid of the outermost
        fibre, while the record reported the section face and the check
        wore the *face's* coordinates.  The assertion was correct and
        the state it pinned was wrong; the fixture existed so the defect
        could not be repaired by accident and reported as D27-1.

        Phase 29 repaired it under ratification (``29_0`` §2, Q1-A).
        The fixture is **inverted, not deleted**: the same two numbers
        are compared at the same point on the same section, and what
        used to be ``assertNotAlmostEqual`` is now ``assertAlmostEqual``.
        Deleting it would have removed the only place in the suite where
        the two accountings are read side by side.

        .. code-block:: text

            pristine   check 10.500279   report 10.535836   gap 0.3375 %
            repaired   check 10.535836   report 10.535836   gap 0.0000 %
        """
        stages = _histories(self.limits)["single"]
        res = _run_quiet(self.sec, stages, {"C35svc": self.lin},
                         self.xr, self.yr)
        sr = res["stages"][-1]
        # Phase 32: one check per zone, named ``concrete_compression
        # [<zone>]``.  This section has a single zone, so there is
        # exactly one record and it is the same record this fixture
        # always read; the stem match is what keeps it reading it.
        checked = [c for c in sr["checks"]
                   if c["name"].split("[")[0] == "concrete_compression"][0]
        self.assertAlmostEqual(
            float(checked["value_MPa"]),
            abs(PRISTINE["lin.report.sig_min"]), places=6,
            msg=("the check and the report disagree: two bookkeepings "
                 "for one question, which is F6 reopened."))
        self.assertNotAlmostEqual(
            float(checked["value_MPa"]),
            PRISTINE["lin.check.value"], places=4,
            msg=("the check still verifies the fibre centroid "
                 f"({PRISTINE['lin.check.value']}), the Phase-28 value."))
        # The label is a polygon *vertex* -- the corner of the section.
        # It always was; what changed is that the value now belongs to
        # it.
        self.assertEqual(tuple(checked["at"]), (B, 0.0))


if __name__ == "__main__":
    unittest.main()
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=0dc11a76b16d65a6062b9ba134efcd840473bfb62d70b714b996206efeab101e body_lines=659
