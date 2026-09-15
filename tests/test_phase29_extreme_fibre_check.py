# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
# This file is part of GenSec.  AGPL-3.0-or-later.
# ---------------------------------------------------------------------------
r"""
Phase-29 fixtures: **F6 on the check path**, and the decompression record.

What the defects are
--------------------
**F6-check.**  :func:`gensec.solver.sls.verify_sls_staged` reported the
concrete extremes from the closed form over the zone's polygon — the
true section face — while ``concrete_compression`` verified
``S_bulk[i_min]``, the stress at the **centroid of the outermost fibre**,
and then labelled the verified value with the *face's* coordinates.  Two
bookkeepings, one question, and a coordinate borrowed from the one that
did not produce the number.

The gap is pure discretisation and it is exactly proportional to the
half-fibre inset :math:`H/2N_y`.  Measured on the 400x1200 reference,
linear declaration, :math:`M = 900` kNm:

.. code-block:: text

    N_y   inset [mm]      face        centroid      gap
     24      25.000    -10.549208    10.193079    3.3759 %
     60      10.000    -10.537860    10.395597    1.3500 %
    240       2.500    -10.535836    10.500279    0.3375 %
    480       1.250    -10.535735    10.517957    0.1687 %

3.3759 / 0.1687 = 20.01 over a 20x refinement.  The direction is
**non-conservative** in every row, against EN 1992-1-1 §7.2, which limits
:math:`\sigma_c` at the extreme fibre of the *section* — not at the
centroid of the last row of a discretisation, which is an artefact of
GenSec and not a point of the member.

**The decompression record.**  The verdict is taken on ``eps_probe``
(Phase 28) while the record also published ``sigma_probe_MPa``, the
zone's law evaluated at the same point.  On a cracking law the two give
**opposite** answers as soon as the probe point cracks: the law returns
zero beyond :math:`\varepsilon_{ct}`, so :math:`\sigma \le 0` passes
while the strain correctly fails.  Measured on the 300x500 prestressed
reference, :math:`\varepsilon_{ct} = 8.971136\times10^{-5}`:

.. code-block:: text

    M [kNm]    eps_probe    sigma_probe   strain   stress
       -80    4.3911e-05      1.571179     fail     fail
       -90    1.6766e-04      0.000000     fail     PASS   <- disagree
      -160    6.8024e-04      0.000000     fail     PASS   <- disagree

And because that record carried no magnitude at all, it never entered
the pool the *governing* check is drawn from: a run that failed
decompression named ``concrete_compression`` — a check that **passed**,
at :math:`\eta = 0.273348`.

What is repaired
----------------
* ``concrete_compression`` verifies the closed form.  One accounting,
  one point, one number (Andrea's ratification, ``29_0`` §2, Q1-A).
* The decompression record carries **one** number, the strain, and
  carries it the way every other check carries its magnitude — value,
  limit and :math:`\eta` — so the check is structurally uniform and the
  governing utilisation can see it (Q3-E).

What is deliberately **not** repaired
-------------------------------------
* **The breakpoint coordinate on a non-convex zone.**  Where a law's
  turning point falls strictly inside the strain interval a holed or
  re-entrant region spans, the interpolated coordinate can leave the
  material — measured at 106.7 mm and 360.0 mm inside a void on
  ``example_custom`` and ``example_annulus``.  The *value* is right
  (the region is connected and the strain continuous, so the breakpoint
  strain is attained by real material), only the label is wrong.  Not
  ratified for this phase; open and named in ``29_0`` §10.
* **One ``sigma_c_comp`` for a section carrying two concretes.**  The
  check reads the global extreme against a single scalar, so it can pass
  a section whose weaker zone is 19.5 % over its own limit.  Repairing
  it changes ``StressLimits``, i.e. the input format.  Open and named.

Red on the pristine tree
------------------------
Every value asserted below was measured on the tree rebuilt from
``gensec_phase28.bundle`` and is written into this file, so the red is
reproducible rather than remembered.  The six repair fixtures fail there
with ``AssertionError`` — none with ``ImportError`` or
``AttributeError``, which is the red the closure gate excludes.  The two
**declared guards** at the end are green on *both* trees by design and
say so: they are falsified by deliberate corruption, not by the repair.
"""
import unittest

import numpy as np

from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.section import RectSection
from gensec.materials.concrete import Concrete
from gensec.materials.ec2_properties import fben2
from gensec.materials.elastic import LinearElastic
from gensec.materials.service_concrete import ServiceConcrete
from gensec.materials.steel import PrestressingSteel, Steel
from gensec.materials.verification_limits import StressLimits
from gensec.solver.section_state import SectionState
from gensec.solver.sls import verify_sls_staged

# ----------------------------------------------------------------------
#  The 400x1200 reference -- the section the Phase-28 numeric probe drives
# ----------------------------------------------------------------------
B, H, NFY = 400.0, 1200.0, 240
AS, Y1, Y2 = 2000.0, 60.0, 1140.0
N_AX, M_AX = -1000.0e3, 900.0e6
AGE, EPS_CU1 = 28.0, 0.0035

# ----------------------------------------------------------------------
#  The 300x500 prestressed reference -- the section the Phase-7
#  decompression validators drive
# ----------------------------------------------------------------------
DB, DH, DNFY = 300.0, 500.0, 200
DEC, DES, DEP = 32000.0, 200000.0, 195000.0
DAS, DYS1, DYS2 = 628.0, 50.0, 450.0
DAP, DYT = 1500.0, 100.0
DEPS_PE, DCDEC = 0.0006, 25.0

#: Values measured on the tree rebuilt from ``gensec_phase28.bundle``.
#: They are the red these fixtures reproduce, written down rather than
#: recalled (``28_0`` §3, E28-1: a figure copied is not a figure
#: measured).
PRISTINE = {
    # -- F6 on the check path --------------------------------------
    "lin.report.sig_min": -10.535836,
    "lin.check.value": 10.500279,
    "lin.check.eta": 0.500013,
    "lin.check.value.nfy60": 10.395597,
    "svc.report.sig_min": -16.459843,
    "svc.check.value": 16.363672,
    "svc.check.eta": 0.779222,
    # -- the decompression record ----------------------------------
    #
    # Read from ``.fingerprints/29_00_numeric_PRE.txt``, keys
    # ``dec.svc[M=-90].*``, and not from a probe run beside it.  The
    # first draft of this table carried 0.273348 against the ``svc``
    # label; that number is real but belongs to ``dec.lin[M=-90]`` --
    # the *linear* declaration on the same section.  Caught by the red
    # run, corrected from the instrument's own output, and recorded as
    # E29-1 in ``29_0`` §3: it is E28-1 committed again, inside the file
    # that exists to stop it.
    "dec.eps_ct": 8.97113587347e-05,
    "dec.svc.M-90.eps_probe": 0.000167662786676,
    "dec.svc.M-90.sigma_probe": 0.0,
    "dec.svc.M-90.governing.check": "concrete_compression",
    "dec.svc.M-90.governing.eta": 0.353984,
    "dec.lin.M-90.governing.eta": 0.273348,
}

#: What the repaired tree must produce.  Measured on the prototype
#: before ratification and re-measured after the repair.
REPAIRED = {
    "lin.check.value": 10.535836,
    "lin.check.eta": 0.501706,
    "svc.check.value": 16.459843,
    "svc.check.eta": 0.783802,
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
    return ServiceConcrete(fcm=float(p.fcm), Ecm=float(p.ecm),
                           eps_c1=abs(float(p.eps_c1)),
                           eps_cu1=EPS_CU1, fct_eff=float(p.fctm),
                           reference_age=age, name=name)


def _reference_section(nfy=NFY):
    r"""
    The 400x1200 C35/45 reference section, two rebar layers.

    Parameters
    ----------
    nfy : int, optional
        Fibre rows.  The mesh is a *parameter* here because the defect
        under repair is mesh-dependent and a fixture that fixed the mesh
        could not exhibit that.

    Returns
    -------
    tuple
        ``(section, x_ref, y_ref, limits, service_law, Ecm)``.
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
    section = RectSection(B, H, conc, rebars, n_fibers_y=nfy)
    # Phase 30: fct_eff removed from StressLimits; the service law
    # built for this bench carries it.
    limits = StressLimits(k_c=0.6,
                          sigma_s_tens=0.8 * 450.0)
    return section, xr, yr, limits, law, float(p.ecm)


def _prestressed_section():
    r"""
    The 300x500 reference with one bonded tendon at ``y = 100``.

    Returns
    -------
    tuple
        ``(section, x_ref, y_ref)``.
    """
    conc = Concrete(fck=35.0, gamma_c=1.5,
                    long_term_reduction_factor=0.85)
    conc.name = "C35"
    st = Steel(fyk=450.0)
    st.name = "B450C"
    ps = PrestressingSteel(f_p01d=1391.3, Ep=DEP)
    ps.name = "Y1860"
    xr, yr = DB / 2.0, DH / 2.0
    reb = [RebarLayer(y=DYS1, As=DAS, material=st, x=xr),
           RebarLayer(y=DYS2, As=DAS, material=st, x=xr)]
    ten = [Tendon(y=DYT, Ap=DAP, material=ps, x=xr,
                  eps_pe=DEPS_PE, name="T1")]
    return (RectSection(DB, DH, conc, reb, n_fibers_y=DNFY,
                        tendons=ten), xr, yr)


def _single_stage(limits):
    """One stage reaching the reference demand."""
    return [{"name": "one",
             "state": SectionState(0, np.ones(2, dtype=bool),
                                   np.ones(2, dtype=bool),
                                   np.zeros(2, dtype=float),
                                   bulk_eps_init=0.0, label="one"),
             "increment": (N_AX, M_AX, 0.0),
             "limits": limits}]


def _run(section, stages, moduli, xr, yr):
    """Run the verification and return the result mapping."""
    return verify_sls_staged(section, stages, moduli=moduli,
                             x_ref=xr, y_ref=yr)


def _named(checks, name):
    """The first check whose ``name`` starts with *name*."""
    hits = [c for c in checks if c.get("name", "").startswith(name)]
    if not hits:
        raise AssertionError(
            f"no check named {name!r}; present: "
            f"{[c.get('name') for c in checks]}")
    return hits[0]


def _dec_run(mat, m_kn):
    """One decompression run at one hogging demand."""
    section, xr, yr = _prestressed_section()
    limits = StressLimits(name="dec", decompression=True, c_dec=DCDEC,
                          k_c=0.6,
                          sigma_s_tens=0.8 * 450.0)
    state = SectionState(0, np.ones(3, dtype=bool),
                         np.ones(3, dtype=bool),
                         np.array([0.0, 0.0, DEPS_PE], dtype=float),
                         bulk_eps_init=0.0, label="d")
    return verify_sls_staged(
        section,
        [{"name": "d", "state": state,
          "increment": (0.0, m_kn * 1.0e6, 0.0), "limits": limits}],
        moduli={"C35": mat}, x_ref=xr, y_ref=yr)


def _service_for_dec():
    """The service law of the prestressed reference."""
    return _service_law(35.0, 28.0, "C35")


# ======================================================================
#  Repair fixtures -- AssertionError-red on gensec_phase28.bundle
# ======================================================================


class TestTheCheckVerifiesTheExtremeFibre(unittest.TestCase):
    r"""
    ``concrete_compression`` must verify the stress the clause names.

    EN 1992-1-1 §7.2 limits :math:`\sigma_c` at the extreme fibre of the
    section.  The extreme fibre is a point of the member; the centroid of
    the outermost row of fibres is a point of GenSec's discretisation.
    Only one of the two exists independently of how the user meshed.
    """

    def setUp(self):
        (self.sec, self.xr, self.yr, self.limits,
         self.law, self.ecm) = _reference_section()
        self.lin = LinearElastic(E=self.ecm, name="C35lin")

    def _checked_and_reported(self, mat, section=None, xr=None, yr=None):
        """``(checked value, reported |sigma_min|, labelled point)``."""
        sec = self.sec if section is None else section
        xr = self.xr if xr is None else xr
        yr = self.yr if yr is None else yr
        res = _run(sec, _single_stage(self.limits),
                   {"C35svc": mat}, xr, yr)
        sr = res["stages"][-1]
        c = _named(sr["checks"], "concrete_compression")
        return (float(c["value_MPa"]),
                abs(float(sr["concrete"]["sigma_min_MPa"])),
                tuple(c["at"]))

    def test_checked_value_equals_the_reported_extreme(self):
        r"""
        The number verified and the number reported are the same number.

        Red on the pristine tree: 10.500279 against 10.535836, an
        under-report of 0.3375 % in the non-conservative direction.
        """
        for tag, mat in (("lin", self.lin), ("svc", self.law)):
            with self.subTest(declaration=tag):
                checked, reported, _ = self._checked_and_reported(mat)
                self.assertAlmostEqual(
                    checked, reported, places=6,
                    msg=(f"[{tag}] the check verifies {checked!r} while "
                         f"the record reports {reported!r}: two "
                         f"bookkeepings for one question (F6)."))
                self.assertAlmostEqual(
                    checked, REPAIRED[f"{tag}.check.value"], places=6)

    def test_the_verdict_does_not_depend_on_the_mesh(self):
        r"""
        Refining the mesh must not move the *reading* of the stress.

        The extreme-fibre stress is a property of the section and the
        solved plane.  A verdict that moves when the user changes
        ``n_fibers_y`` is reporting the discretisation, not the section.

        Two mesh dependences live in this number and only one is F6's:

        * the **reading**, which took the stress half a fibre inside the
          face and therefore scaled with :math:`H/2N_y` -- this is the
          defect, and it is gone;
        * the **equilibrium**, because the solved plane is itself the
          result of a fibre-integrated balance -- this is untouched by
          Phase 29 and is not a defect.

        The residual is measured, not assumed.  On the repaired tree the
        checked values at :math:`N_y = 60/240/480` are 10.537860,
        10.535836, 10.535735 -- byte-for-byte the values the *report*
        path already produced on the pristine tree at the same meshes.
        The whole of what is left is therefore the equilibrium's own,
        and none of it is the reading's.

        The threshold is that residual, expressed against the defect it
        replaces rather than as a round number picked to pass: the
        spread must fall to under a tenth of the pristine one.  Measured
        0.002125 against 0.104682, a factor of 49.
        """
        vals = []
        for nfy in (60, 240, 480):
            sec, xr, yr, limits, law, ecm = _reference_section(nfy)
            res = _run(sec, _single_stage(limits),
                       {"C35svc": LinearElastic(E=ecm, name="C35lin")},
                       xr, yr)
            sr = res["stages"][-1]
            checked = float(_named(sr["checks"],
                                   "concrete_compression")["value_MPa"])
            reported = abs(float(sr["concrete"]["sigma_min_MPa"]))
            # The residual must be the equilibrium's: at every mesh the
            # checked value IS the reported one.
            self.assertAlmostEqual(
                checked, reported, places=6,
                msg=f"at N_y={nfy} the two accountings differ again")
            vals.append(checked)
        spread = max(vals) - min(vals)
        pristine_spread = (PRISTINE["lin.check.value"]
                           - PRISTINE["lin.check.value.nfy60"])
        self.assertLess(
            spread, 0.1 * pristine_spread,
            msg=(f"the verified stress spans {spread:.6f} MPa across "
                 f"N_y = 60/240/480 ({vals}); the pristine spread is "
                 f"{pristine_spread:.6f} and the repair must cut it by "
                 f"at least an order of magnitude."))

    def test_the_label_belongs_to_the_value(self):
        r"""
        The stress at the labelled point *is* the verified stress.

        Re-derived independently from the solved plane and the declared
        law, touching none of the machinery under repair:

        .. math::

            \varepsilon(x, y) = \varepsilon_0
                + \chi_x\,(y - y_{\mathrm{ref}})
                - \chi_y\,(x - x_{\mathrm{ref}})

        Red on the pristine tree, where the label is the section corner
        and the value is the fibre centroid half a fibre inside it.
        """
        for tag, mat in (("lin", self.lin), ("svc", self.law)):
            with self.subTest(declaration=tag):
                res = _run(self.sec, _single_stage(self.limits),
                           {"C35svc": mat}, self.xr, self.yr)
                sr = res["stages"][-1]
                c = _named(sr["checks"], "concrete_compression")
                x, y = (float(v) for v in c["at"])
                pl = sr["plane"]
                eps = (float(pl["eps0"])
                       + float(pl["chi_x"]) * (y - self.yr)
                       - float(pl["chi_y"]) * (x - self.xr))
                sig = float(np.asarray(
                    mat.stress_array(np.array([eps], dtype=float)))[0])
                self.assertAlmostEqual(
                    abs(sig), float(c["value_MPa"]), places=4,
                    msg=(f"[{tag}] the label {(x, y)} carries "
                         f"{abs(sig):.6f} MPa but the check verified "
                         f"{float(c['value_MPa']):.6f}."))


class TestTheDecompressionRecordCarriesOneNumber(unittest.TestCase):
    r"""
    One verdict, one number, and that number is the one it was taken on.

    Two numbers for one question is the shape that produced F6: the
    record stays internally readable while the value and the verdict come
    from different places.
    """

    def test_no_second_number_survives(self):
        r"""
        ``sigma_probe_MPa`` is gone from the record.

        On a cracking law it contradicts the verdict: past
        :math:`\varepsilon_{ct}` the law returns zero, so a reader taking
        :math:`\sigma \le 0` as the criterion reads *pass* where the
        engine correctly reads *fail*.
        """
        res = _dec_run(_service_for_dec(), -90.0)
        c = _named(res["stages"][0]["checks"], "decompression")
        self.assertNotIn(
            "sigma_probe_MPa", c,
            msg=("the record still publishes a stress beside the "
                 "strain the verdict was taken on; on this run the "
                 "stress reads 0.000000 and the verdict is 'failed'."))

    def test_the_check_publishes_its_magnitude(self):
        r"""
        Decompression carries a value and a limit — and **no**
        :math:`\eta`.

        The magnitude is the **strain**: the quantity the verdict is
        taken on, against the tolerance actually applied,
        :math:`\varepsilon_{\mathrm{tol}} = \sigma_{\mathrm{tol}} /
        E_{\mathrm{ref}}`.

        There is deliberately no utilisation.  Every other check divides
        a demand by a capacity — :math:`\sigma_{c,\mathrm{comp}}`,
        :math:`\sigma_{s,\mathrm{tens}}`, :math:`\sigma_{p,\max}`.
        Decompression has none: it asks whether the stress is
        :math:`\le 0`, and ``_DECOMPRESSION_TOL`` is solver round-off,
        not a resistance.  Measured on this exact run, the ratio the
        first draft of Phase 29 proposed evaluates to

        .. math::

            \eta = \frac{1.6766\times10^{-4}}{3.125\times10^{-14}}
                 = 5.4\times10^{9},

        which is a division by zero wearing a number.  Ratified as E'
        (``29_0`` §2, Q3): publish the magnitude, publish no
        utilisation, and repair the governing selection instead.

        Red on the pristine tree, where the record carries neither value
        nor limit and decompression is the only check with no magnitude
        at all.
        """
        res = _dec_run(_service_for_dec(), -90.0)
        c = _named(res["stages"][0]["checks"], "decompression")
        for field in ("value", "limit"):
            self.assertIn(field, c,
                          msg=f"the record carries no {field!r}.")
        self.assertNotIn(
            "eta", c,
            msg=("decompression has no capacity to divide by; an eta "
                 "here would be a ratio against solver round-off."))
        self.assertAlmostEqual(
            float(c["value"]), PRISTINE["dec.svc.M-90.eps_probe"],
            places=12,
            msg="the magnitude is not the strain the verdict was on")
        self.assertFalse(bool(c["passed"]))

    def test_a_failed_decompression_governs(self):
        r"""
        **Behavioural.**  The governing check is one that actually
        governed.

        Red on the pristine tree: at :math:`M = -90` kNm the section is
        ``verified = False`` *because of* decompression, and the run
        names ``concrete_compression`` — which passed, at
        :math:`\eta = 0.273348` — as governing, because a record with no
        :math:`\eta` never enters the pool the maximum is taken over.
        """
        res = _dec_run(_service_for_dec(), -90.0)
        sr = res["stages"][0]
        self.assertFalse(bool(sr["verified"]),
                         "the fixture no longer exhibits a failure")
        dec = _named(sr["checks"], "decompression")
        self.assertFalse(bool(dec["passed"]))
        gov = res["governing"]
        self.assertIsNotNone(gov)
        self.assertTrue(bool(gov["failed"]),
                        "the governing check is one that passed")
        self.assertTrue(
            str(gov["check"]).startswith("decompression"),
            msg=(f"the section failed on decompression but the run "
                 f"names {gov['check']!r} at eta={gov['eta']!r} as "
                 f"governing; on the pristine tree that is "
                 f"{PRISTINE['dec.svc.M-90.governing.check']!r} at "
                 f"{PRISTINE['dec.svc.M-90.governing.eta']!r}, a check "
                 f"that passed."))


# ======================================================================
#  Declared guards -- GREEN ON BOTH TREES, and that is stated
# ======================================================================


class TestElementChecksKeepASingleAccounting(unittest.TestCase):
    r"""
    **Declared guards.**  These pass on the pristine tree *and* on the
    repaired one, and they are not evidence that anything was repaired.

    They exist because Phase 29 asked whether the F6 shape — a value from
    one bookkeeping wearing a label from another — also existed for the
    element checks.  Measured: it does not, and it structurally cannot,
    because an element is a **point** and a point has no face and no
    centroid to disagree about.  The guards pin that answer so the
    pattern cannot appear later without a fixture noticing
    (``29_0`` §2, Q2-B).

    They are falsified by deliberate corruption rather than by the
    repair: see the corruption table in ``29_0`` §7.
    """

    def setUp(self):
        (self.sec, self.xr, self.yr, self.limits,
         self.law, self.ecm) = _reference_section()

    def test_steel_tension_reads_the_recorded_element_stress(self):
        """The checked value is a stress the record already published."""
        for tag, mat in (("lin", LinearElastic(E=self.ecm, name="l")),
                         ("svc", self.law)):
            with self.subTest(declaration=tag):
                res = _run(self.sec, _single_stage(self.limits),
                           {"C35svc": mat}, self.xr, self.yr)
                sr = res["stages"][-1]
                c = _named(sr["checks"], "steel_tension")
                recorded = max(float(e["sigma_MPa"])
                               for e in sr["elements"]
                               if e["kind"] == "rebar")
                self.assertAlmostEqual(float(c["value_MPa"]), recorded,
                                       places=6)

    def test_no_element_check_carries_a_borrowed_coordinate(self):
        """No element check labels its value with a point at all."""
        res = _run(self.sec, _single_stage(self.limits),
                   {"C35svc": self.law}, self.xr, self.yr)
        for c in res["stages"][-1]["checks"]:
            if c.get("name", "").startswith(("steel_tension",
                                             "tendon_stress")):
                self.assertNotIn("at", c)


if __name__ == "__main__":
    unittest.main()
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=a9cb5d71c3022da62c95d9f34e4683184779fb1eb4511842fc3e66d764690bcc body_lines=608
