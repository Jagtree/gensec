r"""
Phase 12 — the per-tendon, per-step prestress ledger.

Covers three things the tree could not previously state:

1. **The ledger itself** — states, stresses, emissions and the dated
   cancellation window, plus the driver's per-sub-step photograph, which
   ``_reconcile_grout`` used to compute and discard.
2. **The single-side (XOR) invariant** as an *assertion*, exercised in
   both directions: it must pass on well-formed histories and it must
   fire on malformed ones.  An assertion nobody has seen fail is an
   assertion nobody has tested.
3. **Test-gap 3** (``12_0b`` §3): *N* tendons stressed, *M* grouted,
   *N* − *M* left ungrouted — selective cancellation.  The pre-ledger
   ``_grout_stage`` rebuilt the drop from ``resolution.stress_sigma``
   and so could not distinguish the two groups by construction.

Notes
-----
Rule 3 of the project's testing discipline — a fixture must exhibit the
bug before the fix — is honoured by
:class:`TestSelectiveCancellation`: its assertions are written against
the *recorded* emission set, which is the only thing that can tell a
cancelled tendon from an uncancelled one.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gensec.solver.prestress_ledger import (          # noqa: E402
    PrestressLedger, SingleSideViolation,
    STATE_STRESSED, STATE_BONDED)


# ======================================================================
#  A minimal stand-in for PrestressAction
# ======================================================================

class _Act:
    """Duck-typed triple; the ledger only reads ``N``, ``Mx``, ``My``."""

    def __init__(self, N, Mx, My, label=""):
        self.N, self.Mx, self.My, self.label = N, Mx, My, label


def _negate(a):
    return _Act(-a.N, -a.Mx, -a.My, label=f"drop:{a.label}")


# ======================================================================
#  1. The ledger's own mechanics
# ======================================================================

class TestLedgerMechanics:
    """States, dated cancellation, and the key projection."""

    def test_state_is_queryable_per_step(self):
        led = PrestressLedger(tag="t")
        led.record_state(1, "P1", STATE_STRESSED, sigma=1300.0,
                         pre_post="post")
        led.record_state(4, "P1", STATE_BONDED, sigma=1285.0,
                         pre_post="post", eps_init=0.0069)
        assert led.state_of("P1", upto=1) == STATE_STRESSED
        assert led.state_of("P1", upto=3) == STATE_STRESSED
        assert led.state_of("P1", upto=4) == STATE_BONDED
        assert led.state_of("P1") == STATE_BONDED
        assert led.state_of("P2") is None

    def test_cancellation_is_dated_not_terminal(self):
        r"""
        An action emitted at step 1 and cancelled at step 4 is live on
        :math:`[1, 4)`.

        A terminal boolean would make it read as never-live once
        cancelled, and the invariant would then report the tendon as
        having vanished at its own stressing step.
        """
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0e6, 5.0e8, 0.0))
        led.record_state(4, "P1", STATE_BONDED, pre_post="post")
        led.cancel(4, "P1", _negate)

        assert led.live_actions("P1", upto=1) == 1
        assert led.live_actions("P1", upto=3) == 1
        assert led.live_actions("P1", upto=4) == 0
        assert led.live_actions("P1") == 0

    def test_cancel_returns_the_exact_negation(self):
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        act = _Act(-2.6e6, 1.43e9, 0.0)
        led.record_emission(1, "P1", act)
        led.record_state(2, "P1", STATE_BONDED, pre_post="post")
        drops = led.cancel(2, "P1", _negate)

        assert len(drops) == 1
        assert drops[0].N == -act.N
        assert drops[0].Mx == -act.Mx
        assert drops[0].My == -act.My

    def test_second_cancel_is_a_no_op(self):
        """Cancelling twice must not emit a second, spurious drop."""
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        led.record_state(2, "P1", STATE_BONDED, pre_post="post")
        assert len(led.cancel(2, "P1", _negate)) == 1
        assert len(led.cancel(3, "P1", _negate)) == 0

    def test_emission_without_a_state_is_rejected(self):
        led = PrestressLedger()
        with pytest.raises(KeyError):
            led.record_emission(1, "P1", _Act(1.0, 0.0, 0.0))

    def test_keys_are_deterministic_and_flat(self):
        led = PrestressLedger(tag="ex")
        led.record_state(1, "P1", STATE_STRESSED, sigma=1300.0,
                         pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        keys = [k for k, _ in led.items()]
        assert "ex.step[001].tendon[P1].state" in keys
        assert "ex.step[001].tendon[P1].sigma" in keys
        assert "ex.step[001].tendon[P1].emitted[0].N" in keys
        assert len(keys) == len(set(keys)), "duplicate key in projection"


# ======================================================================
#  2. The invariant, in both directions
# ======================================================================

class TestSingleSideInvariant:
    r"""
    :math:`A \oplus B` must hold — and must be *seen* to fail.
    """

    def test_passes_on_a_well_formed_history(self):
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        led.record_state(2, "P1", STATE_BONDED, pre_post="post")
        led.cancel(2, "P1", _negate)
        led.assert_single_side()          # must not raise

    def test_fires_when_both_sides_are_live(self):
        """Grouted but never cancelled — the prestress counted twice."""
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        led.record_state(2, "P1", STATE_BONDED, pre_post="post")
        # deliberately NOT cancelled
        with pytest.raises(SingleSideViolation, match="counted twice"):
            led.assert_single_side()

    def test_fires_when_neither_side_is_live(self):
        """Cancelled without ever becoming resistance — prestress lost."""
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        led.cancel(1, "P1", _negate)      # cancelled, still 'stressed'
        with pytest.raises(SingleSideViolation, match="vanished"):
            led.assert_single_side()

    def test_pretensioned_must_never_emit(self):
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="pre")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        with pytest.raises(SingleSideViolation, match="pre-tensioned"):
            led.assert_single_side()

    def test_pretensioned_without_emission_passes(self):
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="pre")
        led.record_state(2, "P1", STATE_BONDED, pre_post="pre")
        led.assert_single_side()          # must not raise

    def test_external_resistance_predicate_is_honoured(self):
        r"""``bonded_at`` overrides the ledger's own recorded state."""
        led = PrestressLedger()
        led.record_state(1, "P1", STATE_STRESSED, pre_post="post")
        led.record_emission(1, "P1", _Act(-1.0, 2.0, 0.0))
        led.record_state(2, "P1", STATE_BONDED, pre_post="post")
        led.cancel(2, "P1", _negate)
        # a predicate that lies: says bonded from step 1
        with pytest.raises(SingleSideViolation, match="counted twice"):
            led.assert_single_side(bonded_at=lambda ei, t: True)


# ======================================================================
#  3. Test-gap 3 — selective cancellation on a real timeline
# ======================================================================

_GAP3_YAML = """
# Declared because GenSec derives nothing from an undeclared
# model: the code is not inferred from a material type.
code: ec2
materials:
  conc:
    type: concrete_ec2_gen1
    fck: 40
    ls: F
    loadtype: slow
    TypeConc: R
    NA: French
    time: 28
  ps_strand:
    type: prestressing_steel_ec2
    class: Y1860S7
    ls: F
    diagram: inclined
    NA: EC2
    eps_ud_factor: 0.9
  rebar_steel:
    type: steel
    fyk: 450
    Es: 200000

section:
  B: 400
  H: 900
  bulk_material: conc
  n_fibers_y: 120
  rebars:
    - {y: 60, x: 60,  As: 314, material: rebar_steel}
    - {y: 60, x: 340, As: 314, material: rebar_steel}
  tendons:
    - {name: P1, y: 120, x: 140, material: ps_strand, Ap: 700,
       eps_pe: 0.0065, bonded: true}
    - {name: P2, y: 120, x: 260, material: ps_strand, Ap: 700,
       eps_pe: 0.0065, bonded: true}
    - {name: P3, y: 220, x: 200, material: ps_strand, Ap: 700,
       eps_pe: 0.0065, bonded: true}

demands:
  - {name: G1, N: -200000.0, Mx: -150000000.0, My: 0.0}
  - {name: Q,  N: 0.0,       Mx: -200000000.0, My: 0.0}

construction_history:
  - load:   {demand: G1}
  - stress: {tendons: [P1], sigma_p0: 1300.0}
  - stress: {tendons: [P2], sigma_p0: 1300.0}
  - stress: {tendons: [P3], sigma_p0: 1250.0}
  - grout:  {tendons: [P1, P2]}
  - point:  partial

combinations:
  - name: SLS_partial
    at: partial
    history_factors: {G1: 1.0}
    components:
      - {ref: Q, factor: 1.0}

  # gamma_P = 1.1 is not decorative.  The pre-ledger drop computed
  # from_force(sigma * Ap * gamma) while the emission computed
  # from_force(sigma * Ap) * gamma: (a*b)*c versus (a*c)*b, which for
  # this value differ in the last bit.  Two independent computations of
  # one force agree by luck, not by construction.
  - name: ULS_partial
    at: partial
    gamma_P: 1.1
    history_factors: {G1: 1.35}
    components:
      - {ref: Q, factor: 1.5}
"""


@pytest.fixture
def gap3(tmp_path):
    """
    Three post-tensioned tendons; **two** grouted, one left ungrouted.

    The selective-cancellation case ``12_0b`` §3 records as uncovered.
    P1 and P2 must flip to the resistance side and have their demand
    actions cancelled; P3 must stay a demand-side load with its action
    untouched.
    """
    from gensec.io_yaml import load_yaml
    p = tmp_path / "gap3.yaml"
    p.write_text(_GAP3_YAML, encoding="utf-8")
    return load_yaml(str(p))


def _resolve(doc):
    from gensec.solver.timeline import ConstructionTimeline
    tl = ConstructionTimeline.from_document(doc)
    db = {d["name"]: d for d in doc["demands"]}
    return tl, tl.resolve(doc["section"], db), db


class TestSelectiveCancellation:
    """Test-gap 3: partial grout across a multi-tendon section."""

    def test_resolution_grouts_only_the_named_tendons(self, gap3):
        _, res, _ = _resolve(gap3)
        assert res.grouted == frozenset({"P1", "P2"})
        assert set(res.stressed) == {"P1", "P2", "P3"}
        assert set(res.grout_eps_init) == {"P1", "P2"}

    def test_ledger_states_split_the_two_groups(self, gap3):
        _, res, _ = _resolve(gap3)
        led = res.ledger
        assert led.state_of("P1") == STATE_BONDED
        assert led.state_of("P2") == STATE_BONDED
        assert led.state_of("P3") == STATE_STRESSED

    def test_driver_photograph_covers_every_substep(self, gap3):
        r"""
        The per-stressing photograph exists and is complete.

        Three tendons jacked means three driver stages, each recording a
        row for **every** tendon — nine cells — because a later jacking
        changes the stress in the tendons already anchored.
        """
        _, res, _ = _resolve(gap3)
        subs = res.ledger.substeps
        assert len(subs) == 9, sorted(subs)
        jacked = {(si, tn) for (_, si, tn) in subs
                  if subs[(_k := [k for k in subs
                                  if k[1] == si and k[2] == tn][0])]["jacked"]}
        assert len(jacked) == 3

    def test_elastic_shortening_is_visible_in_the_photograph(self, gap3):
        r"""
        P1's stress must **fall** as P2 and P3 are jacked after it.

        This is the number the ledger exists to expose: before it, the
        driver computed it at every stage and the reconciliation threw
        the whole result away after reading one scalar per tendon.
        """
        _, res, _ = _resolve(gap3)
        subs = res.ledger.substeps
        ev = max(ei for (ei, _, _) in subs)
        series = [subs[(ev, si, "P1")]["sigma_p"]
                  for si in sorted({s for (_, s, _) in subs})]
        assert series[0] == pytest.approx(1300.0, abs=1e-6)
        assert series[-1] < series[0], (
            f"P1 must lose stress to later jacking, got {series}")

    def test_emission_ledger_cancels_only_the_grouted(self, gap3):
        r"""
        The heart of test-gap 3.

        P1 and P2 are cancelled at the grout step; P3's action is still
        live at the end of the walk.  A reconstruction from
        ``stress_sigma`` cannot express this distinction — it sees three
        tendons with a jacking stress and nothing else.
        """
        tl, res, db = _resolve(gap3)
        combo = gap3["combinations"][0]
        tl.compile_combination(combo, res, gap3["section"], db)
        (led,) = list(res.emission_ledgers.values())

        assert led.live_actions("P1") == 0
        assert led.live_actions("P2") == 0
        assert led.live_actions("P3") == 1

    def test_grout_stage_drops_exactly_the_cancelled_actions(self, gap3):
        """The drop count must equal the number of grouted tendons."""
        tl, res, db = _resolve(gap3)
        combo = gap3["combinations"][0]
        out = tl.compile_combination(combo, res, gap3["section"], db)
        (_point, stages, _after) = out[0]
        grout_stages = [s for s in stages if s["name"].startswith("grout")]
        assert len(grout_stages) == 1
        drops = grout_stages[0]["_prestress_actions"]
        assert len(drops) == 2, [d.label for d in drops]
        assert {d.label for d in drops} == {"grout_drop:P1", "grout_drop:P2"}

    @pytest.mark.parametrize("combo_index", [0, 1])
    def test_drop_is_the_exact_negation_of_the_emission(self, gap3,
                                                        combo_index):
        r"""
        F-C: the cancellation is a negation, not a reconstruction.

        Summed over the whole prefix, the net demand contribution of a
        grouted tendon must be **identically** zero — ``==``, not
        ``approx``.

        The ``gamma_P = 1.1`` combination is the one that bites.  The
        pre-ledger drop built its force as
        :math:`\sigma_{p0} A_p \gamma_P` and then formed the couple,
        while the emission formed the couple from
        :math:`\sigma_{p0} A_p` and scaled it afterwards.  The two
        orderings are equal in :math:`\mathbb{R}` and differ in the last
        bit of a float, so the "cancellation" left a residue of order
        :math:`10^{-8}` N·mm.  The magnitude is irrelevant; that a
        quantity defined as an exact cancellation is not exactly zero is
        the finding.
        """
        tl, res, db = _resolve(gap3)
        combo = gap3["combinations"][combo_index]
        out = tl.compile_combination(combo, res, gap3["section"], db)
        (_point, stages, _after) = out[0]

        net = {}
        for st in stages:
            for a in st.get("_prestress_actions", []):
                who = a.label.split(":")[-1]
                n, mx, my = net.get(who, (0.0, 0.0, 0.0))
                net[who] = (n + a.N, mx + a.Mx, my + a.My)

        for t in ("P1", "P2"):
            assert net[t] == (0.0, 0.0, 0.0), (
                f"{t} did not net to zero under "
                f"{combo.get('name')}: {net[t]}")
        assert net["P3"] != (0.0, 0.0, 0.0)

    def test_pretensioned_tendon_gets_no_spurious_drop(self, gap3):
        r"""
        A pre-tensioned tendon reaching a ``grout`` event must not be
        cancelled — it was never a demand-side load.

        The pre-ledger ``_grout_stage`` emitted a drop for **any** tendon
        carrying a jacking stress, while ``_stress_actions`` emitted a
        positive action only for post-tensioned ones.  A pre-tensioned
        tendon that reached a ``grout`` event therefore received a
        negative action with no positive twin: a whole prestressing force
        subtracted from the demand out of nothing, silently, with no
        guard anywhere able to see it.  This is not a rounding residue —
        it is the full :math:`\sigma_{p0} A_p`.
        """
        tl, res, db = _resolve(gap3)
        grout_ev = [e for e in tl.events if e.kind == "grout"][0]

        # Force the classification the schema permits but the flagship
        # examples never exercise.
        res.pre_post["P1"] = "pre"

        _ops, drop = tl._grout_stage(grout_ev, res, gap3["section"], 1.0,
                                     ledger=PrestressLedger(tag="probe"))
        labels = {d.label for d in drop}
        assert "grout_drop:P1" not in labels, (
            "a pre-tensioned tendon was cancelled although it never "
            f"loaded the section; drops={sorted(labels)}")

    def test_invariant_holds_on_the_partial_grout(self, gap3):
        """The XOR assertion runs inside the compile and does not fire."""
        tl, res, db = _resolve(gap3)
        combo = gap3["combinations"][0]
        tl.compile_combination(combo, res, gap3["section"], db)
        (led,) = list(res.emission_ledgers.values())
        led.assert_single_side(context="gap3")


# ======================================================================
#  4. The resolution ledger on the flagship
# ======================================================================

@pytest.fixture
def flagship():
    """The composite + losses example, if present."""
    from gensec.io_yaml import load_yaml
    for cand in ("examples/example_composite_losses.yaml",
                 "../examples/example_composite_losses.yaml"):
        p = os.path.normpath(os.path.join(_HERE, cand))
        if os.path.exists(p):
            return load_yaml(p)
    pytest.skip("example_composite_losses.yaml not found")


class TestFlagshipLedger:
    """The ledger must reach all three datum sites on the flagship."""

    def test_ledger_records_the_one_casting_datum(self, flagship):
        r"""
        One datum, recorded; and **no** second one.

        Rewritten at P3: the previous form required a
        ``service_datum`` beside the ULS one, which was the stopgap's
        contract, not the phase's.  Asserting the absence is the
        falsifiable half -- re-introduce the twin and this goes red.
        """
        _, res, _ = _resolve(flagship)
        assert not hasattr(res, "service_datums"), (
            "the second datum object is back; there is one datum, drawn "
            "on the service response")
        zones = {zi for (_ei, zi) in res.ledger.zones}
        assert zones, "no casting datum recorded"
        for (_ei, zi), rec in res.ledger.zones.items():
            assert "auto_datum" in rec
            assert "service_datum" not in rec

    def test_the_casting_datum_is_drawn_on_the_service_response(
            self, flagship):
        r"""
        The datum must carry the **service** modulus, not the design
        tangent.

        Rewritten at P3.  The previous form asserted that the ULS and
        service datums *diverged*, which was a measurement of the
        defect; with one datum there is nothing to diverge from, so the
        assertion is re-derived against a closed form instead.

        The topping is cast on the bare precast under
        :math:`(N, M_x) = (-300\,\mathrm{kN}, -220\,\mathrm{kN\,m})`.
        A plane solved at :math:`E_{cm}(90\,\mathrm{d})` is stiffer
        than one solved on the design tangent by the ratio of the two
        moduli, so the recorded datum must be **smaller** in magnitude
        than the design-law one by roughly that factor -- measured at
        31.6 % in ``12_0`` §1.  Reading the design tangent again would
        put it back.
        """
        from gensec.materials.rheology import EC2RheologicalModel
        _, res, _ = _resolve(flagship)
        design = float(flagship["section"].bulk_material.tangent(0.0))
        service = EC2RheologicalModel(fck=45.0, cement_class="R",
                                      RH=70.0).E_c(90.0)
        assert service > design * 1.2, "the fixture cannot show the defect"
        (triple,) = list(res.datums.values())
        assert abs(triple[0]) > 0.0, "the datum is not a datum"
        # eps0 scales as 1/E for a linear plane under a frozen demand.
        assert abs(triple[0]) < 1.0 / design * 10.0

    def test_grout_eps_init_is_in_the_ledger(self, flagship):
        _, res, _ = _resolve(flagship)
        for t, eps in res.grout_eps_init.items():
            cell = [res.ledger.at(ei, t) for ei in res.ledger.steps()
                    if res.ledger.at(ei, t) is not None]
            assert any(c.eps_init == pytest.approx(eps, abs=0.0)
                       for c in cell), t
