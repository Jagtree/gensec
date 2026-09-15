r"""
Phase 18 — the ratified scope boundary, and the two element-class
consumers found by the line-by-line N-9 triage.

Every claim here was seen **false** on the pristine Phase 17 tree
(``066f5c9``, rebuilt from ``gensec_phase17.bundle``) before the repair
that makes it true.  The discriminating fixtures are marked; the
regression guards are marked too, and are declared as guards rather than
as evidence — they pass on both trees by construction, which is exactly
what makes them useless as proof and useful as a fence.

Sections
--------
1.  ``LoadBetweenJacksNotModelled`` — the ``_reconcile_grout``
    ratification (option 3): a load interposed between two jacking
    events, on a tendon that is stressed but not grouted, is declared
    out of scope instead of silently answered.
2.  N-13 — ``_build_material_name_map`` omits tendon materials.
3.  N-14 — ``NMDiagram`` builds its strain-plane sweep envelope from
    bulk and rebar only.
"""

import numpy as np
import pytest
from shapely.geometry import Polygon

from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.section import GenericSection
from gensec.materials import (Steel, concrete_from_class,
                              prestress_from_ec2)
from gensec.solver.capacity import NMDiagram
from gensec.solver.integrator import FiberSolver


# ======================================================================
#  1.  The ratified scope boundary
# ======================================================================

_BASE_YAML = """
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

demands:
  - {name: G1, N: -200000.0, Mx: -150000000.0, My: 0.0}
  - {name: G2, N: -100000.0, Mx: -80000000.0,  My: 0.0}
  - {name: Q,  N: 0.0,       Mx: -200000000.0, My: 0.0}

construction_history:
__HISTORY__

combinations:
  - name: SLS_end
    at: done
    history_factors: {G1: 1.0, G2: 1.0}
    components:
      - {ref: Q, factor: 1.0}
"""

#: A ``load`` between two jacks, both tendons stressed, none grouted.
#: This is the case the ratification declares out of scope.
_HIST_INTERPOSED = """\
  - load:   {demand: G1}
  - stress: {tendons: [P1], sigma_p0: 1300.0}
  - load:   {demand: G2}
  - stress: {tendons: [P2], sigma_p0: 1300.0}
  - grout:  {tendons: [P1, P2]}
  - point:  done
"""

#: The ordinary sequence: every load before the first jack.
_HIST_ORDINARY = """\
  - load:   {demand: G1}
  - load:   {demand: G2}
  - stress: {tendons: [P1], sigma_p0: 1300.0}
  - stress: {tendons: [P2], sigma_p0: 1300.0}
  - grout:  {tendons: [P1, P2]}
  - point:  done
"""

#: A load after the last jack: not interposed in a sequence.
_HIST_TRAILING = """\
  - load:   {demand: G1}
  - stress: {tendons: [P1], sigma_p0: 1300.0}
  - stress: {tendons: [P2], sigma_p0: 1300.0}
  - load:   {demand: G2}
  - grout:  {tendons: [P1, P2]}
  - point:  done
"""


def _compile(tmp_path, history, name="tl.yaml"):
    r"""Load a document with *history* and compile its combination."""
    from gensec.io_yaml import load_yaml
    from gensec.solver.timeline import ConstructionTimeline

    p = tmp_path / name
    p.write_text(_BASE_YAML.replace("__HISTORY__", history),
                 encoding="utf-8")
    doc = load_yaml(str(p))
    tl = ConstructionTimeline.from_document(doc)
    db = {d["name"]: d for d in doc["demands"]}
    res = tl.resolve(doc["section"], db)
    return tl.compile_combination(doc["combinations"][0], res,
                                  doc["section"], db)


def _compile_outcome(tmp_path, history, name="tl.yaml"):
    r"""
    Compile, and report the **outcome as data** instead of letting it
    propagate.

    The point of the indirection is the pristine tree.  A fixture written
    as ``pytest.raises(LoadBetweenJacksNotModelled)`` cannot even be
    *executed* on a tree where that name does not exist: it dies at the
    import, and an :class:`ImportError` is a red for the wrong reason
    (``17_1 §3``, ``16_1 §3``).  The claim this phase makes is not "a
    particular class is importable"; it is **"the driver answers a
    question it has no authority to answer, and answers it in silence"**.
    That claim is observable on both trees, with no new symbol named, and
    that is what this helper exposes.

    Returns
    -------
    tuple of (str, object)
        ``("returned", value)`` if :func:`_compile` completed, where
        *value* is whatever it produced; ``("raised", exc)`` if it did
        not, where *exc* is the exception instance.
    """
    try:
        return ("returned", _compile(tmp_path, history, name))
    except BaseException as exc:            # noqa: BLE001 - it IS the datum
        return ("raised", exc)


def _refusal(tmp_path, history, name="tl.yaml"):
    r"""
    Compile *history*, assert the driver **refused**, and hand back the
    refusal.

    The single assertion in here is the discriminating one for the whole
    of section 1: on the pristine Phase 17 tree it fails reporting what
    the driver returned instead, which is the fact this phase ratifies.
    Everything the callers assert afterwards is detail *of* a refusal
    that has already been established.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kind, out = _compile_outcome(tmp_path, history, name)

    assert kind == "raised", (
        "compile_combination RETURNED IN SILENCE on a history whose "
        "load falls between two jacking events while a tendon is "
        "stressed and not yet grouted: it produced "
        f"{type(out).__name__} (len={len(out)}) and emitted "
        f"{len(caught)} warning(s) {[type(w.message).__name__ for w in caught]}. "
        "An ungrouted tendon has no strain compatibility with the "
        "section, so a sectional model has no authority over this case; "
        "answering it debit-free is an answer, not an abstention.")
    return out


class TestLoadBetweenJacks:
    r"""
    The ratified boundary: declared, not silently answered.

    ``17_0 §4`` measured the alternative and its price: charging the
    interposed load as if it were a jack moves :math:`\sigma_p` by up to
    58.6 MPa, 4.6 % — while option 1 and option 3 both produce
    :math:`\max|\Delta\text{loss}| = 0` exactly.  The difference between
    them is not the number; it is that option 3 says so.
    """

    def test_the_interposed_load_is_refused_instead_of_answered(
            self, tmp_path):
        r"""
        **Discriminating**, and the one that carries section 1.

        Measured on the pristine Phase 17 tree with this exact history:
        ``compile_combination`` **returns** — a ``list`` of one stage,
        zero warnings.  The interposed load is folded into the cumulative
        demand and applied debit-free, so the ungrouted tendon feels
        nothing and nothing anywhere says so.  That silence is the
        defect; the exception class is only how the refusal is spelt.

        The fixture therefore names **no new symbol**: it observes the
        outcome, then reads the taxonomy off the object it caught.  A
        version written as ``pytest.raises(LoadBetweenJacksNotModelled)``
        was tried first and rejected — on the pristine tree it fails with
        ``ImportError``, which is a red for the wrong reason and proves
        only that a name is missing.
        """
        err = _refusal(tmp_path, _HIST_INTERPOSED)

        # The taxonomy, read off the caught object rather than imported.
        # It is load-bearing: a caller that catches ValueError to mean
        # "the engineer typed something wrong" must not catch this, and a
        # caller that catches NotImplementedError to mean "GenSec has no
        # authority here" must.
        assert isinstance(err, NotImplementedError), (
            "the refusal is not in the scope-boundary half of the "
            f"exception taxonomy: {type(err).__mro__}")
        assert not isinstance(err, ValueError), (
            "the refusal is classified as malformed input; it is not — "
            "a load applied mid-sequence is an ordinary construction "
            "situation and the document describing it is well formed")
        assert type(err).__name__ == "LoadBetweenJacksNotModelled"

    def test_the_refusal_names_the_tendon_and_the_two_jacking_events(
            self, tmp_path):
        r"""
        **Discriminating** through :func:`_refusal`, and the reason the
        exception carries attributes instead of only a message.

        The declared way out (``18_0``, ``declared_delta_sigma_p``) is
        **per tendon and per event**.  Whoever builds it must be able to
        ask the refusal *which* tendon and *which* two jacks were
        involved without re-deriving them from the timeline, so the
        identity is pinned here as an interface, not as a nicety.
        """
        err = _refusal(tmp_path, _HIST_INTERPOSED)

        # P2 is jacked *after* the load, so it is not open when the load
        # acts: the set is the tendons open at that instant, not all of
        # them.  Getting this wrong in the widening direction would make
        # the message name a tendon the boundary does not concern.
        assert err.tendons == ("P1",)
        assert err.load_index == 2
        assert err.load_demand == "G2"
        assert err.prev_stress_index == 1
        assert err.next_stress_index == 3
        assert err.combo_name == "SLS_end"

        # ... and in the message, which is what an engineer reads.
        msg = str(err)
        assert "P1" in msg
        assert "G2" in msg
        assert "5.10.3" in msg
        assert "construction_history[2]" in msg

    # -- regression guards: these pass on both trees, by construction --

    def test_guard_loads_before_the_first_jack_do_not_raise(
            self, tmp_path):
        r"""
        **Regression guard, not evidence.**  The semantics GenSec has
        always had: ``base_N``/``base_Mx``/``base_My`` are the state at
        the start of the sequence.
        """
        stages = _compile(tmp_path, _HIST_ORDINARY)
        assert stages

    def test_the_already_grouted_clause_is_exercised_on_the_predicate(
            self):
        r"""
        **Clause unit test — declared as such, not offered as evidence.**

        The third clause of the guard — *stressed and not yet grouted* —
        is exercised on the predicate rather than through the driver, and
        the reason is structural: a timeline of the shape ``stress P1 /
        grout P1 / load / stress P2`` cannot be compiled at all.  It
        lands on a **different and older** scope boundary, the
        intra-sequence bonded-stiffness case of
        ``6_8-WARNING_intra_sequence_bonded.md``, which
        :func:`~gensec.solver.posttension.solve_posttension_sequence`
        already refuses.  Driving the clause through the pipeline would
        measure that older refusal, not this one.

        What this test is worth, stated plainly: it pins the shape of a
        predicate this phase wrote, and on the pristine tree it fails by
        reporting that the predicate is **absent** — an existence claim,
        not a behavioural differential.  The behavioural evidence for
        section 1 is
        :meth:`test_the_interposed_load_is_refused_instead_of_answered`,
        and nothing here is a substitute for it (``16_1 §3``: a red for
        the wrong reason is not evidence, and neither is a green).
        """
        from gensec.solver.timeline import (ConstructionTimeline,
                                            TimelineEvent)

        guard = getattr(ConstructionTimeline,
                        "_check_load_between_jacks", None)
        assert guard is not None, (
            "ConstructionTimeline has no _check_load_between_jacks: the "
            "third clause has no predicate to be exercised on.  This is "
            "an existence claim about a guard, not evidence of the "
            "defect the guard exists for.")

        def ev(kind, payload, i):
            return TimelineEvent(kind, payload, i)

        # stress P1 / grout P1 / load / stress P2 — the open set is
        # empty when the load acts, so the boundary does not apply.
        prefix = [
            ev("stress", {"tendons": ["P1"], "sigma_p0": 1300.0}, 0),
            ev("grout", {"tendons": ["P1"]}, 1),
            ev("load", {"demand": "G2"}, 2),
            ev("stress", {"tendons": ["P2"], "sigma_p0": 1300.0}, 3),
        ]
        guard(prefix, "c")            # must return: nothing is open

        # The same prefix without the grout: now it does apply.  Caught
        # by taxonomy rather than by name, for the same reason as above.
        prefix_open = [prefix[0], prefix[2], prefix[3]]
        with pytest.raises(NotImplementedError) as exc:
            guard(prefix_open, "c")
        assert type(exc.value).__name__ == "LoadBetweenJacksNotModelled"
        assert exc.value.tendons == ("P1",)

    def test_guard_a_trailing_load_does_not_raise(self, tmp_path):
        r"""
        **Regression guard, not evidence.**  A load after the last jack
        is not *between* two jacks: no subsequent debit can be
        contaminated by it.
        """
        stages = _compile(tmp_path, _HIST_TRAILING)
        assert stages


# ======================================================================
#  2.  N-13 — the material name map is blind to tendons
# ======================================================================

def _named_section(mesh=40.0):
    r"""A section whose three materials all carry a declared ``name``."""
    b, h = 400.0, 800.0
    conc = concrete_from_class("C35/45", ls="S", loadtype='slow', TypeConc='R', NA='French', time=28)
    conc.name = "C35/45-deck"
    st = Steel(fyk=450.0, Es=200000.0)
    st.name = "B450C-top"
    ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                            Ep=195000.0, ls='F', NA='EC2', eps_ud_factor=0.9, diagram='horizontal')
    ps.name = "Y1860S7-strand"
    t = Tendon(y=100.0, x=b / 2, Ap=1500.0, material=ps,
               eps_pe=1200.0 / 195000.0, bonded=True, embedded=True,
               name="T1")
    sec = GenericSection(
        polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
        bulk_material=conc,
        rebars=[RebarLayer(y=h - 60.0, As=804.0, material=st, x=b / 2)],
        tendons=[t], mesh_size=mesh)
    return sec, conc, st, ps


class TestTendonMaterialName:
    r"""
    N-9 / N-1 class: a consumer downstream of the solver that enumerates
    two element classes of three.

    Phase 16 gave ``_decompose`` its tendon branch; the name map it reads
    was never extended, so the lookup misses on every tendon and falls
    back to the class name.
    """

    def test_the_name_map_registers_the_tendon_material(self):
        r"""**Discriminating.**  Pristine: the map holds 2 entries."""
        from gensec.solver.analysis import _build_material_name_map

        sec, conc, st, ps = _named_section()
        nm = _build_material_name_map(FiberSolver(sec))
        assert id(conc) in nm
        assert id(st) in nm
        assert id(ps) in nm, (
            "the tendon material is absent from the name map: every "
            "tendon component of the decomposition will report its "
            "class name")

    def test_the_decomposition_reports_the_declared_tendon_name(self):
        r"""
        **Discriminating.**  Pristine reports ``PrestressingSteel``
        while bulk and rebar report their declared names — the
        differential in which the only thing that moves is the element
        class.
        """
        from gensec.solver.analysis import AnalysisEngine

        sec, conc, st, ps = _named_section()
        res = AnalysisEngine(FiberSolver(sec)).analyze(-1.0e6, 200e6, 0.0)
        got = {c["type"]: c["material_name"] for c in res["components"]}
        assert got["bulk"] == conc.name
        assert got["rebar"] == st.name
        assert got["tendon"] == ps.name


# ======================================================================
#  3.  N-14 — the N-M sweep envelope is blind to tendons
# ======================================================================

def _strand_only_slab(mesh=25.0):
    r"""
    A pretensioned slab strip with strands and **no ordinary rebar** —
    the hollow-core / double-tee geometry.

    It is not a laboratory case: it is the section for which the
    envelope's blindness is total rather than partial, because with no
    rebar there is nothing else to widen the tensile end.
    """
    b, h = 1200.0, 200.0
    conc = concrete_from_class("C45/55", ls="S", loadtype='slow', TypeConc='R', NA='French', time=28)
    ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                            Ep=195000.0, ls='F', NA='EC2', eps_ud_factor=0.9, diagram='horizontal')
    tend = [Tendon(y=40.0, x=x, Ap=93.0, material=ps,
                   eps_pe=1100.0 / 195000.0, bonded=True, embedded=True,
                   name=f"S{i}")
            for i, x in enumerate((150., 350., 600., 850., 1050.))]
    return GenericSection(
        polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
        bulk_material=conc, rebars=[], tendons=tend, mesh_size=mesh)


class TestSweepEnvelope:
    r"""
    N-9 / N-2 class, on the sweep instead of the criterion.

    Phase 16 gave the *criterion* one home
    (:meth:`FiberSolver.element_limit_groups`).  The **domain** the
    criterion is applied over — the strain-plane envelope cached in
    :class:`NMDiagram` — still consulted bulk and rebar only, while its
    own docstring promised "all materials".
    """

    def test_the_envelope_reaches_the_tendon_limit(self):
        r"""
        **Discriminating.**  Pristine: ``eps_max_global`` is exactly
        ``0.0`` on a strand-only section — the tensile half of the sweep
        does not exist.
        """
        sec = _strand_only_slab()
        nm = NMDiagram(FiberSolver(sec))
        _emg, exg, _emb, _exb = nm._collect_strain_limits()

        binds_at = max(t.material.eps_max - t.eps_pe for t in sec.tendons)
        assert exg > 0.0, (
            "the tensile end of the strain sweep is bulk-and-rebar "
            "only; on a section with no rebar it collapses to zero")
        assert exg == pytest.approx(binds_at, rel=1e-12)

    def test_a_tendon_widens_the_envelope_on_a_reinforced_section(self):
        r"""
        **Discriminating.**  The rebar's own limit is 1.0e-2; the
        tendon binds at ~2.6e-2, so on an ordinary reinforced
        prestressed section the envelope must move too — the defect is
        not confined to the rebarless case.
        """
        sec, _c, _s, _p = _named_section()
        nm = NMDiagram(FiberSolver(sec))
        _emg, exg, _emb, _exb = nm._collect_strain_limits()
        rebar_max = max(r.material.eps_max for r in sec.rebars)
        tendon_max = max(t.material.eps_max - t.eps_pe
                         for t in sec.tendons)
        assert tendon_max > rebar_max          # the premise, stated
        assert exg == pytest.approx(tendon_max, rel=1e-12)

    def test_guard_a_section_without_tendons_is_untouched(self):
        r"""
        **Regression guard, not evidence.**  The control that makes the
        two above evidence of an omitted class rather than of a broken
        envelope: with no tendons the envelope is what it always was.
        """
        b, h = 400.0, 600.0
        conc = concrete_from_class("C35/45", ls="S", loadtype='slow', TypeConc='R', NA='French', time=28)
        st = Steel(fyk=450.0, Es=200000.0)
        sec = GenericSection(
            polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
            bulk_material=conc,
            rebars=[RebarLayer(y=60.0, As=804.0, material=st, x=b / 2)],
            tendons=[], mesh_size=40.0)
        nm = NMDiagram(FiberSolver(sec))
        emg, exg, _emb, _exb = nm._collect_strain_limits()
        assert exg == pytest.approx(max(conc.eps_max, st.eps_max))
        assert emg == pytest.approx(min(conc.eps_min, st.eps_min))
