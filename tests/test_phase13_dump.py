# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Phase 13 — the machine-readable dump and its projections.

What is pinned here, and why each pin exists rather than a comment:

**The version gate, in both directions.**  A dump with no
``schema_version`` is refused, not read as the current one; a document
declaring ``zone_rheology`` or ``tolerances`` with no version is refused
at parse (D7).  One number, two gates, so a run cannot be
half-versioned.

**Non-parallelism, structurally.**  The projections take one positional
parameter — the dict — *and* their module contains no import of
:mod:`gensec.solver` anywhere in its AST.  The second check is the one
that matters: a clean signature with a hidden import would be
parallelism with an alibi.

**Absences paired with nulls, in both directions.**  Every ``null`` has
a cause and every cause has a ``null``.  A register that may hold stale
rows is a register nobody reads.

**The closure contract.**  A locked-in strain closed by the single-pass
:func:`~gensec.solver.posttension.grout` is refused by the serialiser.
The fixture *exhibits* it — rule 3 — rather than asserting the guard
exists.

**Convergence as residual + tolerance.**  The flagship's topping datum
is the case in point: the solver's flag reads ``False`` after 100
iterations while the relative residual is :math:`1.6\times10^{-15}`.
The dump carries both, so the trap of ``12_12`` §2 is legible instead
of frozen.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys

import pytest

from gensec.io_yaml import load_yaml
from gensec.output import timeline_projections as projections
from gensec.output.timeline_dump import (
    SCHEMA_MAJOR, SCHEMA_NAME, SCHEMA_VERSION, DumpContractViolation,
    build_timeline_dump, dump_json, load_dump,
)
from gensec.solver.timeline import ConstructionTimeline

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EXAMPLES = os.path.join(_ROOT, "examples")

#: The flagship: post-tensioning, grouting, a composite cast, intervals
#: with integrated losses.  Every branch of the schema is reachable on
#: it, which is why it and not a miniature is the fixture.
FLAGSHIP = "example_composite_losses.yaml"

#: The timeline-only twin: a cast with an auto datum, no prestress.
TIMELINE_ONLY = "example_composite_timeline.yaml"


def _demand_db(document):
    r"""Build the SI demand database from a parsed document."""
    return {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                        "My": d.get("My", 0.0)}
            for d in document["demands"]}


def _build(stem):
    r"""
    Resolve an example and build its dump.

    Parameters
    ----------
    stem : str
        Example file name.

    Returns
    -------
    tuple
        ``(document, timeline, resolution, dump)``.
    """
    document = load_yaml(os.path.join(_EXAMPLES, stem))
    timeline = ConstructionTimeline.from_document(document)
    db = _demand_db(document)
    resolution = timeline.resolve(document["section"], db)
    dump = build_timeline_dump(
        timeline, resolution, document["section"], db,
        stem=os.path.splitext(stem)[0],
        combinations=document.get("combinations") or [])
    return document, timeline, resolution, dump


@pytest.fixture(scope="module")
def flagship():
    r"""The flagship dump, built once."""
    return _build(FLAGSHIP)


@pytest.fixture(scope="module")
def timeline_only():
    r"""The prestress-free timeline dump, built once."""
    return _build(TIMELINE_ONLY)


# ======================================================================
#  1. The version gate
# ======================================================================

class TestVersionGate:
    r"""``13_0`` §1 — the versioned schema is the contract."""

    def test_the_dump_declares_schema_and_version(self, flagship):
        _d, _t, _r, dump = flagship
        assert dump["schema"] == SCHEMA_NAME
        assert dump["schema_version"] == SCHEMA_VERSION

    def test_a_dump_without_a_version_is_refused(self):
        r"""
        Refused, **not** read as the current version.

        Assuming is how a format loses the ability to evolve: the first
        reader that guesses turns every later MAJOR bump into a silent
        misreading of every old file.
        """
        with pytest.raises(DumpContractViolation, match="schema_version"):
            load_dump({"schema": SCHEMA_NAME, "steps": []})

    def test_an_unknown_major_is_refused(self):
        with pytest.raises(DumpContractViolation, match="MAJOR"):
            load_dump({"schema": SCHEMA_NAME,
                       "schema_version": f"{SCHEMA_MAJOR + 1}.0"})

    def test_a_higher_minor_is_accepted(self):
        r"""MINOR means *a field was added*; an old reader stays right."""
        out = load_dump({"schema": SCHEMA_NAME,
                         "schema_version": f"{SCHEMA_MAJOR}.99",
                         "something_new": 1})
        assert out["schema_version"] == f"{SCHEMA_MAJOR}.99"

    def test_a_foreign_document_is_not_an_old_dump(self):
        with pytest.raises(DumpContractViolation, match="not a GenSec"):
            load_dump({"schema": "something.else",
                       "schema_version": "1.0"})

    def test_a_built_dump_survives_the_loader(self, flagship):
        _d, _t, _r, dump = flagship
        assert load_dump(dump_json(dump)) == dump


class TestDocumentVersionGate:
    r"""D7 — a hand-written rheology or tolerance needs a contract."""

    HEAD = (
        "code: ec2\n"
        "materials:\n"
        "  conc: {type: concrete_ec2_gen1, fck: 40, ls: F, loadtype: slow, TypeConc: R, NA: French, time: 28}\n"
        "section: {B: 300, H: 500, bulk_material: conc, n_fibers_y: 8}\n"
        "demands: [{name: G, N: -1.0, Mx: 0.0, My: 0.0}]\n"
    )

    def _write(self, tmp_path, body):
        path = tmp_path / "m.yaml"
        path.write_text(self.HEAD + body, encoding="utf-8")
        return str(path)

    def test_tolerances_without_a_version_are_refused(self, tmp_path):
        path = self._write(tmp_path,
                           "tolerances:\n  geometry: {area_rel: 1.0e-11}\n")
        with pytest.raises(ValueError, match="schema_version"):
            load_yaml(path)

    def test_tolerances_with_a_version_are_accepted(self, tmp_path):
        path = self._write(
            tmp_path,
            "schema_version: 1\n"
            "tolerances:\n  geometry: {area_rel: 1.0e-11}\n")
        assert load_yaml(path)["tolerances"].geometry_area_rel == 1e-11

    def test_an_ungated_document_needs_no_version(self, tmp_path):
        r"""D7 is not a tax on the past."""
        path = self._write(tmp_path, "")
        assert load_yaml(path)["section"] is not None

    def test_an_unknown_document_major_is_refused(self, tmp_path):
        path = self._write(
            tmp_path,
            "schema_version: 99\n"
            "tolerances:\n  geometry: {area_rel: 1.0e-11}\n")
        with pytest.raises(ValueError, match="MAJOR"):
            load_yaml(path)

    def test_the_shipped_gated_examples_declare_a_version(self):
        r"""
        The two examples that carry ``zone_rheology`` are versioned.

        Asserted rather than assumed: the gate is only worth having if
        the tree it ships with satisfies it.
        """
        for stem in (FLAGSHIP, TIMELINE_ONLY):
            with open(os.path.join(_EXAMPLES, stem), encoding="utf-8") as h:
                assert "schema_version:" in h.read(), stem


# ======================================================================
#  2. One source, and the projections that derive from it
# ======================================================================

PROJECTIONS = ("project_zones_csv", "project_tendons_csv",
               "project_steps_csv", "project_timeline_figure")


class TestProjectionsAreNotParallel:
    r"""
    ``13_0`` §13 — the dict is the sole input of every projection.

    This is the falsifiable form of *"the CSVs and the figure are
    projections, never built in parallel"*.  Declared, that sentence is
    unverifiable; these three tests make it break loudly if it stops
    being true.
    """

    @pytest.mark.parametrize("name", PROJECTIONS)
    def test_a_projection_takes_only_the_dump(self, name):
        signature = inspect.signature(getattr(projections, name))
        positional = [
            p for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert len(positional) == 1, (
            f"{name} takes {len(positional)} positional parameters; a "
            f"projection takes the dump and nothing else.")
        assert positional[0].name == "dump"

    def test_the_projections_module_cannot_reach_the_solver(self):
        r"""
        No import of :mod:`gensec.solver` anywhere in the module's AST.

        The signature check alone would pass on a module that reaches
        the resolution through an import inside a function body — a
        clean signature with a hidden import is parallelism with an
        alibi.  The AST sees both.
        """
        source = inspect.getsource(projections)
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                offenders += [a.name for a in node.names
                              if "solver" in a.name]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "solver" in module or (node.level and "solver" in
                                          str(node.names)):
                    offenders.append(module)
        assert not offenders, (
            f"the projections module imports {offenders}; it must be "
            f"unable to reach a TimelineResolution even by accident.")

    @pytest.mark.parametrize("name", PROJECTIONS)
    def test_a_projection_runs_on_a_reloaded_dump(self, flagship, name):
        r"""
        A projection works on a dump that went through JSON and back.

        The round trip is the point: anything the projection needed but
        the schema did not serialise would be missing here.
        """
        _d, _t, _r, dump = flagship
        reloaded = load_dump(dump_json(dump))
        assert getattr(projections, name)(reloaded) is not None


class TestRoundTrip:
    r"""JSON → projection → comparison, the closure gate."""

    @pytest.mark.parametrize("name", PROJECTIONS[:3])
    def test_csv_from_dict_equals_csv_from_json(self, flagship, name):
        _d, _t, _r, dump = flagship
        direct = getattr(projections, name)(dump)
        via_json = getattr(projections, name)(
            json.loads(dump_json(dump)))
        assert direct == via_json

    def test_the_json_is_byte_stable(self, flagship):
        r"""Two serialisations of one dump are byte-identical."""
        _d, _t, _r, dump = flagship
        assert dump_json(dump) == dump_json(dump)

    def test_the_json_carries_no_nan_or_infinity(self, flagship):
        r"""
        ``NaN`` and ``Infinity`` are not JSON.

        A file a strict parser rejects is not a machine-readable
        artefact; the unbounded domain bounds become ``null`` with a
        recorded cause instead.
        """
        _d, _t, _r, dump = flagship
        text = dump_json(dump)
        assert "NaN" not in text and "Infinity" not in text

    def test_the_content_digest_is_stable_across_processes(self):
        r"""
        The dump digest does not move with ``PYTHONHASHSEED``.

        The same property ``A7'`` measures, pinned here too so a plain
        ``pytest`` run catches a regression without waiting for the
        validator sweep.  Forced seeds, not the default randomisation:
        two random seeds may coincide, and a check that can pass by
        luck is not a check.
        """
        probe = (
            "from gensec.io_yaml import load_yaml;"
            "from gensec.solver.timeline import ConstructionTimeline;"
            "from gensec.output.timeline_dump import build_timeline_dump;"
            "d = load_yaml(%r);"
            "tl = ConstructionTimeline.from_document(d);"
            "db = {x['name']: {'N': x.get('N', 0.0),"
            " 'Mx': x.get('Mx', 0.0), 'My': x.get('My', 0.0)}"
            " for x in d['demands']};"
            "r = tl.resolve(d['section'], db);"
            "print(build_timeline_dump(tl, r, d['section'], db)"
            "['content_digest'])"
        ) % os.path.join(_EXAMPLES, FLAGSHIP)
        digests = []
        for seed in ("0", "12345"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            env["MPLBACKEND"] = "Agg"
            env["PYTHONPATH"] = (os.path.join(_ROOT, "src") + os.pathsep
                                 + env.get("PYTHONPATH", ""))
            out = subprocess.run([sys.executable, "-c", probe],
                                 capture_output=True, text=True,
                                 env=env, timeout=300)
            assert out.returncode == 0, out.stderr[-800:]
            digests.append(out.stdout.strip())
        assert digests[0] and digests[0] == digests[1]


# ======================================================================
#  3. Absences
# ======================================================================

def _nulls(node, path=""):
    r"""
    Yield the path of every ``None`` in a nested structure.

    Parameters
    ----------
    node : object
    path : str

    Yields
    ------
    str
    """
    if node is None:
        yield path
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _nulls(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _nulls(value, f"{path}[{index}]")


class TestAbsences:
    r"""``13_0`` §11 — a missing value is a row, never a default."""

    def test_every_absence_row_has_a_cause(self, flagship):
        _d, _t, _r, dump = flagship
        for row in dump["absences"]:
            assert row["path"] and row["cause"]
            assert row["cause"].endswith("."), (
                f"the cause for {row['path']} is not a sentence; a "
                f"register an engineer cannot act on is decoration.")

    def test_the_register_is_ordered(self, flagship):
        _d, _t, _r, dump = flagship
        paths = [row["path"] for row in dump["absences"]]
        assert paths == sorted(paths)

    def test_the_register_is_not_empty_on_the_flagship(self, flagship):
        r"""
        The flagship really does have absences.

        A register that is empty because nothing writes to it looks
        exactly like a register that is empty because nothing is
        missing.  Rule 3: the fixture must exhibit the thing.
        """
        _d, _t, _r, dump = flagship
        assert len(dump["absences"]) > 0

    def test_every_null_is_registered(self, flagship):
        r"""
        No ``null`` in the dump without a cause beside it.

        The direction that matters: a silent ``null`` inside a versioned
        contract is the same defect as a silent ``getattr`` default —
        a later reader cannot tell it from a measurement.
        """
        _d, _t, _r, dump = flagship
        registered = {row["path"] for row in dump["absences"]}
        body = {k: v for k, v in dump.items() if k != "absences"}
        unregistered = []
        for path in _nulls(body):
            normalised = path.split(".", 1)[-1] if path.startswith(
                ("zones.", "tendons.", "steps.", "combinations.",
                 "declared.")) else path
            if not any(normalised.startswith(r) or r.startswith(normalised)
                       or _same_slot(r, path) for r in registered):
                unregistered.append(path)
        assert not unregistered, (
            f"{len(unregistered)} null(s) with no recorded cause: "
            f"{unregistered[:8]}")


def _same_slot(registered: str, found: str) -> bool:
    r"""
    Whether a registered path and a found ``null`` name the same slot.

    The register uses the schema's reading order
    (``zones[1].datum.convergence``) while a naive walk of the built
    dict yields the structural order (``zones[1].datum.convergence``
    too, but ``steps[3].rheology.0.model`` against
    ``steps[3].rheology[0].model``).  Comparing on the bracket-stripped
    forms makes the two commensurable without weakening the check.

    Parameters
    ----------
    registered : str
    found : str

    Returns
    -------
    bool
    """
    def key(text):
        return (text.replace("[", ".").replace("]", "")
                .replace("..", ".").strip("."))
    return key(registered) == key(found)


# ======================================================================
#  4. Provenance
# ======================================================================

class TestProvenance:
    r"""``13_0`` §5, §6, §9.1 — where every number came from."""

    def test_every_tolerance_has_a_recorded_origin(self, flagship):
        _d, _t, _r, dump = flagship
        resolved = dump["tolerances"]["resolved"]
        provenance = dump["tolerances"]["provenance"]
        expected = {f"{group}.{key}"
                    for group, keys in resolved.items() for key in keys}
        assert set(provenance) == expected
        assert all(provenance.values())

    def test_a_document_tolerance_is_labelled_as_such(self, tmp_path):
        path = tmp_path / "m.yaml"
        path.write_text(
            "schema_version: 1\n"
            "tolerances:\n  geometry: {area_rel: 1.0e-11}\n"
            "code: ec2\n"
        "materials:\n"
            "  conc: {type: concrete_ec2_gen1, fck: 40, ls: F, loadtype: slow, TypeConc: R, NA: French, time: 28}\n"
            "section: {B: 300, H: 500, bulk_material: conc, n_fibers_y: 8}\n"
            "demands: [{name: G, N: -1.0, Mx: 0.0, My: 0.0}]\n",
            encoding="utf-8")
        tolerances = load_yaml(str(path))["tolerances"]
        document = tolerances.provenance_document()
        assert document["geometry.area_rel"] == "input document"
        assert document["datum.plane_rel"] == "shipped default"

    def test_a_provider_domain_carries_its_citation(self, flagship):
        _d, _t, _r, dump = flagship
        providers = dump["declared"]["providers"]
        assert providers, "the flagship declares loss models"
        for entry in providers.values():
            for quantity in entry["domain"].values():
                assert quantity["source"]
                assert "no citation declared" != quantity["source"]

    def test_a_declared_rule_carries_its_source(self, flagship):
        r"""
        The flagship declares the topping's linearity floor (12_8 §2.6).

        Rule 3 again: the assertion is worth nothing unless a rule is
        actually there to be serialised.
        """
        _d, _t, _r, dump = flagship
        rules = [rule
                 for entry in dump["declared"]["providers"].values()
                 for rule in entry["rules"].values()]
        assert rules, "the flagship declares at least one rule"
        for rule in rules:
            assert rule["source"].strip()
            assert rule["kind"] in ("constant", "callable",
                                    "interpolated")

    def test_declared_and_derived_ages_are_both_recorded(self, flagship):
        r"""
        Time-dependent provenance: an input against a consequence.

        On the flagship the base zone is declared at 28 d and reaches
        25 550 d by the last interval.  A run that does not say at what
        age it let the concrete flow is not reproducible, however exact
        its numbers are.
        """
        _d, _t, _r, dump = flagship
        last = dump["steps"][-1]["rheology"]["0"]
        assert last["age_declared"] == 28.0
        assert last["age_derived"] > last["age_declared"]
        assert last["declared_at"] == "zone_rheology"

    def test_the_derived_age_grows_only_across_intervals(self, flagship):
        r"""
        An instantaneous event does not age the concrete.

        The distinction is the whole reason the two ages are separate
        fields: only the derived one moves, and only when time passes.
        """
        _d, _t, _r, dump = flagship
        previous = None
        for step in dump["steps"]:
            entry = step["rheology"].get("0")
            if entry is None:
                continue
            age = entry["age_derived"]
            if previous is not None:
                if step["duration_days"] == 0.0:
                    assert age == previous, step["index"]
                else:
                    assert age > previous, step["index"]
            previous = age


# ======================================================================
#  5. Convergence, never a bare flag
# ======================================================================

class TestConvergence:
    r"""``13_0`` §7.1 — residual beside the tolerance it was judged by."""

    def test_a_datum_records_residual_and_tolerance(self, flagship):
        _d, _t, _r, dump = flagship
        cast = [z for z in dump["zones"] if z["cast_at_step"] is not None]
        assert cast, "the flagship casts a zone"
        for zone in cast:
            convergence = zone["datum"]["convergence"]
            for field in ("residual_force_rel", "residual_moment_rel",
                          "tolerance_force_rel", "tolerance_moment_rel"):
                assert convergence[field] is not None

    def test_accepted_is_recomputable_from_its_ingredients(self, flagship):
        r"""
        ``accepted`` is redundant on purpose, and must stay so.

        A boolean a reader can re-derive from the numbers beside it is
        a checkable assertion; an opaque one is a request for trust.
        """
        _d, _t, _r, dump = flagship
        for zone in dump["zones"]:
            convergence = (zone["datum"] or {}).get("convergence")
            if convergence is None:
                continue
            expected = bool(
                convergence["solver_flag"]
                or (convergence["residual_force_rel"]
                    < convergence["tolerance_force_rel"]
                    and convergence["residual_moment_rel"]
                    < convergence["tolerance_moment_rel"]))
            assert convergence["accepted"] is expected

    def test_the_flagship_no_longer_exhibits_the_absolute_flag_trap(
            self, flagship):
        r"""
        The trap of ``12_12`` §2 is no longer reproduced on this
        fixture, and the flag is still not trusted on its own.

        Until Phase 27 the topping datum came back exact -- a relative
        residual around :math:`10^{-15}` -- while the solver's absolute
        flag still read ``False`` after its iteration cap: the flag
        disagreed with the residual it was meant to summarise, which is
        why the schema serialises the residuals and the tolerances
        beside it rather than the flag alone.  With the service law in
        the datum view the iteration terminates on its own terms -- six
        iterations, force residual exactly 0, moment residual
        :math:`3.3\cdot10^{-17}` -- and the flag reads ``True``.

        The cause was **isolated, not assumed**.  Moving the topping
        from one day to three, which this phase had to do anyway, does
        not flip the flag: on the tree rebuilt from
        ``gensec_phase26.bundle`` with the age already at three days,
        the old assertion still passed.  The flip belongs to the
        service law.

        What this does *not* establish is that the absolute flag has
        become trustworthy in general: one fixture ceasing to misreport
        is evidence about one fixture.  The schema decision stands, and
        the test now pins the new agreement so that a future divergence
        between flag and residual is caught the way the old one was.
        """
        _d, _t, _r, dump = flagship
        topping = [z for z in dump["zones"] if z["name"] == "topping"][0]
        convergence = topping["datum"]["convergence"]
        assert convergence["accepted"] is True
        assert convergence["residual_force_rel"] < 1e-12
        assert convergence["residual_moment_rel"] < 1e-12
        # The point is the agreement between flag and residual, not the
        # value of either taken alone.
        assert convergence["solver_flag"] is True

    def test_an_unsolved_datum_is_an_absence_not_a_zero(self, timeline_only):
        r"""A zone never cast has no convergence, and says so."""
        _d, _t, _r, dump = timeline_only
        base = [z for z in dump["zones"] if z["index"] == 0][0]
        assert base["datum"] is None
        causes = {row["path"]: row["cause"] for row in dump["absences"]}
        assert "zones[0].datum" in causes


# ======================================================================
#  6. The grout closure contract
# ======================================================================

class TestGroutClosure:
    r"""``13_0`` §8.1 — the primitive stays, its residue does not."""

    def test_the_flagship_datum_is_self_consistent(self, flagship):
        _d, _t, _r, dump = flagship
        tendon = dump["tendons"][0]
        closure = tendon["closure"]
        assert closure["kind"] == "selfconsistent"
        assert closure["residual_rel"] < closure["tolerance_rel"]
        assert closure["iterations"] >= 1

    def test_the_single_pass_primitive_marks_itself(self):
        r"""
        ``grout()`` labels its own output, and is **not** deprecated.

        Deprecating it would be incoherent: ``grout_selfconsistent``
        calls it as its first iterate, so the warning would be one
        nobody could act on.  The marker is what makes the decision
        enforceable instead of merely written down.
        """
        from gensec.solver.posttension import grout
        source = inspect.getsource(grout)
        assert '"closure": {"kind": "single_pass"}' in source

    def test_a_single_pass_datum_is_refused_by_the_dump(self, flagship):
        r"""
        Rule 3: the fixture *exhibits* the violation.

        Asserting that a guard exists is not the same as measuring that
        it fires.  Here the ledger cell is set back to the single-pass
        marker and the serialiser must refuse.
        """
        _document, timeline, resolution, _dump = _build(FLAGSHIP)
        db = _demand_db(_document)
        touched = 0
        for key in list(resolution.ledger.records):
            cell = resolution.ledger.records[key]
            if cell.closure is not None:
                cell.closure = {"kind": "single_pass"}
                touched += 1
        assert touched, "the flagship grouts at least one tendon"
        with pytest.raises(DumpContractViolation, match="single_pass|O\\(n"):
            build_timeline_dump(timeline, resolution,
                                _document["section"], db)


# ======================================================================
#  7. The ledger is read, not reconstructed
# ======================================================================

class TestReadFromLedger:
    r"""``13_0`` §9.2 — the per-tendon action comes from Step 5."""

    def test_the_dump_never_calls_grout(self):
        r"""
        The serialiser holds no call to the post-tensioning primitives.

        Reconstructing :math:`P = \sigma_{p0} A_p \gamma_P` in the dump
        was finding F-C's shape: two independent computations of one
        quantity agree only until one of them is edited.
        """
        from gensec.output import timeline_dump
        source = inspect.getsource(timeline_dump)
        tree = ast.parse(source)
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        assert "grout" not in called
        assert "grout_selfconsistent" not in called
        assert "solve_posttension_sequence" not in called

    def test_the_tendon_state_is_carried_to_every_step(self, flagship):
        r"""
        A tendon bonded at step 2 is still bonded at the last step.

        Carried in the dump rather than left to each consumer, so the
        carry has one home; a dump that spoke only at the steps where
        the ledger holds a cell would have every reader re-derive it,
        differently, eventually.
        """
        _d, _t, _r, dump = flagship
        states = [(s["index"], s["tendons"]["P1"]["state"])
                  for s in dump["steps"] if "P1" in s["tendons"]]
        assert states[0][1] == "stressed"
        assert all(state == "bonded" for _i, state in states[1:])

    def test_the_single_side_invariant_is_checkable_on_the_dump(self):
        r"""
        :math:`A \oplus B` verifiable by the reader, not on trust.

        Checked on the **emission** block, which is where term
        :math:`B` lives: :math:`\gamma_P` belongs to a combination, so
        the demand-side actions are per ``(combination, point)`` and
        not a property of the timeline.  A resolution-only dump cannot
        express the invariant at all, which is why it records the
        demand side as *unknown* rather than as zero — see
        :meth:`test_the_demand_side_is_unknown_not_zero`.
        """
        document, timeline, resolution, _dump = _build(FLAGSHIP)
        db = _demand_db(document)
        for combo in document["combinations"]:
            timeline.compile_combination(combo, resolution,
                                         document["section"], db)
        dump = build_timeline_dump(
            timeline, resolution, document["section"], db,
            combinations=document["combinations"])

        checked = 0
        for combination in dump["combinations"]:
            emissions = combination["emissions"]
            assert emissions is not None, combination["name"]
            for point, by_step in emissions.items():
                for step, cells in by_step.items():
                    for name, cell in cells.items():
                        resistance = cell["state"] in ("bonded", "grouted")
                        demand = cell["live_actions"] > 0
                        assert resistance != demand, (
                            combination["name"], point, step, name)
                        checked += 1
        assert checked, "the flagship emits at least one action"

    def test_the_demand_side_is_unknown_not_zero(self, flagship):
        r"""
        A resolution-only dump says *unknown*, never ``0``.

        Writing ``0`` would read as "this tendon has no demand-side
        action", which is a different claim from "this level does not
        know".  That substitution is the same defect as a silent
        ``getattr`` default, wearing a number's clothes instead of an
        attribute's.
        """
        _d, _t, _r, dump = flagship
        cells = [cell for step in dump["steps"]
                 for cell in step["tendons"].values()]
        assert cells
        assert all(cell["live_actions"] is None for cell in cells)
        causes = {row["path"] for row in dump["absences"]}
        assert any(p.endswith(".live_actions") for p in causes)

    def test_two_unnamed_tendons_do_not_collapse_into_one_cell(self):
        r"""
        Finding F-N1, measured on behaviour rather than on source text.

        ``Tendon.name`` is a dataclass field defaulting to ``None``, so
        the old ``getattr(t, "name", f"T{i}")`` never fired its
        fallback: it returned ``None``, and the ledger keyed the cell
        under the literal string ``"None"``.  Two unnamed tendons
        therefore collapsed into one cell, and that key matched nothing
        in ``_tendon_name_index`` (which yields ``tendon_0`` /
        ``tendon_1``), so the substep rows and the state rows of one
        ledger were keyed differently and a join across them dropped in
        silence.

        Asserted on the ledger the walk actually produces.  A grep over
        the source would pass or fail on prose — including a comment
        *explaining* the defect — which is not a measurement.
        """
        from shapely.geometry import Polygon
        from gensec.geometry.fiber import Tendon
        from gensec.geometry.geometry import GenericSection
        from gensec.materials.codes import resolve_code
        from gensec.materials.concrete import Concrete
        from gensec.materials.steel import PrestressingSteel
        from gensec.solver.timeline import _tendon_name_index

        steel = PrestressingSteel(f_p01d=1391.3, Ep=195000.0)
        concrete = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
        # Stamped by hand because this fixture builds its section in
        # Python instead of loading a document.  Since Phase 21 there
        # is no default code beneath this: an unstamped material makes
        # the timeline decline to state a modulus, which is the whole
        # point and would make this fixture fail for a reason that has
        # nothing to do with tendon names.
        concrete.code = resolve_code(model_code="ec2")
        section = GenericSection(
            polygon=Polygon([(0, 0), (600, 0), (600, 800), (0, 800)]),
            bulk_material=concrete,
            rebars=[], mesh_size=80,
            tendons=[Tendon(x=200.0, y=100.0, Ap=750.0, material=steel),
                     Tendon(x=400.0, y=100.0, Ap=750.0, material=steel)])
        assert [t.name for t in section.tendons] == [None, None], (
            "the fixture must really carry unnamed tendons")

        canonical = _tendon_name_index(section)
        assert canonical == {"tendon_0", "tendon_1"}

        timeline = ConstructionTimeline.from_block([
            {"load": {"demand": "G"}},
            {"stress": {"tendons": sorted(canonical), "sigma_p0": 1250.0}},
            {"grout": {"tendons": sorted(canonical)}},
            {"point": "transfer"},
        ], code="ec2")
        db = {"G": {"N": -80e3, "Mx": -60e6, "My": 0.0}}
        resolution = timeline.resolve(section, db)

        substep_names = {name for _e, _s, name
                         in resolution.ledger.substeps}
        assert "None" not in substep_names, (
            "an unnamed tendon was recorded under the literal key "
            "'None' — F-N1")
        assert substep_names == canonical, (
            f"the driver photograph is keyed {sorted(substep_names)} "
            f"while the rest of the tree uses {sorted(canonical)}; a "
            f"join across the two would drop rows silently")
        assert set(resolution.ledger.tendons()) == canonical


# ======================================================================
#  8. Step content
# ======================================================================

class TestStepContent:
    r"""``13_0`` §9 — active zones, loads, rebars, tendons per step."""

    def test_one_entry_per_physical_event(self, flagship):
        _document, timeline, _r, dump = flagship
        assert len(dump["steps"]) == len(timeline.events)
        assert [s["index"] for s in dump["steps"]] == \
            list(range(len(timeline.events)))

    def test_points_appear_as_anchors_not_as_steps(self, flagship):
        _document, timeline, _r, dump = flagship
        anchored = {a for s in dump["steps"] for a in s["anchors"]}
        assert anchored == set(timeline.points)

    def test_a_zone_is_active_from_its_cast_onward(self, flagship):
        _d, _t, _r, dump = flagship
        topping = [z for z in dump["zones"] if z["name"] == "topping"][0]
        cast_at = topping["cast_at_step"]
        for step in dump["steps"]:
            active = topping["index"] in step["zones_active"]
            assert active == (step["index"] >= cast_at), step["index"]

    def test_a_rebar_is_inactive_before_its_zone_is_cast(self, flagship):
        _d, _t, _r, dump = flagship
        first = dump["steps"][0]
        by_zone = {}
        for bar in first["rebars"]:
            by_zone.setdefault(bar["zone"], []).append(bar["active"])
        assert by_zone, "the flagship carries rebars"
        for zone, flags in by_zone.items():
            assert all(f == (zone in first["zones_active"]) for f in flags)

    def test_the_cumulative_demand_only_grows_at_loads(self, flagship):
        _d, _t, _r, dump = flagship
        previous = None
        for step in dump["steps"]:
            current = step["loads"]["cumulative"]
            if previous is not None and not step["loads"]["applied_here"]:
                assert current == previous, step["index"]
            previous = current


# ======================================================================
#  9. The casting palette
# ======================================================================

class TestCastPalette:
    r"""``13_0`` §13.1 — colour encodes the timeline, not a magnitude."""

    def test_the_base_zone_is_ranked_at_t0(self, flagship):
        r"""
        Zone 0 is *in place at* :math:`T_0`, not *never cast*.

        It is never the target of a ``cast`` event because it is what
        the section already is — which is a position on the timeline,
        and the earliest one.  Colouring it as an absence would say the
        opposite of what is true.
        """
        _d, _t, _r, dump = flagship
        order = projections._cast_order(dump)
        assert order[0] == 0
        assert order[1] == 1

    def test_distinct_casts_get_distinct_colours(self, flagship):
        _d, _t, _r, dump = flagship
        order = projections._cast_order(dump)
        colours = {projections.cast_colour(v) for v in order.values()}
        assert len(colours) == len(set(order.values()))

    def test_a_never_cast_zone_is_not_given_a_rank(self):
        r"""Absence is not a later position in the sequence."""
        dump = {"zones": [{"index": 0, "cast_at_step": None},
                          {"index": 1, "cast_at_step": 3},
                          {"index": 2, "cast_at_step": None}]}
        order = projections._cast_order(dump)
        assert order[2] is None
        assert projections.cast_colour(None) == projections.UNCAST_COLOUR

    def test_the_section_drawing_accepts_cast_steps(self, flagship):
        import matplotlib
        matplotlib.use("Agg")
        from gensec.output.plots import plot_section
        document, _t, _r, dump = flagship
        figure = plot_section(
            document["section"], title="cast order",
            cast_steps=projections.cast_steps_from_dump(dump))
        labels = [t.get_text()
                  for t in figure.axes[0].get_legend().get_texts()]
        assert any("in place at T0" in label for label in labels)
        assert any("cast[" in label for label in labels)


# ======================================================================
# 10. The getattr inventory gate
# ======================================================================

class TestGetattrInventory:
    r"""No substituting ``getattr`` may sit on the dump path."""

    def test_the_gate_is_clean(self):
        r"""
        Run the inventory as the gate, in-process.

        A silent default inside a versioned contract is permanent: a
        later reader cannot tell it from a measurement.
        """
        sys.path.insert(0, _ROOT)
        try:
            from tools.getattr_inventory import walk
        finally:
            sys.path.pop(0)
        findings = [s for s in walk()
                    if s.kind == "substituting" and s.on_dump_path]
        assert not findings, [str(s) for s in findings]
