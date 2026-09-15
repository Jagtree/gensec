r"""
Phase 34, D1b-3: the retrofit's YAML surface.

The defect these fixtures pin
-----------------------------
The physics of the retrofit landed in Phase 24.
:class:`~gensec.solver.posttension.PreexistingState` makes **declarable**
the facts GenSec cannot derive — at what stress an existing cable sits,
how strained the concrete around it already is, and the two clocks that
run from different origins — with eight required fields and no default
anywhere.

No document could reach it.  Measured at the opening of this phase:
``grep -c retrofit`` returns **0** on ``io_yaml.py`` and **0** on
``cli.py``; ``PreexistingState`` is constructed nowhere in ``src/`` and
nowhere in ``tests/``; and the three places ``timeline.py`` names the
retrofit are **all prose inside docstrings**.  The case was therefore
not refused — it was *inexpressible*, which is worse: a document
describing an existing member with new cables loaded, ran, and returned
the analysis of a **virgin** section, with the existing cable silently
demoted to "not yet stressed" and its 1500 mm² of steel at 1100 MPa
appearing neither as a demand action nor as a resisting element.

What the surface is
-------------------
A ``retrofit`` entry of ``construction_history``, constrained to be the
first one::

    construction_history:
      - retrofit:
          concrete:
            eps0:     -3.0e-4
            chi_x:     1.5e-7
            chi_y:     0.0
            x_ref:   200.0
            y_ref:   600.0
            age_days: 14600.0
          tendons:
            P_old: {sigma_pe: 1100.0, age_days: 14200.0}
      - stress: {tendons: [P_new1], sigma_p0: 1300.0}
      ...

Why the file is shaped this way
-------------------------------
Three groups, and the third is what makes the first two mean anything.

``TestReachability``
    The declared retrofit reaches a **verdict**: the pre-existing cable
    is grouted, carries a locked-in strain, and is named
    ``pre-existing``.  On the pristine tree these fail with
    ``AssertionError`` on an unknown event kind.
``TestRefusalsNameTheLever``
    Every refusal names the field, the tendon or the ordinal that is
    actually wrong, not merely the block.  On the pristine tree every
    one of these cases dies on the same generic *unknown event kind
    'retrofit'*, which does name the block and would read as a
    validated refusal — so the assertions are on the **lever**.
``TestTheVirginCaseIsUnchanged``
    The null control.  The same document without the ``retrofit``
    entry must load, run and produce the **same numbers** before and
    after.  Green on both trees, and it is what separates "the
    retrofit landed" from "the walk changed".

Notes
-----
Written **before** the repair and seen RED on a pristine tree rebuilt
from the bundle, which is the order E33-1 records as having been broken
in Phase 33.
"""

import math

import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.timeline_run import run_timeline


# ---------------------------------------------------------------------
#  The bench document
# ---------------------------------------------------------------------

#: A 400x1200 girder carrying three tendons at 150 / 200 / 250 mm.
#: ``P_old`` is the existing cable; ``P_new1`` and ``P_new2`` are jacked
#: and injected one at a time, which is the staged-injection history
#: Phase 22 made expressible.  Geometry and materials are the ones the
#: distributed ``example_staged_injection.yaml`` uses, so that a number
#: read here can be compared against a document a reader can open.
_HEAD = """
code: ec2
materials:
  conc: {type: concrete_ec2_gen1, fck: 40, ls: F, loadtype: slow, TypeConc: R, NA: French, time: 28}
  strand:
    type: prestressing_steel_ec2
    class: Y1860S7
    ls: F
    diagram: inclined
    NA: EC2
    eps_ud_factor: 0.9
section:
  B: 400
  H: 1200
  bulk_material: conc
  n_fibers_y: 240
  tendons:
    - {name: P_old,  y: 150, x: 200, Ap: 1500, material: strand,
       eps_pe: 0.0, bonded: true}
    - {name: P_new1, y: 200, x: 200, Ap: 1500, material: strand,
       eps_pe: 0.0, bonded: true}
    - {name: P_new2, y: 250, x: 200, Ap: 1500, material: strand,
       eps_pe: 0.0, bonded: true}
demands:
  - {name: G1, N_kN: -420, Mx_kNm: -640}
construction_history:
"""

_TAIL = """  - stress: {tendons: [P_new1], sigma_p0: 1300.0}
  - grout:  {tendons: [P_new1]}
  - stress: {tendons: [P_new2], sigma_p0: 1300.0}
  - grout:  {tendons: [P_new2]}
  - point:  service
combinations:
  - name: SLS_char
    at: service
    components: []
"""

#: The fully declared retrofit, written out rather than built from a
#: dict so that the fixtures test the **document**, which is the surface
#: under repair, and not a Python object that bypasses the parse.
_RETROFIT = """  - retrofit:
      concrete:
        eps0:     -3.0e-4
        chi_x:     1.5e-7
        chi_y:     0.0
        x_ref:   200.0
        y_ref:   600.0
        age_days: 14600.0
      tendons:
        P_old: {sigma_pe: 1100.0, age_days: 14200.0}
"""

#: Declared quantities, repeated here so that an expected value below
#: is derived from them rather than copied from a run.
_SIGMA_PE = 1100.0
_EP = 195000.0          # Y1860S7 under EN 1992-1-1
_EPS0 = -3.0e-4
_CHI_X = 1.5e-7
_Y_OLD, _Y_REF = 150.0, 600.0


def _write(tmp_path, body):
    r"""
    Write one bench document and return its path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test temporary directory.
    body : str
        The ``construction_history`` entries, already indented.

    Returns
    -------
    str
        Path of the written document.
    """
    path = tmp_path / "bench.yaml"
    path.write_text(_HEAD + body + _TAIL, encoding="utf-8")
    return str(path)


def _run(tmp_path, body):
    r"""Load and resolve one bench document."""
    return run_timeline(load_yaml(_write(tmp_path, body)))


def _refusal(tmp_path, body):
    r"""
    Return the refusal message a bench document produces.

    Returns
    -------
    str
        ``str(exc)`` of whatever was raised.

    Raises
    ------
    AssertionError
        If the document was **accepted**.  A silence is the defect
        this whole file is about, so it is named as a failure here
        rather than allowed to fall through as ``None``.
    """
    try:
        _run(tmp_path, body)
    except Exception as exc:                # noqa: BLE001 — outcome is data
        return str(exc)
    raise AssertionError(
        "the document was ACCEPTED; a refusal was expected.")


# ---------------------------------------------------------------------
#  1 — the declared retrofit reaches a verdict
# ---------------------------------------------------------------------

class TestReachability:
    r"""The declaration crosses the whole walk and changes numbers."""

    def test_the_existing_cable_is_in_the_resisting_section(
            self, tmp_path):
        r"""
        ``P_old`` is grouted, and named ``pre-existing``.

        Grouted **and** named: the first says it is a resisting
        element, the second says why — it did not become one through a
        ``grout`` event of this history, and a reader of the ledger
        must be able to tell the two apart.
        """
        res = _run(tmp_path, _RETROFIT)["resolution"]
        assert "P_old" in res.grouted, (
            "the declared pre-existing cable is not in the resisting "
            "set: it was declared, parsed, and then dropped.")
        assert res.pre_post.get("P_old") == "pre-existing", (
            f"P_old is recorded as {res.pre_post.get('P_old')!r}; a "
            f"cable bonded before the history began is neither 'pre' "
            f"nor 'post' tensioned in this history's sense.")

    def test_the_locked_in_strain_is_the_declared_one(self, tmp_path):
        r"""
        :math:`\varepsilon_{\mathrm{init}}` equals the declaration.

        .. math::

            \varepsilon_{\mathrm{init}}
                = \frac{\sigma_{pe}}{E_p}
                  - \left[\varepsilon_0
                          + \chi_x\,(y - y_{\mathrm{ref}})\right]

        Computed here from the declared numbers, not read back from a
        run: a test that compares an output to itself measures
        nothing.  The tolerance is 1e-12 relative, which is the
        instrument's own movement threshold.
        """
        res = _run(tmp_path, _RETROFIT)["resolution"]
        want = (_SIGMA_PE / _EP
                - (_EPS0 + _CHI_X * (_Y_OLD - _Y_REF)))
        got = res.grout_eps_init.get("P_old")
        assert got is not None, (
            "P_old carries no locked-in strain: it entered the "
            "resisting section with no datum at all.")
        assert math.isclose(got, want, rel_tol=1e-12), (
            f"locked-in strain {got!r} does not match the declared "
            f"state {want!r}.")

    def test_the_existing_cable_stiffens_the_later_jack(self, tmp_path):
        r"""
        The declaration **changes a published number**.

        A cable bonded before the sequence is a section element from
        the first jack on: it transforms the section the *next* jack is
        pulled against, and it is debited by the sollecitazione the
        walk carries.  So ``P_new2`` must not reconcile to the value it
        has on the same document with no ``retrofit`` entry.

        This is the test that separates "the block is parsed" from
        "the block is consumed".  A surface that validates a
        declaration and then discards it passes every other test in
        this class and fails this one.
        """
        with_ = _run(tmp_path, _RETROFIT)["resolution"]
        without = _run(tmp_path, "")["resolution"]
        a = with_.grout_eps_init["P_new2"]
        b = without.grout_eps_init["P_new2"]
        assert not math.isclose(a, b, rel_tol=1e-9), (
            f"P_new2 reconciles to {a!r} with the retrofit declared "
            f"and {b!r} without it: the declaration reached the parse "
            f"and not the physics.")

    def test_the_first_jack_is_untouched(self, tmp_path):
        r"""
        ``P_new1`` is **not** moved by the declaration.

        The first cable jacked takes no elastic-shortening debit from
        cables stressed after it, and the pre-existing cable is not
        stressed at all — it only stiffens the section.  So the
        movement in the test above must be at ``P_new2`` and not
        everywhere: a bench where every number moves cannot tell a
        repair from a perturbation.
        """
        with_ = _run(tmp_path, _RETROFIT)["resolution"]
        without = _run(tmp_path, "")["resolution"]
        assert math.isclose(with_.grout_eps_init["P_new1"],
                            without.grout_eps_init["P_new1"],
                            rel_tol=1e-12)


# ---------------------------------------------------------------------
#  2 — every refusal names the lever
# ---------------------------------------------------------------------

class TestRefusalsNameTheLever:
    r"""
    A refusal that does not name the lever is a different silence.

    On the pristine tree all six of these die on the same generic
    *unknown event kind* message, which names the block.  Asserting on
    the block would therefore pass before the repair; every assertion
    here is on the field, the tendon or the ordinal.
    """

    def test_a_missing_field_is_named(self, tmp_path):
        r"""``x_ref`` omitted: the refusal says ``x_ref``."""
        body = _RETROFIT.replace("        x_ref:   200.0\n", "")
        msg = _refusal(tmp_path, body)
        assert "x_ref" in msg, msg

    def test_an_unknown_field_is_named_with_the_valid_set(
            self, tmp_path):
        r"""
        A misspelt field is refused **by name**, and the message also
        prints the valid set: naming only the first says what went
        wrong, naming the second says what to write instead.
        """
        body = _RETROFIT.replace("        eps0:     -3.0e-4\n",
                                 "        eps0:     -3.0e-4\n"
                                 "        eps_0:    -3.0e-4\n")
        msg = _refusal(tmp_path, body)
        assert "eps_0" in msg, msg
        assert "chi_x" in msg and "age_days" in msg, (
            f"the refusal names the offending field but not the valid "
            f"set: {msg}")

    def test_an_unknown_tendon_is_named(self, tmp_path):
        r"""A declared cable the section does not carry."""
        body = _RETROFIT.replace(
            "        P_old: {sigma_pe: 1100.0, age_days: 14200.0}\n",
            "        P_old: {sigma_pe: 1100.0, age_days: 14200.0}\n"
            "        P_ghost: {sigma_pe: 900.0, age_days: 100.0}\n")
        msg = _refusal(tmp_path, body)
        assert "P_ghost" in msg, msg

    def test_a_cable_cannot_hold_two_roles(self, tmp_path):
        r"""
        Declared pre-existing **and** stressed by this same history.

        A cable was jacked before the history began or during it; it
        cannot be both, and nothing in the document says which
        declaration to believe.  The driver raises the same refusal one
        level down, but by then the message speaks of indices the
        writer never wrote.
        """
        body = _RETROFIT.replace(
            "        P_old: {sigma_pe: 1100.0, age_days: 14200.0}\n",
            "        P_old: {sigma_pe: 1100.0, age_days: 14200.0}\n"
            "        P_new1: {sigma_pe: 900.0, age_days: 100.0}\n")
        msg = _refusal(tmp_path, body)
        assert "P_new1" in msg, msg

    def test_a_retrofit_must_be_the_first_entry(self, tmp_path):
        r"""
        The position is a constraint, not a convention: an event
        before it would act on a member whose state has not been said.
        """
        body = ("  - load: {demand: G1}\n") + _RETROFIT
        msg = _refusal(tmp_path, body)
        assert "must be the first entry" in msg, msg

    def test_two_retrofits_are_refused_as_two_retrofits(self, tmp_path):
        r"""
        Two declarations are two homes for one fact.

        The diagnosis must be *two retrofits*, not *not first*: a
        second retrofit is also not first, so the ordinal check would
        answer it with the wrong reason.
        """
        msg = _refusal(tmp_path, _RETROFIT + _RETROFIT)
        assert "two 'retrofit' events" in msg, msg

    def test_a_non_positive_effective_stress_is_refused(self, tmp_path):
        r"""
        ``sigma_pe`` is the stress the cable is at **now**; a cable at
        zero or negative stress is not a pre-existing tendon.
        """
        body = _RETROFIT.replace("sigma_pe: 1100.0", "sigma_pe: 0.0")
        msg = _refusal(tmp_path, body)
        assert "sigma_pe" in msg, msg


# ---------------------------------------------------------------------
#  3 — the null control: green on BOTH trees
# ---------------------------------------------------------------------

class TestTheVirginCaseIsUnchanged:
    r"""
    A document that declares no retrofit is untouched by this phase.

    These two must be green before and after.  Without them, "the
    retrofit landed" and "the walk changed" are the same observable.
    """

    def test_the_document_without_a_retrofit_still_runs(self, tmp_path):
        out = _run(tmp_path, "")
        res = out["resolution"]
        assert sorted(res.grouted) == ["P_new1", "P_new2"]
        assert "P_old" not in res.grout_eps_init

    @pytest.mark.parametrize(
        "tendon,expected",
        [("P_new1", 0.006666666666666667),
         ("P_new2", 0.00734671839245194)],
    )
    def test_the_virgin_numbers_are_pinned(self, tmp_path, tendon,
                                           expected):
        r"""
        Pinned to the values measured on the **pristine** tree at the
        opening of this phase, so that a movement here is a regression
        and not a rounding.
        """
        res = _run(tmp_path, "")["resolution"]
        assert math.isclose(res.grout_eps_init[tendon], expected,
                            rel_tol=1e-12)


# ---------------------------------------------------------------------
#  4 — no distributed example declares one
# ---------------------------------------------------------------------

def test_no_distributed_example_declares_a_retrofit():
    r"""
    Zero distributed documents were modified by this phase.

    Held mechanically rather than by intention: if a later phase adds a
    ``retrofit`` to a shipped example, the numeric fingerprint of that
    example moves and this test says why.
    """
    import glob
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for path in sorted(glob.glob(os.path.join(root, "examples",
                                              "*.yaml"))):
        with open(path, encoding="utf-8") as fh:
            if "retrofit:" in fh.read():
                offenders.append(os.path.basename(path))
    assert offenders == [], (
        f"distributed example(s) {offenders} now declare a retrofit; "
        f"their fingerprints move for that reason and the ledger must "
        f"say so.")
# GENSEC-COMPLETE phase=37 utc=2026-09-11T05:30:51Z sha256=a5c8fe2db70cf7b7db0b18fd86496c6416bf1adfe873f2852ce0a911fc81307a body_lines=449
