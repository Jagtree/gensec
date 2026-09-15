r"""
Phase 34, task C (Z-14): one home for ``resolve_components``.

Two functions with the same body are two homes for one fact, which is
the defect this module pins.  ``gensec.solver.analysis`` carried a
module-level ``_resolve_components`` whose body was character-for-
character the body of
:meth:`gensec.solver.check.VerificationEngine.resolve_components`,
docstring included but for its ``Raises`` section.  Nothing read the
second one except ``analysis`` itself, at two call sites, so the
denominator of the deletion is two lines (E30-2: for a function one
intends to delete the denominator is who **uses** it, never who names
it).

What these tests assert, and why each one is here
-------------------------------------------------

The obvious test — "the duplicate is gone" — is necessary and not
sufficient: a duplicate can be removed by deleting the call sites too,
which would also make it pass while removing the feature.  So the file
pins three things at once:

1.  the second home does not exist as a module attribute;
2.  ``analysis`` **reaches** the canonical home, i.e. it imports the
    owning class rather than re-deriving the sum;
3.  the numbers are unchanged, checked by evaluating the canonical
    resolver against an independently written reference sum on a
    demand database that exercises the default factor, an explicit
    factor, a repeated reference and a negative one.

Point 3 is what makes the merge falsifiable rather than cosmetic.  A
merge that silently changed the factor convention — say by treating a
missing ``factor`` as ``0.0`` instead of ``1.0`` — would pass points 1
and 2 and fail here.

Notes
-----
Written **before** the merge and seen RED on the pristine tree, which
is the order E33-1 records as having been broken in the previous
phase.  On that tree the first two fail with ``AssertionError`` and the
third passes: the third is the **null control** and must be green on
both trees, because the canonical resolver is not what this task
changes.
"""

import numbers

import pytest

from gensec.solver import analysis as _analysis
from gensec.solver.check import VerificationEngine


# ---------------------------------------------------------------------
#  The demand database used by the numeric control.
# ---------------------------------------------------------------------

#: Forces in [N] and moments in [N·mm], as everywhere in GenSec.  The
#: values are deliberately not round in the same way as each other, so
#: that an accidental transposition of ``Mx`` and ``My`` cannot cancel.
_DEMAND_DB = {
    "G": {"N": -420.0e3, "Mx": -640.0e6, "My": 12.5e6},
    "Q": {"N": 0.0, "Mx": -520.0e6, "My": -3.25e6},
    "W": {"N": 31.0e3, "Mx": 7.5e6, "My": 88.125e6},
}


def _reference_sum(components):
    r"""
    Independently written reference for

    .. math::

        \mathbf{S} = \sum_i f_i \, \mathbf{d}_i .

    Written out longhand rather than by calling the object under test,
    which is the only way a numeric control can falsify anything.

    Parameters
    ----------
    components : list of dict
        Each entry carries ``ref`` and optionally ``factor``.

    Returns
    -------
    dict
        ``{"N": float, "Mx": float, "My": float}``.
    """
    out = {"N": 0.0, "Mx": 0.0, "My": 0.0}
    for comp in components:
        factor = float(comp["factor"]) if "factor" in comp else 1.0
        demand = _DEMAND_DB[comp["ref"]]
        for axis in ("N", "Mx", "My"):
            out[axis] = out[axis] + factor * float(demand[axis])
    return out


# ---------------------------------------------------------------------
#  1 — the second home is gone
# ---------------------------------------------------------------------

def test_analysis_carries_no_second_resolve_components():
    r"""
    ``gensec.solver.analysis`` exposes no ``_resolve_components``.

    RED on the pristine tree by ``AssertionError``: the attribute is
    there, at ``analysis.py:767``, with the body of the canonical
    resolver.
    """
    assert not hasattr(_analysis, "_resolve_components"), (
        "gensec.solver.analysis still defines `_resolve_components`, "
        "whose body duplicates "
        "VerificationEngine.resolve_components.  Two homes for one "
        "fact: the second one can drift from the first and nothing "
        "would say so."
    )


def test_analysis_source_names_the_canonical_resolver():
    r"""
    ``analysis`` reaches the canonical home rather than re-deriving.

    Asserted on the source text, because the alternative — deleting
    the duplicate *and* its two call sites — would satisfy the
    previous test while removing the behaviour.  The canonical name
    must appear, and the duplicated definition must not.
    """
    import inspect

    src = inspect.getsource(_analysis)
    assert "resolve_components" in src, (
        "gensec.solver.analysis names no resolver at all: the merge "
        "removed the duplicate and its call sites together."
    )
    assert "def _resolve_components" not in src, (
        "gensec.solver.analysis still *defines* a resolver of its "
        "own; it must call VerificationEngine.resolve_components."
    )
    assert "VerificationEngine" in src, (
        "gensec.solver.analysis does not name VerificationEngine, so "
        "it is not reaching the canonical home."
    )


# ---------------------------------------------------------------------
#  2 — the numeric null control: green on BOTH trees
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "components",
    [
        [{"ref": "G"}],
        [{"ref": "G", "factor": 1.35}],
        [{"ref": "G", "factor": 1.35}, {"ref": "Q", "factor": 1.5}],
        [{"ref": "G"}, {"ref": "G", "factor": -1.0}],
        [{"ref": "W", "factor": -0.6}, {"ref": "Q"}, {"ref": "G"}],
        [],
    ],
)
def test_canonical_resolver_matches_an_independent_sum(components):
    r"""
    The canonical resolver reproduces the reference sum exactly.

    Exact equality, not a tolerance: both sides accumulate the same
    products in the same order in IEEE-754 double, so any difference
    is a difference of convention rather than of rounding.  This is
    the **null control** — it must be green before and after the
    merge.
    """
    got = VerificationEngine.resolve_components(components, _DEMAND_DB)
    want = _reference_sum(components)
    assert set(got) == {"N", "Mx", "My"}
    for axis in ("N", "Mx", "My"):
        assert isinstance(got[axis], numbers.Real)
        assert got[axis] == want[axis], (
            f"resolve_components disagrees with the reference sum on "
            f"{axis}: {got[axis]!r} vs {want[axis]!r} for "
            f"{components!r}"
        )
# GENSEC-COMPLETE phase=34 utc=2026-09-06T10:33:16Z sha256=2215e9886b14349734fa637d0e63d82710f07279fde9745e62bb9e875187b102 body_lines=179
