r"""
Phase 35, task Z-D2: the polar-ductility path must bound its own
batch, and its docstring must describe the mechanism that exists.

The defect this module pins
---------------------------

``NMDiagram.generate_polar_curvature`` builds a flat array of
:math:`n_{\text{angles}} \cdot n_{\chi}` curvature configurations and
hands the whole of it to
:meth:`~gensec.solver.capacity.NMDiagram._vectorized_solve_N`, which
passes it unchunked to
:meth:`~gensec.solver.integrator.FiberSolver.integrate_batch`.  That
method materialises the strain field as a dense
``(n_configs, n_fibers)`` array of ``float64`` and holds three such
arrays live at once — strain, stress and fibre force — so the peak
allocation is

.. math::

   M = 3 \, n_{\text{configs}} \, n_{\text{fibers}} \cdot 8\ \text{bytes}

with **no upper bound**: it is set entirely by the input document.
On ``examples/example_prestress.yaml`` the defaults
(:math:`n_{\text{angles}} = 144` from ``io_yaml``,
:math:`n_{\chi} = 100` from the ``run`` verb, 20 000 fibres) give
:math:`M \approx 6.4` GiB, and the process is killed by the kernel
with ``SIGKILL`` — exit 137 — before it can produce a diagram.  The
same happens on ``examples/example_staged_injection.yaml``
(19 200 fibres, :math:`M \approx 6.2` GiB).

The second half of the defect is a declaration contradicted by its
object, which is the species of Z-1 and Z-2.  The docstring of
``generate_polar_curvature`` asserted that a batch exceeding an
"internal chunking threshold of ``integrate_batch``" would be "split
automatically by the solver".  ``integrate_batch`` has never carried
such a threshold.  The chunking it described exists — in
``_mega_batch_integrate``, on the resistance-surface path — which is
why the sentence reads as true to anyone who has seen that method and
is false where it stands.

What these tests assert, and why each one is here
-------------------------------------------------

1.  **The batch is bounded.**  ``_vectorized_solve_N`` never asks
    ``integrate_batch`` for more rows than the declared cap.  This is
    the assertion that is RED on the pristine tree: there, the first
    call carries the whole flat array.
2.  **The cap is a declared quantity, not a literal buried in a
    branch.**  The module must expose the element budget, so that the
    third test can move it and so that a future reader can find the
    one place where the number lives.
3.  **Chunking does not move a number.**  The same problem solved with
    the cap forced below the batch size and with the cap above it must
    return *bit-identical* results.  Without this, a chunking that
    quietly changed the Newton iteration — for instance by restarting
    the active set per chunk — would pass tests 1 and 2.
4.  **The docstring no longer names ``integrate_batch`` as the
    chunker.**  Prose is not usually a test subject, but here the
    prose *is* the defect: it is what told a reader the budget was
    already bounded.

Test 3 is the null control in the sense of `30_0` §5: it must be green
on both trees, because chunking is a change of memory schedule and not
of physics.  On the pristine tree it is green vacuously — no chunking
happens — and on the repaired tree it is green because the schedule is
exact.

Notes
-----
Written **before** the repair and seen RED on the pristine tree
(E33-1: the order is the measurement).  The section used here carries
12 000 fibres so that the cap computed from the declared budget lands
at 1 000 configurations, which a 12 x 100 polar scan exceeds by a
factor of 1.2 — enough to force at least two chunks while keeping the
peak allocation of the test itself around 300 MB.
"""

import numpy as np
import pytest

from gensec.geometry.fiber import RebarLayer
from gensec.geometry.section import RectSection
from gensec.materials import Concrete, Steel
from gensec.solver import capacity as _capacity
from gensec.solver.capacity import NMDiagram
from gensec.solver.integrator import FiberSolver


# ---------------------------------------------------------------------
#  The section under test
# ---------------------------------------------------------------------
#
#: 400 x 800 mm rectangle meshed 100 x 120 = 12 000 fibres.  The count
#: is chosen, not inherited: with the element budget declared by
#: :mod:`gensec.solver.capacity` it puts the configuration cap at
#: 1 000, and the scan below asks for 1 200.
_N_FIBERS_X = 100
_N_FIBERS_Y = 120
_N_FIBERS = _N_FIBERS_X * _N_FIBERS_Y

#: Angles and curvature steps for the scan.  Their product, 1 200,
#: must exceed the cap for test 1 to be able to fail.
_N_ANGLES = 12
_N_CHI = 100


def _build_section():
    """A doubly reinforced rectangular section, biaxially meshed.

    Returns
    -------
    gensec.geometry.section.RectSection
    """
    conc = Concrete(fck=30, gamma_c=1.5, long_term_reduction_factor=0.85)
    steel = Steel(fyk=450, gamma_s=1.15)
    rebars = [
        RebarLayer(x=60.0, y=60.0, As=314.0, material=steel),
        RebarLayer(x=340.0, y=60.0, As=314.0, material=steel),
        RebarLayer(x=60.0, y=740.0, As=314.0, material=steel),
        RebarLayer(x=340.0, y=740.0, As=314.0, material=steel),
    ]
    return RectSection(
        B=400.0,
        H=800.0,
        bulk_material=conc,
        rebars=rebars,
        n_fibers_y=_N_FIBERS_Y,
        n_fibers_x=_N_FIBERS_X,
    )


@pytest.fixture(scope="module")
def nm_gen():
    """An :class:`NMDiagram` over the section above."""
    sec = _build_section()
    assert sec.n_fibers == _N_FIBERS, (
        f"the test's own premise moved: expected {_N_FIBERS} fibres, "
        f"the section reports {sec.n_fibers}")
    return NMDiagram(FiberSolver(sec))


#: The cap the repair is expected to enforce, written here as well as
#: in the module under test.  Duplicating it is deliberate: if the test
#: read the cap only from the module, then on a tree where the module
#: declares nothing the test would fail with ``AttributeError`` — the
#: form of the answer, not the fact being measured (E30-3..E30-5).  With
#: the reference formula written down, the pristine tree fails on the
#: batch size, which is the defect.
_REFERENCE_BUDGET = 12_000_000
_REFERENCE_FLOOR = 500


def _expected_cap(n_fibers):
    """The configuration cap, from the module if declared, else local.

    Parameters
    ----------
    n_fibers : int

    Returns
    -------
    int
    """
    fn = getattr(_capacity, "max_configs_for", None)
    if fn is not None:
        return fn(n_fibers)
    return max(_REFERENCE_FLOOR, _REFERENCE_BUDGET // max(n_fibers, 1))


class _BatchSpy:
    """Records the row count of every ``integrate_batch`` call.

    Wraps the bound method rather than replacing it, so the numbers
    the solver returns are the real ones and only the sizes are
    observed.

    Attributes
    ----------
    sizes : list of int
        One entry per call, in call order.
    """

    def __init__(self, solver):
        self.sizes = []
        self._inner = solver.integrate_batch
        solver.integrate_batch = self

    def __call__(self, eps0, chi_x, chi_y):
        self.sizes.append(len(eps0))
        return self._inner(eps0, chi_x, chi_y)


# ---------------------------------------------------------------------
#  1. The batch is bounded
# ---------------------------------------------------------------------

def test_vectorized_solve_bounds_every_batch(nm_gen):
    r"""No single ``integrate_batch`` call exceeds the declared cap.

    This is the assertion the repair exists to satisfy.  It is stated
    over *every* call rather than the first, because a chunking that
    bounded only the initial iterate and then let the Newton
    perturbation pass the full array would leave the peak untouched.
    """
    cap = _expected_cap(nm_gen.solver.sec.n_fibers)
    assert _N_ANGLES * _N_CHI > cap, (
        "the scan no longer exceeds the cap, so this test could pass "
        "on an unchunked tree; raise _N_ANGLES or _N_CHI")

    spy = _BatchSpy(nm_gen.solver)
    try:
        nm_gen.generate_polar_curvature(
            0.0, n_angles=_N_ANGLES, n_chi=_N_CHI)
    finally:
        del nm_gen.solver.integrate_batch

    assert spy.sizes, "integrate_batch was never called: the scan did not run"
    over = [n for n in spy.sizes if n > cap]
    assert not over, (
        f"{len(over)} of {len(spy.sizes)} batches exceeded the cap of "
        f"{cap} configurations; largest was {max(spy.sizes)}")


# ---------------------------------------------------------------------
#  2. The cap is declared, and it is one number
# ---------------------------------------------------------------------

def test_the_element_budget_is_a_declared_quantity():
    """The budget and the cap derived from it are module-level.

    A cap written as a literal inside the solve loop cannot be moved
    by test 3 and cannot be found by a reader asking "how much memory
    does a batch cost".
    """
    budget = _capacity.MAX_BATCH_ELEMENTS
    assert isinstance(budget, int) and budget > 0

    # The cap must fall with the fibre count and never below the floor,
    # so that a very finely meshed section still integrates.
    assert _capacity.max_configs_for(1) >= _capacity.max_configs_for(10_000)
    assert _capacity.max_configs_for(10 ** 9) >= 1


# ---------------------------------------------------------------------
#  3. Null control: chunking moves no number
# ---------------------------------------------------------------------

def test_chunking_is_bit_identical_to_the_unchunked_solve(nm_gen):
    """Forcing chunks on a small problem changes nothing numerically.

    Green on both trees by construction: on the pristine tree the
    forced budget has no effect because nothing reads it.  Its value
    is on the repaired tree, where it separates "the batch was split"
    from "the batch was split and the Newton iteration was restarted".
    """
    n_angles, n_chi = 6, 20
    monkeyed = getattr(_capacity, "MAX_BATCH_ELEMENTS", None)

    ref = nm_gen.generate_polar_curvature(
        0.0, n_angles=n_angles, n_chi=n_chi)["chi_u"]

    if monkeyed is None:  # pristine tree: nothing to force
        pytest.skip("MAX_BATCH_ELEMENTS not declared on this tree")

    try:
        # A budget of one fibre-element puts the cap at its floor of a
        # single configuration, well below the 120 of the scan, so the
        # comparison is "120 chunks of 1" against "1 chunk of 120".
        _capacity.MAX_BATCH_ELEMENTS = 1
        forced = nm_gen.generate_polar_curvature(
            0.0, n_angles=n_angles, n_chi=n_chi)["chi_u"]
    finally:
        _capacity.MAX_BATCH_ELEMENTS = monkeyed

    assert np.array_equal(ref, forced), (
        "chunking moved a number: max |delta| = "
        f"{np.max(np.abs(ref - forced)):.3e} 1/mm")


# ---------------------------------------------------------------------
#  4. The docstring describes the mechanism that exists
# ---------------------------------------------------------------------

def test_the_docstring_does_not_credit_integrate_batch_with_chunking():
    """The prose that misled a reader is itself under test.

    ``integrate_batch`` allocates in one call and always has.  A
    docstring that sends the reader there for the memory bound is the
    reason the 6.4 GiB allocation was read as impossible.
    """
    doc = NMDiagram.generate_polar_curvature.__doc__ or ""
    lowered = doc.lower()
    assert "chunking threshold of :meth:`integrate_batch`" not in lowered
    assert "automatically split by the solver" not in lowered
    assert "max_configs_for" in doc or "MAX_BATCH_ELEMENTS" in doc, (
        "the docstring must name the quantity that actually bounds the "
        "batch, or the next reader will have to find it by measurement")
