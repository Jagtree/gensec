# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.
#
# GenSec is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# GenSec is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GenSec.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
Closed-form validation of the bulk imposed-strain kernel change.

This is the gate for the Phase-5 integrator change that makes
``bulk_eps_init`` *bite* on resistance (see
``7_1-GENSEC_PHASE5_BULK_KERNEL_PATCH.md``).  It is law-agnostic: every
expectation is built from the section's own ``bulk_material`` evaluated
at the offset argument, so the checks hold for any constitutive law the
section is configured with, not just one parabola-rectangle instance.

Physics being verified
----------------------
A uniform bulk offset :math:`\delta` shifts the argument of the bulk
constitutive law from the kinematic section strain :math:`\varepsilon_b`
to :math:`\varepsilon_b + \delta` at **every** integration site.  The
unambiguous signature of a genuine imposed strain (coazione) is a
**non-zero internal force at zero kinematic strain**:

.. math::

    N(\varepsilon_b=0,\ \chi=0;\ \delta)
        = A_c \, \sigma_b(\delta) \neq 0 ,

i.e. the self-equilibrated autotension/autocompression that an imposed
strain produces with no external load.  Equivalently, the entire
pure-axial response translates by :math:`-\delta` along the strain axis:
:math:`N(\varepsilon_b;\delta) = N(\varepsilon_b+\delta;0)`.  The strain
at which :math:`N=0` (and, on the compression branch, the strain at
which the squash plateau is reached — the "pure-compression intercept")
moves by :math:`-\delta`, in the predicted direction.

Sign note
---------
The kernel is sign-mechanical: it *adds* the offset, exactly as a bonded
tendon's ``eps_pe`` adds to the section strain.  Whether the physical
free shrinkage :math:`\varepsilon_{cs}` maps to ``+`` or ``-`` this
offset is a **provider** decision (the EC2 shrinkage generator), fixed
and validated downstream against the Annex B worked example — NOT here.
This script only proves the kernel reads the offset consistently at all
sites.

Run from the project root, with the package installed / importable,
**after** applying the kernel patch.  Exit code 0 == all checks pass.
"""

import sys
import numpy as np

from gensec.materials import Concrete
from gensec.geometry import RectSection
from gensec.solver import FiberSolver


TOL_N = 1.0e-3      # [N]  — machine-precision band on forces
TOL_M = 1.0e-3      # [N*mm]
TOL_K = 1.0e-3      # [N]  — tangent K[0,0]


def _build_plain_section(delta):
    """Plain concrete rectangle (no rebars/tendons), offset ``delta``.

    The offset is set post-construction exactly as ``io_yaml`` sets it
    (``section.bulk_eps_init = ...``), so the solver picks it up through
    its ``getattr(section, 'bulk_eps_init', 0.0)`` read.
    """
    concrete = Concrete(fck=30.0)
    # Plain concrete: no rebars, no tendons. ``rebars`` is a required
    # positional on RectSection, so pass an empty list explicitly.
    sec = RectSection(B=300.0, H=600.0, bulk_material=concrete,
                      rebars=[], n_fibers_y=200, n_fibers_x=1)
    sec.bulk_eps_init = float(delta)
    return sec, concrete


def _report(name, ok, detail=""):
    flag = "PASS" if ok else "FAIL"
    line = f"  [{flag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def check_offset(delta):
    r"""Run every patched site for a single offset ``delta``."""
    sec, concrete = _build_plain_section(delta)
    sv = FiberSolver(sec)
    A = float(np.sum(sec.A_fibers))
    nfib = int(sec.n_fibers)
    ok = True

    def sigma(eps):
        return float(concrete.stress_array(np.array([float(eps)]))[0])

    def tangent(eps):
        return float(concrete.tangent_array(np.array([float(eps)]))[0])

    # --- 1. integrate(): autotension at zero kinematic strain ----------
    N, Mx, My = sv.integrate(0.0, 0.0, 0.0)
    exp_N0 = sigma(delta) * A
    ok &= _report(
        "integrate: autotension N(0,0) == A*sigma(delta)",
        abs(N - exp_N0) < TOL_N and abs(Mx) < TOL_M and abs(My) < TOL_M,
        f"N={N:.6g}, expected={exp_N0:.6g}",
    )

    # --- 2. integrate(): pure-axial translation N(eb)=A*sigma(eb+d) ----
    trans_ok = True
    for eb in (-0.0008, -0.0003, 0.0, 0.0002):
        N, Mx, My = sv.integrate(eb, 0.0, 0.0)
        exp = sigma(eb + delta) * A
        trans_ok &= (abs(N - exp) < TOL_N
                     and abs(Mx) < TOL_M and abs(My) < TOL_M)
    ok &= _report("integrate: axial response translates by -delta",
                  trans_ok)

    # --- 3. integrate_with_tangent(): stress AND tangent at offset -----
    N, Mx, My, K = sv.integrate_with_tangent(0.0, 0.0, 0.0)
    exp_N = sigma(delta) * A
    exp_K00 = tangent(delta) * A
    ok &= _report(
        "integrate_with_tangent: N and K[0,0] at offset argument",
        abs(N - exp_N) < TOL_N and abs(K[0, 0] - exp_K00) < TOL_K,
        f"K00={K[0, 0]:.6g}, expected={exp_K00:.6g}",
    )

    # --- 4. integrate_batch(): vectorised, same offset -----------------
    ebs = np.array([-0.0008, -0.0003, 0.0, 0.0002])
    zeros = np.zeros_like(ebs)
    Nb, Mxb, Myb = sv.integrate_batch(ebs, zeros, zeros)
    exp_b = np.array([sigma(e + delta) * A for e in ebs])
    ok &= _report("integrate_batch: vector N == A*sigma(eb+delta)",
                  np.allclose(Nb, exp_b, atol=TOL_N)
                  and np.allclose(Mxb, 0.0, atol=TOL_M)
                  and np.allclose(Myb, 0.0, atol=TOL_M))

    # --- 5. get_fiber_results(): reported bulk sigma at offset ---------
    m = sv.get_fiber_results(0.0, 0.0, 0.0)
    sig_field = np.asarray(m["bulk"]["sigma"])
    exp_field = np.full(nfib, sigma(delta))
    ok &= _report("measure: bulk sigma field == sigma(delta)",
                  np.allclose(sig_field, exp_field, atol=1e-6))

    # --- 6. strains_within_limits(): admissibility on offset strain ----
    # Choose eb admissible on its own but driven out of the compression
    # limit by the offset (delta < 0) — proves the check reads delta.
    e_min = float(concrete.eps_min)
    e_max = float(concrete.eps_max)
    # eb inside [e_min, e_max] but eb + delta below e_min:
    eb_out = e_min - delta + 0.5 * delta if delta < 0 else e_min - delta - 1e-4
    # Robust construction independent of delta sign:
    eb_inside = 0.5 * (e_min + e_max) - delta      # eb+delta == midpoint
    eb_below = e_min - delta - 1e-4                 # eb+delta just below e_min
    adm_inside = sv.strains_within_limits(eb_inside, 0.0, 0.0)
    adm_below = sv.strains_within_limits(eb_below, 0.0, 0.0)
    ok &= _report(
        "strains_within_limits: limit applies to eb+delta",
        (adm_inside is True) and (adm_below is False),
        f"inside={adm_inside}, below_limit={adm_below}",
    )

    return ok


def check_zero_field_identity():
    r"""Offset 0.0 must reproduce the bare (pre-change) bulk response.

    This is the regression guarantee: a section that does not declare a
    bulk pre-strain is bit-identical to before the kernel change.  Here
    we assert it against the closed form ``A*sigma(eb)`` directly.
    """
    sec, concrete = _build_plain_section(0.0)
    sv = FiberSolver(sec)
    A = float(np.sum(sec.A_fibers))
    ok = True
    for eb in (-0.001, -0.0005, 0.0, 0.0003):
        N, Mx, My = sv.integrate(eb, 0.0, 0.0)
        exp = float(concrete.stress_array(np.array([eb]))[0]) * A
        ok &= (abs(N - exp) < TOL_N and abs(Mx) < TOL_M
               and abs(My) < TOL_M)
    return _report("zero-field identity: offset 0.0 == bare response", ok)


def main():
    print("Bulk imposed-strain kernel — closed-form validation")
    print("=" * 56)
    all_ok = True

    print("\nZero-field regression")
    all_ok &= check_zero_field_identity()

    for delta in (-0.0006, -0.0002, 0.00015):
        print(f"\nOffset delta = {delta:+.5f}")
        all_ok &= check_offset(delta)

    print("\n" + "=" * 56)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
