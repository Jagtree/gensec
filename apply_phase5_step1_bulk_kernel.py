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
Apply the Phase-5 / Step-1 bulk imposed-strain kernel change.

This patcher edits ``integrator.py`` so the bulk constitutive law is
evaluated at ``eps + bulk_eps_init`` at every site (see
``7_1-GENSEC_PHASE5_BULK_KERNEL_PATCH.md``).  It is **fail-loud**: each
edit asserts its anchor text occurs *exactly once*; on 0 or >1 matches it
refuses and reports, rather than corrupting the most test-covered file in
the package.  It is **idempotent**: an already-applied edit is detected
(its replacement is present) and skipped.  Line endings are preserved
(CRLF stays CRLF).

Ordering is encoded:
  * default run applies ONLY the integrator change;
  * ``--lift-guards`` removes the two ``io_yaml`` no-silent-no-op raises,
    and must be run SEPARATELY, only AFTER ``run_bulk_prestrain_validation.py``
    passes and the test suite stays green.

Usage
-----
    python apply_phase5_step1_bulk_kernel.py  <path-to-gensec-src>
    python apply_phase5_step1_bulk_kernel.py  <path-to-gensec-src> --check
    python apply_phase5_step1_bulk_kernel.py  <path-to-gensec-src> --lift-guards

``<path-to-gensec-src>`` is the directory containing ``integrator.py`` and
``io_yaml.py`` (e.g. ``src/gensec/solver`` — adjust if your layout differs;
the script locates each file by name under the given root, recursively).
"""

import argparse
import os
import sys


class EditError(RuntimeError):
    pass


def _apply_edits(path, edits, check_only):
    r"""Apply ``(label, old, new)`` edits to one file, fail-loud.

    Reads bytes, normalises to ``\n`` for matching, restores the original
    newline on write.  Returns a list of (label, status) tuples.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    if crlf:
        text = text.replace("\r\n", "\n")

    results = []
    for label, old, new in edits:
        n = text.count(old)
        if n == 1:
            text = text.replace(old, new)
            results.append((label, "apply"))
        elif n == 0:
            if new in text:
                results.append((label, "skip (already applied)"))
            else:
                raise EditError(
                    f"{os.path.basename(path)} :: {label}\n"
                    f"  anchor text NOT FOUND and replacement NOT present.\n"
                    f"  The file differs from the version this patch was\n"
                    f"  written against — refusing to guess. No file written."
                )
        else:
            raise EditError(
                f"{os.path.basename(path)} :: {label}\n"
                f"  anchor text found {n} times (expected 1) — ambiguous.\n"
                f"  Refusing to apply. No file written."
            )

    if not check_only:
        out = text.replace("\n", "\r\n") if crlf else text
        with open(path, "wb") as fh:
            fh.write(out.encode("utf-8"))
    return results


def _find(root, name):
    for dirpath, _dirs, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under {root!r}")


# ===========================================================================
#  integrator.py — the kernel change (norm-invariant)
# ===========================================================================

INTEGRATOR_EDITS = [
    # --- E1: FiberSolver.__init__ — the two offset attributes ----------
    (
        "__init__: bulk offset attributes",
        "        # Build bulk material groups for multi-material support\n"
        "        self._bulk_groups = self._build_bulk_groups()\n"
        "        self._is_multi_material = len(self._bulk_groups) > 1\n",
        "        # Build bulk material groups for multi-material support\n"
        "        self._bulk_groups = self._build_bulk_groups()\n"
        "        self._is_multi_material = len(self._bulk_groups) > 1\n"
        "\n"
        "        # ---- Bulk imposed-strain offset (coazione) ----\n"
        "        # Uniform locked-in bulk strain (shrinkage / differential\n"
        "        # thermal), read through getattr so sections built before the\n"
        "        # prestress phase are bit-identical (offset 0.0 -> argument\n"
        "        # unchanged everywhere).  The bulk law is evaluated at\n"
        "        # (eps + offset), exactly as a bonded tendon's law is\n"
        "        # evaluated at (eps_section + eps_pe).\n"
        "        self._bulk_eps_init = float(\n"
        "            getattr(self.sec, 'bulk_eps_init', 0.0))\n"
        "        # Per-group offsets, index-aligned with self._bulk_groups.\n"
        "        # Today every group carries the single section-level value;\n"
        "        # differential shrinkage per zone will populate distinct\n"
        "        # values here WITHOUT touching the kernel sites again.\n"
        "        self._bulk_eps_by_group = [self._bulk_eps_init\n"
        "                                   for _ in self._bulk_groups]\n",
    ),
    # --- E2: integrate() — bulk contribution ---------------------------
    (
        "integrate: bulk stress",
        "        # ---- Bulk contribution ----\n"
        "        if not self._is_multi_material:\n"
        "            # Fast path: single material on all fibers\n"
        "            sb = self.sec.bulk_material.stress_array(eb)\n"
        "        else:\n"
        "            # Multi-material: evaluate each group separately\n"
        "            sb = np.zeros_like(eb)\n"
        "            for mat, idx in self._bulk_groups:\n"
        "                sb[idx] = mat.stress_array(eb[idx])\n",
        "        # ---- Bulk contribution ----\n"
        "        if not self._is_multi_material:\n"
        "            # Fast path: single material on all fibers\n"
        "            sb = self.sec.bulk_material.stress_array(\n"
        "                eb + self._bulk_eps_init)\n"
        "        else:\n"
        "            # Multi-material: evaluate each group at its own offset\n"
        "            sb = np.zeros_like(eb)\n"
        "            for (mat, idx), eps_b in zip(self._bulk_groups,\n"
        "                                         self._bulk_eps_by_group):\n"
        "                sb[idx] = mat.stress_array(eb[idx] + eps_b)\n",
    ),
    # --- E3: integrate() — displaced bulk at rebars --------------------
    (
        "integrate: displaced bulk at rebars",
        "        for mat, bulk_mat, idx in self._rebar_groups:\n"
        "            er_g = er[idx]\n"
        "            s_rebar = mat.stress_array(er_g)\n"
        "            sb_at_rebars = bulk_mat.stress_array(er_g)\n"
        "            a = self.sec.A_rebars[idx]\n",
        "        for mat, bulk_mat, idx in self._rebar_groups:\n"
        "            er_g = er[idx]\n"
        "            s_rebar = mat.stress_array(er_g)\n"
        "            sb_at_rebars = bulk_mat.stress_array(\n"
        "                er_g + self._bulk_eps_init)\n"
        "            a = self.sec.A_rebars[idx]\n",
    ),
    # --- E4: integrate_with_tangent() — bulk stress + tangent ----------
    (
        "integrate_with_tangent: bulk stress + tangent",
        "        # ---- Bulk: stress and tangent ----\n"
        "        if not self._is_multi_material:\n"
        "            sb = self.sec.bulk_material.stress_array(eb)\n"
        "            Et_b = self.sec.bulk_material.tangent_array(eb)\n"
        "        else:\n"
        "            sb = np.zeros_like(eb)\n"
        "            Et_b = np.zeros_like(eb)\n"
        "            for mat, idx in self._bulk_groups:\n"
        "                sb[idx] = mat.stress_array(eb[idx])\n"
        "                Et_b[idx] = mat.tangent_array(eb[idx])\n",
        "        # ---- Bulk: stress and tangent ----\n"
        "        if not self._is_multi_material:\n"
        "            sb = self.sec.bulk_material.stress_array(\n"
        "                eb + self._bulk_eps_init)\n"
        "            Et_b = self.sec.bulk_material.tangent_array(\n"
        "                eb + self._bulk_eps_init)\n"
        "        else:\n"
        "            sb = np.zeros_like(eb)\n"
        "            Et_b = np.zeros_like(eb)\n"
        "            for (mat, idx), eps_b in zip(self._bulk_groups,\n"
        "                                         self._bulk_eps_by_group):\n"
        "                sb[idx] = mat.stress_array(eb[idx] + eps_b)\n"
        "                Et_b[idx] = mat.tangent_array(eb[idx] + eps_b)\n",
    ),
    # --- E5: integrate_with_tangent() — displaced bulk at rebars -------
    (
        "integrate_with_tangent: displaced bulk at rebars",
        "            sb_at_rebars = bulk_mat.stress_array(er_g)\n"
        "            Et_bulk_r = bulk_mat.tangent_array(er_g)\n",
        "            sb_at_rebars = bulk_mat.stress_array(\n"
        "                er_g + self._bulk_eps_init)\n"
        "            Et_bulk_r = bulk_mat.tangent_array(\n"
        "                er_g + self._bulk_eps_init)\n",
    ),
    # --- E6: integrate_batch() — bulk stress (2D) ----------------------
    (
        "integrate_batch: bulk stress",
        "        # Bulk stresses: (n, n_fibers)\n"
        "        if not self._is_multi_material:\n"
        "            sb = self.sec.bulk_material.stress_array(eb)\n"
        "        else:\n"
        "            sb = np.zeros_like(eb)\n"
        "            for mat, idx in self._bulk_groups:\n"
        "                sb[:, idx] = mat.stress_array(eb[:, idx])\n",
        "        # Bulk stresses: (n, n_fibers)\n"
        "        if not self._is_multi_material:\n"
        "            sb = self.sec.bulk_material.stress_array(\n"
        "                eb + self._bulk_eps_init)\n"
        "        else:\n"
        "            sb = np.zeros_like(eb)\n"
        "            for (mat, idx), eps_b in zip(self._bulk_groups,\n"
        "                                         self._bulk_eps_by_group):\n"
        "                sb[:, idx] = mat.stress_array(eb[:, idx] + eps_b)\n",
    ),
    # --- E7: integrate_batch() — displaced bulk at rebars --------------
    (
        "integrate_batch: displaced bulk at rebars",
        "        for mat, bulk_mat, idx in self._rebar_groups:\n"
        "            er_g = er[:, idx]\n"
        "            s_rebar = mat.stress_array(er_g)\n"
        "            sb_at_rebars = bulk_mat.stress_array(er_g)\n",
        "        for mat, bulk_mat, idx in self._rebar_groups:\n"
        "            er_g = er[:, idx]\n"
        "            s_rebar = mat.stress_array(er_g)\n"
        "            sb_at_rebars = bulk_mat.stress_array(\n"
        "                er_g + self._bulk_eps_init)\n",
    ),
    # --- E8a: _tendon_forces() — displaced bulk, batch branch ----------
    (
        "_tendon_forces: displaced bulk (batch)",
        "                s_tendon = mat.stress_array(e_tot_g)\n"
        "                s_bulk = bulk_mat.stress_array(e_sec_g)\n",
        "                s_tendon = mat.stress_array(e_tot_g)\n"
        "                s_bulk = bulk_mat.stress_array(\n"
        "                    e_sec_g + self._bulk_eps_init)\n",
    ),
    # --- E8b: _tendon_forces() — displaced bulk, non-batch -------------
    (
        "_tendon_forces: displaced bulk (non-batch)",
        "            s_tendon = mat.stress_array(e_tot_g)\n"
        "            s_bulk = bulk_mat.stress_array(e_sec_g)\n",
        "            s_tendon = mat.stress_array(e_tot_g)\n"
        "            s_bulk = bulk_mat.stress_array(\n"
        "                e_sec_g + self._bulk_eps_init)\n",
    ),
    # --- E8c: _tendon_forces() — displaced bulk tangent ----------------
    (
        "_tendon_forces: displaced bulk tangent",
        "                Et_tendon = mat.tangent_array(e_tot_g)\n"
        "                Et_bulk = bulk_mat.tangent_array(e_sec_g)\n",
        "                Et_tendon = mat.tangent_array(e_tot_g)\n"
        "                Et_bulk = bulk_mat.tangent_array(\n"
        "                    e_sec_g + self._bulk_eps_init)\n",
    ),
    # --- E9: measure() — bulk stress -----------------------------------
    (
        "measure: bulk stress",
        "        # Bulk stresses\n"
        "        if not self._is_multi_material:\n"
        "            sb = self.sec.bulk_material.stress_array(eb)\n"
        "        else:\n"
        "            sb = np.zeros_like(eb)\n"
        "            for mat, idx in self._bulk_groups:\n"
        "                sb[idx] = mat.stress_array(eb[idx])\n",
        "        # Bulk stresses\n"
        "        if not self._is_multi_material:\n"
        "            sb = self.sec.bulk_material.stress_array(\n"
        "                eb + self._bulk_eps_init)\n"
        "        else:\n"
        "            sb = np.zeros_like(eb)\n"
        "            for (mat, idx), eps_b in zip(self._bulk_groups,\n"
        "                                         self._bulk_eps_by_group):\n"
        "                sb[idx] = mat.stress_array(eb[idx] + eps_b)\n",
    ),
    # --- E10a: measure() — displaced bulk at rebars --------------------
    (
        "measure: displaced bulk at rebars",
        "            sr_ideal_gross[idx] = mat.stress_array(er_g)\n"
        "            sb_at_rebars[idx] = bulk_mat.stress_array(er_g)\n",
        "            sr_ideal_gross[idx] = mat.stress_array(er_g)\n"
        "            sb_at_rebars[idx] = bulk_mat.stress_array(\n"
        "                er_g + self._bulk_eps_init)\n",
    ),
    # --- E10b: measure() — displaced bulk at tendons -------------------
    (
        "measure: displaced bulk at tendons",
        "                sp[idx] = mat.stress_array(e_tot_g)\n"
        "                sb_at_tendons[idx] = bulk_mat.stress_array(e_sec_g)\n",
        "                sp[idx] = mat.stress_array(e_tot_g)\n"
        "                sb_at_tendons[idx] = bulk_mat.stress_array(\n"
        "                    e_sec_g + self._bulk_eps_init)\n",
    ),
    # --- E11: strains_within_limits() — admissibility on offset --------
    (
        "strains_within_limits: bulk admissibility on eps+offset",
        "        for mat, idx in self._bulk_groups:\n"
        "            e_group = eb[idx]\n"
        "            if np.any(e_group < mat.eps_min) or \\\n"
        "               np.any(e_group > mat.eps_max):\n"
        "                return False\n",
        "        for (mat, idx), eps_b in zip(self._bulk_groups,\n"
        "                                     self._bulk_eps_by_group):\n"
        "            e_group = eb[idx] + eps_b\n"
        "            if np.any(e_group < mat.eps_min) or \\\n"
        "               np.any(e_group > mat.eps_max):\n"
        "                return False\n",
    ),
]


# ===========================================================================
#  io_yaml.py — lift the two no-silent-no-op guards (SEPARATE, LATER step)
# ===========================================================================

IOYAML_GUARD_EDITS = [
    (
        "_parse_bulk_prestrain: remove non-zero raise",
        "    if value != 0.0:\n"
        "        raise ValueError(\n"
        "            f\"section: bulk 'prestrain'/'eps_init' = {value:g} is not \"\n"
        "            f\"yet consumed by the fiber solver \u2014 the resistance domain \"\n"
        "            f\"would NOT reflect it. To avoid a silent no-op, non-zero \"\n"
        "            f\"values are rejected until the solver-side support lands \"\n"
        "            f\"(shrinkage/losses phase). Remove the field for now.\"\n"
        "        )\n"
        "    return value\n",
        "    return value\n",
    ),
    (
        "_parse_section_ops_spec: remove non-zero bulk_eps raise",
        "    if \"bulk_eps\" in ops:\n"
        "        val = float(ops[\"bulk_eps\"])\n"
        "        if val != 0.0:\n"
        "            raise ValueError(\n"
        "                f\"{where}: section_ops 'bulk_eps' = {val:g} is not yet \"\n"
        "                f\"consumed by the fiber solver \u2014 the resistance domain \"\n"
        "                f\"would NOT reflect it. To avoid a silent no-op, \"\n"
        "                f\"non-zero values are rejected until the solver-side \"\n"
        "                f\"support lands (Phase 5: bulk prestrain kernel change \"\n"
        "                f\"in integrator.py). Remove the field for now.\"\n"
        "            )\n"
        "        out[\"bulk_eps\"] = val\n",
        "    if \"bulk_eps\" in ops:\n"
        "        out[\"bulk_eps\"] = float(ops[\"bulk_eps\"])\n",
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="root dir containing integrator.py / io_yaml.py")
    ap.add_argument("--check", action="store_true",
                    help="dry run: verify anchors, write nothing")
    ap.add_argument("--lift-guards", action="store_true",
                    help="ALSO remove the two io_yaml guards (run only AFTER "
                         "the validator passes and the suite is green)")
    args = ap.parse_args()

    try:
        integ = _find(args.src, "integrator.py")
        print(f"integrator.py -> {integ}")
        for label, status in _apply_edits(integ, INTEGRATOR_EDITS, args.check):
            print(f"  [{status}] {label}")

        if args.lift_guards:
            ioy = _find(args.src, "io_yaml.py")
            print(f"\nio_yaml.py -> {ioy}")
            for label, status in _apply_edits(ioy, IOYAML_GUARD_EDITS,
                                              args.check):
                print(f"  [{status}] {label}")
            print("\n  NOTE: also update the 'Raises' docstring clauses by "
                  "hand (cosmetic).")
        else:
            print("\nGuards in io_yaml.py NOT lifted (correct default).")
            print("Run, IN ORDER:")
            print("  1. python run_bulk_prestrain_validation.py")
            print("  2. the full test suite (must stay green)")
            print("  3. python apply_phase5_step1_bulk_kernel.py <src> "
                  "--lift-guards")
    except (EditError, FileNotFoundError) as exc:
        print(f"\nABORTED — {exc}", file=sys.stderr)
        return 1

    mode = "CHECK (nothing written)" if args.check else "APPLIED"
    print(f"\n{mode}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
