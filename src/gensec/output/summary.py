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
Verification summary statistics and tiered reporting.

Provides utilities to rank, aggregate, and filter verification results
for scalable reporting of large verification sets (hundreds to thousands
of demands, combinations, or envelopes).

The module is designed around a **three-tier reporting model**:

1. **Summary block** — aggregate statistics printed in a few lines.
   Always displayed regardless of the number of items.
2. **Top-K table** — the *K* items with the highest governing
   :math:`\eta`, printed as a compact table.  Configurable via the
   ``verification_top_k`` YAML flag (default 10).
3. **Full export** — complete CSV/JSON files with all items ranked.
   Never printed to the console.

All functions operate on the result dicts produced by
:class:`~gensec.solver.check.VerificationEngine` without modifying
them.
"""

import numpy as np

# Eta keys for the three result types (demand, combination, envelope).
_DEMAND_ETA_KEYS = ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D")
_COMBINATION_ETA_KEY = "eta_governing"
_ENVELOPE_ETA_KEY = "eta_max"


# ======================================================================
#  Governing eta extraction
# ======================================================================

def governing_eta(result, result_type="demand"):
    r"""
    Extract the governing :math:`\eta` from a single verification result.

    The governing :math:`\eta` is defined as the maximum across all
    enabled utilization metrics for the given result.

    Parameters
    ----------
    result : dict
        A single verification result from
        :meth:`~gensec.solver.check.VerificationEngine.check_demand`,
        :meth:`~gensec.solver.check.VerificationEngine.check_combination`,
        or :meth:`~gensec.solver.check.VerificationEngine.check_envelope`.
    result_type : {'demand', 'combination', 'envelope'}
        Determines which keys to inspect for :math:`\eta` values.

    Returns
    -------
    float
        The governing (worst-case) :math:`\eta`.  Returns ``inf`` if
        no numeric :math:`\eta` is available (should not happen in
        normal operation).

    Notes
    -----
    - For **demands**, the governing :math:`\eta` is
      :math:`\max(\eta_{\text{norm}}, \eta_{\beta}, \eta_{\text{ray}},
      \eta_{\text{2D}})` across whichever metrics are present.
    - For **combinations**, the ``eta_governing`` field is already
      computed by the verification engine (it accounts for both
      point and path metrics across all stages).
    - For **envelopes**, the ``eta_max`` field gives the worst
      :math:`\eta` across all members.
    """
    if result_type == "combination":
        return float(result.get(_COMBINATION_ETA_KEY, float("inf")))

    if result_type == "envelope":
        return float(result.get(_ENVELOPE_ETA_KEY, float("inf")))

    # demand: max across enabled point metrics.
    vals = []
    for key in _DEMAND_ETA_KEYS:
        v = result.get(key)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return max(vals) if vals else float("inf")


# ======================================================================
#  Ranking
# ======================================================================

def rank_results(results, result_type="demand"):
    r"""
    Sort verification results by governing :math:`\eta` (descending).

    Each result dict is augmented with a ``_rank`` key (1-based) and
    a ``_eta_governing`` key for uniform downstream access.  The
    input list is **not** modified; a new sorted list is returned.

    Parameters
    ----------
    results : list of dict
        Verification results from the engine.
    result_type : {'demand', 'combination', 'envelope'}

    Returns
    -------
    list of dict
        Shallow copies of the input dicts, sorted from worst
        (highest :math:`\eta`) to best, with ``_rank`` and
        ``_eta_governing`` injected.
    """
    decorated = []
    for r in results:
        copy = dict(r)
        copy["_eta_governing"] = governing_eta(r, result_type)
        decorated.append(copy)

    decorated.sort(key=lambda x: x["_eta_governing"], reverse=True)

    for i, r in enumerate(decorated, start=1):
        r["_rank"] = i

    return decorated


def top_k_results(results, k, result_type="demand"):
    r"""
    Extract the top-*K* results by governing :math:`\eta`.

    Parameters
    ----------
    results : list of dict
        Verification results from the engine.
    k : int or None
        Number of items to return.  If *None* or >= len(results),
        returns all items (still sorted).
    result_type : {'demand', 'combination', 'envelope'}

    Returns
    -------
    ranked : list of dict
        The *K* worst results, sorted descending by :math:`\eta`,
        with ``_rank`` and ``_eta_governing`` injected.
    n_remaining : int
        Number of items not included (``len(results) - K``).
    n_remaining_verified : int
        How many of the remaining items are verified
        (:math:`\eta \le 1`).
    """
    ranked = rank_results(results, result_type)
    n = len(ranked)

    if k is None or k >= n:
        return ranked, 0, 0

    top = ranked[:k]
    rest = ranked[k:]
    n_remaining = len(rest)
    n_remaining_verified = sum(1 for r in rest if r.get("verified", False))
    return top, n_remaining, n_remaining_verified


# ======================================================================
#  Summary statistics
# ======================================================================

def compute_summary_stats(results, result_type="demand"):
    r"""
    Compute aggregate statistics over a set of verification results.

    Parameters
    ----------
    results : list of dict
        Verification results from the engine.
    result_type : {'demand', 'combination', 'envelope'}

    Returns
    -------
    dict
        Summary with keys:

        - ``n_total`` (int): total number of items.
        - ``n_verified`` (int): items with :math:`\eta \le 1`.
        - ``n_fail`` (int): items with :math:`\eta > 1`.
        - ``eta_max`` (float): worst :math:`\eta`.
        - ``eta_mean`` (float): mean of governing :math:`\eta`.
        - ``eta_p95`` (float): 95th percentile.
        - ``eta_p99`` (float): 99th percentile.
        - ``eta_median`` (float): 50th percentile.
        - ``governing_name`` (str): name of the governing item.
        - ``governing_rank`` (int): rank of the governing item (always 1).
        - ``all_verified`` (bool): True if every item passes.

    Notes
    -----
    Percentile computation uses :func:`numpy.percentile` with linear
    interpolation, consistent with the standard definition.  For a
    single item, all percentiles equal that item's :math:`\eta`.
    """
    if not results:
        return {
            "n_total": 0,
            "n_verified": 0,
            "n_fail": 0,
            "eta_max": 0.0,
            "eta_mean": 0.0,
            "eta_p95": 0.0,
            "eta_p99": 0.0,
            "eta_median": 0.0,
            "governing_name": "",
            "governing_rank": 0,
            "all_verified": True,
        }

    ranked = rank_results(results, result_type)
    etas = np.array([r["_eta_governing"] for r in ranked])

    n_total = len(etas)
    n_fail = int(np.sum(etas > 1.0))
    governing = ranked[0]

    return {
        "n_total": n_total,
        "n_verified": n_total - n_fail,
        "n_fail": n_fail,
        "eta_max": float(etas[0]),
        "eta_mean": float(np.mean(etas)),
        "eta_p95": float(np.percentile(etas, 95)),
        "eta_p99": float(np.percentile(etas, 99)),
        "eta_median": float(np.percentile(etas, 50)),
        "governing_name": governing.get("name", ""),
        "governing_rank": 1,
        "all_verified": n_fail == 0,
    }


# ======================================================================
#  Console printers — tiered output
# ======================================================================

def _format_summary_block(title, stats):
    r"""
    Format the tier-1 summary block as a multi-line string.

    Parameters
    ----------
    title : str
        Section title (e.g. ``"DEMAND VERIFICATION"``).
    stats : dict
        Output of :func:`compute_summary_stats`.

    Returns
    -------
    str
        Ready-to-print summary block.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  {title}")
    lines.append("=" * 80)

    n = stats["n_total"]
    nv = stats["n_verified"]
    nf = stats["n_fail"]
    lines.append(
        f"  Items: {n}    Verified: {nv}    "
        f"Failed: {nf}"
    )
    lines.append(
        f"  eta_max: {stats['eta_max']:.4f}    "
        f"eta_mean: {stats['eta_mean']:.4f}    "
        f"eta_p95: {stats['eta_p95']:.4f}    "
        f"eta_p99: {stats['eta_p99']:.4f}"
    )
    gov = stats["governing_name"]
    if gov:
        lines.append(f"  Governing: {gov}  (eta = {stats['eta_max']:.4f})")

    if stats["all_verified"]:
        lines.append("  --> ALL VERIFIED")
    else:
        lines.append(f"  --> *** {nf} item(s) NOT VERIFIED ***")

    lines.append("-" * 80)
    return "\n".join(lines)


def _eta_columns(results):
    """
    Detect which :math:`\\eta` columns are present in demand results.

    Parameters
    ----------
    results : list of dict
        Demand verification results.

    Returns
    -------
    list of str
        Keys of the :math:`\\eta` types found in at least one result.
    """
    cols = []
    for key in _DEMAND_ETA_KEYS:
        if any(key in r for r in results):
            cols.append(key)
    return cols


def print_demand_summary(results, top_k=10):
    r"""
    Print tiered demand verification output to the console.

    Tier 1: summary statistics block (always).
    Tier 2: top-*K* table sorted by governing :math:`\eta` (always,
    but truncated to *K* rows).

    Parameters
    ----------
    results : list of dict
        Output of :meth:`VerificationEngine.check_demands`.
    top_k : int
        Maximum number of rows in the detail table.  If the total
        number of demands is <= ``top_k``, all demands are shown
        and the output is equivalent to the legacy full table.
    """
    if not results:
        return

    stats = compute_summary_stats(results, "demand")
    print(_format_summary_block("DEMAND VERIFICATION", stats))

    # Tier 2: top-K table.
    top, n_remaining, n_rem_ok = top_k_results(
        results, top_k, "demand")

    eta_cols = _eta_columns(results)
    has_my = any(r.get("My_kNm", 0) != 0 for r in results)

    # Header.
    hdr = f"  {'#':>4} {'Name':>20} {'N[kN]':>8} {'Mx[kNm]':>9}"
    if has_my:
        hdr += f" {'My[kNm]':>9}"
    for ec in eta_cols:
        hdr += f" {ec:>12}"
    hdr += f" {'Status':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in top:
        status = "OK" if r["verified"] else "FAIL"
        line = (f"  {r['_rank']:>4} {r['name']:>20} "
                f"{r['N_kN']:>8.1f} {r['Mx_kNm']:>9.2f}")
        if has_my:
            line += f" {r.get('My_kNm', 0):>9.2f}"
        for ec in eta_cols:
            val = r.get(ec)
            if val is not None:
                line += f" {val:>12.4f}"
            else:
                line += f" {'---':>12}"
        line += f" {status:>8}"
        print(line)

    if n_remaining > 0:
        n_rem_fail = n_remaining - n_rem_ok
        suffix = ""
        if n_rem_fail > 0:
            suffix = f", {n_rem_fail} FAIL"
        print(f"  ... and {n_remaining} more "
              f"({n_rem_ok} verified{suffix})")

    print("=" * 80)


def print_combination_summary(combo_results, top_k=10):
    r"""
    Print tiered combination verification output to the console.

    Tier 1: summary statistics block.
    Tier 2: top-*K* combinations by ``eta_governing``.  For staged
    combinations within the top-*K*, the per-stage detail is printed
    inline.

    Parameters
    ----------
    combo_results : list of dict
        Output of
        :meth:`VerificationEngine.check_combination` calls.
    top_k : int
        Maximum number of combinations to show in detail.
    """
    if not combo_results:
        return

    stats = compute_summary_stats(combo_results, "combination")
    print(_format_summary_block("COMBINATION VERIFICATION", stats))

    top, n_remaining, n_rem_ok = top_k_results(
        combo_results, top_k, "combination")

    for cr in top:
        name = cr["name"]
        ctype = cr.get("type", "simple")
        res = cr.get("resultant", {})
        eta_gov = cr.get("eta_governing", cr.get("eta_norm", "---"))
        status = "OK" if cr.get("verified", False) else "FAIL"
        rank = cr.get("_rank", "?")

        print(f"\n  #{rank} -- {name} ({ctype}) --")
        print(f"     Resultant: N={res.get('N_kN', 0):.1f} kN,"
              f" Mx={res.get('Mx_kNm', 0):.2f} kNm,"
              f" My={res.get('My_kNm', 0):.2f} kNm")

        if "stages" in cr:
            # Detect which eta keys appear in stages.
            stage_etas = []
            for k in ("eta_norm", "eta_norm_beta", "eta_norm_ray",
                      "eta_2D", "eta_path_norm_ray",
                      "eta_path_norm_beta", "eta_path_2D"):
                if any(k in s and s[k] is not None
                       for s in cr["stages"]):
                    stage_etas.append(k)

            hdr = (f"     {'Stage':>20} {'N[kN]':>8} {'Mx':>9}"
                   f" {'My':>9}")
            for se in stage_etas:
                hdr += f" {se:>12}"
            print(hdr)
            print("     " + "-" * (len(hdr) - 5))

            for s in cr["stages"]:
                cum = s.get("cumulative", {})
                line = (f"     {s['name']:>20}"
                        f" {cum.get('N_kN', 0):>8.1f}"
                        f" {cum.get('Mx_kNm', 0):>9.2f}"
                        f" {cum.get('My_kNm', 0):>9.2f}")
                for se in stage_etas:
                    val = s.get(se)
                    if val is not None:
                        line += f" {val:>12.4f}"
                    else:
                        line += f" {'---':>12}"
                print(line)
                if "warning" in s:
                    print(f"       WARNING: {s['warning']}")

        print(f"     eta_governing = {eta_gov}  ->  {status}")

    if n_remaining > 0:
        n_rem_fail = n_remaining - n_rem_ok
        suffix = ""
        if n_rem_fail > 0:
            suffix = f", {n_rem_fail} FAIL"
        print(f"\n  ... and {n_remaining} more combinations "
              f"({n_rem_ok} verified{suffix})")

    print("=" * 80)


def print_envelope_summary(envelope_results, top_k=10):
    r"""
    Print tiered envelope verification output to the console.

    Parameters
    ----------
    envelope_results : list of dict
        Output of
        :meth:`VerificationEngine.check_envelope` calls.
    top_k : int
        Maximum number of envelopes to show.
    """
    if not envelope_results:
        return

    stats = compute_summary_stats(envelope_results, "envelope")
    print(_format_summary_block("ENVELOPE VERIFICATION", stats))

    top, n_remaining, n_rem_ok = top_k_results(
        envelope_results, top_k, "envelope")

    hdr = (f"  {'#':>4} {'Envelope':>25} {'eta_max':>8}"
           f" {'Governing':>25} {'Status':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for er in top:
        status = "OK" if er["verified"] else "FAIL"
        rank = er.get("_rank", "?")
        print(f"  {rank:>4} {er['name']:>25} "
              f"{er['eta_max']:>8.4f} "
              f"{er['governing_member']:>25} {status:>8}")

    if n_remaining > 0:
        n_rem_fail = n_remaining - n_rem_ok
        suffix = ""
        if n_rem_fail > 0:
            suffix = f", {n_rem_fail} FAIL"
        print(f"  ... and {n_remaining} more "
              f"({n_rem_ok} verified{suffix})")

    print("=" * 80)


# ======================================================================
#  Summary export (JSON)
# ======================================================================

def build_verification_summary(demand_results, combination_results,
                               envelope_results):
    r"""
    Build a unified verification summary dict with statistics.

    This is an enriched version of the data exported by
    :func:`~gensec.output.export.export_verification_json`, adding
    aggregate statistics for each result category.  Suitable for
    programmatic consumption by the API or GUI.

    Parameters
    ----------
    demand_results : list of dict
    combination_results : list of dict
    envelope_results : list of dict

    Returns
    -------
    dict
        Top-level keys:

        - ``demands``: ranked demand results with ``_rank`` and
          ``_eta_governing``.
        - ``combinations``: ranked combination results.
        - ``envelopes``: ranked envelope results.
        - ``statistics``: per-category summary stats and a global
          ``all_verified`` flag.
    """
    summary = {}

    d_ranked = rank_results(demand_results, "demand") if demand_results else []
    c_ranked = rank_results(combination_results, "combination") if combination_results else []
    e_ranked = rank_results(envelope_results, "envelope") if envelope_results else []

    if d_ranked:
        summary["demands"] = d_ranked
    if c_ranked:
        summary["combinations"] = c_ranked
    if e_ranked:
        summary["envelopes"] = e_ranked

    # Per-category statistics.
    stats = {}
    if demand_results:
        stats["demands"] = compute_summary_stats(
            demand_results, "demand")
    if combination_results:
        stats["combinations"] = compute_summary_stats(
            combination_results, "combination")
    if envelope_results:
        stats["envelopes"] = compute_summary_stats(
            envelope_results, "envelope")

    # Global pass/fail.
    all_ok = all(
        s.get("all_verified", True) for s in stats.values()
    )
    stats["all_verified"] = all_ok
    summary["statistics"] = stats

    return summary


def select_demands_for_fiber_details(demand_results, top_k=5):
    r"""
    Select the demands that should receive per-fiber post-processing.

    For small demand sets (``n <= top_k``), all demands are selected.
    For large sets, only the *K* with the highest governing
    :math:`\eta` are selected, prioritising unverified demands.

    Parameters
    ----------
    demand_results : list of dict
        Output of :meth:`VerificationEngine.check_demands`.
    top_k : int or None
        Maximum number of demands for fiber details.  ``None`` means
        all demands.

    Returns
    -------
    list of str
        Names of the selected demands, in descending :math:`\eta`
        order.
    """
    if not demand_results:
        return []

    if top_k is None or len(demand_results) <= top_k:
        return [r["name"] for r in demand_results]

    ranked = rank_results(demand_results, "demand")
    return [r["name"] for r in ranked[:top_k]]
