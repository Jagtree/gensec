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
# along with GenSec. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

"""
Numerical export utilities — CSV and JSON.

Provides functions to export N-M domain points, demand verification
results, and per-fiber stress/strain states.
"""

import json
import csv
import numpy as np
import os


def export_nm_domain_csv(nm_data, filepath):
    """
    Export N-M domain point cloud to CSV.

    Columns: ``N_kN``, ``Mx_kNm``.

    Parameters
    ----------
    nm_data : dict
        Output of :meth:`NMDiagram.generate`.
    filepath : str
        Output CSV file path.
    """
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["N_kN", "Mx_kNm"])
        for n, m in zip(nm_data["N_kN"], nm_data["M_kNm"]):
            w.writerow([f"{n:.4f}", f"{m:.6f}"])


def export_nm_domain_json(nm_data, filepath):
    """
    Export N-M domain point cloud to JSON.

    Parameters
    ----------
    nm_data : dict
        Output of :meth:`NMDiagram.generate`.
    filepath : str
        Output JSON file path.
    """
    data = {
        "description": "N-Mx interaction domain point cloud",
        "units": {"N": "kN", "Mx": "kN*m"},
        "n_points": len(nm_data["N_kN"]),
        "N_kN": [round(float(x), 4) for x in nm_data["N_kN"]],
        "Mx_kNm": [round(float(x), 6) for x in nm_data["M_kNm"]],
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def export_demand_results_csv(results, filepath):
    """
    Export demand verification summary to CSV (v2.1 format).

    Columns include all enabled :math:`\\eta` types.

    Parameters
    ----------
    results : list of dict
        Output of :meth:`VerificationEngine.check_demands`.
    filepath : str
    """
    if not results:
        return

    # Determine which η columns are present.
    eta_keys = []
    for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D"):
        if any(key in r for r in results):
            eta_keys.append(key)

    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        header = ["name", "N_kN", "Mx_kNm", "My_kNm"] + eta_keys + [
            "inside", "verified"]
        w.writerow(header)
        for r in results:
            row = [
                r["name"],
                f"{r['N_kN']:.2f}",
                f"{r['Mx_kNm']:.4f}",
                f"{r.get('My_kNm', 0):.4f}",
            ]
            for ek in eta_keys:
                val = r.get(ek)
                row.append(f"{val:.4f}" if val is not None else "")
            row += [r["inside"], r["verified"]]
            w.writerow(row)


def export_demand_results_json(results, filepath):
    """
    Export demand verification summary to JSON (v2.1 format).

    Parameters
    ----------
    results : list of dict
        Output of :meth:`VerificationEngine.check_demands`.
    filepath : str
    """
    clean = []
    for r in results:
        entry = {
            "name": r["name"],
            "N_kN": round(r["N_kN"], 2),
            "Mx_kNm": round(r["Mx_kNm"], 4),
            "My_kNm": round(r.get("My_kNm", 0), 4),
            "inside": bool(r["inside"]),
            "verified": bool(r["verified"]),
        }
        for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D"):
            if key in r:
                entry[key] = r[key]
        clean.append(entry)
    with open(filepath, 'w') as f:
        json.dump({"demands": clean}, f, indent=2)


def export_combination_results_json(results, filepath):
    """
    Export combination verification results to JSON.

    Handles both simple and staged combinations per the v2.1 spec.

    Parameters
    ----------
    results : list of dict
        Output of :meth:`VerificationEngine.check_combination` calls.
    filepath : str
    """
    clean = []
    for r in results:
        entry = {"name": r["name"], "type": r.get("type", "simple")}

        # Resultant in display units.
        res = r.get("resultant", {})
        entry["resultant"] = {
            "N_kN": res.get("N_kN", 0),
            "Mx_kNm": res.get("Mx_kNm", 0),
            "My_kNm": res.get("My_kNm", 0),
        }

        if "stages" in r:
            entry["stages"] = _clean_stages(r["stages"])

        for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D", "eta_governing"):
            if key in r:
                entry[key] = r[key]

        entry["inside"] = bool(r.get("inside", False))
        entry["verified"] = bool(r.get("verified", False))
        clean.append(entry)

    with open(filepath, 'w') as f:
        json.dump({"combinations": clean}, f, indent=2)


def _clean_stages(stages):
    """
    Serialize stage results for JSON export.

    Parameters
    ----------
    stages : list of dict

    Returns
    -------
    list of dict
    """
    out = []
    for s in stages:
        entry = {"name": s["name"]}
        for key in ("increment", "cumulative", "base"):
            if key in s:
                entry[key] = s[key]
        for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray",
                     "eta_2D", "eta_path_norm_ray",
                     "eta_path_norm_beta", "eta_path_2D", "warning"):
            if key in s:
                entry[key] = s[key]
        out.append(entry)
    return out


def export_envelope_results_json(results, filepath):
    """
    Export envelope verification results to JSON.

    Parameters
    ----------
    results : list of dict
        Output of :meth:`VerificationEngine.check_envelope` calls.
    filepath : str
    """
    clean = []
    for r in results:
        entry = {
            "name": r["name"],
            "eta_max": r["eta_max"],
            "governing_member": r["governing_member"],
            "verified": bool(r["verified"]),
            "members": r.get("members", []),
        }
        clean.append(entry)
    with open(filepath, 'w') as f:
        json.dump({"envelopes": clean}, f, indent=2)


def export_verification_json(demand_results, combination_results,
                             envelope_results, filepath):
    """
    Export the complete verification summary to a single JSON file.

    Parameters
    ----------
    demand_results : list of dict
    combination_results : list of dict
    envelope_results : list of dict
    filepath : str
    """
    data = {}
    if demand_results:
        data["demands"] = demand_results
    if combination_results:
        combs = []
        for r in combination_results:
            entry = {"name": r["name"], "type": r.get("type", "simple")}
            res = r.get("resultant", {})
            entry["resultant"] = {
                "N_kN": res.get("N_kN", 0),
                "Mx_kNm": res.get("Mx_kNm", 0),
                "My_kNm": res.get("My_kNm", 0),
            }
            if "stages" in r:
                entry["stages"] = _clean_stages(r["stages"])
            for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D", "eta_governing",
                        "inside", "verified"):
                if key in r:
                    entry[key] = r[key]
            combs.append(entry)
        data["combinations"] = combs
    if envelope_results:
        envs = []
        for r in envelope_results:
            envs.append({
                "name": r["name"],
                "eta_max": r["eta_max"],
                "governing_member": r["governing_member"],
                "verified": bool(r["verified"]),
                "members": r.get("members", []),
            })
        data["envelopes"] = envs
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def export_fiber_results_csv(fiber_results, filepath):
    """
    Export per-fiber strain/stress state to CSV.

    Two sections in the file: BULK and REBARS, separated by a blank line.

    Parameters
    ----------
    fiber_results : dict
        Output of :meth:`FiberSolver.get_fiber_results`.
    filepath : str
    """
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)

        bulk = fiber_results["bulk"]
        has_x = "x" in bulk

        # Bulk fibers
        w.writerow(["# BULK FIBERS"])
        header_b = (["x_mm", "y_mm", "eps", "sigma_MPa", "dA_mm2"]
                    if has_x else
                    ["y_mm", "eps", "sigma_MPa", "dA_mm2"])
        w.writerow(header_b)
        for i in range(len(bulk["y"])):
            row = []
            if has_x:
                row.append(f"{bulk['x'][i]:.2f}")
            row += [
                f"{bulk['y'][i]:.2f}",
                f"{bulk['eps'][i]:.8f}",
                f"{bulk['sigma'][i]:.4f}",
                f"{bulk['dA'][i]:.2f}",
            ]
            w.writerow(row)

        w.writerow([])

        # Rebars
        rb = fiber_results["rebars"]
        has_x_r = "x" in rb
        has_net = "sigma_net" in rb
        w.writerow(["# REBAR LAYERS"])
        header_r = ["bar_#"]
        if has_x_r:
            header_r.append("x_mm")
        header_r += ["y_mm", "eps", "sigma_MPa"]
        if has_net:
            header_r.append("sigma_net_MPa")
        header_r += ["A_mm2", "F_net_kN"]
        w.writerow(header_r)
        for i in range(len(rb["y"])):
            sig_net = rb["sigma_net"][i] if has_net else rb["sigma"][i]
            F_net = sig_net * rb["A"][i] / 1e3
            row = [f"{i+1}"]
            if has_x_r:
                row.append(f"{rb['x'][i]:.2f}")
            row += [
                f"{rb['y'][i]:.2f}",
                f"{rb['eps'][i]:.8f}",
                f"{rb['sigma'][i]:.4f}",
            ]
            if has_net:
                row.append(f"{sig_net:.4f}")
            row += [
                f"{rb['A'][i]:.2f}",
                f"{F_net:.4f}",
            ]
            w.writerow(row)


def export_3d_surface_csv(nm_3d, filepath):
    """
    Export 3D resistance surface (N, Mx, My) to CSV.

    Parameters
    ----------
    nm_3d : dict
        Output of :meth:`NMDiagram.generate_biaxial`.
    filepath : str
    """
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["N_kN", "Mx_kNm", "My_kNm"])
        for n, mx, my in zip(nm_3d["N_kN"], nm_3d["Mx_kNm"],
                              nm_3d["My_kNm"]):
            w.writerow([f"{n:.4f}", f"{mx:.6f}", f"{my:.6f}"])


def export_3d_surface_json(nm_3d, filepath):
    """
    Export 3D resistance surface (N, Mx, My) to JSON.

    Parameters
    ----------
    nm_3d : dict
        Output of :meth:`NMDiagram.generate_biaxial`.
    filepath : str
    """
    data = {
        "description": "N-Mx-My resistance surface point cloud",
        "units": {"N": "kN", "Mx": "kN*m", "My": "kN*m"},
        "n_points": len(nm_3d["N_kN"]),
        "N_kN": [round(float(x), 4) for x in nm_3d["N_kN"]],
        "Mx_kNm": [round(float(x), 6) for x in nm_3d["Mx_kNm"]],
        "My_kNm": [round(float(x), 6) for x in nm_3d["My_kNm"]],
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# ==================================================================
#  Moment-curvature data export
# ==================================================================

def export_moment_curvature_json(mc_data, filepath):
    """
    Export moment-curvature data to JSON.

    The JSON contains the full curve arrays plus all key points
    (cracking, yield, ultimate), sufficient to regenerate the plot
    via ``gensec plot``.

    Parameters
    ----------
    mc_data : dict
        Output of :meth:`NMDiagram.generate_moment_curvature`.
    filepath : str
    """
    data = {
        "type": "moment_curvature",
        "direction": mc_data.get("direction", "x"),
        "N_fixed_kN": float(mc_data["N_fixed_kN"]),
        "units": {"chi": "1/km", "M": "kN*m"},
        "chi_km": [round(float(x), 6) for x in mc_data["chi_km"]],
        "M_kNm": [round(float(x), 6) for x in mc_data["M_kNm"]],
    }
    for prefix in ("cracking", "yield", "ultimate"):
        for suffix in ("_pos", "_neg"):
            chi_key = f"{prefix}_chi{suffix}"
            M_key = f"{prefix}_M{suffix}"
            chi_val = mc_data.get(chi_key)
            M_val = mc_data.get(M_key)
            if chi_val is not None:
                data[f"{chi_key}_km"] = round(chi_val * 1e6, 6)
            if M_val is not None:
                data[f"{M_key}_kNm"] = round(M_val / 1e6, 6)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def export_moment_curvature_csv(mc_data, filepath):
    """
    Export moment-curvature curve to CSV.

    Parameters
    ----------
    mc_data : dict
    filepath : str
    """
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["chi_km", "M_kNm"])
        for c, m in zip(mc_data["chi_km"], mc_data["M_kNm"]):
            w.writerow([f"{c:.6f}", f"{m:.6f}"])


# ==================================================================
#  Mx-My contour data export
# ==================================================================

def export_mx_my_json(mx_my_data, filepath):
    """
    Export Mx-My interaction contour to JSON.

    Parameters
    ----------
    mx_my_data : dict
        Output of :meth:`NMDiagram.generate_mx_my`.
    filepath : str
    """
    data = {
        "type": "mx_my_contour",
        "N_fixed_kN": float(mx_my_data.get("N_fixed_kN", 0)),
        "units": {"Mx": "kN*m", "My": "kN*m"},
        "Mx_kNm": [round(float(x), 6)
                    for x in mx_my_data["Mx_kNm"]],
        "My_kNm": [round(float(x), 6)
                    for x in mx_my_data["My_kNm"]],
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def export_mx_my_csv(mx_my_data, filepath):
    """
    Export Mx-My interaction contour to CSV.

    Parameters
    ----------
    mx_my_data : dict
    filepath : str
    """
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Mx_kNm", "My_kNm"])
        for mx, my in zip(mx_my_data["Mx_kNm"],
                          mx_my_data["My_kNm"]):
            w.writerow([f"{mx:.6f}", f"{my:.6f}"])
