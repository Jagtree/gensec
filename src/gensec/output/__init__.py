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
"""
Output subpackage — reporting, plotting, and numerical export.
"""

from .report import print_section_info, print_fiber_results
from .plots import (plot_nm_diagram, plot_stress_profile,
                    plot_mx_my_diagram, plot_moment_curvature,
                    plot_section, plot_section_state,
                    plot_demand_heatmap, plot_3d_surface,
                    plot_moment_curvature_bundle, plot_polar_ductility,
                    plot_moment_curvature_surface,
                    plot_from_json,)
from .export import (export_nm_domain_csv, export_nm_domain_json,
                    export_demand_results_csv, export_demand_results_json,
                    export_fiber_results_csv,
                    export_3d_surface_csv, export_3d_surface_json,
                    export_verification_json,
                    export_combination_results_json,
                    export_envelope_results_json,
                    export_moment_curvature_json, 
                    export_moment_curvature_csv,
                    export_mx_my_json, 
                    export_mx_my_csv,
                    )

__all__ = [
    "print_section_info", "print_fiber_results",
    "plot_nm_diagram", "plot_stress_profile", "plot_mx_my_diagram",
    "export_nm_domain_csv", "export_nm_domain_json",
    "export_demand_results_csv", "export_demand_results_json",
    "export_fiber_results_csv",
    "export_3d_surface_csv", "export_3d_surface_json",
    "export_verification_json",
    "export_combination_results_json",
    "export_envelope_results_json",
    "export_moment_curvature_json", 
    "export_moment_curvature_csv",
    "export_mx_my_json", 
    "export_mx_my_csv",
    "plot_section_state", "plot_demand_heatmap",
    "plot_3d_surface", "plot_moment_curvature_bundle",
    "plot_polar_ductility", "plot_moment_curvature_surface",
    "plot_from_json",
]
