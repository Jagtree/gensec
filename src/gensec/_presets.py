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

r"""
Computation presets — predefined parameter sets for the solver.

Three named presets control the tradeoff between speed and resolution
across all diagram generators.  ``"standard"`` is the default and
provides a good balance for typical engineering practice.

Usage in YAML
-------------

.. code-block:: yaml

    output:
      preset: rapid                # rapid | standard | accurate
      n_angles_mx_my: 144         # per-parameter override wins

Usage in Python
---------------

.. code-block:: python

    from gensec._presets import PRESETS, resolve_preset

    opts = resolve_preset("standard", {"n_chi": 60})
    # → standard defaults with n_chi overridden to 60

The :func:`resolve_preset` helper merges a named preset with
user-supplied overrides in a single call, so both CLI and YAML
parsers share the same resolution logic.

Preset parameters
-----------------

.. list-table::
   :header-rows: 1

   * - Key
     - Description
     - rapid
     - standard
     - accurate
   * - ``n_points``
     - Edge configs for N-M / M-χ
     - 100
     - 200
     - 400
   * - ``n_angles_3d``
     - Curvature directions for 3D surface
     - 24
     - 36
     - 72
   * - ``n_points_per_angle``
     - Edge configs per direction (3D)
     - 50
     - 80
     - 200
   * - ``n_chi``
     - Curvature steps for Mx-My contour
     - 30
     - 36
     - 50
   * - ``n_angles_mx_my``
     - Output contour angular resolution
     - 36
     - 72
     - 144
   * - ``n_levels_3d``
     - N-level slices for 3D visualisation
     - 10
     - 15
     - 20
   * - ``n_angles_polar``
     - Directions for polar ductility
     - 24
     - 36
     - 72
   * - ``n_chi_polar``
     - Curvature steps for polar ductility
     - 30
     - 50
     - 100
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# -----------------------------------------------------------------------
# Named presets
# -----------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, int]] = {
    "rapid": {
        "n_points":            30,
        "n_angles_3d":         24,
        "n_points_per_angle":  20,
        "n_chi":               10,
        "n_angles_mx_my":      36,
        "n_levels_3d":         5,
        "n_angles_polar":      6,
        "n_chi_polar":         10,
    },
    "standard": {
        "n_points":           200,
        "n_angles_3d":         36,
        "n_points_per_angle":  80,
        "n_chi":               36,
        "n_angles_mx_my":      72,
        "n_levels_3d":         15,
        "n_angles_polar":      36,
        "n_chi_polar":         50,
    },
    "accurate": {
        "n_points":           400,
        "n_angles_3d":         72,
        "n_points_per_angle": 200,
        "n_chi":               50,
        "n_angles_mx_my":     144,
        "n_levels_3d":         20,
        "n_angles_polar":      72,
        "n_chi_polar":        100,
    },
}

DEFAULT_PRESET: str = "rapid"
"""Name of the preset applied when none is specified."""


# -----------------------------------------------------------------------
# Resolver
# -----------------------------------------------------------------------

def resolve_preset(
    preset_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""
    Merge a named preset with user overrides.

    Parameters
    ----------
    preset_name : str or None
        One of ``"rapid"``, ``"standard"``, ``"accurate"``.
        If ``None`` or unrecognised, falls back to
        :data:`DEFAULT_PRESET`.
    overrides : dict or None
        Per-parameter overrides.  Keys present here replace the
        corresponding preset value.  ``None`` values are ignored.

    Returns
    -------
    dict
        Resolved parameter dictionary.

    Examples
    --------
    >>> resolve_preset("rapid", {"n_chi": 60})
    {'n_points': 100, ..., 'n_chi': 60, ...}
    """
    base = dict(PRESETS.get(preset_name or DEFAULT_PRESET,
                            PRESETS[DEFAULT_PRESET]))
    if overrides:
        base.update({k: v for k, v in overrides.items()
                     if v is not None})
    return base
