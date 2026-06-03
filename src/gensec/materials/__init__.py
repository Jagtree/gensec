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
Materials subpackage.

Core constitutive laws:

- :class:`Material` — abstract base class.
- :class:`Concrete` — parabola-rectangle (EC2).
- :class:`Steel` — elastic-plastic with hardening.
- :class:`PrestressingSteel` — generic bilinear diagram (EC2 §3.3).
- :class:`TabulatedMaterial` — arbitrary curve from data points.

EC2 / EN 10025 property classes:

- :class:`fben2` — full EC2 Table 3.1 concrete properties.
- :class:`fbpren` — EC2 §3.3 prestressing-steel design properties.
- :class:`Steel_EN10025_2` — structural steel per EN 10025-2.
- :func:`concrete_from_ec2` — bridge: fben2 -> Concrete.
- :func:`concrete_from_class` — bridge: class name -> Concrete.
- :func:`steel_from_en10025` — bridge: EN 10025 -> Steel.
- :func:`prestress_from_ec2` — bridge: fbpren -> PrestressingSteel.
- :func:`prestress_from_class` — bridge: designation -> PrestressingSteel.
"""

from .base import Material
from .concrete import Concrete
from .steel import Steel, PrestressingSteel
from .tabulated import TabulatedMaterial
from .ec2_properties import fben2, ConcClassFck
from .prestress_properties import fbpren, PrestressClassData
from .en10025_properties import Steel_plate, Steel_EN10025_2
from .ec2_bridge import (
    concrete_from_ec2, concrete_from_class, steel_from_en10025,
    prestress_from_ec2, prestress_from_class,
)

__all__ = [
    "Material", "Concrete", "Steel", "PrestressingSteel",
    "TabulatedMaterial",
    "fben2", "ConcClassFck",
    "fbpren", "PrestressClassData",
    "Steel_plate", "Steel_EN10025_2",
    "concrete_from_ec2", "concrete_from_class", "steel_from_en10025",
    "prestress_from_ec2", "prestress_from_class",
]