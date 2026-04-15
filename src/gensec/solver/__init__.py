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
Solver subpackage.

Provides the fiber integrator (:class:`FiberSolver`), the N-M
interaction diagram generator (:class:`NMDiagram`), and the demand
verification module (:class:`DemandChecker`).
"""

from .integrator import FiberSolver
from .capacity import NMDiagram
from .check import DomainChecker, MxMyContour, VerificationEngine

__all__ = ["FiberSolver", "NMDiagram", "DomainChecker", "MxMyContour", "VerificationEngine"]
