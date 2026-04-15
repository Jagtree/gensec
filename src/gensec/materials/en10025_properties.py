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
#!/usr/bin/python
r'''
"""Steel properties as per NF EN 10025-2.

Implements thickness-dependent yield and ultimate strength for carbon
structural steels (grades S235, S275, S355) according to
NF EN 10025-2 Table 7.

Classes
-------
Steel_plate
    Base class providing the common interface.
Steel_EN10025_2
    Derived class with EN 10025-2 thickness-dependent properties.
    
"""
'''
class Steel_plate:
    """
    Base class for steels.

    Args
    ----
    - grade : str
        The steel grade.
    - t : float
        Thickness of the steel plate [mm].
    - young : float
        Young modulus [MPa].

    """

    def __init__(self, grade, t=0, young=200000):
        """Initialise the base steel object.

        Parameters
        ----------
        grade : str
            The steel grade designation (e.g. ``'S355'``).
        t : float, optional
            Thickness of the steel element [mm].  Default is 0
            (thickness-independent context).
        young : float, optional
            Young's modulus [MPa].  Default is 200 000 MPa.

        """
        self.grade = grade
        self.t = t
        self.young = young

    def __repr__(self):
        """Return a human-readable string representation of the steel object."""
        return (
            f"Steel grade = {self.grade}\n"
            f"Thickness = {self.t} mm\n"
            f"Young's Module = {self.young} MPa\n"
        )


class Steel_EN10025_2(Steel_plate):
    """
    Represents a carbon steel as per the NF EN 10025-2 standard.
    Yield and ultimate resistance are evaluated from steel grade and thickness.
    """

    def __init__(
        self,
        grade,
        t=0,
        young=200000
    ):
        """Initialise a carbon steel object as per NF EN 10025-2.

        Yield strength :math:`f_{y,k}` and ultimate strength :math:`f_{u,k}`
        are computed from the grade and thickness according to EN 10025-2
        Table 7.

        Parameters
        ----------
        grade : str
            The steel grade (``'S235'``, ``'S275'``, or ``'S355'``).
        t : float, optional
            Thickness of the element [mm].  Default is 0.
        young : float, optional
            Young's modulus [MPa].  Default is 200 000 MPa.

        Raises
        ------
        ValueError
            If the grade is not recognized or the thickness exceeds the
            range covered by EN 10025-2.
            
        """
        #
        super().__init__(grade, t, young)
        self.f_yk, self.f_uk = self._calculate_properties()

    def _calculate_properties(self):
        """
        Calculate the yield strength (f_yk) and ultimate strength (f_uk)
        based on grade and thickness.
        """
        if self.t <= 3:
            properties = {
                'S355': (355, 510),
                'S275': (275, 430),
                'S235': (235, 360),
            }
        elif 3 < self.t <= 16:
            properties = {
                'S355': (355, 470),
                'S275': (275, 410),
                'S235': (235, 360)
            }
        elif 16 < self.t <= 40:
            properties = {
                'S355': (345, 470),
                'S275': (265, 410),
                'S235': (215, 360)
            }
        elif 40 < self.t <= 63:
            properties = {
                'S355': (335, 470),
                'S275': (255, 410),
                'S235': (215, 360)
            }
        elif 63 < self.t <= 80:
            properties = {
                'S355': (325, 470),
                'S275': (245, 410),
                'S235': (215, 360)
            }
        elif 80 < self.t <= 100:
            properties = {
                'S355': (315, 470),
                'S275': (235, 410),
                'S235': (215, 360)
            }
        elif 100 < self.t <= 150:
            properties = {
                'S355': (295, 450),
                'S275': (225, 400),
                'S235': (195, 350)
            }
        else:
            raise ValueError('Steel thickness not implemented yet.')

        return properties.get(self.grade, (0, 0))

    def __repr__(self):
        """Return a string representation including yield and ultimate strengths."""
        return (
            f"Steel grade (EN10025-2) = {self.grade}\n"
            f"Thickness = {self.t} mm\n"
            f"Young's Module = {self.young} MPa\n"
            f"f_yk = {self.f_yk} MPa\n"
            f"f_uk = {self.f_uk} MPa)\n"
        )
