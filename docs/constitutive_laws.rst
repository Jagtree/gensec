.. _constitutive_laws:

==================
Constitutive laws
==================

GenSec is material-agnostic: any stress-strain relationship that
implements the :class:`~gensec.materials.Material` interface can be used.
Three built-in laws and a generic tabulated law are provided.


Abstract interface
-------------------

Every material must implement:

- ``stress(eps)`` — evaluate :math:`\sigma(\varepsilon)` for a scalar
  strain.
- ``stress_array(eps)`` — vectorized evaluation over a NumPy array
  of **arbitrary shape** (1-D, 2-D, …).
- ``tangent(eps)`` — scalar tangent modulus
  :math:`E_t = d\sigma/d\varepsilon`.
- ``tangent_array(eps)`` — vectorized tangent modulus over an array of
  arbitrary shape.
- ``eps_min`` / ``eps_max`` — admissible strain range, consumed by the
  N-M diagram generator to determine scan bounds.

The tangent modulus methods are used by the analytical Jacobian in the
Newton-Raphson solver (see :ref:`architecture_solver`).  The base class
provides a finite-difference fallback for both ``tangent`` and
``tangent_array``, so existing custom materials continue to work
without modification.

Optional Numba acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When `numba <https://numba.pydata.org>`_ is installed
(``pip install gensec[fast]``), the built-in ``Concrete`` and ``Steel``
classes automatically use JIT-compiled kernels for ``stress_array`` and
``tangent_array``.  If Numba is not available, pure-NumPy fallbacks are
used transparently.


Concrete — parabola-rectangle (EC2 3.1.7)
-------------------------------------------

The compression branch follows the parabola-rectangle law per
EN 1992-1-1, §3.1.7.  An optional linear-elastic tension branch
can be activated for serviceability checks or nonlinear analyses.


Compression (:math:`\varepsilon \le 0`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \sigma_c(\varepsilon) =
   \begin{cases}
       -f_{cd}\!\left[1 - \left(1 - \dfrac{\varepsilon}
           {\varepsilon_{c2}}\right)^{\!n}\right]
           & \varepsilon_{c2} \le \varepsilon \le 0
           \\[4pt]
       -f_{cd}
           & \varepsilon_{cu2} \le \varepsilon < \varepsilon_{c2}
           \\[4pt]
       0
           & \varepsilon < \varepsilon_{cu2}
   \end{cases}

where:

- :math:`f_{cd} = \alpha_{cc}\,f_{ck}/\gamma_c` is the design
  compressive strength,
- :math:`\varepsilon_{c2}` is the strain at peak stress
  (default :math:`-0.002`),
- :math:`\varepsilon_{cu2}` is the ultimate compressive strain
  (default :math:`-0.0035`),
- :math:`n` is the parabolic exponent (default 2.0).

The stress is negative (compressive) by sign convention.

For high-strength concrete (:math:`f_{ck} > 50\;\text{MPa}`), the
parameters :math:`\varepsilon_{c2}`, :math:`\varepsilon_{cu2}`, and
:math:`n` deviate from the standard values and must be taken from
EC2 Table 3.1.  The factory function
:func:`~gensec.materials.concrete_from_ec2` computes them automatically.


Tension (:math:`\varepsilon > 0`) — optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, concrete carries **no tensile stress** (:math:`\sigma = 0`
for :math:`\varepsilon > 0`).  This is the correct assumption for
ULS verifications with the parabola-rectangle law.

When both ``fct`` and ``Ec`` are set to positive values, a linear
tension branch is activated:

.. math::

   \sigma_c(\varepsilon) =
   \begin{cases}
       E_c \, \varepsilon
           & 0 < \varepsilon \le \varepsilon_{ct}
           \\[4pt]
       0
           & \varepsilon > \varepsilon_{ct}
   \end{cases}

where :math:`\varepsilon_{ct} = f_{ct} / E_c` is the cracking strain.
Beyond :math:`\varepsilon_{ct}` the concrete is fully cracked and
carries no stress.

This branch is useful for:

- **SLS checks** — stress limitation, crack width estimation.
- **Nonlinear analysis** — capturing the uncracked stiffness.
- **Prestressed sections** — where concrete may remain uncracked
  under service loads.

The ``fct`` value should be chosen to match the verification context:
:math:`f_{ctd,0.05}` (design), :math:`f_{ctm}` (mean), or
:math:`f_{ctk,0.05}` (characteristic).

When using the ``concrete_ec2`` YAML type, setting
``enable_tension: true`` automatically populates ``fct`` and ``Ec``
from the EC2 property object.  The ``tension_fct`` field controls
which tensile strength is used (see :doc:`yaml_reference`).

With the tension branch active, the admissible strain range becomes
:math:`[\varepsilon_{cu2},\, \varepsilon_{ct}]`; when disabled it
remains :math:`[\varepsilon_{cu2},\, 0]`.


Tangent modulus (concrete)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The analytical tangent modulus :math:`E_t = d\sigma_c / d\varepsilon`
is:

.. math::

   E_t(\varepsilon) =
   \begin{cases}
       E_c
           & 0 < \varepsilon \le \varepsilon_{ct}
             \;\text{(tension enabled)}
           \\[4pt]
       -\dfrac{f_{cd}\,n}{\varepsilon_{c2}}
           \left(1 - \dfrac{\varepsilon}{\varepsilon_{c2}}\right)^{n-1}
           & \varepsilon_{c2} < \varepsilon \le 0
           \\[4pt]
       0
           & \text{otherwise (plateau, beyond ultimate, post-cracking)}
   \end{cases}

This is used by the analytical Jacobian in the Newton-Raphson solver
to avoid finite-difference perturbations.


Reinforcing steel — elastic-plastic with hardening
----------------------------------------------------

.. math::

   \sigma_s(\varepsilon) =
   \begin{cases}
       E_s\,\varepsilon
           & |\varepsilon| \le \varepsilon_{yd}
           \\[4pt]
       \mathrm{sign}(\varepsilon)\!\left[f_{yd}
           + (f_{td} - f_{yd})\,
           \dfrac{|\varepsilon| - \varepsilon_{yd}}
                 {\varepsilon_{su} - \varepsilon_{yd}}\right]
           & \varepsilon_{yd} < |\varepsilon| \le \varepsilon_{su}
           \\[4pt]
       0
           & |\varepsilon| > \varepsilon_{su}
   \end{cases}

where:

- :math:`f_{yd} = f_{yk}/\gamma_s` is the design yield strength,
- :math:`f_{td} = k \cdot f_{yd}` with :math:`k =` ``k_hardening``
  (ratio :math:`f_t/f_y`; 1.0 = perfectly plastic),
- :math:`\varepsilon_{yd} = f_{yd}/E_s` is the yield strain,
- :math:`\varepsilon_{su}` is the ultimate strain.

The law is **symmetric in tension and compression** by default.  Set
``works_in_compression = False`` to suppress compressive stress
(useful for modelling FRP or tendons that buckle in compression).


Tangent modulus (steel)
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   E_t(\varepsilon) =
   \begin{cases}
       E_s
           & |\varepsilon| \le \varepsilon_{yd}
           \\[4pt]
       \dfrac{f_{td} - f_{yd}}
             {\varepsilon_{su} - \varepsilon_{yd}}
           & \varepsilon_{yd} < |\varepsilon| \le \varepsilon_{su}
           \\[4pt]
       0
           & |\varepsilon| > \varepsilon_{su}
   \end{cases}


Structural steel (EN 10025-2)
------------------------------

Uses the same elastic-plastic model as reinforcing steel, but yield
and ultimate strengths are computed from plate thickness according to
NF EN 10025-2 Table 7.  The factory function
:func:`~gensec.materials.steel_from_en10025` handles the lookup.

Supported grades: **S235**, **S275**, **S355**.

The hardening ratio :math:`k = f_u/f_y` is computed automatically
from the tabulated :math:`f_u` and thickness-dependent :math:`f_y`.


Tabulated material
-------------------

For materials not covered by the built-in laws (CFRP, timber, FRP bars,
shotcrete, etc.), :class:`~gensec.materials.TabulatedMaterial` accepts
an arbitrary set of :math:`(\varepsilon_i,\,\sigma_i)` data points
and interpolates linearly between them.

Strains outside the table range produce :math:`\sigma = 0` (material
failure / rupture).

.. code-block:: python

   from gensec.materials import TabulatedMaterial

   cfrp = TabulatedMaterial(
       strains=[0.0, 0.017],
       stresses=[0.0, 2800.0],
       name="CFRP",
   )

The tabulated law is fully compatible with all solvers and diagram
generators.  The strain limits ``eps_min`` / ``eps_max`` are inferred
from the first and last data points.
