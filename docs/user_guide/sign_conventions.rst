.. _sign_conventions:

=================
Sign conventions
=================

GenSec uses a consistent right-hand coordinate system throughout.
Understanding the sign conventions is essential for interpreting
results correctly.


Coordinate system
-----------------

The origin is at the **bottom-left corner** of the section bounding box.

- :math:`x`-axis: horizontal, pointing right (section width direction).
- :math:`y`-axis: vertical, pointing up (section height direction).

For sections created with parametric primitives
(:mod:`~gensec.geometry.primitives`) or via YAML, this convention is
enforced automatically.


Strain sign convention
-----------------------

.. math::

   \varepsilon > 0 \quad &\Longrightarrow \quad \text{tension} \\
   \varepsilon < 0 \quad &\Longrightarrow \quad \text{compression}


Force and moment signs
-----------------------

+----------------------------+----------------------------+----------------------------+
| Quantity                   | Positive                   | Negative                   |
+============================+============================+============================+
| :math:`N` (axial force)    | Tension                    | Compression                |
+----------------------------+----------------------------+----------------------------+
| :math:`M_x` (moment        | Bottom edge in compression | Top edge in compression    |
| about :math:`x`)           |                            |                            |
+----------------------------+----------------------------+----------------------------+
| :math:`M_y` (moment        | Left edge in compression   | Right edge in compression  |
| about :math:`y`)           |                            |                            |
+----------------------------+----------------------------+----------------------------+

The moment convention follows directly from the fiber integration
(see :doc:`/theory/fiber_method`):

.. math::

   M_x &= \sum_i \sigma_i \, A_i \, (y_i - y_{\text{ref}}) \\
   M_y &= \sum_i \sigma_i \, A_i \, (x_i - x_{\text{ref}})


Strain plane
------------

The strain at any fiber location :math:`(x_i, y_i)` is given by:

.. math::

   \varepsilon(x, y) = \varepsilon_0
       + \chi_x \, (y - y_{\text{ref}})
       + \chi_y \, (x - x_{\text{ref}})

where:

- :math:`\varepsilon_0` is the strain at the reference point
  (centroid by default),
- :math:`\chi_x` is the curvature about the :math:`x`-axis
  [1/mm],
- :math:`\chi_y` is the curvature about the :math:`y`-axis
  [1/mm].

Uniaxial bending corresponds to :math:`\chi_y = 0`.


Embedded vs. external rebars
-----------------------------

For rebars that lie **inside** the bulk material (the default,
``embedded: true``), GenSec subtracts the displaced bulk material to
avoid double-counting the area occupied by the bar:

.. math::

   F_{\text{net},i} =
       \bigl[\sigma_{\text{rebar}}(\varepsilon_i)
       - \sigma_{\text{bulk}}(\varepsilon_i)\bigr] \, A_{s,i}

For **external** elements — CFRP strips bonded to the surface, steel
plates, external tendons — set ``embedded: false``.  The full rebar
stress is used without subtracting the bulk:

.. math::

   F_{\text{net},i} = \sigma_{\text{rebar}}(\varepsilon_i) \, A_{s,i}

This distinction is critical for accurate results when mixing materials.


Units
-----

GenSec works with a consistent set of units internally:

+-------------------+-------+
| Quantity          | Unit  |
+===================+=======+
| Length            | mm    |
+-------------------+-------+
| Stress            | MPa   |
+-------------------+-------+
| Force             | N     |
+-------------------+-------+
| Moment            | N·mm  |
+-------------------+-------+
| Curvature         | 1/mm  |
+-------------------+-------+
| Strain            | —     |
+-------------------+-------+

YAML input uses **kN** and **kN·m** for demands (``N_kN``, ``Mx_kNm``,
``My_kNm``) for convenience.  The conversion happens automatically in
the YAML loader.
