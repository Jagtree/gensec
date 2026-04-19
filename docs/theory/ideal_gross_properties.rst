.. _ideal_gross_properties:

======================================
Homogenized geometric properties
======================================

This page documents the output produced by the ``geometry``
sub-package when querying the geometric properties of a
:class:`~gensec.geometry.geometry.GenericSection`.  All quantities
are computed on the **homogenized section** (sometimes called
*ideal* or *transformed* section in English, *sezione ideale* /
*omogeneizzata* in Italian).  Mono-material sections are the
natural degenerate case.


.. contents::
   :local:
   :depth: 2


Homogenization convention
=========================

GenSec follows the EC2 / NTC 2018 convention for the homogenized
section.  Given a reference modulus :math:`E_{\mathrm{ref}}`, the
polygon area is scaled by

.. math::

    n_{\mathrm{bulk}} = \frac{E_{\mathrm{bulk}}}{E_{\mathrm{ref}}},

and every point fiber (rebar, tendon, FRP strip) of area
:math:`A_s` and modulus :math:`E_s` contributes with its
**differential** area

.. math::

    \Delta A_{s,\mathrm{id}} = \bigl(n_s - n_{\mathrm{bulk}}\bigr)\,
        A_s,\qquad n_s = \frac{E_s}{E_{\mathrm{ref}}}.

The polygon therefore represents the continuous bulk region in
full, and each rebar adds only the *excess* stiffness needed to
avoid double-counting the substrate it physically displaces.
This has three important consequences:

- **Mono-material degeneration.**  When every fiber has
  :math:`E_s = E_{\mathrm{bulk}}`, the factor
  :math:`(n_s - n_{\mathrm{bulk}}) = 0` and the result coincides
  with the pure polygon geometry.
- **Choice of reference.**  The default is
  :math:`E_{\mathrm{ref}} = E_{\mathrm{bulk}}`, giving
  :math:`n_{\mathrm{bulk}} = 1` and the standard RC convention
  :math:`n_s = E_s / E_{\mathrm{cm}}`.  A different reference can
  be selected by passing ``E_ref`` explicitly to
  :func:`~gensec.geometry.properties.compute_section_properties`.
- **Schema stability.**  All output fields are always populated;
  no field is toggled by the presence or absence of rebars.


Method selection: polygon vs. fiber
===================================

GenSec uses two distinct integration machineries in two distinct
parts of the pipeline:

- the **polygon method** (closed-form ring integrals via Green's
  theorem) for the purely geometric/inertial quantities of the
  :class:`SectionProperties` dataclass — :math:`A`, :math:`S_x`,
  :math:`S_y`, :math:`I_x`, :math:`I_y`, :math:`I_{xy}`, extreme
  fiber distances, elastic and plastic section moduli, kern, and
  inertia ellipse;
- the **fiber method** (domain discretization into small regions
  of approximately uniform stress) in the section solver, for
  the response integrals :math:`N, M_x, M_y` under a given
  strain field and a non-linear constitutive law.

The separation is not arbitrary: each method is the right tool
for one and only one of the two problems, and neither can
replace the other cleanly.


Accuracy of the polygon method
------------------------------

For an integrand that is a **polynomial in** :math:`x` and
:math:`y`, Green's theorem reduces a 2-D surface integral over a
polygon :math:`\Omega` to a 1-D circulation along its boundary:

.. math::

    \iint_{\Omega} P(x, y)\, \mathrm{d}A
    = \oint_{\partial\Omega} Q(x, y)\,\mathrm{d}\ell,

and on a polygonal boundary this circulation becomes a finite
sum of monomials evaluated at the vertices.  Every moment of
order up to two — :math:`A`, :math:`S_x`, :math:`S_y`,
:math:`I_{xx}`, :math:`I_{yy}`, :math:`I_{xy}` — falls in this
class.  The result is therefore **exact** for a polygon with
vertices at well-defined coordinates: no discretization error
whatsoever, only the floating-point round-off of the summation.

The limitation is subtler.  When the real section has **curved
boundaries** (circles, annuli, fillets), the polygon is itself
an approximation of the geometry, and the integrals are exact
with respect to the polygon but approximate with respect to the
true shape.  For a circle discretized into :math:`N` sides the
relative error on the area scales as

.. math::

    \varepsilon_A \approx \frac{(2\pi)^2}{6\,N^2}
    = \frac{6.58}{N^2},

giving :math:`\sim 0.16\%` for :math:`N = 64` and
:math:`\sim 0.01\%` for :math:`N = 256`.  Second moments and
the plastic modulus scale similarly.  In GenSec the polygon
resolution is a user choice driven by Shapely's ``resolution``
parameter when buffering.

For :math:`Z` (plastic modulus) the integrand :math:`|y -
y_{\mathrm{pna}}|` is **not** a polynomial because of the
absolute value, but the plastic neutral axis is a *straight
line* that splits :math:`\Omega` into two half-plane
intersections; each half is a new polygon on which Green's
theorem is again exact.  The only numerical error in :math:`Z`
is the bisection tolerance on :math:`y_{\mathrm{pna}}` itself
(currently :math:`10^{-10}` relative on the homogenized area),
which is orders of magnitude below any physical uncertainty.


Accuracy of the fiber method
----------------------------

The fiber method discretizes :math:`\Omega` into small
sub-domains :math:`\Omega_f` and replaces every integral with a
midpoint quadrature:

.. math::

    \iint_{\Omega} g(x, y)\, \mathrm{d}A
    \approx \sum_f g\bigl(x_f, y_f\bigr)\, A_f,

where :math:`(x_f, y_f)` is the centroid of fiber :math:`f` and
:math:`A_f` its area.  For a smooth integrand :math:`g` the
error of each fiber is :math:`O(h_f^{2}\,\nabla^{2} g)`, so the
global error scales as :math:`O(h^{2})` with :math:`h` the
characteristic fiber size.  As a concrete reference, for a
rectangular section :math:`B \times H` discretized in :math:`N`
horizontal strips the relative error on :math:`I_x` is
:math:`1/N^{2}`: :math:`\sim 1\%` for :math:`N = 10`,
:math:`\sim 0.04\%` for :math:`N = 50`.  With a practical
target of a few tenths of a percent, fiber sizes of a few
millimetres are appropriate for typical RC members.

Fibers thus introduce a genuine discretization error that is
always present, but they work for **any** integrand — including
integrands that are not polynomial in :math:`(x, y)`.  This is
what makes them indispensable downstream.


Why the two methods are not interchangeable
-------------------------------------------

The decisive distinction is between **linear** and **non-linear**
integrands along the depth :math:`y`.

1. *Geometric and elastic moments* are monomials in :math:`y`:
   :math:`1`, :math:`y`, :math:`y^{2}`.  The polygon method is
   exact and fast — two or three orders of magnitude faster than
   a fiber sweep on a typical section, because it touches only
   the vertices, not the interior.

2. *Constitutive response integrals* involve the stress
   :math:`\sigma\bigl(\varepsilon(y)\bigr)`.  Under the
   kinematic assumption of plane sections
   :math:`\varepsilon(y) = \varepsilon_0 + \chi_x\, y
   - \chi_y\, x` is linear in the coordinates, **but**
   :math:`\sigma(\cdot)` is emphatically not linear in
   :math:`\varepsilon`: EC2 concrete is parabola–rectangle,
   confined concrete is Sargin / Mander, reinforcing steel is
   bilinear with strain hardening, prestressing steel follows a
   Ramberg–Osgood law.  There is no closed-form antiderivative
   of
   :math:`\sigma\bigl(\varepsilon_0 + \chi_x\, y
   - \chi_y\, x\bigr)`
   that can be turned into a boundary circulation.  Fibers are
   the only practical choice, and their :math:`O(h^{2})`
   discretization error is accepted as the price of admission.

3. *Cracking* makes the integration domain itself depend on the
   strain field: the effective region is
   :math:`\{(x, y) \in \Omega \,:\, \varepsilon(x, y)
   > \varepsilon_{\mathrm{cr}}\}`.
   The polygon must be re-clipped at every iteration, and the
   integrand is already non-linear — so one is firmly in fiber
   territory.

Conversely, a fiber-based calculation of :math:`I_x` or
:math:`Z_x` would waste precision: it would converge to the
polygon-exact value only in the limit :math:`h \to 0`, at a cost
growing as :math:`1/h^{2}`.  For the quantities collected in
:class:`SectionProperties`, the polygon method is both faster
*and* more accurate — an unusual combination that justifies
keeping the two machineries strictly separate.


Where the two methods meet
--------------------------

The two methods communicate through the **centroid** and the
**principal axes** reported in :class:`SectionProperties`.  The
solver translates the reference point of the applied action
:math:`(N, M_x, M_y)` to the centroid and aligns the strain
field with the principal frame when needed; the fiber
integration then proceeds in a frame where :math:`I_{xy} = 0`.
Accuracy of the centroid — which comes from the polygon method
— therefore propagates into the accuracy of every solver output.
A mismatch of a few microns in :math:`y_G` has negligible effect
on an :math:`N`–:math:`M` envelope, but it would be visible in a
first-cracking computation where the governing quantity is
:math:`\varepsilon_{\mathrm{cr}}
= f_{\mathrm{ctm}} / E_{\mathrm{cm}}` — a small strain is
amplified by the lever arm, so the centroid must be right.

For the same reason, the kern and the inertia ellipse — both
derived from :class:`SectionProperties` — are honest exact
surrogates of the *elastic* behaviour of the homogenized section
and can be used as early-stage checks before running the
non-linear solver.


API reference
=============

.. autoclass:: gensec.geometry.properties.HomogenizedRebar
   :members:

.. autoclass:: gensec.geometry.properties.SectionProperties
   :members:

.. autofunction::
   gensec.geometry.properties.compute_section_properties

.. autofunction::
   gensec.geometry.properties.compute_inertia_ellipse

.. autofunction::
   gensec.geometry.properties.compute_kern_polygon


Area, centroid and second-moments
=================================

The area of the homogenized section is

.. math::

    A_{\mathrm{id}} = n_{\mathrm{bulk}}\!\int_{\Omega}\!\mathrm{d}A
        + \sum_i \bigl(n_{s,i} - n_{\mathrm{bulk}}\bigr) A_{s,i},

where :math:`\Omega` is the polygon and the sum runs over all
point fibers.  Static moments and second-moments follow the same
additive structure.  In practice the polygon integrals are
evaluated in closed form via Green's theorem on every ring
(exterior CCW, holes CW):

.. math::

    A &= \tfrac{1}{2}\sum_i c_i,
    \qquad
    S_x = \tfrac{1}{6}\sum_i (y_i + y_{i+1})\, c_i,\\[2pt]
    S_y &= \tfrac{1}{6}\sum_i (x_i + x_{i+1})\, c_i,\\[2pt]
    I_{xx,O} &= \tfrac{1}{12}\sum_i
                (y_i^2 + y_i y_{i+1} + y_{i+1}^2)\, c_i,\\[2pt]
    I_{yy,O} &= \tfrac{1}{12}\sum_i
                (x_i^2 + x_i x_{i+1} + x_{i+1}^2)\, c_i,\\[2pt]
    I_{xy,O} &= \tfrac{1}{24}\sum_i
                (x_i y_{i+1} + 2 x_i y_i + 2 x_{i+1} y_{i+1}
                 + x_{i+1} y_i)\, c_i,

with :math:`c_i = x_i y_{i+1} - x_{i+1} y_i`.

Centroidal quantities are obtained by Huygens' translation:

.. math::

    I_x = I_{xx,O} - A y_G^2,\qquad
    I_y = I_{yy,O} - A x_G^2,\qquad
    I_{xy} = I_{xy,O} - A x_G y_G.


Principal axes and radii of gyration
====================================

The principal centroidal second-moments and their rotation angle
are the eigenvalues/eigenvectors of the centroidal inertia
tensor:

.. math::

    I_{\xi,\eta} = \frac{I_x + I_y}{2}
                 \pm \sqrt{\!\left(\frac{I_x - I_y}{2}\right)^{2}
                          + I_{xy}^{2}},\qquad
    \alpha = \tfrac{1}{2}\operatorname{atan2}\bigl(
        -2 I_{xy},\, I_x - I_y\bigr),

with :math:`I_\xi \ge I_\eta` and
:math:`\alpha \in (-\pi/2, +\pi/2]`.  Radii of gyration are
:math:`\rho_\bullet = \sqrt{I_\bullet / A_{\mathrm{id}}}`.

.. note::

    When :math:`I_x \approx I_y` and :math:`I_{xy} \approx 0`
    (isotropic inertia tensor — circle, annulus, square, regular
    polygons), :math:`\alpha` is mathematically indeterminate.
    GenSec detects this case by checking that the Mohr radius is
    below :math:`10^{-10}` times the mean of
    :math:`I_x, I_y`, and snaps :math:`\alpha = 0` so the
    principal axes coincide with the user axes.


Extreme-fiber distances and elastic section moduli
==================================================

For each of the four reference axes — user :math:`x, y` and
principal :math:`\xi, \eta` — GenSec reports the two extreme
fiber distances from the centroid:

.. math::

    c_y^{\mathrm{top}}
    = \max_i (y_i - y_G)^+,\qquad
    c_y^{\mathrm{bot}}
    = \max_i -(y_i - y_G)^-,

and analogously for :math:`x, \xi, \eta`.  Both polygon vertices
(exterior and interior rings) **and** point fibers are included
in the maximum, so that a rebar sitting outside the polygon
contour is correctly picked up as the outermost fiber.

The elastic section moduli follow the convention
:math:`W = I / c`:

.. math::

    W_x^{\mathrm{top}} = \frac{I_x}{c_y^{\mathrm{top}}},
    \qquad
    W_x^{\mathrm{bot}} = \frac{I_x}{c_y^{\mathrm{bot}}},
    \qquad
    W_\xi^{\pm} = \frac{I_\xi}{c_\eta^{\pm}},
    \qquad
    W_\eta^{\pm} = \frac{I_\eta}{c_\xi^{\pm}}.

Both sides are reported separately: asymmetric sections have
distinct top/bottom moduli, and engineering practice normally
selects the smaller of the two for resistance verifications.


Plastic section moduli
======================

The plastic modulus :math:`Z` of the homogenized section is
computed about each of the four reference axes through the
centroid by locating the **plastic neutral axis** (PNA) — the
line that splits the homogenized area in two equal halves — and
then summing the absolute first moments of the two halves about
it:

.. math::

    A_{\mathrm{id}}^+(t_{\mathrm{pna}})
    = A_{\mathrm{id}}^-(t_{\mathrm{pna}})
    = \frac{A_{\mathrm{id}}}{2},
    \qquad
    Z = \bigl|S_{\mathrm{id}}^+\bigr|
      + \bigl|S_{\mathrm{id}}^-\bigr|.

The PNA is found numerically by bisection on the offset
:math:`t_{\mathrm{pna}}`.  The polygon contribution is evaluated
via Shapely intersection with a half-plane, and rebar
contributions are summed analytically according to which side of
the PNA each rebar falls on.

.. important::

    The *homogenized plastic modulus* is a purely geometric
    quantity; it has direct engineering meaning only for
    mono-material sections (steel, timber, aluminium), where it
    combines with a uniform yield stress to give the plastic
    bending moment :math:`M_{\mathrm{pl}} = Z \sigma_y`.  For
    reinforced concrete and other composite sections the actual
    ultimate moment must be evaluated with the fiber solver — the
    two constitutive laws of concrete and steel are not
    proportional, so the plastic-modulus approach no longer
    applies.  GenSec still reports :math:`Z` in all cases because
    it is a useful diagnostic quantity.


Torsional constant
==================

The attribute :attr:`SectionProperties.I_t` is a placeholder that
is currently always ``None``.  Computing :math:`I_t` for an
arbitrary shape requires solving the St.-Venant warping problem

.. math::

    \nabla^{2}\psi = -2 \quad\text{in }\Omega,\qquad
    \psi = 0 \quad\text{on }\partial\Omega,\qquad
    I_t = 2\int_{\Omega}\!\psi\,\mathrm{d}A,

which in turn requires a 2-D FEM mesher and a Poisson solver.
The solver is planned for a future development phase; the field
is kept in the dataclass now so that JSON exports and
downstream consumers (including the future GUI) have a stable
schema.

For common shapes closed-form formulas are available (rectangle:
:math:`I_t = k_1\,b t^3`; circle: :math:`I_t = \pi R^4 / 2`;
thin-walled closed: Bredt; thin-walled open:
:math:`I_t = \tfrac{1}{3}\sum_i b_i t_i^3`).  These will be
provided as helper functions callable by name once the main
solver lands.


Kern of the cross-section
=========================

The central kern is the antipolar dual of the exterior's convex
hull with respect to the central inertia ellipse.  For each edge
of the hull with outward unit normal :math:`\mathbf{n}` and
signed distance :math:`d > 0` from the centroid, the associated
kern vertex in the principal frame is

.. math::

    \xi_p = -\rho_\eta^{2}\, n_\xi / d,\qquad
    \eta_p = -\rho_\xi^{2}\, n_\eta / d.

.. warning::

    When the exterior is **non-convex**, the algorithm returns
    the kern of the convex hull — this is an **upper bound** of
    the true kern and may be noticeably too large.  The
    :attr:`SectionProperties.is_convex` flag is provided so that
    the plot and the report can warn the user explicitly.


Example
=======

.. code-block:: python

    from shapely.geometry import box
    from gensec.geometry.properties import (
        compute_section_properties, HomogenizedRebar,
    )

    # RC rectangle 300×500 with 4φ20 symmetric rebars.
    poly = box(0, 0, 300, 500)
    As = 3.14159 * 20 ** 2 / 4.0
    rebars = [
        HomogenizedRebar( 40,  40, As, 200_000.0),
        HomogenizedRebar(260,  40, As, 200_000.0),
        HomogenizedRebar( 40, 460, As, 200_000.0),
        HomogenizedRebar(260, 460, As, 200_000.0),
    ]
    props = compute_section_properties(
        poly, rebars=rebars,
        E_bulk=30_000.0,   # E_cm of concrete [MPa]
    )
    print(f"A_id = {props.area:.1f} mm^2")
    print(f"Ix   = {props.Ix:.3e} mm^4")
    print(f"Z_x  = {props.Z_x:.3e} mm^3")
