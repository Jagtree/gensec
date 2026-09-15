.. _integration_methods:

======================================================
Integration methods: polygon vs fiber
======================================================

GenSec evaluates two very different families of integrals over
the cross-section :math:`\Omega`.  The geometric quantities
(area, static moments, second-moments, elastic and plastic
moduli) involve integrands that are **polynomial in the
coordinates and independent of the deformation state**, and
therefore admit closed-form evaluation via Green's theorem on a
polygon.  The mechanical response quantities (axial force,
bending moments, moment–curvature and interaction surfaces)
involve integrands of the form :math:`\sigma(\varepsilon(x, y))`
that are **non-linear in the coordinates**, typically only
:math:`C^0` at the branch points of the constitutive law, and
dependent on the current deformation state; they admit no
closed form and must be evaluated by numerical quadrature, which
in this context is universally done by **fiber summation**.

GenSec uses each method exclusively where it is best suited.
This page documents the two methods, their accuracy, their cost,
and the reasons they are not mixed.


The polygonal method (Green's theorem)
======================================

Formulation
-----------

For any polynomial integrand :math:`P(x, y)` and any simply
connected region :math:`\Omega` bounded by a closed curve
:math:`\partial\Omega`, Green's theorem converts the 2-D area
integral into a 1-D line integral:

.. math::

    \iint_{\Omega} P(x, y)\,\mathrm{d}A
    \;=\; \oint_{\partial\Omega} Q(x, y)\,\mathrm{d}l,

with :math:`Q` any anti-derivative of :math:`P` with respect to
one of the coordinates.  When :math:`\partial\Omega` is a polygon
with vertices :math:`(x_i, y_i),\; i = 1, \dots, n`
(indexed modulo :math:`n`), the line integral reduces to a sum
of elementary integrals over straight segments, each of which
is itself a *closed-form* polynomial expression.

For the six moment integrals needed to compute
:math:`A, S_x, S_y, I_{xx,O}, I_{yy,O}, I_{xy,O}` of the polygon
about the origin, one obtains

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

with :math:`c_i = x_i y_{i+1} - x_{i+1} y_i`.  Rings with holes
are handled by *summing* the integrals of every ring, using
counter-clockwise orientation for the exterior and clockwise for
every hole, so that the signed contribution of a hole subtracts
automatically.

Accuracy
--------

The formulas above are **exact**: for any polygon with vertices
given exactly, the result is the true value of the integral, up
to floating-point round-off of order :math:`10^{-15}` relative
for standard double-precision arithmetic.

When the section boundary is a smooth curve that has been
approximated by a polygon of :math:`N` vertices — for instance a
circle of radius :math:`R` discretised as a regular
:math:`N`-gon — the *only* error is the boundary approximation
error.  For a regular :math:`N`-gon inscribed in a circle,

.. math::

    A_N = \frac{N R^2}{2}\sin\!\frac{2\pi}{N}
        = \pi R^2 \,\frac{\sin(2\pi/N)}{2\pi/N}
        \;\xrightarrow[N\to\infty]{}\;
        \pi R^2 \Bigl(1 - \frac{2\pi^2}{3 N^2} + O(N^{-4})\Bigr),

so the relative error of the area decays as
:math:`\mathcal{O}(N^{-2})`.  The same rate governs all
second-moment integrals.  Numerically (see :ref:`the table below
<integration_disc_table>`), 64 vertices give an error of
:math:`\approx 1.6\times 10^{-3}`, 256 vertices give
:math:`\approx 10^{-4}`.

Cost
----

Each moment integral is a single pass over the
:math:`n`-vertex boundary, i.e. :math:`\mathcal{O}(n)` in time
and space.  A typical RC cross-section has
:math:`n \lesssim 100` even for complex boundaries, so all
geometric properties are computed in milliseconds.  Full
computation of :func:`compute_section_properties` (area,
moments, principal axes, :math:`W`, kern) on a 100-vertex
polygon is :math:`\lesssim 3\,\mathrm{ms}` on a 2024 laptop
CPU.


The fiber method
================

Formulation
-----------

When the integrand depends on the deformation state through a
constitutive law :math:`\sigma = \hat\sigma(\varepsilon)`, and
the strain is a linear function of the fiber coordinates,

.. math::

    \varepsilon(y, z) = \varepsilon_0 + \chi_y\, y - \chi_z\, z,

the resultant axial force and bending moments are integrals of
the form

.. math::

    N(\mathbf{e}) &= \iint_{\Omega}
        \hat\sigma\!\bigl(\varepsilon(y, z)\bigr)\,\mathrm{d}A,\\
    M_y(\mathbf{e}) &= \iint_{\Omega}
        \hat\sigma\!\bigl(\varepsilon(y, z)\bigr)\, y\,\mathrm{d}A,\\
    M_z(\mathbf{e}) &= -\iint_{\Omega}
        \hat\sigma\!\bigl(\varepsilon(y, z)\bigr)\, z\,\mathrm{d}A,

with :math:`\mathbf{e} = (\varepsilon_0, \chi_y, \chi_z)`.  The
integrand is **not polynomial** in :math:`y, z`; it has
piecewise-smooth branches separated by the pivot points of the
constitutive law (e.g. :math:`\varepsilon_{cr}` for concrete
cracking, :math:`\varepsilon_{c2}, \varepsilon_{cu2}` for EC2
parabola–rectangle, :math:`\varepsilon_{sy}, \varepsilon_{su}`
for elastoplastic steel).  Green's theorem cannot be applied.

The fiber method partitions :math:`\Omega` into a finite set of
cells :math:`\{\Omega_k\}` with centroids :math:`(y_k, z_k)` and
areas :math:`A_k`, and approximates each integral by a
**midpoint sum**:

.. math::

    N \;\approx\; \sum_k
        \hat\sigma\!\bigl(\varepsilon(y_k, z_k)\bigr)\, A_k.

Point fibers (rebars, tendons, FRP strips) are added
analytically: they are not quadrature points but *exact* point
contributions :math:`A_s\,\hat\sigma_s(\varepsilon(y_s, z_s))`.
Embedded rebars additionally subtract the bulk contribution at
the same location to avoid double-counting the area they
physically displace, so the total contribution is
:math:`A_s\,[\hat\sigma_s - \hat\sigma_c(\varepsilon_s)]`.


Boundary fitting — what GenSec actually does
--------------------------------------------

Naïve fiber meshes — a rectangular :math:`N \times N` grid with
every cell contributing either "fully in" or "fully out" based on
whether the cell centre lies inside the polygon — suffer from a
boundary-fitting pathology: the sign of the local quadrature
error depends on how each boundary cell happens to straddle
:math:`\partial\Omega`, and the pattern changes erratically with
:math:`N`.  Convergence drops to :math:`\mathcal{O}(h)` or
worse, and can be non-monotone (see
:ref:`the disc table below <integration_disc_table>`).

GenSec avoids this completely.  The grid mesher in
:mod:`gensec.geometry.geometry` **intersects every cell with the
polygon using Shapely**, and each fiber inherits the *centroid
and area of the intersection*:

.. code-block:: python

    cell    = box(x0, y0, x1, y1)
    clipped = polygon.intersection(cell)
    if not clipped.is_empty:
        x_fib, y_fib = clipped.centroid.x, clipped.centroid.y
        A_fib        = clipped.area

This is exact up to the polygon's own discretisation of the
boundary.  Every fiber sits on a polygonal sub-domain, and the
midpoint rule recovers its theoretical :math:`\mathcal{O}(h^2)`
rate — monotonic, without the fluctuations seen in the naïve
version.  The clipping is a one-time cost at section assembly:
the subsequent Newton iterations in the solver operate on the
pre-computed fiber arrays :math:`(x_k, y_k, A_k)` and never
invoke Shapely.

For sections where a structured grid is inadequate (complex
boundaries, locally refined regions), GenSec also supports a
constrained Delaunay mesh via the ``triangle`` backend; the
accuracy discussion below applies identically.

Accuracy — branch points of the constitutive law
------------------------------------------------

Even with perfect boundary fitting, the constitutive law
introduces a second source of error.  Where :math:`\hat\sigma`
is only :math:`C^0` — the parabola/plateau transition at
:math:`\varepsilon = \varepsilon_{c2}`, the yield point at
:math:`\varepsilon = \varepsilon_{sy}`, the tension cut-off at
:math:`\varepsilon = 0` for concrete — the midpoint rule has a
local error that degrades to :math:`\mathcal{O}(h)` inside a
strip of width :math:`\mathcal{O}(h)` around each iso-strain
line :math:`\varepsilon(y, z) = \varepsilon_b`.  Integrating the
strip over its :math:`\mathcal{O}(h)` width gives an
:math:`\mathcal{O}(h^2)` contribution to the total error, so the
global rate is preserved; the constant is larger than for a
smooth integrand, but the asymptotic behaviour is still
quadratic.

The RC benchmark in :ref:`integration_rc_table` confirms this:
convergence is :math:`\mathcal{O}(h^2)` with a constant that
would be no worse if the integrand were polynomial.

Accuracy — coupling with the non-linear solver
----------------------------------------------

The fiber sum is evaluated many times inside a Newton or
bisection loop on :math:`\mathbf{e}`.  Residual errors in the
quadrature propagate into the equilibrium solution, so
over-coarse meshes manifest as spurious oscillation or failure
of the non-linear solver, not just a slightly wrong answer.  In
practice, a mesh size of a few millimetres on an RC column
gives :math:`10^{-5}`–:math:`10^{-4}` relative error on
:math:`(N, M_y, M_z)`, which is more than enough: the
engineering uncertainty on material constants is several orders
of magnitude larger.


Why not use higher-order quadrature (Gauss, Simpson)?
-----------------------------------------------------

For a *smooth* integrand on a rectangular domain, quadrature
rules of order :math:`p` achieve error :math:`\mathcal{O}(h^{p})`
per cell with a fixed number of evaluation points per cell,
vastly outperforming midpoint (:math:`p = 2`).  Gauss–Legendre
with :math:`k` points per axis, for instance, integrates
polynomials of degree :math:`2k-1` exactly.  One might ask why
GenSec does not use such a rule.

The answer is in the constitutive law.

**Higher-order rules gain nothing across branch points.**  Their
exactness proofs assume the integrand is a polynomial — or at
least :math:`C^p`-smooth — over the cell.  Across an iso-strain
line where :math:`\hat\sigma` is only :math:`C^0`, every
quadrature rule has local error :math:`\mathcal{O}(h^2)`
regardless of its nominal order.  The total error then
integrates as :math:`\mathcal{O}(h^2)` globally, the same as
midpoint.  Investing in Gauss points buys nothing on the strips
that dominate the error, and pays extra evaluations of
:math:`\hat\sigma` everywhere else.

**Higher-order rules can be strictly worse.**  When
:math:`\hat\sigma` is non-smooth *and* non-monotonic — for
instance the descending branch of a concrete law past
:math:`\varepsilon_{cu2}`, or a damage model with load reversal
— Gauss rules can produce spurious cancellations between points
on opposite sides of the discontinuity.  Midpoint, with a single
sample per cell, cannot exhibit this pathology; it systematically
over- or under-estimates each cell consistently with the
iso-strain structure.

**Mesh refinement near branch points is the right remedy.**  If
the user truly needs higher accuracy near a transition line, the
right tool is local mesh refinement (smaller cells in the strip
where :math:`\varepsilon \approx \varepsilon_b`), not a
higher-order rule on the coarse mesh.  GenSec's triangular
mesher supports non-uniform sizing for exactly this purpose;
cells can be refined to align with the expected branch-point
bands.

The short answer, then, is that midpoint on a conforming mesh is
the optimal rule for this class of integrands: it is robust
against non-smoothness, achieves the best asymptotic rate any
quadrature can achieve here, and offers the most predictable
behaviour inside the non-linear solver.


Cost
----

Each evaluation of :math:`(N, M_y, M_z)` is
:math:`\mathcal{O}(n_{\mathrm{fib}})` where
:math:`n_{\mathrm{fib}}` is the total fiber count.  A full
verification — build-up of a 3-D interaction surface with
:math:`n_{\mathrm{pts}}` sampled points, each requiring tens of
Newton iterations — therefore costs
:math:`\mathcal{O}(n_{\mathrm{pts}} \cdot n_{\mathrm{iter}}
\cdot n_{\mathrm{fib}})` constitutive-law evaluations.  See
:ref:`integration_rc_table` for per-call timings on a
representative RC section, and :ref:`integration_engineering`
for the full-surface cost budget.


Why the two methods are not mixed
=================================

Two consistency arguments and one efficiency argument.

**Consistency.**  All the geometric quantities
:math:`A, S, I, W, Z` derive from polynomial integrands on the
same domain :math:`\Omega`.  If the area were computed by fiber
summation and the moments of inertia by Green's theorem, the
relation :math:`\rho = \sqrt{I/A}` would mix two
incommensurable approximations; the reported radius of gyration
would inherit two different error contributions with different
rates.

Similarly, the plastic neutral axis in :math:`Z` is found by
splitting :math:`A_{\mathrm{id}}` in half.  If :math:`A` came
from a fiber sum with :math:`10^{-3}` relative error, the PNA
position would have matching error, and the resulting :math:`Z`
would lose the exactness property that Green's theorem
naturally provides on a polygon.

**Regularity.**  The fiber method is designed to tolerate
constitutive-law branch points.  Applying it to a polynomial
integrand is not wrong, just wasteful: the branch-point
machinery is carrying weight it does not need.  Conversely,
Green's theorem is exact for polynomial integrands and strictly
inapplicable to non-polynomial ones — there is no way to bend
the formulation to accept
:math:`\sigma(\varepsilon(y, z))`.

**Efficiency.**  A fiber mesh of :math:`10^{4}` cells inside a
rectangular polygon costs :math:`10^{4}` floating-point
operations per moment integral; Green's theorem on the same
polygon costs 4.  For quantities that never depend on the
deformation state, there is no argument for the more expensive
option.


GenSec's allocation of methods
==============================

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Quantity
     - Method
     - Reason
   * - :math:`A, S_x, S_y, x_G, y_G`
     - Polygonal
     - Polynomial integrand; exact on a polygon.
   * - :math:`I_x, I_y, I_{xy}, I_\xi, I_\eta, \alpha`
     - Polygonal
     - Polynomial integrand; exact on a polygon.
   * - :math:`c_y^{\pm}, c_x^{\pm}, c_\xi^{\pm}, c_\eta^{\pm}`
     - Vertex enumeration
     - Deterministic max over the vertex + rebar set.
   * - :math:`W_x^{\pm}, W_y^{\pm}, W_\xi^{\pm}, W_\eta^{\pm}`
     - Polygonal
     - Derived from :math:`I / c`; inherits the exactness of
       the polygonal moments.
   * - :math:`Z_x, Z_y, Z_\xi, Z_\eta`
     - Polygonal + bisection
     - Polygon split by a half-plane (exact); PNA located by
       bisection on a monotone objective; first-moment integrals
       computed in closed form on each half.
   * - :math:`(N, M_y, M_z)` for a given :math:`\mathbf{e}`
     - Fiber (midpoint, Shapely-clipped mesh)
     - Non-polynomial integrand with constitutive-law branch
       points.
   * - Moment–curvature :math:`M(\chi)`,
       failure surface, utilisation ratio
     - Fiber + non-linear solver
     - Built on top of the fiber response.
   * - Torsional constant :math:`I_t`
     - (Future) St.-Venant FEM
     - Requires solving
       :math:`\nabla^2 \psi = -2` with
       :math:`\psi = 0` on :math:`\partial\Omega`; neither
       polynomial nor expressible through fiber sums.


Numerical illustration
======================

.. _integration_disc_table:

Disc — polygonal vs fiber (with and without clipping)
-----------------------------------------------------

Disc of radius :math:`R = 500\,\mathrm{mm}` (exact
:math:`A = \pi R^2` and :math:`I = \pi R^4 / 4`).  For the
polygonal method, :math:`N` is the number of boundary vertices
of the regular :math:`N`-gon approximating the disc.  For the
fiber methods, :math:`N` is the number of cells per side of the
bounding square grid.  The naïve column uses a cell-centre
mask (in/out); the clipped column uses Shapely intersection to
assign each fiber the area and centroid of its cell ∩ disc.

.. list-table:: Relative error on the second-moment :math:`I`.
   :header-rows: 1
   :widths: 10 22 22 24

   * - :math:`N`
     - Polygonal :math:`N`-gon
     - Fiber, naïve mask
     - Fiber, Shapely clipping
   * - 8
     - :math:`1.88\!\times\!10^{-1}`
     - :math:`5.94\!\times\!10^{-2}`
     - :math:`1.96\!\times\!10^{-2}`
   * - 16
     - :math:`5.02\!\times\!10^{-2}`
     - :math:`6.81\!\times\!10^{-2}`
     - :math:`5.06\!\times\!10^{-3}`
   * - 32
     - :math:`1.28\!\times\!10^{-2}`
     - :math:`1.90\!\times\!10^{-2}`
     - :math:`1.29\!\times\!10^{-3}`
   * - 64
     - :math:`3.21\!\times\!10^{-3}`
     - :math:`6.84\!\times\!10^{-3}`
     - :math:`3.26\!\times\!10^{-4}`
   * - 128
     - :math:`8.03\!\times\!10^{-4}`
     - :math:`3.73\!\times\!10^{-3}`
     - :math:`8.42\!\times\!10^{-5}`
   * - 256
     - :math:`2.01\!\times\!10^{-4}`
     - :math:`1.50\!\times\!10^{-4}`
     - :math:`2.34\!\times\!10^{-5}`
   * - 512
     - :math:`5.02\!\times\!10^{-5}`
     - :math:`4.45\!\times\!10^{-5}`
     -
   * - 1024
     - :math:`1.26\!\times\!10^{-5}`
     - :math:`1.03\!\times\!10^{-4}`
     -

Three observations.

The polygonal column shows a clean, monotone
:math:`\mathcal{O}(N^{-2})` decay: every doubling of :math:`N`
reduces the error by a factor close to 4 throughout.  The only
limit is boundary approximation; it is predictable and stable.

The naïve mask column shows *non-monotone* convergence with
fluctuations caused by the boundary-fitting effect: the sign and
magnitude of the local error depend on how each cell straddles
the disc boundary, and that pattern changes erratically with
:math:`N`.  At :math:`N = 1024` the error is actually *larger*
than at :math:`N = 512`.  This is the pathology GenSec avoids.

The Shapely-clipped column recovers the clean
:math:`\mathcal{O}(h^2)` rate of the clipped-mesh midpoint rule.
The ratio between successive rows is again close to 4.  For a
given :math:`N`, the clipped fiber is about four times more
accurate than the polygonal :math:`N`-gon; this is because the
:math:`N \times N` grid produces :math:`\mathcal{O}(N^2)` cells
instead of :math:`\mathcal{O}(N)` boundary vertices.


.. _integration_rc_table:

RC benchmark — fiber method vs exact PR integration
---------------------------------------------------

Rectangular section :math:`B \times H = 300 \times 500\,
\mathrm{mm}`, centred on the origin, cover 40 mm.  Two rebars
φ12 at :math:`y = +210\,\mathrm{mm}` (compression side) and
three rebars φ20 at :math:`y = -210\,\mathrm{mm}` (tension side),
all embedded.  Concrete EC2 parabola–rectangle with
:math:`f_{cd} = 17\,\mathrm{MPa}`,
:math:`\varepsilon_{c2} = 0.002`,
:math:`\varepsilon_{cu2} = 0.0035`, no tension capacity.  Steel
B450C (:math:`f_{yd} = 391.3\,\mathrm{MPa}`,
:math:`E_s = 200\,000\,\mathrm{MPa}`), symmetric elastoplastic
law.  Prescribed ultimate strain profile:
:math:`\varepsilon_{\mathrm{top}} = \varepsilon_{cu2} = 0.0035`,
neutral axis at :math:`y = 50\,\mathrm{mm}` (NA depth from
compressed edge :math:`x = 200\,\mathrm{mm}`), giving
:math:`\varepsilon_0 = -8.75\!\times\!10^{-4}`,
:math:`\chi = 1.75\!\times\!10^{-5}\,\mathrm{mm}^{-1}`.

The concrete integrand contains all three regions in the
compressed zone — tension cut-off from :math:`y = -250` to
:math:`y = 50`, parabola from :math:`y = 50` to :math:`y =
164.29`, plateau from :math:`y = 164.29` to :math:`y = 250` —
so both of the concrete branch points at :math:`\varepsilon = 0`
and :math:`\varepsilon = \varepsilon_{c2}` lie inside the
integration domain.  The tension rebar yields
(:math:`\varepsilon_s = -4.55\!\times\!10^{-3}`); the
compression rebar also yields
(:math:`\varepsilon_s = +2.80\!\times\!10^{-3}`).

Exact reference obtained by closed-form integration of each
region of the concrete law and exact point evaluation of the
rebar contributions (with embedded bulk subtraction):

.. math::

    N_{\mathrm{exact}} = 541.587\,\mathrm{kN},
    \qquad
    M_{\mathrm{exact}} = 232.961\,\mathrm{kN}\!\cdot\!\mathrm{m}.

.. list-table:: Fiber-sum convergence against the PR analytical
                reference (timings on a 2024 laptop CPU, NumPy
                vectorised).
   :header-rows: 1
   :widths: 7 7 10 12 12 12 12 10

   * - :math:`N_x`
     - :math:`N_y`
     - :math:`n_{\mathrm{fib}}`
     - :math:`N` [kN]
     - :math:`M` [kN·m]
     - :math:`N` rel. err.
     - :math:`M` rel. err.
     - :math:`t` [ms]
   * - 30
     - 50
     - 1\,500
     - 541.956
     - 232.957
     - :math:`6.80\!\times\!10^{-4}`
     - :math:`1.38\!\times\!10^{-5}`
     - 0.12
   * - 60
     - 100
     - 6\,000
     - 541.681
     - 232.960
     - :math:`1.73\!\times\!10^{-4}`
     - :math:`2.47\!\times\!10^{-6}`
     - 0.15
   * - 120
     - 200
     - 24\,000
     - 541.611
     - 232.960
     - :math:`4.31\!\times\!10^{-5}`
     - :math:`6.43\!\times\!10^{-7}`
     - 4.00
   * - 240
     - 400
     - 96\,000
     - 541.593
     - 232.960
     - :math:`1.07\!\times\!10^{-5}`
     - :math:`1.83\!\times\!10^{-7}`
     - 9.63
   * - 480
     - 800
     - 384\,000
     - 541.589
     - 232.960
     - :math:`2.68\!\times\!10^{-6}`
     - :math:`4.38\!\times\!10^{-8}`
     - 26.27
   * - 960
     - 1\,600
     - 1\,536\,000
     - 541.588
     - 232.960
     - :math:`6.71\!\times\!10^{-7}`
     - :math:`1.10\!\times\!10^{-8}`
     - 133.7

The ratio between consecutive rows is :math:`\approx 4` for both
:math:`N` and :math:`M`, confirming the :math:`\mathcal{O}(h^2)`
rate despite the two concrete branch points inside the
integration domain.

A second observation worth noting: up to about 10\,000 fibers
the per-call time is dominated by Python and NumPy setup
overhead rather than by the actual floating-point work.  The
step from 1\,500 to 6\,000 fibers costs essentially nothing in
wall time but buys four times more accuracy; the step from
6\,000 to 24\,000 costs 25× more time for the next factor of 4.
This explains why the default GenSec mesh sizes never go below
2–3 mm: there is no reason to.


Comparison with the EC2 rectangular stress-block approximation
--------------------------------------------------------------

The rectangular stress-block method — used in engineering hand
calculations and in most national annexes to EC2 — replaces the
full parabola–rectangle law by an equivalent uniform stress
:math:`\eta f_{cd}` acting over a reduced depth
:math:`\lambda x` from the compressed edge, with
:math:`\lambda = 0.8`, :math:`\eta = 1.0` for concrete classes
up to C50/60.  Applying it to the same RC section at the same
strain state:

.. math::

    N_{\mathrm{SB}} &= 531.873\,\mathrm{kN}
        \quad(\Delta = -1.79\%\text{ vs PR}),\\
    M_{\mathrm{SB}} &= 233.946\,\mathrm{kN}\!\cdot\!\mathrm{m}
        \quad(\Delta = +0.42\%\text{ vs PR}).

Two points.

The stress block is calibrated to match the bending moment
accurately at ultimate — the reported :math:`0.4\%` error on
:math:`M` is fully within engineering tolerance.  The
:math:`1.8\%` discrepancy on :math:`N`, on the other hand, is
the model-choice price paid for the simplification: the block
has the correct resultant *moment* but not the correct
resultant *axial force*.  This is harmless in pure flexural
verification (where :math:`N = 0` is imposed by equilibrium and
the neutral axis is recalibrated to match) but becomes
significant in flexural–axial verification of columns, where a
:math:`1.8\%` error in :math:`N` shifts the interaction curve
visibly.

The fiber solver, in contrast, integrates the *real* PR law
with relative error of :math:`10^{-4}` at 6\,000 fibers and
:math:`10^{-5}` at 24\,000 fibers — several orders of magnitude
tighter than the stress-block intrinsic error, and orders of
magnitude tighter than any plausible modelling uncertainty on
the constitutive parameters.


.. _integration_engineering:

Engineering cost budget
-----------------------

A concrete example.  A full 3-D N–M\ :sub:`x`\ –M\ :sub:`y`
interaction surface generated by GenSec (:mod:`capacity.py`)
typically samples a few hundred axial-force levels times a few
hundred angular directions, giving
:math:`n_{\mathrm{pts}} \sim 10^{4}` points on the surface.
Each point costs one capacity search that internally runs a
Newton solver with :math:`n_{\mathrm{iter}} \lesssim 15`
iterations, each requiring one fiber integration.

With 6\,000 fibers (~5 mm mesh, error :math:`10^{-4}`):

.. math::

    T_{\mathrm{surface}} \;\approx\;
    10^{4} \cdot 15 \cdot 0.15\,\mathrm{ms}
    \;\approx\; 23\,\mathrm{s}.

With 24\,000 fibers (~2.5 mm mesh, error :math:`6\!\times\!10^{-7}`):

.. math::

    T_{\mathrm{surface}} \;\approx\;
    10^{4} \cdot 15 \cdot 4\,\mathrm{ms}
    \;\approx\; 10\,\mathrm{min}.

The first is the regime GenSec targets by default; the second
is unnecessary for any realistic engineering verification and
is reserved for convergence studies or for sections where
fiber-level accuracy matters (e.g. high-strength concrete with
sharp softening branches, or sections near the boundary of the
stress-block validity range).


Summary
=======

Use Green's theorem on the polygon whenever the integrand is
polynomial and the domain is polygonal: it is exact, cheap, and
stable.  Use fiber summation — midpoint rule on a Shapely-clipped
mesh — whenever the integrand depends on a non-polynomial
constitutive law: clipping eliminates the boundary error, and
the branch points of the constitutive law still allow
:math:`\mathcal{O}(h^2)` global convergence.  Do not mix the two
for the same quantity; the accuracy and consistency arguments
above make the separation worth enforcing.  Do not reach for
higher-order quadrature rules: they provide no benefit on
:math:`C^0` integrands and can in fact be worse.

On a representative RC section with realistic rebars, 6\,000
fibers give :math:`\sim 10^{-4}` accuracy in 0.15 ms per
integration — about three orders of magnitude more accurate than
the EC2 rectangular stress-block hand calculation, and several
orders of magnitude more accurate than the modelling uncertainty
on the constitutive constants themselves.
