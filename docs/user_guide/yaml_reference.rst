.. _yaml_reference:

====================
YAML input reference
====================

A GenSec YAML file has four top-level blocks: ``materials``, ``section``,
``demands`` (optional), and ``combinations`` (optional).  An ``output``
block controls which post-processing artefacts are generated.

.. note::

   The same YAML input file is used by both ``gensec run`` and
   ``gensec analyze``.

   - ``gensec run`` reads all six blocks and uses ``output`` flags to
     control which post-processing artefacts are generated.
   - ``gensec analyze`` reads ``materials``, ``section``, ``demands``
     and ``combinations`` only.  The ``output`` block is **ignored**
     entirely: no resistance domain, no diagrams, no η flags.

   If your goal is only force decomposition (which material carries
   how much force for each combination), ``gensec analyze`` is the
   right command — you don't need to touch the ``output`` block.


``code`` — the normative declaration
------------------------------------

A GenSec model declares **which structural code it is verified to**, and
everything code-dependent is derived from that one key.  The alternative
is to declare that a material is ``custom``, in which case GenSec
derives nothing for it and the input supplies everything.  There is no
third option, and in particular no implicit one.

.. code-block:: yaml

   code: aci318-19          # ec2 | ntc18 | aci318-19 | custom

   materials:
     conc: {type: concrete_ec2_gen1_custom, fck: 28.0}

Why this is a declaration and not a preference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same characteristic strength gives different derived quantities
under different codes.  For the conventional secant modulus of
concrete:

.. math::

   E_{cm} = 22\,000\left(\frac{f_{ck} + 8}{10}\right)^{0.3}
   \qquad \text{EN 1992-1-1:2004 §3.1.3, Table 3.1}

.. math::

   E_{c} = 4700\sqrt{f'_{c}}
   \qquad \text{ACI 318-19 §19.2.2.1, normalweight, SI}

+---------------------+--------------------+--------------------+-----------+
| :math:`f_{ck}` [MPa]| EN 1992-1-1 [MPa]  | ACI 318-19 [MPa]   | gap       |
+=====================+====================+====================+===========+
| 25                  | 31 475.8           | 23 500.0           | +33.9 %   |
+---------------------+--------------------+--------------------+-----------+
| 40                  | 35 220.5           | 29 725.4           | +18.5 %   |
+---------------------+--------------------+--------------------+-----------+
| 60                  | 39 099.9           | 36 406.0           | +7.4 %    |
+---------------------+--------------------+--------------------+-----------+

:math:`f_{ck}` is an **input**; the code is what turns it into a
modulus.  GenSec will not choose one on your behalf.

How the code is resolved
~~~~~~~~~~~~~~~~~~~~~~~~~

There are exactly **two** sources, and no default beneath them:

1. ``code:`` **on the material**.  One material under another code is a
   legitimate model — a precast beam under EN 1992-1-1 carrying an
   ACI 318-19 topping — and each element then answers under its own.
2. ``code:`` **at the top level** of the input file.  The ordinary
   case: declared once for the whole model.

A material with neither is **undeclared**, and
:func:`~gensec.io_yaml.load_yaml` **refuses to build the model**,
naming the material.  The refusal is at the door rather than at the
first quantity that needs a modulus: a deferred refusal would surface
halfway through an analysis, attached to whatever happened to ask
first.

.. important::

   The **material type is not a declaration.**  ``type:
   concrete_ec2_gen1`` names a *constitutive shape* — a curve — and
   not a verification code.  Which partial factors, which conventional
   modulus and which creep law accompany that curve are separate
   facts, and a model may legitimately pair it with a code that did
   not invent it.  Reading the code out of a concrete type name would
   also be a rule that only works for the sections somebody happened
   to think of first: nothing says a GenSec section is concrete at
   all.

   Inference from the type was tried, and removed for those two
   reasons.  If you want the Eurocode, write ``code: ec2``.

Accepted values
~~~~~~~~~~~~~~~~

+----------------+----------------------------------------------------+
| Value          | Meaning                                            |
+================+====================================================+
| ``ec2``        | EN 1992-1-1:2004.  Aliases: ``en1992``,            |
|                | ``en1992-1-1``, ``eurocode2``, ``eurocode``.       |
+----------------+----------------------------------------------------+
| ``ntc18``      | NTC 2018 (D.M. 17/01/2018).  For the conventional  |
|                | modulus and for creep and shrinkage §11.2.10.3     |
|                | adopts the Eurocode formulation, so the **value**  |
|                | is identical and only the **citation** differs.    |
|                | Aliases: ``ntc``, ``ntc2018``, ``ntc-18``.         |
+----------------+----------------------------------------------------+
| ``aci318-19``  | ACI 318-19.  Aliases: ``aci``, ``aci318``,         |
|                | ``aci-318``, ``aci318-19m``.                       |
+----------------+----------------------------------------------------+
| ``custom``     | The explicit opt-out: GenSec derives nothing and   |
|                | the input supplies every code-dependent value.     |
|                | Aliases: ``none``, ``user``.                       |
+----------------+----------------------------------------------------+

An unrecognised value is an **error**, never a fallback.  A mistyped
code name that silently became the Eurocode is precisely the failure
this declaration exists to prevent.

There is **no default anywhere**: not for a missing ``code:``, not for
a material type that happens to name one, and not beneath a timeline.
Every input file must declare.

.. note::

   ``custom`` and *undeclared* both decline to derive, and they are
   deliberately distinct.  ``custom`` means *you said you would supply
   this and you did not*; undeclared means *you never said which code
   to read it under*.  The remedy differs, so the messages differ.

What a code governs
~~~~~~~~~~~~~~~~~~~~

* the conventional secant modulus of every concrete, with the citation
  that appears in reports;
* the creep, shrinkage and relaxation provider, when the model declares
  a time axis.

.. warning::

   The ACI 209R-92 rheological provider is a **declared empty slot** in
   this build.  Asking for creep or shrinkage under ``code: aci318-19``
   raises and names the omission rather than falling back to the
   Eurocode's.  Supply one explicitly with
   ``ConstructionTimeline(rheology=...)``, or keep the model
   time-independent.

Adding a code
~~~~~~~~~~~~~~

Subclass :class:`~gensec.materials.codes.NormativeCode` and call
:func:`~gensec.materials.codes.register_code`.  Nothing in the
resolver, the solver or the loader changes — which is the property that
makes normative agnosticism checkable rather than aspirational.

See ``examples/example_aci318.yaml`` for a complete model under a
non-Eurocode declaration.


``materials`` block
-------------------

A dictionary of named material definitions.  Each key becomes the
identifier used in the ``section`` block to assign materials to the
bulk concrete or to individual rebar layers.

Concrete — direct parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   materials:
     concrete_1:
       type: concrete_ec2_gen1_custom
       fck: 25.0             # characteristic cylinder strength [MPa]
       gamma_c: 1.5          # partial safety factor (default 1.5)
       alpha_cc: 0.85        # long-term coefficient (default 0.85)
       n_parabola: 2.0       # parabolic exponent (default 2.0)
       eps_c2: -0.002        # peak-stress strain (default -0.002)
       eps_cu2: -0.0035      # ultimate strain (default -0.0035)
       fct: 0.0              # tensile strength [MPa] (default 0 = no tension)
       Ec: 0.0               # elastic modulus [MPa] for tension (default 0)

Only ``fck`` is required; all other fields have the defaults shown above.
The legacy type name ``concrete`` is accepted as an alias.

The design compressive strength is computed internally as:

.. math::

   f_{cd} = \alpha_{cc} \, \frac{f_{ck}}{\gamma_c}

When both ``fct`` and ``Ec`` are positive, a linear tension branch is
activated up to the cracking strain :math:`\varepsilon_{ct} = f_{ct}/E_c`.
See :ref:`constitutive_laws` for the full piecewise definition.


Concrete — from EC2 class
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   materials:
     concrete_1:
       type: concrete_ec2_gen1
       class: C30/37          # EC2 class name (C12/15 … C90/105)
       ls: F                  # limit state: 'F' fundamental, 'A' accidental
       loadtype: slow         # 'slow' or 'fast' (default 'slow')
       TypeConc: R            # 'R' rapid, 'N' normal, 'S' slow hardening
       enable_tension: false  # activate linear tension branch (default false)
       tension_fct: fctd      # which fct: 'fctd', 'fctm', or 'fctk'

The legacy type name ``concrete_ec2`` is accepted as an alias.

All Table 3.1 parameters (:math:`f_{cm}`, :math:`f_{ctm}`, :math:`E_{cm}`,
:math:`\varepsilon_{c2}`, :math:`\varepsilon_{cu2}`, :math:`n`, etc.) are
computed automatically by the :mod:`~gensec.materials.ec2_properties` module.

When ``enable_tension: true``, the elastic modulus :math:`E_{cm}` and the
chosen tensile strength (:math:`f_{ctd}`, :math:`f_{ctm}`, or
:math:`f_{ctk,0.05}`) are extracted from the EC2 property object and passed
to the :class:`~gensec.materials.Concrete` constructor.  The ``tension_fct``
field controls which value is used:

+-----------------+-----------------------------------------------------------+
| ``tension_fct`` | Tensile strength used                                     |
+=================+===========================================================+
| ``fctd``        | :math:`f_{ctd,0.05} = \alpha_{ct}\,f_{ctk,0.05}/\gamma_c` |
+-----------------+-----------------------------------------------------------+
| ``fctm``        | :math:`f_{ctm}` (mean tensile strength)                   |
+-----------------+-----------------------------------------------------------+
| ``fctk``        | :math:`f_{ctk,0.05}` (characteristic, 5 % fractile)       |
+-----------------+-----------------------------------------------------------+

.. note::

   Only the **French National Annex** (NF EN 1992-1-1/NA) is implemented.
   Other national annexes may be added in the future.


Reinforcing steel
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   materials:
     steel_1:
       type: steel
       fyk: 450.0               # characteristic yield strength [MPa]
       gamma_s: 1.15            # partial safety factor (default 1.15)
       Es: 200000               # Young's modulus [MPa] (default 200000)
       k_hardening: 1.0         # ft/fy ratio (default 1.0 = perfectly plastic)
       eps_su: 0.01             # ultimate strain (default 0.01)
       works_in_compression: true  # if false, σ = 0 for ε < 0

Only ``fyk`` is required.


Structural steel (EN 10025-2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   materials:
     steel_plate:
       type: steel_en10025
       grade: S355              # S235, S275, or S355
       t: 20                    # plate thickness [mm]
       gamma_s: 1.0             # partial safety factor (default 1.0)

Yield and ultimate strengths are automatically adjusted for plate thickness
according to NF EN 10025-2 Table 7.


Tabulated material
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   materials:
     cfrp:
       type: tabulated
       name: CFRP_sheet
       strains:  [0.0, 0.017]
       stresses: [0.0, 2800.0]

Piecewise-linear interpolation between data points.  Strains outside the
table range produce zero stress (material failure).  The ``strains`` array
must be strictly increasing.


``section`` block
-----------------

Two formats are supported: the **legacy rectangular** format (backward
compatible) and the **generic section** format (arbitrary polygons).


Legacy rectangular section
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   section:
     B: 300                   # width [mm]
     H: 600                   # height [mm]
     bulk_material: concrete_1
     n_fibers_y: 50            # fiber count along height
     n_fibers_x: 20            # explicit x resolution (optional)
     rebars:
       - y: 40
         As: 942.5
         material: steel_1
         n_bars: 3             # optional, for reporting
         diameter: 20          # optional, for reporting [mm]
       - y: 560
         As: 942.5
         material: steel_1

The origin is at the **bottom-left corner** of the bounding box.

Grid resolution is controlled by ``n_fibers_y``, which sets the mesh
size :math:`s = H / n_y`.  When ``n_fibers_x`` is omitted or set to
``1``, an **isotropic grid** is used: the x-resolution is derived from
the same mesh size, giving approximately square cells.  For example,
``B=300, H=600, n_fibers_y=50`` produces ``mesh_size=12`` and
``n_fibers_x=25``, for a total of 1250 fibers.

Set ``n_fibers_x`` explicitly to override the x-resolution (e.g.,
``n_fibers_x: 30`` for a 30×50 grid).

.. note::

   ``RectSection`` is internally a convenience wrapper around
   :class:`~gensec.geometry.GenericSection` with a rectangular
   polygon.  All attributes (``n_fibers_x``, ``dx``, ``polygon``,
   etc.) are available directly on the returned object.


Generic section — parametric shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   section:
     shape: <shape_name>
     params:
       <shape-specific parameters>
     bulk_material: concrete_1
     mesh_size: 15             # target fiber edge length [mm]
     mesh_method: grid         # 'grid' (default) or 'triangle'
     rebars:
       - y: 40
         x: 150               # x required for 2D shapes
         As: 314.16
         material: steel_1

Available shapes and their parameters:

+-----------------+------------------------------------------------------+
| ``shape``       | Required ``params``                                  |
+=================+======================================================+
| ``rect``        | ``B``, ``H``                                         |
+-----------------+------------------------------------------------------+
| ``circle``      | ``D``, ``segments_per_quadrant`` (4n vertices)       |
+-----------------+------------------------------------------------------+
| ``annulus``     | ``D_ext``, ``D_int``, ``segments_per_quadrant``      |
+-----------------+------------------------------------------------------+
| ``tee``         | ``bf``, ``hf``, ``bw``, ``hw``                       |
+-----------------+------------------------------------------------------+
| ``inv_tee``     | ``bf``, ``hf``, ``bw``, ``hw``                       |
+-----------------+------------------------------------------------------+
| ``h_section``   | ``bf``, ``tf``, ``bw``, ``hw``                       |
+-----------------+------------------------------------------------------+
| ``box``         | ``B``, ``H``, ``tw``, ``tf``                         |
+-----------------+------------------------------------------------------+
| ``single_tee``  | ``bf``, ``hf``, ``bw``, ``hw``, ``hw2``              |
+-----------------+------------------------------------------------------+
| ``double_tee``  | ``bf``, ``hf``, ``bw``, ``hw``, ``hw2``, ``s``       |
+-----------------+------------------------------------------------------+

All dimensions in millimetres.  See :mod:`~gensec.geometry.primitives`
for the full parameter description of each factory function.


Generic section — custom polygon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   section:
     shape: custom
     params:
       exterior:
         - [0, 0]
         - [400, 0]
         - [400, 700]
         - [0, 700]
       holes:
         - - [80, 80]
           - [320, 80]
           - [320, 620]
           - [80, 620]
     bulk_material: concrete_1
     mesh_size: 12

Vertex coordinates are ``[x, y]`` in millimetres.  The ``holes`` list is
optional; each hole is a list of vertices defining an interior ring.


Meshing methods
~~~~~~~~~~~~~~~~

- **grid** (default): rectangular grid clipped to the polygon boundary.
  Fast and robust.  Resolution is controlled by ``mesh_size``.
- **triangle**: constrained Delaunay triangulation via the ``triangle``
  library.  Better suited for curved boundaries (circles, annuli).
  Requires ``triangle`` to be installed.


Rebar fields
~~~~~~~~~~~~~

Each rebar entry supports:

+---------------------+---------+------------------------------------------+
| Field               | Default | Description                              |
+=====================+=========+==========================================+
| ``y``               | —       | Vertical coordinate [mm] (required)      |
+---------------------+---------+------------------------------------------+
| ``As``              | —       | Cross-sectional area [mm²].  If omitted, |
|                     |         | computed from ``diameter`` and           |
|                     |         | ``n_bars``.  If both ``As`` and          |
|                     |         | ``diameter`` are given, ``As`` prevails. |
+---------------------+---------+------------------------------------------+
| ``diameter``        | —       | Bar diameter [mm].  When ``As`` is       |
|                     |         | omitted, used to compute                 |
|                     |         | :math:`A_s = n_b \cdot \pi d^2/4`.       |
+---------------------+---------+------------------------------------------+
| ``material``        | —       | Material identifier (required)           |
+---------------------+---------+------------------------------------------+
| ``x``               | centroid| Horizontal coordinate [mm]               |
+---------------------+---------+------------------------------------------+
| ``embedded``        | ``true``| Subtract displaced bulk? See             |
|                     |         | :doc:`sign_conventions`.                 |
+---------------------+---------+------------------------------------------+
| ``n_bars``          | ``1``   | Number of physical bars.  Used for       |
|                     |         | automatic ``As`` computation and         |
|                     |         | reporting.                               |
+---------------------+---------+------------------------------------------+

Either ``As`` or ``diameter`` must be provided.  If only ``diameter``
is given, ``n_bars`` defaults to 1.


``demands`` block
-----------------

A list of individual load demands to verify against the resistance domain.

.. code-block:: yaml

   demands:
     - name: Gravity
       N_kN: -1500
       Mx_kNm: 200
       My_kNm: 0
     - name: Seismic_X
       N_kN: -1200
       Mx_kNm: 280
       My_kNm: 0

Units: ``N_kN`` in kN, ``Mx_kNm`` and ``My_kNm`` in kN·m.

Both ``Mx_kNm`` and ``My_kNm`` are required.  For uniaxial analysis,
set ``My_kNm: 0``.  The legacy field ``M_kNm`` is accepted as an alias
for ``Mx_kNm`` with ``My_kNm = 0``.


``combinations`` block
----------------------

Groups of load triples under a single name.  Typical use case: the same
section is used on multiple columns in a building, each with its own
internal forces.

.. code-block:: yaml

   combinations:
     - name: Seismic_X_envelope
       demands:
         - {N_kN: -1800, Mx_kNm: 180, My_kNm: 30}
         - {N_kN: -1500, Mx_kNm: 250, My_kNm: 45}
         - {N_kN: -1200, Mx_kNm: 300, My_kNm: 20}

GenSec reports the utilization ratio :math:`\eta` for every triple, plus
:math:`\eta_{\max}` per combination.


``output`` block
----------------

Controls optional post-processing.  All fields are optional and
default to the values shown below.

.. code-block:: yaml

   output:
     # ── Utilization ratio flags ──
     # 3-D family (operate in anisotropy-corrected normalised space)
     eta_norm: true              # alpha: linear distance to boundary (default: true)
     eta_norm_beta: true         # composite ratio F_SU/(F_SU+d_min) (default: true)
     eta_norm_ray: false         # ray-cast from origin (default: false)
     # 2-D family (operate in (Mx,My) plane at fixed N)
     eta_2D: false               # ray in plane at demand's N (default: false)
     # Path metrics (staged combinations)
     eta_path_norm_ray: false    # 3-D ray-cast staged (default: false)
     eta_path_norm_beta: false   # composite ratio along stage segment (default: false)
     eta_path_2D: false          # 2-D ray staged (default: false)
     delta_N_tol: 0.03           # ΔN tolerance for eta_path_2D (default: 0.03)

     # ── Domain generation ──
     generate_mx_my: false       # produce Mx-My contour diagrams (default: false)
     generate_3d_surface: false  # produce 3-D resistance surface (default: false)
     n_angles_mx_my: 144         # angular resolution for Mx-My (default: 144)

     # ── Moment-curvature and ductility ──
     generate_moment_curvature: true      # M-χ diagrams per N level (default: true)
     generate_polar_ductility: true       # polar ductility plot (default: true)
     generate_3d_moment_curvature: true   # 3D M-χ-N surface (default: true)

     # ── N-level strategy ──
     n_levels_mode: demands     # "demands", "auto", or "explicit" (default: "demands")
     # n_levels_count: 10       # for "auto" mode
     # n_levels_values: [-3000, -1500, 0]  # for "explicit" mode [kN]

     # ── Section/state figure rendering ──
     show_rebar_labels: true    # per-rebar name/value annotations (default: true)

``show_rebar_labels`` controls the per-rebar text annotation (index,
plus strain/stress value where applicable) drawn on
``section_geometry.png``, ``section_<demand>.png`` and
``state_<demand>.png``.  The rebar markers themselves are always
drawn; set to ``false`` for heavily reinforced sections where the
labels overlap and make the figure unreadable.

Utilization metrics
~~~~~~~~~~~~~~~~~~~

GenSec computes up to seven utilization metrics, organised in two
geometric families plus path-aware variants for staged loading.
Each metric answers a *different* geometric question: they are
complementary, not redundant, and no strict ordering between them
holds in general.  All metrics share the convention
:math:`\eta \le 1` safe / :math:`\eta > 1` unsafe; ``verified`` is
decided by the worst of the enabled metrics.

**3-D family** — operates in *anisotropy-corrected normalised
space*: the axial axis is rescaled by
:math:`u_x = \Delta M_x / \Delta N` and the :math:`M_y` axis by
:math:`v_y = \Delta M_x / \Delta M_y`, so that the bounding box of
the resistance domain has the same numeric extent on every axis.
Euclidean distances in this space are physically meaningful even
for strongly anisotropic sections (walls, slabs, T-beams).

* ``eta_norm`` (alpha) — Linear distance from the demand to the
  nearest boundary face, expressed as a fraction of the Chebyshev
  radius :math:`D_{\max}` of the normalised domain:
  :math:`\eta = 1 - d_{\min}/D_{\max}` (interior) or
  :math:`\eta = 1 + d_{\min}/D_{\max}` (exterior).  A true
  geometric distance: monotone in proximity to the boundary,
  reaches 0 at the Chebyshev centre, 1 on the boundary.  The
  recommended principal 3-D metric.

* ``eta_norm_beta`` — Composite ratio
  :math:`F_{SU} / (F_{SU} + d_{\min})`, mixing the demand norm
  :math:`F_{SU}` and the face-distance :math:`d_{\min}`.  Not a
  pure distance: a ratio that grows both with proximity to the
  boundary and with the demand magnitude.  The semantic reading
  is "sensitivity to perturbation in proportion to the demand
  magnitude".  Useful for cross-software validation, since
  equivalent formulations are reported by some commercial RC
  software.

* ``eta_norm_ray`` — Ray-cast from the origin to the demand in
  normalised space:
  :math:`\eta = |\mathbf{S}_u| / |\mathbf{R}_u|`.  Answers
  "if I scale all three force components proportionally, when does
  the demand exit the domain?".  Linear in demand magnitude along
  any fixed ray.

**2-D family** — operates in the :math:`(M_x, M_y)` plane at the
demand's :math:`N`.  Both axes share physical units, no
normalisation needed.

* ``eta_2D`` — Ray-cast from the origin in the :math:`(M_x, M_y)`
  plane to the contour at the demand's :math:`N`.  Useful for
  flexural verifications at fixed axial force.  Requires a
  biaxial section.

**Path metrics** (staged combinations only):

* ``eta_path_norm_ray`` — 3-D ray-cast in normalised space, from the
  previous stage's cumulative demand to the current one.  
  Path-direction-aware analogue of ``eta_norm_ray``.

* ``eta_path_norm_beta`` — Composite ratio along the stage
  segment :math:`B \to T`:
  :math:`L / (L + d_{\text{seg}})`, with :math:`L = |T - B|` and
  :math:`d_{\text{seg}}` the minimum signed distance of the
  segment from the boundary.  Path-direction-aware analogue of
  ``eta_norm_beta``.  Particularly informative in seismic
  verification: when gravity is applied as a first stage and
  seismic action as a second stage, this metric flags the
  "structure already at the limit before the seismic increment"
  case distinctly from ``eta_path_norm_ray``.

* ``eta_path_2D`` — 2-D variant on the Mx-My contour at
  cumulative :math:`N`.  Skipped when the axial-force change
  between stages exceeds ``delta_N_tol``.

The default flags reflect the recommendation that ``eta_norm`` and
``eta_norm_beta`` are the principal metrics for routine
verification, while the others are opt-in for specific analyses.
Path-aware metrics are off by default; if you process a staged
combination without enabling at least one ``eta_path_*`` flag,
GenSec issues an informational warning, since the resulting
report would only contain point metrics at each cumulative state
and would not characterise the load history.

The ``generate_moment_curvature``, ``generate_polar_ductility``, and
``generate_3d_moment_curvature`` flags control whether the
computationally expensive moment-curvature diagrams and related outputs
are produced.  Setting all three to ``false`` significantly reduces
run time for cases where only the resistance domain and demand
verification are needed.

.. note::

   When ``eta_2D`` is enabled, GenSec generates an Mx-My contour (via
   :meth:`~gensec.solver.NMDiagram.generate_mx_my`) for each unique N
   level encountered in demands, combinations, and envelopes.  This can
   be expensive for many distinct N values.  Contours are cached so that
   the same N level is never computed twice.

   For sections where the 3-D domain is 2-D (no My data),
   ``eta_2D`` and ``eta_path_2D`` are automatically disabled with an
   informative message.


Mandatory vs. optional generation (``gensec run``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When running ``gensec run``, some computations are **always**
performed because the verification engine depends on them:

+----------------------------------+----------+------------------------------------------+
| Computation                      | Skip?    | Why                                      |
+==================================+==========+==========================================+
| N-Mx diagram                     | **No**   | Domain boundary, always needed           |
+----------------------------------+----------+------------------------------------------+
| N-My diagram (biaxial only)      | **No**   | Biaxial domain boundary                  |
+----------------------------------+----------+------------------------------------------+
| 3-D surface (biaxial only)       | **No**   | ConvexHull for VerificationEngine        |
+----------------------------------+----------+------------------------------------------+
| Moment-curvature (M-χ)           | Yes      | ``generate_moment_curvature: false``     |
+----------------------------------+----------+------------------------------------------+
| Polar ductility                  | Yes      | ``generate_polar_ductility: false``      |
+----------------------------------+----------+------------------------------------------+
| 3-D M-χ-N surface                | Yes      | ``generate_3d_moment_curvature: false``  |
+----------------------------------+----------+------------------------------------------+
| Mx-My contour diagrams           | Yes      | ``generate_mx_my: false`` (default off)  |
+----------------------------------+----------+------------------------------------------+
| 3-D surface CSV/JSON export      | Yes      | ``generate_3d_surface: false`` (dflt off)|
+----------------------------------+----------+------------------------------------------+

To skip domain generation entirely, use ``gensec analyze`` instead.


Minimal fast verification
~~~~~~~~~~~~~~~~~~~~~~~~~~

For the fastest possible ``gensec run`` (domain + verification,
no moment-curvature diagrams):

.. code-block:: yaml

   output:
     # Only the two default η metrics — no optional ones.
     eta_norm: true
     eta_norm_beta: true

     # Disable all expensive optional generators.
     generate_moment_curvature: false
     generate_polar_ductility: false
     generate_3d_moment_curvature: false
     generate_mx_my: false

This produces:

- N-Mx and N-My interaction diagrams (CSV + JSON + plot)
- 3-D resistance surface (in memory, for verification)
- Verification results with ``eta_norm`` and ``eta_norm_beta``
- Per-fiber stress states at each demand

Estimated speedup vs. full defaults: **3–10×** depending on
section complexity and number of N levels.


Force decomposition only (``gensec analyze``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the lightest possible analysis — no domain, no diagrams, just
per-material force breakdown:

.. code-block:: bash

   uv run gensec analyze input.yaml --output-dir results

The ``output`` block in the YAML file is **ignored**.  Output:

- ``analysis_results.json`` — per-demand and per-combination force
  decomposition (bulk zones, rebar groups, per-bar detail)

Add ``--eta`` to also compute on-demand :math:`\eta` via
ray–bisection (~15–25 equilibrium solves per demand):

.. code-block:: bash

   uv run gensec analyze input.yaml --output-dir results --eta

See :ref:`architecture_solver` for a comparison of the two
pipelines.
