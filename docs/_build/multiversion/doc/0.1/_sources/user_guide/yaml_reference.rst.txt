.. _yaml_reference:

====================
YAML input reference
====================

A GenSec YAML file has four top-level blocks: ``materials``, ``section``,
``demands`` (optional), and ``combinations`` (optional).  An ``output``
block controls which post-processing artefacts are generated.

.. contents:: On this page
   :local:
   :depth: 2


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
       type: concrete
       fck: 25.0             # characteristic cylinder strength [MPa]
       gamma_c: 1.5          # partial safety factor (default 1.5)
       alpha_cc: 0.85        # long-term coefficient (default 0.85)
       n_parabola: 2.0       # parabolic exponent (default 2.0)
       eps_c2: -0.002        # peak-stress strain (default -0.002)
       eps_cu2: -0.0035      # ultimate strain (default -0.0035)
       fct: 0.0              # tensile strength [MPa] (default 0 = no tension)
       Ec: 0.0               # elastic modulus [MPa] for tension (default 0)

Only ``fck`` is required; all other fields have the defaults shown above.

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
       type: concrete_ec2
       class: C30/37          # EC2 class name (C12/15 … C90/105)
       ls: F                  # limit state: 'F' fundamental, 'A' accidental
       loadtype: slow         # 'slow' or 'fast' (default 'slow')
       TypeConc: R            # 'R' rapid, 'N' normal, 'S' slow hardening
       enable_tension: false  # activate linear tension branch (default false)
       tension_fct: fctd      # which fct: 'fctd', 'fctm', or 'fctk'

All Table 3.1 parameters (:math:`f_{cm}`, :math:`f_{ctm}`, :math:`E_{cm}`,
:math:`\varepsilon_{c2}`, :math:`\varepsilon_{cu2}`, :math:`n`, etc.) are
computed automatically by the :mod:`~gensec.materials.ec2_properties` module.

When ``enable_tension: true``, the elastic modulus :math:`E_{cm}` and the
chosen tensile strength (:math:`f_{ctd}`, :math:`f_{ctm}`, or
:math:`f_{ctk,0.05}`) are extracted from the EC2 property object and passed
to the :class:`~gensec.materials.Concrete` constructor.  The ``tension_fct``
field controls which value is used:

+---------------+-----------------------------------------------------------+
| ``tension_fct``| Tensile strength used                                    |
+===============+===========================================================+
| ``fctd``      | :math:`f_{ctd,0.05} = \alpha_{ct}\,f_{ctk,0.05}/\gamma_c`|
+---------------+-----------------------------------------------------------+
| ``fctm``      | :math:`f_{ctm}` (mean tensile strength)                   |
+---------------+-----------------------------------------------------------+
| ``fctk``      | :math:`f_{ctk,0.05}` (characteristic, 5 % fractile)      |
+---------------+-----------------------------------------------------------+

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
     n_fibers_y: 100          # fiber count along height
     n_fibers_x: 1            # 1 = uniaxial, >1 = biaxial grid
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
| ``circle``      | ``D`` (diameter), ``resolution`` (polygon vertices)  |
+-----------------+------------------------------------------------------+
| ``annulus``     | ``D_ext``, ``D_int``, ``resolution``                 |
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
| ``As``              | —       | Cross-sectional area [mm²] (required)    |
+---------------------+---------+------------------------------------------+
| ``material``        | —       | Material identifier (required)           |
+---------------------+---------+------------------------------------------+
| ``x``               | centroid| Horizontal coordinate [mm]               |
+---------------------+---------+------------------------------------------+
| ``embedded``        | ``true``| Subtract displaced bulk? See             |
|                     |         | :doc:`sign_conventions`.                 |
+---------------------+---------+------------------------------------------+
| ``n_bars``          | —       | Number of physical bars (reporting only) |
+---------------------+---------+------------------------------------------+
| ``diameter``        | —       | Bar diameter [mm] (reporting only)       |
+---------------------+---------+------------------------------------------+


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

Controls optional post-processing.

.. code-block:: yaml

   output:
     generate_mx_my: true         # produce Mx-My contour diagram
     generate_3d_surface: true    # produce 3-D resistance surface
     n_angles_mx_my: 144          # angular resolution for Mx-My contour

All fields are optional and default to ``false`` / sensible values.
