.. _examples:

========
Examples
========

This section presents worked examples covering common use cases.  Each
example can be run directly from the command line or reproduced via the
Python API.

The YAML input files referenced below are distributed in the
``examples/`` directory of the repository.


Rectangular RC column — uniaxial bending
-----------------------------------------

**File**: ``examples/example_input.yaml``

A 300×600 mm reinforced concrete column with three layers of
reinforcement (bottom, mid-height, top), verified under gravity and
seismic demands in uniaxial bending.

.. code-block:: bash

   uv run gensec examples/example_input.yaml --output-dir results/rect_column

Key features exercised:

- Direct concrete parameters (``type: concrete``).
- Uniaxial :math:`N\text{-}M` diagram.
- Individual demands and combinations with envelope verification.


T-section beam
---------------

**File**: ``examples/example_tee.yaml``

A reinforced concrete T-beam (flange 800×150 mm, web 300×450 mm) using
EC2 class-based concrete (``type: concrete_ec2, class: C25/30``).

.. code-block:: bash

   uv run gensec examples/example_tee.yaml --output-dir results/tee_beam

Key features exercised:

- Parametric shape (``shape: tee``).
- EC2 automatic property computation from class name.
- Grid meshing of a non-rectangular section.
- :math:`M_x\text{-}M_y` interaction contour and 3-D surface.


Annular (hollow circular) pile
-------------------------------

**File**: ``examples/example_annulus.yaml``

A 1200/800 mm annular pile section with 12 equally-spaced Φ25 bars.

.. code-block:: bash

   uv run gensec examples/example_annulus.yaml --output-dir results/annulus

Key features exercised:

- Parametric shape (``shape: annulus``).
- Circular rebar layout at explicit coordinates.
- Steel with hardening (``k_hardening: 1.05``).
- Biaxial demand verification.


Custom polygon with hole (box section)
----------------------------------------

**File**: ``examples/example_custom.yaml``

A 400×700 mm box section defined by explicit vertex coordinates with a
rectangular hole.

.. code-block:: bash

   uv run gensec examples/example_custom.yaml --output-dir results/custom_box

Key features exercised:

- ``shape: custom`` with ``exterior`` and ``holes`` vertex lists.
- Grid meshing on a section with an interior void.


Biaxial column with 3-D surface
---------------------------------

**File**: ``examples/example_biaxial_column.yaml``

Full biaxial analysis of a rectangular column including the 3-D
:math:`(N, M_x, M_y)` resistance surface, the :math:`M_x\text{-}M_y`
contour at constant :math:`N`, and convex-hull demand verification.

.. code-block:: bash

   uv run gensec examples/example_biaxial_column.yaml --output-dir results/biaxial

Key features exercised:

- 2-D fiber grid (``n_fibers_x > 1``).
- ``generate_3d_surface: true`` and ``generate_mx_my: true``.
- Multiple demands with :math:`M_y \neq 0`.
