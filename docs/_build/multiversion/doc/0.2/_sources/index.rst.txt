.. GenSec documentation master file.

=====================================
GenSec — Generic Section Calculator
=====================================

**Version**: |version|

GenSec is a modular Python package for **fiber-based cross-section analysis**
of composite structural members under combined axial force and biaxial bending.

It computes the resistance domain :math:`(N, M_x, M_y)` of any cross-section
composed of arbitrary materials — concrete, reinforcing steel, structural
steel, CFRP, timber, or any user-defined :math:`\sigma\text{-}\varepsilon`
curve — and verifies load demands against it.

.. important::

   GenSec is under active development and **has not yet been fully validated**
   for production use.  Always cross-check results against independent
   calculations.


Key capabilities
----------------

- **Material-agnostic constitutive laws**: parabola-rectangle concrete
  (EC2 / NTC 2018), elastic-plastic steel with optional hardening,
  EN 10025-2 structural steel, and arbitrary tabulated
  :math:`\sigma\text{-}\varepsilon` curves.
- **Biaxial bending**: full :math:`(N,\,M_x,\,M_y)` interaction surface
  with convex-hull demand verification.
- **Uniaxial bending**: classic :math:`N\text{-}M` interaction diagram as a
  special case.
- **Utilization ratios**: :math:`\eta = d/r` for each demand point, where
  :math:`d` is the distance from the domain centroid and :math:`r` the
  distance to the boundary.
- **Per-fiber post-processing**: strain and stress at every fiber and rebar
  for any load state.
- **YAML-driven input**: section geometry, materials, and demands in a
  single file.
- **CLI and Python API**: use from the command line or import as a library.
- **EC2 Table 3.1 integration**: automatic computation of all concrete
  parameters from :math:`f_{ck}`, including high-strength classes
  (:math:`f_{ck} > 50\;\text{MPa}`).


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/yaml_reference
   user_guide/sign_conventions
   user_guide/examples


.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/architecture_overview
   architecture/architecture_io
   architecture/architecture_solver


.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/fiber_method
   theory/constitutive_laws
   theory/nm_diagram
   theory/demand_verification


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/gensec.materials
   api/gensec.geometry
   api/gensec.solver
   api/gensec.output
   api/gensec.io_yaml
   api/gensec.cli


.. toctree::
   :maxdepth: 1
   :caption: Development

   developer
   ci_and_versioning
   changelog


.. toctree::
   :hidden:
   :caption: Help

   about


Indices and tables
------------------

| :ref:`genindex`
| :ref:`modindex`
| :ref:`search`
