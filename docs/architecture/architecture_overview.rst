.. _architecture_overview:

=============================
Architecture overview
=============================

This document describes the internal architecture of GenSec at a
level of detail that allows a civil engineer to understand what each
software component does and how data flows through the program.

No prior software engineering knowledge is required beyond a basic
familiarity with the concept of *functions* and *objects* (containers
that group data and operations together).



What GenSec does
-----------------

GenSec receives a description of a cross-section (geometry, materials,
reinforcement) and a set of load demands, and answers two questions:

1. **What is the section's resistance domain?**  The set of all
   :math:`(N, M_x, M_y)` combinations the section can carry at the
   ultimate limit state.

2. **Are the demands inside that domain?**  For each demand point,
   GenSec computes one or more utilization ratios :math:`\eta` that
   measure how close the demand is to the boundary.

Everything else — moment-curvature diagrams, stress maps, CSV/JSON
exports, plots — is built on top of these two capabilities.


Package structure
------------------

GenSec is organized into four *subpackages* (folders), each with a
well-defined responsibility:

.. mermaid::

   graph TB
       subgraph materials["materials/"]
           base["base.py<br/>Abstract Material"]
           concrete["concrete.py<br/>Parabola-rectangle"]
           steel["steel.py<br/>Elastic-plastic"]
           tabulated["tabulated.py<br/>Arbitrary σ-ε"]
           ec2props["ec2_properties.py<br/>EC2 Table 3.1"]
           ec2bridge["ec2_bridge.py<br/>EC2 → GenSec factory"]
       end

       subgraph geometry["geometry/"]
           fiber["fiber.py<br/>RebarLayer"]
           primitives["primitives.py<br/>Shape factories"]
           geom["geometry.py<br/>GenericSection + mesher"]
           section["section.py<br/>RectSection factory"]
       end

       subgraph solver["solver/"]
           integrator["integrator.py<br/>FiberSolver"]
           capacity["capacity.py<br/>NMDiagram"]
           check["check.py<br/>VerificationEngine"]
       end

       subgraph output["output/"]
           plots["plots.py<br/>matplotlib"]
           export["export.py<br/>CSV / JSON"]
           report["report.py<br/>Terminal"]
       end

       io_yaml["io_yaml.py<br/>YAML loader"]
       cli["cli.py<br/>CLI: run / plot"]

       cli --> io_yaml
       io_yaml --> materials
       io_yaml --> geometry
       cli --> solver
       cli --> output
       solver --> materials
       solver --> geometry

Think of these subpackages as departments in a design office:

- **materials/** — the *material laboratory*: knows how to convert a
  strain value into a stress value for any material (concrete, steel,
  CFRP, etc.).
- **geometry/** — the *drafter's desk*: defines section shapes, places
  rebars, and discretizes the section into fibers.
- **solver/** — the *calculation team*: takes a section and its
  materials, computes internal forces, builds resistance domains,
  verifies demands.
- **output/** — the *reporting team*: produces plots, data files, and
  terminal summaries.
- **io_yaml.py** — the *secretary*: reads the YAML input file and hands
  the right objects to each department.
- **cli.py** — the *project manager*: coordinates the entire workflow.


Main workflow
--------------

When you run ``gensec run input.yaml``, the following happens:

.. mermaid::

   flowchart TD
       A["Read YAML file<br/>(io_yaml.py)"] --> B["Build Material objects<br/>(concrete, steel, ...)"]
       B --> C["Build Section object<br/>(meshed polygon + rebars)"]
       C --> D["Create FiberSolver<br/>(strain → forces integrator)"]
       D --> E["Generate N-M diagrams<br/>(NMDiagram)"]
       D --> F["Generate 3D surface<br/>(NMDiagram.generate_biaxial)"]
       D --> G["Generate M-χ curves<br/>(NMDiagram.generate_moment_curvature)"]
       E --> H["Build VerificationEngine"]
       F --> H
       H --> I["Check demands<br/>(η_3D, η_2D)"]
       H --> J["Check combinations<br/>(η_path, staged)"]
       H --> K["Check envelopes<br/>(max η among members)"]
       I --> L["Export JSON + CSV"]
       J --> L
       K --> L
       L --> M["Generate plots<br/>(plots.py)"]
       M --> N["Done: all outputs<br/>in output directory"]

       style A fill:#e1f5fe
       style N fill:#c8e6c9


Class hierarchy
----------------

The key classes and their relationships:

.. mermaid::

   classDiagram
       class Material {
           <<abstract>>
           +stress(eps) float
           +stress_array(eps) ndarray
           +tangent(eps) float
           +tangent_array(eps) ndarray
           +eps_min float
           +eps_max float
       }
       class Concrete {
           +fck: float
           +fcd: float
           +eps_c2: float
           +eps_cu2: float
           +n_parabola: float
           +fct: float
           +Ec: float
           +tension_enabled: bool
       }
       class Steel {
           +fyk: float
           +fyd: float
           +Es: float
           +eps_yd: float
       }
       class TabulatedMaterial {
           +strains: ndarray
           +stresses: ndarray
       }

       Material <|-- Concrete
       Material <|-- Steel
       Material <|-- TabulatedMaterial

       class RebarLayer {
           +x: float
           +y: float
           +As: float
           +material: Material
           +embedded: bool
       }

       class GenericSection {
           +polygon: Polygon
           +bulk_material: Material
           +rebars: list~RebarLayer~
           +x_fibers: ndarray
           +y_fibers: ndarray
           +A_fibers: ndarray
           +n_grid_x: int or None
           +n_grid_y: int or None
           +dx: float
           +dy: float
       }

       GenericSection o-- Material : bulk
       GenericSection o-- RebarLayer : rebars

       note for GenericSection "RectSection() is a factory function\nthat returns a GenericSection\nwith a rectangular polygon."

       class FiberSolver {
           +integrate(eps0, chi_x, chi_y)
           +integrate_batch(eps0[], chi_x[], chi_y[])
           +integrate_with_tangent(eps0, chi_x, chi_y)
           +solve_equilibrium(N, Mx, My)
           +get_fiber_results(...)
       }
       class NMDiagram {
           +generate()
           +generate_biaxial()
           +generate_mx_my()
           +generate_moment_curvature()
       }
       class VerificationEngine {
           +check_demand()
           +check_combination()
           +check_envelope()
       }
       class DomainChecker {
           +eta_3D()
           +eta_path()
           +is_inside()
       }
       class MxMyContour {
           +eta_2D()
           +eta_path_2D()
       }

       FiberSolver --> GenericSection : reads
       NMDiagram --> FiberSolver : uses
       VerificationEngine --> DomainChecker : 3D checks
       VerificationEngine --> MxMyContour : 2D checks
       VerificationEngine --> NMDiagram : generates contours


Units and sign conventions
---------------------------

All internal computations use **basic SI units**:

+------------------+--------+---------------------------------------------+
| Quantity         | Unit   | Sign convention                             |
+==================+========+=============================================+
| Length           | mm     |                                             |
+------------------+--------+---------------------------------------------+
| Force            | N      | Positive = tension                          |
+------------------+--------+---------------------------------------------+
| Moment           | N·mm   | Right-hand rule around each axis            |
+------------------+--------+---------------------------------------------+
| Stress           | MPa    | Positive = tension                          |
+------------------+--------+---------------------------------------------+
| Strain           | —      | Positive = tension, negative = compression  |
+------------------+--------+---------------------------------------------+
| Curvature        | 1/mm   |                                             |
+------------------+--------+---------------------------------------------+

YAML input uses **engineering units** (kN, kN·m) for convenience.
The parser converts automatically.  All output files provide values
in both unit systems.

The coordinate origin is at the **bottom-left corner** of the
section's bounding box:

- :math:`x` axis: horizontal (width direction).
- :math:`y` axis: vertical (height direction, upward).
- Reference point for moments: the **geometric centroid** of the
  gross section (computed from the polygon, not from the bounding box).
