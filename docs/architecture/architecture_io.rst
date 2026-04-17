.. _architecture_io:

============================
I/O and CLI pipeline
============================

This document describes how data enters GenSec (YAML input), how
it is processed (CLI orchestration), and how results leave (JSON,
CSV, plots).

.. contents:: On this page
   :local:
   :depth: 2


Data flow
----------

.. mermaid::

   flowchart LR
       subgraph INPUT
           YAML["input.yaml"]
       end

       subgraph PARSE["io_yaml.py"]
           PM["Parse materials"]
           PS["Parse section"]
           PD["Parse demands"]
           PC["Parse combinations"]
           PE["Parse envelopes"]
           PO["Parse output flags"]
       end

       subgraph COMPUTE["solver/"]
           FS["FiberSolver"]
           NM["NMDiagram"]
           VE["VerificationEngine"]
       end

       subgraph EXPORT["output/"]
           JSON["JSON files"]
           CSV["CSV files"]
           PNG["PNG plots"]
       end

       YAML --> PM & PS & PD & PC & PE & PO
       PM --> FS
       PS --> FS
       FS --> NM --> VE
       PD & PC & PE --> VE
       PO --> VE
       VE --> JSON & CSV
       NM --> JSON & CSV
       JSON --> PNG

Each box is a distinct software module.  The arrows show data
dependencies: "A → B" means "B needs output from A".


YAML input structure
---------------------

A GenSec YAML file has up to six top-level blocks:

.. mermaid::

   graph TD
       YAML["input.yaml"]
       YAML --> MAT["materials:<br/>concrete, steel, ..."]
       YAML --> SEC["section:<br/>geometry + rebars"]
       YAML --> DEM["demands:<br/>individual load cases"]
       YAML --> CMB["combinations:<br/>factored sums / stages"]
       YAML --> ENV["envelopes:<br/>max η among members"]
       YAML --> OUT["output:<br/>flags and options"]

       MAT --> SEC
       DEM --> CMB
       DEM --> ENV
       CMB --> ENV

       style MAT fill:#fff3e0
       style SEC fill:#e8f5e9
       style DEM fill:#e3f2fd
       style CMB fill:#e3f2fd
       style ENV fill:#e3f2fd
       style OUT fill:#f3e5f5

The parser (``io_yaml.py``) reads each block and constructs the
corresponding Python objects:

+-------------------+--------------------------------------------------+
| YAML block        | Python object produced                           |
+===================+==================================================+
| ``materials``     | dict of ``Material`` instances                   |
+-------------------+--------------------------------------------------+
| ``section``       | ``GenericSection`` or ``RectSection``             |
+-------------------+--------------------------------------------------+
| ``demands``       | list of ``{"name", "N", "Mx", "My"}`` dicts      |
+-------------------+--------------------------------------------------+
| ``combinations``  | list of parsed combination specs                 |
+-------------------+--------------------------------------------------+
| ``envelopes``     | list of parsed envelope specs                    |
+-------------------+--------------------------------------------------+
| ``output``        | dict of flags with defaults applied              |
+-------------------+--------------------------------------------------+


Demand resolution
~~~~~~~~~~~~~~~~~~

Combinations and envelopes reference demands by name (``ref: G``).
The resolution order is:

1. Look up ``ref`` in the ``demands`` block.
2. If not found, look up in previously computed ``combination_results``
   (using the combination's final resultant).
3. If not found → error.

This means combinations can reference demands, and envelopes can
reference both demands and combinations.


CLI subcommands
----------------

The CLI (``cli.py``) uses Python's ``argparse`` with two subcommands:

.. mermaid::

   flowchart TD
       CMD["gensec"]
       RUN["gensec run input.yaml<br/>--n-points 400<br/>--output-dir results/"]
       PLOT["gensec plot data.json<br/>-o custom.png<br/>--dpi 300"]
       COMPAT["gensec input.yaml<br/>(auto-detected → run)"]

       CMD --> RUN
       CMD --> PLOT
       CMD -.-> COMPAT

``gensec run``
~~~~~~~~~~~~~~~

The full analysis pipeline:

1. Load YAML → build section, materials, demands.
2. Generate N-Mx diagram (always).
3. Generate N-My diagram (biaxial sections).
4. Generate 3D surface (biaxial sections).
5. Create VerificationEngine → check demands, combinations, envelopes.
6. Export verification JSON/CSV.
7. Per-demand fiber analysis (solve inverse → stress/strain state).
8. Export fiber CSV + section state plots.
9. Generate N-M diagram plot.
10. Generate Mx-My contour plots (if ``generate_mx_my: true``).
11. Generate M-χ diagrams (JSON + CSV + plot).
12. Generate 3D surface plot (if ``generate_3d_surface: true``).
13. Generate bundle, polar ductility, and 3D M-χ-N surface plots.
14. Print summary table.

``gensec plot``
~~~~~~~~~~~~~~~~

Regenerates a single plot from a previously exported JSON file:

1. Read JSON.
2. Detect data type from the ``"type"`` field (or infer from keys).
3. Reconstruct the data dict expected by the plotting function.
4. Call the appropriate plot function.
5. Save PNG.

Supported types: ``moment_curvature``, ``mx_my_contour``, N-M domain,
3D surface.


Data-first design
------------------

Every analysis step exports its numerical data (JSON + CSV) **before**
generating the plot:

.. mermaid::

   flowchart LR
       CALC["Compute M-χ curve"]
       JSON["Export mx_chi_N0.json"]
       CSV["Export mx_chi_N0.csv"]
       PLOT["Generate mx_chi_N0.png"]
       REPLOT["gensec plot mx_chi_N0.json<br/>(later, without recomputing)"]

       CALC --> JSON --> PLOT
       CALC --> CSV
       JSON -.-> REPLOT

This design has three benefits:

1. **No recomputation**: change plot aesthetics without re-running
   the analysis.
2. **External tools**: import JSON/CSV into Excel, MATLAB, Python
   scripts, web dashboards, etc.
3. **GUI-ready**: a future GUI reads JSON data and renders its own
   interactive plots.


Output files
-------------

A typical run with all options enabled produces:

+------------------------------------+------------------------------------------------+
| File                               | Content                                        |
+====================================+================================================+
| ``n_mx_domain.csv`` / ``.json``    | N-Mx interaction point cloud                   |
+------------------------------------+------------------------------------------------+
| ``n_my_domain.csv``                | N-My interaction point cloud                   |
+------------------------------------+------------------------------------------------+
| ``surface_3d.csv`` / ``.json``     | 3D surface (N, Mx, My) point cloud             |
+------------------------------------+------------------------------------------------+
| ``mx_chi_N{label}.json`` / ``.csv``| M-χ curve at fixed N (with key points)         |
+------------------------------------+------------------------------------------------+
| ``mx_my_N{label}.json`` / ``.csv`` | Mx-My contour at fixed N                       |
+------------------------------------+------------------------------------------------+
| ``demand_summary.csv`` / ``.json`` | Per-demand η values                            |
+------------------------------------+------------------------------------------------+
| ``combination_summary.json``       | Staged combination results                     |
+------------------------------------+------------------------------------------------+
| ``envelope_summary.json``          | Envelope results with governing member          |
+------------------------------------+------------------------------------------------+
| ``verification_summary.json``      | Unified export (all three above)               |
+------------------------------------+------------------------------------------------+
| ``fibers_{name}.csv``              | Per-fiber strain/stress for each demand         |
+------------------------------------+------------------------------------------------+

PNG plots: ``n_mx_diagram.png``, ``n_my_diagram.png``,
``surface_3d.png``, ``mx_my_N{label}.png``, ``mx_chi_N{label}.png``,
``section_{name}.png``, ``state_{name}.png``,
``demand_utilization.png``, ``section_geometry.png``,
``mx_chi_bundle.png``, ``polar_ductility_N{label}.png``, etc.


Verification JSON structure
----------------------------

The ``verification_summary.json`` file contains all three verification
types in a single file:

.. code-block:: json

   {
     "demands": [
       {
         "name": "G",
         "N_kN": -2000.0,
         "Mx_kNm": 150.0,
         "My_kNm": 0.0,
         "eta_3D": 0.42,
         "eta_2D": 0.38,
         "inside": true,
         "verified": true
       }
     ],
     "combinations": [
       {
         "name": "SLU_sismico",
         "type": "staged",
         "resultant": {"N_kN": -1850.0, "Mx_kNm": 350.0, "My_kNm": 100.0},
         "stages": [
           {
             "name": "gravitazionale",
             "cumulative": {"N_kN": -2150.0, "Mx_kNm": 174.0, "My_kNm": 9.0},
             "eta_3D": 0.35
           },
           {
             "name": "sisma",
             "cumulative": {"N_kN": -1850.0, "Mx_kNm": 350.0, "My_kNm": 100.0},
             "base": {"N_kN": -2150.0, "Mx_kNm": 174.0, "My_kNm": 9.0},
             "eta_path": 0.72,
             "eta_3D": 0.81
           }
         ],
         "eta_governing": 0.81,
         "verified": true
       }
     ],
     "envelopes": [
       {
         "name": "Envelope_SLU",
         "eta_max": 0.81,
         "governing_member": "SLU_sismico",
         "verified": true
       }
     ]
   }

The ``eta_governing`` of a combination is the maximum across all
stages and all enabled η types.  The ``eta_max`` of an envelope is
the maximum across all members.


3D surface rendering
---------------------

The 3D resistance surface plot uses a **lofted surface** technique
instead of directly rendering the ConvexHull triangulation:

.. mermaid::

   flowchart TD
       H["3D ConvexHull<br/>of (Mx, My, N) point cloud"]
       S["Slice hull at 20 regular N levels"]
       C["For each slice:<br/>2D ConvexHull → clean boundary"]
       R["Resample each contour<br/>to 72 equi-spaced arc-length points"]
       G["Build structured grid<br/>(20 rows × 72 columns)"]
       P["plot_surface with<br/>face colours from N"]
       M["Add contour rings (parallels)<br/>and meridians (longitudes)"]

       H --> S --> C --> R --> G --> P --> M

This produces a smooth surface free of the spurious triangles that
plague raw ConvexHull rendering, especially in the tension zone
where the point cloud is sparse.
