import sys, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
for f in ("C1_example_input_lumped.yaml", "C1_example_input_distributed.yaml"):
    d = load_yaml(sys.argv[1] + "/" + f); sec = d["section"]
    nm = NMDiagram(FiberSolver(sec))
    print(f, "rebar x:", sorted(set(np.round(sec.x_rebars, 1))),
          "n bulk fibres", len(sec.x_fibers))
    for Nq in (-1500e3, -1200e3, -900e3):
        c = nm.generate_mx_my(Nq, n_angles=144)
        My = np.asarray(c["My"]) / 1e6; Mx = np.asarray(c["Mx"]) / 1e6
        print(f"  N={Nq/1e3:.0f} kN: My_max {My.max():.1f}  Mx_max {Mx.max():.1f} kNm")
