"""AUD-F4: which GenSec sweep planes form the hull of the prestressed F1 section
near N = -1390 kN, M > 0, and what are the element strains/stresses there."""
import sys, numpy as np
from scipy.spatial import ConvexHull
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram

d = load_yaml(sys.argv[1]); sec = d["section"]; sv = FiberSolver(sec); nm = NMDiagram(sv)
print(f"global strain envelope used by the sweep: emg={nm._emg:.5f} exg={nm._exg:.5f} emb={nm._emb:.5f}")
e0, chi = nm._ultimate_strain_configs_1d("x", 400)
N, M, _ = sv.integrate_batch(e0, chi, np.zeros_like(chi)); N /= 1e3; M /= 1e6
P = np.column_stack([N, M]); H = ConvexHull(P)
ps = sec.tendons[0].material; ei = float(sv._eps_init_tendon[0]); lt = float(sv._ly_tendon[0])
ytop = sec.bbox[3] - sv.y_ref; ybot = sec.bbox[1] - sv.y_ref
rows = []
for v in H.vertices:
    if M[v] > 0 and -2600 < N[v] < -400:
        et = e0[v] + chi[v] * lt + ei
        rows.append((N[v], M[v], e0[v] + chi[v] * ybot, e0[v] + chi[v] * ytop, et, ps.stress(et)))
rows.sort()
print(" N[kN]     M[kNm]   eps_bottom  eps_top   eps_tendon  sigma_tendon[MPa]")
for r in rows:
    print(f"{r[0]:8.1f} {r[1]:9.1f} {r[2]*1e3:9.3f}‰ {r[3]*1e3:8.3f}‰ {r[4]*1e3:9.3f}‰ {r[5]:10.1f}")
