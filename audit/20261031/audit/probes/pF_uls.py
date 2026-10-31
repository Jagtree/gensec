import sys, numpy as np
from scipy.spatial import ConvexHull
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
d = load_yaml(sys.argv[1] + "/F1_pretensioned_uls.yaml"); sec = d["section"]
ps = sec.tendons[0].material
print("tendon law: f_p01d", ps.f_p01d, "eps_ud", ps.eps_ud, "Ep", ps.Ep, "sigma_ud", ps.sigma_ud)
st = d["combinations"][0]["stages"][0]
print("stage keys:", sorted(st.keys()))
print("resolved prestress actions:", st.get("_prestress_actions"))
sv = FiberSolver(sec); nm = NMDiagram(sv)
N0, M0, _ = sv.integrate(0.0, 0.0, 0.0)
print(f"zero-strain-plane resultant (prestrain only): N={N0/1e3:.3f} kN Mx={M0/1e6:.3f} kNm  (expected +Ap*Ep*eps_pe = {1000*195000*0.006/1e3:.1f} kN at y=100)")
r = nm.generate(n_points=1500, direction="x"); P = np.column_stack([r["N_kN"], r["M_kNm"]]); H = ConvexHull(P)
best = 0
for s in H.simplices:
    (n1, m1), (n2, m2) = P[s[0]], P[s[1]]
    if n1 * n2 <= 0 and n1 != n2:
        m = m1 - n1 / (n2 - n1) * (m2 - m1)
        if m < best: best = m
print(f"GenSec M_Rd(N=0), bottom-tension side: {best:.3f} kNm")
