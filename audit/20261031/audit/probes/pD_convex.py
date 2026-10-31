"""AUD-D2: convex hull vs attainable set for a brittle-tension section.

Oracle: exhaustive enumeration of strain planes (eps_bot, eps_top) on a
dense grid inside the admissible box (bulk >= eps_cu2, rebar |eps| <=
eps_su), integrated with GenSec's own direct problem.  For each target N
the attainable max |M| is the max over planes whose N lies within the
target band.  This is an UPPER bound on what the section can carry at
that N (it ignores pivot C).  The GenSec hull value is then compared.
"""
import sys, numpy as np
from scipy.spatial import ConvexHull
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
from gensec.solver.check import DomainChecker

d = load_yaml(sys.argv[1] + "/" + sys.argv[2]); sec = d["section"]
sv = FiberSolver(sec); nm = NMDiagram(sv)
r = nm.generate(n_points=800, direction="x")
P = np.column_stack([r["N_kN"], r["M_kNm"]])
H = ConvexHull(P)
dc = DomainChecker({"N": r["N"], "M": r["M"]})

def hull_M(Nq, sgn):
    best = None
    for s in H.simplices:
        (n1, m1), (n2, m2) = P[s[0]], P[s[1]]
        if (n1 - Nq) * (n2 - Nq) <= 0 and n1 != n2:
            m = m1 + (Nq - n1) / (n2 - n1) * (m2 - m1)
            if sgn * m > 0 and (best is None or abs(m) > abs(best)): best = m
    return best

c = sec.bulk_material; s_mat = sec.rebars[0].material
H_ = sec.bbox[3] - sec.bbox[1]; lever = sv.y_ref - sec.bbox[1]
# dense grid of edge strains; tension side refined near cracking strain
neg = np.linspace(c.eps_cu2, 0, 120)
pos = np.unique(np.concatenate([np.linspace(0, 4 * c.eps_ct, 120), np.linspace(0, s_mat.eps_su * 1.2, 200)]))
grid = np.unique(np.concatenate([neg, pos]))
EB, ET = np.meshgrid(grid, grid); EB = EB.ravel(); ET = ET.ravel()
chi = (ET - EB) / H_; e0 = EB + chi * lever
N, M, _ = sv.integrate_batch(e0, chi, np.zeros_like(chi))
# admissibility on actual element strains
eb_min = np.minimum(EB, ET)
er_top = e0 + chi * (sec.y_rebars.max() - sv.y_ref); er_bot = e0 + chi * (sec.y_rebars.min() - sv.y_ref)
adm = (eb_min >= c.eps_cu2 - 1e-12) & (np.maximum(np.abs(er_top), np.abs(er_bot)) <= s_mat.eps_su + 1e-12)
N = N[adm] / 1e3; M = M[adm] / 1e6
print(f"section {sys.argv[2]}: fct={c.fct} eps_ct={c.eps_ct:.3e} As_tot={sec.A_rebars.sum():.0f}; planes enumerated={adm.sum()}")
print(f"  hull N_max={P[:,0].max():.2f} kN ; enumerated N_max={N.max():.2f} kN")
for Nq in (50.0, 100.0, 150.0, 200.0, 300.0):
    band = np.abs(N - Nq) <= 2.0
    att = M[band].max() if band.any() else float("nan")
    hm = hull_M(Nq, +1)
    eta_ray = dc.eta_norm_ray(Nq * 1e3, att * 1e6) if band.any() else float("nan")
    # a demand at 0.98 x hull capacity: what does the attainable set say?
    if hm is not None:
        dem = 0.98 * hm
        ok_hull = dc.is_inside(Nq * 1e3, dem * 1e6)
        print(f"  N={Nq:6.1f} kN: hull M_Rd={hm:8.3f}  attainable max M={att:8.3f} kNm  hull/attainable-1={hm/att-1:+.1%}  demand 0.98*hull inside hull={ok_hull}, exceeds attainable by {dem/att-1:+.1%}")
