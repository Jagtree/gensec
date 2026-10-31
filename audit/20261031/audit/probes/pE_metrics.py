"""AUD-E1: what do the eta metrics report for demands at a KNOWN fraction of
the radial (proportional-loading) capacity?

Domain: default CLI resolution generate_biaxial(36, 50).  For random
directions, demand = lam * R where R is the ray exit point from the origin,
so eta_ray = lam exactly.  eta_2D from MxMyContour at the demand's N.
Also: coincidence checks eta_2D == eta_ray at N=0, eta_path(origin->T) == eta_ray.
"""
import sys, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
from gensec.solver.check import DomainChecker, MxMyContour

d = load_yaml(sys.argv[1] + "/" + sys.argv[2]); sv = FiberSolver(d["section"]); nm = NMDiagram(sv)
dom = nm.generate_biaxial(n_angles=36, n_points_per_angle=50); D = DomainChecker(dom)
rng = np.random.default_rng(7); Nmin = float(np.min(dom["N"]))
print("eta_norm at origin:", round(D.eta_norm(0, 0, 0), 4), " eta_norm_beta at origin:", D.eta_norm_beta(0, 0, 0))
for lam in (0.1, 0.5, 0.9, 0.98):
    en, eb, e2 = [], [], []
    for k in range(60):
        th = rng.uniform(0, 2 * np.pi); nf = rng.uniform(0.05, 0.6)
        p = np.array([-nf * -Nmin * -1, np.cos(th) * 1e8, np.sin(th) * 1e8])
        p[0] = -abs(p[0])
        er = D.eta_norm_ray(*p); R = p / er; T = lam * R
        en.append(D.eta_norm(*T)); eb.append(D.eta_norm_beta(*T))
        if k < 12:
            c = nm.generate_mx_my(T[0], n_angles=72)
            e2.append(MxMyContour(c).eta_2D(T[1], T[2]))
    print(f"eta_ray={lam:4.2f}: eta_norm in [{min(en):.3f},{max(en):.3f}]  eta_norm_beta in [{min(eb):.3f},{max(eb):.3f}]  eta_2D in [{min(e2):.3f},{max(e2):.3f}]")
# coincidence at N=0
c0 = nm.generate_mx_my(0.0, n_angles=72); C0 = MxMyContour(c0)
dev = []
for th in np.linspace(0, 2 * np.pi, 13)[:-1]:
    p = np.array([0.0, np.cos(th) * 1e8, np.sin(th) * 1e8])
    a = D.eta_norm_ray(*p); b = C0.eta_2D(p[1], p[2]); c = D.eta_path_norm_ray(0, 0, 0, *p)
    dev.append((a / b - 1, c / a - 1))
dev = np.array(dev)
print(f"N=0: eta_ray/eta_2D-1 in [{dev[:,0].min():+.3%},{dev[:,0].max():+.3%}] ; eta_path(origin)/eta_ray-1 max |{np.abs(dev[:,1]).max():.1e}|")
