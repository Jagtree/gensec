"""AUD-D3/D4: chord error of the default 3-D domain, and symmetry.

Default CLI domain: generate_biaxial(n_angles=36, n_points_per_angle=50).
Dense reference : generate_biaxial(n_angles=144, n_points_per_angle=400).
For demand points ON the dense boundary (eta_ray(dense)=1 by construction)
in several directions, eta_ray(default) > 1 means the default domain is
smaller (conservative); < 1 would be non-conservative.
Symmetry: doubly symmetric sections -> max|Mx|, max|My| per sign.
"""
import sys, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
from gensec.solver.check import DomainChecker

for f in sys.argv[2:]:
    d = load_yaml(sys.argv[1] + "/" + f); sv = FiberSolver(d["section"]); nm = NMDiagram(sv)
    dflt = nm.generate_biaxial(n_angles=36, n_points_per_angle=50)
    dens = nm.generate_biaxial(n_angles=144, n_points_per_angle=400)
    Dd, Dr = DomainChecker(dflt), DomainChecker(dens)
    print(f"== {f}: default pts={len(dflt['N'])} dense pts={len(dens['N'])}")
    for nm_, D in (("default", dflt), ("dense", dens)):
        Mx = np.asarray(D["Mx"]) / 1e6; My = np.asarray(D["My"]) / 1e6; N = np.asarray(D["N"]) / 1e3
        print(f"   {nm_:7s}: Mx [{Mx.min():9.3f}, {Mx.max():9.3f}]  My [{My.min():9.3f}, {My.max():9.3f}]  N [{N.min():9.2f}, {N.max():8.2f}]  asym Mx {Mx.max()/-Mx.min()-1:+.2e} My {My.max()/-My.min()-1:+.2e}")
    worst = (0, None); best = (9, None)
    rng = np.random.default_rng(1)
    Nmin = float(np.min(dens["N"]))
    for k in range(400):
        # direction in (N,Mx,My) with N in compression half mostly
        th = rng.uniform(0, 2 * np.pi); nfrac = rng.uniform(-0.9, 0.1)
        p = np.array([nfrac * -Nmin * -1.0, np.cos(th) * 1e8, np.sin(th) * 1e8])
        e = Dr.eta_norm_ray(*p)
        if not np.isfinite(e) or e <= 0: continue
        pb = p / e   # on dense boundary
        ed = Dd.eta_norm_ray(*pb)
        if ed > worst[0]: worst = (ed, pb)
        if ed < best[0]: best = (ed, pb)
    print(f"   default eta_ray on dense-boundary points: max {worst[0]:.4f} (chord loss, conservative), min {best[0]:.4f} (<1 => default domain LARGER = non-conservative)")
