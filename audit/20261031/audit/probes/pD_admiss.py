"""AUD-D1: are inadmissible strain planes on the N-M hull?

For every (eps0, chi) the N-Mx sweep of NMDiagram.generate() integrates,
compute whether any element exceeds ITS OWN limit (strict, 1e-9 margin) or
whether uniform compression exceeds eps_c2 (EN 1992-1-1 6.1(5), pivot C).
Compare the convex hull of all points with the hull of admissible points.
"""
import sys, numpy as np
from scipy.spatial import ConvexHull
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram

for f in sys.argv[2:]:
    d = load_yaml(sys.argv[1] + "/" + f); sec = d["section"]
    sv = FiberSolver(sec); nm = NMDiagram(sv)
    e0, chi = nm._ultimate_strain_configs_1d("x", 400)
    N, Mx, My = sv.integrate_batch(e0, chi, np.zeros_like(chi))
    bad = np.zeros(len(e0), bool); pivc = np.zeros(len(e0), bool)
    groups = sv.element_limit_groups()
    for i in range(len(e0)):
        eb, er = sv.strain_field(e0[i], chi[i], 0.0)
        arrs = {"bulk": eb, "rebar": er}
        for kind, idx, lo, hi in groups:
            if kind not in arrs or len(idx) == 0: continue
            s = arrs[kind][idx]
            up = (kind != "bulk") and s.max() > hi * (1 + 1e-9)
            if s.min() < lo * (1 + 1e-9) or up:
                bad[i] = True
        # pivot C (EC2 6.1(5)): for fully compressed sections the strain at
        # 3/7 h from the compressed face must not exceed eps_c2
        c = sec.bulk_material
        if eb.max() <= 0 and hasattr(c, "eps_c2"):
            ymin, ymax = sec.y_fibers.min(), sec.y_fibers.max()
            # strain at 3/7 depth from the more compressed face
            e_top = e0[i] + chi[i] * (sec.bbox[3] - sv.y_ref)
            e_bot = e0[i] + chi[i] * (sec.bbox[1] - sv.y_ref)
            if e_top <= e_bot: e37 = e_top + (e_bot - e_top) * 3 / 7
            else: e37 = e_bot + (e_top - e_bot) * 3 / 7
            if e37 < c.eps_c2 * (1 + 1e-6): pivc[i] = True
    P = np.column_stack([N / 1e3, Mx / 1e6])
    adm = ~bad & ~pivc
    Hall = ConvexHull(P); Hadm = ConvexHull(P[adm])
    verts_bad = [v for v in Hall.vertices if not adm[v]]
    print(f"== {f}: configs={len(e0)} inadmissible(own-limit)={bad.sum()} pivotC-violating={pivc.sum()} hull vertices={len(Hall.vertices)} of which inadmissible={len(verts_bad)}")
    print(f"   N_min all={P[:,0].min():.2f}  admissible={P[adm,0].min():.2f} kN  ({P[:,0].min()/P[adm,0].min()-1:+.3%})")
    print(f"   N_max all={P[:,0].max():.2f}  admissible={P[adm,0].max():.2f} kN")
    print(f"   |M|max all={np.abs(P[:,1]).max():.2f}  admissible={np.abs(P[adm,1]).max():.2f} kNm")
    print(f"   hull area all={Hall.volume:.1f} admissible={Hadm.volume:.1f}  ({Hall.volume/Hadm.volume-1:+.3%})")
    # max outward excess of the all-hull over the admissible hull along rays from the admissible centroid
    ctr = P[adm].mean(axis=0); eq = Hadm.equations
    worst = 0.0; wv = None
    for v in Hall.vertices:
        p = P[v]; dvec = p - ctr
        den = eq[:, :2] @ dvec; num = -(eq[:, :2] @ ctr + eq[:, 2])
        t = np.where(den > 1e-15, num / den, np.inf).min()
        if 1 / t > 1 + 1e-9 and (1 / t - 1) > worst: worst = 1 / t - 1; wv = p
    print(f"   worst radial excess of reported hull over admissible hull = {worst:+.3%} at (N,M)={wv}")
    for lab, mask in (("own-limit only", ~bad), ("pivot-C only", ~pivc)):
        Hm = ConvexHull(P[mask]); eqm = Hm.equations; ctr2 = P[mask].mean(axis=0); w2 = 0.0
        for v in Hall.vertices:
            dvec = P[v] - ctr2; den = eqm[:, :2] @ dvec; num = -(eqm[:, :2] @ ctr2 + eqm[:, 2])
            t = np.where(den > 1e-15, num / den, np.inf).min(); w2 = max(w2, 1 / t - 1)
        print(f"   excluding {lab:15s}: worst radial excess {w2:+.3%}; N_min filtered {P[mask,0].min():.2f}")
