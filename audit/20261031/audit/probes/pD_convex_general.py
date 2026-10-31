"""AUD-D2bis: convexity check of the N-Mx domain for arbitrary sections
(bulk zones, rebars, bonded tendons with prestrain).

Oracle: enumeration of strain planes (eps at bbox bottom, eps at bbox top)
on an n x n grid spanning GenSec's own global strain envelope; a plane is
ADMISSIBLE when every element group respects its own law limits
(bulk: compression limit only, at eps_sec + bulk offset; rebar: both;
tendon: both, at eps_sec + eps_init).  Admissibility is evaluated exactly
from the extreme lever arms of each group (the strain is linear).

Two measures per N slice (band = 0.5 % of the N range):
  conv  = hull(admissible cloud) / max attained by the admissible cloud - 1
          -> pure non-convexity of the true domain (grid noise ~ control)
  gen   = GenSec hull (NMDiagram.generate) / max attained - 1
          -> what GenSec reports in excess (includes AUD-D-1)
Reported: maximum over slices with |M_att| > 2 % of max|M|.
"""
import sys, numpy as np
from scipy.spatial import ConvexHull
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram

NG = int(sys.argv[2]) if len(sys.argv) > 2 else 360


def hull_M_at(P, H, Nq):
    hi, lo = None, None
    for s in H.simplices:
        (n1, m1), (n2, m2) = P[s[0]], P[s[1]]
        if (n1 - Nq) * (n2 - Nq) <= 0 and n1 != n2:
            m = m1 + (Nq - n1) / (n2 - n1) * (m2 - m1)
            hi = m if hi is None else max(hi, m)
            lo = m if lo is None else min(lo, m)
    return hi, lo


def run(path):
    d = load_yaml(path); sec = d["section"]
    sv = FiberSolver(sec); nm = NMDiagram(sv)
    ymin, ymax = sec.bbox[1], sec.bbox[3]; H_ = ymax - ymin; lev = sv.y_ref - ymin
    lo, hi = nm._emg, nm._exg
    g = np.unique(np.concatenate([np.linspace(lo, hi, NG), np.linspace(nm._emb, 0, NG // 3),
                                  np.linspace(0, min(hi, 0.003), NG // 3)]))
    EB, ET = np.meshgrid(g, g); EB = EB.ravel(); ET = ET.ravel()
    chi = (ET - EB) / H_; e0 = EB + chi * lev
    adm = np.ones(e0.size, bool)
    off_all = sv.bulk_strain_offset()
    for kind, idx, emin, emax in sv.element_limit_groups():
        if len(idx) == 0: continue
        if kind == "bulk":
            ly = sv._ly_bulk[idx]
            off = np.asarray(off_all)[idx] if np.ndim(off_all) else float(off_all)
            offmin = float(np.min(off)) if np.ndim(off) else off
            emn = e0 + np.minimum(chi * ly.min(), chi * ly.max()) + offmin
            adm &= emn >= emin - 1e-12
        elif kind == "rebar":
            ly = sv._ly_rebar[idx]
            a = e0 + chi * ly.min(); b = e0 + chi * ly.max()
            adm &= (np.minimum(a, b) >= emin - 1e-12) & (np.maximum(a, b) <= emax + 1e-12)
        else:
            ly = sv._ly_tendon[idx]; ei = sv._eps_init_tendon[idx]
            s = e0[:, None] + chi[:, None] * ly[None, :] + ei[None, :]
            adm &= (s.min(axis=1) >= emin - 1e-12) & (s.max(axis=1) <= emax + 1e-12)
    N = np.empty(e0.size); M = np.empty(e0.size)
    for k in range(0, e0.size, 4000):
        sl = slice(k, k + 4000)
        n_, m_, _ = sv.integrate_batch(e0[sl], chi[sl], np.zeros_like(chi[sl]))
        N[sl], M[sl] = n_, m_
    N, M = N[adm] / 1e3, M[adm] / 1e6
    P = np.column_stack([N, M]); Hc = ConvexHull(P)
    r = nm.generate(n_points=400, direction="x")
    G = np.column_stack([r["N_kN"], r["M_kNm"]]); Hg = ConvexHull(G)
    Nr = N.max() - N.min(); band = 0.005 * Nr; Mref = np.abs(M).max()
    wc = (0.0, None, 0.0); wg = (0.0, None, 0.0)
    for Nq in np.linspace(N.min() + 2 * band, N.max() - 2 * band, 300):
        sel = np.abs(N - Nq) <= band
        if sel.sum() < 5: continue
        for sgn in (+1, -1):
            att = (M[sel] * sgn).max()
            hc = hull_M_at(P, Hc, Nq)[0 if sgn > 0 else 1]
            hg = hull_M_at(G, Hg, Nq)[0 if sgn > 0 else 1]
            if hc is not None:
                ex = (sgn * hc - att) / Mref
                if ex > wc[0]: wc = (ex, (round(float(Nq), 1), sgn), (sgn * hc - att))
            if hg is not None:
                ex = (sgn * hg - att) / Mref
                if ex > wg[0]: wg = (ex, (round(float(Nq), 1), sgn), (sgn * hg - att))
    name = path.replace("\\", "/").split("/")[-1]
    print(f"{name:40s} NG={NG} planes={e0.size:7d} adm={adm.sum():7d} tendons={len(sv._eps_init_tendon)} Mref={Mref:.1f} kNm | "
          f"TRUE-domain concavity: {wc[0]:+.2%} of Mref ({wc[2]:.1f} kNm) at (N,side)={wc[1]} | "
          f"GenSec hull excess: {wg[0]:+.2%} of Mref ({wg[2]:.1f} kNm) at {wg[1]}", flush=True)


for p in sys.argv[1].split(","):
    try:
        run(p)
    except Exception as e:
        print(f"{p}: ERROR {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
