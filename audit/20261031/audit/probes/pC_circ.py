"""AUD-C circle: M_Rd at N=0 vs mesh, against an independent oracle.

Oracle: exact chord-width integration of the parabola-rectangle block over
the TRUE circle (and over the 256-gon), 1-D adaptive quadrature in y,
neutral axis by bisection, ultimate state = min(concrete eps_cu2 at the
extreme fibre, steel eps_su at the extreme bar).  Embedded bars subtract
the concrete stress at their location.
"""
import sys, math, numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.spatial import ConvexHull
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram

C = sys.argv[1]
fcd = 0.85 * 30 / 1.5; fyd = 450 / 1.15; Es = 2e5; eyd = fyd / Es
ec2, ecu, esu = 0.002, 0.0035, 0.01
R = 250.0; rb = 200.0; Ab = 314.159265
bars = [(R + rb * math.cos(k * math.pi / 4), R + rb * math.sin(k * math.pi / 4)) for k in range(8)]


def sc(e):  # concrete, compression negative
    if e >= 0: return 0.0
    if e >= -ec2: return -fcd * (1 - (1 + e / ec2) ** 2)
    if e >= -ecu: return -fcd
    return 0.0


def ss(e):
    if abs(e) <= eyd: return Es * e
    if abs(e) <= esu: return math.copysign(fyd, e)
    return 0.0


def forces(et, eb):
    """strain at y=500 (top) et, at y=0 (bottom) eb; returns N [kN], M about centroid [kNm] (GenSec sign)."""
    k = (et - eb) / (2 * R)
    eps = lambda y: eb + k * y
    w = lambda y: 2 * math.sqrt(max(R * R - (y - R) ** 2, 0.0))
    brk = sorted({min(max(y, 0), 2 * R) for y in (( -ec2 - eb) / k if k else 0, (-ecu - eb) / k if k else 0, (0 - eb) / k if k else 0)})
    pts = [p for p in brk if 0 < p < 2 * R]
    N = quad(lambda y: sc(eps(y)) * w(y), 0, 2 * R, points=pts or None, limit=400, epsabs=1e-6)[0]
    M = quad(lambda y: sc(eps(y)) * w(y) * (y - R), 0, 2 * R, points=pts or None, limit=400, epsabs=1e-6)[0]
    for (x, y) in bars:
        e = eps(y); f = (ss(e) - sc(e)) * Ab
        N += f; M += f * (y - R)
    return N / 1e3, M / 1e6


# ultimate states with top compressed (Mx<0? GenSec: Mx = sum sigma A ly -> top compression gives Mx<0)
ymax_bar = R + rb
def N_of_pivot(par):
    # par in [0,1): pivot B (top at -ecu, bottom varies) ; return N, M for given bottom strain
    pass

def ultimate(eb_free):
    # top fibre at -ecu, bottom strain eb_free; check bar strain at y=50 not beyond esu
    et = -ecu
    k = (et - eb_free) / (2 * R)
    e_bar_bot = eb_free + k * (R - rb)
    if e_bar_bot > esu:
        return None
    return forces(et, eb_free)

# Find bottom strain such that N=0 (pivot B: top at eps_cu2)
f = lambda eb: ultimate(eb)[0] if ultimate(eb) else float("nan")
eb0 = brentq(lambda eb: forces(-ecu, eb)[0], 0.0, 0.05, xtol=1e-12)
k = (-ecu - eb0) / (2 * R); ebar = eb0 + k * (R - rb)
N0, M0 = forces(-ecu, eb0)
print(f"ORACLE true circle, pivot B (top -3.5 permil): eb={eb0*1e3:.4f} permil, bottom bar strain={ebar*1e3:.4f} permil (<10 ok={ebar<esu}), N={N0:.5f} kN, |M|={abs(M0):.4f} kNm")

for hm in ("50", "25", "12.5", "6.25", "3.125"):
    dd = load_yaml(f"{C}/C2_circ_h{hm}.yaml"); sec = dd["section"]
    nm = NMDiagram(FiberSolver(sec))
    r = nm.generate(n_points=1500, direction="x")
    P = np.column_stack([r["N_kN"], r["M_kNm"]])
    H = ConvexHull(P)
    best = 0.0
    for s in H.simplices:
        (n1, m1), (n2, m2) = P[s[0]], P[s[1]]
        if (n1 - 0) * (n2 - 0) <= 0 and n1 != n2:
            m = m1 + (0 - n1) / (n2 - n1) * (m2 - m1)
            best = max(best, abs(m))
    print(f"GenSec circ h={hm:>6}: n_fibres={len(sec.A_fibers):6d}  hull |M(N=0)|={best:.4f} kNm  rel={best/abs(M0)-1:+.3e}")

# ---- corrected oracle: pivot B (top at -ecu) with bottom-bar strain bracketed in [0, esu]
def state_B(ebar):
    yb = R - rb
    k = (-ecu - ebar) / (2 * R - yb); eb = ebar - k * yb
    return forces(-ecu, eb)
print("N at A/B corner (top -3.5, bar +10 permil):", round(state_B(esu)[0], 4), "kN")
ebarB = brentq(lambda e: state_B(e)[0], 1e-6, esu, xtol=1e-14)
NB, MB = state_B(ebarB)
print(f"ORACLE pivot B: bottom-bar strain={ebarB*1e3:.5f} permil, N={NB:.6f} kN, |M_Rd(N=0)|={abs(MB):.4f} kNm")
