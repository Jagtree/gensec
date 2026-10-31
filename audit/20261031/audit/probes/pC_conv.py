"""AUD-C convergence: N_Rd,c and M_Rd(N=0) vs grid mesh size, against closed form."""
import sys, math, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram

C = sys.argv[1]
fcd = 0.85 * 30 / 1.5; fyd = 450 / 1.15
ec2, ecu = 0.002, 0.0035
alpha = 1 - ec2 / (3 * ecu)                      # stress-block area factor
# centroid depth factor of the parabola-rectangle block (n=2)
beta = (ecu**2 / 2 - ec2**2 / 12) / (ecu * (ecu - ec2 / 3))
beta = 1 - beta                                   # depth from compressed edge / x
print(f"alpha={alpha:.6f} beta={beta:.6f}")

# rectangle: closed form
b, h, As, d = 300.0, 500.0, 1500.0, 450.0
x = As * fyd / (alpha * b * fcd)
eps_s = ecu * (d - x) / x
M_ref = As * fyd * (d - beta * x) / 1e6
N_ref = -(b * h * fcd + As * (fyd - fcd)) / 1e3   # uniform -2 permil, steel yielded
print(f"RECT closed form: x={x:.3f} mm eps_s={eps_s*1e3:.3f} permil  M_Rd={M_ref:.4f} kNm  N_Rd,c={N_ref:.4f} kN")


def m_at_n0(N, M, sign):
    best = None
    for i in range(len(N) - 1):
        if (N[i] <= 0 <= N[i + 1]) or (N[i + 1] <= 0 <= N[i]):
            if N[i + 1] == N[i]:
                continue
            t = -N[i] / (N[i + 1] - N[i]); m = M[i] + t * (M[i + 1] - M[i])
            if sign * m > 0 and (best is None or abs(m) > abs(best)):
                best = m
    return best


for shape in ("rect", "circ"):
    rows = []
    for hm in ("50", "25", "12.5", "6.25", "3.125"):
        dd = load_yaml(f"{C}/C2_{shape}_h{hm}.yaml"); sec = dd["section"]
        nm = NMDiagram(FiberSolver(sec))
        r = nm.generate(n_points=1500, direction="x")
        N = r["N_kN"]; M = r["M_kNm"]
        Aflt = float(np.sum(sec.A_fibers))
        Nmin = float(N.min()); m0 = m_at_n0(N, M, -1); m0 = m0 if m0 is not None else m_at_n0(N, M, +1)
        rows.append((float(hm), len(sec.A_fibers), Aflt, Nmin, m0))
    if shape == "rect":
        for hm, nf, A, Nmin, m0 in rows:
            print(f"rect h={hm:7.3f} nf={nf:6d} A={A:.1f} N_Rd={Nmin:.4f} ({Nmin/N_ref-1:+.2e})  |M(N=0)|={abs(m0):.4f} ({abs(m0)/M_ref-1:+.2e})")
    else:
        poly_A = float(dd["section"].polygon.area)
        As_c = 8 * 314.159265
        Nc_ref_poly = -(poly_A * fcd + As_c * (fyd - fcd)) / 1e3
        Nc_ref_true = -(math.pi * 250**2 * fcd + As_c * (fyd - fcd)) / 1e3
        print(f"CIRC polygon area={poly_A:.3f} vs pi r^2={math.pi*250**2:.3f} ({poly_A/(math.pi*250**2)-1:+.2e});"
              f" N_Rd,c(poly)={Nc_ref_poly:.4f} kN, N_Rd,c(true circle)={Nc_ref_true:.4f} kN")
        ms = []
        for hm, nf, A, Nmin, m0 in rows:
            ms.append(abs(m0))
            print(f"circ h={hm:7.3f} nf={nf:6d} A={A:.1f} ({A/poly_A-1:+.2e}) N_Rd={Nmin:.4f} ({Nmin/Nc_ref_poly-1:+.2e})  |M(N=0)|={abs(m0):.4f}")
        # observed order from successive differences
        for i in range(len(ms) - 2):
            e1 = abs(ms[i] - ms[i + 1]); e2 = abs(ms[i + 1] - ms[i + 2])
            if e2 > 0:
                print(f"  observed order p (h={rows[i][0]}..{rows[i+2][0]}) = {math.log(e1/e2, 2):.2f}")
