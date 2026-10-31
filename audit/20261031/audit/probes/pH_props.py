"""AUD-H: section properties vs closed form (oracle written from first principles)."""
import sys, math
from gensec.io_yaml import load_yaml

def rect(b, h, xc, yc):  # area, Sx (about x=0 axis -> y moment), Sy, own Ix, Iy about own centroid
    return dict(A=b*h, x=xc, y=yc, Ix=b*h**3/12, Iy=h*b**3/12, Ixy=0.0)

def combine(parts):  # parts: (sign, dict)
    A = sum(s*p["A"] for s, p in parts)
    xg = sum(s*p["A"]*p["x"] for s, p in parts) / A
    yg = sum(s*p["A"]*p["y"] for s, p in parts) / A
    Ix = sum(s*(p["Ix"] + p["A"]*(p["y"]-yg)**2) for s, p in parts)
    Iy = sum(s*(p["Iy"] + p["A"]*(p["x"]-xg)**2) for s, p in parts)
    Ixy = sum(s*(p["Ixy"] + p["A"]*(p["x"]-xg)*(p["y"]-yg)) for s, p in parts)
    return dict(A=A, xg=xg, yg=yg, Ix=Ix, Iy=Iy, Ixy=Ixy)

C = sys.argv[1]
cases = {
  "H1_box.yaml": combine([(+1, rect(400, 700, 200, 350)), (-1, rect(240, 540, 200, 350))]),
  "H2_tee.yaml": combine([(+1, rect(300, 550, 400, 275)), (+1, rect(800, 150, 400, 625))]),
  "H4_L.yaml":   combine([(+1, rect(300, 500, 150, 250)), (-1, rect(200, 350, 200, 325))]),
}
# H3: homogenised RC rect 300x500, Ec=33000, 2 bars As=1500 at y=50 and 600 at y=450, Es=200000
n = 200000/33000
c = rect(300, 500, 150, 250)
bars = [dict(A=(n-1)*1500, x=150, y=50, Ix=0, Iy=0, Ixy=0), dict(A=(n-1)*600, x=150, y=450, Ix=0, Iy=0, Ixy=0)]
cases["H3_rc_homog.yaml"] = combine([(+1, c)] + [(+1, b) for b in bars])
for f, ref in cases.items():
    p = load_yaml(f"{C}/{f}")["section"].compute_ideal_properties()
    g = dict(A=p.area, xg=p.xg, yg=p.yg, Ix=p.Ix, Iy=p.Iy, Ixy=p.Ixy)
    worst = max(abs(g[k]-ref[k])/max(abs(ref[k]), 1e-9*max(abs(ref['Ix']),1)) for k in ref)
    print(f"== {f}: max rel dev {worst:.2e}")
    for k in ref:
        print(f"   {k:4s} closed={ref[k]:.6e}  GenSec={g[k]:.6e}")
    if f == "H4_L.yaml":
        Ix, Iy, Ixy = ref["Ix"], ref["Iy"], ref["Ixy"]
        a = 0.5*math.atan2(-2*Ixy, Ix-Iy)
        I1 = (Ix+Iy)/2 + math.hypot((Ix-Iy)/2, Ixy); I2 = (Ix+Iy)/2 - math.hypot((Ix-Iy)/2, Ixy)
        print(f"   principal closed: I_xi={I1:.6e} I_eta={I2:.6e} alpha(=0.5 atan2(-2Ixy,Ix-Iy))={math.degrees(a):.4f} deg")
        print(f"   principal GenSec: I_xi={p.I_xi:.6e} I_eta={p.I_eta:.6e} alpha={math.degrees(p.alpha):.4f} deg (n_bulk={p.n_bulk}, E_ref={p.E_ref})")
