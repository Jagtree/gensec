import sys, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
C=sys.argv[1]
for f in ["A1_sign_right_rebar.yaml","A2_sign_top_rebar.yaml"]:
    d=load_yaml(C+"/"+f); sec=d["section"]
    print("==",f, "demands parsed:", [(x["name"],x["N"]/1e3,x["Mx"]/1e6,x["My"]/1e6) for x in d["demands"]])
    sv=FiberSolver(sec); nm=NMDiagram(sv)
    for dirn in "xy":
        r=nm.generate(n_points=400,direction=dirn)
        N=r["N_kN"]; M=r["M_kNm"]
        # moment capacity near N=0
        m=np.abs(N)<20
        print(f" dir {dirn}: M range near N~0 (|N|<20kN): min {M[m].min():.2f}  max {M[m].max():.2f} kNm; Nmin {N.min():.1f} Nmax {N.max():.1f}")
    # direct integration: compress right edge with chi_y>0
    for chiy in (+1e-5,-1e-5):
        eb,er=sv.strain_field(0.0,0.0,chiy)
        i=np.argmax(sec.x_fibers)
        print(f"  chi_y={chiy:+g}: strain at max-x fibre {eb[i]:+.2e}")
