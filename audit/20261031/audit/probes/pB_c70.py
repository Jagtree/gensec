import sys, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
res={}
for f in ("B3_steel_nodecl.yaml","B5_c70_explicit.yaml"):
    d=load_yaml(sys.argv[1]+"/"+f); nm=NMDiagram(FiberSolver(d["section"]))
    r=nm.generate(n_points=600,direction="x"); N=r["N_kN"]; M=r["M_kNm"]
    out={"Nmin":N.min(),"Mmax":M.max()}
    for Nq in (-1000,-2000,-3000,-4000):
        # max M on the positive branch at N=Nq by interpolation of upper envelope
        o=np.argsort(N); 
        sel=M>0; Ns=N[sel]; Ms=M[sel]; o=np.argsort(Ns)
        # upper envelope via binning
        bins=np.linspace(Ns.min(),Ns.max(),400); idx=np.digitize(Ns,bins)
        env=np.array([[Ns[idx==k].mean(),Ms[idx==k].max()] for k in np.unique(idx)])
        out[f"M@{Nq}"]=float(np.interp(Nq,env[:,0],env[:,1]))
    res[f]=out; print(f,{k:round(float(v),2) for k,v in out.items()})
a,b=res["B3_steel_nodecl.yaml"],res["B5_c70_explicit.yaml"]
for k in a: print(k, f"defaults/explicit-1 = {a[k]/b[k]-1:+.3%}")
