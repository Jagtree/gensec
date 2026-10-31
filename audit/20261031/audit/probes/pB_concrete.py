import math
from gensec.materials.ec2_bridge import concrete_from_ec2, service_concrete_from_ec2
from gensec.materials.concrete import Concrete
def ref(fck):
    fcm=fck+8
    if fck<=50: n,ec2,ecu2,fctm=2.0,2.0,3.5,0.3*fck**(2/3)
    else:
        n=1.4+23.4*((90-fck)/100)**4; ec2=2.0+0.085*(fck-50)**0.53; ecu2=2.6+35*((90-fck)/100)**4; fctm=2.12*math.log(1+fcm/10)
    Ecm=22000*(fcm/10)**0.3; ec1=min(0.7*fcm**0.31,2.8)
    ecu1=3.5 if fck<50 else 2.8+27*((98-fcm)/100)**4
    return dict(n=n,eps_c2=ec2,eps_cu2=ecu2,fctm=fctm,Ecm=Ecm,eps_c1=ec1,eps_cu1=ecu1)
print("fck | quantity | EN1992 Tab3.1 | GenSec | rel")
for fck in (30,45,50,55,70,90):
    c=concrete_from_ec2(fck,ls="F",loadtype="slow",TypeConc="N",NA="French",time=28)
    r=ref(fck); e=c.ec2
    g=dict(n=c.n_parabola,eps_c2=-c.eps_c2*1e3,eps_cu2=-c.eps_cu2*1e3,fctm=e.fctm,Ecm=e.ecm,eps_c1=e.eps_c1*1e3,eps_cu1=e.eps_cu1*1e3)
    for k in r: print(f"{fck} | {k} | {r[k]:.4f} | {g[k]:.4f} | {(g[k]-r[k])/r[k]:+.2e}")
    print(f"{fck} | fcd(t=28) | {0.85*fck/1.5:.4f} (alpha_cc=0.85 rec.) | {c.fcd:.4f} | alpha_cc={c.long_term_reduction_factor}")
# age discontinuity
for t in (27.0,28.0,28.001,29.0,365.0):
    c=concrete_from_ec2(30,ls="F",loadtype="slow",TypeConc="N",NA="French",time=t)
    print(f"C30 time={t}: fck_used={c.fck:.3f} alpha_cc={c.long_term_reduction_factor} fcd={c.fcd:.3f}")
# ls handling
for ls in ("F","A","QP","C","S","xyz"):
    c=concrete_from_ec2(30,ls=ls,loadtype="slow",TypeConc="N",NA="French",time=28)
    print(f"ls={ls!r}: gamma_c={c.gamma_c}")
# custom container silent defaults at fck=70
c=Concrete(fck=70,gamma_c=1.5,long_term_reduction_factor=0.85)
print("Concrete(fck=70) custom container: n",c.n_parabola,"eps_c2",c.eps_c2,"eps_cu2",c.eps_cu2)
# Service law points
s=service_concrete_from_ec2(30.0,fct_eff="fctm",TypeConc="N",time=28.0)
print("Service C30: fcm",s.fcm,"Ecm",s.Ecm,"eps_c1",s.eps_c1,"eps_cu1",s.eps_cu1,"k",s.k)
print(" sigma(-eps_c1)=",s.stress(-s.eps_c1)," tangent(0)/Ecm=",s.tangent(0.0)/s.Ecm, " sigma(eps_ct)=",s.stress(s.eps_ct), "fct_eff",s.fct_eff, "sigma(-1.0001 eps_cu1)",s.stress(-1.0001*s.eps_cu1))
import inspect
print("service_concrete_from_ec2 signature:",inspect.signature(service_concrete_from_ec2))
