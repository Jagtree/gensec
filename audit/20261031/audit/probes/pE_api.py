import sys
from gensec import api
p=sys.argv[1]
r=api.verify_point(0.0,0.0,150.0,yaml_path=p)
print("verify_point(0,0,150): eta_3D",r.eta_3D,"eta_2D",r.eta_2D,"status",r.status)
r=api.verify_point(-10000.0,0.0,0.0,yaml_path=p)
print("verify_point(-10000,0,0) [beyond N_Rd,min=-3469]: eta_3D",r.eta_3D,"eta_2D",r.eta_2D,"status",r.status)
a=api.analyze(yaml_path=p)
for row in a.verification:
    print("analyze row",row.kind,row.name,"eta_3D",row.eta_3D,"eta_2D",row.eta_2D,"status",row.status)
