import zipfile,os
z=zipfile.ZipFile("gensec_audit_20261028.zip"); s=d=m=0; dl=[]
for n in z.namelist():
    if n.endswith("/"): continue
    if not os.path.exists(n): m+=1; dl.append("MISSING "+n); continue
    a=open(n,"rb").read(); b=z.read(n)
    if a.replace(b"\r\n",b"\n")==b.replace(b"\r\n",b"\n"): s+=1
    else: d+=1; dl.append("DIFF "+n)
print("worktree same",s,"diff",d,"missing",m); print(dl)
