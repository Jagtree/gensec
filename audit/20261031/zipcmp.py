import zipfile, subprocess, hashlib, sys
z = zipfile.ZipFile("gensec_audit_20261028.zip")
same=diff=untracked=0; dl=[]; ul=[]
tracked=set(subprocess.run(["git","ls-files"],capture_output=True,text=True).stdout.split("\n"))
for n in z.namelist():
    if n.endswith("/"): continue
    zb=z.read(n)
    if n not in tracked: untracked+=1; ul.append(n); continue
    hb=subprocess.run(["git","show","HEAD:"+n],capture_output=True).stdout
    if hb==zb or hb.replace(b"\r\n",b"\n")==zb.replace(b"\r\n",b"\n"): same+=1
    else: diff+=1; dl.append(n)
print("same",same,"diff",diff,"not-tracked",untracked)
print("DIFF:",dl[:20]); print("UNTRACKED:",ul[:20])
tz=[f for f in tracked if f.startswith(("src/","tests/")) and f not in set(z.namelist())]
print("tracked src/tests missing from zip:",len(tz),tz[:10])
