import zipfile, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
z = zipfile.ZipFile("AUDIT_ENG_01_bundle.zip", "w", zipfile.ZIP_DEFLATED)
n = 0
for d, _, fs in os.walk("audit"):
    for f in fs:
        z.write(os.path.join(d, f)); n += 1
for f in os.listdir("base"):
    z.write(os.path.join("base", f), os.path.join("audit", "baseline", f)); n += 1
z.close(); print(n, "entries")
