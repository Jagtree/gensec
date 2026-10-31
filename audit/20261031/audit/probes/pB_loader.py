import sys
from gensec.io_yaml import load_yaml
for f in sys.argv[2:]:
    try:
        d=load_yaml(sys.argv[1]+"/"+f)
        for k,m in d["materials"].items():
            attrs={a:getattr(m,a) for a in ("fck","gamma_c","fcd","n_parabola","eps_c2","eps_cu2","fct","eps_ct","fyk","gamma_s","Es","k_hardening","eps_su","fyd","ftd","eps_yd") if hasattr(m,a)}
            print(f, k, type(m).__name__, attrs)
    except Exception as e:
        print(f, "REFUSED:", type(e).__name__, str(e).splitlines()[0])
