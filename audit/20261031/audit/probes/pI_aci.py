import sys
from gensec.io_yaml import load_yaml
d = load_yaml(sys.argv[1])
for x in d["demands"]: print("demand", x["name"], "N=", x["N"], "Mx=", x["Mx"], "My=", x["My"])
for k, m in d["materials"].items():
    print("material", k, type(m).__name__, "code=", getattr(m, "code", None).key,
          {a: getattr(m, a) for a in ("fcd", "gamma_c", "long_term_reduction_factor", "n_parabola", "eps_c2", "eps_cu2", "gamma_s", "eps_su") if hasattr(m, a)})
