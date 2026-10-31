"""AUD-F2: grout_selfconsistent on a hand-controllable case.

Physics (oracle): grouting is not a mechanical event.  Under the SAME
external demand the strain plane before and after grouting is the same,
and the bonded tendon reads back the stress it carried as an external
load.  Hence eps_init = eps_pe_eff - eps_sec,grout, and the datum must be
a CONTINUOUS function of the demand: demand 0 N and demand 1 N must give
the same eps_init to ~1e-9.
"""
import sys, numpy as np
from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver
from gensec.solver.posttension import (solve_posttension_sequence,
                                       grout_selfconsistent, _bonded_state)
from gensec.solver.section_state import materialize_view

d = load_yaml(sys.argv[1]); sec = d["section"]
SIG = 1400.0
res = solve_posttension_sequence(sec, sigma_p0=[SIG], order=[0],
                                 base_N=0.0, base_Mx=0.0, base_My=0.0)
print("PostTensionResult fields:", [a for a in dir(res) if not a.startswith("_")])
eps_pe = float(np.asarray(res.eps_pe_eff)[0]); Ep = sec.tendons[0].material.Ep
print(f"eps_pe_eff={eps_pe:.9e} (sigma={eps_pe*Ep:.4f} MPa)")
out = {}
for lab, dem in (("demand 0 N", (0.0, 0.0, 0.0)), ("demand 1 N (compr.)", (-1.0, 0.0, 0.0))):
    g, by_local, k, r = grout_selfconsistent(sec, res, [0], demand_after=dem)
    ei = by_local[0]
    st = _bonded_state(sec, None, [0], [ei]); view = materialize_view(sec, st)
    sv = FiberSolver(view)
    sol = sv.solve_equilibrium(*dem, tol=1e-6)
    e0, cx = float(sol["eps0"]), float(sol["chi_x"])
    eps_sec = e0 + cx * (sec.y_tendons[0] - sv.y_ref)
    sig_read = Ep * (eps_sec + ei)
    out[lab] = ei
    print(f"{lab:20s}: iters={k} eps_init={ei:.9e}  post-grout plane eps0={e0:.4e} chi={cx:.4e}  tendon read-back={sig_read:.4f} MPa (jack {eps_pe*Ep:.4f})")
a, b = out.values()
print(f"eps_init jump between demand 0 N and 1 N: {a-b:+.4e} ({(a-b)/b:+.3%}); stress equivalent {(a-b)*Ep:+.2f} MPa")
