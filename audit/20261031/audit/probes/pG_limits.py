from gensec.io_yaml import _parse_stress_limits, _parse_sls_moduli
for blk in ({"kind": "characteristic"}, {"kind": "characteristic", "fyk": 450.0, "fpk": 1860.0}):
    L = _parse_stress_limits("probe", blk, "ec2")
    print(blk, "->", {k: v for k, v in vars(L).items() if not k.startswith("_")})
for bad in ({}, {"z": 0.0}):
    try: _parse_sls_moduli("probe", bad); print(bad, "ACCEPTED")
    except ValueError as e: print(bad, "REFUSED:", str(e).splitlines()[0][:90])
