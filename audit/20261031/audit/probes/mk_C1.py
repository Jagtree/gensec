import sys
p=sys.argv[1]
t=open(p+"/C1_example_input_lumped.yaml").read()
r={"    - {y: 40,  As: 942.5, material: steel_1, n_bars: 3, diameter: 20}\n":
 "    - {y: 40, x: 40,  As: 314.17, material: steel_1}\n    - {y: 40, x: 150, As: 314.17, material: steel_1}\n    - {y: 40, x: 260, As: 314.16, material: steel_1}\n",
 "    - {y: 560, As: 942.5, material: steel_1, n_bars: 3, diameter: 20}\n":
 "    - {y: 560, x: 40,  As: 314.17, material: steel_1}\n    - {y: 560, x: 150, As: 314.17, material: steel_1}\n    - {y: 560, x: 260, As: 314.16, material: steel_1}\n",
 "    - {y: 300, As: 402.1, material: steel_1, n_bars: 2, diameter: 16}\n":
 "    - {y: 300, x: 40,  As: 201.05, material: steel_1}\n    - {y: 300, x: 260, As: 201.05, material: steel_1}\n"}
for a,b in r.items():
    assert a in t, a; t=t.replace(a,b)
open(p+"/C1_example_input_distributed.yaml","w").write("# AUD-C: example_input.yaml with the same bars placed at explicit x (cover 40)\n"+t)
