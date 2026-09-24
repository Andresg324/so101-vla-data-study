import json, sys
p = "/workspace/august_cfg/train_config_runpod.json"
c = json.load(open(p))
for key in sys.argv[1:]:
    *parents, last = key.split(".")
    d = c
    for k in parents:
        d = d[k]
    print(f"dropped {key} = {d.pop(last)!r}")
json.dump(c, open(p, "w"), indent=4)
