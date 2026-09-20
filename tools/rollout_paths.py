# tools/rollout_paths.py
# Shared resolution of timestamped rollout datasets.

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_results import ROLLOUTS

CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/Andresg324"))
CELLS = ["in_distribution", "new_positions", "reduced_lighting",
         "different_object", "distractors", "near_1in", "near_2in", "trained_t2", "spot_check"]
PATTERN = re.compile(r"^rollout_(?P<policy>.+?)_(?P<cell>" + "|".join(CELLS) + r")_(?P<stamp>\d{8}_\d{6})$")

def parse_policy(policy):
    for suffix, seed in (("-seed3000", 3000), ("-seed2000", 2000)):
        if policy.endswith(suffix):
            return policy[: -len(suffix)], seed
    return policy, 1000

def discover(manifest=None):
    """
    {(policy, cell): path} for every rollout named in the ROLLOUT manifest in export_results.

    Confirms that policies used in analyses have the correct timestamp, so redoing any policies,
    or having other versions on a local cache won't leak into analyses unless explictely added.
    """

    names = set(ROLLOUTS if manifest is None else manifest)
    found, missing = {}, []

    for name in sorted(names):
        m = PATTERN.match("rollout_" + name)
        if not m:
            raise ValueError(f"manifest entry does not parse: {name}")
        path = os.path.join(CACHE, "rollout_" + name)
        if not os.path.isdir(path):
            missing.append(name)
            continue
        found[(m["policy"], m["cell"])] = path

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest rollouts not in cache: {sorted(missing)[:5]}"
            + (" ..." if len(missing) > 5 else "")
            )

    return found

def resolve(policy, cell):
    # Path to the latest dataset for one (policy, cell)
    hit = discover().get((policy, cell))
    if hit is None:
        raise FileNotFoundError(f"No dataset for rollout_{policy}_{cell}_*")
    return hit