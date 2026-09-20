#!/usr/bin/env python3
"""tools/azimuth_analysis.py — aiming error in calibrated azimuth. """

import glob
import os
import sys

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibrate_pose import azimuth_fit, azimuth, BASE_X
from rollout_paths import resolve, parse_policy

# ------------------------ Set up ---------------------------------------

OUTDIR = "analysis/out_azimuth"

POS = {"E1": (2.0, 7.5), "E2": (6.5, 2.5), "E3": (12.0, 10.0), "E4": (15.5, 6.5), "E5": (19.5, 13.5)}

CELL_POS = {"in_distribution": (15.5, 10.0), "near_1in": (15.5, 9.0), "near_2in": (15.5, 8.0)}

INTERP = {"E2", "E3", "E4"}

TRAIN_POS = {"T1": (2.0, 2.5), "T2": (6.5, 7.5), "T3": (8.5, 15.0), "T4": (12.0, 14.0),
             "T5": (15.5, 2.5), "T6": (15.5, 10.0), "T7": (15.5, 14.25),
             "T8": (20.5, 2.5), "T9": (20.5, 6.5), "T10": (20.5, 10.0)}

SWEEP = ["density5", "density10", "density25", "density50",
         "density5-seed2000", "density10-seed2000", "density25-seed2000",
         "density50-seed2000", "density25-fixedstep", "density50-fixedstep"]

CLEAN = ("clean", "clean-seed2000")

def rollout_actions(policy, cell):
    # Episode indices and the action array for the newest rollout of this pair
    root = resolve(policy, cell)
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.episode_index != 0].sort_values(["episode_index", "frame_index"], kind="stable")
    return df.episode_index.to_numpy(), np.stack(df["action"].to_numpy()).astype(float)

os.makedirs(OUTDIR, exist_ok=True)

TRAIN_AZ = {k: azimuth(*xy) for k, xy in TRAIN_POS.items()}     # Azimuth for all trained positions
TEN = list(TRAIN_AZ)

BRACKET = {}
for _inst, _xy in POS.items():
    _t = azimuth(*_xy)
    _lo = [t for t in TEN if TRAIN_AZ[t] <= _t]
    _hi = [t for t in TEN if TRAIN_AZ[t] >= _t]
    BRACKET[_inst] = (max(_lo, key=lambda t: TRAIN_AZ[t]) if _lo else None,
                      min(_hi, key=lambda t: TRAIN_AZ[t]) if _hi else None)

def nearest_trained(az, trained=TEN):
    return min(trained, key=lambda t: abs(TRAIN_AZ[t] - az))

def frac(az, inst):
    lo, hi = BRACKET[inst]
    if lo is None or hi is None or lo == hi:
        return np.nan
    return (az - TRAIN_AZ[lo]) / (TRAIN_AZ[hi] - TRAIN_AZ[lo])

# ---------------------- Calibrate -------------------------------------

fit, X, az_true = azimuth_fit()
pan = X[:, [0]]
to_az = lambda p: fit.predict(np.asarray(p, float).reshape(-1, 1))

loo = cross_val_predict(LinearRegression(), pan, az_true, cv=LeaveOneOut())
resid = np.abs(loo - az_true)
r2 = 1 - ((loo - az_true) ** 2).sum() / ((az_true - az_true.mean()) ** 2).sum()

pd.DataFrame([{
    "slope_deg_per_unit": fit.coef_[0],
    "intercept_deg": fit.intercept_,
    "n": len(X),
    "loo_median_abs_err_deg": float(np.median(resid)),
    "loo_p90_abs_err_deg": float(np.percentile(resid, 90)),
    "loo_r2": float(r2),
}]).round(4).to_csv(os.path.join(OUTDIR, "calibration.csv"), index=False)

print("--- Calibration ---")
print(f"Azimuth = {fit.coef_[0]:.4f} * pan + {fit.intercept_:.3f} (n={len(X)})")
print(f"leave-one-out | error |: median {np.median(resid):.2f} deg,",
      f"90th percentile {np.percentile(resid, 90):.2f} deg, R2 {r2:.4f}")

ends = []
for path in sorted(glob.glob("analysis/out_endpoints/endpoints_*.csv")):
    df = pd.read_csv(path)
    df["policy"] = os.path.basename(path)[len("endpoints_"):-len(".csv")]
    df["az"] = to_az(df["pan"].to_numpy())
    ends.append(df)

ends = pd.concat(ends, ignore_index=True)

LAB = pd.concat([pd.read_csv("documents/results_full.csv"),
                 pd.read_csv("documents/exploratory.csv"),
                 pd.read_csv("documents/density.csv")], ignore_index=True)

cov = []
for p in SWEEP:
    cond, seed = parse_policy(p)
    n_tot = len(LAB[(LAB.condition == cond) & (LAB.seed == seed)
                    & (LAB.eval_cell == "new_positions")])
    n_g = len(ends[(ends.policy == p) & (ends.cell == "new_positions")])
    cov.append({"policy": p, "grasps": n_g, "episodes": n_tot,
                "coverage": round(n_g / n_tot, 2) if n_tot else np.nan})
print("\n--- Grasp coverage at held-out positions ---")
print(pd.DataFrame(cov).to_string(index=False))

v = ends[ends.policy.isin(CLEAN) & (ends.cell == "in_distribution")]
print(f"\nvalidation: clean grasp azimuth at T6, true value {azimuth(15.5, 10.0):.2f} deg")
print(v.groupby("policy")["az"].agg(n="size", median="median").round(2).to_string())

# --------------------- Clean Trajectory Envelope -----------------------
# The grasp event is not comparable across these cells: the arm closes at frame ~215
# in distribution and at ~470 when the cube is displaced, because it reaches the location,
# doesn't find the cube, hovers, then closes late, so we're comparing trajectories and not
# grasp events

print("\n--- Clean trajectory envelope: furthest azimuth reached ---")
rows, store = [], {}
for pol in CLEAN:
    for cell, xy in CELL_POS.items():
        req = azimuth(*xy)
        ep, A = rollout_actions(pol, cell)
        mx = pd.Series(to_az(A[:, 0]), index=ep).groupby(level=0).max()
        store[(pol, cell)] = mx
        rows.append({
            "policy": pol,
            "cell": cell,
            "n": len(mx),
            "median_max_az": mx.median(),
            "q1": mx.quantile(0.25),
            "q3": mx.quantile(0.75),
            "required_az": req,
            "episodes_reaching_target": int((mx >= req).sum())
        })

env = pd.DataFrame(rows).round(2)
print(env.to_string(index=False))
env.to_csv(os.path.join(OUTDIR, "clean_envelope.csv"), index=False)

print()
for pol in CLEAN:
    a, b = store[(pol, "in_distribution")], store[(pol, "near_2in")]
    u = mannwhitneyu(b, a, alternative="greater")
    print(f"{pol}: near_2in vs in_distribution max azimuth, "
          f"median difference {b.median() - a.median():+.2f} deg"
          f" (required {azimuth(*CELL_POS['near_2in']) - azimuth(*CELL_POS['in_distribution']):+.2f}), "
          f"U={u.statistic:.0f}, p={u.pvalue:.3f}")

# ----------------------- Randomized Aiming ----------------------------

r = ends[ends.policy.str.startswith("randomized") & (ends.cell == "new_positions")].copy()
n_all = len(r)
r = r[r.instance.isin(POS)]
if len(r) < n_all:
    print(f"dropped {n_all - len(r)} new_positions rows with no recognized instance: "
          f"{sorted(set(ends.loc[ends.cell == 'new_positions', 'instance']) - set(POS))}")
    
r["true_az"] = [azimuth(*POS[i]) for i in r.instance]
r["err"] = r["az"] - r["true_az"]
r["split"] = np.where(r.instance.isin(INTERP), "interpolation", "extrapolation")
r["dist_in"] = [np.hypot(POS[i][0] - BASE_X, POS[i][1]) for i in r.instance]
r["cube_widths"] = r.err.abs() / np.degrees(2 * np.arctan(0.5 / r.dist_in))


print("\n--- Randomized Aiming Error at Held-out Positions ---")
print(r[["policy", "episode", "instance", "split", "az", "true_az", "err"]].round(1).to_string(index=False))
print("\n" + r.groupby("split")["err"].agg(
    n="size", mean_abs=lambda s: s.abs().mean(), median_abs=lambda s: s.abs().median()).round(2).to_string())

i = r.loc[r.split == "interpolation", "err"].abs()
e = r.loc[r.split == "extrapolation", "err"].abs()

print("\nMann-Whitney (extrapolation > interpolation):", mannwhitneyu(e, i, alternative="greater"))
print("extrapolation signed errors:", r.loc[r.split == "extrapolation", "err"].round(1).to_list())

print(r.groupby("split")["cube_widths"].agg(n="size", mean="mean", median="median").round(2))

r.round(2).to_csv(os.path.join(OUTDIR, "randomized_aiming.csv"), index=False)

# ------------------------- Aim Invariance Across Cells ----------------------

# Absolute pose is not comparable across policies as clean grasps with lift ~9 and elbow ~25, but recovery with lift ~25 and elbow ~8,
# This is a difference in teleoperated grasp style rather than aiming, as such, everything here is within a policy across cells

# New positions is excluded from IQR comparisons because the cube is at five difference places, and as such the policy will
# have a large spread

SAME_TARGET = ["in_distribution", "reduced_lighting", "different_object", "distractors"]
REGISTERED = SAME_TARGET + ["new_positions"]

med = ends.pivot_table(index="policy", columns="cell", values="az", aggfunc="median")
iqr = ends.pivot_table(index="policy", columns="cell", values="az", aggfunc=lambda s: s.quantile(0.75) - s.quantile(0.25))
same = [c for c in SAME_TARGET if c in med.columns]
allc = [c for c in REGISTERED if c in med.columns]

print(f"\n--- Commanded Azimuth by Cell (cube at T6 = {azimuth(15.5, 10.0):.2f} except new_positions) ---")
print(med[allc].round(1).to_string())

print("\nspread of cell medians within each policy, same-target cells only (deg)")
print((med[same].max(axis=1) - med[same].min(axis=1)).round(1).to_string())

print("\nsame, including new_positions (deg)")
print((med[allc].max(axis=1) - med[allc].min(axis=1)).round(1).to_string())

print("\nIQR width of commanded azimuth, same-target cells only (deg)")
print(iqr[same].round(1).to_string())

med[allc].round(3).to_csv(os.path.join(OUTDIR, "aim_by_cell.csv"))
iqr[allc].round(3).to_csv(os.path.join(OUTDIR, "aim_iqr_by_cell.csv"))

# ------------------------- Density Probe (PROTOCOL.md §8.30) ----------------------
# Amendment 30 chose T2 because it sits on the opposite side of the arm from T6, so one
# fixed sweep cannot succeed at both. Two questions: does it aim correctly at each
# trained position, and what does it do at a position it never saw.

T6_AZ_D, T2_AZ_D = azimuth(15.5, 10.0), azimuth(6.5, 7.5)
DENS_POS = {"in_distribution": (15.5, 10.0), "trained_t2": (6.5, 7.5)}

d_ends = ends[ends.policy == "density"]

if len(d_ends):
    rows = []
    for cell, xy in DENS_POS.items():
        g = d_ends[d_ends.cell == cell]
        if not len(g):
            continue
        true_az = azimuth(*xy)
        rows.append({"cell": cell, "n": len(g), "true_az": true_az,
                     "median_az": g.az.median(), "err": g.az.median() - true_az})

    if rows:
        dens = pd.DataFrame(rows).round(2)
        
        print("\n--- Density: grasp azimuth at the two trained positions ---")
        print(dens.to_string(index=False))
        dens.to_csv(os.path.join(OUTDIR, "density_aim.csv"), index=False)

# ------------------- Density sweep aiming (PROTOCOL.md §8.32) -------------------
# Grasp bearing is primary for the sweep: coverage is high enough to use an
# event-defined measure. Conditioned on the gripper having closed.

d = ends[ends.policy.isin(SWEEP) & (ends.cell == "new_positions")].copy()
d = d[d.instance.isin(POS)]
d["true_az"] = [azimuth(*POS[i]) for i in d.instance]
d["err"] = d.az - d.true_az
d["split"] = np.where(d.instance.isin(INTERP), "interpolation", "extrapolation")
d["dist_in"] = [np.hypot(POS[i][0] - BASE_X, POS[i][1]) for i in d.instance]
d["cube_widths"] = d.err.abs() / np.degrees(2 * np.arctan(0.5 / d.dist_in))
d["near"] = [nearest_trained(a) for a in d.az]
d["frac_obs"] = [frac(a, i) for a, i in zip(d.az, d.instance)]
d["frac_true"] = [frac(t, i) for t, i in zip(d.true_az, d.instance)]

print("\n--- Density sweep aiming error at held-out positions ---")
print(d.groupby(["policy", "instance"]).agg(
    n=("az", "size"), median_az=("az", "median"), true=("true_az", "first"),
    median_abs_err=("err", lambda s: s.abs().median()),
    frac_obs=("frac_obs", "median"), frac_true=("frac_true", "first"),
    picked=("near", lambda s: s.mode().iat[0])).round(2).to_string())

di = d.loc[d.split == "interpolation", "err"].abs()
de = d.loc[d.split == "extrapolation", "err"].abs()
print("\nMann-Whitney (extrapolation > interpolation), exploratory, episodes pooled "
      "across policies:", mannwhitneyu(de, di, alternative="greater"))
d.round(2).to_csv(os.path.join(OUTDIR, "density_aiming.csv"), index=False)

# ------------------- Sweep aim at trained positions (validation) -------------------

print("\nspot_check instance values:",
      sorted(ends.loc[ends.cell == "spot_check", "instance"].dropna().unique()))

t = ends[ends.policy.isin(SWEEP) & ends.cell.isin(["in_distribution", "spot_check"])].copy()
t["target"] = np.where(t.cell == "in_distribution", "T6", t.instance)
t = t[t.target.isin(TRAIN_AZ)]
if t.empty:
    print("\nno sweep trained-position rows matched; check the spot_check instance format")
else:
    t["true_az"] = [TRAIN_AZ[k] for k in t.target]
    t["err"] = t.az - t.true_az
    print("\n--- Sweep aim at trained positions ---")
    print(t.groupby(["policy", "target"])["err"].agg(
        n="size", median="median", median_abs=lambda s: s.abs().median()).round(2).to_string())
    t.round(2).to_csv(os.path.join(OUTDIR, "sweep_trained_aim.csv"), index=False)

# ---- Settled bearing at the held-out positions, same measure for every policy ----
# Almost nothing grasps at a held-out position, so the endpoint table is too thin to
# compare policies. Use the bearing the arm settles on instead: the median commanded
# azimuth over the last quarter of the episode, after it has committed to a direction.


def settled_new_positions(policy):
    ep_idx, A = rollout_actions(policy, "new_positions")
    s = pd.Series(to_az(A[:, 0]), index=ep_idx)
    s.index.name = "episode"
    settled = s.groupby(level=0).apply(lambda g: g.iloc[int(len(g) * 0.75):].median())
    cond, seed = parse_policy(policy)
    lab = (LAB[(LAB.condition == cond) & (LAB.seed == seed)
               & (LAB.eval_cell == "new_positions")].set_index("episode"))
    if not lab.index.is_unique:
        dup = lab.index[lab.index.duplicated()].unique().tolist()
        raise SystemExit(f"{policy}: duplicate episode labels {dup}")
    d = pd.DataFrame({"settled_az": settled}).join(lab[["instance", "success"]])
    d["policy"] = policy
    d["true_az"] = [azimuth(*POS[i]) if i in POS else np.nan for i in d.instance]
    return d.reset_index()

frames = []
for pol in ["randomized", "randomized-seed2000", "density"] + SWEEP:
    try:
        frames.append(settled_new_positions(pol))
    except FileNotFoundError:
        print(f" no new_positions rollout for {pol}, skipped")

if frames:
    sn = pd.concat(frames, ignore_index=True)
    n_drop = int((sn.success == 1).sum())
    print(f"\ndropping {n_drop} successful episodes from the settled measure "
          f"(delivery puts the cup bearing in the window)")

    if n_drop != 3:
        raise SystemExit(f"expected 3 successful held-out episodes, found {n_drop}")
    
    sn = sn[sn.success != 1].copy()
    sn["err"] = sn.settled_az - sn.true_az
    TRAINED = {"density": ["T6", "T2"]}
    sn["near"] = [nearest_trained(a, TRAINED.get(p, TEN))
                  for p, a in zip(sn.policy, sn.settled_az)]
    sn.round(2).to_csv(os.path.join(OUTDIR, "settled_new_positions.csv"), index=False)

    print(f"\n--- Settled bearing at held-out positions "
          f"(density probe trained at T6 {T6_AZ_D:.1f} deg and T2 {T2_AZ_D:.1f} deg, density sweep trained at all 10 training locations) ---")
    print(sn.groupby(["policy", "instance"]).agg(
        n=("settled_az", "size"),
        settled=("settled_az", "median"),
        true=("true_az", "first"),
        miss_vs_true=("err", lambda s: s.abs().median()),
        picked=("near", lambda s: s.mode().iat[0]),
    ).round(1).to_string())

print(f"\nSaved to {OUTDIR}/")