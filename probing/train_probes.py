"""
Trains linear probes to predict episode success from a policy's hidden activations, per
condition, with controls from the deception-probe evaluation protocol:
  - AUROC only (threshold-free). Accuracy is threshold-dependent and misleads (see paper's Qwen-3B exhibit).
  - Episode-grouped splits (no timestep leakage across train/test).
  - Repeated-split 95% CI on AUROC (small-n uncertainty).
  - De-confound: within-cell vs pooled AUROC. A big pooled>within-cell gap means the probe is
    reading 'which cell' (difficulty), not genuine self-knowledge of failure.
RUN: python probing/train_probes.py probing/out/activations_synthetic.npz --outdir probing/out
"""

import argparse, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", message=".*matmul.*")
np.seterr(all="ignore")

def grouped_auroc(X, y, groups, seed):
    # One episode=grouped split to test AUROC (nan if a split is degenerate).
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])

def auroc_with_ci(X, y, groups, n_repeats=100):
    # Repeat the grouped splits and return the median, 2.5% and 97.5% for a 95% CI
    vals = [grouped_auroc(X, y, groups, s) for s in range(n_repeats)]
    vals = [v for v in vals if v == v]
    if not vals:
        return(np.nan, np.nan, np.nan)
    return(float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def within_cell_auroc(X, y, cells, groups, n_repeats=40):
    # De-confound: AUROC inside each cell (control for difficulty), and then averaged
    per = []
    for cl in np.unique(cells):
        m = cells == cl
        if len(set(y[m])) < 2:
            continue
        med, _, _ = auroc_with_ci(X[m], y[m], groups[m], n_repeats)
        if med == med:
            per.append(med)
    return float(np.mean(per)) if per else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("activations"); ap.add_argument("--outdir", default="probing/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.activations, allow_pickle=True)
    X, condition, eval_cell = d["X"], d["condition"], d["eval_cell"]
    episode, success, tfe = d["episode"], d["success"].astype(int), d["t_from_end"]

    print(f"{'condition':12s} {'pooled AUROC [95% CI]':30s} {'within-cell':12s} gap")
    results = []
    for c in sorted(set(condition)):
        m = condition == c
        med, lo, hi = auroc_with_ci(X[m], success[m], episode[m])
        wc = within_cell_auroc(X[m], success[m], eval_cell[m], episode[m])
        gap = med - wc if (med == med and wc == wc) else np.nan
        results.append((c, med, lo, hi, wc, gap))
        ci = f"{med:.3f} [{lo:.3f}, {hi:.3f}]"
        print(f"{c:12s} {ci:30s} {wc:<12.3f}  {gap:+.3f}")

    # Lead-time curve, how early is failure decodable
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, success, episode))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], success[tr])
    p, yt, bt = clf.predict_proba(X[te])[:, 1], success[te], tfe[te]

    # Bin t_from_end instead of using exact values. Activations are captured at
    # SmolVLA's action-chunk boundaries, so the raw t_from_end values are sparse
    # and irregular. Grouping by exact value leaves one or two samples per value,
    # nearly all single-class, and the curve collapses to a few points.
    
    BIN = 10 # Steps per bucket (e.g., 1/3 of a second at 30 FPS)
    MIN_PER_BIN = 20

    xs, ys = [], []
    buckets = (bt // BIN) * BIN
    for b in sorted(set(buckets)):
        mm = buckets == b
        if mm.sum() >= MIN_PER_BIN and len(set(yt[mm])) == 2:
            xs.append(b + BIN / 2) # Plot at the center of the bucket
            ys.append(roc_auc_score(yt[mm], p[mm]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.invert_xaxis()
    ax.axhline(0.5, ls="--", c="grey")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel(f"Steps before end of episode (binned, width {BIN})")
    ax.set_ylabel("Probe AUROC")
    ax.set_title("How early is success or failure decodable from a policy's internal activations?")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "lead_time.png"), dpi=150)

    with open(os.path.join(args.outdir, "probe_auroc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "pooled_auroc", "ci_low", "ci_high", "within_cell_auroc", "gap"])
        for r in results:
            w.writerow(r)
    print(f"\nSaved probe_auroc.csv and lead_time.png to {args.outdir}/")

if __name__ == "__main__":
    main()
