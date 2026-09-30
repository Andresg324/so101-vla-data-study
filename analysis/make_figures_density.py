#!/usr/bin/env python3
"""
analysis/make_figures_density.py

Figures and tables for the density sweep and its follow-ups. Kept separate from
make_figures.py, which covers the August grid.

Aim error at the held-out positions: grasp bearing (first close-on-target event)
minus the cube's true bearing, one panel per position (E1 to E5), one row per policy.
Every episode is plotted; the thin line is the range, the thick line the interquartile
range, the diamond the median. Episodes with no detected grasp are absent, so the table
reports grasps / episodes scored.

  figures/aim_error_density.{png,pdf}       density sweep, both seeds
  figures/aim_error_fixedstep.{png,pdf}     fixed-step controls beside their epoch-matched pair
  figures/aim_error_randomized.{png,pdf}    August Randomized, both seeds, and on the rebuilt bench
  figures/table_<name>.md                   median [IQR] (grasps/scored) per policy and position
  analysis/out_azimuth/<name>.csv           the same statistics in long form, plus range and median |error|

Two axes of a held-out position (PROTOCOL.md §8.35, §8.38):

  figures/positions_two_axes.{png,pdf}      the board with the E5 ray, the reach point on it and
                                            the furthest trained radius; geometry only, no data
  figures/reach_probe.{png,pdf}             success at E5 and at the reach point by density and
                                            seed, and grasp bearing error at the reach point
  analysis/out_azimuth/reach_probe_success.csv, reach_probe_aim.csv

RUN: python analysis/make_figures_density.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_pose import azimuth_fit, azimuth, T as TRAIN_XY, BASE_X
from rollout_paths import parse_policy
from analyze_results import wilson_ci

ENDS = "analysis/out_endpoints/endpoints_{}.csv"
TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv",
            "documents/density.csv", "documents/followup.csv"]
FIGDIR = "figures"
CSVDIR = "analysis/out_azimuth"

POS = {"E1": (2.0, 7.5), "E2": (6.5, 2.5), "E3": (12.0, 10.0), "E4": (15.5, 6.5), "E5": (19.5, 13.5)}
ORANGE, GREY, DARK, GREEN = "#D9693A", "#8C8C8C", "#444444", "#2E7D5B"

FIGURES = {
    "aim_error_density": [
        ("density5", "new_positions", "Density 5"),
        ("density5-seed2000", "new_positions", "Density 5 (s2000)"),
        ("density10", "new_positions", "Density 10"),
        ("density10-seed2000", "new_positions", "Density 10 (s2000)"),
        ("density25", "new_positions", "Density 25"),
        ("density25-seed2000", "new_positions", "Density 25 (s2000)"),
        ("density50", "new_positions", "Density 50"),
        ("density50-seed2000", "new_positions", "Density 50 (s2000)"),
    ],
    "aim_error_fixedstep": [
        ("density25", "new_positions", "Density 25, epoch-matched"),
        ("density25-fixedstep", "new_positions", "Density 25, fixed-step"),
        ("density50", "new_positions", "Density 50, epoch-matched"),
        ("density50-fixedstep", "new_positions", "Density 50, fixed-step"),
    ],
    "aim_error_randomized": [
        ("randomized", "new_positions", "Randomized, August"),
        ("randomized-seed2000", "new_positions", "Randomized (s2000), August"),
        ("randomized", "new_positions_rebuilt", "Randomized, rebuilt bench"),
    ],
}

# Geometry for the two-axes figures. TRAIN_XY comes from calibrate_pose so the positions
# have one source of truth.
TRAIN_POS = {f"T{k}": v for k, v in TRAIN_XY.items()}
BASE = (BASE_X, 0.0)
CUP = (5.0, 12.5)
REACH = (19.0, 12.7)                                   # §8.35 reach point, on E5's bearing
BOARD = (22.0, 17.0)                                   # overhead frame, inches
R_MAX = max(np.hypot(x - BASE_X, y) for x, y in TRAIN_POS.values())   # 15.21 in, T3
DENSITIES = [5, 10, 25, 50]
SEED_OFFSET = {1000: 0.93, 2000: 1.07}                 # seed 1000 left of each tick, seed 2000 right


def n_scored(lab, policy, cell, inst):
    cond, seed = parse_policy(policy)
    return int(((lab.condition == cond) & (lab.seed == seed)
                & (lab.eval_cell == cell) & (lab.instance == inst)).sum())


def aim_errors(rows, to_az):
    out = []
    for pol, cell, label in rows:
        df = pd.read_csv(ENDS.format(pol))
        df = df[(df.cell == cell) & df.instance.isin(POS)].copy()
        if df.empty:
            raise SystemExit(f"no {cell} grasps in {ENDS.format(pol)}")
        true = np.array([azimuth(*POS[i]) for i in df.instance])
        df["err"] = to_az(df["pan"]) - true
        df["label"] = label
        out.append(df[["label", "instance", "episode", "success", "err"]])
    return pd.concat(out, ignore_index=True)


def aim_error_table(name, rows, err, lab):
    long_rows, md_rows = [], []
    for pol, cell, label in rows:
        md = [label]
        for inst in POS:
            e = err[(err.label == label) & (err.instance == inst)].err
            n_ep = n_scored(lab, pol, cell, inst)
            long_rows.append({"policy": label, "instance": inst, "grasps": len(e), "scored": n_ep,
                              "median_err": e.median(), "q1": e.quantile(0.25), "q3": e.quantile(0.75),
                              "min": e.min(), "max": e.max(), "median_abs_err": e.abs().median()})
            md.append(f"{e.median():+.1f} [{e.quantile(0.25):+.1f}, {e.quantile(0.75):+.1f}] ({len(e)}/{n_ep})"
                      if len(e) else f"n/a (0/{n_ep})")
        md_rows.append(md)

    os.makedirs(CSVDIR, exist_ok=True)
    pd.DataFrame(long_rows).round(2).to_csv(os.path.join(CSVDIR, f"{name}.csv"), index=False)

    head = ["Policy"] + [f"{i} ({azimuth(*POS[i]):.0f}°)" for i in POS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in md_rows]
    lines.append("")
    lines.append("Signed error in degrees, grasp bearing minus cube bearing: median [interquartile range] "
                 "(episodes with a detected grasp / episodes scored).")
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, f"table_{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {CSVDIR}/{name}.csv and {FIGDIR}/table_{name}.md")


def fig_aim_error(name, rows, err):
    labels = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]            # first row in the list is drawn at the top
    fig, axes = plt.subplots(1, len(POS), sharey=True, sharex=True,
                             figsize=(2.6 * len(POS), 0.5 * len(rows) + 1.6))

    for ax, inst in zip(axes, POS):
        for yi, label in zip(y, labels):
            e = err[(err.label == label) & (err.instance == inst)]
            if e.empty:
                continue
            ax.hlines(yi, e.err.min(), e.err.max(), color=GREY, lw=1.0, zorder=1)
            ax.hlines(yi, e.err.quantile(0.25), e.err.quantile(0.75), color=GREY, lw=4, alpha=0.6, zorder=2)
            fail, win = e[e.success != 1], e[e.success == 1]
            ax.scatter(fail.err, np.full(len(fail), yi), facecolors="white", edgecolors=GREY,
                       s=22, linewidths=1.0, zorder=3)
            ax.scatter(win.err, np.full(len(win), yi), color=GREEN, s=22, zorder=3)
            ax.scatter([e.err.median()], [yi], marker="D", color=ORANGE, s=50, zorder=4)
        ax.axvline(0, ls="--", color=DARK, lw=1.2, zorder=0)
        ax.set_title(f"{inst}, cube at {azimuth(*POS[inst]):.0f}°", fontsize=10)
        ax.grid(axis="x", color="#E3E3E3")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.supxlabel("Grasp bearing minus cube bearing (deg)")

    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="Episode, failed"),
        Line2D([], [], marker="o", ls="", color=GREEN, label="Episode, succeeded"),
        Line2D([], [], marker="D", ls="", color=ORANGE, label="Median"),
        Line2D([], [], color=GREY, lw=4, alpha=0.6, label="Interquartile range"),
        Line2D([], [], color=GREY, lw=1.0, label="Range"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR}/{name}.png and .pdf")


def fig_positions(name="positions_two_axes"):
    """Figure 1: the board with the E5 ray, the reach point on it, and the furthest trained radius.
    Geometry only; every annotation that is not a position label lives in the legend below."""
    fig, ax = plt.subplots(figsize=(3.6, 3.7))
    ax.add_patch(plt.Rectangle((0, 0), *BOARD, facecolor="#F1F1F1", edgecolor="#BBBBBB", lw=0.8, zorder=0))
    ax.add_patch(plt.Rectangle((BASE_X - 1.5, 0), 3.0, 2.4, facecolor="#CFCFCF", edgecolor=GREY, lw=0.8, zorder=1))
    ax.text(BASE_X, 1.2, "arm", ha="center", va="center", fontsize=7, color=DARK)
    ax.add_patch(plt.Circle(CUP, 1.75, facecolor="white", edgecolor=GREY, lw=1.0, zorder=1))
    ax.text(*CUP, "cup", ha="center", va="center", fontsize=7, color=DARK)

    pts = np.array(list(TRAIN_POS.values()))
    hull = ConvexHull(pts)
    poly = np.vstack([pts[hull.vertices], pts[hull.vertices[:1]]])
    ax.plot(poly[:, 0], poly[:, 1], ls="--", lw=0.8, color="#B5B5B5", zorder=1)

    th = np.linspace(-np.pi / 2, np.pi / 2, 800)
    cx, cy = BASE_X + R_MAX * np.sin(th), R_MAX * np.cos(th)
    m = (cx >= 0) & (cx <= BOARD[0]) & (cy <= BOARD[1])
    ax.plot(cx[m], cy[m], ls=":", lw=1.3, color=ORANGE, zorder=2)

    d = np.array(POS["E5"]) - np.array(BASE)
    t_end = min((BOARD[0] - BASE_X) / d[0], BOARD[1] / d[1])
    end = np.array(BASE) + d * t_end
    ax.plot([BASE_X, end[0]], [0, end[1]], ls="--", lw=1.0, color=ORANGE, zorder=2)

    for k, (x, y) in TRAIN_POS.items():
        ax.scatter(x, y, s=24, color=DARK, zorder=3)
        ax.annotate(k, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6, color=DARK)
    for k, (x, y) in POS.items():
        if k == "E5":
            continue
        ax.scatter(x, y, s=24, facecolors="white", edgecolors=DARK, linewidths=1.0, zorder=3)
        ax.annotate(k, (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6, color=DARK)
    ax.scatter(*POS["E5"], s=34, facecolors="white", edgecolors=ORANGE, linewidths=1.4, zorder=4)
    ax.annotate("E5", POS["E5"], xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=DARK)
    ax.scatter(*REACH, s=34, marker="D", color=ORANGE, zorder=4)

    ax.set_xlim(-0.4, BOARD[0] + 0.4)
    ax.set_ylim(-0.4, BOARD[1] + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (in)", fontsize=8)
    ax.set_ylabel("y (in)", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", color=DARK, label="training position"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=DARK, label="held-out position"),
        Line2D([], [], ls="--", color="#B5B5B5", label="training convex hull"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.4, label="E5"),
        Line2D([], [], marker="D", ls="", color=ORANGE, label="reach point, 1 in inward from E5"),
        Line2D([], [], ls="--", color=ORANGE, label=f"E5 bearing, {azimuth(*POS['E5']):.1f}°"),
        Line2D([], [], ls=":", color=ORANGE, lw=1.3, label=f"furthest trained radius, {R_MAX:.1f} in (T3)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5, frameon=False,
               handletextpad=0.5, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR}/{name}.png and .pdf")


def fig_reach_probe(to_az, lab, name="reach_probe"):
    """Figure 3: success at E5 (registered cell) and at the reach point by density and seed,
    and grasp bearing error at the reach point. Reads the trackers and the endpoint files."""
    rows = []
    for dens in DENSITIES:
        for seed in SEED_OFFSET:
            cond = f"density{dens}"
            base = lab[(lab.condition == cond) & (lab.seed == seed)]
            cells = {"E5": base[(base.eval_cell == "new_positions") & (base.instance == "E5")],
                     "reach": base[base.eval_cell == "reach_e5"]}
            for cell, g in cells.items():
                if g.empty:
                    raise SystemExit(f"no rows for {cond} seed {seed} at {cell}")
                k, n = int(g.success.sum()), len(g)
                lo, hi = wilson_ci(k, n)
                rows.append({"density": dens, "seed": seed, "cell": cell, "successes": k, "n": n,
                             "rate": k / n, "ci_low": lo, "ci_high": hi})
    succ = pd.DataFrame(rows)

    true = float(azimuth(*REACH))
    parts = []
    for dens in DENSITIES:
        for seed in SEED_OFFSET:
            pol = f"density{dens}" + ("" if seed == 1000 else "-seed2000")
            df = pd.read_csv(ENDS.format(pol))
            df = df[df.cell == "reach_e5"]
            if df.empty:
                raise SystemExit(f"no reach_e5 grasps in {ENDS.format(pol)}")
            parts.append(pd.DataFrame({"density": dens, "seed": seed, "episode": df.episode.to_numpy(),
                                       "success": df.success.to_numpy(), "err": to_az(df["pan"]) - true}))
    aim = pd.concat(parts, ignore_index=True)

    os.makedirs(CSVDIR, exist_ok=True)
    succ.round(4).to_csv(os.path.join(CSVDIR, f"{name}_success.csv"), index=False)
    aim.round(2).to_csv(os.path.join(CSVDIR, f"{name}_aim.csv"), index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7))

    for seed, off in SEED_OFFSET.items():
        # E5 as a hollow ring slightly left, the reach point as a filled dot slightly right, so the
        # two stay visible when both sit at zero
        for cell, fc, ec, sz, nudge, z in (("E5", "white", GREY, 60, 0.985, 4), ("reach", ORANGE, ORANGE, 34, 1.015, 3)):
            s = succ[(succ.seed == seed) & (succ.cell == cell)].sort_values("density")
            xs = s.density.to_numpy() * off * nudge
            a1.vlines(xs, s.ci_low, s.ci_high, color=ec, lw=0.9, zorder=2)
            a1.scatter(xs, s.rate, s=sz, facecolors=fc, edgecolors=ec, linewidths=1.2, zorder=z)
    a1.set_ylim(-0.04, 1.04)
    a1.set_ylabel("Success rate, 5 episodes (Wilson 95 %)", fontsize=8)
    a1.set_title("E5 against the reach point", fontsize=9)
    a1.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="E5 (registered cell)"),
        Line2D([], [], marker="o", ls="", color=ORANGE, label="reach point, one inch inward"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=7, frameon=False)

    half = float(np.degrees(np.arctan(0.5 / np.hypot(REACH[0] - BASE_X, REACH[1]))))   # cube half-width, deg
    a2.axhspan(-half, half, color="#E6E6E6", zorder=0)
    a2.axhline(0, ls="--", color=DARK, lw=1.0, zorder=1)
    for seed, off in SEED_OFFSET.items():
        g = aim[aim.seed == seed]
        xs = g.density.to_numpy() * off
        fail, win = g.success != 1, g.success == 1
        a2.scatter(xs[fail.to_numpy()], g.err[fail], s=18, facecolors="white", edgecolors=GREY, linewidths=0.9, zorder=2)
        a2.scatter(xs[win.to_numpy()], g.err[win], s=18, color=GREEN, zorder=2)
        med = g.groupby("density").err.median()
        a2.scatter(med.index.to_numpy() * off, med.to_numpy(), marker="D", s=52, color=ORANGE, zorder=4)
    a2.set_ylabel("Grasp bearing minus cube bearing (deg)", fontsize=8)
    a2.set_title("Aim at the reach point", fontsize=9)
    a2.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="episode, failed"),
        Line2D([], [], marker="o", ls="", color=GREEN, label="episode, succeeded"),
        Line2D([], [], marker="D", ls="", color=ORANGE, label="median"),
    ], loc="lower right", fontsize=7, frameon=False)

    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(DENSITIES)
        ax.set_xticklabels(DENSITIES)
        ax.minorticks_off()
        ax.set_xlim(3.8, 66)
        ax.set_xlabel("Demonstrations per position", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="#E3E3E3")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGDIR}/{name}.png and .pdf")


def main():
    fit, _, _ = azimuth_fit()
    to_az = lambda p: fit.predict(np.asarray(p, float).reshape(-1, 1))
    lab = pd.concat([pd.read_csv(p) for p in TRACKERS], ignore_index=True)
    for name, rows in FIGURES.items():
        err = aim_errors(rows, to_az)
        aim_error_table(name, rows, err, lab)
        fig_aim_error(name, rows, err)
    fig_positions()
    fig_reach_probe(to_az, lab)


if __name__ == "__main__":
    main()
    