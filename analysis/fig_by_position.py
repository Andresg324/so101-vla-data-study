#!/usr/bin/env python3
"""
analysis/fig_aim_by_position.py

Grasp bearing at each held-out position: one panel per position (E1 to E5), one row per
policy. Hollow circles are failed episodes, filled circles successful ones, at the first
close-on-target event. The diamond is the median and the dashed line the cube's true
bearing. Episodes with no detected grasp are absent, so each row is labeled with
grasps / scored episodes.

  figures/aim_by_position_density.{png,pdf}      density sweep, both seeds
  figures/aim_by_position_fixedstep.{png,pdf}    fixed-step controls beside their epoch-matched pair
  figures/aim_by_position_randomized.{png,pdf}   August Randomized, both seeds, and on the rebuilt bench

RUN: python analysis/fig_aim_by_position.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from calibrate_pose import azimuth_fit, azimuth
from rollout_paths import parse_policy

ENDS = "analysis/out_endpoints/endpoints_{}.csv"
TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv",
            "documents/density.csv", "documents/followup.csv"]
FIGDIR = "figures"

POS = {"E1": (2.0, 7.5), "E2": (6.5, 2.5), "E3": (12.0, 10.0), "E4": (15.5, 6.5), "E5": (19.5, 13.5)}
TRAIN_POS = {"T1": (2.0, 2.5), "T2": (6.5, 7.5), "T3": (8.5, 15.0), "T4": (12.0, 14.0),
             "T5": (15.5, 2.5), "T6": (15.5, 10.0), "T7": (15.5, 14.25),
             "T8": (20.5, 2.5), "T9": (20.5, 6.5), "T10": (20.5, 10.0)}

ORANGE, GREY, DARK, GREEN = "#D9693A", "#8C8C8C", "#444444", "#2E7D5B"
ANNOTATE_ABOVE = 5.0      # label the median's offset when |offset| is at least this, degrees
SHOW_TRAINED = False      # faint dotted lines at trained bearings inside each panel's range
PANEL_W, ROW_H = 3.0, 0.55

FIGURES = {
    "aim_by_position_density": [
        ("density5", "new_positions", "Density 5"),
        ("density5-seed2000", "new_positions", "Density 5 (s2000)"),
        ("density10", "new_positions", "Density 10"),
        ("density10-seed2000", "new_positions", "Density 10 (s2000)"),
        ("density25", "new_positions", "Density 25"),
        ("density25-seed2000", "new_positions", "Density 25 (s2000)"),
        ("density50", "new_positions", "Density 50"),
        ("density50-seed2000", "new_positions", "Density 50 (s2000)"),
    ],
    "aim_by_position_fixedstep": [
        ("density25", "new_positions", "Density 25, epoch-matched"),
        ("density25-fixedstep", "new_positions", "Density 25, fixed-step"),
        ("density50", "new_positions", "Density 50, epoch-matched"),
        ("density50-fixedstep", "new_positions", "Density 50, fixed-step"),
    ],
    "aim_by_position_randomized": [
        ("randomized", "new_positions", "Randomized, August"),
        ("randomized-seed2000", "new_positions", "Randomized (s2000), August"),
        ("randomized", "new_positions_rebuilt", "Randomized, rebuilt bench"),
    ],
}


def load_grasps(policy, cell, to_az):
    df = pd.read_csv(ENDS.format(policy))
    df = df[df.cell == cell].copy()
    if df.empty:
        raise SystemExit(f"no {cell} rows in {ENDS.format(policy)}")
    df["az"] = to_az(df["pan"])
    return df


def n_scored(lab, policy, cell, inst):
    cond, seed = parse_policy(policy)
    m = (lab.condition == cond) & (lab.seed == seed) & (lab.eval_cell == cell) & (lab.instance == inst)
    return int(m.sum())


def draw(name, rows, to_az, lab):
    data = {(p, c): load_grasps(p, c, to_az) for p, c, _ in rows}
    y = np.arange(len(rows))[::-1]           # first row in the list is drawn at the top

    fig, axes = plt.subplots(1, len(POS), sharey=True,
                             figsize=(PANEL_W * len(POS), ROW_H * len(rows) + 1.6))

    for ax, inst in zip(axes, POS):
        true = azimuth(*POS[inst])
        xs = [true]
        labels = []
        for yi, (pol, cell, _) in zip(y, rows):
            g = data[(pol, cell)]
            g = g[g.instance == inst]
            n_ep = n_scored(lab, pol, cell, inst)
            if len(g):
                fail, win = g[g.success != 1], g[g.success == 1]
                ax.scatter(fail.az, np.full(len(fail), yi), facecolors="none", edgecolors=GREY,
                           s=36, linewidths=1.3, zorder=2)
                ax.scatter(win.az, np.full(len(win), yi), color=GREEN, s=36, zorder=3)
                med = g.az.median()
                ax.scatter([med], [yi], marker="D", color=ORANGE, s=70, zorder=4)
                if abs(med - true) >= ANNOTATE_ABOVE:
                    ax.annotate(f"{med - true:+.1f}°", (med, yi), xytext=(0, 7),
                                textcoords="offset points", ha="center", fontsize=8, color=ORANGE)
                xs.extend(g.az.tolist())
            labels.append((yi, f"{len(g)}/{n_ep}"))

        lo, hi = min(xs), max(xs)
        pad = max(5.0, 0.08 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad + 0.18 * (hi - lo + 2 * pad))   # room on the right for n
        for yi, txt in labels:
            ax.annotate(txt, (1.0, yi), xycoords=("axes fraction", "data"), xytext=(-2, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=7, color=GREY)

        ax.axvline(true, ls="--", color=DARK, lw=1.3, zorder=1)
        if SHOW_TRAINED:
            x0, x1 = ax.get_xlim()
            for t, xy in TRAIN_POS.items():
                b = azimuth(*xy)
                if x0 < b < x1:
                    ax.axvline(b, color="#BBBBBB", lw=0.8, ls=":", zorder=0)
                    ax.text(b, 1.0, t, transform=ax.get_xaxis_transform(), ha="center",
                            va="bottom", fontsize=7, color="#999999")
        ax.set_title(f"{inst}, cube at {true:.0f}°", fontsize=10, pad=14 if SHOW_TRAINED else 6)
        ax.grid(axis="x", color="#E3E3E3")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[2] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.supxlabel("Grasp bearing (deg)")

    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=GREY, label="Episode, failed"),
        Line2D([], [], marker="o", ls="", color=GREEN, label="Episode, succeeded"),
        Line2D([], [], marker="D", ls="", color=ORANGE, label="Median"),
        Line2D([], [], ls="--", color=DARK, label="Cube bearing"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.995, 0.005, "n = episodes with a detected grasp / episodes scored",
             ha="right", va="bottom", fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

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
        draw(name, rows, to_az, lab)


if __name__ == "__main__":
    main()