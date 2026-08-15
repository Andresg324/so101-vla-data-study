#!/usr/bin/env python3
"""
analysis/make_figures.py

Builds Table 1 and Figures 1 to 3 from the derived CSVs. Nothing here recomputes a
result: every number is read from analysis/out_*/ so the figures and the text cannot
drift apart. Run the analysis pipeline first (see SETUP.md).

    Table 1  registered grid, both seeds, with the two matched comparisons beneath

    Dissociation:
    Fig 1    Aim invariance
    Fig 2    Aiming error 
    Fig 3    Probe decoding

    Mechanism and Inheritance:
    Fig 4    Trajectory envelope 
    Fig 5    Release point

Outputs PDF's and PNG's in figures/.

RUN: python analysis/make_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "tools")
from calibrate_pose import BASE_X, T          # the ten training positions

OUT = "figures"

# Validated categorical slots (dataviz reference palette, light mode). At most two hues
# are ever on screen together; NEUTRAL is context, not a category.
BLUE, ORANGE = "#2a78d6", "#eb6834"
NEUTRAL = "#8a8985"
INK, MUTED = "#0b0b0b", "#52514e"

T6_AZ = np.degrees(np.arctan2(15.5 - BASE_X, 10.0))                              # +24.23
CUP_AZ = np.degrees(np.arctan2(5.0 - BASE_X, 12.5))                              # -25.60
MID_AZ = np.degrees(np.arctan2((15.5 + 5.0) / 2 - BASE_X, (10.0 + 12.5) / 2))    #  -3.81

SAME_TARGET = ["in_distribution", "reduced_lighting", "different_object", "distractors"]
CELLS = SAME_TARGET + ["new_positions"]
CONDITIONS = ["clean", "color", "recovery", "randomized"]
# Invariant policies first, so the eye lands on the outliers last.
POLICIES = ["clean", "clean-seed2000", "color", "color-seed2000",
            "recovery", "recovery-seed2000", "randomized", "randomized-seed2000"]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "grid.color": "#e3e2df",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
})


def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}. Run the analysis pipeline first (SETUP.md).")
    return path


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf and .png")


def pretty(name):
    return name.replace("-seed2000", " (s2000)").replace("_", " ")


# ----------------------------------------------------------------------------
# Table 1
# ----------------------------------------------------------------------------

def table1():
    print("Table 1")
    grids, matched = {}, {}
    for seed in (1000, 2000):
        d = pd.read_csv(need(f"analysis/out_seed{seed}/success_by_cell.csv"))
        grids[seed] = d.pivot(index="condition", columns="eval_cell", values="successes")
        matched[seed] = pd.read_csv(need(f"analysis/out_seed{seed}/matched_comparisons.csv"))

    head = ["condition"] + [c.replace("_", " ") for c in CELLS] + ["total"]
    md, tex = [], []

    for seed in (1000, 2000):
        g = grids[seed].reindex(index=CONDITIONS, columns=CELLS)
        md.append(f"\n**Seed {seed}** (successes out of 15)\n")
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "---|" * len(head))
        tex.append(rf"\multicolumn{{{len(head)}}}{{l}}{{\textit{{Seed {seed}}}}} \\")
        for cond in CONDITIONS:
            vals = [int(g.loc[cond, c]) for c in CELLS]
            md.append(f"| {cond} | " + " | ".join(str(v) for v in vals)
                      + f" | {sum(vals)}/75 |")
            tex.append(f"{cond} & " + " & ".join(str(v) for v in vals)
                       + rf" & {sum(vals)}/75 \\")
        tex.append(r"\midrule")

    md.append("\n**Matched-axis comparisons** (pre-registered)\n")
    md.append("| seed | cell | matched | clean | difference | 95% CI | Fisher p |")
    md.append("|---|---|---|---|---|---|---|")
    for seed in (1000, 2000):
        for _, r in matched[seed].iterrows():
            ci = f"{r.diff_ci_low:+.3f} to {r.diff_ci_high:+.3f}"
            md.append(f"| {seed} | {r.eval_cell.replace('_', ' ')} "
                      f"({r.matched_condition}) | **{r.matched}** | {r.baseline_counts} "
                      f"| **{r.difference:+.3f}** | {ci} | {r.fisher_p:.3f} |")
            tex.append(rf"\textbf{{{r.matched_condition} vs clean, {r.eval_cell}}} (s{seed}) "
                       rf"& \multicolumn{{5}}{{l}}{{{r.matched} vs {r.baseline_counts}, "
                       rf"$\Delta$ {r.difference:+.3f} "
                       rf"[{r.diff_ci_low:+.3f}, {r.diff_ci_high:+.3f}], "
                       rf"$p$ = {r.fisher_p:.3f}}} \\")

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "table1.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(OUT, "table1.tex"), "w").write("\n".join(tex) + "\n")
    print(f"  wrote {OUT}/table1.md and table1.tex")


# ----------------------------------------------------------------------------
# Aim invariance
# ----------------------------------------------------------------------------

def fig_aim_invariance():
    print("Aim invariance")
    aim = pd.read_csv(need("analysis/out_azimuth/aim_by_cell.csv"), index_col=0)
    order = [p for p in POLICIES if p in aim.index]
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.axvline(T6_AZ, color=MUTED, lw=1.0, ls="--", zorder=1)

    for cell in [c for c in SAME_TARGET if c in aim.columns]:
        ax.scatter(aim[cell].reindex(order), y, s=30, facecolors="none",
                   edgecolors=NEUTRAL, linewidths=1.0, zorder=2)
    if "new_positions" in aim.columns:
        ax.scatter(aim["new_positions"].reindex(order), y, s=34, marker="D",
                   color=ORANGE, zorder=3)
        for p, yy in zip(order, y):
            v = float(aim.loc[p, "new_positions"])
            if abs(v - T6_AZ) > 5:                      # label only what actually moved
                ax.annotate(f"{v:.1f}°", (v, yy), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=7, color=ORANGE)

    ax.set_yticks(y, [pretty(p) for p in order])
    ax.set_xlabel("median commanded bearing (deg)")
    ax.set_xlim(-6, 34)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.text(T6_AZ + 0.7, len(order) - 0.55, "cube at T6",
            fontsize=7, color=MUTED, va="top")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=NEUTRAL, markersize=5.5, label="cube at T6 (4 cells)"),
        Line2D([], [], marker="D", ls="", color=ORANGE, markersize=5,
               label="new positions"),
    ], loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    ax.set_title("Only Randomized's aim moves when the cube does", loc="center", pad=8)
    save(fig, "fig_aim_invariance")


# ----------------------------------------------------------------------------
# Aiming error, commanded vs required
# ----------------------------------------------------------------------------

def fig_aiming_error():
    print("Aiming error")
    aiming = pd.read_csv(need("analysis/out_azimuth/randomized_aiming.csv"))
    centroid = float(np.mean([np.degrees(np.arctan2(x - BASE_X, yy))
                              for x, yy in T.values()]))
    lim = (-82, 62)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot(lim, lim, ls="--", lw=1.0, color=MUTED, zorder=1)
    ax.axhline(centroid, ls=":", lw=1.0, color=NEUTRAL, zorder=1)
    ax.text(lim[0] + 3, centroid + 4, f"mean training bearing ({centroid:.0f}°)",
            fontsize=6.5, color=NEUTRAL)
    ax.text(lim[1] - 3, lim[1] - 10, "no error", fontsize=6.5, color=MUTED,
            ha="right", rotation=45, rotation_mode="anchor")

    for split, colour in (("interpolation", BLUE), ("extrapolation", ORANGE)):
        s = aiming[aiming.split == split]
        ax.scatter(s.true_az, s.az, s=30, color=colour, label=f"{split} (n={len(s)})",
                   edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("bearing required (deg)")
    ax.set_ylabel("bearing commanded (deg)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Randomized regresses to the centre\noutside its training hull",
                 loc="center", pad=8)
    save(fig, "fig_aiming_error")


# ----------------------------------------------------------------------------
# Probe decoding
# ----------------------------------------------------------------------------

def fig_probe():
    print("Probe")
    probe = pd.read_csv(need("analysis/out_probe/position_probe.csv"))
    pr = probe.set_index("policy").reindex([p for p in POLICIES if p in set(probe.policy)])
    x = np.arange(len(pr))
    w = 0.38

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    ax.bar(x - w / 2, pr.probe_by_episode_deg, w, color=BLUE, label="held out by episode")
    ax.bar(x + w / 2, pr.probe_by_position_deg, w, color=ORANGE, label="held out by position")

    chance = float(pr.chance_med_deg.mean())
    ax.axhline(chance, ls="--", lw=1.0, color=MUTED, zorder=4)
    ax.text(len(pr) - 0.45, chance + 1.0, f"chance ({chance:.0f}°)",
            ha="right", fontsize=7, color=MUTED)

    ax.set_xticks(x, [pretty(p).replace(" (s2000)", "\n(s2000)") for p in pr.index])
    ax.set_ylabel("median bearing error (deg)")
    ax.set_ylim(0, 56)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Every policy encodes position", loc="center", pad=8)
    save(fig, "fig_probe")


# ----------------------------------------------------------------------------
# Trajectory envelope
# ----------------------------------------------------------------------------

def fig_envelope():
    print("Envelope")
    env = pd.read_csv(need("analysis/out_azimuth/clean_envelope.csv"))
    cells = ["in_distribution", "near_1in", "near_2in"]
    xs = np.array([0.0, 1.0, 2.0])
    req = [float(env[env.cell == c].required_az.iloc[0]) for c in cells]

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.plot(xs, req, ls="--", marker="s", ms=5.5, color=INK, label="bearing required",
            zorder=4)

    # The two seeds sit within 0.3 deg of each other, so dodge them apart on the
    # categorical x axis rather than letting the markers and labels overlap.
    for pol, colour, dx, dy in (("clean", BLUE, -0.055, 13),
                                ("clean-seed2000", ORANGE, 0.055, -17)):
        e = env[env.policy == pol].set_index("cell").reindex(cells)
        yerr = [e.median_max_az - e.q1, e.q3 - e.median_max_az]
        ax.errorbar(xs + dx, e.median_max_az, yerr=yerr, marker="o", ms=5.5,
                    capsize=3, color=colour, zorder=3,
                    label=f"furthest reached, {pretty(pol)}")
        for xx, cc in zip(xs + dx, cells):
            ax.annotate(f"{int(e.loc[cc, 'episodes_reaching_target'])}"
                        f"/{int(e.loc[cc, 'n'])}",
                        (xx, e.loc[cc, "median_max_az"]), xytext=(0, dy),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=colour, zorder=5)

    ax.set_xticks(xs, ["0", "1", "2"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(23.4, 30.6)
    ax.set_xlabel("cube displacement from T6 (inches)")
    ax.set_ylabel("bearing (deg)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("The arm never turns further\n(labels: episodes reaching the cube)",
                 loc="center", pad=8)
    save(fig, "fig_envelope")


# ----------------------------------------------------------------------------
# Release point
# ----------------------------------------------------------------------------

def fig_release():
    print("Release point")
    demo = pd.read_csv(need("analysis/out_drops/demo_drops.csv"))
    loc = pd.read_csv(need("analysis/out_drops/drop_locations.csv"))
    roll = loc[loc.condition == "recovery"]

    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    rng = np.random.default_rng(0)          # jitter only; no inference depends on it
    labels = []
    series = [(roll.az.to_numpy(), f"Recovery policy\n(n={len(roll)})", ORANGE),
              (demo.az.to_numpy(), f"demonstrations\n(n={len(demo)})", BLUE)]

    for i, (vals, label, colour) in enumerate(series):
        yy = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, yy, s=14, color=colour, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)
        m = float(np.median(vals))
        ax.plot([m, m], [i - 0.27, i + 0.27], color=colour, lw=2.2, zorder=4)
        ax.annotate(f"{m:.1f}°", (m, i + 0.33), ha="center", fontsize=7.5, color=colour)
        labels.append(label)

    for v, lab in ((CUP_AZ, "cup"), (MID_AZ, "carry midpoint"), (T6_AZ, "cube")):
        ax.axvline(v, color=NEUTRAL, lw=1.0, ls=":", zorder=1)
        ax.text(v, 1.74, lab, ha="center", va="center", fontsize=7, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.6), zorder=5)

    ax.set_yticks([0, 1], labels)
    ax.set_ylim(-0.6, 1.95)
    ax.set_xlim(-40, 30)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("release bearing (deg)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("The policy releases where the demonstrator released", loc="center", pad=8)
    save(fig, "fig_release")


if __name__ == "__main__":
    table1()
    fig_aim_invariance()
    fig_aiming_error()
    fig_probe()
    fig_envelope()
    fig_release()
    print(f"\nall outputs in {OUT}/")