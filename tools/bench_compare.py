#!/usr/bin/env python3
"""
bench_compare.py

Compares the rebuilt bench against the August reference on two axes:
geometry (has the camera moved) and luminosity (how much light reaches the
surface, as the camera records it).

Both frames come from tools/check_cameras.py, so the capture path, resolution
and warmup are identical and the comparison is not confounded by how the image
was taken.

Geometry: click the same three features in each frame; pixel offsets convert to
inches through the known 22 in frame width. Cup and cube sit at fixed marked
positions, so any offset is the camera, not the scene. A common shift across
all three means the camera translated. A change in the cup-to-cube distance
means height or angle changed, which rescales what the policy sees.

Luminosity: mean grayscale over the whole frame and over three fixed patches of
bare work surface. The scale to read these against is the paper's reduced
lighting cell, a fall from 144.8 to 101.0.

Workflow: adjust the gantry, run check_cameras.py, run this. Each new capture
is archived with a timestamp so the sequence of adjustments is kept.

Usage:
    python bench_compare.py                    # defaults below
    python bench_compare.py ref.jpg new.jpg    # explicit paths
    python bench_compare.py --no-archive
    python bench_compare.py --lum-only         # skip the clicking
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REFERENCE = "media/overhead_baseline_lighting.jpg"
CURRENT = "tools/preview_overhead.jpg"
ARCHIVE = Path("media/bench_rebuild")

FRAME_WIDTH_IN = 22.0
FEATURES = ["T1 mark", "T3 mark", "T8 mark"]

# Bare-surface patches as fractions of the frame (x0, y0, x1, y1).
PATCHES = {
    "upper right": (0.70, 0.10, 0.90, 0.25),
    "mid left":    (0.08, 0.55, 0.25, 0.70),
    "lower right": (0.72, 0.70, 0.92, 0.85),
}


def archive(path):
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE / f"overhead_{stamp}.jpg"
    shutil.copy2(path, dest)
    print(f"archived: {dest}")
    return dest


def luminosity(img, tag):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    print(f"\n{tag}")
    print(f"  whole frame  mean {g.mean():6.1f}   median {np.median(g):6.1f}"
          f"   p5 {np.percentile(g,5):5.1f}   p95 {np.percentile(g,95):5.1f}")
    out = {"whole frame": g.mean()}
    for name, (x0, y0, x1, y1) in PATCHES.items():
        patch = g[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
        out[name] = patch.mean()
        print(f"  {name:<13} mean {patch.mean():6.1f}   sd {patch.std():5.1f}")
    return out


def pick(img, window):
    pts, disp = [], img.copy()

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(FEATURES):
            pts.append((x, y))
            cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(disp, str(len(pts)), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(window, disp)

    cv2.namedWindow(window)
    cv2.imshow(window, disp)
    cv2.setMouseCallback(window, on_click)
    for i, f in enumerate(FEATURES):
        print(f"  [{i+1}/{len(FEATURES)}] click: {f}")
        while len(pts) <= i:
            if cv2.waitKey(20) == 27:
                cv2.destroyAllWindows()
                sys.exit("aborted")
    cv2.waitKey(400)
    cv2.destroyWindow(window)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", nargs="?", default=REFERENCE)
    ap.add_argument("current", nargs="?", default=CURRENT)
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--lum-only", action="store_true",
                    help="luminosity only, skip the geometry clicking")
    args = ap.parse_args()

    a, b = cv2.imread(args.reference), cv2.imread(args.current)
    for p, im in ((args.reference, a), (args.current, b)):
        if im is None:
            sys.exit(f"could not read {p}")
        print(f"{p}: {im.shape[1]}x{im.shape[0]}")
    if a.shape != b.shape:
        sys.exit("frames differ in size; recapture at 640x480")

    if not args.no_archive:
        archive(args.current)

    la = luminosity(a, f"LUMINOSITY  reference  {args.reference}")
    lb = luminosity(b, f"LUMINOSITY  current    {args.current}")
    print("\n  change (current minus reference):")
    for k in la:
        d = lb[k] - la[k]
        print(f"    {k:<13} {la[k]:6.1f} -> {lb[k]:6.1f}   "
              f"{d:+6.1f}  ({100*d/la[k]:+5.1f}%)")
    frame_pct = 100 * (lb["whole frame"] - la["whole frame"]) / la["whole frame"]
    print(f"\n  reference point: the reduced-lighting cell was 144.8 -> 101.0, "
          f"a 30% drop.\n  this rebuild is {frame_pct:+.1f}% on frame mean.")

    if args.lum_only:
        return

    print(f"\nGEOMETRY: click the three features in the REFERENCE frame")
    pa = pick(a, "reference")
    print(f"GEOMETRY: click the same three in the CURRENT frame")
    pb = pick(b, "current")

    ppi = a.shape[1] / FRAME_WIDTH_IN
    print(f"\n  scale: {ppi:.1f} px/inch")
    print(f"  {'feature':<22}{'dx px':>8}{'dy px':>8}{'dx in':>8}{'dy in':>8}")
    dxs, dys = [], []
    for f, (x0, y0), (x1, y1) in zip(FEATURES, pa, pb):
        dx, dy = x1 - x0, y1 - y0
        dxs.append(dx); dys.append(dy)
        print(f"  {f:<22}{dx:>8}{dy:>8}{dx/ppi:>8.2f}{dy/ppi:>8.2f}")

    print(f"\n  mean shift:  {np.mean(dxs)/ppi:+.2f} in x, "
          f"{np.mean(dys)/ppi:+.2f} in y")
    print(f"  disagreement between features (sd): {np.std(dxs)/ppi:.2f} in x, "
          f"{np.std(dys)/ppi:.2f} in y")

    da = np.hypot(pa[0][0]-pa[1][0], pa[0][1]-pa[1][1])
    db = np.hypot(pb[0][0]-pb[1][0], pb[0][1]-pb[1][1])
    print(f"  cup-to-cube distance: {da:.1f} -> {db:.1f} px "
          f"({100*(db-da)/da:+.1f}%)")
    print("\n  A common shift across features is translation, correctable by "
          "moving the gantry.\n  A change in cup-to-cube distance is height or "
          "angle, which rescales the scene.\n  Feature disagreement above about "
          "0.2 in is your clicking precision, not the bench.")


if __name__ == "__main__":
    main()