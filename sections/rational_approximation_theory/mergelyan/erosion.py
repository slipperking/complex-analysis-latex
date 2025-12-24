# Disclaimer this file is AI generated :)

import os
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from scipy.interpolate import CubicSpline

###############################################
# 1. READ INPUT
###############################################

def read_curves_json(filename):
    with open(filename) as f:
        data = json.load(f)

    global_settings = data.get("global", {})
    curves = data["curves"]

    # Fill defaults
    for c in curves:
        c.setdefault("tension", global_settings.get("tension", 0.55))
        c.setdefault("samples_per_segment", global_settings.get("samples_per_segment", 20))
        c.setdefault("resample_points", global_settings.get("resample_points", 200))

        # Convert points to numpy array
        c["points"] = np.array(c["points"], dtype=float)

    return global_settings, curves

###############################################
# 2. BEZIER SAMPLING TO IMITATE PGFPLOTS
###############################################

def tikz_bezier_sample(points, tension=0.55, samples_per_seg=20):
    pts = np.array(points)
    n = len(pts)
    out = []

    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        s1 = tension * (p2 - p0)
        s2 = tension * (p3 - p1)

        c1 = p1 + s1 / 3
        c2 = p2 - s2 / 3

        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            B = (1 - t)**3 * p1 \
                + 3*(1 - t)**2 * t * c1 \
                + 3*(1 - t) * t**2 * c2 \
                + t**3 * p2
            out.append(B)

    return np.array(out)


###############################################
# 3. RESAMPLING
###############################################

def resample_ring(ring, n=200):
    coords = np.array(ring.coords)
    d = np.cumsum(np.r_[0, np.linalg.norm(np.diff(coords, axis=0), axis=1)])
    t = d / d[-1]

    csx = CubicSpline(t, coords[:,0], bc_type='periodic')
    csy = CubicSpline(t, coords[:,1], bc_type='periodic')

    ts = np.linspace(0, 1, n, endpoint=False)
    return np.vstack([csx(ts), csy(ts)]).T


###############################################
# 4. COMPUTATION
###############################################

def compute_offsets(curves):
    outer_polys = []
    hole_polys = []
    neutral_polys = []

    # Build polygons with TikZ sampling + offset
    for c in curves:
        samp = tikz_bezier_sample(c["points"], tension=c["tension"], samples_per_seg=c["samples_per_segment"])
        poly = Polygon(samp)

        # Orientation
        if c["type"] == "outer":
            poly = orient(poly, sign=1.0)
            poly = poly.buffer(-c["offset"], resolution=256)
            outer_polys.append(poly)

        elif c["type"] == "hole":
            poly = orient(poly, sign=-1.0)
            poly = poly.buffer(c["offset"], resolution=256)
            hole_polys.append(poly)

        else:  # neutral
            neutral_polys.append(poly)

    # Combine all outer boundaries
    if outer_polys:
        outer_union = unary_union(outer_polys)
    else:
        raise ValueError("No outer curves defined.")

    # Combine all holes
    if hole_polys:
        hole_union = unary_union(hole_polys)
    else:
        hole_union = None

    # Subtract holes from outer region
    if hole_union:
        final = outer_union.difference(hole_union)
    else:
        final = outer_union

    # Add neutral curves (rare)
    if neutral_polys:
        final = unary_union([final] + neutral_polys)

    # Extract all rings
    rings = []
    if isinstance(final, Polygon):
        rings.append(final.exterior)
        rings.extend(final.interiors)
    elif isinstance(final, MultiPolygon):
        for p in final.geoms:
            rings.append(p.exterior)
            rings.extend(p.interiors)

    return rings

###############################################
# 5. WRITE FILES
###############################################

def number_to_letters(n):
    """Convert a positive integer n to letters (A=1, ..., Z=26, AA=27, etc.)"""
    if n < 1:
        raise ValueError("n must be >= 1")
    result = ""
    while n > 0:
        n -= 1  # adjust because it's 1-indexed
        n, remainder = divmod(n, 26)
        result = chr(65 + remainder) + result  # 65 is ord('A')
    return result

def write_raw_outputs(rings, folder="erosion_output/eroded"):
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    for i, ring in enumerate(rings, start=1):
        pts = resample_ring(ring, 200)
        coords = "\\def\\ErosionOutputCoordinates" + number_to_letters(i) + "{" + " ".join(f"({x:.4f},{y:.4f})" for x, y in pts) + "}"

        filename = os.path.join(folder, f"{i}.tex") # save as tex so importing is simpler
        with open(filename, "w") as f:
            f.write(coords)

    print(f"Wrote {len(rings)} rings to folder '{folder}'")

def write_original_inputs(curves, folder="erosion_output/original"):
    os.makedirs(folder, exist_ok=True)

    for i, c in enumerate(curves, start=1):
        pts = c["points"]
        coords = "\\def\\ErosionInputCoordinates" + number_to_letters(i) + "{" + " ".join(f"({x:.4f},{y:.4f})" for x, y in pts) + "}"

        filename = os.path.join(folder, f"{i}.tex")
        with open(filename, "w") as f:
            f.write(coords)

    print(f"Wrote {len(curves)} original curves to folder '{folder}'")

###############################################
# 6. RUN
###############################################

if __name__ == "__main__":
    global_settings, curves = read_curves_json("curves.json")
    # write original inputs
    write_original_inputs(curves)
    rings = compute_offsets(curves)
    write_raw_outputs(rings)