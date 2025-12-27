#!/usr/bin/env python3
import os
import json
import shutil
import time
from typing import List
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

def read_input_json(filename: Path):
    with open(filename, "r") as f:
        data = json.load(f)

    global_offset = float(data.get("global_offset", 0.4))
    input_disk_radius_scalar = float(data.get("input_disk_radius_scalar", 1.0))
    subcover_disk_radius_scalar = float(data.get("subcover_disk_radius_scalar", 1.0))

    samples_per_segment = int(data.get("samples_per_segment", 20))

    pre_erosion_curves = data.get("pre_erosion_curves", [])
    support_curves = data.get("support_curves", [])
    disk_points = data.get("disk_points", [])

    for c in pre_erosion_curves:
        c["points"] = np.array(c["points"], dtype=float)
    for c in support_curves:
        c["points"] = np.array(c["points"], dtype=float)
    disk_points = [np.array(p, dtype=float) for p in disk_points]

    return {
        "global_offset": global_offset,
        "input_disk_radius_scalar": input_disk_radius_scalar,
        "subcover_disk_radius_scalar": subcover_disk_radius_scalar,
        "samples_per_segment": samples_per_segment,
        "pre_erosion_curves": pre_erosion_curves,
        "support_curves": support_curves,
        "disk_points": disk_points,
    }

def tikz_bezier_sample(points, tension: float = 0.55, samples_per_seg: int = 20):
    pts = np.array(points)
    n = len(pts)
    out = []
    if n == 0:
        return np.array(out)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        s1 = tension * (p2 - p0)
        s2 = tension * (p3 - p1)
        c1 = p1 + s1 / 3.0
        c2 = p2 - s2 / 3.0
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            B = (1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p2
            out.append(B)
    return np.array(out)

def number_to_letters(n: int) -> str:
    result = ""
    n = int(n)
    while n > 0:
        n -= 1
        n, rem = divmod(n, 26)
        result = chr(65 + rem) + result
    return result if result else "A"

def compute_polygon_region(curves, samples_per_segment: int):
    outer_polys = []
    hole_polys = []
    for c in curves:
        pts = c["points"]
        samp = tikz_bezier_sample(pts, samples_per_seg=samples_per_segment)
        poly = Polygon(samp)
        typ = c.get("type", "outer")
        if typ == "outer":
            poly = orient(poly, sign=1.0)
            outer_polys.append(poly)
        elif typ == "hole":
            poly = orient(poly, sign=-1.0)
            hole_polys.append(poly)
    if not outer_polys:
        return None
    outer = unary_union(outer_polys)
    holes = unary_union(hole_polys) if hole_polys else None
    return outer.difference(holes) if holes else outer

def compute_input_region(curves, samples_per_segment: int):
    return compute_polygon_region(curves, samples_per_segment)

def compute_eroded_region(curves, global_offset: float, samples_per_segment: int):
    region = compute_polygon_region(curves, samples_per_segment)
    if region is None:
        return None
    return region.buffer(-global_offset, resolution=80)

def compute_support_region(curves, samples_per_segment: int):
    return compute_polygon_region(curves, samples_per_segment)

def coords_to_tikz_path(coords: np.ndarray) -> str:
    return " ".join(f"({x:.6f},{y:.6f})" for x, y in coords)

def write_region_boundaries(folder: str, geom_list, prefix: str, list_macro: str):
    os.makedirs(folder, exist_ok=True)
    print(f"Writing region boundaries to {folder}")

    if all(geom is None or geom.is_empty for geom in geom_list):
        path = os.path.join(folder, "all.tex")
        with open(path, "w") as f:
            f.write(f"\\def\\{list_macro}{{}}\n")
        print(f"  Empty geometry → empty list in {path}")
        return

    rings = []
    polys = [
        g for geom in geom_list for g in (geom.geoms if isinstance(geom, (MultiPolygon, GeometryCollection)) else [geom])
    ]
    for poly in polys:
        if not isinstance(poly, Polygon):
            continue
        rings.append(poly.exterior)
        rings.extend(poly.interiors)

    macros = []
    for i, ring in enumerate(rings, 1):
        letter = number_to_letters(i)
        macro = f"{prefix}{letter}"

        coords = np.array(ring.coords)
        path_str = coords_to_tikz_path(coords)

        file_path = os.path.join(folder, f"{i}.tex")
        with open(file_path, "w") as f:
            f.write(f"\\def\\{macro}{{{path_str}}}\n")
        print(f"  Written curve {i} to {file_path}")
        macros.append(f"\\{macro}")

    all_path = os.path.join(folder, "all.tex")
    with open(all_path, "w") as f:
        list_content = ",".join(macros)
        f.write(f"\\def\\{list_macro}{{{list_content}}}\n")
    print(f"  Written list macro \\{list_macro} with {len(macros)} curves to {all_path}")

def write_input_disks(folder: str, disk_points: List[np.ndarray], disk_r: float):
    os.makedirs(folder, exist_ok=True)
    print(f"Writing input disks to {folder}")

    macros = []
    for i, center in enumerate(disk_points, 1):
        letter = number_to_letters(i)
        macro = f"InputDisk{letter}"
        macros.append(f"\\{macro}")

        angles = np.linspace(0, 2 * np.pi, 100)
        circle_x = center[0] + disk_r * np.cos(angles)
        circle_y = center[1] + disk_r * np.sin(angles)
        coords = np.column_stack((circle_x, circle_y))
        path_str = coords_to_tikz_path(coords) + f" ({circle_x[0]:.6f},{circle_y[0]:.6f})"

        file_path = os.path.join(folder, f"{i}.tex")
        with open(file_path, "w") as f:
            f.write(f"\\def\\{macro}{{{path_str}}}\n")
        print(f"  Written disk boundary {i} to {file_path}")

    all_path = os.path.join(folder, "all.tex")
    with open(all_path, "w") as f:
        f.write(f"\\def\\InputDiskList{{{','.join(macros)}}}\n")
    print(f"  Written \\InputDiskList to {all_path}")

    centers_path = os.path.join(folder, "centers.tex")
    items = ",".join(f"{p[0]:.6f}/{p[1]:.6f}" for p in disk_points)
    with open(centers_path, "w") as f:
        f.write(f"\\def\\InputDiskCenters{{{items}}}\n")
    print(f"  Written {len(disk_points)} centers to {centers_path}")

def write_subcover_centers(folder: str, sub_centers: List[np.ndarray]):
    os.makedirs(folder, exist_ok=True)
    print(f"Writing subcover centers to {folder}")

    centers_path = os.path.join(folder, "centers.tex")
    if sub_centers:
        items = ",".join(f"{p[0]:.6f}/{p[1]:.6f}" for p in sub_centers)
        with open(centers_path, "w") as f:
            f.write(f"\\def\\SubcoverDiskCenters{{{items}}}\n")
        print(f"  Written {len(sub_centers)} subcover centers to {centers_path}")
    else:
        with open(centers_path, "w") as f:
            f.write(f"\\def\\SubcoverDiskCenters{{}}\n")
        print(f"  Written empty subcover centers to {centers_path}")

def write_wkt(folder: str, name: str, geom):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.wkt")
    content = "EMPTY\n" if geom is None or geom.is_empty else geom.wkt + "\n"
    with open(path, "w") as f:
        f.write(content)
    print(f"Written WKT for {name} to {path} (empty: {geom is None or geom.is_empty})")

def write_metadata_tex_all(root: str):
    os.makedirs(root, exist_ok=True)
    imports_path = os.path.join(root, "imports.tex")
    tex_files = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(".tex") and f.lower() != "imports.tex":
                rel = os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/")
                tex_files.append(rel)
    lines = ["% AUTO-GENERATED", "% DO NOT EDIT", ""]
    for rel in tex_files:
        d, f = os.path.split(rel)
        lines.append(f"\\subimport{{{d}/}}{{{f}}}" if d else f"\\subimport{{}}{{{f}}}")
    if os.path.exists(imports_path):
        with open(imports_path) as f:
            existing = f.read().strip()
        if existing == "\n".join(lines).strip():
            print(f"No changes to {imports_path}")
            return
    with open(imports_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written metadata to {imports_path}")

def prune_redundant_disks(centers: List[np.ndarray], r_sub: float, covered_region, nocenters_region) -> List[np.ndarray]:
    if not centers:
        return []

    points = [Point(p) for p in centers]
    disks = [p.buffer(r_sub, resolution=50) for p in points]
    useful_areas = [disk.intersection(covered_region) for disk in disks]
    useful_area_values = [ua.area for ua in useful_areas]

    sorted_idx = sorted(range(len(centers)), key=lambda i: useful_area_values[i], reverse=True)

    kept_idx = []
    current_union = None
    epsilon = 1e-8

    for i in sorted_idx:
        contrib = useful_areas[i]
        if contrib.is_empty or contrib.area < epsilon:
            continue
        if current_union is None:
            remaining = contrib
        else:
            remaining = contrib.difference(current_union)
        if not remaining.is_empty and remaining.area >= epsilon:
            kept_idx.append(i)
            if current_union is None:
                current_union = disks[i]
            else:
                current_union = current_union.union(disks[i])

    pruned = [centers[i] for i in kept_idx]
    print(f"Pruned subcover disks: {len(centers)} → {len(pruned)} (strict: zero-contribution only)")
    return pruned

def main():
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = Path.cwd()
    INPUT_JSON = SCRIPT_DIR / "curves.json"
    OUTPUT_ROOT = PROJECT_ROOT / "build" / "visual_output"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = read_input_json(INPUT_JSON)

    K = compute_input_region(cfg["pre_erosion_curves"], cfg["samples_per_segment"])
    Eroded = compute_eroded_region(cfg["pre_erosion_curves"], cfg["global_offset"], cfg["samples_per_segment"])
    Support = compute_support_region(cfg["support_curves"], cfg["samples_per_segment"])

    CoveredRegion = Support.difference(Eroded) if Support and Eroded else Support

    disk_r = cfg["input_disk_radius_scalar"] * cfg["global_offset"]
    D = unary_union([Point(p).buffer(disk_r, resolution=50) for p in cfg["disk_points"]]) if cfg["disk_points"] else None
    NoCentersRegion = unary_union([g for g in [K, D] if g])

    for name, geom in [("K_region", K), ("ErodedRegion", Eroded), ("support_region", Support),
                       ("CoveredRegion", CoveredRegion), ("D_region", D), ("NoCentersRegion", NoCentersRegion)]:
        write_wkt(str(OUTPUT_ROOT), name, geom)

    write_input_disks(str(OUTPUT_ROOT / "input_disks"), cfg["disk_points"], disk_r)

    write_region_boundaries(str(OUTPUT_ROOT / "pre_erosion_region"), [K], "PreErosionRegion", "PreErosionRegionList")
    write_region_boundaries(str(OUTPUT_ROOT / "eroded_region"), [Eroded], "ErodedRegion", "ErodedRegionList")
    write_region_boundaries(str(OUTPUT_ROOT / "support_region"), [Support], "SupportRegion", "SupportRegionList")
    write_region_boundaries(str(OUTPUT_ROOT / "covered_region"), [CoveredRegion], "CoveredRegion", "CoveredRegionList")
    write_region_boundaries(str(OUTPUT_ROOT / "no_centers_region"), [NoCentersRegion], "NoCentersRegion", "NoCentersRegionList")

    r_sub = cfg["global_offset"] * cfg["subcover_disk_radius_scalar"]
    l = max(0, (cfg["subcover_disk_radius_scalar"] - 1) / 8 * cfg["global_offset"])

    H_prime = CoveredRegion.buffer(l, resolution=80) if CoveredRegion else Polygon()
    H_doubleprime = H_prime.boundary
    N = NoCentersRegion.buffer(l, resolution=80) if NoCentersRegion else Polygon()
    N_prime = N.boundary
    curves = unary_union([H_doubleprime, N_prime]).difference(NoCentersRegion)

    linestrings = []
    if curves.geom_type == "LineString":
        linestrings.append(curves)
    elif curves.geom_type == "MultiLineString":
        linestrings.extend(curves.geoms)
    elif curves.geom_type == "GeometryCollection":
        for g in curves.geoms:
            if g.geom_type == "LineString":
                linestrings.append(g)

    sub_centers = []
    d = r_sub * np.sqrt(3)
    step = d * 0.8

    for ls in linestrings:
        length = ls.length
        if length == 0:
            continue
        distances = np.arange(0, length, step)
        for dist in distances:
            pt = ls.interpolate(dist)
            if not NoCentersRegion.contains(pt):
                sub_centers.append(np.array([pt.x, pt.y]))
        last_dist = distances[-1] if len(distances) > 0 else 0
        if length - last_dist > 1e-6:
            pt = ls.interpolate(length)
            if not NoCentersRegion.contains(pt):
                sub_centers.append(np.array([pt.x, pt.y]))

    allowed_interior = CoveredRegion.difference(NoCentersRegion)
    if not allowed_interior.is_empty:
        minx, miny, maxx, maxy = allowed_interior.bounds
        row_spacing = (np.sqrt(3) / 2) * d
        col_spacing = d
        row = 0
        y = miny - row_spacing
        while y <= maxy + row_spacing:
            offset = (col_spacing / 2) if row % 2 == 1 else 0
            x = minx - col_spacing + offset
            while x <= maxx + col_spacing:
                pt = Point(x, y)
                if allowed_interior.contains(pt):
                    sub_centers.append(np.array([x, y]))
                x += col_spacing
            y += row_spacing
            row += 1

    for _ in range(10):
        sub_disks = [Point(p).buffer(r_sub, resolution=50) for p in sub_centers]
        union_sub = unary_union(sub_disks) if sub_disks else Polygon()
        uncovered = CoveredRegion.difference(union_sub)
        if uncovered.is_empty:
            break
        geoms = uncovered.geoms if hasattr(uncovered, "geoms") else [uncovered]
        for g in geoms:
            if g.is_empty:
                continue
            cent = g.centroid
            if not NoCentersRegion.contains(cent):
                sub_centers.append(np.array([cent.x, cent.y]))
            else:
                possible = cent.buffer(r_sub).difference(NoCentersRegion)
                if not possible.is_empty:
                    sub_centers.append(np.array([possible.centroid.x, possible.centroid.y]))
    sub_centers = prune_redundant_disks(sub_centers, r_sub, CoveredRegion, NoCentersRegion)
    write_subcover_centers(str(OUTPUT_ROOT / "subcover"), sub_centers)
    final_union = unary_union([Point(p).buffer(r_sub, resolution=80) for p in sub_centers]) if sub_centers else Polygon()
    write_wkt(str(OUTPUT_ROOT), "SubcoverRegion", final_union)
    
    covered_disjoint_union_region = []
    current_union = None

    for center in sub_centers:
        disk = Point(center).buffer(r_sub, resolution=80)
        contrib = disk.intersection(CoveredRegion)

        if contrib.is_empty:
            covered_disjoint_union_region.append(Polygon())  # empty for consistency
            continue

        if current_union is None:
            remaining = contrib
        else:
            remaining = contrib.difference(current_union)

        covered_disjoint_union_region.append(remaining)

        if current_union is None:
            current_union = disk.intersection(CoveredRegion)
        else:
            current_union = current_union.union(contrib)
    write_region_boundaries(str(OUTPUT_ROOT / "covered_disjoint_union_region"), covered_disjoint_union_region, "CoveredDisjointUnionRegion", "CoveredDisjointUnionRegionList")
    write_metadata_tex_all(str(OUTPUT_ROOT))

if __name__ == "__main__":
    main()