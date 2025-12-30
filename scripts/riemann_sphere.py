import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xml.etree.ElementTree as ET
import subprocess
from pathlib import Path

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False}
)

OUT_SVG = Path("build/svg/riemann_sphere.svg")
FIGSIZE = (10, 10)

SPHERE_RES_U = 100
SPHERE_RES_V = 100
NEIGHBOR_RES = 100
PLANE_RANGE = 3
ARROW_COUNTS = 7

AXIS_LENGTH = PLANE_RANGE
AXIS_RADIUS = 0.04
AXIS_HEIGHT = 0.13

CONE_RES = 20


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def stereographic_proj(x, y, z):
    denom = 1 - z
    return x / denom, y / denom


def make_sphere(res_u=SPHERE_RES_U, res_v=SPHERE_RES_V):
    u = np.linspace(0, 2 * np.pi, res_u)
    v = np.linspace(0, np.pi, res_v)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def make_neighborhood(phi_center=np.pi / 3, theta_center=np.pi / 3, r=0.2, n=NEIGHBOR_RES):
    t = np.linspace(0, 2 * np.pi, n)
    phi = phi_center + r * np.cos(t)
    theta = theta_center + r * np.sin(t)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def draw_cone(ax, base_center, direction, height=0.1, radius=0.03, resolution=CONE_RES, color="black"):
    base_center = np.asarray(base_center, dtype=float)
    d = unit(direction)
    up = np.array([0.0, 0.0, 1.0])
    if np.allclose(np.abs(np.dot(d, up)), 1.0):
        up = np.array([1.0, 0.0, 0.0])
    ortho1 = unit(np.cross(d, up))
    ortho2 = np.cross(d, ortho1)

    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    circle = [base_center + radius *
              (np.cos(a) * ortho1 + np.sin(a) * ortho2) for a in angles]
    tip = base_center + d * height
    faces = [[circle[i], circle[(i + 1) % resolution], tip]
             for i in range(resolution)]
    ax.add_collection3d(Poly3DCollection(faces, color=color, edgecolor="none"))


def add_axes_arrows(ax, length=AXIS_LENGTH, radius=AXIS_RADIUS, height=AXIS_HEIGHT):
    axes = {
        r"$x_1$": (np.array([1, 0, 0]), np.array([length, 0, 0])),
        r"$-x_1$": (np.array([-1, 0, 0]), np.array([-length, 0, 0])),
        r"$x_2$": (np.array([0, 1, 0]), np.array([0, length, 0])),
        r"$-x_2$": (np.array([0, -1, 0]), np.array([0, -length, 0])),
        r"$x_3$": (np.array([0, 0, 1]), np.array([0, 0, length])),
        r"$-x_3$": (np.array([0, 0, -1]), np.array([0, 0, -length])),
    }
    ax.plot([-PLANE_RANGE, PLANE_RANGE], [0, 0], [0, 0],
            color='black', linewidth=0.9, zorder=20)
    ax.plot([0, 0], [-PLANE_RANGE, PLANE_RANGE], [0, 0],
            color='black', linewidth=0.9, zorder=20)
    ax.plot([0, 0], [0, 0], [-PLANE_RANGE, PLANE_RANGE],
            color='black', linewidth=0.9, zorder=20)
    for label, (vec, tip) in axes.items():
        base = tip - height * vec
        draw_cone(ax, base, vec, height=height, radius=radius)
        if label in (r"$x_1$", r"$x_2$", r"$x_3$"):
            pos = tip + 0.1 * vec
            ax.text(*pos, label, fontsize=14, ha="center", va="center")


def plot_riemann_sphere(
    out_svg=OUT_SVG,
    remove_patches=True,
):
    x_s, y_s, z_s = make_sphere()
    x_nbhd, y_nbhd, z_nbhd = make_neighborhood()
    x_proj, y_proj = stereographic_proj(x_nbhd, y_nbhd, z_nbhd)
    z_proj = np.zeros_like(x_proj)

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    top_mask = z_s >= 0
    bottom_mask = z_s < 0
    ax.plot_surface(np.where(top_mask, x_s, np.nan), np.where(top_mask, y_s, np.nan), np.where(top_mask, z_s, np.nan),
                    color="black", alpha=0.12, edgecolor="none")
    ax.plot_surface(np.where(bottom_mask, x_s, np.nan), np.where(bottom_mask, y_s, np.nan), np.where(bottom_mask, z_s, np.nan),
                    color="black", alpha=0.03, edgecolor="none")

    u_eq = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(u_eq), np.sin(u_eq), np.zeros_like(u_eq),
            color="black", linestyle="dotted", linewidth=1.3)

    xx, yy = np.meshgrid(np.linspace(-PLANE_RANGE, PLANE_RANGE, PLANE_RANGE * 8 + 1),
                         np.linspace(-PLANE_RANGE, PLANE_RANGE, PLANE_RANGE * 8 + 1))
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="white",
                    edgecolor="black", alpha=0, linewidth=0.1)

    ax.plot(x_nbhd, y_nbhd, z_nbhd, color="gray",
            linewidth=0.6, linestyle="dashed")
    ax.plot(x_proj, y_proj, z_proj, color="black",
            linewidth=0.7, linestyle="dashed")

    idx = np.linspace(0, len(x_nbhd) - 2, ARROW_COUNTS, dtype=int)
    for i in idx:
        d_nb = -np.array([x_nbhd[i + 1] - x_nbhd[i],
                         y_nbhd[i + 1] - y_nbhd[i], z_nbhd[i + 1] - z_nbhd[i]])
        draw_cone(ax, np.array(
            [x_nbhd[i], y_nbhd[i], z_nbhd[i]]), d_nb, height=0.07, radius=0.015)
        d_pr = -np.array([x_proj[i + 1] - x_proj[i],
                         y_proj[i + 1] - y_proj[i], z_proj[i + 1] - z_proj[i]])
        draw_cone(ax, np.array(
            [x_proj[i], y_proj[i], z_proj[i]]), d_pr, height=0.15, radius=0.025)

    meridian_longs = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    theta_samples = np.linspace(0, np.pi, 100)
    for theta in meridian_longs:
        x_mer = np.sin(theta_samples) * np.cos(theta)
        y_mer = np.sin(theta_samples) * np.sin(theta)
        z_mer = np.cos(theta_samples)
        ax.plot(x_mer, y_mer, z_mer, color="black",
                linestyle="dotted", linewidth=0.8)

    for phi in (np.pi / 6, np.pi / 3, 2 * np.pi / 3):
        theta_lat = np.linspace(0, 2 * np.pi, 200)
        x_lat = np.sin(phi) * np.cos(theta_lat)
        y_lat = np.sin(phi) * np.sin(theta_lat)
        z_lat = np.full_like(x_lat, np.cos(phi))
        ax.plot(x_lat, y_lat, z_lat, color="black",
                linestyle="dotted", linewidth=0.8)

    ax.scatter(0, 0, 1, color="black", s=7)
    add_axes_arrows(ax)

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=40, azim=66)
    ax.set_axis_off()
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    fig.patch.set_facecolor("none")

    plt.savefig(out_svg, dpi=600, pad_inches=0, transparent=True)

    if remove_patches:
        _inkscape_export_area(out_svg)


def remove_patch_groups(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    def recurse(elem):
        for child in list(elem):
            if child.tag == "{http://www.w3.org/2000/svg}g" and child.get("id", "").startswith("patch_"):
                elem.remove(child)
            else:
                recurse(child)

    recurse(root)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def _inkscape_export_area(svg_path: Path):
    remove_patch_groups(svg_path)
    cmd = ["inkscape", str(svg_path), "--export-area-drawing",
           f"--export-filename={svg_path}"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    out_svg = Path(OUT_SVG)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    plot_riemann_sphere(out_svg, remove_patches=True)
