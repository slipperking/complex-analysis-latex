import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
})

OUT_SVG = Path("build/svg/second_fundamental_form.svg")
FIGSIZE = (8, 6)
DPI = 600

U_RANGE = (-1, 1)
V_RANGE = (-1, 1)
SURF_RES = 21
PLANE_EXT = 1.1
PLANE_RES = 11


def make_paraboloid(u_range=U_RANGE, v_range=V_RANGE, res=SURF_RES):
    u = np.linspace(u_range[0], u_range[1], res)
    v = np.linspace(v_range[0], v_range[1], res)
    U, V = np.meshgrid(u, v)
    Z = U**2 + V**2
    return U, V, Z


def make_plane(extent=PLANE_EXT, res=PLANE_RES):
    uu = np.linspace(-extent, extent, res)
    vv = np.linspace(-extent, extent, res)
    UU, VV = np.meshgrid(uu, vv)
    return UU, VV, np.zeros_like(UU)


def plot_second_fundamental_form(out_svg: Path = OUT_SVG, remove_patches: bool = True):
    U, V, Z = make_paraboloid()
    Xp, Yp, Zp = make_plane()

    P = np.array([0.0, 0.0, 0.0])
    Q = np.array([0.5, 0.3, 0.5**2 + 0.3**2])

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(U, V, Z, facecolor=(0, 0, 0, 0.05),
                    edgecolor=(0, 0, 0, 0.4), linewidth=0.25)

    t = np.linspace(0, 1, 200)
    x_curve = t * Q[0]
    y_curve = t * Q[1]
    z_curve = x_curve**2 + y_curve**2
    ax.plot(x_curve, y_curve, z_curve, color="k", linewidth=0.5)

    ax.scatter(*P, color="k", s=3)
    ax.scatter(*Q, color="k", s=3)
    ax.text(*(P + np.array([-0.3, 0.0, 0.1])),
            r"$P=\vec{\mathbf{r}}(u,v)$", fontsize=12)
    ax.text(*(Q + np.array([-0.3, 0.0, 0.1])),
            r"$Q=\vec{\mathbf{r}}(u+\Delta u,v+\Delta v)$", fontsize=12)

    Q_proj = np.array([Q[0], Q[1], 0.0])
    ax.plot([Q[0], Q_proj[0]], [Q[1], Q_proj[1]], [
            Q[2], Q_proj[2]], "k:", linewidth=0.5)
    ax.plot([P[0], Q_proj[0]], [P[1], Q_proj[1]], [
            P[2], Q_proj[2]], "k:", linewidth=0.5)

    ax.plot_surface(Xp, Yp, Zp, facecolor=(0, 0, 0, 0.05),
                    edgecolor=(0, 0, 0, 0.5), linewidth=0.5)
    ax.text(P[0] - 0.8, P[1] - 0.8, P[2] + 0.1, r"$T_P\Sigma$", fontsize=12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.view_init(elev=25, azim=-60)
    ax.set_axis_off()

    plt.savefig(out_svg, dpi=DPI, transparent=True,
                bbox_inches="tight", pad_inches=0)
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
    plot_second_fundamental_form(out_svg, remove_patches=True)
