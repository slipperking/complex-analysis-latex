import subprocess
import shutil
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
BUILD_DIR = ROOT / "build"
SVG_INKSCAPE_DIR = ROOT / "svg-inkscape"

def clean_latex():
    print("Cleaning LaTeX auxiliary files...")
    result = subprocess.run(["latexmk", "-c"], cwd=ROOT)
    if result.returncode != 0:
        print("latexmk -c failed")
        sys.exit(result.returncode)

def clean_build_dir():
    if BUILD_DIR.exists():
        print("Removing build directory...")
        shutil.rmtree(BUILD_DIR)
    else:
        print("No build directory was found")

def clean_svg_inkscape():
    if SVG_INKSCAPE_DIR.exists():
        print("Removing SVG Inkscape directory...")
        shutil.rmtree(SVG_INKSCAPE_DIR)
    else:
        print("No SVG Inkscape directory was found")

def main():
    clean_latex()
    clean_build_dir()
    clean_svg_inkscape()
    print("Cleanup complete.")

if __name__ == "__main__":
    main()