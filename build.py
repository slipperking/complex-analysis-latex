import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"

def run_python_scripts():
    print("Running preprocessing scripts...")
    for script in sorted(SCRIPTS_DIR.rglob("*.py")):
        print(f"  → {script}")
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"Script failed: {script}")
            sys.exit(result.returncode)

def run_latex():
    main_tex = ROOT / "main.tex"
    print("Running LaTeX build...")
    cmd = [
        "latexmk",
        "--max-print-line=10000",
        "-interaction=nonstopmode",
        "-file-line-error",
        "--shell-escape",
        "-pdf",
        f"-outdir={ROOT}",
        str(main_tex),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("LaTeX build failed")
        sys.exit(result.returncode)

def main():
    run_python_scripts()
    run_latex()
    print("Build complete.")

if __name__ == "__main__":
    main()