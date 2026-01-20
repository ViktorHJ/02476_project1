import subprocess
import tomllib
from pathlib import Path
import sys


def check_dependencies() -> int:
    # -----------------------------
    # Load dependencies from pyproject.toml
    # -----------------------------
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    declared = set()

    for dep in pyproject["project"].get("dependencies", []):
        name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("<")[0].split(">")[0]
        declared.add(name.lower())

    # -----------------------------
    # Get installed packages via uv
    # -----------------------------
    result = subprocess.run(
        ["uv", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True,
    )

    installed = set()
    for line in result.stdout.splitlines():
        if "==" in line:
            pkg = line.split("==")[0].lower()
            installed.add(pkg)

    # -----------------------------
    # Compare declared vs installed
    # -----------------------------
    missing = sorted(pkg for pkg in declared if pkg not in installed)

    print("Checking installed project dependencies:\n")

    if missing:
        print("Missing packages:")
        for pkg in missing:
            print(f" - {pkg}")
        return 1  # ❌ Missing dependencies

    print("All main project dependencies are installed.")
    return 0  # ✅ Everything OK


if __name__ == "__main__":
    sys.exit(check_dependencies())
