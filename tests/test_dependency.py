import importlib

packages = [
    "coverage",
    "invoke",
    "mkdocs",
    "pytest",
    "pre_commit",
]

for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"{pkg}: OK")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
# This file is used to verify that all required packages are installed.