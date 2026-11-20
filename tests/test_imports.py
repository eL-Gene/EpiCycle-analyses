"""Every import in the notebooks and scripts resolves to a package in environment.yml or a helper file next to the notebook.

The packages count as installed when the environment file installs them directly or through the
pinned EpiCycle v0.1.2 (its plugin packages and u-Segment3D included). An import of a private
module fails the test.

Imports are read with `ast` (IPython magics and shell lines dropped first), so an import inside
a function, a `try:` block or an `if` counts, and so does `importlib.import_module("name")` /
`__import__("name")` with a literal name. A cell that still does not parse falls back to a
line regex.

Run from the repository root:  python -m pytest tests -q
"""
import ast, importlib.util, json, pkgutil, re, sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STDLIB = set(sys.stdlib_module_names) | {"IPython", "ipywidgets"}          # the notebook runtime
PUBLIC_TOP = {                                                              # distributions the environment installs
    "numpy", "pandas", "scipy", "skimage", "sklearn", "matplotlib", "pylab", "seaborn", "plotly", "kaleido",
    "tifffile", "tqdm", "natsort", "cellpose", "torch", "torchvision", "umap", "leidenalg", "igraph", "networkx",
    "pyvista", "trimesh", "ptitprince", "PIL", "cv2", "yaml", "joblib", "h5py", "zarr", "dask", "napari", "imageio",
    "napari_epicycle",                                                      # EpiCycle v0.1.2
    "segment3D",                                                            # u-Segment3D
}
PRIVATE_TOP = {"epicycle", "epicycle_local", "epicycle_training", "epicycle_core", "largescale_segmentation"}

# The EpiCycle plugin's packages/ ships a fixed set of subpackages: `packages.<sub>` must be one of them.
_PACKAGES_SPEC = importlib.util.find_spec("packages")
EPICYCLE_SUBPACKAGES = (None if _PACKAGES_SPEC is None or not _PACKAGES_SPEC.submodule_search_locations else
                        {m.name for m in pkgutil.iter_modules(_PACKAGES_SPEC.submodule_search_locations)})

# Files known to import something a reader cannot install, with exactly what they import.
# The test fails if such a file imports more or fewer modules than listed here, and if a file
# not listed here imports anything private or unknown.
EXPECTED_PRIVATE: dict = {}

_IMPORT_RE = re.compile(r"^\s*(?:from\s+(\.*[\w\.]*)\s+import|import\s+([\w\.]+(?:\s*,\s*[\w\.]+)*))", re.M)


def _strip_magics(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))


def _regex_imports(src: str):
    src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for m in _IMPORT_RE.finditer(src):
        if m.group(1):
            yield m.group(1)
        else:
            for mod in m.group(2).split(","):
                yield mod.strip()


def _ast_imports(tree):
    """Module names as written ('.x' for a relative import), from import statements and literal dynamic imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            if node.module:
                yield dots + node.module
            else:                                            # from . import x, y
                for a in node.names:
                    yield dots + a.name
        elif isinstance(node, ast.Call) and node.args:
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            arg = node.args[0]
            if name in ("import_module", "__import__") and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                yield arg.value


def imported_modules(chunks):
    """Every module the code chunks (notebook cells, or one script) import."""
    for chunk in chunks:
        code = _strip_magics(chunk)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            yield from _regex_imports(code)
            continue
        yield from _ast_imports(tree)


def sources():
    for p in sorted(REPO.rglob("*")):
        if p.suffix not in (".ipynb", ".py") or ".ipynb_checkpoints" in p.parts or ".git" in p.parts:
            continue
        if p.parts[len(REPO.parts)] in ("tests", "data") or p.name.startswith("tools_"):
            continue
        if p.suffix == ".ipynb":
            cells = json.loads(p.read_text(encoding="utf-8"))["cells"]
            chunks = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        else:
            chunks = [p.read_text(encoding="utf-8")]
        yield p, chunks


def _resolves_under(folder: Path, dotted: str) -> bool:
    """`dotted` is a module file (or a package with __init__.py) under folder; a bare directory is not enough.

    A directory without __init__.py (a namespace package) counts only through a module file
    inside it: `benchmarking_scripts.notebook_workers` resolves to benchmarking_scripts/notebook_workers.py.
    """
    target = folder.joinpath(*dotted.split("."))
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


def classify(path: Path, mod: str) -> str:
    if mod.startswith("."):                                  # relative import: only a module next to the file
        level = len(mod) - len(mod.lstrip("."))
        folder = path.parent
        for _ in range(level - 1):
            folder = folder.parent
        rest = mod.lstrip(".")
        return "local module next to the file" if rest and _resolves_under(folder, rest) else "UNKNOWN"
    top = mod.split(".")[0]
    if top == "packages":
        if EPICYCLE_SUBPACKAGES is None:
            return "EPICYCLE NOT INSTALLED"
        parts = mod.split(".")
        return "public" if len(parts) > 1 and parts[1] in EPICYCLE_SUBPACKAGES else "UNKNOWN"
    if top in STDLIB or top in PUBLIC_TOP:
        return "public"
    for folder in (path.parent, path.parent.parent):      # next to the file, or the notebook folder a script sits under
        if _resolves_under(folder, mod):
            return "local module next to the file"
    if top in PRIVATE_TOP:
        return "PRIVATE"
    return "UNKNOWN"


FILES = list(sources())


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def test_expected_private_lists_existing_files():
    assert set(EXPECTED_PRIVATE) <= {_rel(p) for p, _ in FILES}, "a listed file is gone: remove it from EXPECTED_PRIVATE"


def test_the_detector_sees_every_import_form():
    chunks = [
        "%matplotlib inline\n!pip list\nimport os, json\nfrom a.b import c\n",
        "def f():\n    import inside_function\ntry:\n    from in_try import x\nexcept ImportError:\n    pass\n",
        "import importlib\nm = importlib.import_module('dynamic_mod')\nn = __import__('dunder_mod')\n",
        "from . import sibling\nfrom ..up import thing\n",
        "x = 1 +\nimport after_syntax_error\n",                    # does not parse: regex fallback
    ]
    assert set(imported_modules(chunks)) == {"os", "json", "a.b", "inside_function", "in_try", "importlib",
                                             "dynamic_mod", "dunder_mod", ".sibling", "..up", "after_syntax_error"}


def test_classify_rules(tmp_path):
    folder = tmp_path / "nb"
    (folder / "bare_dir").mkdir(parents=True)
    (folder / "helpers.py").touch()
    (folder / "ns_pkg").mkdir(); (folder / "ns_pkg" / "mod.py").touch()
    nb = folder / "x.ipynb"
    assert classify(nb, "helpers") == "local module next to the file"
    assert classify(nb, ".helpers") == "local module next to the file"
    assert classify(nb, ".missing") == "UNKNOWN"
    assert classify(nb, "bare_dir") == "UNKNOWN"                              # a directory is not a module
    assert classify(nb, "ns_pkg.mod") == "local module next to the file"
    assert classify(nb, "ns_pkg.other") == "UNKNOWN"
    assert classify(nb, "largescale_segmentation") == "PRIVATE"
    assert classify(nb, "some_unlisted_dist") == "UNKNOWN"
    if EPICYCLE_SUBPACKAGES is not None:
        assert classify(nb, "packages.epicycle_scaledseg.core.pipeline") == "public"
        assert classify(nb, "packages.epicycle_private_thing") == "UNKNOWN"
        assert classify(nb, "packages") == "UNKNOWN"


@pytest.mark.parametrize("path", [p for p, _ in FILES], ids=_rel)
def test_imports_are_installable(path):
    rel = _rel(path)
    chunks = dict((str(p), c) for p, c in FILES)[str(path)]
    verdicts = {mod: classify(path, mod) for mod in imported_modules(chunks)}
    if "EPICYCLE NOT INSTALLED" in verdicts.values():
        pytest.skip("EpiCycle v0.1.2 is not installed: its packages.* imports cannot be checked")
    bad = {mod for mod, v in verdicts.items() if v in ("PRIVATE", "UNKNOWN")}
    expected = EXPECTED_PRIVATE.get(rel, set())
    assert bad == expected, (f"{rel} imports {sorted(bad)}, expected {sorted(expected)}: nothing private, and every "
                             "public package must be in environment.yml")
