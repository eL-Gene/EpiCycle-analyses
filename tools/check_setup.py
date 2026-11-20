"""Checks a fresh clone: can this machine import what the notebooks import, and does it see the data?

    python tools/check_setup.py [--data FOLDER] [--figure fig2|fig3|fig4|fig5|fig6_7 ...]

Prints three tables and exits 1 if anything a notebook needs is missing:

1. Python, platform, torch and its GPU, and the versions of the pinned packages.
2. Every top-level module the notebooks and scripts import (read from their source with `ast`),
   with OK / MISSING for this interpreter. Local helper modules next to a notebook are checked as files.
3. The data layout of data/README.md: for each figure, the folders and files the figure code READS
   (required), the ones only the GPU steps read (optional, with a note), and the ones the notebooks
   WRITE (regenerated; reported, not required), with file counts.

The data root is --data, else the EPICYCLE_ANALYSES_DATA variable, else <repo>/data, the same
rule the notebooks use. Read-only: nothing is written anywhere.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata as md
import json
import os
import platform
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Distribution names whose versions matter for reproducing the figures.
PINNED = ["torch", "cellpose", "napari-epicycle", "u-Segment3D", "umap-learn", "leidenalg", "igraph",
          "scikit-image", "scikit-learn", "numpy", "pandas", "trimesh", "pyvista", "plotly", "seaborn"]

# Import name -> distribution name where they differ (so a MISSING row says what to pip install).
DIST_OF = {"skimage": "scikit-image", "sklearn": "scikit-learn", "umap": "umap-learn", "PIL": "pillow",
           "cv2": "opencv-python", "yaml": "pyyaml", "segment3D": "u-Segment3D", "napari_epicycle": "napari-epicycle",
           "packages": "napari-epicycle"}

STDLIB = set(sys.stdlib_module_names) | {"IPython", "ipywidgets"}

# What each figure's code reads (required), what only its GPU steps read (optional), and what it
# writes (regenerated), relative to the data root. Mirrors data/README.md; update both together.
LAYOUT: dict[str, dict[str, list[str]]] = {
    "ground_truth": {
        "reads": ["ground_truth/zebrafish/development", "ground_truth/zebrafish/held_out",
                  "ground_truth/wound/crops/img/patch_wound_3-2.tif", "ground_truth/wound/crops/img/patch_wound_3-3.tif",
                  "ground_truth/wound/crops/mask/patch_wound_3-2_mask.tif", "ground_truth/wound/crops/mask/patch_wound_3-3_mask.tif",
                  "ground_truth/human/crop/img/human_patch1.tif", "ground_truth/human/crop/mask/human_patch1_mask.tif"],
        "writes": [],
    },
    "fig2": {
        "reads": ["fig2/cross_round_metrics/spatial_cross_round_metrics.csv",
                  "fig2/cross_round_metrics/axial_cross_round_metrics.csv"],
        "writes": ["fig2/training_data_spatial", "fig2/training_data_axial", "fig2/training_rounds_results",
                   "fig2/cross_round_metrics/spatial_cross_round_metrics_filtered.csv",
                   "fig2/cross_round_metrics/axial_cross_round_metrics_filtered.csv"],
    },
    "fig3": {
        "reads": [],                                    # step 5 reads only 4_metrics/, which is in git
        "optional": ["fig3/bootstrap_lateral_models", "fig3/bootstrap_axial_models"],
        "writes": ["fig3/metrics_outputs/3DRecon_metrics_BootStrapped.csv",
                   "fig3/metrics_outputs/3DRecon_Metrics_Finetuned.csv",
                   "fig3/decomposed_2D",
                   "fig3/domain-specialized_lateral_models/rescale_true/models", "fig3/domain-specialized_axial_models/rescale_true/models",
                   "fig3/domain-specialized_finetuned_bootstrap/models", "fig3/domain-specialized_finetuned_CPCyto3/models",
                   "fig3/benchmark_outputs", "fig3/benchmark_outputs_finetuned"],
    },
    "fig4": {
        "reads": ["fig4/domain-specialized_finetuned_bootstrap/models",
                  "fig4/adult_skin_strip/input.tif", "fig4/wound_timepoint1/input.tif", "fig4/scale_preplacode/input.tif",
                  "fig4/scale_placode/input.tif", "fig4/scale_condensate/input.tif", "fig4/scale_maturing/input.tif"],
        "writes": ["fig4/adult_skin_strip/WT_AdultSkin_Strip/stage4_global/global_segmentation.tif",
                   "fig4/wound_timepoint1/Wound_timepoint1/stage4_global/global_segmentation.tif",
                   "fig4/scale_preplacode/Preplacode/stage4_global/global_segmentation.tif",
                   "fig4/scale_placode/Placode/stage4_global/global_segmentation.tif",
                   "fig4/scale_condensate/ScaleCondensate/stage4_global/global_segmentation.tif",
                   "fig4/scale_maturing/Maturing_Scale/stage4_global/global_segmentation.tif",
                   "fig4/wound_finetuning/multi_stack_models/models", "fig4/wound_finetuning/exports", "fig4/wound_finetuning/exports-axial"],
    },
    "fig5": {
        "reads": ["fig5/salivary_gland/SG.tif",
                  "fig5/salivary_gland/cell_models/lateral", "fig5/salivary_gland/cell_models/axial",
                  "fig5/salivary_gland/nuclei_models/lateral", "fig5/salivary_gland/nuclei_models/axial",
                  "fig5/domain-specialized_finetuned_bootstrap/models"],
        "writes": ["fig5/salivary_gland/epicycle_output_cell", "fig5/salivary_gland/epicycle_output_nuclei",
                   "fig5/salivary_gland/epicycle_output_cell/reconcile",
                   "fig5/human_finetuning/multi_stack_models-finetune_Zf_to_human/models",
                   "fig5/human_finetuning/dense_FT_ZfHuman-lateral", "fig5/human_finetuning/dense_FT_ZfHuman-axial"],
    },
    "fig6_7": {
        "reads": ["fig6_7/all_stacks_features.csv", "fig6_7/workspace/stack_segmentation_state.json",
                  "fig6_7/workspace/artifacts/step4/meshes", "fig6_7/workspace/artifacts/step4/runs",
                  "fig6_7/workspace/artifacts/step4/qc", "fig6_7/workspace/ProcessedData"],
        "writes": ["fig6_7/run/morphospace_results.pkl", "fig6_7/run/niche_results.pkl",
                   "fig6_7/run/all_features_multi_leiden.csv", "fig6_7/run/placode_vs_ridge", "fig6_7/figures"],
    },
}
# Why an optional path may be absent.
OPTIONAL_NOTE = {
    "fig3": "absent (read only by steps 2 and 3, which need a GPU; the plots of step 5 read 4_metrics/)",
}
FIGURE_FOLDERS = {"fig2": "fig2_hitl_2d_benchmark", "fig3": "fig3_3d_reconstruction_benchmark",
                  "fig4": "fig4_zebrafish_tissue_scale", "fig5": "fig5_mammalian_epithelia",
                  "fig6_7": "fig6_7_human_morphospace"}


# ----------------------------------------------------------------------------- imports
def _chunks(path: Path):
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    return [path.read_text(encoding="utf-8")]


def _strip_magics(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))


_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", re.M)


def imported_modules(path: Path) -> set[str]:
    """Dotted module names imported by one notebook or script (absolute imports only)."""
    mods: set[str] = set()
    for chunk in _chunks(path):
        code = _strip_magics(chunk)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            for m in _IMPORT_RE.finditer(code):
                mods.add(m.group(1) or m.group(2))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module)
    return mods


def source_files(folders):
    for folder in folders:
        for p in sorted((REPO / folder).rglob("*")):
            if p.suffix in (".ipynb", ".py") and ".ipynb_checkpoints" not in p.parts:
                yield p


def _local_module(path: Path, dotted: str) -> bool:
    for folder in (path.parent, path.parent.parent):
        target = folder.joinpath(*dotted.split("."))
        if target.with_suffix(".py").is_file() or (target / "__init__.py").is_file():
            return True
    return False


def check_imports(folders) -> tuple[list[tuple[str, str, str]], int]:
    """Rows (module, status, used by) for every top-level module; the count of MISSING rows."""
    users: dict[str, set[str]] = {}
    local: dict[str, set[str]] = {}
    for p in source_files(folders):
        rel = str(p.relative_to(REPO)).replace("\\", "/")
        for mod in imported_modules(p):
            top = mod.split(".")[0]
            if top in STDLIB:
                continue
            if _local_module(p, mod):
                local.setdefault(mod, set()).add(rel)
                continue
            users.setdefault(top, set()).add(rel)
    rows, missing = [], 0
    for top in sorted(users):
        try:
            importlib.import_module(top)
            status = "OK"
        except Exception as e:                                       # ImportError, DLL errors, Qt without display
            status = f"MISSING ({type(e).__name__}: pip install {DIST_OF.get(top, top)})"
            missing += 1
        rows.append((top, status, _short(users[top])))
    for mod in sorted(local):
        rows.append((mod, "OK (module next to the notebook)", _short(local[mod])))
    return rows, missing


def _short(files: set[str]) -> str:
    files = sorted(files)
    return files[0] + (f" (+{len(files) - 1})" if len(files) > 1 else "")


# ----------------------------------------------------------------------------- environment
def environment_rows() -> list[tuple[str, str]]:
    rows = [("python", sys.version.split()[0] + f"  ({sys.executable})"),
            ("platform", platform.platform())]
    try:
        import torch  # noqa: WPS433
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA device (training and segmentation need one)"
        rows.append(("torch GPU", gpu))
    except Exception as e:
        rows.append(("torch GPU", f"torch not importable ({type(e).__name__})"))
    for dist in PINNED:
        try:
            rows.append((dist, md.version(dist)))
        except md.PackageNotFoundError:
            rows.append((dist, "not installed"))
    return rows


# ----------------------------------------------------------------------------- data
NOTE_FILES = {"_WHAT_GOES_HERE.txt", "_STAGED_NOTE.txt"}  # the deposit skeleton's notes are not data


def _count(path: Path) -> str:
    if path.is_file():
        return f"file, {path.stat().st_size / 1e6:.1f} MB"
    n = sum(1 for _ in path.rglob("*") if _.is_file() and _.name not in NOTE_FILES)
    return f"{n} files"


def check_data(root: Path, figures) -> tuple[list[tuple[str, str, str, str]], int]:
    """Rows (figure, role, path, status); the count of missing REQUIRED paths."""
    rows, missing = [], 0
    for fig in figures:
        for role, key in (("reads", "required"), ("optional", "optional"), ("writes", "regenerated")):
            for rel in LAYOUT[fig].get(role, []):
                p = root / rel
                if p.exists() and (p.is_file() or _count(p) != "0 files"):
                    status = _count(p)
                elif role == "reads":
                    status = "MISSING"
                    missing += 1
                elif role == "optional":
                    status = OPTIONAL_NOTE.get(fig, "absent (not needed for the figures)")
                else:
                    status = "absent (the notebooks write it)"
                rows.append((fig, key, rel, status))
    return rows, missing


# ----------------------------------------------------------------------------- report
def _table(title: str, header: tuple, rows: list) -> str:
    rows = [tuple(str(c) for c in r) for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    return "\n".join([f"== {title}", line(header), line(tuple("-" * w for w in widths)), *map(line, rows), ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="data root (default: $EPICYCLE_ANALYSES_DATA, else <repo>/data)")
    ap.add_argument("--figure", action="append", choices=sorted(FIGURE_FOLDERS),
                    help="check only these figures (repeatable; default: all)")
    args = ap.parse_args(argv)

    figures = args.figure or sorted(FIGURE_FOLDERS)
    data_root = Path(args.data or os.environ.get("EPICYCLE_ANALYSES_DATA") or REPO / "data").resolve()

    print(_table("environment", ("item", "value"), environment_rows()))

    import_rows, n_import_missing = check_imports([FIGURE_FOLDERS[f] for f in figures])
    print(_table("imports used by the notebooks and scripts", ("module", "status", "used by"), import_rows))

    data_rows, n_data_missing = check_data(data_root, ["ground_truth", *figures] if not args.figure else figures)
    print(f"data root: {data_root}" + ("" if data_root.is_dir() else "  (DOES NOT EXIST)"))
    print(_table("data layout (data/README.md)", ("figure", "role", "path", "status"), data_rows))

    problems = []
    if n_import_missing:
        problems.append(f"{n_import_missing} module(s) missing: recreate the environment from environment.yml")
    if n_data_missing:
        problems.append(f"{n_data_missing} required data path(s) missing: download the deposit folders into {data_root} "
                        "or point EPICYCLE_ANALYSES_DATA at it")
    print("RESULT: " + ("ready" if not problems else "; ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
