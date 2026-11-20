"""Shared helpers for the tests that execute a notebook's own code cells.

The notebooks are run cell by cell in one namespace, as Jupyter would, with the heavy calls
(the Cellpose trainer, the engine stages, the u-Segment3D solvers) replaced by fakes that record
what they were called with. Nothing here needs a GPU or the data deposit.
"""
import json, os, sys, types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

# The notebooks' first cell finds the repository root by its environment.yml and its data/ folder
# (the deposit itself is redirected to tmp_path with EPICYCLE_ANALYSES_DATA).
needs_data_root = pytest.mark.skipif(
    not (REPO / "data").is_dir(),
    reason="the notebooks locate the repository by its data/ folder: create an empty data/ to run this test")


def code_cells(path: Path):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run_cells(cells, namespace, stop_after: str):
    """Execute the cells in order up to and including the first one containing stop_after."""
    for src in cells:
        exec(compile(src, "<cell>", "exec"), namespace)
        if stop_after in src:
            return
    raise AssertionError(f"no cell contains {stop_after!r}")


def only_cell(cells, marker: str) -> int:
    """Index of the one cell containing marker (fails if there are none or several)."""
    hits = [i for i, c in enumerate(cells) if marker in c]
    assert len(hits) == 1, f"{len(hits)} cells contain {marker!r}"
    return hits[0]


def tag_of(arr) -> int:
    return int(np.asarray(arr).flat[0])


class CellposeRecorder:
    """Stands in for the cellpose package: CellposeModel and train.train_seg record their calls.

    The fake modules are put in sys.modules (restored after the test), so the notebooks'
    `from cellpose import models, train` picks them up without importing Cellpose and torch,
    which takes ~20 s. `models` holds the keyword arguments of every model built, `trains` those
    of every train_seg call plus `model` (the kwargs of the model whose net was trained). The
    fake train_seg creates <save_path>/models/<model_name>, as Cellpose does.
    """

    def __init__(self, monkeypatch):
        self.models, self.trains = [], []
        rec = self

        class FakeCellposeModel:
            def __init__(self, *args, **kwargs):
                assert not args, f"CellposeModel called with positional arguments {args}"
                self.kwargs = kwargs
                self.net = ("fake-net", len(rec.models))
                rec.models.append(kwargs)

        def fake_train_seg(net, *args, **kwargs):
            assert not args, "train_seg called with positional arguments"
            rec.trains.append(dict(kwargs, model=rec.models[net[1]]))
            out = Path(kwargs["save_path"]) / "models" / kwargs["model_name"]
            out.mkdir(parents=True, exist_ok=True)
            return str(out), np.zeros(3), np.zeros(3)

        pkg = types.ModuleType("cellpose")
        pkg.__path__ = []
        for name in ("models", "train", "io", "core"):
            mod = types.ModuleType(f"cellpose.{name}")
            setattr(pkg, name, mod)
            monkeypatch.setitem(sys.modules, f"cellpose.{name}", mod)
        pkg.models.CellposeModel = FakeCellposeModel
        pkg.train.train_seg = fake_train_seg
        monkeypatch.setitem(sys.modules, "cellpose", pkg)


def fake_torch(monkeypatch):
    """A torch whose CUDA reports one GPU: the training notebooks print the pinned device."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(device_count=lambda: 1, get_device_name=lambda i=0: "fake GPU")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))  # restored after


def posix(p) -> str:
    return str(p).replace("\\", "/")


def stub_plotly(monkeypatch):
    """Put a stand-in plotly in sys.modules (restored after the test) and return its submodules.

    Every attribute of plotly.express / graph_objects / subplots / figure_factory / io is a
    MagicMock, so building, showing and saving a figure does nothing and writes no file. Plotly and
    kaleido are not needed: the figure notebooks import plotly in their setup cells. A test can set
    a real function on a submodule (e.g. figure_factory.create_dendrogram) where a cell reads the
    figure back.
    """
    from unittest import mock

    def _getattr(modname):
        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return mock.MagicMock(name=f"{modname}.{attr}")
        return __getattr__

    pkg = types.ModuleType("plotly")
    pkg.__path__ = []
    subs = {}
    for name in ("express", "graph_objects", "subplots", "figure_factory", "io"):
        mod = types.ModuleType(f"plotly.{name}")
        mod.__getattr__ = _getattr(f"plotly.{name}")
        setattr(pkg, name, mod)
        subs[name] = mod
        monkeypatch.setitem(sys.modules, f"plotly.{name}", mod)
    pkg.__getattr__ = _getattr("plotly")
    monkeypatch.setitem(sys.modules, "plotly", pkg)
    return subs


# ----------------------------------------------------------------------------- the whole-tissue engine
class _FakeStage:
    """An engine stage that only keeps the keyword arguments it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.name = type(self).__name__


class TilingStage(_FakeStage): ...
class SegmentationStage(_FakeStage): ...
class ReconstructionStage(_FakeStage): ...
class AssemblyStage(_FakeStage): ...
class GlobalReconstructionStage(_FakeStage): ...
class IoUStitchStage(_FakeStage): ...


class FakePipeline:
    """EpiCyclePipeline: keeps its stages and the tile folders run() was given; runs nothing."""

    def __init__(self, work_dir, n_gpus, log_dir, mode="coarse"):
        self.work_dir, self.n_gpus, self.log_dir, self.mode = work_dir, n_gpus, log_dir, mode
        self.stages, self.runs = [], []

    def add_stage(self, stage):
        self.stages.append(stage)

    def run(self, tile_dirs, resume=False):
        self.runs.append(list(tile_dirs))

        class R:
            def summary(self_):
                return "fake run"
        return R()


class _RecordingStage:
    """A post-stitch stage: records its kwargs and its run() call; run() returns a result with summary()."""
    built = []

    def __init__(self, **kwargs):
        self.kwargs, self.runs = kwargs, []
        type(self).built.append(self)

    def run(self, dirs, n_gpus=None):
        self.runs.append((list(dirs), n_gpus))

        class Result:
            def summary(self_):
                return "fake run"
        return Result()


class PostprocessStage(_RecordingStage):
    built = []


class FeatureStage(_RecordingStage):
    built = []


def fake_engine(monkeypatch, n_gpus=3):
    """Replace the engine's pipeline, every stage the run notebooks build and the GPU count by the
    fakes above (restored after the test), so a run notebook builds its pipeline and runs nothing.
    Needs the EpiCycle package (packages.epicycle_scaledseg)."""
    import packages.epicycle_scaledseg.core.pipeline as pipeline_mod
    import packages.epicycle_scaledseg.core.tiling as tiling_mod
    import packages.epicycle_scaledseg.core.segmentation as seg_mod
    import packages.epicycle_scaledseg.core.reconstruction as recon_mod
    import packages.epicycle_scaledseg.core.assembly as asm_mod
    import packages.epicycle_scaledseg.core.global_reconstruction as glob_mod
    import packages.epicycle_scaledseg.core.iou_stitch_stage as stitch_mod
    import packages.epicycle_scaledseg.core.postprocess_stage as pp_mod
    import packages.epicycle_scaledseg.core.feature_stage as feat_mod
    import packages.epicycle_scaledseg.utils.resources as res_mod
    PostprocessStage.built.clear()
    FeatureStage.built.clear()
    for mod, name, fake in [(pipeline_mod, "EpiCyclePipeline", FakePipeline), (tiling_mod, "TilingStage", TilingStage),
                            (seg_mod, "SegmentationStage", SegmentationStage),
                            (recon_mod, "ReconstructionStage", ReconstructionStage), (asm_mod, "AssemblyStage", AssemblyStage),
                            (glob_mod, "GlobalReconstructionStage", GlobalReconstructionStage),
                            (stitch_mod, "IoUStitchStage", IoUStitchStage), (pp_mod, "PostprocessStage", PostprocessStage),
                            (feat_mod, "FeatureStage", FeatureStage), (res_mod, "get_allocated_gpus", lambda: n_gpus)]:
        monkeypatch.setattr(mod, name, fake)


def quiet_pyplot(monkeypatch):
    """matplotlib on Agg, and plt.show closes the figures instead of opening a window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    return plt
