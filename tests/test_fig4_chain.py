"""Checks for the Fig. 4 chain: every run notebook builds the whole-tissue pipeline from the public
EpiCycle package with the settings of Table S6, and the wound fine-tuning notebooks train from the
DSM pair with the settings of Table S5.

The notebooks' own code cells are executed with the engine stages and the Cellpose trainer
replaced by fakes that record what they were built with: no GPU, no data, a few seconds.

Run from the repository root:  python -m pytest tests -q
"""
import json, os, re, sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("packages.epicycle_scaledseg", reason="the EpiCycle plugin (its packages/) is not installed")
pytest.importorskip("packages.epicycle_training", reason="the EpiCycle plugin (its packages/) is not installed")

from _nbexec import code_cells, fake_engine, needs_data_root, run_cells  # noqa: E402  (fig5 imports them from here)

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "fig4_zebrafish_tissue_scale"
FT = FIG / "wound_finetuning"
DSM_LATERAL = "domain-specialized_finetuned_bootstrap/models/domain-specialized_Bootstrap4Finetuned_lateral_500epoch_scratch_lenient_rescale_True"
DSM_AXIAL = "domain-specialized_finetuned_bootstrap/models/domain-specialized_Bootrap4Finetuned_axial_500epoch_scratch_lenient_rescale_True"


# ----------------------------------------------------------------------------- layout
REVIEWED = sorted(FIG.glob("run_*.ipynb")) + sorted(FT.glob("finetune_wound_*.ipynb"))


def test_notebooks_import_only_the_public_packages():
    for nb in REVIEWED:
        code = "\n".join(code_cells(nb))
        assert "from epicycle." not in code and "import epicycle." not in code, nb.name
        assert "epicycle_training.data" not in code and "from epicycle_training import" not in code, nb.name
    for nb in FIG.glob("run_*.ipynb"):
        code = "\n".join(code_cells(nb))
        assert "from packages.epicycle_scaledseg.core.pipeline import EpiCyclePipeline" in code, nb.name
    for nb in FT.glob("finetune_wound_*.ipynb"):
        code = "\n".join(code_cells(nb))
        assert "from packages.epicycle_training import TrainingConfig, CellposeTrainer" in code, nb.name
        assert "from finetune_helpers import BootstrappedSliceLoader, export_split_dataset" in code, nb.name


def test_every_notebook_resolves_the_data_root_the_same_way():
    for nb in REVIEWED:
        first = code_cells(nb)[0]
        assert "EPICYCLE_ANALYSES_DATA" in first and '/ "fig4"' in first, nb.name


def test_no_machine_paths_or_emoji():
    emoji = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
    machine = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/]|/gpfs/|/home/[a-z]|/content/drive)")
    for p in FIG.rglob("*"):
        if p.suffix in (".ipynb", ".py", ".md"):
            s = p.read_text(encoding="utf-8")
            assert not emoji.search(s), p
            assert not machine.search(s), p


# ----------------------------------------------------------------------------- the engine runs (Table S6)
# The pipeline and its stages are replaced by recorders (fake_engine in tests/_nbexec.py).
# notebook -> (sample folder, output name, tile size, overlap, full size, gradient decay, stages built)
ALL = ["TilingStage", "SegmentationStage", "ReconstructionStage", "AssemblyStage", "GlobalReconstructionStage"]
EXPECTED = {
    "run_adult_skin_strip": ("adult_skin_strip", "WT_AdultSkin_Strip", 1280, 0.5, (1280, 7680), 0.3, ALL),
    "run_wound_timepoint1": ("wound_timepoint1", "Wound_timepoint1", 1024, 0.5, (4096, 4096), 0.25, ALL),
    "run_scale_preplacode": ("scale_preplacode", "Preplacode", 1024, 0.25, (1792, 1792), 0.25, ALL),
    "run_scale_placode": ("scale_placode", "Placode", 1024, 0.25, (1792, 1792), 0.25, ALL),
    "run_scale_condensate": ("scale_condensate", "ScaleCondensate", 1024, 0.25, (1792, 1792), 0.25, ALL),
    "run_scale_maturing": ("scale_maturing", "Maturing_Scale", 1024, 0.5, (2560, 2560), 0.25, ALL),
}


def test_every_run_notebook_is_listed():
    assert sorted(p.stem for p in FIG.glob("run_*.ipynb")) == sorted(EXPECTED)


@needs_data_root
@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_run_notebook_builds_the_pipeline_as_in_table_s6(stem, tmp_path, monkeypatch):
    fake_engine(monkeypatch, n_gpus=3)

    sample, out_name, patch, overlap, full_size, decay, stages = EXPECTED[stem]
    (tmp_path / "fig4" / sample / "tiles").mkdir(parents=True)     # what the notebook expects to find or create
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(REPO)

    ns = {}
    run_cells(code_cells(FIG / f"{stem}.ipynb"), ns, stop_after="pipeline.run(")
    assert Path.cwd() == tmp_path / "fig4"
    assert ns["INPUT_TIF"] == f"{sample}/input.tif" and ns["WORK_DIR"] == sample and ns["OUTPUT_NAME"] == out_name
    assert ns["LATERAL_MODEL"] == DSM_LATERAL and ns["AXIAL_MODEL"] == DSM_AXIAL

    built = {s.name: s.kwargs for s in ns["pipeline"].stages}
    assert list(built) == stages

    assert ns["PATCH_SIZE"] == patch and ns["OVERLAP"] == overlap and ns["FULL_SIZE"] == full_size
    assert ns["Z_THRESHOLD_METHOD"] == "hard" and ns["Z_THRESHOLD_VALUE"] == 0.07 and ns["CHANNEL_INDEX"] == 1
    if "TilingStage" in built:
        t = built["TilingStage"]
        assert (t["patch_size"], t["overlap"], t["threshold_value"], t["channel_index"]) == (patch, overlap, 0.07, 1)
    if "SegmentationStage" in built:
        s = built["SegmentationStage"]
        assert s["lateral_model"] == DSM_LATERAL and s["axial_model"] == DSM_AXIAL
        assert "segment_params" not in s                      # the engine defaults were used (see the README)
    if "ReconstructionStage" in built:
        rs = built["ReconstructionStage"]
        assert (rs["binary_erode"], rs["binary_kernel_size"], rs["binary_border_margin"], rs["binary_fill_holes"]) == (True, 3, 2, True)
        r = rs["reconstruct_params"]
        assert r["gradient_descent"]["renorm_gradient"] is False
        assert r["combine_cell_probs"]["smooth_sigma"] == 0.8 and r["combine_cell_gradients"]["smooth_sigma"] == 1.5
    if "AssemblyStage" in built:
        a = built["AssemblyStage"]
        assert (a["full_size"], a["tile_size"], a["overlap"], a["prob_stitch_mode"]) == (full_size, patch, overlap, "max")

    g = built["GlobalReconstructionStage"]
    assert (g["downsample_factor"], g["min_size"], g["use_precomputed_binary"]) == (2, 1000, True)
    gd = g["indirect_params"]["gradient_descent"]
    assert (gd["n_iter"], gd["momenta"], gd["gradient_decay"], gd["renorm_gradient"]) == (300, 0.9, decay, True)
    assert g["indirect_params"]["indirect_method"]["dtform_method"] == "edt"
    assert g["indirect_params"]["indirect_method"]["smooth_skel_sigma"] == 1


# ----------------------------------------------------------------------------- the wound fine-tuning (Table S5)
class FakeTrainer:
    calls = []

    def __init__(self, config, use_gpu=True):
        self.config = config

    def fit(self, train_data, train_labels, val_data=None, pretrained_model=None, base_model=None,
            model_name=None, save_path=None, epoch_callback=None):
        FakeTrainer.calls.append(dict(config=self.config, n_train=len(train_data), n_val=len(val_data[0]),
                                      pretrained_model=pretrained_model, base_model=base_model,
                                      model_name=model_name, save_path=save_path))


def _write_synthetic_crops(root: Path, z=12, h=40, w=40):
    """Two tiny image/mask pairs: every XY and XZ slice holds two labels of more than 100 px."""
    (root / "img").mkdir(parents=True); (root / "mask").mkdir(parents=True)
    import tifffile
    for name in ["patch_wound_a", "patch_wound_b"]:
        img = np.random.default_rng(0).integers(0, 255, size=(z, 2, h, w), dtype=np.uint8)
        mask = np.zeros((z, h, w), dtype=np.uint32)
        mask[:, :, : w // 2] = 1
        mask[:, :, w // 2:] = 2
        tifffile.imwrite(root / "img" / f"{name}.tif", img)
        tifffile.imwrite(root / "mask" / f"{name}_mask.tif", mask)


EXPECTED_FT = {  # notebook -> (view, edge buffer, pretrained DSM, model name, export folder)
    "finetune_wound_lateral": ("XY", 0, DSM_AXIAL.replace("Bootrap4Finetuned_axial", "Bootstrap4Finetuned_lateral"), "multi_stack_lateral_finetune", "wound_finetuning/exports/multi_stack_split"),
    "finetune_wound_axial": ("XZ", 50, DSM_AXIAL, "multi_stack_axial_finetune", "wound_finetuning/exports-axial/multi_stack_split"),
}


@needs_data_root
@pytest.mark.parametrize("stem", sorted(EXPECTED_FT))
def test_finetune_notebook_trains_from_the_dsm_as_in_table_s5(stem, tmp_path, monkeypatch):
    import packages.epicycle_training as training
    monkeypatch.setattr(training, "CellposeTrainer", FakeTrainer)
    FakeTrainer.calls.clear()
    _write_synthetic_crops(tmp_path / "ground_truth" / "wound" / "crops")
    (tmp_path / "fig4").mkdir(exist_ok=True)           # the notebook chdirs into data/fig4
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(FT)              # Jupyter starts in the notebook's folder; the first cell adds it to sys.path
    monkeypatch.syspath_prepend(str(FT))

    view, buffer, pretrained, model_name, export_dir = EXPECTED_FT[stem]
    ns = {}
    run_cells(code_cells(FT / f"{stem}.ipynb"), ns, stop_after="export_split_dataset(")

    cfg = ns["training_config"]
    assert (cfg.n_epochs, cfg.learning_rate, cfg.batch_size, cfg.channels) == (500, 1e-2, 16, [2, 1])
    assert (cfg.weight_decay, cfg.momentum, cfg.SGD) == (1e-5, 0.9, True)
    assert (ns["TRAIN_FRAC"], ns["VAL_FRAC"], ns["RANDOM_SEED"]) == (0.8, 0.2, 42)
    assert ns["VIEW"] == view and ns["EDGE_BUFFER"] == buffer
    assert ns["USE_CUSTOM_PRETRAINED"] and not ns["USE_CELLPOSE_BASE"]
    assert str(ns["CUSTOM_PRETRAINED_MODEL"]).replace("\\", "/") == pretrained
    loader = ns["loader"]                                     # the last stack's loader
    assert (loader.view, loader.min_size, loader.smoothing_disk_size, loader.edge_buffer, loader.relabel_masks) == (view, 100, 1, buffer, True)

    n = len(ns["imgs"])
    assert n == 2 * (12 if view == "XY" else 40)              # every slice of the view, both stacks
    [call] = FakeTrainer.calls
    assert call["config"] is cfg                              # the trainer is built with the checked config
    assert (call["n_train"], call["n_val"]) == (int(0.8 * n), int(0.2 * n))
    assert call["model_name"] == model_name and str(call["pretrained_model"]).replace("\\", "/") == pretrained
    assert str(call["save_path"]).replace("\\", "/") == "wound_finetuning/multi_stack_models"
    assert (tmp_path / "fig4" / export_dir / "metadata.json").exists()


def test_engine_segment_defaults_behind_the_fig4_runs():
    """The Fig. 4 runs pass no segment_params: Cellpose runs with the engine's defaults (Table S6)."""
    from packages.epicycle_scaledseg.config.defaults import get_default_segment_params
    seg = get_default_segment_params()
    assert seg["channels"] == [2, 1] and seg["cellprob_threshold"] == 0.0 and seg["rescale"] is True
