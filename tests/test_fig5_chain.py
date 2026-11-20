"""Checks for the Fig. 5 chain: the two salivary-gland run notebooks build the whole-tissue
pipeline (full mode: IoU stitch) from the public EpiCycle package with the settings of Table S6,
the clean-up notebook carries the reconciliation settings of Table S6, and the two human
fine-tuning notebooks train from the DSM pair with the settings of Table S5.

As for Fig. 4, the notebooks' own code cells are executed with the engine stages and the Cellpose
trainer replaced by fakes that record what they were built with: no GPU, no data.

Run from the repository root:  python -m pytest tests -q
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("packages.epicycle_scaledseg", reason="EpiCycle v0.1.2 (the plugin's packages/) is not installed")
pytest.importorskip("packages.epicycle_training", reason="EpiCycle v0.1.2 (the plugin's packages/) is not installed")

from _nbexec import FeatureStage, PostprocessStage, fake_engine  # noqa: E402  (the engine fakes, shared with Fig. 4)
from test_fig4_chain import (FakeTrainer, _write_synthetic_crops, code_cells, needs_data_root, run_cells,  # noqa: E402
                             DSM_LATERAL, DSM_AXIAL)

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "fig5_mammalian_epithelia"
SG = FIG / "salivary_gland"
FT = FIG / "human" / "finetuning_zf_to_human"


def test_every_notebook_resolves_the_data_root_the_same_way():
    for nb in FIG.rglob("*.ipynb"):
        first = code_cells(nb)[0]
        assert "EPICYCLE_ANALYSES_DATA" in first and '/ "fig5"' in first, nb.name


# ----------------------------------------------------------------------------- the salivary-gland runs (Table S6)
EXPECTED_SG = {  # notebook -> (output name, lateral model, axial model, Cellpose channels)
    "run_salivary_gland_cells": ("epicycle_output_cell", "salivary_gland/cell_models/lateral", "salivary_gland/cell_models/axial", [2, 1]),
    "run_salivary_gland_nuclei": ("epicycle_output_nuclei", "salivary_gland/nuclei_models/lateral", "salivary_gland/nuclei_models/axial", [1, 2]),
}


@needs_data_root
@pytest.mark.parametrize("stem", sorted(EXPECTED_SG))
def test_salivary_gland_run_builds_the_full_mode_pipeline_as_in_table_s6(stem, tmp_path, monkeypatch):
    fake_engine(monkeypatch, n_gpus=3)
    (tmp_path / "fig5" / "salivary_gland").mkdir(parents=True)
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(REPO)

    out_name, lateral, axial, channels = EXPECTED_SG[stem]
    ns = {}
    run_cells(code_cells(SG / f"{stem}.ipynb"), ns, stop_after="feat_stage.run(")   # through the post-stitch cells
    assert ns["INPUT_TIF"] == "salivary_gland/SG.tif" and ns["WORK_DIR"] == "salivary_gland" and ns["OUTPUT_NAME"] == out_name
    assert ns["MODE"] == "full" and ns["pipeline"].mode == "full"
    built = {s.name: s.kwargs for s in ns["pipeline"].stages}
    assert list(built) == ["TilingStage", "SegmentationStage", "ReconstructionStage", "IoUStitchStage"]

    t = built["TilingStage"]
    assert (t["patch_size"], t["overlap"], t["threshold_value"], t["channel_index"]) == (1024, 0.25, 0.07, 1)
    s = built["SegmentationStage"]
    assert (s["lateral_model"], s["axial_model"]) == (lateral, axial)
    assert s["segment_params"]["channels"] == channels and s["segment_params"]["cellprob_threshold"] == 0.0
    assert s["segment_params"]["rescale"] is True                 # Table S6: diameter not set, rescale on
    r = built["ReconstructionStage"]
    assert (r["binary_erode"], r["binary_kernel_size"], r["binary_border_margin"], r["binary_fill_holes"]) == (True, 3, 2, True)
    assert (r["n_iter"], r["gradient_decay"], r["momenta"]) == (None, None, None)   # the engine defaults: 25 / 0.001 / 0.98
    assert (r["pre_smooth_probs"], r["pre_smooth_flows"], r["pre_smooth_sigma_perp"], r["pre_smooth_sigma_inplane"]) == (True, False, 2.5, 1.0)
    st = built["IoUStitchStage"]
    assert (st["tau"], st["vmin"], st["one_to_many"], st["emit_provenance"]) == (0.25, 10, "max_iou", True)

    # the post-stitch cleanup and the feature extraction are set up in the same notebook
    assert (ns["PP_REMOVE_SMALL"], ns["PP_MIN_VOXELS"], ns["PP_REMOVE_ISLANDS"], ns["PP_ISLAND_MIN_FRAC"]) == (True, 200, True, 0.01)
    assert (ns["PP_SMOOTH"], ns["PP_SMOOTH_SIGMA"], ns["PP_PRESERVE_TOPOLOGY"]) == (True, 1.5, True)
    assert ns["FEAT_VOXEL_SIZE_UM"] == (0.26, 0.26, 0.26)
    stitch_dir = Path("salivary_gland") / out_name / "stage4_full_stitch"

    # the post-stitch cleanup cell ran, with the Table S6 settings
    assert ns["RUN_POSTPROCESS"] is True and ns["PP_DRY_RUN"] is False
    [pp] = PostprocessStage.built
    k = pp.kwargs
    assert Path(k["input_dir"]) == stitch_dir and Path(k["output_dir"]) == stitch_dir
    assert (k["remove_small"], k["min_voxels"], k["min_percentile"], k["remove_large"]) == (True, 200, 0.0, False)
    assert (k["remove_islands"], k["island_min_component_frac"]) == (True, 0.01)
    assert (k["smooth"], k["smooth_sigma"], k["smooth_sigma_z"], k["smooth_sigma_xy"]) == (True, 1.5, None, None)
    assert (k["preserve_topology"], k["dry_run"]) == (True, False)
    assert [(Path(dirs[0]), g) for dirs, g in pp.runs] == [(stitch_dir, 3)]

    # then the feature extraction, on the cleaned labels, in microns
    assert ns["RUN_FEATURES"] is True and ns["FEAT_DRY_RUN"] is False
    [feat] = FeatureStage.built
    f = feat.kwargs
    assert Path(f["input_dir"]) == stitch_dir and f["prefer_clean"] is True and f["dry_run"] is False
    assert f["voxel_size_um"] == (0.26, 0.26, 0.26) and f["include_surface"] is True
    assert len(feat.runs) == 1


def test_engine_defaults_behind_the_salivary_gland_runs():
    """The per-tile settings the notebooks leave at None are the engine's; pin them (Table S6)."""
    from packages.epicycle_scaledseg.config.defaults import get_default_reconstruct_params, get_default_segment_params
    gd = get_default_reconstruct_params()["gradient_descent"]
    assert (gd["n_iter"], gd["gradient_decay"], gd["momenta"]) == (25, 0.001, 0.98)
    seg = get_default_segment_params()
    assert seg["rescale"] is True and seg["cellprob_threshold"] == 0.0 and seg["channels"] == [2, 1]


@needs_data_root
def test_cleanup_notebook_carries_the_reconciliation_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path)); (tmp_path / "fig5").mkdir()
    monkeypatch.chdir(REPO)
    ns = {}
    run_cells(code_cells(SG / "cleanup_consensus.ipynb"), ns, stop_after="CELL_TILES")   # the paths-and-knobs cell
    assert str(ns["CELL_RUN_DIR"]).replace("\\", "/") == "salivary_gland/epicycle_output_cell"
    assert str(ns["NUCLEI_RUN_DIR"]).replace("\\", "/") == "salivary_gland/epicycle_output_nuclei"
    assert str(ns["CELL_LABELS_NPY"]).replace("\\", "/").endswith("epicycle_output_cell/stage4_full_stitch/global_labels_clean.npy")
    assert ns["VOXEL_SIZE_UM"] == (0.26, 0.26, 0.26)
    assert (ns["SMOOTH"], ns["SIGMA"], ns["MIN_CELL_VOX"], ns["RECOVER"]) == (False, 1.5, 100, True)
    assert (ns["SPLIT_STRADDLING"], ns["MIN_PIECE_FRAC"], ns["SMOOTH_SPLIT_NUCLEI"], ns["NUCLEI_CHANNEL"]) == (True, 0.2, True, 0)
    code = "\n".join(code_cells(SG / "cleanup_consensus.ipynb"))
    assert "from packages.epicycle_scaledseg.reconcile import cleanup_volume" in code


@needs_data_root
def test_cleanup_notebook_calls_cleanup_volume_with_those_settings(tmp_path, monkeypatch):
    import packages.epicycle_scaledseg.reconcile as reconcile
    calls = []

    def fake_cleanup_volume(cells, nuclei, **kwargs):
        calls.append(dict(kwargs, cells=cells, nuclei=nuclei))
        return cells, nuclei, {}, []

    monkeypatch.setattr(reconcile, "cleanup_volume", fake_cleanup_volume)
    cells = np.zeros((2, 3, 3), np.int32); cells[:, :2, :2] = 1; cells[:, 2:, :] = 2
    nuclei = np.zeros_like(cells); nuclei[0, 0, 0] = 7
    for run, arr in [("epicycle_output_cell", cells), ("epicycle_output_nuclei", nuclei)]:
        d = tmp_path / "fig5" / "salivary_gland" / run / "stage4_full_stitch"
        d.mkdir(parents=True)
        np.save(d / "global_labels_clean.npy", arr)
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(REPO)

    ns = {}
    run_cells(code_cells(SG / "cleanup_consensus.ipynb"), ns, stop_after="= cleanup_volume(")
    [call] = calls
    assert np.array_equal(call["cells"], cells) and np.array_equal(call["nuclei"], nuclei)   # the two stitched runs
    assert (call["smooth"], call["sigma"], call["min_cell_vox"], call["recover"]) == (False, 1.5, 100, True)
    assert (call["split_straddling_nuclei"], call["min_piece_frac"], call["smooth_split_nuclei"]) == (True, 0.2, True)
    assert call["nuclei_raw"] is None and call["return_triage"] is True
    assert str(call["cells_path"]).replace("\\", "/").endswith("epicycle_output_cell/stage4_full_stitch/global_labels_clean.npy")
    assert str(call["nuclei_path"]).replace("\\", "/").endswith("epicycle_output_nuclei/stage4_full_stitch/global_labels_clean.npy")


# ----------------------------------------------------------------------------- the human fine-tuning (Table S5)
EXPECTED_FT = {  # notebook -> (view, edge buffer, pretrained DSM, model name, export folder)
    "finetune_zf_to_human_lateral": ("XY", 0, DSM_LATERAL, "multi_stack_lateral_finetune_Zf2human", "human_finetuning/dense_FT_ZfHuman-lateral"),
    "finetune_zf_to_human_axial": ("XZ", 50, DSM_AXIAL, "multi_stack_axial_finetune_Zf2human", "human_finetuning/dense_FT_ZfHuman-axial"),
}


@needs_data_root
@pytest.mark.parametrize("stem", sorted(EXPECTED_FT))
def test_human_finetune_notebook_trains_from_the_dsm_as_in_table_s5(stem, tmp_path, monkeypatch):
    import packages.epicycle_training as training
    monkeypatch.setattr(training, "CellposeTrainer", FakeTrainer)
    FakeTrainer.calls.clear()
    root = tmp_path / "ground_truth" / "human" / "crop"
    _write_synthetic_crops(root)
    (tmp_path / "fig5").mkdir(exist_ok=True)           # the notebook chdirs into data/fig5
    for extra in ["patch_wound_b.tif"]:                       # one crop, as run
        (root / "img" / extra).unlink(); (root / "mask" / extra.replace(".tif", "_mask.tif")).unlink()
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(FT)
    monkeypatch.syspath_prepend(str(FT))

    view, buffer, pretrained, model_name, export_dir = EXPECTED_FT[stem]
    ns = {}
    run_cells(code_cells(FT / f"{stem}.ipynb"), ns, stop_after="export_split_dataset(")
    cfg = ns["training_config"]
    assert (cfg.n_epochs, cfg.learning_rate, cfg.batch_size, cfg.channels) == (500, 1e-2, 16, [2, 1])
    assert (cfg.weight_decay, cfg.momentum, cfg.SGD) == (1e-5, 0.9, True)
    assert (ns["TRAIN_FRAC"], ns["VAL_FRAC"], ns["RANDOM_SEED"]) == (0.8, 0.2, 42)
    assert ns["VIEW"] == view and ns["EDGE_BUFFER"] == buffer
    assert str(ns["CUSTOM_PRETRAINED_MODEL"]).replace("\\", "/") == pretrained
    loader = ns["loader"]
    assert (loader.view, loader.min_size, loader.smoothing_disk_size, loader.edge_buffer) == (view, 100, 1, buffer)
    n = len(ns["imgs"])
    assert n == (12 if view == "XY" else 40)
    [call] = FakeTrainer.calls
    assert call["config"] is cfg                              # the trainer is built with the checked config
    assert (call["n_train"], call["n_val"]) == (int(0.8 * n), int(0.2 * n))
    assert call["model_name"] == model_name and str(call["pretrained_model"]).replace("\\", "/") == pretrained
    assert str(call["save_path"]).replace("\\", "/") == "human_finetuning/multi_stack_models-finetune_Zf_to_human"
    assert (tmp_path / "fig5" / export_dir / "metadata.json").exists()
