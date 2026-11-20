# Fig. 5: mammalian epithelia

Panels: Fig. 5A-F; Videos S5, S6.

Contents: salivary_gland/ (cell and nucleus runs, consensus clean-up); human/finetuning_zf_to_human/ (the model pair used to segment the human stacks in the plugin's stack-segmentation step (Step 4)).

Words used in the code vs the manuscript: see `TERMINOLOGY.md` at the repository root.

## Notebooks and panels

| Notebook | Panel | What it does |
|---|---|---|
| `salivary_gland/run_salivary_gland_cells.ipynb` | 5A, 5B, Video S5 | Whole-tissue run on the mouse salivary gland explant (`SG`) with the salivary cell model pair |
| `salivary_gland/run_salivary_gland_nuclei.ipynb` | 5A, 5B, Video S5 | The same run with the salivary nucleus model pair |
| `salivary_gland/cleanup_consensus.ipynb` | 5A, 5B | Makes the cell and nucleus masks consistent: every nucleus inside exactly one cell |
| `human/finetuning_zf_to_human/finetune_zf_to_human_lateral.ipynb`, `finetune_zf_to_human_axial.ipynb` | 5C-F, Video S6 | Fine-tune the zebrafish DSM pair on one corrected human crop |

The human stacks of Fig. 5C-F were then segmented with the fine-tuned pair in the EpiCycle
plugin (Step 4). That workspace is deposited as `data/fig6_7/workspace/` and is described in
`../fig6_7_human_morphospace/README.md`.

The two run notebooks are the EpiCycle whole-tissue engine notebook (`packages.epicycle_scaledseg`
in the EpiCycle plugin) in its full-resolution mode: tiling (stage 0), 2D segmentation of every tile
in the three views with the lateral and axial models (stage 1), per-tile direct 3D reconstruction
with u-Segment3D (stage 2), then the tile masks joined at full resolution where tiles overlap, cells
matched by overlap (IoU), in place of the assembly and global reconstruction used for Fig. 4, a post-stitch clean-up and smoothing,
and per-cell features. The saved printouts are from the original runs, made with an earlier
build of the engine; the notebooks import the plugin's engine.

## Where the data for this section should be located

Everything is read from and written under `data/fig5/` (or the folder named by the
`EPICYCLE_ANALYSES_DATA` environment variable); the corrected human crop is read from
`data/ground_truth/human/crop/`. The first cell of every notebook moves to `data/fig5/`.

```
data/fig5/
  domain-specialized_finetuned_bootstrap/models/      a copy of the zebrafish DSM pair (deposited, read by the fine-tuning)
  salivary_gland/
    SG.tif                                            the processed, isotropic explant volume, 2 channels (deposited, read)
    cell_models/lateral, cell_models/axial            the salivary cell pair: checkpoint files, with loss curves,
    nuclei_models/lateral, nuclei_models/axial        training configs and manifest.json next to them (deposited, read)
    epicycle_output_cell/                             the cell run (regenerated)
      tiles/                                          stage 0 tiles and tiling_config.json
      logs/, stage1_segmentation/, stage2_reconstruction/
      stage4_full_stitch/                             global_labels.{tif,npy}, label_provenance.csv, result_manifest.json;
                                                      global_labels_clean.{tif,npy}, label_provenance_clean.csv,
                                                      postprocess_manifest.json; cell_features.{parquet,csv}, feature_manifest.json
      reconcile/cleanup_cell_labels.tif,
                cleanup_nuclei_labels.tif             the consensus masks (deposited; the .npy copies are regenerated)
    epicycle_output_nuclei/                           the nucleus run, same tree without reconcile/ (regenerated, not deposited)
  human_finetuning/
    multi_stack_models-finetune_Zf_to_human/models/
      multi_stack_lateral_finetune_Zf2human, multi_stack_axial_finetune_Zf2human
                                                      the fine-tuned pair, with loss curves and training configs (deposited; made by the fine-tuning notebooks)
    dense_FT_ZfHuman-lateral/, dense_FT_ZfHuman-axial/  the slice splits: train/, val/, test/ and metadata.json (deposited; made by the fine-tuning notebooks)
data/ground_truth/human/crop/                         (read by the fine-tuning notebooks)
  img/human_patch1.tif, mask/human_patch1_mask.tif    the corrected human crop and its instance mask
```

The checkpoints in `cell_models/` and `nuclei_models/` must be files named exactly `lateral` and
`axial` (the engine hands the path to Cellpose as a checkpoint). The nucleus run's log line names
its pair `nuclei_models/lateral (1)` and `axial (1)`; these are the files `nuclei_models/lateral` and
`axial`, the names the notebook and the deposit use.

## The setup

| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 1a. Cell run | `salivary_gland/run_salivary_gland_cells.ipynb` | `salivary_gland/SG.tif`, `salivary_gland/cell_models/{lateral,axial}` | `salivary_gland/epicycle_output_cell/` |
| 1b. Nucleus run | `salivary_gland/run_salivary_gland_nuclei.ipynb` | `salivary_gland/SG.tif`, `salivary_gland/nuclei_models/{lateral,axial}` | `salivary_gland/epicycle_output_nuclei/` |
| 2. Consensus | `salivary_gland/cleanup_consensus.ipynb` | `salivary_gland/epicycle_output_{cell,nuclei}/stage4_full_stitch/global_labels_clean.npy`; the raw channels from `salivary_gland/epicycle_output_cell/tiles/` | `salivary_gland/epicycle_output_cell/reconcile/cleanup_{cell,nuclei}_labels.{npy,tif}` |
| Human fine-tuning | `human/finetuning_zf_to_human/finetune_zf_to_human_lateral.ipynb` (XY), `finetune_zf_to_human_axial.ipynb` (XZ) | `../ground_truth/human/crop/{img,mask}/`, the DSM checkpoint of the view under `domain-specialized_finetuned_bootstrap/models/` | `human_finetuning/multi_stack_models-finetune_Zf_to_human/models/`, `human_finetuning/dense_FT_ZfHuman-{lateral,axial}/` |

Only the two consensus `.tif` of the salivary runs are deposited, so the consensus notebook needs
both run notebooks first: it reads the cleaned stitch of each run as `.npy`, and the cell run's
`tiles/` for nucleus recovery. The engine skips any stage whose output already exists, so
re-running a run notebook on a folder that already holds stage outputs skips those stages
silently: delete `epicycle_output_cell/` or `epicycle_output_nuclei/` to recompute.

## Settings as run

Salivary gland, both runs (the nucleus run differs only in its model pair, its channels and its output folder):

| Setting | Value |
|---|---|
| Tiles | 1024 px, 25 % overlap (42 tiles, 7 rows x 6 columns, on the 266 x 5631 x 4709 volume) |
| Tissue depth range | slices with intensity > 0.07 in channel 1 (hard threshold) |
| 2D segmentation | channels `[2, 1]` (cells) or `[1, 2]` (nuclei), flow threshold 0.2, cell probability 0, batch 1024, rescale on, diameter not set |
| Per-tile reconstruction | u-Segment3D direct method; gradient descent at the engine defaults (25 iterations, decay 0.001, momentum 0.98); probability maps pre-smoothed (sigma 2.5 across the slicing plane, 1.0 in plane), flows not; foreground eroded (kernel 3), border margin 2, holes filled |
| Stitch | tile masks joined where tiles overlap, cells matched by overlap (IoU), at full resolution: IoU threshold 0.25, at least 10 shared voxels, one-to-one by highest IoU, per-cell provenance table |
| Post-stitch clean-up | objects under 200 voxels removed; detached islands removed (kept if at least 1 % of the largest component; occupancy grid stride 8, one dilation); per-cell surface smoothing, sigma 1.5, topology preserved. Cells 90,862 -> 82,165; nuclei 87,411 -> 80,169 |
| Features | voxel size 0.26 µm in every axis, surface family included |

Consensus (`cleanup_consensus.ipynb`): the cell labels are trusted and the nuclei made to conform.
A nucleus straddling two or more cells is cut at the cell boundary (a piece must hold at least
20 % of the nucleus's in-cell voxels; the cut pieces are smoothed with sigma 1.5); cells under 100
voxels and nuclei outside every cell are dropped; holes are filled and cells are not smoothed; a
cell left without a nucleus gets one recovered from the nuclear stain (channel 0) where possible;
cells with two or more nuclei are kept and reported. Result: 82,063 cells, 77,490 of them
with a nucleus (94.4 %); 1,867 cells hold two or more nuclei.

Human fine-tuning (Table S5): both notebooks start from the DSM checkpoint of their view and
train for 500 epochs, learning rate 0.01, batch 16, SGD with momentum 0.9 and weight decay 1e-5,
channels `[2, 1]`, on an 80/20 train/validation split (seed 42) of every slice of the one
corrected crop `human_patch1`: 127/31 lateral (XY) slices and 204/51 axial (XZ) slices, each
slice's labels cleaned as for the dense training sets (relabelled, objects under 100 px removed,
smoothing with a disk of radius 1). The one slice left over by the integer split is written to
the `test/` folder of the export and not used.

## Notes

- The axial fine-tuning used XZ slices only. The notebook computes `VIEWS_TO_LOAD = ["XZ", "YZ"]`
  and its comment says pooled axial, but the loader is given `VIEW = "XZ"` and `VIEWS_TO_LOAD` is
  never used. `EDGE_BUFFER = 50` has no effect either: every slice of the loader is used (256 XZ
  slices, the full Y extent of the crop).
- Fine-tuning is seeded only for the split. The trainer's seed is left unset, so re-training gives a
  different pair; Figs. 5C-F, 6 and 7 use the deposited pair.
- The export line in the axial notebook's printout names the folder `dense_FT_ZfHuman`; the code
  and the deposit name it `dense_FT_ZfHuman-axial`.
- Sections 6 and 7 of `cleanup_consensus.ipynb` compute per-cell and per-nucleus feature tables
  for inspection; section 7 writes them only with `WRITE_FEATURES = True`, which is False, and
  nothing in this repository reads them. They show no saved output.
- The inline viewer of `cleanup_consensus.ipynb` (section 5) reads the consensus `.npy` files
  from `reconcile/`, which are not deposited; it falls back to the arrays in memory.
- The nucleus notebook's saved printout is from a resumed run: its log shows tiling, segmentation and
  reconstruction already complete (0.0 s); the stitch and the feature stage ran, and the clean-up was
  skipped as already complete (its counts come from the existing manifest). The segmentation settings
  of that run are in its parameters log line.
- The run notebooks do not pass a `feature_set` to the feature stage. The original runs measured
  the families morphology, surface, discrete compactness, skeleton, cross-section, crowding and
  provenance (log line of section 12); the plugin engine's default set is `"full"`, so a re-run
  writes more columns.
- The `FLOW_THRESHOLD`, `CELLPROB_THRESHOLD`, `CHANNELS` and `BATCH_SIZE` variables are passed to
  the segmentation stage here (unlike the Fig. 4 notebooks). The coarse-mode cells (assembly and
  global reconstruction parameters) are not used, since `MODE = "full"`.
- Resources: the runs used 3 GPUs (as their log shows), and stage 1 runs at batch size 1024. The consensus
  notebook holds the two int32 label volumes of 266 x 5631 x 4709 voxels (about 28 GB each) in
  memory, plus their cleaned copies.
- The fine-tuning notebooks import `TrainingConfig` and `CellposeTrainer` from the EpiCycle
  plugin (`packages.epicycle_training`) and the slice loader and split exporter from
  `finetune_helpers.py` next to them, the same file as `../fig4_zebrafish_tissue_scale/wound_finetuning/finetune_helpers.py`.

`python -m pytest tests -q` from the repository root steps through the notebooks with the segmentation
stages and the trainer replaced by stand-ins that only capture their settings (`tests/test_fig5_chain.py`),
so it checks the settings above without data or a GPU.
