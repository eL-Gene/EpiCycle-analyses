# Data deposit map

Everything below is deposited in the BioImage Archive, accession `S-BIAD____` (to be filled in), and is
**not** in this git repository, except `fig6_7/all_stacks_features.csv`. Download each folder into
its place here (`data/ground_truth/`, `data/fig2/`, `data/fig3/`, ...); every script and notebook
resolves its inputs from this folder. To keep the data elsewhere, set the `EPICYCLE_ANALYSES_DATA`
environment variable to the folder that contains both `ground_truth/` and the `fig*/` subfolders.

## Deposit facts

- Accession: deposited in the BioImage Archive, accession `S-BIAD____` (to be filled in). The raw images are in the same study, under `Raw_images/`.
- Licence: to be filled in (the archive's default is CC0).
- Size per top-level folder:

| Folder | Files | Size |
|---|---|---|
| `fig2/` | 30,080 | 30.4 GB |
| `fig3/` | 86,794 | 91.2 GB |
| `fig4/` | 3,932 | 175.1 GB |
| `fig5/` | 867 | 66.1 GB |
| `fig6_7/` | 18,051 | 115.0 GB |
| `ground_truth/` | 15,660 | 21.9 GB |
| total | 155,384 | 499.7 GB |

- Minimum download to run every notebook of one figure (the table below lists what each folder holds and which code reads it):
  - Fig. 2: `fig2/` and `ground_truth/zebrafish/`
  - Fig. 3: `fig3/` and `ground_truth/zebrafish/`
  - Fig. 4: `fig4/` and `ground_truth/wound/` (only its `crops/` is read)
  - Fig. 5: `fig5/` and `ground_truth/human/` (only its `crop/` is read)
  - Fig. 6 and 7: `fig6_7/` (no other folder is read)
- Code words in folder and file names (`spatial` = lateral, `lenient` / `strict`, `7patch`, ...) are
  explained in `TERMINOLOGY.md` at the repository root; the axial DSM checkpoint keeps the typo
  `Bootrap4Finetuned` in its name, as every notebook reads it.
- Stale values kept in deposited files:
  - The deposited `metadata.json` files record `"chunk_size": 5`; the splitting used blocks of 10,
    which is what the notebook records. The same files of the
    7-stack regimes also record 11 patches (`num_patches`) where 7 stacks were used; the notebook still
    writes `len(stacks)`, which is 11 in those cells.
  - The `patch_path` column of the Fig. 3 metric CSVs names the stacks in an older layout
    (`data/patch-N.tif`, `data_holdout/patch-N.tif`) that no longer exists. The paths are
    informational only: no code opens them, and the plot scripts read from them only whether a
    stack is held out.

| Folder | What it holds | Read or written by | Notes |
|---|---|---|---|
| `ground_truth/zebrafish/` | the 11 curated zebrafish stacks, image (2 channels) + instance mask: `development/` patch-1 to 5, 7, 8 (`_masks.tif`) and `held_out/` patch-6, 9, 10, 14 (`_mask.tif`) | read by Fig. 2 step 1 (development only) and Fig. 3 training, reconstruction and scoring (all 11) | |
| `ground_truth/wound/` | `crops/img/` and `crops/mask/`: the two corrected wound crops `patch_wound_3-2`, `patch_wound_3-3`; `curated_tile/`: the corrected wound tile `patch_wound-3.tif` + `patch_wound-3_GT.tif` (the Fig. 4C region); `curated_tile_partitions/`: that tile cut into the four crops 3-1 to 3-4 with masks, of which 3-2 and 3-3 were used | `crops/` read by the Fig. 4C fine-tuning notebooks | `curated_tile/` and `curated_tile_partitions/` are provenance, read by no code |
| `ground_truth/human/` | `crop/img/human_patch1.tif`, `crop/mask/human_patch1_mask.tif`; `hitl_workspace/`: the curated lateral and axial slice pairs of the 5 human HITL rounds (`LateralPairs/`, `AxialPairs/`); `hitl_rounds/`: the round checkpoints with their training slices | `crop/` read by the Fig. 5 fine-tuning notebooks | `hitl_workspace/` and `hitl_rounds/` are provenance, read by no code |
| `ground_truth/salivary_gland/` | `curated_stacks/{cell,nuclei}/`: the curated 3D sub-volumes (`small_patch_N.tiff` + `small_patch_N_mask.tif`; cells 1, 3, 4, 5; nuclei 1, 4); `final_training/{cell,nuclei}/pairs/`: the 2D slice pairs decomposed from them for the final training rounds (with their manifests); `hitl_workspace/`: the curated slice pairs of the Cell and Nuclei HITL runs | none | provenance, read by no code |
| `fig2/` | `training_data_spatial/`, `training_data_axial/` (sampled slices per round 0 to 9), `training_rounds_results/` (the 20 round checkpoints with loss curves), `cross_round_metrics/` (the two raw per-slice metric CSVs, and the two `*_filtered.csv` step 3 writes next to them) | written by step 1 (slices, checkpoints), step 2 (metric CSVs) and step 3 (the filtered CSVs); read by steps 2 to 4 | layout in `fig2_hitl_2d_benchmark/README.md` |
| `fig3/` | `decomposed_2D/` (the four strict / lenient, 7-stack / 11-stack regime folders, each with `lateral_dataset_v1/`, `axial_dataset_v1/` and the `<stack>_<view>_splits.json` files), `bootstrap_{lateral,axial}_models/` (Fig. 2 rounds 0, 4 and 9), the trained model folders (from-scratch, DSM, cyto3 fine-tunes), `benchmark_outputs[_finetuned]/` (3D predictions per model and stack), `metrics_outputs/` (the two metric CSVs) | written by steps 1 to 4; the bootstrap folders are read | metric CSV copies are in git under `4_metrics/` |
| `fig4/` | a copy of the DSM pair (`domain-specialized_finetuned_bootstrap/models/`); one folder per whole-tissue sample (adult skin strip, wound time point 1, four scale stages) with `input.tif` (the processed isotropic volume) and the final mask `<engine>/stage4_global/global_segmentation.tif` only (at 2x downsampling of the input on every axis; no tiles or stage 1-3 outputs are deposited, the run notebook regenerates them from `input.tif`); `wound_finetuning/` (the fine-tuned pair, the slice splits `exports/`, `exports-axial/`, and `fig4c_results/`: the corrected wound tile `patch_wound-3` (1024 x 1024 x Z) segmented with the DSM pair before and after the wound fine-tuning; the tile divides into four quadrants, 3-1 top left, 3-2 top right, 3-3 bottom left, 3-4 bottom right, of which 3-2 and 3-3 are the corrected crops the fine-tuning used, and Fig. 4C shows one quadrant) | `input.tif` and the DSM pair read; the masks, the fine-tuned pair and the slice splits written by the Fig. 4 notebooks | layout and original file names in `fig4_zebrafish_tissue_scale/README.md`; to recompute a mask, delete the sample's engine output folder first, the engine skips a stage whose output exists |
| `fig5/salivary_gland/` | `SG.tif` (processed, isotropic explant volume, 2 channels); `cell_models/` and `nuclei_models/`, each holding the checkpoint files `lateral` and `axial` with their loss curves, training configs and `manifest.json`; `epicycle_output_cell/reconcile/cleanup_cell_labels.tif` and `cleanup_nuclei_labels.tif`, the final consensus-cleaned masks | `SG.tif` and the four checkpoint files read; the cell run notebook regenerates `epicycle_output_cell/` (its `tiles/`, the stage outputs and `stage4_full_stitch/`), the nuclei run notebook `epicycle_output_nuclei/` (same tree), and the clean-up notebook `epicycle_output_cell/reconcile/` (the consensus as `.npy` and `.tif`) | `fig5/` also holds a copy of the DSM pair; of the run folders only the two consensus `.tif` in `epicycle_output_cell/reconcile/` are deposited (`epicycle_output_nuclei/` is not), so the clean-up notebook needs both run notebooks first; it also reads the cell run's `tiles/` for nucleus recovery |
| `fig5/human_finetuning/` | the zebrafish-to-human fine-tuned pair with loss files, and the lateral and axial dense slice sets | written by the Fig. 5 fine-tuning notebooks | |
| `fig6_7/` | `all_stacks_features.csv` (17,436 cells x 113 columns, in git); `workspace/`: the one workspace of the plugin's stack-segmentation step (Step 4), holding the 14 human stacks (13 analysed; `(14 wks) front skin` was segmented but not smoothed, meshed or measured, and is in no figure) (`stack_segmentation_state.json`, feature tables, adjacency CSVs, `artifacts/step4/meshes/`); `run/`: what the notebooks write and read back (the pickles, `all_features_multi_leiden.csv`, cluster tables, galleries); `figures/`: the saved panels; `supp_tables/Table_S3_Feature_dictionary.csv` (Table S3, read only by the tests) | the CSV read by notebook 01; `workspace/` read by 01, 02, 03, 04; `run/` written by 01, 02 and read by 03, 04, 05 and `stats/` | the deposited `run/` holds the original outputs of notebooks 01, 02 and 05; rerunning 01 or 02 overwrites them (copy `run/` first to keep them) |


`python tools/check_setup.py` reports which folders above are present and which packages are missing.
