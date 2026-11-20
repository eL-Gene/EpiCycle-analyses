# Fig. 4: zebrafish whole-tissue segmentation

Panels: Fig. 4A-D; Videos S1-S4; Tables S1, S5 and S6.

Words used in the code vs the manuscript: see `TERMINOLOGY.md` at the repository root.

## Notebooks and panels

| Notebook | Panel | What it does |
|---|---|---|
| `run_adult_skin_strip.ipynb` | 4A, Video S1 | Whole-tissue run on the adult trunk epidermis strip with the DSM pair |
| `run_wound_timepoint1.ipynb` | 4B, Video S2 | Whole-tissue run on the scratch-wounded epidermis (one time point) with the DSM pair |
| `wound_finetuning/finetune_wound_lateral.ipynb`, `finetune_wound_axial.ipynb` | 4C, Video S3 | Fine-tune the DSM pair on the manually corrected wound sub-region |
| `run_scale_preplacode.ipynb`, `run_scale_placode.ipynb`, `run_scale_condensate.ipynb`, `run_scale_maturing.ipynb` | 4D, Video S4 | Whole-tissue runs on the four scale-development stages with the DSM pair |

Every run notebook is the same engine notebook (the EpiCycle whole-tissue pipeline,
`packages.epicycle_scaledseg` in the EpiCycle plugin) with the paths and settings of that sample:
tiling (stage 0), 2D segmentation of every tile in the three views with the lateral and axial
models (stage 1), per-tile direct 3D reconstruction with u-Segment3D (stage 2), assembly of the
tile masks and probability maps into the full volume (stage 3), and the global reconstruction
of the assembled volume at 2x downsampling (stage 4, the indirect method). The whole-tissue
panels are visual; the notebooks document the settings and reproduce the output masks.

## Settings as run (Table S6)

| Setting | Strip (4A) | Wound (4B) | Scale stages (4D) |
|---|---|---|---|
| Tile size / overlap | 1280 px / 50 % | 1024 px / 50 % | 1024 px / 25 % (maturing scale: 50 %) |
| Tissue depth range | slices with mean intensity > 0.07 in channel 1 | same | same |
| 2D segmentation | DSM pair, engine defaults (channels [2, 1], cell probability 0) | same | same |
| Per-tile reconstruction | u-Segment3D direct method, engine defaults, gradient renormalisation off | same | same |
| Global reconstruction | 2x downsampling, objects < 1,000 voxels removed, gradient descent, 300 iterations / momentum 0.9 / decay 0.3 | decay 0.25 | decay 0.25 |
| Post-stitch cleanup | none | none | none |

Fine-tuning (Table S5): both wound notebooks start from the DSM checkpoint of their view and
train for 500 epochs, learning rate 0.01, batch 16, SGD with momentum 0.9 and weight decay 1e-5,
on an 80/20 train/validation split (seed 42) of every slice of the two corrected crops
(`patch_wound_3-2`, `patch_wound_3-3`): 737/184 lateral (XY) slices and 819/204 axial (XZ)
slices, each slice's labels cleaned as for the dense training sets (boundaries removed,
objects under 100 px removed, opening and closing with a disk of radius 1). The one slice left
over by the integer split is written to the `test/` folder of the export and not used.

Notes for a reader comparing the notebooks with the tables:

- The `FLOW_THRESHOLD`, `CELLPROB_THRESHOLD`, `CHANNELS` and `BATCH_SIZE` variables of the run
  notebooks are not passed to the segmentation stage, so the engine defaults were used: flow
  threshold 0.2, cell probability 0, channels [2, 1]. The engine's log line in the printed output
  shows them, except in the strip notebook, whose printed output covers stage 4 only. The flow threshold acts on Cellpose's 2D masks only; the reconstruction
  consumes the probability and flow maps, so it has no effect on the result.
- The strip notebook runs all stages from `input.tif`, like the other five. The original run of
  February 2026 ran only the last stage (on 4 February), on tiles and stage 1-3 outputs the same
  notebook had made earlier with the same settings; its printed output shows only that stage.
- The wound notebook ran all five stages on 3 to 4 February 2026; the scale-stage notebooks on
  4 February 2026.
- The engine the February runs used was an earlier build of the one now in the plugin.
  The notebooks import the plugin's engine, whose only behavioural difference in the stages
  used is the stage-2 default for gradient renormalisation (off then, on now); the notebooks pass
  it explicitly.
- The fine-tuning notebooks import `TrainingConfig` and `CellposeTrainer` from the EpiCycle
  plugin (`packages.epicycle_training`) and the slice loader and split exporter from
  `wound_finetuning/finetune_helpers.py`.

## Data layout

Everything is read from and written under `data/fig4/` (or the folder named by the
`EPICYCLE_ANALYSES_DATA` environment variable); the corrected wound crops are read from `data/ground_truth/wound/crops/`. The first cell of every notebook moves to `data/fig4/`.

```
data/fig4/
  domain-specialized_finetuned_bootstrap/models/                 (deposited, read)
      domain-specialized_Bootstrap4Finetuned_lateral_500epoch_scratch_lenient_rescale_True
      domain-specialized_Bootrap4Finetuned_axial_500epoch_scratch_lenient_rescale_True   (the DSM pair; the name typo is the file's)
  <sample>/                                                      one folder per sample, names in the table below
      input.tif                                                  the processed volume (deposited, read)
      tiles/                                                     stage 0: the tiles and tiling_config.json (regenerated)
      <engine>/                                                  the notebook's OUTPUT_NAME
          stage1_segmentation/, stage2_reconstruction/,
          stage3_assembly/, logs/                                (regenerated)
          stage4_global/global_segmentation.tif                  the final whole-tissue mask (deposited)
  wound_finetuning/
      multi_stack_models/models/multi_stack_lateral_finetune, multi_stack_axial_finetune
                     the fine-tuned pair with its loss curves and training configs (written by the notebooks; deposited)
      exports/multi_stack_split/, exports-axial/multi_stack_split/
                     the lateral and axial slice splits: train/, val/, test/ and metadata.json (written by the notebooks; deposited)
      fig4c_results/result_crop_dsm_raw.tif, result_crop_post_finetune.tif
                     the corrected wound tile patch_wound-3 segmented with the DSM pair before and after the wound
                     fine-tuning (deposited); its quadrants are numbered 3-1 top left, 3-2 top right, 3-3 bottom left,
                     3-4 bottom right; 3-2 and 3-3 are the fine-tuning crops; Fig. 4C shows one quadrant
data/ground_truth/wound/crops/                                   (read by the fine-tuning notebooks)
  img/patch_wound_3-2.tif, patch_wound_3-3.tif                   the two corrected crops (image, 2 channels)
  mask/patch_wound_3-2_mask.tif, patch_wound_3-3_mask.tif
```

| Sample folder | Engine folder (`<engine>`) |
|---|---|
| `adult_skin_strip` | `WT_AdultSkin_Strip` |
| `wound_timepoint1` | `Wound_timepoint1` |
| `scale_preplacode` | `Preplacode` |
| `scale_placode` | `Placode` |
| `scale_condensate` | `ScaleCondensate` |
| `scale_maturing` | `Maturing_Scale` |

Per sample the deposit holds `input.tif` and the final mask `<engine>/stage4_global/global_segmentation.tif`
only. For all six samples, the strip included, the mask is at 2x downsampling of `input.tif` in every
axis: Z, Y and X are each halved (the strip's input is 1886 x 1280 x 7680 voxels per channel, its mask
943 x 640 x 3840). The tiles and the stage 1-3 outputs are not deposited; the run notebooks regenerate
them from `input.tif`. The engine skips any stage whose output already exists, so re-running an engine
notebook on a folder that already holds stage outputs skips every stage (one INFO log line per skip) and
ends with the mask that was already there; on the deposit as downloaded, stages 0-3 are regenerated but
stage 4 is skipped because the deposited mask exists. Delete the sample's engine output folder
(`<sample>/<engine>/`) to recompute.

`input.tif` is the processed, isotropic microscope volume (Z, C, Y, X; channel 0 nuclei,
channel 1 membrane) each notebook tiles. The deposit names them as above; the files as
acquired were:

| Sample | Original file name |
|---|---|
| adult_skin_strip | `3-24-25_1650-1811_cdh1GFP_actb2-H2BmScar_SL18.25_M_001 - Denoised.nd2 - Stitched --Strip Perfect Cropped -- AdjustedContrast -- 16bit-1-1_PROCESSED ISO.tif` |
| wound_timepoint1 | `12-2-25 cdh1 h2b - Denoised.nd2 - Stitched-1-01_processed-4096x4096.tif` |
| scale_preplacode | `8-15-25_1791-2012_cdh1GFP_actb-h2b-mSc_SL7.7_PREplacode_002 - Stitched - Denoised.nd2-PROCESSED-ISO-1792x1792.tif` |
| scale_placode | `8-15-25_1791-2012_cdh1GFP_actb-h2b-mSc_SL8.5_placode_003 - Stitched - Denoised.nd2-PROCESSED-ISO-1792x1792.tif` |
| scale_condensate | `8-15-25_1791-2012_cdh1GFP_actb-h2b-mSc_SL8._scale condensat - Stitched - Denoised.nd2-PROCESSED-ISO-1792x1792.tif` |
| scale_maturing | `8-15-25_1791-2012_cdh1GFP_actb-h2b-mSc_SL8.9_maturing scale - Stitched - Denoised.nd2-PROCESSED-ISO-2560x2560.tif` |

## Running

The runs were executed on a cluster node with 3 or 6 GPUs; the engine detects the GPUs it is
given and runs on one as well. Tiles of 1024 px need roughly 8 GB of GPU memory for stage 1 and
tens of GB of RAM per tile for stage 2. The tests in `tests/test_fig4_chain.py` step through every
run and fine-tuning notebook with the segmentation stages replaced by stand-ins that only record
their settings, so they check the settings above without data or a GPU.
