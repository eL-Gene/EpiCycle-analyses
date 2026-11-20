# Fig. 3: 3D reconstruction benchmark

Every model pair of the paper (HITL bootstrap rounds 0, 4, 9; the domain-specialised models;
the from-scratch ceilings; the Cellpose cyto3 fine-tunes) is applied to all 11 ground-truth
stacks in three orthogonal views, the 2D predictions are fused into 3D instances with
u-Segment3D's direct method, and each reconstruction is scored against the curated 3D ground
truth with 3D AP@50. Panels: Fig. 3C, 3E, S3; Table S2b, c. Methods §3.2-3.4, §4.2-4.4, §5.2.

Words: the code says `lenient` / `strict` where the manuscript says permissive /
conservative, and `7stacks` / `11stacks` in the notebook names (`7patch` in the model names, no token for the eleven-stack models) for the seven development / all eleven stacks; see
`TERMINOLOGY.md` at the root.

## Where the data for this section should be located

Download the Fig. 3 deposit folder into its place, `data/fig3/` at the repository root (or anywhere, with the
`EPICYCLE_ANALYSES_DATA` environment variable pointing at the folder that contains `fig3/`).
The notebooks change their working directory to `data/fig3/` in their first cell and read the
11 stacks from the shared `data/ground_truth/zebrafish/` folder.

```
data/ground_truth/zebrafish/development/           patch-1,2,3,4,5,7,8 .tif + _masks.tif   (steps 1, 3, 4)
data/ground_truth/zebrafish/held_out/              patch-6,9,10,14     .tif + _mask.tif    (steps 1, 3, 4)
data/fig3/
  decomposed_2D/                   dense 2D slices and splits                               (step 1 -> 2)
    decomposed2D_benchmark-strict_splits/           all 11 stacks, 40/30/30 (conservative)
    decomposed2D_benchmark-lenient_splits/          all 11 stacks, 80/20/0  (permissive)
    decomposed2D_benchmark-7_patch-strict_splits/   7 development stacks, 40/30/30
    decomposed2D_benchmark-7_patch-lenient_splits/  7 development stacks, 80/20/0
      each: lateral_dataset_v1/ and axial_dataset_v1/ with train|val|test/images|masks, metadata.json,
            and one <stack>_<view>_splits.json per stack and view
  bootstrap_lateral_models/        spatial_model_round0, _round4, _round9  (copies of the Fig. 2 checkpoints)
  bootstrap_axial_models/          axial_model_round0, _round4, _round9
  domain-specialized_lateral_models/rescale_true/models/           from-scratch models        (step 2 -> 3)
  domain-specialized_axial_models/rescale_true/models/
  domain-specialized_finetuned_bootstrap/models/                   round-4 + dense fine-tune (step 2 -> 3)
  domain-specialized_finetuned_CPCyto3/models/                     cyto3 fine-tunes           (step 2 -> 3)
  benchmark_outputs/<stack>/<model_id>_mask.tif + _meta.json       reconstructions           (step 3 -> 4)
  benchmark_outputs_finetuned/<stack>/...
  metrics_outputs/                 the two metric tables written by step 4 (copies in 4_metrics/ in this repository)
```

To reproduce only the figures, nothing from the deposit is needed: `4_metrics/` is in the
repository and step 5 runs from it. Steps 2 and 3 need GPUs; step 3 runs the three views of one stack
on three GPUs in parallel. Step 4 runs on a CPU and
takes hours. It writes its two tables to `metrics_outputs/`; copy them into `4_metrics/` before
step 5, which reads only `4_metrics/`.

## The setup

| Step | Notebook / script | Reads | Writes |
|---|---|---|---|
| 1. Dense 2D decomposition and splits | `1_dense_splits/prepare_dense_2d_splits.ipynb` | the 11 stacks | `decomposed_2D/` (four split regimes, lateral and axial) |
| 2. Train the model pairs | `2_training/train_scratch_{lateral,axial}_{11stacks,7stacks}_{lenient,strict}.ipynb` (ceilings, from scratch), `finetune_bootstrap_round4_11stacks_{lenient,strict}.ipynb` (DSM), `finetune_cyto3_{lateral,axial}.ipynb` (cyto3 at learning rate 0.01, then 0.1) | `decomposed_2D/...`; the round-4 checkpoints; Cellpose cyto3 weights | `domain-specialized_*/models/<model_name>` + loss curves |
| 3. Reconstruct every stack with every pair | `3_reconstruction/reconstruct_benchmark.ipynb` (bootstrap + scratch pairs), `reconstruct_benchmark_finetuned.ipynb` (DSM + cyto3 pairs); helpers in `benchmarking_scripts/` | the 11 stacks, the model pairs | `benchmark_outputs*/<stack>/<model_id>_mask.tif` + `_meta.json` |
| 4. Score | `3_reconstruction/score_benchmark.ipynb`, `score_benchmark_finetuned.ipynb` | the reconstructions and the ground-truth masks | `metrics_outputs/3DRecon_metrics_BootStrapped.csv`, `3DRecon_Metrics_Finetuned.csv` (one row per stack per model: `ap50`, `tp`, `fp`, `fn`, model and checkpoint names) |
| 5. Statistics and plots | `5_plots/compute_boxplot_stats.py` (Table S2c), `epicycle_benchmark_main_vertical.py` (Fig. 3C, E), `epicycle_benchmark_supp_vertical.py` (Fig. S3) | `4_metrics/*.csv` | next to the scripts: `boxplot_stats_main.csv`, `boxplot_stats_supp.csv` (included here) and `boxplot_stats_all.csv`; each figure as HTML, plus SVG when kaleido can export static images |

## Settings as run, checked against Methods and Tables S5, S6

Step 1: per stack and view, the slices are cut into blocks of 10 consecutive slices. Within each
block the slices are shuffled and divided 40/30/30 (strict) or 80/20/0 (lenient) among train,
validation and test, so every block contributes to every split. A 50-slice edge buffer is kept on
XZ / YZ, and the slices get the same clean-up as Fig. 2 (boundaries between touching instances
removed, relabel, < 100 px removed, open/close radius 1). The shuffle is not seeded, so a rerun of
step 1 draws a different split: it deletes the exported `decomposed_2D/<regime>/{lateral,axial}_dataset_v1`
folders and writes new ones; copy `decomposed_2D/` first to keep the deposited split the models were
trained on.

Step 2, every model: 500 epochs, SGD, batch size 16, weight decay 1e-5, momentum 0.9,
channels `[2, 1]`, `min_train_masks = 0`, validation split passed for loss convergence monitoring only.
Learning rate 0.01, except the second cyto3 model of each view ("HLR") at 0.1. The DSM
notebooks start from `bootstrap_*_models/*_model_round4`; the scratch notebooks from no
weights; the cyto3 notebooks from `model_type='cyto3'`. All models were trained with `rescale=True` (Table S6: rescale on).

Step 3, prediction: `channels=[2, 1]`, `normalize=True`, `cellprob_threshold=0.0`,
`flow_threshold=0.4`, `rescale` as the model was trained. Aggregation (u-Segment3D direct
method): probability maps combined with `ksize=25`, `alpha=0.5`, `smooth_sigma=0.8`; flows
combined with `ksize=25`, `smooth_sigma=1.5`, `post_sigma=0.5`; binary post-processing with
hole filling and u-Segment3D's default object-size filters (Table S6); gradient descent 50
iterations, momentum 0.98, decay 0.001; no flow-consistency removal. No per-view pre-smoothing (Table S6, 3D benchmark rows).

Step 4: 3D AP@50 per stack from `benchmarking_scripts/reconstruction_metrics.py`
(u-Segment3D's `average_precision` at IoU 0.5; `ap50 = tp / (tp + fp + fn)`).

## Tests

`python -m pytest tests -q` from the repository root checks the layout, that every training
notebook carries the Table S5 settings for its condition, that model names agree from the
training notebooks through the reconstruction notebooks to the metric tables and the plot
scripts, that the metric tables in `4_metrics/` are well formed and carry Table S2b values, and
that `compute_boxplot_stats.py` reproduces the Table S2c tables included in `5_plots/`.

## Notes

- The cyto3 fine-tunes were run with Cellpose v3.1.1.2 (the version `environment.yml`
  installs), one notebook per view, the 0.01 model first and the 0.1 model after it. The lateral notebook keeps its printed logs; the axial
  notebook shows only its first two outputs.

