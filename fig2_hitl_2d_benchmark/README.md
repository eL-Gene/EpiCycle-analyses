# Fig. 2: 2D human-in-the-loop (HITL) benchmark

Simulated bootstrap HITL training on the seven development ground-truth stacks (patches 1-5,
7, 8), ten rounds, one lateral and one axial model per round, then every checkpoint evaluated
on every round's cumulative slice set. Panels: Fig. 2D, 2E, S2A, S2B; Table S2a.

Words: the code says `spatial` where the manuscript says lateral; see `TERMINOLOGY.md` at the root.

## Where the data for this section should be located

Download the Fig. 2 deposit folder into its place, `data/fig2/` at the repository root (or anywhere, with the
`EPICYCLE_ANALYSES_DATA` environment variable pointing at the folder that contains `fig2/`).
Every notebook and script below resolves its inputs from there; the notebooks change their
working directory to `data/fig2/` in their first cell.

```
data/ground_truth/zebrafish/development/       patch-N.tif + patch-N_masks.tif, the 7 development stacks (1-5, 7, 8); shared with Fig. 3 (input to step 1)
data/fig2/
  training_data_spatial/       round_0 ... round_9: sampled XY slices + selected_indices.json (step 1 -> 2)
  training_data_axial/         round_0 ... round_9: sampled XZ + YZ slices                   (step 1 -> 2)
  training_rounds_results/     spatial_models/ and axial_models/: the 20 checkpoints + loss curves (step 1 -> 2)
  cross_round_metrics/         spatial_ / axial_cross_round_metrics.csv, one row per slice per model per round (step 2 -> 3, 4)
                               + the *_filtered.csv tables that step 3 writes next to them (step 3 -> 4)
```

To reproduce only the figures, the deposit's `cross_round_metrics/` folder is enough (steps 3
and 4). Steps 1 and 2 need a GPU and take days.

### What step 1 produces (file by file)

The two training notebooks are identical except for one line, `train_mode = "spatial"` or
`"axial"`, which picks the views (`["XY"]` or `["XZ", "YZ"]`) and the output folders. Run
with the working directory `data/fig2/`, each round N:

```
training_data_<mode>/round_N/
  spatial/ or axial/           <VIEW>_<patch>_<index>_img.tif (float32, 2 channels) + ..._mask.tif (uint16)
                               the new slices of this round plus copies of every earlier round's slices
                               (the cumulative set; the empty sibling folder of the other view is created but unused)
  selected_indices.json        every index sampled so far, per view and stack
training_rounds_results/<mode>_models/models/
  <mode>_model_roundN          the Cellpose checkpoint (round N > 0 is initialised from round N-1)
  <mode>_model_roundN_train_loss.npy
```

The loop is resumable: on restart it counts the rounds whose checkpoint `<mode>_model_roundN` and
loss curve `_train_loss.npy` both exist and continues from the first round that lacks either,
discarding a half-written `round_N` data folder. Cellpose saves the checkpoint during training, so a
checkpoint without its loss curve is from a killed job, and that round is trained again.

No validation set is passed to Cellpose (Table S5: "cumulative, no split"), so no
validation-loss file is written.
Slice sampling is random and unseeded, so a rerun of step 1 draws different slices and trains
different checkpoints: it replicates the experiment rather than reproducing the deposited
checkpoints bit for bit.

## The setup

| Step | Notebook / script | Reads | Writes |
|---|---|---|---|
| 1. Sample slices and train, round by round | `1_training/train_hitl_rounds_lateral.ipynb` (XY) and `train_hitl_rounds_axial.ipynb` (XZ + YZ) | `../ground_truth/zebrafish/development/` | `training_data_{spatial,axial}/`, `training_rounds_results/` |
| 2. Cross-round evaluation | `2_cross_round_benchmark/cross_round_benchmark.ipynb` | the 20 checkpoints and the 10 cumulative slice sets per view | `cross_round_metrics/{spatial,axial}_cross_round_metrics.csv` |
| 3. Filter and summarise | `3_metrics/process_2D_cross_round_metrics.py` (rationale in `2D_cross_round_processing_notes.md`) | the two raw CSVs | `cross_round_metrics/*_filtered.csv` (large, stay in the data folder) and, next to the script, `2D_cross_round_heatmap_{lateral,axial}.csv` and `2D_final_round_summary.csv` (= Table S2a; committed) |
| 4. Plot | `4_plots/figure2_main.py`, `figure2_supp.py` (Fig. 2D, 2E, S2B) and `figure2_cumulative_training.py` (S2A) | the filtered CSVs; the raw CSVs for S2A | next to the scripts: `figure2_main` and `figure2_supp` as HTML, plus SVG / PDF / PNG when kaleido can export static images; `figure2_cumulative_training.html` only (HTML, no static image) |

## Settings as run (step 1), checked against Methods and Table S5

- 15 new slices per stack per round, per view; 105 lateral and 210 axial slices per round for rounds 0 to 5. The candidate pool of 100 per stack and view then runs out: the lateral set stops at 700 slices from round 6 and the axial set at 1,400 from round 7, and the later rounds retrain on the same set.
- Minimum inter-slice distance 5 within a round's picks; 50-slice edge buffer on XZ / YZ only (none on XY).
- Candidate pool: 100 evenly spaced indices per stack per view; sampling without replacement,
  accumulated across rounds (`selected_indices.json`).
- Slice label clean-up: boundaries between touching instances removed, relabel, remove instances < 100 px, opening / closing with disk radius 1.
- Round 0 from scratch; rounds 1-9 start from the previous round's checkpoint.
- 1,000 epochs per round, SGD, learning rate 0.01 (fixed; `use_dynamic_lr = False`),
  batch size 32, weight decay 1e-5, momentum 0.9, channels `[2, 1]` (membrane, nucleus).

Step 2 predicts with `channels=[2, 1]`, `normalize=True`, `diameter=None`,
`cellprob_threshold=0.0`, `flow_threshold=0.4` and scores AP@50 with
`cellpose.metrics.average_precision`.

## Tests

`python -m pytest tests -q` from the repository root checks that the file names agree from
step to step, that the filter keeps exactly the slices the Methods describe (synthetic data),
and, when `data/fig2/cross_round_metrics/` is present, that the filter script reproduces the
committed summary tables.

