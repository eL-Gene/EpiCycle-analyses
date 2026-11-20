# EpiCycle-analyses

Code behind the figures of *Quitevis et al., EpiCycle* (in preparation). This repository holds
the analysis scripts, the notebooks, with the printed output of the original runs, and the small
summary tables. The segmentation method itself is the EpiCycle napari plugin
(https://github.com/eL-Gene/EpiCycle); nothing from it is copied here, and `environment.yml` installs the
release the notebooks were checked with.
The image data, segmentation masks, model checkpoints and intermediate pickles are deposited
separately (see `data/README.md`).

## Environment

One conda environment runs every script and notebook here, and it includes the EpiCycle plugin. Create
it with `conda env create -f environment.yml`. Saving figures as images (SVG, PDF, PNG) needs Chrome or
Edge on the machine; if neither is installed, run `kaleido_get_chrome` once in the environment.

The printed output saved in the training and whole-tissue notebooks comes from the original runs on
other machines, so the folder names in it differ from `data/`; the code reads from `data/` as described
below. Running those notebooks again needs a GPU.

## Running anything

1. Download the data folders you need from the archive and place them under `data/`, one folder per
   figure (`data/fig2/`, `data/fig3/`, ...). `data/README.md` lists what each folder holds and which
   folders each figure needs. If you keep the data somewhere else, set the environment variable
   `EPICYCLE_ANALYSES_DATA` to that folder.
2. Create the environment (see above).
3. Run `python tools/check_setup.py`. It tells you which Python packages are still missing and which
   data folders it cannot find, so you know before opening a notebook.
4. Open the notebooks or run the scripts of a figure folder. For Fig. 2 and 3 the subfolders are
   numbered and run in that order. Each figure folder has a README that walks through its steps.

`python -m pytest tests -q` runs the tests. They step through each notebook on tiny stand-in data,
with the training and segmentation calls replaced by stand-ins that only record their settings, and
check that every step reads the files the previous step wrote and uses the settings the paper reports.
They need no deposit folder and no GPU.

Some words in the code and file names differ from the manuscript (`spatial` for lateral, `lenient`
for permissive, and so on); `TERMINOLOGY.md` is the translation table.

## Figure map

| Figure / table | What makes it | Where | Needs a GPU |
|---|---|---|---|
| Fig. 2 models (rounds 0 to 9, lateral and axial) | simulated human-in-the-loop (HITL) training rounds: sample slices from the 7 training stacks, train per round | `fig2_hitl_2d_benchmark/1_training/` (one notebook per view); per-round slices and checkpoints in the deposit | yes |
| Fig. 2 cross-round metrics | every round model evaluated on every round's cumulative training slice set | `fig2_hitl_2d_benchmark/2_cross_round_benchmark/cross_round_benchmark.ipynb`; writes the per-slice metric CSVs (in the deposit) | yes |
| Fig. 2D, 2E, S2B; Table S2a | lateral empty-slice filter, then plots | `fig2_hitl_2d_benchmark/3_metrics/process_2D_cross_round_metrics.py` then `4_plots/figure2_main.py`, `4_plots/figure2_supp.py` | no |
| Fig. S2A | cumulative training-set size per round | `fig2_hitl_2d_benchmark/4_plots/figure2_cumulative_training.py` | no |
| Fig. 3 training data | dense 2D decomposition of the 11 stacks into strict / lenient, 7-stack / all-stack splits | `fig3_3d_reconstruction_benchmark/1_dense_splits/prepare_dense_2d_splits.ipynb` | no |
| Fig. 3 models | from-scratch ceilings (lateral / axial x 7 / all stacks x strict / lenient), the domain-specialised models (DSM: the round-4 model fine-tuned on dense slices), cyto3 fine-tunes, lateral and axial, learning rates 0.01 and 0.1 | `fig3_3d_reconstruction_benchmark/2_training/` | yes |
| Fig. 3 reconstructions and metrics | reconstruct the 11 stacks with each model pair, score against ground truth | `fig3_3d_reconstruction_benchmark/3_reconstruction/` (+ `benchmarking_scripts/`) | yes (the scoring runs on a CPU) |
| Fig. 3C, 3E, S3; Table S2b, c | box-plot statistics and plots from the metric CSVs | `fig3_3d_reconstruction_benchmark/4_metrics/*.csv`, `5_plots/compute_boxplot_stats.py`, `5_plots/epicycle_benchmark_{main,supp}_vertical.py` | no |
| Fig. 4A, 4B, 4D; Videos S1, S2, S4 | whole-tissue runs: adult skin strip, the wounded sample (first time point), scale stages | `fig4_zebrafish_tissue_scale/` (run notebooks) | yes |
| Fig. 4C; Video S3 | wound-region fine-tuning of the lateral and axial models | `fig4_zebrafish_tissue_scale/wound_finetuning/` | yes |
| Fig. 5A, 5B; Video S5 | salivary gland cell and nucleus runs, consensus clean-up | `fig5_mammalian_epithelia/salivary_gland/` | yes |
| Fig. 5C to F; Video S6 | zebrafish-to-human fine-tuning, then segmentation of the human stacks in the EpiCycle plugin (its Step 4) | `fig5_mammalian_epithelia/human/finetuning_zf_to_human/`; the plugin workspace is `data/fig6_7/workspace/` in the deposit | yes |
| Fig. 6, 7; Videos S6, S7 | human cell morphospace and niche analysis | `fig6_7_human_morphospace/notebooks/01`, `02`, `03`, `04`, `05`; `stats/`; input is `data/fig6_7/all_stacks_features.csv` (in git) plus the plugin workspace in the deposit | no |
| Fig. 7 I to L statistics | group descriptives, omnibus and post hoc tests | `fig6_7_human_morphospace/stats/` | no |
| Tables S1, S3, S5, S6 | the settings and feature lists as run | the "Settings as run" sections of the figure READMEs | no |

The run notebooks under `fig4_zebrafish_tissue_scale/` and `fig5_mammalian_epithelia/` document
the settings used for each volume. They are not meant to be re-executed to reproduce a number
(the Fig. 5 clean-up is the exception: it needs both salivary-gland runs first):
the whole-tissue panels are visual, and the deposit carries both the input volumes and the
output masks.

The Fig. 3 scoring notebooks take hours on a CPU. Their output goes to `data/fig3/metrics_outputs/`;
the tables in `fig3_3d_reconstruction_benchmark/4_metrics/` are copies of that output (step 5 reads the
copies).

## Notes on provenance

- The Fig. 6 and 7 notebooks here are the copies that produced the submitted figures
  (31 morphological features, niche clustering on the 3-hop contact graph at resolution 1.5).
  They differ from the tutorial notebooks shipped with the plugin.
- The Fig. 6 and 7 notebooks are committed without outputs; the run and training notebooks keep
  their printed text, and their images were removed to keep the files small.
- The Fig. 4C and Fig. 5 fine-tuning notebooks import the trainer and its config from the EpiCycle
  plugin (`packages.epicycle_training`) and the slice loader and split exporter from a
  `finetune_helpers.py` next to the notebooks.
- The zebrafish whole-tissue runs (February 2026) used an earlier build of the engine that is now part
  of the plugin (`packages.epicycle_scaledseg`). The notebooks import the plugin's engine and set
  explicitly the one stage default that changed since (see `fig4_zebrafish_tissue_scale/README.md`).
- After the runs the code was tidied: paths were rewritten to the deposit layout, the engine import
  was pointed at the plugin, a resume check was made stricter, and images were removed from the
  outputs; the GPU pins were removed from the Fig. 3 training notebooks, the strip notebook's stage
  switches were turned on to match the other five, a rerun no longer overwrites a trained model or
  leaves stale split files, and figure saves fall back to HTML when the image exporter is missing.
  No analysis setting changed.

## Layout

```
environment.yml                     the EpiCycle plugin environment
TERMINOLOGY.md                      code words vs manuscript words
tests/                              the tests described above (python -m pytest tests -q)
tools/check_setup.py                checks that the packages and data folders are in place
data/                               deposit map; only the 28 MB feature table is in git
fig2_hitl_2d_benchmark/             1_training/ 2_cross_round_benchmark/ 3_metrics/ 4_plots/
fig3_3d_reconstruction_benchmark/   1_dense_splits/ 2_training/ 3_reconstruction/ 4_metrics/ 5_plots/
fig4_zebrafish_tissue_scale/        run notebooks, wound_finetuning/
fig5_mammalian_epithelia/           salivary_gland/ human/finetuning_zf_to_human/
fig6_7_human_morphospace/           notebooks/ stats/
fig6_7_human_morphospace/tools/fix_step4_paths.py   makes the plugin workspace's record paths absolute
```
