# Terminology: code words vs manuscript words

The notebooks, model folder names and metric tables use some words that differ from the
manuscript's terms. This table maps one to the other; the code keeps its own words so that every
notebook, checkpoint name and CSV column agree with each other.

| In the code, file names and tables | In the manuscript | Where it appears |
|---|---|---|
| `spatial` | **lateral** (XY view) | Fig. 2 `training_data_spatial/`, `spatial_models/`, `spatial_cross_round_metrics.csv`, the `mode` column (`spatial` / `axial`); `train_hitl_rounds_lateral.ipynb` sets `train_mode = "spatial"` |
| `axial` | **axial** (XZ + YZ views, pooled) | same places |
| `lenient` | **permissive** training-data regime (Table S5) | Fig. 3 `train_scratch_*_lenient.ipynb`, model folders `*_lenient_*`, `decomposed2D_benchmark-lenient_splits/` |
| `strict` | **conservative** training-data regime (Table S5) | Fig. 3 `train_scratch_*_strict.ipynb`, model folders `*_strict_*`, `decomposed2D_benchmark-strict_splits/` |
| `7stacks` (notebook names) / `7patch` (model names) | the **seven development stacks** (patches 1-5, 7, 8) | Fig. 3 training notebooks and model folders |
| `11stacks` (notebook names; the 11-stack model names carry no stack token) | **all eleven** ground-truth stacks | Fig. 3 training notebooks and model folders |
| `bootstrap` / `bootstrap_models_round4` | the **HITL round-4 checkpoint** used as the DSM starting point | Fig. 3 `finetune_bootstrap_round4_*.ipynb`, prediction folders `bootstrap_models_round4-*` |
| `domain-specialized`, `DSM` | **domain-specialised model** | Fig. 3 and 4 model folders |
| `CPcyto3_Finetuned` / `-HLR` | **Cellpose cyto3 fine-tuned**, learning rate 0.01 / **0.1** (higher) | Fig. 3 `finetune_cyto3_lateral.ipynb`, `finetune_cyto3_axial.ipynb` (each trains 0.01 then 0.1), prediction folders |
| `500epoch_scratch` | **500 training epochs**; the from-scratch ceiling models carry it, and so do the DSM and cyto3 fine-tunes, so it does not mean from scratch | Fig. 3 model and prediction folders |
| `patch-N` | **stack N** (ground-truth stack) | everywhere |
| `_masks.tif` (development) / `_mask.tif` (held-out) | instance ground truth | `data/ground_truth/zebrafish/` |
| `ap_50`, `ap_75`, `ap_90` (Fig. 2) / `ap50` (Fig. 3) | AP@50, AP@75, AP@90 (`cellpose.metrics.average_precision`); Fig. 3 tables have AP@50 only | Fig. 2 and 3 metric tables |
| `model_round` / `eval_round` | HITL round of the **model** / of the **cumulative dataset** it is scored on | Fig. 2 metric tables |
| `wound_timepoint1` / `Wound_timepoint1` | the **wound time point** of Fig. 4B (only one exists) | Fig. 4 run notebooks and data folders |
| `Preplacode`, `Placode`, `ScaleCondensate`, `Maturing_Scale` | the **scale morphogenesis stages** of Fig. 4D | Fig. 4 run notebooks (output folder names) |
| `SG` | mouse **salivary gland** explant (Fig. 5A, B) | Fig. 5 notebooks and data |
| `leiden` | **shape cluster** (Fig. 6D) | Fig. 6/7 tables |
| `niche_leiden_r1.5` (`hop3_pure`) | **niche cluster** (Fig. 7G), 3-hop contact graph, resolution 1.5 | Fig. 6/7 tables |
| `area` (3D) | cell **volume** (µm³) | Fig. 6/7 feature tables (regionprops naming) |
