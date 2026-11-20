# 2D Cross-Round Metrics — Data Processing Notes

## Input files
- `spatial_cross_round_metrics.csv` — Lateral (XY) model evaluations, 50,050 slice-level rows
- `axial_cross_round_metrics.csv` — Axial (XZ/YZ) model evaluations, 100,030 slice-level rows

Each CSV contains per-slice metrics (dice, iou, aji, ap_50, ap_75, ap_90) for every combination of model round (0–9) × evaluation dataset round (0–9).

## Filtering: empty-slice exclusion (lateral only)

### Problem
8.7% of lateral (spatial) slices have AP@50 = 0. They fall into two distinct groups:

| Category | Count | Description |
|---|---|---|
| dice = 0 AND ap_50 = 0 | 4,046 | No overlap at all: the truth or the prediction is empty, or the two do not overlap (edge slices with no tissue) |
| dice > 0 AND ap_50 = 0 | 310 | Model produced some output but no instance matched at IoU ≥ 0.50 |

The first category is dominated by empty edge slices at the top and bottom of the confocal z-stacks (no tissue), where the model produced stray predictions, but also includes slices where the prediction and the ground truth simply do not overlap. A degenerate zero score of this kind is uninformative for the comparison across rounds (Methods 5.1).

The axial data has zero instances of AP@50 = 0, so no filtering is needed there. This is expected because axial slices (XZ/YZ cross-sections) always intersect tissue at some depth, whereas lateral slices at stack extremes can be entirely outside the tissue volume.

### Decision
**Exclude slices with no overlap at all** (dice == 0 AND ap_50 == 0). Slices where the truth and the prediction are both empty score AP = 0/0 = NaN; they are not matched by this rule and are kept.

- This removes 4,046 / 50,050 spatial slices (8.1%)
- Retains the 310 slices with dice > 0 but AP@50 = 0 (genuine detection failures)
- No filtering applied to axial data (0 affected slices)

### Justification
1. **Standard practice**: Excluding degenerate zero-overlap cases is standard in instance segmentation benchmarks (e.g., COCO excludes images with no annotations from per-category AP computation)
2. **Consistency across rounds**: The zero-rate is roughly uniform across model rounds (7.8–10.3%), so filtering does not differentially benefit any model
3. **Conservative approach**: We only exclude the degenerate zero-overlap cases (dice 0 and AP@50 0). Every slice with any overlap, including the 310 with dice > 0 but AP@50 = 0, is retained
4. **Impact on medians**: Filtering primarily affects the lower tail of distributions; median AP@50 values change by ~0.01–0.02, preserving the same relative ordering and saturation pattern across rounds

## Output files
- `spatial_cross_round_metrics_filtered.csv` — Lateral data, empty slices excluded
- `axial_cross_round_metrics_filtered.csv` — Axial data (unchanged, included for pipeline consistency)
- `2D_cross_round_heatmap_lateral.csv` — Median AP@50 pivot table (model_round × eval_round), lateral
- `2D_cross_round_heatmap_axial.csv` — Median AP@50 pivot table (model_round × eval_round), axial
- `2D_final_round_summary.csv` — Summary statistics for all models evaluated on the final (round 9) dataset

## Processing script
See `process_2D_cross_round_metrics.py`
