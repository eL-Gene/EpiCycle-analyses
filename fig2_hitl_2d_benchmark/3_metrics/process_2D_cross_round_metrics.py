"""
Process 2D Cross-Round Metrics for EpiCycle benchmark figures.

Inputs (read from data/fig2/cross_round_metrics/, written by step 2):
    spatial_cross_round_metrics.csv   — lateral (XY) per-slice metrics
    axial_cross_round_metrics.csv     — axial (XZ/YZ) per-slice metrics

Outputs:
    data/fig2/cross_round_metrics/
        spatial_cross_round_metrics_filtered.csv  — lateral with empty slices removed
        axial_cross_round_metrics_filtered.csv    — axial (unchanged)
    3_metrics/ (next to this script; these small tables are committed)
    2D_cross_round_heatmap_lateral.csv        — median AP@50 pivot (model × dataset round)
    2D_cross_round_heatmap_axial.csv          — median AP@50 pivot (model × dataset round)
    2D_final_round_summary.csv                — summary stats, all models on final dataset

Filtering rationale:
    Lateral slices with no overlap at all (dice == 0 AND ap_50 == 0)
    are excluded. These represent edge slices outside the tissue volume and are
    uninformative for model evaluation. See 2D_cross_round_processing_notes.md.

Usage:
    python process_2D_cross_round_metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# The deposit folders are downloaded into <repo>/data/ (override with the EPICYCLE_ANALYSES_DATA variable).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("EPICYCLE_ANALYSES_DATA", REPO_ROOT / "data"))
IN_DIR = DATA_ROOT / "fig2" / "cross_round_metrics"   # raw per-slice metrics from step 2
FILTERED_DIR = IN_DIR                                  # the large filtered tables go next to the raw ones
OUT_DIR = Path(__file__).resolve().parent              # 3_metrics/: the small summary tables

# ── Load ──────────────────────────────────────────────────────────────
df_sp = pd.read_csv(IN_DIR / "spatial_cross_round_metrics.csv")
df_ax = pd.read_csv(IN_DIR / "axial_cross_round_metrics.csv")

print(f"Loaded spatial: {df_sp.shape[0]:,} slices")
print(f"Loaded axial:   {df_ax.shape[0]:,} slices")

# ── Filter: remove empty slices (lateral only) ────────────────────────
# No overlap at all: the truth or the prediction is empty, or they do not overlap → dice == 0 AND ap_50 == 0
# Note: slices where the truth and the prediction are both empty have AP = 0/0 = NaN
# (1,737 lateral rows) and are not caught by this test. They are kept in the filtered
# tables; pandas skips NaN in the medians below, so the AP@50 values are unaffected,
# while n_slices counts those rows. Left as is deliberately (matches the published tables).
empty_mask = (df_sp["dice"] == 0) & (df_sp["ap_50"] == 0)
n_empty = empty_mask.sum()
n_genuine_zero = ((df_sp["ap_50"] == 0) & (df_sp["dice"] > 0)).sum()

print(f"\nSpatial filtering:")
print(f"  Total slices:                  {df_sp.shape[0]:,}")
print(f"  Empty slices (dice=0, ap=0):   {n_empty:,} ({n_empty/len(df_sp)*100:.1f}%)")
print(f"  Genuine failures (dice>0, ap=0): {n_genuine_zero:,}")
print(f"  Retained after filtering:      {(~empty_mask).sum():,}")

df_sp_filtered = df_sp[~empty_mask].copy()

# Axial: verify no filtering needed
ax_empty = ((df_ax["dice"] == 0) & (df_ax["ap_50"] == 0)).sum()
print(f"\nAxial: {ax_empty} empty slices (no filtering needed)")
df_ax_filtered = df_ax.copy()

# ── Save filtered data ────────────────────────────────────────────────
df_sp_filtered.to_csv(FILTERED_DIR / "spatial_cross_round_metrics_filtered.csv", index=False)
df_ax_filtered.to_csv(FILTERED_DIR / "axial_cross_round_metrics_filtered.csv", index=False)
print(f"\nSaved: spatial_cross_round_metrics_filtered.csv ({len(df_sp_filtered):,} rows)")
print(f"Saved: axial_cross_round_metrics_filtered.csv ({len(df_ax_filtered):,} rows)")

# ── Heatmaps: median AP@50 per model_round × eval_round ──────────────
heatmap_sp = (df_sp_filtered
              .groupby(["model_round", "eval_round"])["ap_50"]
              .median()
              .unstack()
              .round(3))

heatmap_ax = (df_ax_filtered
              .groupby(["model_round", "eval_round"])["ap_50"]
              .median()
              .unstack()
              .round(3))

heatmap_sp.to_csv(OUT_DIR / "2D_cross_round_heatmap_lateral.csv")
heatmap_ax.to_csv(OUT_DIR / "2D_cross_round_heatmap_axial.csv")
print(f"\nSaved: 2D_cross_round_heatmap_lateral.csv")
print(f"Saved: 2D_cross_round_heatmap_axial.csv")

print("\nLateral heatmap (median AP@50):")
print(heatmap_sp.round(2).to_string())
print("\nAxial heatmap (median AP@50):")
print(heatmap_ax.round(2).to_string())

# ── Final round summary: all models evaluated on dataset round 9 ──────
summary_rows = []
for mode_name, df_mode in [("lateral", df_sp_filtered), ("axial", df_ax_filtered)]:
    final_data = df_mode[df_mode["eval_round"] == 9]
    for r in range(10):
        sub = final_data[final_data["model_round"] == r]
        if sub.empty:
            continue
        summary_rows.append({
            "mode": mode_name,
            "model_round": r,
            "n_slices": len(sub),
            "median_ap50": round(sub["ap_50"].median(), 3),
            "mean_ap50": round(sub["ap_50"].mean(), 3),
            "std_ap50": round(sub["ap_50"].std(), 3),
            "min_ap50": round(sub["ap_50"].min(), 3),
            "max_ap50": round(sub["ap_50"].max(), 3),
            "Q25_ap50": round(sub["ap_50"].quantile(0.25), 3),
            "Q75_ap50": round(sub["ap_50"].quantile(0.75), 3),
            "median_ap75": round(sub["ap_75"].median(), 3),
            "median_ap90": round(sub["ap_90"].median(), 3),
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "2D_final_round_summary.csv", index=False)
print(f"\nSaved: 2D_final_round_summary.csv")
print("\nFinal round summary (eval on dataset round 9):")
print(summary.to_string(index=False))

# ── Impact check: compare filtered vs unfiltered medians ──────────────
print("\n" + "=" * 70)
print("IMPACT CHECK: filtered vs unfiltered median AP@50 (lateral, diagonal)")
print("=" * 70)
for r in range(10):
    unfilt = df_sp[(df_sp["model_round"] == r) & (df_sp["eval_round"] == r)]["ap_50"].median()
    filt = df_sp_filtered[(df_sp_filtered["model_round"] == r) & (df_sp_filtered["eval_round"] == r)]["ap_50"].median()
    print(f"  Round {r}: unfiltered={unfilt:.3f}  filtered={filt:.3f}  diff={filt-unfilt:+.3f}")
