"""
Aggregate box-and-whisker statistics for the EpiCycle 3D reconstruction benchmark.

For every model_id in the bootstrap + finetuned CSVs, computes the standard
descriptive stats that define a box plot (Tukey-style) on AP@50:
    n, mean, std, min, Q1, median, Q3, max, IQR, whisker_low, whisker_high,
    n_outliers_low, n_outliers_high

Three breakdowns are produced (long format) per output CSV:
    group = "All"          -> all patches pooled
    group = "Development"  -> dev patches only (7 patches)
    group = "Test"         -> holdout patches only (4 patches)

Outputs (next to this script):
    boxplot_stats_all.csv   -> every model_id present in the CSVs
    boxplot_stats_main.csv  -> only models shown in the main figure (6 unique)
    boxplot_stats_supp.csv  -> only models shown in the supp figure (11)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "4_metrics"

# ── Load & merge ───────────────────────────────────────────────────────
df_boot = pd.read_csv(DATA_DIR / "3DRecon_metrics_BootStrapped.csv")
df_ft = pd.read_csv(DATA_DIR / "3DRecon_Metrics_Finetuned.csv")
df = pd.concat([df_boot, df_ft], ignore_index=True)
df["patch_type"] = df["patch_path"].apply(
    lambda p: "Test" if ("holdout" in str(p) or "held_out" in str(p)) else "Development"
)

METRIC = "ap50"


# ── Figure membership + display labels (mirrors the figure scripts) ────
def _clean(lbl: str) -> str:
    """Strip plotly <br> tags for CSV-friendly labels."""
    return lbl.replace("<br>", " ").strip()


# Main figure (epicycle_benchmark_main_vertical.py): Panel C + Panel E
MAIN_LABELS = {
    "bootstrap_models0":                          "BootstrapModel [Round 0]",
    "bootstrap_models1":                          "BootstrapModel [Round 4]",
    "bootstrap_models2":                          "BootstrapModel [Round 9]",
    "bootstrap_models_round4-0":                  "BootstrapModel [Round 4] + Dense Finetuning (11 Patch)",
    "500epoch_scratch_lenient_CPcyto3_Finetuned": "CellPose Cyto3 Dense Finetuning (11 Patch)",
    "500epoch_scratch_lenient_rescale_True":      "Scratch Dense Training (11 Patch)",
}
MAIN_ORDER = list(MAIN_LABELS.keys())

# Supplementary figure (epicycle_benchmark_supp_vertical.py): all 11 conditions
SUPP_LABELS_RAW = {
    "bootstrap_models0":        "Bootstrap Round 0<br>HITL on 7 patches",
    "bootstrap_models1":        "Bootstrap Round 4<br>HITL on 7 patches",
    "bootstrap_models2":        "Bootstrap Round 9<br>HITL on 7 patches",
    "bootstrap_models_round4-0":"Bootstrap Round 4 — Finetuned<br>on all 11, Data Permissive",
    "bootstrap_models_round4-1":"Bootstrap Round 4 — Finetuned<br>on all 11, Data Conservative",
    "500epoch_scratch_lenient_CPcyto3_Finetuned":
        "Cellpose Cyto3 — Finetuned<br>on all 11, Data Permissive",
    "500epoch_scratch_lenient_CPcyto3_Finetuned-HLR":
        "Cellpose Cyto3 — Finetuned<br>on all 11 (HLR), Data Permissive",
    "500epoch_scratch_7patch_lenient_rescale_True":
        "Scratch — Trained on 7 Patches<br>Data Permissive",
    "500epoch_scratch_7patch_strict_rescale_True":
        "Scratch — Trained on 7 Patches<br>Data Conservative",
    "500epoch_scratch_lenient_rescale_True":
        "Scratch — Trained on all 11<br>Data Permissive",
    "500epoch_scratch_strict_rescale_True":
        "Scratch — Trained on all 11<br>Data Conservative",
}
SUPP_LABELS = {k: _clean(v) for k, v in SUPP_LABELS_RAW.items()}
SUPP_ORDER = list(SUPP_LABELS.keys())


# ── Stats ──────────────────────────────────────────────────────────────
def box_stats(values: np.ndarray) -> dict:
    """Tukey box plot summary statistics for a 1-D array of values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return dict(
            n=0, mean=np.nan, std=np.nan, min=np.nan,
            q1=np.nan, median=np.nan, q3=np.nan, max=np.nan,
            iqr=np.nan, whisker_low=np.nan, whisker_high=np.nan,
            n_outliers_low=0, n_outliers_high=0,
        )

    q1, median, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    inside = v[(v >= fence_low) & (v <= fence_high)]
    whisker_low = inside.min() if inside.size else np.nan
    whisker_high = inside.max() if inside.size else np.nan
    n_out_low = int((v < fence_low).sum())
    n_out_high = int((v > fence_high).sum())

    return dict(
        n=int(n),
        mean=float(v.mean()),
        std=float(v.std(ddof=1)) if n > 1 else 0.0,
        min=float(v.min()),
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        max=float(v.max()),
        iqr=float(iqr),
        whisker_low=float(whisker_low),
        whisker_high=float(whisker_high),
        n_outliers_low=n_out_low,
        n_outliers_high=n_out_high,
    )


def stats_table(model_order: list[str], label_map: dict | None) -> pd.DataFrame:
    """Build the long-format stats table for a given model ordering."""
    group_filters = {
        "All": df,
        "Development": df[df["patch_type"] == "Development"],
        "Test": df[df["patch_type"] == "Test"],
    }
    rows = []
    for mid in model_order:
        label = label_map[mid] if label_map and mid in label_map else mid
        for group_name, sub in group_filters.items():
            vals = sub[sub["model_id"] == mid][METRIC].values
            rows.append({
                "model_id": mid,
                "display_label": label,
                "group": group_name,
                **box_stats(vals),
            })
    out = pd.DataFrame(rows)
    col_order = [
        "model_id", "display_label", "group", "n", "mean", "std",
        "min", "q1", "median", "q3", "max",
        "iqr", "whisker_low", "whisker_high",
        "n_outliers_low", "n_outliers_high",
    ]
    out = out[col_order]
    num_cols = ["mean", "std", "min", "q1", "median", "q3", "max",
                "iqr", "whisker_low", "whisker_high"]
    out[num_cols] = out[num_cols].round(4)
    return out


# ── Build the three tables ─────────────────────────────────────────────
all_models = list(df["model_id"].unique())
# Use supp labels where available, otherwise fall back to model_id
all_label_map = {m: SUPP_LABELS.get(m, m) for m in all_models}

table_all = stats_table(all_models, all_label_map)
table_main = stats_table(MAIN_ORDER, MAIN_LABELS)
table_supp = stats_table(SUPP_ORDER, SUPP_LABELS)

# Sanity check: every figure model exists in the CSVs
missing_main = [m for m in MAIN_ORDER if m not in all_models]
missing_supp = [m for m in SUPP_ORDER if m not in all_models]
if missing_main:
    print(f"[WARN] Main figure models not found in CSVs: {missing_main}")
if missing_supp:
    print(f"[WARN] Supp figure models not found in CSVs: {missing_supp}")

# ── Save ───────────────────────────────────────────────────────────────
out_all = SCRIPT_DIR / "boxplot_stats_all.csv"
out_main = SCRIPT_DIR / "boxplot_stats_main.csv"
out_supp = SCRIPT_DIR / "boxplot_stats_supp.csv"

table_all.to_csv(out_all, index=False)
table_main.to_csv(out_main, index=False)
table_supp.to_csv(out_supp, index=False)

print(f"[OK] Saved: {out_all}  ({len(all_models)} models, {len(table_all)} rows)")
print(f"[OK] Saved: {out_main}  ({len(MAIN_ORDER)} models, {len(table_main)} rows)")
print(f"[OK] Saved: {out_supp}  ({len(SUPP_ORDER)} models, {len(table_supp)} rows)")

# ── Pretty print to stdout ─────────────────────────────────────────────
with pd.option_context("display.max_rows", None,
                       "display.max_columns", None,
                       "display.width", 220,
                       "display.float_format", "{:.4f}".format):
    print()
    print("=== AP@50 box-plot statistics — SUPP figure (11 models) ===")
    print(table_supp.to_string(index=False))
    print()
    print("=== AP@50 box-plot statistics — MAIN figure (6 models) ===")
    print(table_main.to_string(index=False))
