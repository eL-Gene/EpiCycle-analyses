"""
EpiCycle Figure 2 — Cumulative training dataset size per round.
Shows how the evaluation dataset grows as slices are added each round.
Lateral and Axial shown as two lines on one plot.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os

# The deposit folders are downloaded into <repo>/data/ (override with the EPICYCLE_ANALYSES_DATA variable).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("EPICYCLE_ANALYSES_DATA", REPO_ROOT / "data"))
SCRIPT_DIR = Path(__file__).resolve().parent
IN_DIR = DATA_ROOT / "fig2" / "cross_round_metrics"   # raw per-slice metrics from step 2

# ── Font hierarchy (matching main/supp figures) ──────────────────────
FONT_FAMILY = "Helvetica Neue, Arial"
FONT_TITLE = 13
FONT_AXIS_TITLE = 12
FONT_TICK = 10

# ── Load data ─────────────────────────────────────────────────────────
# Use UNFILTERED data — these represent the actual training set sizes
# (filtering is an evaluation-time decision, not training-time)
df_sp = pd.read_csv(IN_DIR / "spatial_cross_round_metrics.csv")
df_ax = pd.read_csv(IN_DIR / "axial_cross_round_metrics.csv")

# ── Compute cumulative dataset size per round ─────────────────────────
lat_sizes = []
ax_sizes = []
for r in range(10):
    # Number of slices in the eval dataset at each round
    # (use a single model_round to avoid counting duplicates)
    n_lat = len(df_sp[(df_sp["eval_round"] == r) & (df_sp["model_round"] == 0)])
    n_ax = len(df_ax[(df_ax["eval_round"] == r) & (df_ax["model_round"] == 0)])
    lat_sizes.append(n_lat)
    ax_sizes.append(n_ax)

rounds = list(range(10))

# ── Build figure ──────────────────────────────────────────────────────
fig = go.Figure()

# Lateral — step line with markers
fig.add_trace(go.Scatter(
    x=rounds, y=lat_sizes,
    mode="lines+markers",
    line=dict(color="#0077BB", width=2, shape="hv"),
    marker=dict(size=7, color="#0077BB", symbol="circle"),
    name="Lateral (XY)",
    hovertemplate="Round %{x}<br>%{y} slices<extra>Lateral</extra>",
))

# Axial — step line with markers
fig.add_trace(go.Scatter(
    x=rounds, y=ax_sizes,
    mode="lines+markers",
    line=dict(color="#CC3311", width=2, shape="hv"),
    marker=dict(size=7, color="#CC3311", symbol="square"),
    name="Axial (XZ/YZ)",
    hovertemplate="Round %{x}<br>%{y} slices<extra>Axial</extra>",
))

# ── Layout ────────────────────────────────────────────────────────────
fig.update_layout(
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    width=500,
    height=350,
    margin=dict(l=60, r=20, t=40, b=60),
    font=dict(family=FONT_FAMILY),
    xaxis=dict(
        title=dict(text="Dataset round", font=dict(size=FONT_AXIS_TITLE)),
        tickvals=rounds,
        ticktext=[str(r) for r in rounds],
        tickfont=dict(size=FONT_TICK),
    ),
    yaxis=dict(
        title=dict(text="Cumulative training slices", font=dict(size=FONT_AXIS_TITLE)),
        tickfont=dict(size=FONT_TICK),
        gridcolor="rgba(200,200,200,0.3)",
        gridwidth=0.5,
        zeroline=True,
        zerolinecolor="rgba(200,200,200,0.5)",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.0,
        font=dict(size=11, family=FONT_FAMILY),
        bgcolor="rgba(255,255,255,0.8)",
        borderwidth=0,
    ),
)

fig.write_html(SCRIPT_DIR / "figure2_cumulative_training.html", include_plotlyjs="cdn")
print("Saved: figure2_cumulative_training.html")

# Print the data for reference
print("\nCumulative training dataset sizes:")
print(f"{'Round':<8} {'Lateral':<10} {'Axial':<10}")
for r in range(10):
    print(f"{r:<8} {lat_sizes[r]:<10} {ax_sizes[r]:<10}")
