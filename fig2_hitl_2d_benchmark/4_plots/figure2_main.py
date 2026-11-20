"""
EpiCycle Figure 2 — Main: Truncated cross-evaluation (Rounds 0, 4, 9).
Left: Heatmap (median AP@50, selected model_round x all dataset_rounds)
Right: Distribution on final dataset — violin + boxplot + jittered points

UPDATED (n-range): the distribution-panel "n" annotation now shows the per-round
slice-count RANGE across all 10 rounds, since empty-slice exclusion is per model
so n varies by round (lateral). The plotted values are unchanged — only the n label.
"""

from pathlib import Path
import os

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- Font hierarchy ----------------------------------------------------
FONT_FAMILY = "Helvetica Neue, Arial"
FONT_SUBPLOT_TITLE = 13
FONT_AXIS_TITLE = 12
FONT_TICK = 10
FONT_HEATMAP_CELL = 10

# -- Paths -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# The deposit folders are downloaded into <repo>/data/ (override with the EPICYCLE_ANALYSES_DATA variable).
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("EPICYCLE_ANALYSES_DATA", REPO_ROOT / "data"))
DATA_DIR = DATA_ROOT / "fig2" / "cross_round_metrics"  # filtered CSVs written by 3_metrics/process_2D_cross_round_metrics.py

# -- Load filtered data ------------------------------------------------
df_sp = pd.read_csv(DATA_DIR / "spatial_cross_round_metrics_filtered.csv")
df_ax = pd.read_csv(DATA_DIR / "axial_cross_round_metrics_filtered.csv")

SELECTED_ROUNDS = [0, 4, 9]

# -- Heatmap data (selected rows, all columns) ------------------------
heat_sp_full = (df_sp.groupby(["model_round", "eval_round"])["ap_50"]
                .median().unstack())
heat_ax_full = (df_ax.groupby(["model_round", "eval_round"])["ap_50"]
                .median().unstack())

heat_sp = heat_sp_full.loc[SELECTED_ROUNDS].values
heat_ax = heat_ax_full.loc[SELECTED_ROUNDS].values

colorscale = [
    [0.0, "#eff3ff"],
    [0.25, "#bdd7e7"],
    [0.5, "#6baed6"],
    [0.75, "#3182bd"],
    [1.0, "#08519c"],
]

# -- Build figure ------------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    column_widths=[0.62, 0.38],
    row_heights=[0.50, 0.50],
    horizontal_spacing=0.10,
    vertical_spacing=0.18,
    subplot_titles=[
        "Cross-round evaluation - Lateral (XY)",
        "Distribution on final dataset - Lateral (XY)",
        "Cross-round evaluation - Axial (XZ/YZ)",
        "Distribution on final dataset - Axial (XZ/YZ)",
    ],
)

def add_heatmap(fig, data, row, col):
    zmin_local = data.min()
    zmax_local = data.max()

    fig.add_trace(go.Heatmap(
        z=data,
        x=list(range(10)),
        y=list(range(len(SELECTED_ROUNDS))),
        zmin=zmin_local, zmax=zmax_local,
        colorscale=colorscale,
        showscale=(row == 1 and col == 1),
        colorbar=dict(
            orientation="h",
            x=0.28, xanchor="center",
            y=0.50, yanchor="middle",
            len=0.40, thickness=18,
            title=dict(text=""),
            tickvals=[], ticks="", outlinewidth=0,
        ) if (row == 1 and col == 1) else None,
        hovertemplate=(
            "Model Round %{y}<br>"
            "Dataset Round %{x}<br>"
            "Median AP@50: %{z:.3f}<extra></extra>"
        ),
    ), row=row, col=col)

    norm = (data - zmin_local) / (zmax_local - zmin_local) if zmax_local > zmin_local else np.full_like(data, 0.5)
    ax_idx = 1 if (row == 1 and col == 1) else 3
    for yi in range(data.shape[0]):
        for xi in range(data.shape[1]):
            val = data[yi, xi]
            n = norm[yi, xi]
            color = "white" if n > 0.6 else "#333"
            fig.add_annotation(
                text=f"{val:.2f}",
                x=xi, y=yi,
                xref=f"x{ax_idx}", yref=f"y{ax_idx}",
                showarrow=False,
                font=dict(size=FONT_HEATMAP_CELL, color=color, family=FONT_FAMILY),
            )

    fig.update_xaxes(
        tickvals=list(range(10)),
        ticktext=[str(i) for i in range(10)],
        tickfont=dict(size=FONT_TICK),
        title_text="Dataset round" if row == 2 else None,
        title_font=dict(size=FONT_AXIS_TITLE),
        row=row, col=col,
    )
    fig.update_yaxes(
        tickvals=list(range(len(SELECTED_ROUNDS))),
        ticktext=[f"Round {r}" for r in SELECTED_ROUNDS],
        tickfont=dict(size=FONT_TICK),
        title_text="Model",
        title_font=dict(size=FONT_AXIS_TITLE),
        autorange="reversed",
        row=row, col=col,
    )

add_heatmap(fig, heat_sp, row=1, col=1)
add_heatmap(fig, heat_ax, row=2, col=1)

# -- Distribution plots: violin + box + jitter -------------------------
legend_added = {"median": False, "baseline": False}

def add_distribution(fig, df, row, col, baseline_median):
    final = df[df["eval_round"] == 9]

    for i, r in enumerate(SELECTED_ROUNDS):
        sub = final[final["model_round"] == r]["ap_50"]

        fig.add_trace(go.Violin(
            y=sub, x0=i,
            side="positive", width=0.6,
            line=dict(color="rgba(0, 119, 187, 0.4)", width=1),
            fillcolor="rgba(0, 119, 187, 0.08)",
            meanline_visible=False,
            showlegend=False, hoverinfo="skip", points=False,
            scalemode="width",
        ), row=row, col=col)

        fig.add_trace(go.Box(
            y=sub, x0=i, orientation="v",
            boxpoints=False, width=0.3, whiskerwidth=0.4,
            line=dict(color="#666666", width=1.0),
            fillcolor="rgba(100,100,100,0.04)",
            showlegend=False, hoverinfo="skip",
            marker=dict(opacity=0),
        ), row=row, col=col)

        if len(sub) > 200:
            sub_sample = sub.sample(200, random_state=42)
        else:
            sub_sample = sub
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(sub_sample))
        fig.add_trace(go.Scatter(
            x=i + jitter - 0.15, y=sub_sample.values,
            mode="markers",
            marker=dict(size=3.5, color="#0077BB", opacity=0.3),
            showlegend=False,
            hovertemplate=f"Round {r}<br>AP@50: %{{y:.3f}}<extra></extra>",
        ), row=row, col=col)

    medians = [final[final["model_round"] == r]["ap_50"].median() for r in SELECTED_ROUNDS]
    show_median_legend = not legend_added["median"]
    fig.add_trace(go.Scatter(
        x=list(range(len(SELECTED_ROUNDS))), y=medians,
        mode="lines+markers",
        line=dict(color="#CC3311", width=1.5, dash="dash"),
        marker=dict(symbol="circle-open", size=7, color="#CC3311",
                    line=dict(width=1.5, color="#CC3311")),
        showlegend=show_median_legend,
        name="Median AP@50",
        legendgroup="median",
        hovertemplate="Round %{x}<br>Median AP@50: %{y:.3f}<extra></extra>",
    ), row=row, col=col)
    legend_added["median"] = True

    yref_idx = 2 if row == 1 else 4
    fig.add_shape(
        type="line",
        xref=f"x{yref_idx}", yref=f"y{yref_idx}",
        x0=-0.5, x1=len(SELECTED_ROUNDS) - 0.5,
        y0=baseline_median, y1=baseline_median,
        line=dict(color="#999", width=1, dash="dashdot"),
    )
    if not legend_added["baseline"]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="#999", width=1, dash="dashdot"),
            showlegend=True,
            name="Round 0 baseline",
            legendgroup="baseline",
        ), row=row, col=col)
        legend_added["baseline"] = True

    fig.update_xaxes(
        tickvals=list(range(len(SELECTED_ROUNDS))),
        ticktext=[f"Round {r}" for r in SELECTED_ROUNDS],
        tickfont=dict(size=FONT_TICK),
        range=[-0.45, len(SELECTED_ROUNDS) - 0.55],
        showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.5,
        title_text="Model" if row == 2 else None,
        title_font=dict(size=FONT_AXIS_TITLE),
        row=row, col=col,
    )
    fig.update_yaxes(
        title_text="AP@50",
        title_font=dict(size=FONT_AXIS_TITLE),
        range=[-0.02, 1.05], dtick=0.2,
        gridcolor="rgba(200,200,200,0.3)", gridwidth=0.5,
        zeroline=False, tickfont=dict(size=FONT_TICK),
        row=row, col=col,
    )

    # -- n annotation: RANGE across ALL rounds (0-9) ----------------------
    # Empty-slice exclusion (dice=0 & ap=0) is per model, so the retained slice
    # count varies by round. Report the full per-model range across the benchmark
    # (all 10 rounds) so the MAIN and SUPP figures — and the methods text — all show
    # the same n. The plotted rounds (0/4/9) fall within this range. Plotted values
    # are unchanged.
    xref_idx = 2 if row == 1 else 4
    yref_idx = xref_idx
    n_by_round = [len(final[final["model_round"] == r]) for r in range(10)]
    n_min, n_max = min(n_by_round), max(n_by_round)
    n_text = f"n = {n_min:,} slices" if n_min == n_max else f"n = {n_min:,}–{n_max:,} slices"
    print(f"  {'lateral' if row == 1 else 'axial'} n by round 0-9: {n_by_round}  ->  '{n_text}'")
    fig.add_annotation(
        text=n_text,
        x=0.98, y=0.03,
        xref=f"x{xref_idx} domain", yref=f"y{yref_idx} domain",
        showarrow=False,
        font=dict(size=12, color="#555", family=FONT_FAMILY),
        xanchor="right", yanchor="bottom",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(200,200,200,0.5)",
        borderwidth=1,
        borderpad=4,
    )

sp_baseline = df_sp[(df_sp["model_round"] == 0) & (df_sp["eval_round"] == 9)]["ap_50"].median()
ax_baseline = df_ax[(df_ax["model_round"] == 0) & (df_ax["eval_round"] == 9)]["ap_50"].median()

add_distribution(fig, df_sp, row=1, col=2, baseline_median=sp_baseline)
add_distribution(fig, df_ax, row=2, col=2, baseline_median=ax_baseline)

# -- Layout ------------------------------------------------------------
fig.update_layout(
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    width=1100,
    height=650,
    margin=dict(l=60, r=20, t=50, b=70),
    font=dict(family=FONT_FAMILY),
    legend=dict(
        orientation="h",
        yanchor="middle",
        y=0.50,
        xanchor="center",
        x=0.82,
        font=dict(size=11, family=FONT_FAMILY),
        bgcolor="rgba(255,255,255,0.8)",
        borderwidth=0,
        itemwidth=40,
    ),
)

for ann in list(fig.layout.annotations)[:4]:
    ann.font = dict(size=FONT_SUBPLOT_TITLE, family=FONT_FAMILY)

OUT_BASE = SCRIPT_DIR / "figure2_main"
fig.write_html(f"{OUT_BASE}.html", include_plotlyjs="cdn")
print(f"Saved: {OUT_BASE.name}.html")

try:
    fig.write_image(f"{OUT_BASE}.svg")
    fig.write_image(f"{OUT_BASE}.pdf")
    fig.write_image(f"{OUT_BASE}.png", scale=3)
    print("Saved: svg + pdf + png")
except Exception as e:
    print(f"(static export skipped - kaleido unavailable: {type(e).__name__}: {e}; HTML saved)")
