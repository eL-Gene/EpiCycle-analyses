"""
EpiCycle Figure 2 — Supplementary: Full 10-round cross-evaluation.
Left: Heatmap (median AP@50, model_round x dataset_round) — shared colorscale
Right: Distribution on final dataset — violin + boxplot + jittered points

UPDATED (n-range): the distribution-panel "n" annotation now shows the per-round
slice-count RANGE across all 10 rounds (empty-slice exclusion is per model, so n
varies by round). Plotted values are unchanged — only the n label.
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
FONT_HEATMAP_CELL = 9

# -- Paths -------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# The deposit folders are downloaded into <repo>/data/ (override with the EPICYCLE_ANALYSES_DATA variable).
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(os.environ.get("EPICYCLE_ANALYSES_DATA", REPO_ROOT / "data"))
DATA_DIR = DATA_ROOT / "fig2" / "cross_round_metrics"  # filtered CSVs written by 3_metrics/process_2D_cross_round_metrics.py

# -- Load filtered data ------------------------------------------------
df_sp = pd.read_csv(DATA_DIR / "spatial_cross_round_metrics_filtered.csv")
df_ax = pd.read_csv(DATA_DIR / "axial_cross_round_metrics_filtered.csv")

# -- Heatmap data ------------------------------------------------------
heat_sp = (df_sp.groupby(["model_round", "eval_round"])["ap_50"]
           .median().unstack().values)
heat_ax = (df_ax.groupby(["model_round", "eval_round"])["ap_50"]
           .median().unstack().values)

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
    column_widths=[0.45, 0.55],
    row_heights=[0.50, 0.50],
    horizontal_spacing=0.08,
    vertical_spacing=0.18,
    subplot_titles=[
        "Cross-round evaluation - Lateral (XY)",
        "Distribution on final dataset - Lateral (XY)",
        "Cross-round evaluation - Axial (XZ/YZ)",
        "Distribution on final dataset - Axial (XZ/YZ)",
    ],
)

def add_heatmap(fig, data, row, col):
    n_rounds = data.shape[0]
    zmin_local = data.min()
    zmax_local = data.max()

    show_bar = (row == 1 and col == 1)

    fig.add_trace(go.Heatmap(
        z=data,
        x=list(range(10)),
        y=list(range(n_rounds)),
        zmin=zmin_local, zmax=zmax_local,
        colorscale=colorscale,
        showscale=show_bar,
        colorbar=dict(
            orientation="h",
            x=0.20, xanchor="center",
            y=0.50, yanchor="middle",
            len=0.35, thickness=22,
            title=dict(text=""),
            tickvals=[], ticks="", outlinewidth=0,
        ) if show_bar else None,
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
        tickvals=list(range(n_rounds)),
        ticktext=[str(i) for i in range(n_rounds)],
        tickfont=dict(size=FONT_TICK),
        title_text="Model round",
        title_font=dict(size=FONT_AXIS_TITLE),
        autorange="reversed",
        row=row, col=col,
    )

add_heatmap(fig, heat_sp, row=1, col=1)
add_heatmap(fig, heat_ax, row=2, col=1)

# -- Distribution plots ------------------------------------------------
legend_added = {"median": False, "baseline": False}

def add_distribution(fig, df, row, col, baseline_median):
    final = df[df["eval_round"] == 9]

    for r in range(10):
        sub = final[final["model_round"] == r]["ap_50"]

        fig.add_trace(go.Violin(
            y=sub, x0=r,
            side="positive", width=0.6,
            line=dict(color="rgba(0, 119, 187, 0.4)", width=1),
            fillcolor="rgba(0, 119, 187, 0.08)",
            meanline_visible=False,
            showlegend=False, hoverinfo="skip", points=False,
            scalemode="width",
        ), row=row, col=col)

        fig.add_trace(go.Box(
            y=sub, x0=r, orientation="v",
            boxpoints=False, width=0.3, whiskerwidth=0.4,
            line=dict(color="#666666", width=1.0),
            fillcolor="rgba(100,100,100,0.04)",
            showlegend=False, hoverinfo="skip",
            marker=dict(opacity=0),
        ), row=row, col=col)

        if len(sub) > 150:
            sub_sample = sub.sample(150, random_state=42)
        else:
            sub_sample = sub
        jitter = np.random.default_rng(42).uniform(-0.10, 0.10, size=len(sub_sample))
        fig.add_trace(go.Scatter(
            x=r + jitter - 0.15, y=sub_sample.values,
            mode="markers",
            marker=dict(size=3, color="#0077BB", opacity=0.25),
            showlegend=False,
            hovertemplate=f"Round {r}<br>AP@50: %{{y:.3f}}<extra></extra>",
        ), row=row, col=col)

    medians = [final[final["model_round"] == r]["ap_50"].median() for r in range(10)]
    show_median_legend = not legend_added["median"]
    fig.add_trace(go.Scatter(
        x=list(range(10)), y=medians,
        mode="lines+markers",
        line=dict(color="#CC3311", width=1.5, dash="dash"),
        marker=dict(symbol="circle-open", size=6, color="#CC3311",
                    line=dict(width=1.5, color="#CC3311")),
        showlegend=show_median_legend,
        name="Median AP@50",
        legendgroup="median",
        hovertemplate="Round %{x}<br>Median AP@50: %{y:.3f}<extra></extra>",
    ), row=row, col=col)
    legend_added["median"] = True

    yref_idx = 2 if row == 1 else 4
    fig.add_shape(
        type="line", xref=f"x{yref_idx}", yref=f"y{yref_idx}",
        x0=-0.5, x1=9.5,
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
        tickvals=list(range(10)),
        ticktext=[f"Round {i}" for i in range(10)],
        tickfont=dict(size=FONT_TICK), tickangle=45,
        range=[-0.6, 9.6],
        showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.5,
        title_text="Model" if row == 2 else None,
        title_font=dict(size=FONT_AXIS_TITLE),
        row=row, col=col,
    )
    fig.update_yaxes(
        title_text="AP@50", title_font=dict(size=FONT_AXIS_TITLE),
        range=[-0.02, 1.05], dtick=0.2,
        gridcolor="rgba(200,200,200,0.3)", gridwidth=0.5,
        zeroline=False, tickfont=dict(size=FONT_TICK),
        row=row, col=col,
    )

    # -- n annotation: RANGE across all 10 rounds -------------------------
    # Empty-slice exclusion (dice=0 & ap=0) is per model, so the retained slice
    # count varies by round. Show the min-max range across all rounds instead of a
    # single (round-0) count. Values plotted are unchanged.
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
    title=dict(
        text="<b>Supplementary:</b> 2D Cross-Round Model Evaluation - All Rounds",
        font=dict(size=14, family=FONT_FAMILY, color="#333"),
        x=0.02,
    ),
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    width=1100,
    height=950,
    margin=dict(l=60, r=20, t=60, b=80),
    font=dict(family=FONT_FAMILY),
    legend=dict(
        orientation="h",
        yanchor="middle",
        y=0.50,
        xanchor="center",
        x=0.77,
        font=dict(size=11, family=FONT_FAMILY),
        bgcolor="rgba(255,255,255,0.8)",
        borderwidth=0,
        itemwidth=40,
    ),
)

for ann in list(fig.layout.annotations)[:4]:
    ann.font = dict(size=FONT_SUBPLOT_TITLE, family=FONT_FAMILY)

OUT_BASE = SCRIPT_DIR / "figure2_supp"
fig.write_html(f"{OUT_BASE}.html", include_plotlyjs="cdn")
print(f"Saved: {OUT_BASE.name}.html")

try:
    fig.write_image(f"{OUT_BASE}.svg")
    fig.write_image(f"{OUT_BASE}.pdf")
    fig.write_image(f"{OUT_BASE}.png", scale=3)
    print("Saved: svg + pdf + png")
except Exception as e:
    print(f"(static export skipped - kaleido unavailable: {type(e).__name__}: {e}; HTML saved)")
