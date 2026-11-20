"""
EpiCycle 3D Reconstruction Benchmark — Main Figure
VERTICAL boxplots with jittered points.
Rotated x-labels, independent y-axes, uniform skinny box widths.

Usage:  python epicycle_benchmark_main_vertical.py
Requires: 3DRecon_metrics_BootStrapped.csv, 3DRecon_Metrics_Finetuned.csv
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "4_metrics"

# ── Load & merge ───────────────────────────────────────────────────────
df_boot = pd.read_csv(DATA_DIR / "3DRecon_metrics_BootStrapped.csv")
df_ft = pd.read_csv(DATA_DIR / "3DRecon_Metrics_Finetuned.csv")
df = pd.concat([df_boot, df_ft], ignore_index=True)
df["patch_type"] = df["patch_path"].apply(
    lambda p: "Test" if ("holdout" in str(p) or "held_out" in str(p)) else "Development"
)

# ── Patch config ───────────────────────────────────────────────────────
DEV_PATCHES = ["patch-7","patch-5","patch-8","patch-3","patch-1","patch-4","patch-2"]
TEST_PATCHES = ["patch-9","patch-14","patch-10","patch-6"]
ALL_PATCHES = DEV_PATCHES + TEST_PATCHES

PATCH_COLORS = {
    "patch-7":"#0077BB","patch-5":"#EE7733","patch-8":"#009988",
    "patch-3":"#CC3311","patch-1":"#EE3377","patch-4":"#33BBEE",
    "patch-2":"#AA3377","patch-9":"#0055AA","patch-14":"#BB5566",
    "patch-10":"#44AA99","patch-6":"#DDCC77",
}

TRAINED_ON_ALL_11 = {
    "bootstrap_models_round4-0",
    "500epoch_scratch_lenient_CPcyto3_Finetuned",
    "500epoch_scratch_lenient_rescale_True",
}

MK_SIZE = 10
MK_LINE_W = 1.3   # circle-open stroke
MK_X_W = 0.6     # x-thin stroke

def get_marker(patch_name):
    """Always O or X based on dev/test — no filled circles."""
    c = PATCH_COLORS[patch_name]
    is_test = patch_name in TEST_PATCHES
    if is_test:
        return dict(symbol="x-thin", size=MK_SIZE, color=c,
                    line=dict(width=MK_X_W, color=c))
    else:
        return dict(symbol="circle-open", size=MK_SIZE, color=c,
                    line=dict(width=MK_LINE_W, color=c))


# ── Panel definitions ──────────────────────────────────────────────────
PANEL_C_ORDER = ["bootstrap_models0","bootstrap_models1","bootstrap_models2"]
PANEL_C_LABELS = {
    "bootstrap_models0": "Bootstrap [Round 0]",
    "bootstrap_models1": "Bootstrap [Round 4]",
    "bootstrap_models2": "Bootstrap [Round 9]",
}

PANEL_E_ORDER = [
    "bootstrap_models2",
    "bootstrap_models_round4-0",
    "500epoch_scratch_lenient_CPcyto3_Finetuned",
    "500epoch_scratch_lenient_rescale_True",
]
PANEL_E_LABELS = {
    "bootstrap_models2":
        "Bootstrap [Round 9]",
    "bootstrap_models_round4-0":
        "Bootstrap [Round 4] + Dense FT (11 Patch)",
    "500epoch_scratch_lenient_CPcyto3_Finetuned":
        "CellPose Cyto3 Dense FT (11 Patch)",
    "500epoch_scratch_lenient_rescale_True":
        "Scratch Dense Training (11 Patch)",
}

JITTER = {p: (i / (len(ALL_PATCHES) - 1) - 0.5) * 0.42
           for i, p in enumerate(ALL_PATCHES)}

# Category colors — matching supplementary figure
CATEGORY_COLOR = {}
for m in PANEL_C_ORDER + PANEL_E_ORDER:
    if "CPcyto3" in m:       CATEGORY_COLOR[m] = "#C75A3A"
    elif "round4" in m:      CATEGORY_COLOR[m] = "#5B8DBE"
    elif "bootstrap" in m:   CATEGORY_COLOR[m] = "#666666"
    elif "7patch" in m:      CATEGORY_COLOR[m] = "#E8915A"
    else:                    CATEGORY_COLOR[m] = "#6FB57A"

# Box width: equalize physical pixel width across panels.
# Left panel: 3 models, Right panel: 4 models
# Give proportional width so density feels similar
LEFT_FRAC, RIGHT_FRAC = 0.46, 0.54
LEFT_SPAN, RIGHT_SPAN = 3.0, 4.0  # number of x-axis data units
BASE_WIDTH = 0.45  # base box width in data units for the left panel
BOX_WIDTH_LEFT = BASE_WIDTH
BOX_WIDTH_RIGHT = BASE_WIDTH * (LEFT_FRAC / LEFT_SPAN) / (RIGHT_FRAC / RIGHT_SPAN)
BOX_WHISKER_W = 0.4

# ── Build figure — independent y-axes ─────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[LEFT_FRAC, RIGHT_FRAC],
    horizontal_spacing=0.07,
    shared_yaxes=False,
)

def add_panel(fig, model_order, label_map, col, box_width):
    """Draw one panel of the figure: a box per model (median and IQR of 3D AP@50 over the 11
    stacks), the per-stack points jittered beside it (circles development, crosses test), and
    a grey band behind the models trained on all 11 stacks. `col` selects the subplot column.
    """
    n = len(model_order)
    xpos = {m: i for i, m in enumerate(model_order)}
    is_panel_e = (col == 2)

    # Background highlight for "all 11" models
    HIGHLIGHT_COLOR = "rgba(0, 0, 0, 0.04)"
    all11_x = sorted([xpos[m] for m in model_order if m in TRAINED_ON_ALL_11])
    runs = []
    for idx in all11_x:
        if runs and idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    for run in runs:
        fig.add_shape(
            type="rect", xref=f"x{col}" if col > 1 else "x", yref=f"y{col}" if col > 1 else "y",
            x0=run[0] - 0.5, x1=run[-1] + 0.5,
            y0=0, y1=1.05,
            fillcolor=HIGHLIGHT_COLOR,
            line=dict(width=0),
            layer="below",
        )

    for mid in model_order:
        mdata = df[df["model_id"] == mid]
        x = xpos[mid]
        cc = CATEGORY_COLOR[mid]
        r, g, b = int(cc[1:3], 16), int(cc[3:5], 16), int(cc[5:7], 16)

        fig.add_trace(go.Box(
            y=mdata["ap50"], x0=x, orientation="v",
            boxpoints=False,
            width=box_width, whiskerwidth=BOX_WHISKER_W,
            line=dict(color=cc, width=0.7),
            fillcolor=f"rgba({r},{g},{b},0.06)",
            showlegend=False, hoverinfo="skip",
            marker=dict(opacity=0),
        ), row=1, col=col)

    for mid in model_order:
        x = xpos[mid]
        for pname in ALL_PATCHES:
            pdata = df[(df["model_id"] == mid) & (df["patch"] == pname)]
            if pdata.empty:
                continue
            mk = get_marker(pname)
            jx = x + JITTER[pname]
            is_test = pname in TEST_PATCHES
            role = "Holdout" if is_test else "Development"

            fig.add_trace(go.Scatter(
                x=[jx], y=pdata["ap50"].values,
                mode="markers", marker=mk, showlegend=False,
                hovertemplate=(
                    f"<b>{pname}</b> ({role})<br>"
                    f"AP@50: %{{y:.3f}}<extra></extra>"
                ),
            ), row=1, col=col)

    fig.update_xaxes(
        tickvals=[xpos[m] for m in model_order],
        ticktext=[label_map[m] for m in model_order],
        tickfont=dict(size=10, family="Helvetica Neue, Arial"),
        tickangle=90,
        range=[-0.5, n - 0.5],
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
        gridwidth=0.3,
        row=1, col=col,
    )

add_panel(fig, PANEL_C_ORDER, PANEL_C_LABELS, col=1, box_width=BOX_WIDTH_LEFT)
add_panel(fig, PANEL_E_ORDER, PANEL_E_LABELS, col=2, box_width=BOX_WIDTH_RIGHT)

# ── Divider in Panel E: separate reference (Bootstrap R9) from dense-trained ──
fig.add_shape(
    type="line", xref="x2", yref="paper",
    x0=0.5, x1=0.5, y0=0, y1=1,
    line=dict(color="#ccc", width=0.6, dash="dot"),
)

# ── Y-axes — both independent ─────────────────────────────────────────
ykw = dict(range=[-0.02, 1.05], dtick=0.2,
           gridcolor="rgba(200,200,200,0.3)", gridwidth=0.3,
           zeroline=False, tickfont=dict(size=11))
fig.update_yaxes(title_text="AP@50",
                 title_font=dict(size=14, family="Helvetica Neue, Arial"),
                 **ykw, row=1, col=1)
fig.update_yaxes(title_text="AP@50",
                 title_font=dict(size=14, family="Helvetica Neue, Arial"),
                 **ykw, row=1, col=2)

# ── Legend ─────────────────────────────────────────────────────────────
for pname in DEV_PATCHES:
    c = PATCH_COLORS[pname]
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle-open", size=MK_SIZE, color=c,
                    line=dict(width=MK_LINE_W, color=c)),
        name=pname,
        legendgroup="stacks",
        legendgrouptitle=dict(
            text="<b>Stacks</b><br>"
                 "<span style='font-size:10px;color:#888'>"
                 "○ = Development  × = Test</span>",
            font=dict(size=11),
        ),
        showlegend=True,
    ), row=1, col=1)

for pname in TEST_PATCHES:
    c = PATCH_COLORS[pname]
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="x-thin", size=MK_SIZE, color=c,
                    line=dict(width=MK_X_W, color=c)),
        name=pname,
        legendgroup="stacks",
        showlegend=True,
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(symbol="square", size=12,
                color="rgba(0, 0, 0, 0.06)",
                line=dict(color="rgba(0, 0, 0, 0.15)", width=0.6)),
    name="  = trained on all 11 patches",
    legendgroup="note",
    showlegend=True,
))

# ── Layout ────────────────────────────────────────────────────────────
fig.update_layout(
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    width=880,
    height=720,
    margin=dict(l=60, r=20, t=30, b=200),
    legend=dict(
        font=dict(size=10.5, family="Helvetica Neue, Arial"),
        tracegroupgap=8,
        x=1.01, y=0.98,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.9)",
    ),
)

# ── Save ──────────────────────────────────────────────────────────────
html_out = SCRIPT_DIR / "epicycle_benchmark_main_vertical.html"
svg_out = SCRIPT_DIR / "epicycle_benchmark_main_vertical.svg"
fig.write_html(str(html_out), include_plotlyjs="cdn")
print(f"[OK] Saved: {html_out}")
try:
    fig.write_image(str(svg_out), width=880, height=620)
    print(f"[OK] Saved: {svg_out}")
except Exception as e:
    print(f"[WARN] SVG export failed: {e}")
