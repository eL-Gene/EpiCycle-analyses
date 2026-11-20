"""
EpiCycle 3D Reconstruction Benchmark — Supplementary Figure
All 11 model conditions, VERTICAL boxplots, category grouping.
90° rotated x-labels.

Usage:  python epicycle_benchmark_supp_vertical.py
Requires: 3DRecon_metrics_BootStrapped.csv, 3DRecon_Metrics_Finetuned.csv
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import numpy as np

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
    "bootstrap_models_round4-0","bootstrap_models_round4-1",
    "500epoch_scratch_lenient_CPcyto3_Finetuned",
    "500epoch_scratch_lenient_CPcyto3_Finetuned-HLR",
    "500epoch_scratch_lenient_rescale_True",
    "500epoch_scratch_strict_rescale_True",
}

MK_SIZE = 9
MK_LINE_W = 1.2   # circle-open stroke
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


# ── Model config ──────────────────────────────────────────────────────
MODEL_ORDER = [
    "bootstrap_models0",
    "bootstrap_models1",
    "bootstrap_models2",
    "bootstrap_models_round4-0",
    "bootstrap_models_round4-1",
    "500epoch_scratch_lenient_CPcyto3_Finetuned",
    "500epoch_scratch_lenient_CPcyto3_Finetuned-HLR",
    "500epoch_scratch_7patch_lenient_rescale_True",
    "500epoch_scratch_7patch_strict_rescale_True",
    "500epoch_scratch_lenient_rescale_True",
    "500epoch_scratch_strict_rescale_True",
]

# Single-line labels for 90° rotation
LABELS = {
    "bootstrap_models0":        "Bootstrap Round 0 (HITL)",
    "bootstrap_models1":        "Bootstrap Round 4 (HITL)",
    "bootstrap_models2":        "Bootstrap Round 9 (HITL)",
    "bootstrap_models_round4-0":"Bootstrap Round 4 FT All 11 (Permissive)",
    "bootstrap_models_round4-1":"Bootstrap Round 4 FT All 11 (Conservative)",
    "500epoch_scratch_lenient_CPcyto3_Finetuned":
        "CP Cyto3 FT All 11 (Permissive)",
    "500epoch_scratch_lenient_CPcyto3_Finetuned-HLR":
        "CP Cyto3 FT All 11 HLR (Permissive)",
    "500epoch_scratch_7patch_lenient_rescale_True":
        "Scratch 7-Patch (Permissive)",
    "500epoch_scratch_7patch_strict_rescale_True":
        "Scratch 7-Patch (Conservative)",
    "500epoch_scratch_lenient_rescale_True":
        "Scratch All 11 (Permissive)",
    "500epoch_scratch_strict_rescale_True":
        "Scratch All 11 (Conservative)",
}

CATEGORY_COLOR = {}
for m in MODEL_ORDER:
    if "CPcyto3" in m:       CATEGORY_COLOR[m] = "#C75A3A"
    elif "round4" in m:      CATEGORY_COLOR[m] = "#5B8DBE"
    elif "bootstrap" in m:   CATEGORY_COLOR[m] = "#666666"
    elif "7patch" in m:      CATEGORY_COLOR[m] = "#E8915A"
    else:                    CATEGORY_COLOR[m] = "#6FB57A"

# Category groups: (label, model indices, color)
GROUPS = [
    ("Bootstrap (HITL)",      [0, 1, 2],   "#666666"),
    ("Bootstrap (Finetuned)", [3, 4],       "#5B8DBE"),
    ("Cellpose Cyto3",        [5, 6],       "#C75A3A"),
    ("Scratch 7-patch",       [7, 8],       "#E8915A"),
    ("Scratch 11-patch",      [9, 10],      "#6FB57A"),
]

JITTER = {p: (i / (len(ALL_PATCHES) - 1) - 0.5) * 0.38
           for i, p in enumerate(ALL_PATCHES)}

# ── Build figure ──────────────────────────────────────────────────────
fig = go.Figure()

n = len(MODEL_ORDER)
xpos = {m: i for i, m in enumerate(MODEL_ORDER)}


# ── Background highlight bands for "trained on all 11" models ─────────
HIGHLIGHT_COLOR = "rgba(0, 0, 0, 0.04)"
all11_indices = sorted([xpos[m] for m in MODEL_ORDER if m in TRAINED_ON_ALL_11])
runs = []
for idx in all11_indices:
    if runs and idx == runs[-1][-1] + 1:
        runs[-1].append(idx)
    else:
        runs.append([idx])
for run in runs:
    fig.add_shape(
        type="rect", xref="x", yref="paper",
        x0=run[0] - 0.5, x1=run[-1] + 0.5,
        y0=0, y1=1,
        fillcolor=HIGHLIGHT_COLOR,
        line=dict(width=0),
        layer="below",
    )

# ── Vertical divider lines between groups ─────────────────────────────
divider_positions = []
for gi in range(len(GROUPS) - 1):
    last_in_group = GROUPS[gi][1][-1]
    first_in_next = GROUPS[gi + 1][1][0]
    x_mid = (xpos[MODEL_ORDER[last_in_group]] + xpos[MODEL_ORDER[first_in_next]]) / 2
    divider_positions.append(x_mid)

# Boxes
for mid in MODEL_ORDER:
    mdata = df[df["model_id"] == mid]
    x = xpos[mid]
    cc = CATEGORY_COLOR[mid]
    r, g, b = int(cc[1:3], 16), int(cc[3:5], 16), int(cc[5:7], 16)

    fig.add_trace(go.Box(
        y=mdata["ap50"], x0=x, orientation="v",
        boxpoints=False, width=0.5, whiskerwidth=0.4,
        line=dict(color=cc, width=0.7),
        fillcolor=f"rgba({r},{g},{b},0.06)",
        showlegend=False, hoverinfo="skip",
        marker=dict(opacity=0),
    ))

# Scatter
for mid in MODEL_ORDER:
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
        ))

# ── Divider lines (vertical) ─────────────────────────────────────────
for x_div in divider_positions:
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=x_div, x1=x_div,
        y0=0, y1=1,
        line=dict(color="#ddd", width=0.6, dash="dot"),
    )

# ── Category bracket labels (bottom, below rotated tick labels) ───────
for label, indices, color in GROUPS:
    x_vals = [xpos[MODEL_ORDER[i]] for i in indices]
    mid_x = np.mean(x_vals)
    fig.add_annotation(
        text=f'<b style="color:{color}">{label}</b>',
        xref="x", yref="paper",
        x=mid_x, y=-0.52,
        showarrow=False, yanchor="top",
        font=dict(size=10, family="Helvetica Neue, Arial"),
    )

# ── X-axis — 90° rotated labels ──────────────────────────────────────
fig.update_xaxes(
    tickvals=[xpos[m] for m in MODEL_ORDER],
    ticktext=[LABELS[m] for m in MODEL_ORDER],
    tickfont=dict(size=9, family="Helvetica Neue, Arial"),
    tickangle=90,
    range=[-0.6, n - 0.4],
    showgrid=True,
    gridcolor="rgba(200,200,200,0.3)",
    gridwidth=0.3,
)

# ── Y-axis ────────────────────────────────────────────────────────────
fig.update_yaxes(
    title_text="AP@50",
    title_font=dict(size=14, family="Helvetica Neue, Arial"),
    range=[-0.02, 1.05], dtick=0.2,
    gridcolor="rgba(200,200,200,0.3)", gridwidth=0.3,
    zeroline=False, tickfont=dict(size=11),
)

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
    ))

for pname in TEST_PATCHES:
    c = PATCH_COLORS[pname]
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="x-thin", size=MK_SIZE, color=c,
                    line=dict(width=MK_X_W, color=c)),
        name=pname,
        legendgroup="stacks",
        showlegend=True,
    ))

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
    title=dict(
        text="<b>Supplementary:</b> 3D Reconstruction Benchmark"
             " — All Model Comparisons",
        font=dict(size=14, family="Helvetica Neue, Arial", color="#333"),
        x=0.02,
    ),
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    width=1050,
    height=720,
    margin=dict(l=60, r=20, t=50, b=280),
    legend=dict(
        font=dict(size=10.5, family="Helvetica Neue, Arial"),
        tracegroupgap=8,
        x=1.01, y=0.98,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.9)",
    ),
)

# ── Save ──────────────────────────────────────────────────────────────
html_out = SCRIPT_DIR / "epicycle_benchmark_supp_vertical.html"
svg_out = SCRIPT_DIR / "epicycle_benchmark_supp_vertical.svg"
fig.write_html(str(html_out), include_plotlyjs="cdn")
print(f"[OK] Saved: {html_out}")
try:
    fig.write_image(str(svg_out), width=1050, height=720)
    print(f"[OK] Saved: {svg_out}")
except Exception as e:
    print(f"[WARN] SVG export failed: {e}")
