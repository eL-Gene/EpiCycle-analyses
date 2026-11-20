"""Fig. 7 I-L violins with significance bars (notebook 05's violin_panel, 560 x 460).

Bars: Games-Howell post hoc (see fig7_stats.py), drawn ABOVE the axis (the axis itself stops at the
data / physical bound, so no impossible tick values appear).  Default comparisons are the four the text
makes: each structure vs its own background, placode vs ridge, background vs background.
--all-pairs draws all six.  Densities are clipped to the data range (no mass beyond 90 deg etc).
Group n is written under each label.  Size: positions are log1p(volume), the ticks read the actual volume in µm³ (log axis stated in the legend, not the title).

Run:  python -X utf8 fig7_violins_stats.py [--all-pairs] [--no-labels] [--no-axis-lines]
                                           [--stars dunn_holm_stars] [--features col ...]
"""
import argparse, os
import numpy as np, pandas as pd
import plotly.graph_objects as go
import fig7_stats as fs
import nb05_helpers as H

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")   # next to the scripts

GROUP_PALETTE = {                      # notebook 05 / Fig. 7G
    "hair_placode":       "#AA3377",
    "placode_background": "#E3B5CF",
    "ridge":              "#CC3311",
    "ridge_background":   "#F3B9A9",
}
STYLE = dict(point_opacity=0.75, point_size=3.0, max_points=500,   # up to 500 cells per group as dots, 75 % opaque
             violin_line_width=2.0, show_grid=False, axis_lines=True,
             bar_level=0.055, bar_font=12, show_n=True, width=560, height=460)   # figure size in px
# y axis per feature: (range or None = data range with 3 % padding, tick values or None, title)
AXES = {
    "major_axis_z_angle": ((0, 90), list(range(0, 91, 15)), "Long-axis angle to z (°)"),
    "axis_minor_length":  (None, None, "Cell width, minor axis (µm)"),
    "area":               (None, [50, 100, 200, 500, 1000, 2000], "Cell volume (µm³)"),
    "true_sphericity":    ((0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "True sphericity"),
}
FOUR = [("placode_background", "hair_placode"), ("ridge", "ridge_background"),
        ("hair_placode", "ridge"), ("placode_background", "ridge_background")]
LABELS = {"placode_background": "Front-skin epidermis", "hair_placode": "Hair placode",
          "ridge": "Ridge (basal)", "ridge_background": "Ridge sample, other cells"}   # as on the 19-Oct panels


def pack_levels(comparisons, pos):
    """Assign each comparison a level so bars on one level never overlap. Short bars first."""
    levels, placed = [], []
    for ga, gb in sorted(comparisons, key=lambda p: abs(pos[p[0]] - pos[p[1]])):
        x0, x1 = sorted((pos[ga], pos[gb]))
        for li, lv in enumerate(levels):
            if all(x1 < a or x0 > b for a, b in lv):
                lv.append((x0, x1)); placed.append((ga, gb, x0, x1, li)); break
        else:
            levels.append([(x0, x1)]); placed.append((ga, gb, x0, x1, len(levels) - 1))
    return placed, len(levels)


def add_bars_above(fig, pairs_df, feature, comparisons, group_order, star_col="games_howell_stars",
                   level=STYLE["bar_level"], font_family=H.FONT_FAMILY):
    """Bars in paper coordinates above the plot area; shrinks the y-axis domain to make room."""
    pos = {g: i for i, g in enumerate(group_order)}
    rows = pairs_df[pairs_df.feature == feature].set_index(["group_a", "group_b"])
    placed, n_levels = pack_levels(comparisons, pos)
    top = 1 - (n_levels + 0.3) * level            # axis domain ends here
    fig.update_yaxes(domain=[0, top])
    for ga, gb, x0, x1, li in placed:
        key = (ga, gb) if (ga, gb) in rows.index else (gb, ga)
        txt = rows.loc[key, star_col]
        y = top + level * (li + 0.45); tick = 0.3 * level
        fig.add_shape(type="path", path=f"M {x0} {y - tick} L {x0} {y} L {x1} {y} L {x1} {y - tick}",
                      line=dict(color="#333", width=1), xref="x", yref="paper")
        fig.add_annotation(x=(x0 + x1) / 2, y=y, text=txt, showarrow=False, xref="x", yref="paper",
                           yanchor="bottom", yshift=-1 if txt == "ns" else -5,
                           font=dict(size=STYLE["bar_font"] - 3 if txt == "ns" else STYLE["bar_font"], color="#333", family=font_family))


def style_violin(fig, col, g, group_order, labels, axes=AXES, style=STYLE, font_family=H.FONT_FAMILY):
    """Figure style: densities clipped to the data, axes stopped at the data or physical bound, ticks in raw units, n under each group."""
    rng_, ticks, title = axes.get(col, (None, None, col))
    tf = fs.FEATURES[col][2] if col in fs.FEATURES else None
    vals = np.log1p(g[col]) if tf == "log1p" else g[col]
    fig.update_traces(selector=dict(type="violin"), spanmode="hard", line_width=style["violin_line_width"])
    if rng_ is None:
        lo, hi = float(vals.min()), float(vals.max()); pad = 0.03 * (hi - lo); rng_ = (lo - pad, hi + pad)
    fig.update_yaxes(range=list(rng_), title_text=title)
    if ticks is not None:
        tv = [np.log1p(t) for t in ticks] if tf == "log1p" else ticks
        fig.update_yaxes(tickvals=tv, ticktext=[f"{t:,}" if isinstance(t, int) else str(t) for t in ticks])
    n = g["group"].value_counts()
    text = [labels[x] + (f"<br>n = {n[x]:,}" if style["show_n"] else "") for x in group_order]
    fig.update_xaxes(title_text="", ticktext=text, tickangle=0, showgrid=style["show_grid"])
    fig.update_yaxes(showgrid=style["show_grid"])
    if style["axis_lines"]:   # open frame: left + bottom lines, outward ticks
        fig.update_yaxes(showline=True, linecolor="#333", linewidth=1, ticks="outside", tickcolor="#333", ticklen=4)
        fig.update_xaxes(showline=True, linecolor="#333", linewidth=1, ticks="outside", tickcolor="#333", ticklen=4)
    fig.update_layout(margin=dict(b=70 if style["show_n"] else 50, t=10))


def star_key(test_line="Games-Howell post hoc after Welch's ANOVA", width=300, height=150):
    """Standalone key for the significance bars (a separate figure file, not drawn on the panels)."""
    rows = [("****", "p < 0.0001"), ("***", "p < 0.001"), ("**", "p < 0.01"), ("*", "p < 0.05"), ("ns", "not significant")]
    fig = go.Figure()
    fig.add_annotation(x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                       text=test_line, font=dict(size=11, color="#333", family=H.FONT_FAMILY))
    for i, (sym, txt) in enumerate(rows):
        y = 0.78 - i * 0.17
        fig.add_annotation(x=0.02, y=y, xref="paper", yref="paper", xanchor="left", yanchor="middle", showarrow=False,
                           text=sym, font=dict(size=10 if sym == "ns" else 14, color="#333", family=H.FONT_FAMILY))
        fig.add_annotation(x=0.28, y=y, xref="paper", yref="paper", xanchor="left", yanchor="middle", showarrow=False,
                           text=txt, font=dict(size=11, color="#333", family=H.FONT_FAMILY))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(width=width, height=height, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-pairs", action="store_true")
    ap.add_argument("--no-labels", action="store_true", help="hide the group names + n under the x axis")
    ap.add_argument("--no-axis-lines", action="store_true", help="drop the left + bottom axis lines")
    ap.add_argument("--stars", default="games_howell_stars", choices=["games_howell_stars", "dunn_holm_stars"])
    ap.add_argument("--features", nargs="*", default=list(fs.FEATURES))
    ap.add_argument("--width", type=int, default=STYLE["width"], help="panel width in px (default 560)")
    ap.add_argument("--height", type=int, default=STYLE["height"], help="panel height in px (default 460)")
    a = ap.parse_args()
    if a.no_axis_lines: STYLE["axis_lines"] = False
    g = fs.load_groups()
    _, _, pairs = fs.run(g)
    comparisons = fs.PAIRS if a.all_pairs else FOUR
    suffix = ("_allpairs" if a.all_pairs else "") + ("_nolabels" if a.no_labels else "") \
        + ("_noaxislines" if a.no_axis_lines else "") + ("_dunn" if a.stars.startswith("dunn") else "")         + (f"_{a.width}x{a.height}" if (a.width, a.height) != (STYLE["width"], STYLE["height"]) else "")
    for col in a.features:
        panel, label, tf = fs.FEATURES[col]
        fig = H.violin_panel(g, col, "group", palette=GROUP_PALETTE, group_order=fs.GROUP_ORDER,
                             y_transform=tf, x_title="", include_zero=False, width=a.width, height=a.height,
                             show_n_annotation=False, random_seed=0, point_opacity=STYLE["point_opacity"],
                             point_size=STYLE["point_size"], max_points=STYLE["max_points"])
        style_violin(fig, col, g, fs.GROUP_ORDER, LABELS)
        add_bars_above(fig, pairs, col, comparisons, fs.GROUP_ORDER, star_col=a.stars)
        if a.no_labels:
            fig.update_xaxes(showticklabels=False); fig.update_layout(margin=dict(b=30))
        H.save_fig(fig, f"Fig7{panel}_{col}_stats{suffix}", out_dir=FIG_DIR)
    H.save_fig(star_key(), "Fig7_significance_key" + ("_dunn" if a.stars.startswith("dunn") else ""), out_dir=FIG_DIR)


if __name__ == "__main__":
    main()
