FIG_PREFIX = "Fig7_"
OUT_TABLES = "."
import os, pickle, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go

FONT_FAMILY       = "Helvetica Neue, Arial"
FONT_AXIS_TITLE   = 13
FONT_TICK         = 11
FONT_LEGEND       = 10.5
FONT_LEGEND_TITLE = 11
_CLUSTER_BASE = [
    "#0077BB", "#EE7733", "#009988", "#CC3311", "#EE3377",
    "#33BBEE", "#AA3377", "#BB5566", "#44AA99", "#DDCC77",
    "#882255", "#117733", "#88CCEE", "#332288", "#CC6677",
    "#EE9944", "#99DDBB", "#AA4499", "#661100", "#6699CC",
    "#888888", "#D4B483",
]
DONOR_PALETTE = {
    "donor1": "#332288",
    "donor2": "#CC6677",
    "donor3": "#44AA99",
    "donor4": "#88CCEE",
    "donor5": "#DDCC77",
}
MM_PER_INCH = 25.4
SVG_DPI = 72
PNG_DPI = 300
FIG_OUT_DIR = 'figures'
_CAM_CFGS = {
    "iso": dict(pos=(2.8, 2.2, 1.6), up=(0, 0, 1)),
    "xz":  dict(pos=(0.0, -3.2, 0.0), up=(0, 0, 1)),
    "xy":  dict(pos=(0.0, 0.0, 3.2), up=(0, 1, 0)),
    "yz":  dict(pos=(3.2, 0.0, 0.0), up=(0, 0, 1)),
}
SEED_HIGHLIGHT = 0


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _fmt_cluster(v):
    """Format a cluster label: integer-valued floats become int strings.

    Niche cluster columns are stored as float so they can hold NaN for
    cells outside the valid mask. Without this helper, str(1.0) shows as
    "1.0" which looks wrong in tick labels.
    """
    if v is None:
        return "NA"
    try:
        import math as _math
        f = float(v)
        if _math.isnan(f):
            return "NA"
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(v)


def apply_transform(values, method="raw", winsorize_sigma=5.0):
    """
    Feature transform matching the EpiCycle GUI (see
    napari_epicycle/segmentation/feature_stats.py:179-226).
    Preserves NaN.

    method : 'raw' | 'log1p' | 'sqrt' | 'rank' | 'zscore' | 'winsorize'

    Notes
    -----
    - log1p / sqrt shift by min if any finite value is negative so the
      transform is always defined.
    - rank normalizes to (0, 1] via scipy.stats.rankdata(x) / n.
    - winsorize clips to +/- winsorize_sigma std-devs then unscales.
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    finite_mask = np.isfinite(arr)
    if not finite_mask.any() or method == "raw":
        return arr
    finite = arr[finite_mask]

    if method in ("log", "log1p"):
        if finite.min() < 0:
            arr[finite_mask] = arr[finite_mask] - finite.min()
            finite = arr[finite_mask]
        arr[finite_mask] = np.log1p(finite)
        return arr

    if method == "sqrt":
        if finite.min() < 0:
            arr[finite_mask] = arr[finite_mask] - finite.min()
            finite = arr[finite_mask]
        arr[finite_mask] = np.sqrt(finite)
        return arr

    if method == "rank":
        from scipy.stats import rankdata
        arr[finite_mask] = rankdata(finite) / len(finite)
        return arr

    if method == "zscore":
        if len(finite) > 1:
            mu = np.mean(finite)
            std = np.std(finite, ddof=1)
            if std > 0:
                arr[finite_mask] = (finite - mu) / std
        return arr

    if method == "winsorize":
        if len(finite) > 1:
            mu = np.mean(finite)
            std = np.std(finite, ddof=1)
            if std > 0:
                z = (finite - mu) / std
                z = np.clip(z, -winsorize_sigma, winsorize_sigma)
                arr[finite_mask] = z * std + mu
        return arr

    raise ValueError(f"Unknown transform method: {method!r}")


def _base_layout(width, height, legend_title=None, show_legend=True):
    return dict(
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        width=width, height=height,
        margin=dict(l=65, r=180, t=30, b=60),  # r=180 so external legends don't clip in SVG
        font=dict(family=FONT_FAMILY),
        showlegend=show_legend,
        legend=dict(
            font=dict(size=FONT_LEGEND, family=FONT_FAMILY),
            title=dict(
                text=legend_title or "",
                font=dict(size=FONT_LEGEND_TITLE, family=FONT_FAMILY),
            ),
            tracegroupgap=4,
            x=1.01, y=0.98, xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            itemsizing="constant",
        ),
    )


def _apply_umap_axes(fig, x_title="UMAP 1", y_title="UMAP 2"):
    axkw = dict(
        showgrid=True, gridcolor="rgba(200,200,200,0.25)", gridwidth=0.5,
        zeroline=False,
        showticklabels=False, ticks="",
        showline=False,
        title_font=dict(size=FONT_AXIS_TITLE, family=FONT_FAMILY),
    )
    fig.update_xaxes(title_text=x_title, **axkw)
    fig.update_yaxes(title_text=y_title, **axkw)


def violin_panel(df, feature, group_col, show_legend=True,
                 palette=None, group_order=None,
                 show_box=True, show_points=True, max_points=200,
                 show_median_line=False, baseline=None,
                 violin_width=0.6, box_width=0.3, jitter_frac=0.12,
                 violin_fill_alpha=0.08, violin_line_alpha=0.4,
                 point_size=3.5, point_opacity=0.32,
                 y_transform=None,
                 width=760, height=500,
                 y_title=None, x_title=None,
                 tick_rotation=0,
                 show_n_annotation=True,
                 include_zero=True,
                 random_seed=42):
    """
    Single-feature violin across the categories in `group_col`.
    Aesthetic: half-violin + overlaid narrow box + jittered scatter (side-offset),
    optional median trend line + baseline reference.

    Parameters
    ----------
    df : DataFrame
    feature : str  — numeric column to plot on y
    group_col : str  — categorical column; one violin per value
    palette : dict {group_value: hex_color} or None (auto)
    group_order : list of group values in display order (default: sorted)
    show_box / show_points : toggle overlay elements
    max_points : subsample threshold for the jitter scatter
    show_median_line : connect group medians (dashed red) — useful for
                      ordered groups like timepoint, off for Leiden clusters
    baseline : optional horizontal reference line (y-value)
    violin_width / box_width / jitter_frac : geometry controls
    violin_fill_alpha / violin_line_alpha / point_opacity : visual weights
    y_transform : None | 'raw' | 'log1p' | 'sqrt' | 'rank' | 'zscore' | 'winsorize'
        Matches the EpiCycle GUI transform set (feature_stats.apply_transform).
    tick_rotation : degrees on x labels (useful for sample names)
    """
    import numpy as _np
    d = df[[feature, group_col]].dropna().copy()

    if y_transform and y_transform != "raw":
        d[feature] = apply_transform(d[feature].values, method=y_transform)

    if group_order is None:
        try:
            group_order = sorted(d[group_col].unique(),
                                 key=lambda x: float(x))
        except (TypeError, ValueError):
            group_order = sorted(d[group_col].unique(), key=str)

    if palette is None:
        palette = {g: _CLUSTER_BASE[i % len(_CLUSTER_BASE)]
                   for i, g in enumerate(group_order)}

    rng = _np.random.default_rng(random_seed)
    fig = go.Figure()
    legend_added_median = False

    for i, g in enumerate(group_order):
        sub = d[d[group_col] == g][feature]
        if sub.empty:
            continue
        c_hex = palette.get(g, "#888888")
        r, gr, b = _hex_to_rgb(c_hex)
        fill_rgba = f"rgba({r},{gr},{b},{violin_fill_alpha:.3f})"
        line_rgba = f"rgba({r},{gr},{b},{violin_line_alpha:.3f})"

        # Half-violin (positive side)
        fig.add_trace(go.Violin(
            y=sub, x0=i,
            side="positive", width=violin_width,
            line=dict(color=line_rgba, width=1),
            fillcolor=fill_rgba,
            meanline_visible=False,
            showlegend=False, hoverinfo="skip", points=False,
            scalemode="width",
        ))

        if show_box:
            fig.add_trace(go.Box(
                y=sub, x0=i, orientation="v",
                boxpoints=False, width=box_width, whiskerwidth=0.4,
                line=dict(color="#666666", width=1.0),
                fillcolor="rgba(100,100,100,0.04)",
                showlegend=False, hoverinfo="skip",
                marker=dict(opacity=0),
            ))

        if show_points:
            pts = sub.sample(max_points, random_state=random_seed) \
                if len(sub) > max_points else sub
            jitter = rng.uniform(-jitter_frac, jitter_frac, size=len(pts))
            fig.add_trace(go.Scatter(
                x=i + jitter - 0.15,
                y=pts.values, mode="markers",
                marker=dict(size=point_size, color=c_hex,
                            opacity=point_opacity, line=dict(width=0)),
                showlegend=False,
                hovertemplate=(
                    f"<b>{g}</b><br>"
                    f"{feature}: %{{y:.3f}}<extra></extra>"
                ),
            ))

    if show_median_line:
        medians = [d[d[group_col] == g][feature].median() for g in group_order]
        fig.add_trace(go.Scatter(
            x=list(range(len(group_order))), y=medians,
            mode="lines+markers",
            line=dict(color="#CC3311", width=1.5, dash="dash"),
            marker=dict(symbol="circle-open", size=7, color="#CC3311",
                        line=dict(width=1.5, color="#CC3311")),
            showlegend=True, name="Median",
            hovertemplate="Group: %{x}<br>Median: %{y:.3f}<extra></extra>",
        ))
        legend_added_median = True

    if baseline is not None:
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(group_order) - 0.5,
            y0=baseline, y1=baseline,
            line=dict(color="#999", width=1, dash="dashdot"),
        )

    fig.update_layout(
        plot_bgcolor="#fff", paper_bgcolor="#fff",
        width=width, height=height,
        margin=dict(l=70, r=20, t=30, b=80 if tick_rotation else 60),
        font=dict(family=FONT_FAMILY),
        showlegend=legend_added_median,
        legend=dict(
            font=dict(size=FONT_LEGEND, family=FONT_FAMILY),
            x=1.01, y=0.98, xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
        ),
    )
    fig.update_xaxes(
        tickvals=list(range(len(group_order))),
        ticktext=[_fmt_cluster(g) for g in group_order],
        title_text=x_title or group_col,
        title_font=dict(size=FONT_AXIS_TITLE, family=FONT_FAMILY),
        tickfont=dict(size=FONT_TICK, family=FONT_FAMILY),
        tickangle=tick_rotation,
        range=[-0.6, len(group_order) - 0.4],
        showgrid=True, gridcolor="rgba(200,200,200,0.3)", gridwidth=0.5,
        zeroline=False,
    )
    _y_suffix = "" if not y_transform or y_transform == "raw" else f" ({y_transform})"
    fig.update_yaxes(
        title_text=y_title or (feature + _y_suffix),
        title_font=dict(size=FONT_AXIS_TITLE, family=FONT_FAMILY),
        tickfont=dict(size=FONT_TICK, family=FONT_FAMILY),
        gridcolor="rgba(200,200,200,0.3)", gridwidth=0.5,
        zeroline=False,
        rangemode="tozero" if include_zero else "normal",
    )

    if show_n_annotation:
        fig.add_annotation(
            text=f"n = {len(d):,}",
            x=0.98, y=0.03, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=11, color="#555", family=FONT_FAMILY),
            xanchor="right", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1, borderpad=4,
        )
    fig.update_layout(showlegend=show_legend)
    return fig


def mm_to_px(mm, dpi=SVG_DPI):
    return int(round(mm * dpi / MM_PER_INCH))


def save_fig(fig, name, *, mm=None, width_px=None, height_px=None,
             out_dir=None, formats=("svg", "png")):
    """Save a Plotly figure as SVG (vector, Illustrator-editable) + high-dpi PNG.

    Sizing:
        mm=(w, h)            -> sized to those mm at 72 dpi (preferred).
        width_px / height_px -> raw pixel size.
        neither              -> uses fig.layout.width / height.

    PNG is rendered at PNG_DPI / SVG_DPI (~4.17x) for crisp previews.
    Requires kaleido:  pip install -U kaleido
    """
    out_dir = out_dir if out_dir is not None else FIG_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    if mm is not None:
        width_px = mm_to_px(mm[0])
        height_px = mm_to_px(mm[1])
    png_scale = PNG_DPI / SVG_DPI

    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{fmt}")
        kw = {}
        if width_px:  kw["width"] = width_px
        if height_px: kw["height"] = height_px
        if fmt == "png":
            kw["scale"] = png_scale
        fig.write_image(path, format=fmt, **kw)
        paths.append(path)
        print(f"  -> {path}")
    return paths


def _axis_titles(x_col, y_col):
    """Axis titles by the notebooks' convention: niche_* columns -> 'Niche UMAP'."""
    def word(c):
        return "Niche UMAP" if c.startswith("niche_") else "UMAP"
    return f"{word(x_col)} 1", f"{word(y_col)} 2"


def umap_highlight(df, highlight_col, highlight_values,
                   x_col="umap_1", y_col="umap_2",
                   palette=None, bg_color="#D9D9D9", bg_opacity=0.25, bg_size=3,
                   fg_size=5, fg_opacity=0.85, mask=None, title=None,
                   legend_title=None, width=900, height=700,
                   on_top=None, names=None, seed=None, show_legend=True):
    """Highlight-vs-grey UMAP.  Highlights chosen groups on a UMAP (one-vs-rest views,
    the within-donor view of claim 3, ridged vs whorl for claim 4, the
    structures of claim 5).

    Every cell is drawn first in light grey; cells whose `highlight_col` is in
    `highlight_values` (and, if given, where the boolean Series `mask` is True)
    are drawn on top in their `palette` colour, in a seeded random order so no
    group is consistently on top (notebook 03's "intermix").
    on_top : values drawn after the others (e.g. small structures over their
             tissue backgrounds); each layer is shuffled on its own.
    names  : {value: legend text}; the legend always adds the cell count.
    Works for shape (umap_1/2) and niche (niche_umap_1/2) coordinates.
    """
    seed = SEED_HIGHLIGHT if seed is None else seed
    palette, names = palette or {}, names or {}
    values = list(highlight_values)
    d = df.dropna(subset=[x_col, y_col])
    eligible = d[highlight_col].isin(values)
    if mask is not None:
        eligible &= mask.astype(bool).reindex(d.index, fill_value=False)

    fig = go.Figure()
    bg = d[~eligible]
    fig.add_trace(go.Scatter(
        x=bg[x_col], y=bg[y_col], mode="markers",
        marker=dict(size=bg_size, color=bg_color, opacity=bg_opacity,
                    line=dict(width=0)),
        name="other cells", showlegend=False, hoverinfo="skip"))

    fg = d[eligible]
    counts = fg[highlight_col].value_counts()
    top = [v for v in (on_top or []) if v in values]
    rng = np.random.default_rng(seed)
    for layer in ([v for v in values if v not in top], top):
        sub = fg[fg[highlight_col].isin(layer)]
        if sub.empty:
            continue
        sub = sub.iloc[rng.permutation(len(sub))]
        grp = sub[highlight_col]
        fig.add_trace(go.Scatter(
            x=sub[x_col].to_numpy(), y=sub[y_col].to_numpy(), mode="markers",
            marker=dict(size=fg_size, opacity=fg_opacity, line=dict(width=0),
                        color=grp.map(lambda v: palette.get(v, "#888")).to_numpy()),
            customdata=grp.map(lambda v: names.get(v, str(v))).to_numpy(),
            hovertemplate="<b>%{customdata}</b><br>%{x:.2f}, %{y:.2f}<extra></extra>",
            name="highlighted", showlegend=False))
    for rank, v in enumerate(values):          # legend-only entries, in order
        n = int(counts.get(v, 0))
        if n == 0:
            print(f"[umap_highlight]: no cells for {highlight_col} = {v!r}")
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=fg_size, color=palette.get(v, "#888"),
                        opacity=fg_opacity, line=dict(width=0)),
            name=f"{names.get(v, str(v))} (n={n:,})", legendrank=rank,
            hoverinfo="skip"))

    fig.update_layout(**_base_layout(width, height,
                                     legend_title=legend_title or f"<b>{highlight_col}</b>",
                                     show_legend=show_legend))
    _apply_umap_axes(fig)
    xt, yt = _axis_titles(x_col, y_col)
    fig.update_xaxes(title_text=xt)
    fig.update_yaxes(title_text=yt)
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left",
                                     font=dict(size=FONT_AXIS_TITLE, family=FONT_FAMILY)),
                          margin=dict(t=55))
    return fig

os.makedirs(OUT_TABLES, exist_ok=True)
print('helpers ready')