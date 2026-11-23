# Fig. 7 I-L: four-group statistics

The two scripts here test four cell features across the four groups of Fig. 7 I-L. All cells come
from one donor at 13 weeks. The groups are niche clusters of the Fig. 7 run (`hop3_pure`,
resolution 1.5), as in notebook 05:

| Group | Stack | Niche clusters | Cells |
|---|---|---|---|
| front-skin epidermis (surrounding the hair placode) | `(13 wks) Front skin` | N0 | 393 |
| hair placode | `(13 wks) Front skin` | N17 | 276 |
| ridge basal cells | `(13 wks) D3L ventral ridges` | N3, 7, 10, 13, 14 | 1,903 |
| other cells of the ridge stack | `(13 wks) D3L ventral ridges` | the rest of that stack | 1,006 |

The features are tilt (`major_axis_z_angle`, panel I), width (`axis_minor_length`, panel J), size
(`area`, the cell volume, panel K) and true sphericity (`true_sphericity`, panel L). The same
analysis is section 6 of `../notebooks/05_placode_vs_ridge.ipynb` ("Four-group statistics for the
Fig. 7 violins").

## What the scripts compute

`fig7_stats.py` runs the tests and writes the three tables:

- Omnibus test: **Welch's one-way ANOVA**, since the groups differ in size and spread. The plain
  ANOVA F is also given and reaches the same verdict.
- Post hoc test: **Games-Howell**, which compares each pair with Welch's test. Its p-value floors at
  about 1e-13 in scipy, so it is reported as "<1e-12".
- A check that assumes no distribution: **Kruskal-Wallis**, then **Dunn's test, Holm-adjusted**,
  for the six pairs.
- Effect sizes: Hedges' g per pair; omega-squared (ANOVA) and epsilon-squared (Kruskal-Wallis) per
  feature.
- Size (K) is tested on log1p(volume), as plotted. Tilt, width and sphericity are tested raw.
- Stars: * p < 0.05, ** p < 0.01, *** p < 0.001, **** p < 0.0001; ns otherwise.

`fig7_violins_stats.py` draws the four violins of Fig. 7 I-L with bars for the four comparisons the
text makes: each structure against its own surroundings, hair placode against ridge, and the two
surroundings against each other. Each group is a half-violin (a kernel density) with a box
(median and quartiles; whiskers to the furthest cell within 1.5 x IQR). Each panel is 560 x 460 px. Up to 500 cells per group are drawn as
dots at 75 % opacity; the boxes and densities use all cells. The density line is 2 px wide, there is
no grid, and the frame is open. The densities stop at the data. The y axis stops at the data or at
the physical bound (tilt 0-90, sphericity 0.5-1), and the bars sit above it. The number of cells is
written under each group. The size axis is titled "Cell volume (µm³)"; its ticks read actual volumes
on a log-spaced axis.

## Result (Games-Howell; Dunn-Holm agrees on 23 of 24 pairs)

| Panel | placode vs front-skin | ridge vs ridge others | placode vs ridge | front-skin vs ridge others |
|---|---|---|---|---|
| I tilt | **** | **** | ns (p = 0.052; Dunn-Holm p = 0.039, *) | ns |
| J width | **** | ns | **** | **** |
| K size | **** | **** | **** | **** |
| L sphericity | ** (Dunn-Holm *) | **** | **** | **** |

All four omnibus tests give p <= 1.5e-150. Both structures tilt away from their surroundings, and the
two surroundings do not differ in tilt. Only the hair placode is narrower than its surroundings.
Placode cells are larger than ridge cells, and the surroundings differ in the same way. Ridge cells
are rounder than placode cells.

Cells are the replicates and all four groups come from one donor, so the p-values describe this
donor and these samples, not donors in general. With hundreds to thousands of cells per group nearly
every difference is significant, which is why notebook 05 also shows the distributions and the
effect sizes.

## Inputs

The two pickles written by notebooks 01 and 02: `morphospace_results.pkl` and `niche_results.pkl`
in `data/fig6_7/run/` (or in `fig6_7/run/` under the folder named by `EPICYCLE_ANALYSES_DATA`).

## Outputs

| File | What it holds |
|---|---|
| `Fig7_IL_omnibus_tests.csv` | one row per feature: Welch ANOVA (F, df, p), plain ANOVA, omega-squared, Kruskal-Wallis (H, p), epsilon-squared |
| `Fig7_IL_pairwise_posthoc.csv` | one row per feature and pair (6 pairs): means, medians, Hedges' g, Games-Howell t / df / p, Dunn z / raw p / Holm p, stars for both, `tests_agree` |
| `Fig7_IL_group_descriptives.csv` | n, mean, SD, median and IQR per group (size on the log1p scale; `median_raw` is the volume in µm³; width in µm) |
| `figures/Fig7{I,J,K,L}_*_stats.svg` and `.png` | the four violins; `figures/Fig7_significance_key` explains the stars |
| `figures/*_allpairs.*` | the same with all six pairs (with `--all-pairs`) |

The three tables are included in this folder. `figures/` is written when `fig7_violins_stats.py` runs.

## How to run

From this folder, in the environment of `environment.yml`:

```
python -X utf8 fig7_stats.py
python -X utf8 fig7_violins_stats.py
```

Options of `fig7_violins_stats.py`: `--all-pairs` draws all six pairs, `--no-labels` hides the group
names under the x axis, `--no-axis-lines` drops the axis lines, `--stars dunn_holm_stars` uses the
Dunn-Holm stars, `--features col ...` draws other features, and `--width` / `--height` set the panel
size in px.

`fig7_stats.py` needs numpy, pandas and scipy. `fig7_violins_stats.py` also needs plotly, and kaleido
to save the images; it uses the plotting helpers of `nb05_helpers.py`, shared with notebook 05.
