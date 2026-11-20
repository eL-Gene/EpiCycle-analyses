"""Four-group statistics for Fig. 7 I-L (hair placode vs fingerprint ridge, one donor, 13 weeks).

Groups (niche clusters of the Fig. 7 run, hop3_pure r1.5):
  placode_background = N0 in the front-skin stack, hair_placode = N17,
  ridge = N3,7,10,13,14 in the ventral-ridge stack, ridge_background = the rest of that stack.
Features: major_axis_z_angle (tilt), axis_minor_length (width), area (size, log1p as plotted),
  true_sphericity.

Tests per feature
  omnibus : Welch's one-way ANOVA (unequal variances; the plain F is also given) and Kruskal-Wallis
  post hoc: Games-Howell (parametric, pairs with Welch) and Dunn with Holm (pairs with Kruskal-Wallis)
  effect  : Hedges' g per pair, omega-squared / epsilon-squared per feature
Cells are the replicates and there is one donor, so these p-values describe this donor only.

Run:  python -X utf8 fig7_stats.py        (needs numpy, pandas, scipy)
"""
import os, pickle, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

# The pickles of the Fig. 7 run: data/fig6_7/run/ (or $EPICYCLE_ANALYSES_DATA/fig6_7/run/).
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("EPICYCLE_ANALYSES_DATA", REPO_ROOT / "data"))
ANALYSES = str(DATA_ROOT / "fig6_7" / "run")
RIDGE_STACK   = "(13 wks) D3L ventral ridges x40-Smooth05_Resampled.tif"
PLACODE_STACK = "(13 wks) Front skin x40-Smooth05_Resampled.tif"
RIDGE_CLUSTERS, PLACODE_CLUSTERS, PLACODE_BG = [3, 10, 13, 14, 7], [17], [0]
GROUP_ORDER = ["placode_background", "hair_placode", "ridge", "ridge_background"]
EXPECTED_N = [393, 276, 1903, 1006]
FEATURES = {                       # column -> (panel, label, transform)
    "major_axis_z_angle": ("I", "tilt: long-axis angle to z (deg)", None),
    "axis_minor_length":  ("J", "width: minor axis length", None),
    "area":               ("K", "size: cell volume (µm³), log1p", "log1p"),
    "true_sphericity":    ("L", "true sphericity", None),
}
PAIRS = list(itertools.combinations(GROUP_ORDER, 2))


def load_groups():
    with open(os.path.join(ANALYSES, "morphospace_results.pkl"), "rb") as fh:
        m = pickle.load(fh)
    df = m["df"].copy().reset_index(drop=True)
    with open(os.path.join(ANALYSES, "niche_results.pkl"), "rb") as fh:
        n = pickle.load(fh)
    nr = n["niche_results"]["hop3_pure"]
    col = np.full(len(df), np.nan)
    col[np.asarray(nr["valid_mask"])] = np.asarray(nr["clusters"][1.5]["membership"], float)
    df["niche_leiden_r1.5"] = col
    in_p, in_r = df.stack_name == PLACODE_STACK, df.stack_name == RIDGE_STACK
    df["group"] = None
    df.loc[in_p & df["niche_leiden_r1.5"].isin(PLACODE_CLUSTERS), "group"] = "hair_placode"
    df.loc[in_p & df["niche_leiden_r1.5"].isin(PLACODE_BG), "group"] = "placode_background"
    df.loc[in_r & df["niche_leiden_r1.5"].isin(RIDGE_CLUSTERS), "group"] = "ridge"
    df.loc[in_r & df["group"].isna(), "group"] = "ridge_background"
    g = df[df["group"].notna()].copy()
    g["area_log1p"] = np.log1p(g["area"])
    counts = g["group"].value_counts().reindex(GROUP_ORDER)
    assert counts.tolist() == EXPECTED_N, counts.to_dict()
    return g


# ---------- omnibus ----------
def welch_anova(samples):
    """Welch (1951) one-way ANOVA. Returns F, df1, df2, p."""
    k = len(samples)
    n = np.array([len(s) for s in samples], float)
    m = np.array([s.mean() for s in samples])
    v = np.array([s.var(ddof=1) for s in samples])
    w = n / v
    mw = (w * m).sum() / w.sum()
    num = ((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = (3 * ((1 - w / w.sum()) ** 2 / (n - 1)).sum()) / (k * k - 1)
    F = num / (1 + 2 * (k - 2) * lam / 3)
    df1, df2 = k - 1, 1 / lam
    return F, df1, df2, stats.f.sf(F, df1, df2)


def omega_squared(samples):
    allv = np.concatenate(samples); N = len(allv); k = len(samples)
    ss_b = sum(len(s) * (s.mean() - allv.mean()) ** 2 for s in samples)
    ss_w = sum(((s - s.mean()) ** 2).sum() for s in samples)
    ms_w = ss_w / (N - k)
    return (ss_b - (k - 1) * ms_w) / (ss_b + ss_w + ms_w)


# ---------- post hoc ----------
def games_howell(a, b, k):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = stats.studentized_range.sf(abs(t) * np.sqrt(2), k, df)
    return t, df, float(min(p, 1.0))


def dunn(ranks, groups):
    """Dunn z for every pair on the pooled ranks (tie-corrected); returns {pair: (z, p_raw)}."""
    N = len(ranks)
    _, cnt = np.unique(ranks, return_counts=True)
    tie = 1 - (cnt ** 3 - cnt).sum() / (N ** 3 - N)
    out = {}
    for ga, gb in PAIRS:
        ra, rb = ranks[groups == ga], ranks[groups == gb]
        se = np.sqrt((N * (N + 1) / 12) * tie * (1 / len(ra) + 1 / len(rb)))
        z = (ra.mean() - rb.mean()) / se
        out[(ga, gb)] = (z, 2 * stats.norm.sf(abs(z)))
    return out


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(running, 1.0)
    return adj


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    g = (a.mean() - b.mean()) / sp
    return g * (1 - 3 / (4 * (na + nb) - 9))


def stars(p):
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def fmt_p(p):
    return "<1e-300" if p < 1e-300 else f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def run(g):
    desc, omni, pairs = [], [], []
    for col, (panel, label, tf) in FEATURES.items():
        valcol = "area_log1p" if tf == "log1p" else col
        samples = [g.loc[g.group == grp, valcol].dropna().to_numpy(float) for grp in GROUP_ORDER]
        for grp, s in zip(GROUP_ORDER, samples):
            q1, q3 = np.percentile(s, [25, 75])
            desc.append(dict(panel=panel, feature=col, value=valcol, group=grp, n=len(s),
                             mean=s.mean(), sd=s.std(ddof=1), median=np.median(s), q1=q1, q3=q3,
                             median_raw=np.median(np.expm1(s)) if tf else np.median(s)))
        F, d1, d2, pW = welch_anova(samples)
        Fc, pc = stats.f_oneway(*samples)
        H, pH = stats.kruskal(*samples)
        N = sum(len(s) for s in samples); k = len(samples)
        omni.append(dict(panel=panel, feature=col, value=valcol, n_total=N, k=k,
                         welch_F=F, welch_df1=d1, welch_df2=d2, welch_p=pW, welch_stars=stars(pW),
                         anova_F=Fc, anova_df1=k - 1, anova_df2=N - k, anova_p=pc,
                         omega_squared=omega_squared(samples),
                         kruskal_H=H, kruskal_df=k - 1, kruskal_p=pH, kruskal_stars=stars(pH),
                         epsilon_squared=(H - k + 1) / (N - k)))
        sub = g[g.group.isin(GROUP_ORDER)]
        vals = sub[valcol].to_numpy(float); grps = sub["group"].to_numpy()
        dz = dunn(stats.rankdata(vals), grps)
        rows = []
        for ga, gb in PAIRS:
            a = sub.loc[sub.group == ga, valcol].to_numpy(float)
            b = sub.loc[sub.group == gb, valcol].to_numpy(float)
            t, df, pgh = games_howell(a, b, k)
            z, pd_raw = dz[(ga, gb)]
            rows.append(dict(panel=panel, feature=col, value=valcol, group_a=ga, group_b=gb,
                             n_a=len(a), n_b=len(b), mean_a=a.mean(), mean_b=b.mean(),
                             median_a=np.median(a), median_b=np.median(b),
                             mean_diff_a_minus_b=a.mean() - b.mean(), hedges_g=hedges_g(a, b),
                             games_howell_t=t, games_howell_df=df, games_howell_p=pgh,
                             dunn_z=z, dunn_p_raw=pd_raw))
        pr = pd.DataFrame(rows)
        pr["dunn_p_holm"] = holm(pr["dunn_p_raw"])
        # scipy's studentized-range survival function bottoms out near 1e-13; below that report a bound
        pr["games_howell_p_report"] = pr["games_howell_p"].map(lambda p: "<1e-12" if p < 1e-12 else f"{p:.3g}")
        pr["games_howell_stars"] = pr["games_howell_p"].map(stars)
        pr["dunn_holm_stars"] = pr["dunn_p_holm"].map(stars)
        pr["tests_agree"] = pr["games_howell_stars"].eq("ns") == pr["dunn_holm_stars"].eq("ns")
        pairs.append(pr)
    return pd.DataFrame(desc), pd.DataFrame(omni), pd.concat(pairs, ignore_index=True)


if __name__ == "__main__":
    g = load_groups()
    desc, omni, pairs = run(g)
    desc.round(5).to_csv(HERE / "Fig7_IL_group_descriptives.csv", index=False)
    omni.to_csv(HERE / "Fig7_IL_omnibus_tests.csv", index=False)
    pairs.to_csv(HERE / "Fig7_IL_pairwise_posthoc.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(desc.pivot(index=["panel", "feature"], columns="group", values="median_raw").reindex(columns=GROUP_ORDER).round(3))
    print()
    o = omni[["panel", "feature", "welch_F", "welch_df2", "welch_p", "omega_squared", "kruskal_H", "kruskal_p", "epsilon_squared"]].copy()
    for c in ("welch_p", "kruskal_p"): o[c] = o[c].map(fmt_p)
    print(o.round(3).to_string(index=False))
    print()
    p = pairs[["panel", "group_a", "group_b", "mean_diff_a_minus_b", "hedges_g", "games_howell_p", "games_howell_stars", "dunn_p_holm", "dunn_holm_stars", "tests_agree"]].copy()
    for c in ("games_howell_p", "dunn_p_holm"): p[c] = p[c].map(fmt_p)
    print(p.round(3).to_string(index=False))
