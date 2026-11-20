"""Checks for the Fig. 6 / 7 chain: the shape morphospace (notebook 01), the niche embedding
(notebook 02), the figure notebooks that read their pickles back (03, 04, 05) and the four-group
statistics script of Fig. 7 I-L (stats/fig7_stats.py).

Three kinds of test:
  * the settings: each notebook's configuration cells are executed and their values compared with
    Table S6, Methods 7.4-7.7 and the Fig. 7 legend;
  * a real small run: notebook 01 on 60 cells per stack of the committed feature table, then
    notebook 02 on that output with a synthetic workspace (state file + adjacency tables), then the
    loading cells of 03, 04 and 05 on what the two wrote. Cluster numbers on a subsample are not the
    paper's, so only the structure and the arithmetic are checked;
  * the Fig. 7 group guard: notebook 05 and fig7_stats.py on synthetic pickles sized to the four
    groups of Fig. 7, with decoy configurations that would change the counts if the wrong niche
    configuration or resolution were read.

plotly is replaced by a stand-in (no figure is drawn or written) and matplotlib draws on Agg.
No GPU is needed.  Run from the repository root:  python -m pytest tests -q
"""
import importlib.util
import json
import os
import pickle
import re
import shutil
import types
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _nbexec import REPO, code_cells, needs_data_root, only_cell, quiet_pyplot, run_cells, stub_plotly

FIG = REPO / "fig6_7_human_morphospace"
NBS = FIG / "notebooks"
NB01 = NBS / "01_shape_morphospace.ipynb"
NB02 = NBS / "02_niche_embedding.ipynb"
NB03 = NBS / "03_figure_panels.ipynb"
NB04 = NBS / "04_niche_figure_panels.ipynb"
NB05 = NBS / "05_placode_vs_ridge.ipynb"
STATS = FIG / "stats" / "fig7_stats.py"
TABLE = REPO / "data" / "fig6_7" / "all_stacks_features.csv"
TABLE_S3 = Path(os.environ.get(
    "EPICYCLE_SUPP_TABLES_CSV", REPO / "data" / "fig6_7" / "supp_tables")) / "Table_S3_Feature_dictionary.csv"

# The 31 shape features of the morphospace (Methods 7.4, Table S3 "Used in morphospace analysis").
MORPHOSPACE_31 = {
    "area", "area_filled", "equivalent_diameter", "extent", "euler_number", "bbox_volume", "fill_ratio",
    "sphericity", "aspect_ratio", "z_span", "z_elongation", "xy_elongation", "ellipsoid_solidity",
    "inertia_elongation", "inertia_flatness", "axis_major_length", "axis_minor_length", "major_axis_z_angle",
    "surface_area", "true_sphericity", "solidity", "skeleton_branches", "skeleton_nodes", "skeleton_length",
    "apical_area", "basal_area", "apical_basal_area_ratio", "max_cross_section_z_relative",
    "cross_section_taper", "cross_section_bulge", "roughness",
}
RESOLUTIONS = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PLACODE_STACK = "(13 wks) Front skin x40-Smooth05_Resampled.tif"
RIDGE_STACK = "(13 wks) D3L ventral ridges x40-Smooth05_Resampled.tif"
FIG7_GROUP_N = [393, 276, 1903, 1006]   # placode background, hair placode, ridge, ridge background
OLD_ROOTS = re.compile(r"\d{4}-\d{2}-\d{2}-EpiCycle_Analyses|morphospace_output")   # the folders the notebooks once read

PER_STACK = 60            # cells per stack in the small run
NO_ADJ_STACK = "(12 wks) D3L dorsal x40 stack-Smooth05_Resampled.tif"   # run B: no adjacency table
KEPT_CONFIGS = [(0.0, 1), (0.0, 3)]   # (self_weight, n_iter) kept in the small run of notebook 02


def _need_ml():
    for mod in ("umap", "leidenalg", "igraph", "sklearn"):
        pytest.importorskip(mod)


def _exec(cells, ns, first, last, skip=()):
    """Execute cells[first..last] (inclusive) in ns, leaving out the indices in skip."""
    for i in range(first, last + 1):
        if i not in skip:
            exec(compile(cells[i], f"<cell {i}>", "exec"), ns)


def _into_data_root(mp, root):
    """The notebooks' first cell: EPICYCLE_ANALYSES_DATA = root, started from the repository root."""
    (root / "fig6_7").mkdir(parents=True, exist_ok=True)
    mp.setenv("EPICYCLE_ANALYSES_DATA", str(root))
    mp.chdir(REPO)


class Calls:
    """Records the keyword arguments of UMAP, NearestNeighbors and leidenalg.find_partition (all three
    still run)."""

    def __init__(self, ns):
        self.umap, self.nn, self.leiden = [], [], []
        real_umap, real_nn, real_leiden = ns["UMAP"], ns["NearestNeighbors"], ns["leidenalg"]

        def UMAP(**kw):
            self.umap.append(kw)
            return real_umap(**kw)

        def NearestNeighbors(**kw):
            self.nn.append(kw)
            return real_nn(**kw)

        def find_partition(graph, partition_type, **kw):
            self.leiden.append(dict(kw, partition_type=partition_type))
            return real_leiden.find_partition(graph, partition_type, **kw)

        ns["UMAP"], ns["NearestNeighbors"] = UMAP, NearestNeighbors
        ns["leidenalg"] = types.SimpleNamespace(
            find_partition=find_partition, RBConfigurationVertexPartition=real_leiden.RBConfigurationVertexPartition)


def _fake_dendrogram(X, orientation, labels, linkagefun):
    """plotly.figure_factory.create_dendrogram, as far as notebook 01 reads it back."""
    from scipy.spatial.distance import pdist
    linkagefun(pdist(X))                         # the notebook's ward linkage runs as plotly would run it
    n = len(labels)
    tickvals = tuple(-(5.0 + 10.0 * k) for k in range(n))
    trace = types.SimpleNamespace(x=(0.0, 1.0, 1.0, 0.0), y=(tickvals[0],) * 2 + (tickvals[1],) * 2,
                                  line=types.SimpleNamespace(color=None))
    yaxis = types.SimpleNamespace(ticktext=tuple(reversed(labels)), tickvals=tickvals)
    return types.SimpleNamespace(layout=types.SimpleNamespace(yaxis=yaxis), data=[trace])


# ============================================================================ layout
def test_the_figure_folder_holds_the_five_notebooks():
    in_figure = {p.relative_to(FIG).as_posix() for p in FIG.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts}
    assert in_figure == {f"notebooks/{p.name}" for p in (NB01, NB02, NB03, NB04, NB05)}


@pytest.mark.parametrize("nb", [NB01, NB02, NB03, NB04, NB05], ids=lambda p: p.stem)
def test_every_notebook_resolves_the_data_root_and_names_no_old_folder(nb):
    cells = code_cells(nb)
    assert "EPICYCLE_ANALYSES_DATA" in cells[0] and '/ "fig6_7"' in cells[0]
    for i, src in enumerate(cells):
        found = OLD_ROOTS.search(src)
        assert found is None, f"{nb.name} cell {i} names {found.group(0)!r}"


def test_the_stats_scripts_name_no_old_folder():
    for py in (FIG / "stats").glob("*.py"):
        text = py.read_text(encoding="utf-8")
        found = OLD_ROOTS.search(text)
        assert found is None, f"{py.name} names {found.group(0)!r}"


# ============================================================================ A. the settings
@needs_data_root
def test_notebook_01_settings_match_table_s6(tmp_path, monkeypatch):
    _need_ml()
    stub_plotly(monkeypatch)
    _into_data_root(monkeypatch, tmp_path)
    cells = code_cells(NB01)
    ns = {}
    run_cells(cells, ns, stop_after="def save_fig(")
    assert Path.cwd() == tmp_path / "fig6_7"
    assert len(ns["TARGET_FEATURES"]) == 31 == len(set(ns["TARGET_FEATURES"]))
    assert set(ns["TARGET_FEATURES"]) == MORPHOSPACE_31
    assert (ns["N_PCS"], ns["KNN"], ns["ACTIVE_RES"]) == (10, 15, 1.0)
    assert ns["LEIDEN_RESOLUTIONS"] == RESOLUTIONS
    assert (ns["UMAP_NEIGHBORS"], ns["UMAP_MIN_DIST"], ns["METRIC"]) == (15, 0.1, "euclidean")
    assert (ns["RANDOM_SEED"], ns["WINSOR_CLIP"], ns["USE_RESIDUAL_PCA"]) == (42, 5.0, False)
    assert ns["FEATURES_CSV"] == "all_stacks_features.csv"
    assert ns["MESHES_ROOT"] == "workspace/artifacts/step4/meshes/"
    assert (ns["OUTPUT_DIR"], ns["WORKSPACE_DIR"], ns["FIG_OUT_DIR"]) == ("run", "workspace", "figures")


def test_the_31_features_are_the_morphospace_rows_of_table_s3():
    if not TABLE_S3.is_file():
        pytest.skip(f"{TABLE_S3} not found (set EPICYCLE_SUPP_TABLES_CSV to the supplementary tables' csv folder)")
    s3 = pd.read_csv(TABLE_S3, encoding="utf-8-sig")
    used = set(s3.loc[s3["Used in morphospace analysis"].astype(str).str.strip().str.lower() == "yes", "Feature"])
    assert used == MORPHOSPACE_31


@needs_data_root
def test_notebook_02_settings_match_table_s6(tmp_path, monkeypatch):
    _need_ml()
    _into_data_root(monkeypatch, tmp_path)
    cells = code_cells(NB02)
    ns = {}
    run_cells(cells, ns, stop_after="product(SELF_WEIGHTS, N_ITERS)")
    assert ns["SELF_WEIGHTS"] == [0.0, 0.3, 0.5, 0.7]
    assert ns["N_ITERS"] == list(range(1, 11))
    names = [c["name"] for c in ns["NICHE_CONFIGS"]]
    assert len(names) == 40 == len(set(names))
    assert {(c["self_weight"], c["n_iter"]) for c in ns["NICHE_CONFIGS"]} == set(product(ns["SELF_WEIGHTS"], ns["N_ITERS"]))
    assert {"name": "hop3_pure", "self_weight": 0.0, "n_iter": 3} in [dict(c) for c in ns["NICHE_CONFIGS"]]
    assert (ns["MIN_CONTACT"], ns["WEIGHT_BY_CONTACT"], ns["RANDOM_SEED"]) == (100, True, 42)
    assert ns["NICHE_LEIDEN_RES"] == RESOLUTIONS
    assert (ns["OUTPUT_DIR"], ns["WORKSPACE_DIR"]) == ("run", "workspace")
    state_cell = cells[only_cell(cells, "_state['stacks']")]
    assert "os.path.join(WORKSPACE_DIR, 'stack_segmentation_state.json')" in state_cell
    assert "stack_name + '_adjacency.csv'" in state_cell
    emb_cell = cells[only_cell(cells, "G_niche = ig.Graph(")]
    assert "n_neighbors=15, min_dist=0.1" in emb_cell and "NearestNeighbors(n_neighbors=16" in emb_cell


@needs_data_root
def test_notebook_04_featured_niche_configuration(tmp_path, monkeypatch):
    _into_data_root(monkeypatch, tmp_path)
    cells = code_cells(NB04)
    knob_cell = cells[only_cell(cells, 'NICHE_CONFIG      = "hop3_pure"')]
    head, sep, _ = knob_cell.partition("# -- Load shape morphospace --")
    assert sep, "notebook 04: the knob block no longer ends at '# -- Load shape morphospace --'"
    ns = {"os": os}
    exec(compile(head, "<04 knobs>", "exec"), ns)
    assert (ns["NICHE_CONFIG"], ns["NICHE_RESOLUTION"], ns["SHAPE_RESOLUTION"]) == ("hop3_pure", 1.5, 1.0)
    assert ns["MORPHO_OUTPUT_DIR"] == "run"


@needs_data_root
def test_notebook_05_groups_match_the_fig7_legend(tmp_path, monkeypatch):
    _into_data_root(monkeypatch, tmp_path)
    cells = code_cells(NB05)
    ns = {}
    run_cells(cells, ns, stop_after="PLACODE_CLUSTERS            =")
    assert (ns["NICHE_CONFIG"], ns["NICHE_RESOLUTION"], ns["NICHE_COL"]) == ("hop3_pure", 1.5, "niche_leiden_r1.5")
    assert (ns["MORPHO_PKL"], ns["NICHE_PKL"]) == ("run/morphospace_results.pkl", "run/niche_results.pkl")
    assert ns["PLACODE_STACK"] == PLACODE_STACK and ns["RIDGE_STACK"] == RIDGE_STACK
    assert ns["PLACODE_CLUSTERS"] == [17] and ns["PLACODE_BACKGROUND_CLUSTERS"] == [0]
    assert set(ns["RIDGE_CLUSTERS"]) == {3, 10, 13, 14, 7} and len(ns["RIDGE_CLUSTERS"]) == 5
    assert ns["RIDGE_BACKGROUND_CLUSTERS"] is None
    assert ns["N_REPRESENTATIVES"] == 10 and ns["N_PCS"] == 10
    guard = [c for c in cells if "assert _group_n ==" in c]
    assert len(guard) == 1 and "assert _group_n == [393, 276, 1903, 1006]" in guard[0]


def _load_fig7_stats(monkeypatch, data_root):
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(data_root))
    spec = importlib.util.spec_from_file_location("fig7_stats_under_test", STATS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fig7_stats_settings_and_paths(tmp_path, monkeypatch):
    fs = _load_fig7_stats(monkeypatch, tmp_path)
    assert Path(fs.ANALYSES) == tmp_path / "fig6_7" / "run"
    assert fs.EXPECTED_N == FIG7_GROUP_N
    assert fs.GROUP_ORDER == ["placode_background", "hair_placode", "ridge", "ridge_background"]
    assert fs.PLACODE_STACK == PLACODE_STACK and fs.RIDGE_STACK == RIDGE_STACK
    assert fs.PLACODE_CLUSTERS == [17] and fs.PLACODE_BG == [0] and set(fs.RIDGE_CLUSTERS) == {3, 10, 13, 14, 7}
    monkeypatch.delenv("EPICYCLE_ANALYSES_DATA")
    spec = importlib.util.spec_from_file_location("fig7_stats_default_root", STATS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert Path(mod.ANALYSES) == REPO / "data" / "fig6_7" / "run"


# ============================================================================ B + C. a small real run of 01 then 02
def _subsample(dest_csv):
    full = pd.read_csv(TABLE)
    complete = full.dropna(subset=sorted(MORPHOSPACE_31))
    parts = [g.sample(n=min(PER_STACK, len(g)), random_state=0) for _, g in complete.groupby("stack_name", sort=True)]
    sub = pd.concat(parts).reset_index(drop=True)
    sub.loc[5, "skeleton_length"] = np.nan          # one incomplete cell: notebook 01 drops it
    sub.to_csv(dest_csv, index=False)
    return sub


def _synthetic_workspace(ws, sub, rng):
    """The Step 4 state file (stacks -> qc_features_csv, relative to the workspace) and one
    <stack>_adjacency.csv next to each stack's features table: random pairs of the stack's cells with
    50-500 contact voxels, plus four pairs whose second cell is not in the table."""
    state = {"stacks": {}}
    adjacency = {}
    for i, (stack, g) in enumerate(sorted(sub.groupby("stack_name"), key=lambda t: t[0])):
        rel = f"stack_{i:02d}/qc/features.csv"
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        g[["label", "area"]].to_csv(ws / rel, index=False)
        state["stacks"][stack] = {"qc_features_csv": rel}
        labels = g["label"].astype(int).to_numpy()
        pairs = set()
        while len(pairs) < 3 * len(labels):
            a, b = rng.choice(labels, 2, replace=False)
            pairs.add((int(min(a, b)), int(max(a, b))))
        adj = pd.DataFrame(sorted(pairs), columns=["label_a", "label_b"])
        adj["contact_voxels"] = rng.integers(50, 501, len(adj))
        stray = pd.DataFrame({"label_a": labels[:4], "label_b": 10_000_000 + np.arange(4), "contact_voxels": 400})
        adj = pd.concat([adj, stray], ignore_index=True)       # neighbours outside the table: dropped
        adj.to_csv(ws / rel.replace("features.csv", stack + "_adjacency.csv"), index=False)
        adjacency[stack] = adj
    (ws / "stack_segmentation_state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    return adjacency


def _run_nb02(mp, root, configs):
    """Notebook 02 through its pickle, with the niche configurations cut down to configs."""
    mp.setenv("EPICYCLE_ANALYSES_DATA", str(root))
    os.chdir(REPO)                                       # the first cell searches up from the repository
    cells = code_cells(NB02)
    i_imp = only_cell(cells, "from umap import UMAP")
    i_cfg = only_cell(cells, "product(SELF_WEIGHTS, N_ITERS)")
    i_pkl = only_cell(cells, "'niche_results.pkl'")
    ns = {}
    _exec(cells, ns, 0, i_imp)
    calls = Calls(ns)
    _exec(cells, ns, i_imp + 1, i_cfg)
    ns["NICHE_CONFIGS"] = [{"self_weight": w, "n_iter": k, "name": ns["_make_name"](w, k)} for w, k in configs]
    _exec(cells, ns, i_cfg + 1, i_pkl)
    return ns, calls


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Run A: notebook 01 then notebook 02 (hop1_pure, hop3_pure) with adjacency for all 13 stacks.
    Run B: notebook 02 (hop3_pure) on run A's morphospace with no adjacency table for NO_ADJ_STACK."""
    if not TABLE.is_file():
        pytest.skip(f"{TABLE} is not in the checkout")
    if not (REPO / "data").is_dir() or not (REPO / "environment.yml").is_file():
        pytest.skip("the notebooks locate the repository by environment.yml and data/")
    _need_ml()
    root = tmp_path_factory.mktemp("fig6_7_chain")
    data = root / "fig6_7"
    (data / "workspace").mkdir(parents=True)
    (data / "run").mkdir()
    sub = _subsample(data / "all_stacks_features.csv")
    adjacency = _synthetic_workspace(data / "workspace", sub, np.random.default_rng(0))

    with pytest.MonkeyPatch.context() as mp:
        _into_data_root(mp, root)
        plotly = stub_plotly(mp)
        mp.setattr(plotly["figure_factory"], "create_dendrogram", _fake_dendrogram, raising=False)
        quiet_pyplot(mp)

        # notebook 01, every cell (the dendrogram is plotly's stand-in, the meshes are absent)
        c1 = code_cells(NB01)
        i_imp, i_pkl = only_cell(c1, "from umap import UMAP"), only_cell(c1, "'morphospace_results.pkl'")
        ns1 = {}
        _exec(c1, ns1, 0, i_imp)
        calls1 = Calls(ns1)
        _exec(c1, ns1, i_imp + 1, i_pkl)

        ns2, calls2 = _run_nb02(mp, root, KEPT_CONFIGS)

        root_b = tmp_path_factory.mktemp("fig6_7_chain_no_adj")
        data_b = root_b / "fig6_7"
        shutil.copytree(data / "workspace", data_b / "workspace")
        (data_b / "run").mkdir()
        for name in ("morphospace_results.pkl", "all_features_multi_leiden.csv"):
            shutil.copy2(data / "run" / name, data_b / "run" / name)
        state = json.loads((data_b / "workspace" / "stack_segmentation_state.json").read_text(encoding="utf-8"))
        rel = state["stacks"][NO_ADJ_STACK]["qc_features_csv"]
        (data_b / "workspace" / rel.replace("features.csv", NO_ADJ_STACK + "_adjacency.csv")).unlink()
        ns2b, _ = _run_nb02(mp, root_b, [(0.0, 3)])
    return types.SimpleNamespace(root=root, data=data, sub=sub, adjacency=adjacency, ns1=ns1, calls1=calls1,
                                 ns2=ns2, calls2=calls2, root_b=root_b, data_b=data_b, ns2b=ns2b)


def _downstream_keys(pattern, sources):
    return {k for src in sources for k in re.findall(pattern, src)}


def test_notebook_01_preprocesses_the_31_features_as_in_methods(chain):
    ns, sub = chain.ns1, chain.sub
    assert ns["avail_feat"] == ns["TARGET_FEATURES"] and set(ns["avail_kf"]) == MORPHOSPACE_31
    assert len(ns["df"]) == len(sub) - 1                     # the one incomplete cell is dropped
    # log1p where min >= 0 and |skew| > 1, then RobustScaler, then clip at +-5
    from scipy.stats import skew
    from sklearn.preprocessing import RobustScaler
    X = ns["df"][ns["TARGET_FEATURES"]].to_numpy(dtype=float)
    logged = [(X[:, i].min() >= 0 and abs(skew(X[:, i])) > 1.0) for i in range(X.shape[1])]
    assert 0 < sum(logged) < 31                                # the rule bites both ways on this subsample
    X = np.where(logged, np.log1p(np.clip(X, 0, None)), X)
    expected = np.clip(RobustScaler().fit_transform(X), -5.0, 5.0)
    np.testing.assert_allclose(ns["X_all"], expected, rtol=1e-10, atol=1e-10)
    assert np.abs(ns["X_all"]).max() == pytest.approx(5.0)   # the winsorising clip is reached
    assert ns["pcs"].shape == (len(ns["df"]), 10)


def test_notebook_01_graph_clustering_and_umap(chain):
    ns, calls = chain.ns1, chain.calls1
    pcs, G = ns["pcs"], ns["G"]
    assert calls.nn[0] == {"n_neighbors": 16, "metric": "euclidean"}       # 15 neighbours + the cell itself
    assert G.vcount() == len(pcs) and min(G.degree()) >= 15
    e = np.array(G.get_edgelist())
    d = np.linalg.norm(pcs[e[:, 0]] - pcs[e[:, 1]], axis=1)
    np.testing.assert_allclose(G.es["weight"], np.exp(-d), rtol=1e-9)     # w = exp(-d)
    assert [c["resolution_parameter"] for c in calls.leiden] == RESOLUTIONS
    assert all(c["seed"] == 42 and c["weights"] == "weight"
               and c["partition_type"].__name__ == "RBConfigurationVertexPartition" for c in calls.leiden)
    for res in RESOLUTIONS:
        col = f"leiden_r{res}"
        assert col in ns["df"] and ns["df"][col].notna().all() and ns["leiden_results"][res]["col"] == col
    assert (ns["df"]["leiden"] == ns["df"]["leiden_r1.0"]).all()
    umap = [c for c in calls.umap if c.get("n_components") == 2]
    assert umap and umap[0]["n_neighbors"] == 15 and umap[0]["min_dist"] == 0.1 and umap[0]["random_state"] == 42
    assert {"umap_1", "umap_2"} <= set(ns["df"]) and np.isfinite(ns["df"][["umap_1", "umap_2"]].to_numpy()).all()


def test_notebook_01_pickle_carries_what_the_downstream_notebooks_read(chain):
    with open(chain.data / "run" / "morphospace_results.pkl", "rb") as f:
        res = pickle.load(f)
    downstream = [s for nb in (NB02, NB03, NB04, NB05) for s in code_cells(nb)]
    keys = _downstream_keys(r"\b(?:results|morpho)(?:\[|\.get\()\s*[\"']([A-Za-z_]+)[\"']", downstream)
    keys |= _downstream_keys(r"\bm\[\s*[\"']([A-Za-z_]+)[\"']", [STATS.read_text(encoding="utf-8")])
    assert {"df", "pcs", "avail_kf", "leiden_results", "config", "mesh_lookup"} <= keys
    assert keys <= set(res), f"missing from morphospace_results.pkl: {keys - set(res)}"
    assert len(res["df"]) == len(chain.sub) - 1 == res["pcs"].shape[0]
    assert res["pcs"].shape[1] == 10 and set(res["leiden_results"]) == set(RESOLUTIONS)
    assert res["config"]["ACTIVE_RES"] == 1.0
    assert (chain.data / "run" / "all_features_multi_leiden.csv").is_file()


def _expected_niche(df, pcs, adjacency, stack, n_iter, min_contact=100):
    """Contact-weighted, row-normalised neighbour averaging with self weight 0, as in Methods 7.6."""
    idx = np.where((df["stack_name"] == stack).to_numpy())[0]
    local = {int(l): i for i, l in enumerate(df["label"].to_numpy()[idx])}
    adj = adjacency[stack]
    adj = adj[(adj["contact_voxels"] >= min_contact) & adj["label_a"].isin(local) & adj["label_b"].isin(local)]
    W = np.zeros((len(idx), len(idx)))
    for a, b, c in adj[["label_a", "label_b", "contact_voxels"]].itertuples(index=False):
        W[local[a], local[b]] += c
        W[local[b], local[a]] += c
    s = W.sum(axis=1)
    iso = s == 0
    Wn = W / np.where(iso, 1.0, s)[:, None]
    X0 = pcs[idx]
    X = X0.copy()
    for _ in range(n_iter):
        X = Wn @ X
        X[iso] = X0[iso]
    return idx, X, len(adj)


def test_notebook_02_drops_small_contacts_and_averages_by_contact(chain):
    ns = chain.ns2
    df, pcs, nres = ns["df"], ns["pcs"], ns["niche_results"]
    stacks = sorted(chain.adjacency)
    for name, n_iter in (("hop1_pure", 1), ("hop3_pure", 3)):
        kept = {}
        for stack in stacks:
            idx, X, n_kept = _expected_niche(df, pcs, chain.adjacency, stack, n_iter)
            np.testing.assert_allclose(nres[name]["niche_pcs"][idx], X, rtol=1e-9, atol=1e-12)
            kept[stack] = n_kept
            _, X_all_edges, _ = _expected_niche(df, pcs, chain.adjacency, stack, n_iter, min_contact=0)
            assert not np.allclose(X, X_all_edges), "the >= 100 voxel filter does not change this stack"
        stats = nres[name]["stats"]
        assert [s["edges_kept"] for s in stats] == [kept[s] for s in stacks]
        assert [s["edges_total"] for s in stats] == [len(chain.adjacency[s]) for s in stacks]
        assert all(s["edges_kept"] < s["edges_total"] for s in stats)


def test_notebook_02_niche_pickle_has_the_configs_downstream_reads(chain):
    with open(chain.data / "run" / "niche_results.pkl", "rb") as f:
        out = pickle.load(f)
    downstream = [s for nb in (NB04, NB05) for s in code_cells(nb)] + [STATS.read_text(encoding="utf-8")]
    top = _downstream_keys(r"\b(?:niche_data|n)(?:\[|\.get\()\s*[\"']([A-Za-z_]+)[\"']", downstream) - {"EPICYCLE_ANALYSES_DATA"}
    assert "niche_results" in top and top <= set(out)
    assert out["MIN_CONTACT"] == 100 and out["WEIGHT_BY_CONTACT"] is True and out["NICHE_LEIDEN_RES"] == RESOLUTIONS
    assert set(out["niche_results"]) == {"hop1_pure", "hop3_pure"}
    assert [c["name"] for c in out["NICHE_CONFIGS"]] == ["hop1_pure", "hop3_pure"]
    n = len(chain.ns2["df"])
    nr_keys = _downstream_keys(r"\bnr(?:\[|\.get\()\s*[\"']([A-Za-z_]+)[\"']", downstream)
    for name, nr in out["niche_results"].items():
        assert nr_keys <= set(nr), f"{name}: missing {nr_keys - set(nr)}"
        assert nr["valid_mask"].all() and nr["n_valid"] == n and not np.isnan(nr["niche_pcs"]).any()
        assert nr["umap"].shape == (n, 2) and set(nr["clusters"]) == set(RESOLUTIONS)
        for res in RESOLUTIONS:
            assert len(nr["clusters"][res]["membership"]) == n
    calls = chain.calls2
    assert len(calls.umap) == 2
    assert all(c["n_neighbors"] == 15 and c["min_dist"] == 0.1 and c["random_state"] == 42 for c in calls.umap)
    assert all(c["n_neighbors"] == 16 for c in calls.nn)                   # niche kNN: 15 + the cell itself
    assert [c["resolution_parameter"] for c in calls.leiden] == RESOLUTIONS * 2
    assert all(c["seed"] == 42 for c in calls.leiden)


def test_notebook_02_leaves_out_the_cells_of_a_stack_without_adjacency(chain):
    df = chain.ns2b["df"]
    nr = chain.ns2b["niche_results"]["hop3_pure"]
    with_adj = (df["stack_name"] != NO_ADJ_STACK).to_numpy()
    assert 0 < (~with_adj).sum() < len(df)
    np.testing.assert_array_equal(nr["valid_mask"], with_adj)
    assert nr["n_valid"] == with_adj.sum() and np.isnan(nr["niche_pcs"][~with_adj]).all()
    assert nr["umap"].shape == (with_adj.sum(), 2)
    for res in RESOLUTIONS:
        assert len(nr["clusters"][res]["membership"]) == with_adj.sum()
    # the other stacks' niche vectors do not depend on the missing table
    np.testing.assert_allclose(nr["niche_pcs"][with_adj],
                               chain.ns2["niche_results"]["hop3_pure"]["niche_pcs"][with_adj], rtol=1e-12)


def test_notebooks_03_04_05_load_the_small_run(chain, monkeypatch):
    _into_data_root(monkeypatch, chain.root)
    stub_plotly(monkeypatch)
    quiet_pyplot(monkeypatch)
    n = len(chain.ns2["df"])

    ns3 = {}
    run_cells(code_cells(NB03), ns3, stop_after='FIG_OUT_DIR = "figures"')
    assert ns3["RESOLUTION"] == 1.0 and "leiden_r1.0" in ns3["df"] and len(ns3["df"]) == n
    assert ns3["FIG_OUT_DIR"] == "figures"

    os.chdir(REPO)
    ns4 = {}
    run_cells(code_cells(NB04), ns4, stop_after="[Step 0]")             # loading + the 13-stack metadata check
    df4 = ns4["df"]
    assert ns4["NICHE_CONFIG"] == "hop3_pure" and ns4["NICHE_RESOLUTION"] == 1.5
    assert df4["niche_leiden_r1.5"].notna().all() and len(df4) == n
    assert {f"PC{k}" for k in range(1, 11)} | {f"niche_PC{k}" for k in range(1, 11)} <= set(df4)

    os.chdir(REPO)
    ns5 = {}
    run_cells(code_cells(NB05), ns5, stop_after="with open(NICHE_PKL")
    np.testing.assert_array_equal(ns5["df"]["niche_leiden_r1.5"].to_numpy(), df4["niche_leiden_r1.5"].to_numpy())


def test_notebook_05_loads_a_run_with_cells_without_adjacency(chain, monkeypatch):
    _into_data_root(monkeypatch, chain.root_b)
    stub_plotly(monkeypatch)
    quiet_pyplot(monkeypatch)
    ns5 = {}
    run_cells(code_cells(NB05), ns5, stop_after="with open(NICHE_PKL")
    col = ns5["df"]["niche_leiden_r1.5"]
    assert col.isna().to_numpy().tolist() == (ns5["df"]["stack_name"] == NO_ADJ_STACK).tolist()


def test_notebook_04_loads_a_run_with_cells_without_adjacency(chain, monkeypatch):
    """nr['niche_pcs'] has one row per cell (NaN without adjacency): notebook 04 copies its valid rows."""
    _into_data_root(monkeypatch, chain.root_b)
    stub_plotly(monkeypatch)
    quiet_pyplot(monkeypatch)
    ns4 = {}
    run_cells(code_cells(NB04), ns4, stop_after="# -- Merge chosen niche config into df --")
    df = ns4["df"]
    no_adj = (df["stack_name"] == NO_ADJ_STACK).to_numpy()
    nr = chain.ns2b["niche_results"]["hop3_pure"]
    assert df.loc[no_adj, ["niche_PC1", "niche_umap_1", "niche_leiden_r1.5"]].isna().all().all()
    np.testing.assert_allclose(df.loc[~no_adj, [f"niche_PC{k}" for k in range(1, 11)]].to_numpy(),
                               nr["niche_pcs"][~no_adj], rtol=1e-12)


# ============================================================================ the Fig. 7 group guard
def _fig7_pickles(run_dir, rng):
    """Synthetic run pickles whose hop3_pure r1.5 niche clusters give the four groups of Fig. 7."""
    blocks = [  # (stack, niche cluster, n)
        (PLACODE_STACK, 0, 393), (PLACODE_STACK, 17, 276), (PLACODE_STACK, 4, 120),
        (RIDGE_STACK, 3, 400), (RIDGE_STACK, 10, 400), (RIDGE_STACK, 13, 400), (RIDGE_STACK, 14, 400), (RIDGE_STACK, 7, 303),
        (RIDGE_STACK, 1, 600), (RIDGE_STACK, 2, 406),
        ("(14 wks) digit 5 unpatterned skin x40 stack-Smooth05_Resampled.tif", 0, 80),
        ("(14 wks) digit 5 unpatterned skin x40 stack-Smooth05_Resampled.tif", 17, 40),
    ]
    stack = np.concatenate([[s] * n for s, _, n in blocks])
    member = np.concatenate([[c] * n for _, c, n in blocks]).astype(int)
    order = rng.permutation(len(stack))
    stack, member = stack[order], member[order]
    n = len(stack)
    df = pd.DataFrame({"stack_name": stack, "label": np.arange(1, n + 1)})
    df["donor"] = np.where(df["stack_name"].str.startswith("donor2"), "donor2", "donor1")
    df["timepoint"] = np.where(df["donor"] == "donor2", "13wk", "14wk")
    df["tissue"] = np.where(df["stack_name"] == PLACODE_STACK, "front", "volar")
    df["pattern"] = df["tissue"]
    for f in ("z_span", "area", "true_sphericity", "aspect_ratio", "z_elongation", "xy_elongation",
              "apical_basal_area_ratio", "cross_section_taper", "solidity", "major_axis_z_angle",
              "inertia_elongation", "axis_minor_length", "axis_major_length"):
        df[f] = rng.uniform(1.0, 2.0, n)
    df["leiden"] = rng.integers(0, 5, n)
    df["leiden_r1.0"] = df["leiden"]
    morpho = {"df": df, "pcs": rng.normal(size=(n, 10)), "avail_kf": sorted(MORPHOSPACE_31), "mesh_lookup": {}}
    every = np.ones(n, bool)
    decoy = {"membership": np.zeros(n, int), "n_clusters": 1}   # read by mistake, the counts change
    niche = {"niche_results": {
        "hop3_pure": {"valid_mask": every, "clusters": {1.5: {"membership": member, "n_clusters": 20}, 1.0: decoy}},
        "hop3_w0.3": {"valid_mask": every, "clusters": {1.5: decoy, 1.0: decoy}},
    }, "NICHE_LEIDEN_RES": [1.0, 1.5]}
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("morphospace_results.pkl", morpho), ("niche_results.pkl", niche)):
        with open(run_dir / name, "wb") as f:
            pickle.dump(obj, f)


@needs_data_root
def test_notebook_05_group_guard_passes_on_the_fig7_counts(tmp_path, monkeypatch):
    _into_data_root(monkeypatch, tmp_path)
    _fig7_pickles(tmp_path / "fig6_7" / "run", np.random.default_rng(1))
    stub_plotly(monkeypatch)
    quiet_pyplot(monkeypatch)
    cells = code_cells(NB05)
    ns = {}
    run_cells(cells, ns, stop_after="assert _group_n ==")
    assert ns["_group_n"] == FIG7_GROUP_N
    assert ns["g"]["group"].value_counts().reindex(ns["GROUP_ORDER"]).tolist() == FIG7_GROUP_N


@needs_data_root
def test_notebook_05_group_guard_fails_when_the_counts_differ(tmp_path, monkeypatch):
    _into_data_root(monkeypatch, tmp_path)
    _fig7_pickles(tmp_path / "fig6_7" / "run", np.random.default_rng(1))
    stub_plotly(monkeypatch)
    quiet_pyplot(monkeypatch)
    cells = code_cells(NB05)
    ns = {}
    run_cells(cells, ns, stop_after="PLACODE_CLUSTERS            =")
    i_guard = only_cell(cells, "assert _group_n ==")
    ns["PLACODE_BACKGROUND_CLUSTERS"] = [0, 4]                 # 120 more background cells
    with pytest.raises(AssertionError, match="differ from the deposited run"):
        _exec(cells, ns, only_cell(cells, "PLACODE_CLUSTERS            =") + 1, i_guard)


def test_fig7_stats_reads_hop3_pure_r1_5_and_finds_the_four_groups(tmp_path, monkeypatch):
    _fig7_pickles(tmp_path / "fig6_7" / "run", np.random.default_rng(1))
    fs = _load_fig7_stats(monkeypatch, tmp_path)
    g = fs.load_groups()
    assert g["group"].value_counts().reindex(fs.GROUP_ORDER).tolist() == FIG7_GROUP_N
    assert set(g.loc[g["group"] == "hair_placode", "stack_name"]) == {PLACODE_STACK}
    np.testing.assert_allclose(g["area_log1p"], np.log1p(g["area"]))


def test_notebook_05_draws_the_four_groups_on_the_shape_and_the_niche_umap():
    """Fig. 7G: the niche-UMAP cell repeats the shape-UMAP call with the niche coordinates."""
    cells = code_cells(NB05)
    i_shape, i_niche = only_cell(cells, 'save(fig, "shape_umap_groups")'), only_cell(cells, 'save(fig, "niche_umap_groups")')
    assert i_niche == i_shape + 1
    call = lambda src: src[src.index("fig = umap_highlight("):]
    shape, niche = call(cells[i_shape]), call(cells[i_niche])
    assert 'x_col="umap_1", y_col="umap_2"' in shape and 'x_col="niche_umap_1", y_col="niche_umap_2"' in niche
    assert niche.replace("niche_umap_", "umap_") == shape.replace("shape_umap_groups", "umap_groups")
    assert 'df.loc[_valid, f"niche_umap_{_k}"] = np.asarray(_nr["umap"])[:, _k - 1]' in cells[i_niche]
