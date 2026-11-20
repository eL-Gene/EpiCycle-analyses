"""Checks for the Fig. 2 chain: file names agree between steps, the filter script does what the
Methods say, and (when the raw per-slice tables are available) it reproduces the committed
summary tables exactly.

Run from the repository root:  python -m pytest tests -q
"""
import json, os, re, subprocess, sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "fig2_hitl_2d_benchmark"
TRAIN = FIG / "1_training"
BENCH = FIG / "2_cross_round_benchmark"
METRICS = FIG / "3_metrics"
PLOTS = FIG / "4_plots"
FILTER = METRICS / "process_2D_cross_round_metrics.py"


def notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


# ----------------------------------------------------------------------------- names agree
def test_folders_are_in_run_order():
    assert [p.name for p in sorted(FIG.iterdir()) if p.is_dir()] == [
        "1_training", "2_cross_round_benchmark", "3_metrics", "4_plots"]


def test_training_writes_what_the_benchmark_reads():
    for nb in TRAIN.glob("train_hitl_rounds_*.ipynb"):
        code = notebook_code(nb)
        assert 'data_dir = "../ground_truth/zebrafish/development"' in code
        for name in ["training_data_spatial", "training_data_axial",
                     "training_rounds_results/spatial_models/", "training_rounds_results/axial_models/",
                     "selected_indices.json", "_img.tif", "_mask.tif"]:
            assert name in code, f"{nb.name} lacks {name}"
    bench = notebook_code(BENCH / "cross_round_benchmark.ipynb")
    for name in ["training_rounds_results", "training_data_spatial", "training_data_axial",
                 "_img.tif", "_mask.tif", '"cross_round_metrics"', "{view}_cross_round_metrics.csv"]:
        assert name in bench, f"benchmark notebook lacks {name}"


def test_benchmark_writes_what_the_filter_and_plots_read():
    filt = FILTER.read_text(encoding="utf-8")
    cum = (PLOTS / "figure2_cumulative_training.py").read_text(encoding="utf-8")
    for view in ("spatial", "axial"):
        assert f'"{view}_cross_round_metrics.csv"' in filt
        assert f'"{view}_cross_round_metrics.csv"' in cum
    for script in ("figure2_main.py", "figure2_supp.py"):
        s = (PLOTS / script).read_text(encoding="utf-8")
        for view in ("spatial", "axial"):
            assert f'"{view}_cross_round_metrics_filtered.csv"' in s
            assert f'"{view}_cross_round_metrics_filtered.csv"' in filt
        assert 'DATA_ROOT / "fig2" / "cross_round_metrics"' in s


def test_every_step_resolves_the_data_root_the_same_way():
    for script in [FILTER, PLOTS / "figure2_cumulative_training.py", PLOTS / "figure2_main.py", PLOTS / "figure2_supp.py"]:
        s = script.read_text(encoding="utf-8")
        assert 'os.environ.get("EPICYCLE_ANALYSES_DATA"' in s, script.name
        assert 'DATA_ROOT / "fig2"' in s, script.name
    for nb in list(TRAIN.glob("*.ipynb")) + list(BENCH.glob("*.ipynb")):
        cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
        first = "".join(next(c for c in cells if c["cell_type"] == "code")["source"])
        assert "EPICYCLE_ANALYSES_DATA" in first and '/ "fig2"' in first, nb.name


def test_no_machine_paths_or_emoji():
    emoji = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
    machine = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/]|/gpfs/|/home/[a-z])")
    for p in FIG.rglob("*"):
        if p.suffix not in (".ipynb", ".py", ".md"):
            continue
        s = p.read_text(encoding="utf-8")
        assert not emoji.search(s), p
        assert not machine.search(s), p


# ----------------------------------------------------------------------------- the filter
def run_filter(data_root: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, EPICYCLE_ANALYSES_DATA=str(data_root), PYTHONUTF8="1")
    return subprocess.run([sys.executable, str(FILTER)], env=env, capture_output=True, text=True)


def make_raw(rows, path: Path):
    cols = ["slice_idx", "dice", "iou", "aji", "ap_50", "ap_75", "ap_90", "model_round", "eval_round", "mode", "model_name"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_filter_removes_only_lateral_empty_slices(tmp_path):
    cr = tmp_path / "fig2" / "cross_round_metrics"
    cr.mkdir(parents=True)
    sp = [
        ["s0", 0.5, 0.2, 0.3, 0.40, 0.1, 0.0, 0, 9, "spatial", "m0"],   # kept
        ["s1", 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0, 9, "spatial", "m0"],   # empty: dropped
        ["s2", 0.3, 0.1, 0.1, 0.00, 0.0, 0.0, 0, 9, "spatial", "m0"],   # genuine failure: kept
        ["s3", 0.9, 0.8, 0.8, 0.80, 0.5, 0.2, 1, 9, "spatial", "m1"],
        ["s4", 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 1, 9, "spatial", "m1"],   # empty: dropped
        ["s5", 0.9, 0.8, 0.8, 0.60, 0.5, 0.2, 1, 0, "spatial", "m1"],   # not final round
    ]
    ax = [
        ["a0", 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0, 9, "axial", "a0"],     # axial is never filtered
        ["a1", 0.7, 0.5, 0.5, 0.50, 0.2, 0.1, 0, 9, "axial", "a0"],
    ]
    make_raw(sp, cr / "spatial_cross_round_metrics.csv")
    make_raw(ax, cr / "axial_cross_round_metrics.csv")

    # the script writes its small tables next to itself: run it on a copy so the repo stays clean
    import shutil
    work = tmp_path / "3_metrics"; shutil.copytree(METRICS, work)
    env = dict(os.environ, EPICYCLE_ANALYSES_DATA=str(tmp_path), PYTHONUTF8="1")
    # REPO_ROOT is derived from the script location; the data root override makes that irrelevant
    r = subprocess.run([sys.executable, str(work / FILTER.name)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    filt_sp = pd.read_csv(cr / "spatial_cross_round_metrics_filtered.csv")
    assert sorted(filt_sp.slice_idx) == ["s0", "s2", "s3", "s5"]
    filt_ax = pd.read_csv(cr / "axial_cross_round_metrics_filtered.csv")
    assert len(filt_ax) == 2

    summary = pd.read_csv(work / "2D_final_round_summary.csv")
    lat0 = summary[(summary["mode"] == "lateral") & (summary.model_round == 0)].iloc[0]
    assert lat0.n_slices == 2 and lat0.median_ap50 == pytest.approx(0.20)
    lat1 = summary[(summary["mode"] == "lateral") & (summary.model_round == 1)].iloc[0]
    assert lat1.n_slices == 1 and lat1.median_ap50 == pytest.approx(0.80)
    ax0 = summary[(summary["mode"] == "axial") & (summary.model_round == 0)].iloc[0]
    assert ax0.n_slices == 2 and ax0.median_ap50 == pytest.approx(0.25)

    heat = pd.read_csv(work / "2D_cross_round_heatmap_lateral.csv", index_col=0)
    assert heat.loc[1, "9"] == pytest.approx(0.80) and heat.loc[1, "0"] == pytest.approx(0.60)


RAW = Path(os.environ.get("EPICYCLE_ANALYSES_DATA", REPO / "data")) / "fig2" / "cross_round_metrics"


@pytest.mark.skipif(not (RAW / "spatial_cross_round_metrics.csv").exists(), reason="raw per-slice tables not downloaded")
def test_filter_reproduces_the_committed_tables(tmp_path):
    import shutil
    work = tmp_path / "3_metrics"; shutil.copytree(METRICS, work)
    # copy the two raw tables so the filtered CSVs are written under tmp_path, not into the deposit
    cr = tmp_path / "fig2" / "cross_round_metrics"; cr.mkdir(parents=True)
    for view in ("spatial", "axial"):
        shutil.copy2(RAW / f"{view}_cross_round_metrics.csv", cr)
    env = dict(os.environ, EPICYCLE_ANALYSES_DATA=str(tmp_path), PYTHONUTF8="1")
    r = subprocess.run([sys.executable, str(work / FILTER.name)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "4,046" in r.stdout and "50,050" in r.stdout          # Methods §5.1
    for name in ["2D_final_round_summary.csv", "2D_cross_round_heatmap_lateral.csv", "2D_cross_round_heatmap_axial.csv"]:
        new = pd.read_csv(work / name)
        old = pd.read_csv(METRICS / name)
        pd.testing.assert_frame_equal(new, old, check_exact=False, atol=5e-4)


@pytest.mark.skipif(not (RAW / "spatial_cross_round_metrics.csv").exists(), reason="raw per-slice tables not downloaded")
def test_raw_metric_tables_are_well_formed():
    expected = {"spatial": [105, 210, 315, 420, 525, 630, 700, 700, 700, 700],
                "axial": [210, 420, 630, 840, 1050, 1260, 1393, 1400, 1400, 1400]}
    for view, sizes in expected.items():
        df = pd.read_csv(RAW / f"{view}_cross_round_metrics.csv")
        assert not df.duplicated(["slice_idx", "model_round"]).any()
        grid = df.groupby(["model_round", "eval_round"]).size().unstack()
        assert grid.shape == (10, 10)
        assert (grid.nunique() == 1).all(), "every model must be scored on the same slices"
        assert grid.iloc[0].tolist() == sizes
        # Slices where both the truth and the prediction are empty score AP = 0/0 = NaN
        # (lateral only: 1,737 rows). The filter in step 3 does not remove them; pandas skips
        # them in the medians but counts them in n_slices.
        nan_rows = df.ap_50.isna()
        assert nan_rows.sum() == (1737 if view == "spatial" else 0)
        assert (df.loc[nan_rows, "dice"] == 0).all() and df.loc[nan_rows, "aji"].isna().all()
        ok = df[~nan_rows]
        for col in ("ap_50", "ap_75", "ap_90", "dice", "iou", "aji"):
            assert ok[col].between(0, 1).all(), col
        assert (ok.ap_50 >= ok.ap_75).all() and (ok.ap_75 >= ok.ap_90).all()
        empty = ((df.ap_50 == 0) & (df.dice == 0)).sum()
        assert empty == (4046 if view == "spatial" else 0)


# ----------------------------------------------------------------------------- the HITL round loop (Table S5)
# The training notebooks are executed cell by cell: the slice sampler and the round loader are
# replaced by recorders, Cellpose's model and train_seg by fakes (tests/_nbexec.py), and the number
# of rounds is cut from 10 to 2 so the loop runs once from scratch and once from a checkpoint.
from _nbexec import CellposeRecorder, code_cells, needs_data_root, only_cell, posix, run_cells, tag_of  # noqa: E402

EXPECTED_HITL = {  # notebook -> (train_mode, views sampled, model folder, training-data folder)
    "train_hitl_rounds_lateral": ("spatial", ["XY"], "training_rounds_results/spatial_models/", "training_data_spatial"),
    "train_hitl_rounds_axial": ("axial", ["XZ", "YZ"], "training_rounds_results/axial_models/", "training_data_axial"),
}


def test_every_hitl_training_notebook_is_listed():
    assert sorted(p.stem for p in TRAIN.glob("*.ipynb")) == sorted(EXPECTED_HITL)


@needs_data_root
@pytest.mark.parametrize("stem", sorted(EXPECTED_HITL))
def test_hitl_round_loop_trains_as_in_table_s5(stem, tmp_path, monkeypatch):
    import numpy as np
    rec = CellposeRecorder(monkeypatch)
    dev = tmp_path / "ground_truth" / "zebrafish" / "development"
    dev.mkdir(parents=True)
    for i in (1, 2, 3):                                   # only listed, never read: the sampler is faked
        (dev / f"patch-{i}.tif").touch(); (dev / f"patch-{i}_masks.tif").touch()
    (tmp_path / "fig2").mkdir()
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(REPO)

    mode, views, model_dir, data_dir_ = EXPECTED_HITL[stem]
    cells = code_cells(TRAIN / f"{stem}.ipynb")
    loop_i = only_cell(cells, "for round_number in range(completed_rounds, num_rounds)")
    ns = {}
    run_cells(cells[:loop_i], ns, stop_after="def train_cellpose_model(")
    assert [s["name"] for s in ns["stacks"]] == ["patch-1", "patch-2", "patch-3"]

    sampled, loaded = [], []

    def fake_sampler(**kw):
        sampled.append(kw)
        return {}, {}

    def fake_loader(round_number, base_dir="training_data", mode="both"):
        imgs = [np.full((2, 4, 4), 100 * round_number + i, dtype=np.float32) for i in range(3)]
        masks = [np.full((4, 4), 100 * round_number + i, dtype=np.uint16) for i in range(3)]
        loaded.append(dict(round_number=round_number, base_dir=base_dir, mode=mode, imgs=imgs, masks=masks))
        return imgs, masks

    ns["sample_slices_for_round"] = fake_sampler
    ns["load_training_pairs"] = fake_loader
    loop, n = re.subn(r"^num_rounds = 10\b", "num_rounds = 2", cells[loop_i], flags=re.M)
    assert n == 1, "the loop cell must set num_rounds = 10 (ten HITL rounds)"
    exec(compile(loop, "<cell>", "exec"), ns)

    # the configuration (Table S5 / Methods)
    assert ns["train_mode"] == mode and ns["model_save_path"] == model_dir and ns["training_data_dir"] == data_dir_
    assert ns["use_dynamic_lr"] is False
    assert [ns["get_training_schedule"](r) for r in range(10)] == [(1000, 1e-2)] * 10   # fixed LR, every round

    # the sampler, once per round
    assert [kw["round_number"] for kw in sampled] == [0, 1]
    for kw in sampled:
        assert kw["stacks"] is ns["stacks"] and kw["views"] == views and kw["base_dir"] == data_dir_
        assert (kw["slices_per_stack"], kw["edge_buffer"], kw["min_distance"], kw["allowed_per_view"]) == (15, 50, 5, 100)

    # the round's cumulative slices, of this view only
    assert [(d["round_number"], d["base_dir"], d["mode"]) for d in loaded] == [(0, data_dir_, mode), (1, data_dir_, mode)]

    # what reaches Cellpose
    assert len(rec.trains) == 2
    for r, (call, data) in enumerate(zip(rec.trains, loaded)):
        assert call["train_data"] is data["imgs"] and call["train_labels"] is data["masks"]
        assert [tag_of(a) for a in call["train_data"]] == [100 * r, 100 * r + 1, 100 * r + 2]
        assert call["test_data"] is None and call["test_labels"] is None          # no validation data
        assert call["channels"] == [2, 1]
        assert (call["n_epochs"], call["learning_rate"], call["batch_size"]) == (1000, 1e-2, 32)
        assert (call["weight_decay"], call["momentum"], call["SGD"]) == (1e-5, 0.9, True)
        assert (call["normalize"], call["rescale"], call["min_train_masks"]) == (True, True, 0)
        assert call["model_name"] == f"{mode}_model_round{r}" and call["save_path"] == model_dir
    # round 0 from scratch, round 1 from the round-0 checkpoint
    assert rec.trains[0]["model"] == {"gpu": True, "model_type": None}
    assert set(rec.trains[1]["model"]) == {"gpu", "pretrained_model"}
    assert posix(rec.trains[1]["model"]["pretrained_model"]) == f"{model_dir}models/{mode}_model_round0"
