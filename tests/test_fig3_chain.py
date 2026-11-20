"""Checks for the Fig. 3 chain: the training notebooks carry the settings of Table S5, the
model names agree from training to reconstruction to the metric tables to the plots, and the
committed metric tables carry the Table S2b values.

Run from the repository root:  python -m pytest tests -q
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "fig3_3d_reconstruction_benchmark"
SPLITS, TRAIN, RECON, METRICS, PLOTS = (FIG / d for d in ["1_dense_splits", "2_training", "3_reconstruction", "4_metrics", "5_plots"])


def code_cells(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def notebook_code(path: Path) -> str:
    return "\n".join(code_cells(path))


def kwargs_of_calls(src: str, fn="train_cellpose_model("):
    """Every keyword=value pair of each call of fn in src, as a list of dicts (literal text)."""
    out = []
    src = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))   # skip commented-out calls
    for m in re.finditer(r"(?<!def )" + re.escape(fn), src):
        body = src[m.end():]
        depth, i = 1, 0
        while depth and i < len(body):
            depth += {"(": 1, ")": -1}.get(body[i], 0); i += 1
        call = body[:i - 1]
        kv = {}
        for line in call.split("\n"):
            line = line.split("#")[0].strip().rstrip(",")
            if "=" in line:
                k, v = line.split("=", 1); kv[k.strip()] = v.strip()
        out.append(kv)
    return out


# ----------------------------------------------------------------------------- layout
def test_folders_are_in_run_order():
    assert [p.name for p in sorted(FIG.iterdir()) if p.is_dir()] == [
        "1_dense_splits", "2_training", "3_reconstruction", "4_metrics", "5_plots"]


def test_every_notebook_resolves_the_data_root_the_same_way():
    for nb in FIG.rglob("*.ipynb"):
        first = code_cells(nb)[0]
        assert "EPICYCLE_ANALYSES_DATA" in first and '/ "fig3"' in first, nb.name
    for script in PLOTS.glob("*.py"):
        assert 'SCRIPT_DIR.parent / "4_metrics"' in script.read_text(encoding="utf-8"), script.name


def test_stacks_are_read_from_the_shared_folder():
    for nb in [SPLITS / "prepare_dense_2d_splits.ipynb", *RECON.glob("*.ipynb")]:
        code = notebook_code(nb)
        assert "../ground_truth/zebrafish/development" in code and "../ground_truth/zebrafish/held_out" in code, nb.name
        assert not re.search(r"['\"]data_holdout|['\"]data/|listdir\('data'\)", code), nb.name


def test_no_machine_paths_or_emoji():
    emoji = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
    machine = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/]|/gpfs/|/home/[a-z]|/content/drive)")
    for p in FIG.rglob("*"):
        if p.suffix in (".ipynb", ".py", ".md"):
            s = p.read_text(encoding="utf-8")
            assert not emoji.search(s), p
            assert not machine.search(s), p


# ----------------------------------------------------------------------------- Table S5
EXPECTED_TRAIN = {  # notebook -> (dataset folder, view folder, pretrained, learning rate)
    "train_scratch_lateral_11stacks_lenient": ("decomposed2D_benchmark-lenient_splits", "lateral_dataset_v1", "None", "1e-2"),
    "train_scratch_lateral_11stacks_strict": ("decomposed2D_benchmark-strict_splits", "lateral_dataset_v1", "None", "1e-2"),
    "train_scratch_lateral_7stacks_lenient": ("decomposed2D_benchmark-7_patch-lenient_splits", "lateral_dataset_v1", "None", "1e-2"),
    "train_scratch_lateral_7stacks_strict": ("decomposed2D_benchmark-7_patch-strict_splits", "lateral_dataset_v1", "None", "1e-2"),
    "train_scratch_axial_11stacks_lenient": ("decomposed2D_benchmark-lenient_splits", "axial_dataset_v1", "None", "1e-2"),
    "train_scratch_axial_11stacks_strict": ("decomposed2D_benchmark-strict_splits", "axial_dataset_v1", "None", "1e-2"),
    "train_scratch_axial_7stacks_lenient": ("decomposed2D_benchmark-7_patch-lenient_splits", "axial_dataset_v1", "None", "1e-2"),
    "train_scratch_axial_7stacks_strict": ("decomposed2D_benchmark-7_patch-strict_splits", "axial_dataset_v1", "None", "1e-2"),
    "finetune_bootstrap_round4_11stacks_lenient": ("decomposed2D_benchmark-lenient_splits", None, "bootstrap", "1e-2"),
    "finetune_bootstrap_round4_11stacks_strict": ("decomposed2D_benchmark-strict_splits", None, "bootstrap", "1e-2"),
    "finetune_cyto3_lateral": ("decomposed2D_benchmark-lenient_splits", "lateral_dataset_v1", "None", ["1e-2", "1e-1"]),
    "finetune_cyto3_axial": ("decomposed2D_benchmark-lenient_splits", "axial_dataset_v1", "None", ["1e-2", "1e-1"]),
}


def test_training_notebooks_match_table_s5():
    for stem, (dataset, view, pretrained, lr) in EXPECTED_TRAIN.items():
        code = notebook_code(TRAIN / f"{stem}.ipynb")
        assert dataset in code, stem
        if view:
            assert view in code, stem
        calls = kwargs_of_calls(code)
        assert len(calls) >= 1, stem
        assert [kv["learning_rate"] for kv in calls] == (lr if isinstance(lr, list) else [lr] * len(calls)), stem
        for kv in calls:
            assert kv["n_epochs"] == "500", (stem, kv)
            assert kv["batch_size"] == "16", (stem, kv)
            assert kv["weight_decay"] == "1e-5" and kv["momentum"] == "0.9", (stem, kv)
            assert kv["channels"] == "[2,1]", (stem, kv)
            assert kv["val_data"] == "val_imgs" and kv["val_labels"] == "val_masks", (stem, "validation split must be passed")
            assert kv["min_train_masks"] == "0", (stem, kv)
            if pretrained == "None":
                assert kv["pretrained_model_path"] == "None", (stem, kv)
            else:
                assert "model_round4" in kv["pretrained_model_path"], (stem, kv)
        if stem.startswith("train_scratch"):
            assert [kv["rescale"] for kv in calls] == ["True"], stem   # the rescale=False runs were deleted from the notebook and the deposit
    # the cyto3 notebook starts from the cyto3 weights, not from scratch
    for view in ("lateral", "axial"):
        assert "model_type='cyto3'" in notebook_code(TRAIN / f"finetune_cyto3_{view}.ipynb")


def test_split_fractions_match_methods_3_4():
    code = notebook_code(SPLITS / "prepare_dense_2d_splits.ipynb")
    assert code.count("splits=[0.4, 0.3, 0.3]") == 4 and code.count("splits=[0.8, 0.2, 0.0]") == 4
    assert 'spatial_split="chunked"' in code
    assert 'edge_buffer = 50 if view in ["XZ", "YZ"] else 0' in code


# ----------------------------------------------------------------------------- names agree
MODEL_IDS_BOOT = ["bootstrap_models0", "bootstrap_models1", "bootstrap_models2"] + [
    f"500epoch_scratch_{s}_rescale_True" for s in ("7patch_lenient", "7patch_strict", "lenient", "strict")]
MODEL_IDS_FT = ["bootstrap_models_round4-0", "bootstrap_models_round4-1",
                "500epoch_scratch_lenient_CPcyto3_Finetuned", "500epoch_scratch_lenient_CPcyto3_Finetuned-HLR"]


def test_model_names_agree_from_training_to_tables_to_plots():
    # training writes <folder>/models/<model_name>; the reconstruction notebooks list those folders
    names = []
    for nb in TRAIN.glob("*.ipynb"):
        for kv in kwargs_of_calls(notebook_code(nb)):
            names.append((kv["model_save_path"].strip('"'), kv["model_name"].strip('"')))
    folders = {f for f, _ in names}
    assert folders == {"domain-specialized_lateral_models/rescale_true", "domain-specialized_axial_models/rescale_true",
                       "domain-specialized_finetuned_bootstrap", "domain-specialized_finetuned_CPCyto3"}
    recon = notebook_code(RECON / "reconstruct_benchmark.ipynb") + notebook_code(RECON / "reconstruct_benchmark_finetuned.ipynb")
    for f in folders:
        assert f.split("/")[0] in recon, f
    # the metric tables carry the model ids the reconstruction notebooks build, and the plots name them
    boot = pd.read_csv(METRICS / "3DRecon_metrics_BootStrapped.csv")
    ft = pd.read_csv(METRICS / "3DRecon_Metrics_Finetuned.csv")
    assert sorted(boot.model_id.unique()) == sorted(MODEL_IDS_BOOT)
    assert sorted(ft.model_id.unique()) == sorted(MODEL_IDS_FT)
    stats = (PLOTS / "compute_boxplot_stats.py").read_text(encoding="utf-8")
    for mid in MODEL_IDS_BOOT + MODEL_IDS_FT:
        assert f'"{mid}"' in stats, mid
    # the lateral/axial checkpoint names in the tables are the training model names
    trained = {n for _, n in names}
    for col in ("lateral_model", "axial_model"):
        for path in pd.concat([boot, ft])[col].unique():
            leaf = path.rstrip("/").split("/")[-1]
            assert leaf in trained or leaf.endswith(("model_round0", "model_round4", "model_round9")), leaf


# ----------------------------------------------------------------------------- Table S2b
def test_metric_tables_are_well_formed_and_carry_table_s2b():
    boot = pd.read_csv(METRICS / "3DRecon_metrics_BootStrapped.csv")
    ft = pd.read_csv(METRICS / "3DRecon_Metrics_Finetuned.csv")
    df = pd.concat([boot, ft])
    stacks = {"patch-1", "patch-2", "patch-3", "patch-4", "patch-5", "patch-7", "patch-8", "patch-6", "patch-9", "patch-10", "patch-14"}
    assert set(df.patch) == stacks
    assert (df.groupby("model_id").size() == 11).all() and df.model_id.nunique() == 11
    assert df.ap50.between(0, 1).all() and (df[["tp", "fp", "fn"]] >= 0).all().all()
    assert ((df.tp / (df.tp + df.fp + df.fn)) - df.ap50).abs().max() < 1e-6   # ap50 = tp / (tp + fp + fn)
    held = df[df.patch_path.str.contains("holdout|held_out")]
    assert set(held.patch) == {"patch-6", "patch-9", "patch-10", "patch-14"}
    # a few cells of Table S2b (3D AP@50 per stack), as published
    def ap(mid, patch): return round(float(df[(df.model_id == mid) & (df.patch == patch)].ap50.iloc[0]), 3)
    assert ap("bootstrap_models0", "patch-1") == 0.427 and ap("bootstrap_models0", "patch-9") == 0.845
    assert ap("bootstrap_models2", "patch-9") == 0.909 and ap("bootstrap_models2", "patch-14") == 0.440
    assert ap("bootstrap_models_round4-0", "patch-10") == 0.932 and ap("bootstrap_models_round4-0", "patch-6") == 0.678
    assert ap("500epoch_scratch_lenient_CPcyto3_Finetuned", "patch-2") == 0.313
    assert ap("500epoch_scratch_lenient_rescale_True", "patch-14") == 0.701
    assert ap("500epoch_scratch_7patch_strict_rescale_True", "patch-3") == 0.732


def test_boxplot_stats_script_reproduces_the_committed_tables(tmp_path):
    work = tmp_path / "5_plots"; shutil.copytree(PLOTS, work)
    shutil.copytree(METRICS, tmp_path / "4_metrics")
    r = subprocess.run([sys.executable, str(work / "compute_boxplot_stats.py")], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONUTF8="1"))
    assert r.returncode == 0, r.stderr
    for name in ("boxplot_stats_main.csv", "boxplot_stats_supp.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(work / name), pd.read_csv(PLOTS / name), check_exact=False, atol=5e-4)


# ----------------------------------------------------------------------------- executed: Table S5 training
# The training notebooks are executed cell by cell with Cellpose replaced by recorders
# (tests/_nbexec.py) and the exported datasets they load written under tmp_path as tiny TIFFs whose
# pixels carry a tag (dataset, split, index): the tags tell which split reached train_seg as what.
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from _nbexec import (CellposeRecorder, fake_torch, needs_data_root, only_cell, posix,  # noqa: E402
                     run_cells, tag_of)

SCRATCH = {"gpu": True, "model_type": None}
CYTO3 = {"gpu": True, "model_type": "cyto3", "nchan": 2}


def _dataset(view, n_stacks, regime):
    seven = "7_patch-" if n_stacks == 7 else ""
    return f"decomposed2D_benchmark-{seven}{regime}_splits/{view}_dataset_v1"


def _scratch_runs():
    runs = {}
    for view in ("lateral", "axial"):
        for n in (11, 7):
            for regime in ("lenient", "strict"):
                runs[f"train_scratch_{view}_{n}stacks_{regime}"] = [
                    (_dataset(view, n, regime), SCRATCH, 1e-2, f"domain-specialized_{view}_models/rescale_true")]
    return runs


# notebook -> the train_seg calls in order: (dataset loaded, model start, learning rate, output folder)
# model start: the CellposeModel kwargs (from scratch, from cyto3, or from the round-4 HITL checkpoint)
EXPECTED_RUNS = {
    **_scratch_runs(),
    **{f"finetune_bootstrap_round4_11stacks_{regime}": [
        (_dataset("lateral", 11, regime), {"gpu": True, "pretrained_model": "bootstrap_lateral_models/spatial_model_round4"},
         1e-2, "domain-specialized_finetuned_bootstrap"),
        (_dataset("axial", 11, regime), {"gpu": True, "pretrained_model": "bootstrap_axial_models/axial_model_round4"},
         1e-2, "domain-specialized_finetuned_bootstrap")] for regime in ("lenient", "strict")},
    **{f"finetune_cyto3_{view}": [
        (_dataset(view, 11, "lenient"), CYTO3, 1e-2, "domain-specialized_finetuned_CPCyto3"),
        (_dataset(view, 11, "lenient"), CYTO3, 1e-1, "domain-specialized_finetuned_CPCyto3")] for view in ("lateral", "axial")},
}
DATASETS = sorted({d for calls in EXPECTED_RUNS.values() for d, *_ in calls})
SPLIT_SIZES = {"train": 3, "val": 2, "test": 1}
SPLIT_CODE = {"train": 1, "val": 2, "test": 3}


def _tag(dataset, split, i):
    return 100 * DATASETS.index(dataset) + 10 * SPLIT_CODE[split] + i + 1


def _write_exported_datasets(fig3: Path):
    for d in DATASETS:
        root = fig3 / "decomposed_2D" / d
        for split, n in SPLIT_SIZES.items():
            (root / split / "images").mkdir(parents=True); (root / split / "masks").mkdir(parents=True)
            for i in range(n):
                t = _tag(d, split, i)
                tifffile.imwrite(root / split / "images" / f"s{i}_img.tif", np.full((2, 4, 4), t, np.uint16))
                tifffile.imwrite(root / split / "masks" / f"s{i}_mask.tif", np.full((4, 4), t, np.uint16))
        (root / "metadata.json").write_text("{}")


def test_every_training_notebook_is_listed():
    stems = sorted(p.stem for p in TRAIN.glob("*.ipynb"))
    assert stems == sorted(EXPECTED_TRAIN) == sorted(EXPECTED_RUNS)


@needs_data_root
@pytest.mark.parametrize("stem", sorted(EXPECTED_RUNS))
def test_training_notebook_trains_as_in_table_s5(stem, tmp_path, monkeypatch):
    rec = CellposeRecorder(monkeypatch)
    fake_torch(monkeypatch)
    _write_exported_datasets(tmp_path / "fig3")
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(TRAIN)
    monkeypatch.setattr(sys, "path", list(sys.path))            # the first cell inserts its folder

    ns = {}
    cells = code_cells(TRAIN / f"{stem}.ipynb")
    run_cells(cells, ns, stop_after=cells[-1])                  # the whole notebook
    expected = EXPECTED_RUNS[stem]
    assert len(rec.trains) == len(expected), stem
    for call, (dataset, start, lr, out) in zip(rec.trains, expected):
        train_tags = [_tag(dataset, "train", i) for i in range(3)]
        val_tags = [_tag(dataset, "val", i) for i in range(2)]
        assert [tag_of(a) for a in call["train_data"]] == train_tags, (stem, dataset)
        assert [tag_of(a) for a in call["train_labels"]] == train_tags
        assert [tag_of(a) for a in call["test_data"]] == val_tags, "the validation split is passed as test_data"
        assert [tag_of(a) for a in call["test_labels"]] == val_tags
        model = {k: (posix(v) if k == "pretrained_model" else v) for k, v in call["model"].items()}
        assert model == start, (stem, model)
        assert (call["n_epochs"], call["learning_rate"], call["batch_size"]) == (500, lr, 16), stem
        assert (call["weight_decay"], call["momentum"], call["SGD"]) == (1e-5, 0.9, True), stem
        assert (call["normalize"], call["rescale"], call["min_train_masks"]) == (True, True, 0), stem
        assert call["channels"] == [2, 1] and call["save_path"] == out, stem


# ----------------------------------------------------------------------------- executed: Table S6 reconstruction
RECON_NBS = ["reconstruct_benchmark", "reconstruct_benchmark_finetuned"]


def _recon_params(stem):
    params = pytest.importorskip("segment3D.parameters")
    cells = code_cells(RECON / f"{stem}.ipynb")
    assert any("import segment3D.parameters as uSegment3D_params" in c for c in cells)
    ns = {"uSegment3D_params": params}
    exec(compile(cells[only_cell(cells, "base_segment_params = {")], "<cell>", "exec"), ns)
    return cells, ns


@pytest.mark.parametrize("stem", RECON_NBS)
def test_reconstruction_parameters_match_table_s6(stem):
    _, ns = _recon_params(stem)
    seg = ns["base_segment_params"]
    assert (seg["channels"], seg["normalize"], seg["cellprob_threshold"], seg["flow_threshold"]) == ([2, 1], True, 0.0, 0.4)
    agg = ns["direct_aggregation_params"]
    p, g, gd = agg["combine_cell_probs"], agg["combine_cell_gradients"], agg["gradient_descent"]
    assert (p["ksize"], p["alpha"], p["smooth_sigma"], p["cellpose_prob_mask"]) == (25, 0.5, 0.8, True)
    assert (g["ksize"], g["smooth_sigma"], g["post_sigma"]) == (25, 1.5, 0.5)
    assert (gd["n_iter"], gd["momenta"], gd["gradient_decay"], gd["renorm_gradient"]) == (50, 0.98, 0.001, False)
    assert agg["postprocess_binary"]["binary_fill_holes"] is True
    assert ns["post_process_params"]["flow_consistency"]["do_flow_remove"] is False


@pytest.mark.parametrize("stem", RECON_NBS)
def test_reconstruction_passes_those_parameters_to_u_segment3d(stem, tmp_path, monkeypatch):
    """Run the notebook's loop over one stack and one pair, the view workers and u-Segment3D faked in-process."""
    import multiprocessing, queue
    cells, ns = _recon_params(stem)
    exec(compile(cells[only_cell(cells, "def run_one_patch_one_pair_notebook(")], "<cell>", "exec"), ns)

    workers, solver = [], {}

    def fake_worker(patch_path, model_path, view, segment_params, gpu_id, out_queue, verbose):
        workers.append(dict(view=view, model_path=model_path, segment_params=segment_params))
        out_queue.put({"view": view, "probs": f"probs-{view}", "flows": f"flows-{view}"})

    class InlineProcess:
        def __init__(self, target, kwargs):
            self.target, self.kwargs = target, kwargs

        def start(self):
            self.target(**self.kwargs)

        def join(self):
            pass

    class FakeUSegment3D:
        @staticmethod
        def aggregate_2D_to_3D_segmentation_direct_method(probs, gradients, params):
            solver["aggregate"] = dict(probs=probs, gradients=gradients, params=params)
            return np.ones((2, 2, 2), np.uint32), ("prob3D", "grad3D")

        @staticmethod
        def postprocess_3D_cell_segmentation(seg, aggregation_params, postprocess_params, cell_gradients, savefolder, basename):
            solver["post"] = dict(aggregation_params=aggregation_params, postprocess_params=postprocess_params,
                                  cell_gradients=cell_gradients)
            return seg, None

    monkeypatch.setattr(multiprocessing, "Process", InlineProcess)
    monkeypatch.setattr(multiprocessing, "Queue", queue.Queue)
    ns.update(gpu_worker_one_view_notebook=fake_worker, uSegment3D=FakeUSegment3D, np=np, tifffile=tifffile,
              json=json, Path=Path)
    pair = {"id": "m0", "lateral": "lat/model", "axial": "ax/model", "rescale": True, "group": "g"}

    loop = cells[only_cell(cells, "for mpair in model_pairs:")]
    out_root = re.search(r'out_root="([^"]+)"', loop).group(1)
    assert out_root == {"reconstruct_benchmark": "benchmark_outputs",
                        "reconstruct_benchmark_finetuned": "benchmark_outputs_finetuned"}[stem]
    monkeypatch.chdir(tmp_path)
    ns.update(patches=["stacks/patch-1.tif"], model_pairs=[pair])
    exec(compile(loop, "<cell>", "exec"), ns)

    assert [(w["view"], w["model_path"]) for w in workers] == [("xy", "lat/model"), ("xz", "ax/model"), ("yz", "ax/model")]
    for w in workers:
        assert w["segment_params"] == dict(ns["base_segment_params"], rescale=True)
    assert solver["aggregate"]["params"] is ns["direct_aggregation_params"]
    assert solver["aggregate"]["probs"] == ["probs-xy", "probs-xz", "probs-yz"]
    assert solver["aggregate"]["gradients"] == ["flows-xy", "flows-xz", "flows-yz"]
    assert solver["post"]["aggregation_params"] is ns["direct_aggregation_params"]
    assert solver["post"]["postprocess_params"] is ns["post_process_params"]
    assert solver["post"]["cell_gradients"] == "grad3D"
    assert (tmp_path / out_root / "patch-1" / "m0_mask.tif").exists()


# ----------------------------------------------------------------------------- executed: the dense splits (Methods 3.4)
DEV_STACKS = ["patch-1", "patch-2", "patch-3", "patch-4", "patch-5", "patch-7", "patch-8"]
HELD_OUT = ["patch-10", "patch-14", "patch-6", "patch-9"]          # sorted as names, as the notebook lists them
SPLITS_OF = {"strict": [0.4, 0.3, 0.3], "lenient": [0.8, 0.2, 0.0]}


def _run_splits_notebook(tmp_path, monkeypatch):
    gt = tmp_path / "ground_truth" / "zebrafish"
    (gt / "development").mkdir(parents=True); (gt / "held_out").mkdir(parents=True)
    for s in DEV_STACKS:                                     # only listed: the loader is faked
        (gt / "development" / f"{s}.tif").touch(); (gt / "development" / f"{s}_masks.tif").touch()
    for s in HELD_OUT:
        (gt / "held_out" / f"{s}.tif").touch(); (gt / "held_out" / f"{s}_mask.tif").touch()
    (tmp_path / "fig3").mkdir()
    monkeypatch.setenv("EPICYCLE_ANALYSES_DATA", str(tmp_path))
    monkeypatch.chdir(SPLITS)
    monkeypatch.setattr(sys, "path", list(sys.path))
    CellposeRecorder(monkeypatch)                            # the notebook imports cellpose, never calls it
    cells = code_cells(SPLITS / "prepare_dense_2d_splits.ipynb")
    ns = {}
    run_cells(cells, ns, stop_after="def load_exported_dataset(")
    return cells, ns


@needs_data_root
def test_split_notebook_builds_the_eight_regimes(tmp_path, monkeypatch):
    cells, ns = _run_splits_notebook(tmp_path, monkeypatch)
    assert [s["name"] for s in ns["stacks1"]] == DEV_STACKS
    assert [s["name"] for s in ns["stacks"]] == DEV_STACKS + HELD_OUT

    loaders, prepared, exported = [], [], []

    class FakeLoader:
        def __init__(self, **kw):
            loaders.append(kw)
            self.info = {"stack": kw["stack_name"], "view": kw["view"], "slice_idx": 0}

        def _one(self):
            return [np.zeros((2, 2, 2))], [np.zeros((2, 2))], [self.info]
        get_training = get_validation = get_test_data = _one

    real_prepare = ns["prepare_dense_data_across_stacks"]

    def prepare(**kw):
        first = len(loaders)
        out = real_prepare(**kw)
        prepared.append(dict(kw, loaders=loaders[first:]))
        return out

    ns.update(LoadTrainingData=FakeLoader, prepare_dense_data_across_stacks=prepare,
              export_split_dataset=lambda **kw: exported.append(kw))
    start = only_cell(cells, "def load_exported_dataset(") + 1
    for src in cells[start:]:
        exec(compile(src, "<cell>", "exec"), ns)

    expected = [(view, n, regime) for view in ("lateral", "axial") for n in (11, 7) for regime in ("strict", "lenient")]
    assert len(prepared) == len(exported) == 8
    for call, exp, (view, n, regime) in zip(prepared, exported, expected):
        seven = "7_patch-" if n == 7 else ""
        folder = f"decomposed_2D/decomposed2D_benchmark-{seven}{regime}_splits"
        assert call["save_dir"] == folder and call["splits"] == SPLITS_OF[regime], (view, n, regime)
        assert call["mode"] == ("spatial" if view == "lateral" else "axial")
        assert [s["name"] for s in call["stacks"]] == (DEV_STACKS if n == 7 else DEV_STACKS + HELD_OUT), (view, n, regime)
        views = ["XY"] if view == "lateral" else ["XZ", "YZ"]
        assert [(ld["stack_name"], ld["view"]) for ld in call["loaders"]] == [(s["name"], v) for s in call["stacks"] for v in views]
        for ld in call["loaders"]:
            assert [ld["train_split"], ld["validation_split"], ld["test_split"]] == SPLITS_OF[regime]
            assert ld["spatial_split"] == "chunked" and ld["edge_buffer"] == (0 if ld["view"] == "XY" else 50)
            assert (ld["relabel_masks"], ld["size_filter"], ld["smoothing_disk_size"]) == (True, 100, 1)
        assert exp["dataset_root"] == folder and exp["dataset_name"] == f"{view}_dataset_v1"
        assert exp["metadata"]["split_type"] == regime


@needs_data_root
def test_split_notebook_splits_in_chunks_of_ten(tmp_path, monkeypatch):
    """The notebook's own LoadTrainingData on 40 synthetic slices: every block of 10 is split 4 / 3 / 3."""
    _, ns = _run_splits_notebook(tmp_path, monkeypatch)

    class Synthetic(ns["LoadTrainingData"]):
        def load_data(self):
            n = 40
            return [np.zeros((2, 2, 2))] * n, [np.zeros((2, 2))] * n, [{"slice_idx": i} for i in range(n)]

    np.random.seed(0)
    ld = Synthetic(stack_name="s", image_path="i.tif", mask_path="m.tif", view="XY", spatial_split="chunked",
                   train_split=0.4, validation_split=0.3, test_split=0.3)
    for block in range(0, 40, 10):
        counts = [sum(block <= info["slice_idx"] < block + 10 for _, _, info in part)
                  for part in (ld.train, ld.validation, ld.test)]
        assert counts == [4, 3, 3], (block, counts)
