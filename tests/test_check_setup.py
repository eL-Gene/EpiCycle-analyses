"""tools/check_setup.py: the check a reader runs on a fresh clone before the notebooks."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import check_setup  # noqa: E402


def _make(root: Path, rels):
    for rel in rels:
        p = root / rel
        if Path(rel).suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        else:
            p.mkdir(parents=True, exist_ok=True)
            (p / "a.tif").write_bytes(b"x")


def test_layout_mirrors_data_readme():
    """Every top folder the layout names is a row of data/README.md's table, and nothing is listed twice."""
    readme = (REPO / "data" / "README.md").read_text(encoding="utf-8")
    seen = set()
    for fig, roles in check_setup.LAYOUT.items():
        for rel in roles["reads"] + roles.get("optional", []) + roles["writes"]:
            assert rel not in seen, f"{rel} listed twice"
            seen.add(rel)
            assert rel.split("/")[0] in readme, f"{rel}: its top folder is not in data/README.md"


def test_missing_required_paths_are_counted(tmp_path):
    rows, missing = check_setup.check_data(tmp_path, ["fig6_7"])
    required = [r for r in rows if r[1] == "required"]
    assert missing == len(required) == len(check_setup.LAYOUT["fig6_7"]["reads"])
    assert all(r[3] == "MISSING" for r in required)
    assert all(r[3].startswith("absent") for r in rows if r[1] == "regenerated")


def test_complete_layout_passes_and_counts_files(tmp_path):
    _make(tmp_path, check_setup.LAYOUT["fig6_7"]["reads"] + check_setup.LAYOUT["fig6_7"]["writes"])
    rows, missing = check_setup.check_data(tmp_path, ["fig6_7"])
    assert missing == 0
    by_path = {r[2]: r[3] for r in rows}
    assert by_path["fig6_7/all_stacks_features.csv"].startswith("file,")
    assert by_path["fig6_7/workspace/artifacts/step4/meshes"] == "1 files"


def test_fig3_needs_nothing_from_the_deposit_for_the_plots(tmp_path):
    """Step 5 reads 4_metrics/ in git: an empty data folder is not a failure; the GPU-step inputs are optional."""
    rows, missing = check_setup.check_data(tmp_path, ["fig3"])
    assert missing == 0
    assert not any(r[1] == "required" for r in rows)
    optional = [r for r in rows if r[1] == "optional"]
    assert {r[2] for r in optional} == {"fig3/bootstrap_lateral_models", "fig3/bootstrap_axial_models"}
    assert all("4_metrics" in r[3] for r in optional)


def test_imports_are_collected_from_every_figure():
    rows, _ = check_setup.check_imports(check_setup.FIGURE_FOLDERS.values())
    modules = {r[0] for r in rows}
    assert {"numpy", "pandas", "cellpose", "umap", "leidenalg", "packages"} <= modules
    assert not any(m in check_setup.STDLIB for m in modules)
    assert any("module next to the notebook" in r[1] for r in rows)          # finetune_helpers, benchmarking_scripts


def test_main_exit_code_follows_the_data(tmp_path, capsys):
    _make(tmp_path, check_setup.LAYOUT["fig4"]["reads"])
    code = check_setup.main(["--data", str(tmp_path), "--figure", "fig4"])
    out = capsys.readouterr().out
    assert "data root:" in out and "== imports" in out
    data_ok = "required data path(s) missing" not in out
    assert data_ok
    assert code in (0, 1)                                 # 1 only when this interpreter lacks a module
    if code == 1:
        assert "module(s) missing" in out


def test_main_rejects_an_unknown_figure():
    with pytest.raises(SystemExit):
        check_setup.main(["--figure", "fig9"])
