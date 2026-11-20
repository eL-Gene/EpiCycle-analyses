"""Make the workspace-relative record paths in stack_segmentation_state.json absolute.

Usage: python fix_step4_paths.py <workspace>   (the folder holding stack_segmentation_state.json)
Also writes workspace_root and moves an existing step4_state.json aside (.pre-fix), so the
EpiCycle plugin migrates the fixed legacy file instead of reading a stale one.
"""
import json, os, re, sys, time
from pathlib import Path

KEYS = ("source_stack_path", "labels_path", "prob_path", "qc_features_csv",
        "original_labels_path", "smoothed_cache_path", "border_cells_path")
ws = Path(sys.argv[1]).resolve()
state = ws / "stack_segmentation_state.json"
d, n = json.loads(state.read_text(encoding="utf-8-sig")), 0

def absolute(v):
    global n
    if not isinstance(v, str) or not v or os.path.isabs(v) or re.match(r"[A-Za-z]:[\\/]", v):
        return v
    n += 1
    return os.path.normpath(os.path.join(ws, v.replace("\\", "/")))

for rec in (d.get("stacks") or {}).values():
    for k in KEYS:
        if k in rec:
            rec[k] = absolute(rec[k])
session = d.get("session") or {}
if "input_stacks_dir" in session: session["input_stacks_dir"] = absolute(session["input_stacks_dir"])
if isinstance(session.get("stack_folders"), list):
    session["stack_folders"] = [absolute(v) for v in session["stack_folders"]]
d["workspace_root"] = str(ws)
for old in (p for p in (ws / "step4_state.json", ws / "artifacts/step4/step4_state.json") if p.exists()):
    aside = old.with_name("step4_state.json.pre-fix" + (time.strftime("-%Y%m%d%H%M%S") if old.with_name("step4_state.json.pre-fix").exists() else ""))
    old.rename(aside)
    print(f"moved aside: {old} -> {aside.name}")
tmp = state.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
os.replace(tmp, state)
print(f"{state}: {n} paths made absolute under {ws}; workspace_root set")
