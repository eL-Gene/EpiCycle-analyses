import gc
import numpy as np
from tqdm import tqdm
from cellpose import io, models
import torch


def segment_one_view_fullZ(
    patch_path: str,
    model_path: str,
    view: str,                  # "xy" | "xz" | "yz"
    segment_params: dict,
    gpu: bool = True,
    verbose: bool = True,
):
    """
    Run Cellpose on a single view over the FULL Z stack.
    Uses explicit indexing exactly like the original working code.
    """
    assert view in ("xy", "xz", "yz")

    # --------------------------------------------------
    # Load patch: (Z, C, Y, X)
    # --------------------------------------------------
    img = io.imread(patch_path)
    assert img.ndim == 4, f"Expected (Z,C,Y,X), got shape {img.shape}"

    Z, C, Y, X = img.shape

    # (C, Z, Y, X)
    image_xy = img.transpose(1, 0, 2, 3)

    # --------------------------------------------------
    # Prepare view-specific stacks (EXACTLY like old code)
    # --------------------------------------------------
    if view == "xy":
        stack = image_xy                    # (C, Z, Y, X)
        n_slices = Z

    elif view == "xz":
        stack = image_xy.transpose(0, 2, 1, 3)  # (C, Y, Z, X)
        n_slices = Y

    else:  # "yz"
        stack = image_xy.transpose(0, 3, 1, 2)  # (C, X, Z, Y)
        n_slices = X

    # --------------------------------------------------
    # Load Cellpose model
    # --------------------------------------------------
    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model=model_path,
    )

    flows_list = []
    probs_list = []

    desc = f"[{view.upper()}] Cellpose"
    for i in tqdm(range(n_slices), desc=desc, disable=not verbose):
        slice_img = stack[:, i, :, :]   # EXPLICIT, SAME AS OLD CODE

        _, flow_out, _ = model.eval(
            slice_img,  # (C, H, W)
            channels=segment_params["channels"],
            normalize=segment_params["normalize"],
            # rescale=segment_params["rescale"],
            rescale=True,
            cellprob_threshold=segment_params["cellprob_threshold"],
            flow_threshold=segment_params["flow_threshold"],
            batch_size=segment_params["batch_size"],
            do_3D=False,
            compute_masks=True,
            diameter=None,
        )

        _, flow_comp, cellprob = flow_out
        flows_list.append(flow_comp)
        probs_list.append(cellprob)

    # --------------------------------------------------
    # Reassemble outputs (same logic as before)
    # --------------------------------------------------
    if view == "xy":
        probs = np.stack(probs_list, axis=0)            # (Z, Y, X)
        flows = np.stack(flows_list, axis=0)            # (Z, 2, Y, X)
        flows = flows.transpose(1, 0, 2, 3)

    elif view == "xz":
        probs = np.stack(probs_list, axis=0)            # (Y, Z, X)
        probs = probs.transpose(1, 0, 2)                # (Z, Y, X)

        flows = np.stack(flows_list, axis=0)            # (Y, 2, Z, X)
        flows = flows.transpose(1, 2, 0, 3)             # (2, Z, Y, X)

    else:  # "yz"
        probs = np.stack(probs_list, axis=0)            # (X, Z, Y)
        probs = probs.transpose(1, 2, 0)                # (Z, Y, X)

        flows = np.stack(flows_list, axis=0)            # (X, 2, Z, Y)
        flows = flows.transpose(1, 2, 3, 0)             # (2, Z, Y, X)

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    del model, img, image_xy, stack
    torch.cuda.empty_cache()
    gc.collect()

    return flows.astype(np.float32), probs.astype(np.float32)
