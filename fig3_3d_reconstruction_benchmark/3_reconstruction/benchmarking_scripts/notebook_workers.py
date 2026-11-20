# notebook_workers.py

import os
import torch
from benchmarking_scripts.multiview_segmentation_utils import segment_one_view_fullZ

def gpu_worker_one_view_notebook(
    patch_path,
    model_path,
    view,
    segment_params,
    gpu_id,
    out_queue,
    verbose=True,
):
    """Process target: predict one view of one stack on one GPU and put the result on the queue.

    Pins `CUDA_VISIBLE_DEVICES` to `gpu_id`, runs `segment_one_view_fullZ` and puts
    {"view", "flows", "probs"} (or {"view", "error"}) on `out_queue`.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if not torch.cuda.is_available():
            raise RuntimeError(f"no CUDA GPU visible as device {gpu_id}; this step needs one GPU per view")
        torch.cuda.set_device(0)
        flows, probs = segment_one_view_fullZ(
            patch_path=patch_path,
            model_path=model_path,
            view=view,
            segment_params=segment_params,
            gpu=True,
            verbose=verbose,
        )
        out_queue.put({"view": view, "flows": flows, "probs": probs})
    except Exception as e:
        out_queue.put({"view": view, "error": str(e)})
