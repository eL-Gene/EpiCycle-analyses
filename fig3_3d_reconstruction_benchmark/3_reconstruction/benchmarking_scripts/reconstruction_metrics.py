import numpy as np
import tqdm

import segment3D.metrics as uSegment3D_metrics
import segment3D.segmentation as uSegment3D_segment

def compute_AP(
    reference_list,
    predictions_list,
    iou_threshold_samples=11,
    ap50_only=True,
):
    """
    Compute Average Precision (AP), TP, FP, FN for 3D segmentations.

    Parameters
    ----------
    reference_list : list[np.ndarray]
        List of GT 3D label volumes
    predictions_list : list[np.ndarray]
        List of predicted 3D label volumes
    iou_threshold_samples : int
        Number of IoU thresholds between 0.5 and 1.0
    ap50_only : bool
        If True, return AP@0.5 only

    Returns
    -------
    ap, tp, fp, fn : lists
    """

    iou_thresholds = np.linspace(0.5, 1, iou_threshold_samples)

    ap, tp, fp, fn = [], [], [], []

    for ref, pred in tqdm.tqdm(
        zip(reference_list, predictions_list),
        total=len(reference_list),
        desc="Computing 3D metrics"
    ):
        ref_labels = uSegment3D_segment.reorder_labels_3D(ref)
        pred_labels = uSegment3D_segment.reorder_labels_3D(pred)

        ap_r, tp_r, fp_r, fn_r = uSegment3D_metrics.average_precision(
            [ref_labels * 1],
            [pred_labels],
            threshold=iou_thresholds
        )

        if ap50_only:
            ap.append(ap_r[0][0])
            tp.append(tp_r[0][0])
            fp.append(fp_r[0][0])
            fn.append(fn_r[0][0])
        else:
            ap.append(ap_r)
            tp.append(tp_r)
            fp.append(fp_r)
            fn.append(fn_r)

    return ap, tp, fp, fn
