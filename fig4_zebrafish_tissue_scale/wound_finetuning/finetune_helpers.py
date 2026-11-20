"""Slice loader and split exporter for the fine-tuning notebooks next to this file.

The trainer and its config come from the EpiCycle plugin (packages.epicycle_training).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tifffile
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening
from skimage.segmentation import find_boundaries

ViewType = Literal["XY", "XZ", "YZ"]


class BootstrappedSliceLoader:
    """Load 3D stacks and extract 2D slices from different views.

    This loader handles:
    - Loading multi-channel volumetric data
    - Transposing to XY, XZ, or YZ views
    - Optional mask cleaning (relabeling, smoothing, size filtering)
    - Edge buffering for axial views

    Args:
        image_path: Path to the image stack (Z, C, H, W) or (C, Z, H, W)
        mask_path: Path to the mask stack (Z, H, W)
        view: Orthogonal view to extract ("XY", "XZ", or "YZ")
        relabel_masks: Whether to relabel connected components
        min_size: Minimum region size to keep (pixels)
        smoothing_disk_size: Disk size for morphological smoothing (0 to disable)
        edge_buffer: Number of edge slices to exclude for axial views

    Example:
        >>> loader = BootstrappedSliceLoader(
        ...     image_path="stack.tif",
        ...     mask_path="stack_masks.tif",
        ...     view="XZ",
        ...     edge_buffer=50
        ... )
        >>> img, mask = loader.get_slice(100)
        >>> print(img.shape)  # (C, H, W)
    """

    def __init__(
        self,
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        view: ViewType = "XY",
        relabel_masks: bool = True,
        min_size: int = 100,
        smoothing_disk_size: int = 1,
        edge_buffer: int = 20,
    ):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)
        self.view = view.upper()
        self.relabel_masks = relabel_masks
        self.min_size = min_size
        self.smoothing_disk_size = smoothing_disk_size
        self.edge_buffer = edge_buffer

        # Load and process
        self.images, self.masks = self._load_and_prepare()
        self.depth = self.masks.shape[0]  # Number of slices in the view

    def _load_and_prepare(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and transpose data for the specified view."""
        image = tifffile.imread(self.image_path)  # (Z, C, H, W) or (C, Z, H, W)
        mask = tifffile.imread(self.mask_path)    # (Z, H, W)

        # Fix channel order if needed (expect Z, C, H, W)
        if image.shape[0] == 2 and image.shape[1] != 2:
            image = np.transpose(image, (1, 0, 2, 3))  # (Z, C, H, W)
        if image.ndim == 4 and image.shape[1] != 2:
            raise ValueError(f"Expected image with 2 channels, got shape {image.shape}")

        # Transpose for the desired view
        if self.view == "XY":
            pass  # Already (Z, C, H, W) and (Z, H, W)
        elif self.view == "XZ":
            # Iterate over Y dimension
            mask = np.transpose(mask, (1, 0, 2))        # (Y, Z, X) -> (D, H, W)
            image = np.transpose(image, (2, 1, 0, 3))   # (Y, C, Z, X) -> (D, C, H, W)
        elif self.view == "YZ":
            # Iterate over X dimension
            mask = np.transpose(mask, (2, 0, 1))        # (X, Z, Y) -> (D, H, W)
            image = np.transpose(image, (3, 1, 0, 2))   # (X, C, Z, Y) -> (D, C, H, W)
        else:
            raise ValueError(f"Invalid view: {self.view}. Must be 'XY', 'XZ', or 'YZ'")

        # Optionally clean masks
        if self.relabel_masks:
            mask = self._clean_mask_stack(mask)

        return image, mask

    def _clean_mask_stack(self, mask_stack: np.ndarray) -> np.ndarray:
        """Clean a stack of 2D masks (relabel, filter, smooth)."""
        cleaned_stack = np.zeros_like(mask_stack, dtype=np.uint16)

        for i in range(mask_stack.shape[0]):
            cleaned_stack[i] = self._clean_single_mask(mask_stack[i])

        return cleaned_stack

    def _clean_single_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean a single 2D mask."""
        # Remove boundaries between instances
        boundary = find_boundaries(mask, mode='inner')
        cleaned = mask.copy()
        cleaned[boundary] = 0

        # Relabel connected components
        relabeled = label(cleaned, connectivity=1)
        smoothed = np.zeros_like(relabeled)

        selem = disk(self.smoothing_disk_size) if self.smoothing_disk_size > 0 else None

        for region in regionprops(relabeled):
            if region.area >= self.min_size:
                region_mask = (relabeled == region.label)
                if selem is not None:
                    region_mask = opening(region_mask, selem)
                    region_mask = closing(region_mask, selem)
                smoothed[region_mask] = region.label

        return smoothed.astype(np.uint16)

    def get_valid_indices(self, total: int = 100) -> List[int]:
        """Get valid slice indices, respecting edge buffer for axial views.

        Args:
            total: Maximum number of indices to return (evenly spaced)

        Returns:
            List of valid slice indices
        """
        if self.view in ["XZ", "YZ"]:
            valid = np.arange(self.edge_buffer, self.depth - self.edge_buffer)
        else:
            valid = np.arange(0, self.depth)

        if len(valid) == 0:
            return []

        # Return evenly spaced indices
        return np.linspace(valid[0], valid[-1], min(total, len(valid)), dtype=int).tolist()

    def get_slice(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single slice (image and mask) at the given index.

        Args:
            index: Slice index in the current view

        Returns:
            Tuple of (image, mask) where:
                - image has shape (C, H, W)
                - mask has shape (H, W)
        """
        if index < 0 or index >= self.depth:
            raise IndexError(f"Index {index} out of range [0, {self.depth})")

        img = self.images[index]   # (C, H, W)
        mask = self.masks[index]   # (H, W)
        return img, mask

    def get_slice_cleaned(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a slice with mask cleaned individually (for lazy cleaning).

        Useful when relabel_masks=False but you want per-slice cleaning.
        """
        img, raw_mask = self.get_slice(index)
        cleaned_mask = self._clean_single_mask(raw_mask)
        return img, cleaned_mask

    def __len__(self) -> int:
        return self.depth

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_slice(index)


def export_split_dataset(
    train_imgs: List[np.ndarray],
    train_masks: List[np.ndarray],
    val_imgs: List[np.ndarray],
    val_masks: List[np.ndarray],
    output_dir: Union[str, Path],
    test_imgs: Optional[List[np.ndarray]] = None,
    test_masks: Optional[List[np.ndarray]] = None,
    metadata: Optional[Dict] = None,
    prefix: str = "sample",
):
    """Export a train/val/test split to disk.

    Creates a standardized directory structure:
        output_dir/
            train/
                images/
                masks/
            val/
                images/
                masks/
            test/  (if provided)
                images/
                masks/
            metadata.json

    Args:
        train_imgs: List of training images
        train_masks: List of training masks
        val_imgs: List of validation images
        val_masks: List of validation masks
        output_dir: Output directory path
        test_imgs: Optional list of test images
        test_masks: Optional list of test masks
        metadata: Optional metadata dict to save
        prefix: Filename prefix for samples
    """
    output_dir = Path(output_dir)

    def save_split(imgs, masks, split_name):
        img_dir = output_dir / split_name / "images"
        mask_dir = output_dir / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for i, (img, mask) in enumerate(zip(imgs, masks)):
            img_path = img_dir / f"{prefix}_{i:04d}_img.tif"
            mask_path = mask_dir / f"{prefix}_{i:04d}_mask.tif"

            tifffile.imwrite(str(img_path), img.astype(np.float32))
            tifffile.imwrite(str(mask_path), mask.astype(np.uint16))

    # Save splits
    save_split(train_imgs, train_masks, "train")
    save_split(val_imgs, val_masks, "val")

    if test_imgs is not None and test_masks is not None:
        save_split(test_imgs, test_masks, "test")

    # Build metadata
    full_metadata = {
        "created": datetime.now().isoformat(),
        "splits": {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs) if test_imgs else 0,
        },
        "total_samples": len(train_imgs) + len(val_imgs) + (len(test_imgs) if test_imgs else 0),
    }

    if metadata:
        full_metadata.update(metadata)

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(full_metadata, f, indent=4)

    print(f"Exported dataset to {output_dir}")
    print(f"  Train: {len(train_imgs)} samples")
    print(f"  Val: {len(val_imgs)} samples")
    if test_imgs:
        print(f"  Test: {len(test_imgs)} samples")


def load_exported_dataset(
    dataset_root: Union[str, Path],
) -> Tuple[
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    Dict,
]:
    """Load an exported dataset from disk.

    Args:
        dataset_root: Root directory of the exported dataset

    Returns:
        Tuple of (train_imgs, train_masks, val_imgs, val_masks,
                  test_imgs, test_masks, metadata)
    """
    dataset_root = Path(dataset_root)

    def load_split(split_name):
        img_dir = dataset_root / split_name / "images"
        mask_dir = dataset_root / split_name / "masks"

        if not img_dir.exists():
            return [], []

        imgs, masks = [], []
        img_files = sorted(img_dir.glob("*_img.tif"))

        for img_path in img_files:
            mask_path = mask_dir / img_path.name.replace("_img.tif", "_mask.tif")

            img = tifffile.imread(img_path)
            mask = tifffile.imread(mask_path)

            imgs.append(img)
            masks.append(mask)

        return imgs, masks

    train_imgs, train_masks = load_split("train")
    val_imgs, val_masks = load_split("val")
    test_imgs, test_masks = load_split("test")

    # Load metadata
    metadata_path = dataset_root / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    print(f"Loaded dataset from {dataset_root}")
    print(f"  Train: {len(train_imgs)} samples")
    print(f"  Val: {len(val_imgs)} samples")
    print(f"  Test: {len(test_imgs)} samples")

    return (
        train_imgs, train_masks,
        val_imgs, val_masks,
        test_imgs, test_masks,
        metadata,
    )
