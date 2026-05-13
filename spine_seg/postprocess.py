"""Anatomically-informed post-processing — exact 3-step procedure from the PDF.

Algorithm 1 (Section 5.4):
  1. Sagittal strip corridor: zero out labeled voxels outside +/-fraction*N_LR of
     the LR centerline. LR axis = axis with smallest span of labeled voxels.
  2. Largest connected component per label (26-connectivity), drop labels whose
     largest component is smaller than min_component_voxels.
  3. Vertical SI ordering: drop labels whose centroid violates monotonic order.
     SI axis = axis with largest span of labeled voxels.

Operates in voxel space (matches reference pipeline). Returns a new array.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from .config import PostProcessConfig, VERTEBRA_LABELS

log = logging.getLogger(__name__)


def label_summary(seg: np.ndarray) -> Dict[int, int]:
    """Voxel count per vertebra label (1..5)."""
    return {int(lbl): int((seg == lbl).sum()) for lbl in VERTEBRA_LABELS}


def find_lr_si_axes(seg: np.ndarray) -> Tuple[int, int]:
    """Identify (LR axis, SI axis) from labeled voxel spans.

    PDF Section 5.4:
      LR axis = arg min over axes of span(seg > 0, axis=a)
      SI axis = arg max over axes of span(seg > 0, axis=a)
    """
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        # No labels — fall back to a sensible default (axis 0 LR, axis 2 SI)
        return 0, 2
    spans = coords.max(0) - coords.min(0)
    lr_axis = int(np.argmin(spans))
    si_axis = int(np.argmax(spans))
    if lr_axis == si_axis:
        # Degenerate (very rare); pick a different axis for SI.
        si_axis = int(np.argmax([s if i != lr_axis else -1 for i, s in enumerate(spans)]))
    return lr_axis, si_axis


def sagittal_strip_constraint(
    seg: np.ndarray,
    lr_axis: int,
    fraction: float = 0.20,
) -> np.ndarray:
    """Zero out labels outside the central sagittal corridor.

    corridor = [c_LR - fraction*N_LR, c_LR + fraction*N_LR]
    where c_LR is the centroid of labeled voxels along the LR axis.
    """
    out = seg.copy()
    coords = np.argwhere(out > 0)
    if coords.size == 0:
        return out
    c_lr = float(coords[:, lr_axis].mean())
    n_lr = out.shape[lr_axis]
    half = fraction * n_lr
    lo = c_lr - half
    hi = c_lr + half
    # Build a 1D mask along the LR axis, then broadcast.
    idx = np.arange(n_lr)
    keep_strip = (idx >= lo) & (idx <= hi)
    # Construct the broadcast-safe shape: 1s on every axis except lr_axis.
    shape = [1] * out.ndim
    shape[lr_axis] = n_lr
    keep_strip = keep_strip.reshape(shape)
    out = np.where(keep_strip, out, 0).astype(out.dtype)
    return out


def largest_connected_component_per_label(
    seg: np.ndarray,
    min_voxels: int = 50,
    connectivity: int = 26,
) -> np.ndarray:
    """Per label l in {1..5}, keep only the largest 26-connected component.

    Drop the entire label if its largest component has fewer than min_voxels.
    """
    if connectivity == 26:
        structure = ndimage.generate_binary_structure(3, 3)  # 3x3x3, all neighbors
    elif connectivity == 18:
        structure = ndimage.generate_binary_structure(3, 2)
    else:
        structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity

    out = seg.copy()
    for lbl in VERTEBRA_LABELS:
        mask = out == lbl
        if not mask.any():
            continue
        labeled, n_cc = ndimage.label(mask, structure=structure)
        if n_cc == 0:
            continue
        # Voxel counts per CC (excluding background).
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        biggest = int(np.argmax(sizes))
        biggest_size = int(sizes[biggest])
        if biggest_size < min_voxels:
            log.debug("Label %d: largest CC has %d voxels < %d -> dropping label",
                      lbl, biggest_size, min_voxels)
            out[mask] = 0
            continue
        # Zero everything in the label except the biggest CC.
        keep = labeled == biggest
        drop_in_label = mask & ~keep
        out[drop_in_label] = 0
    return out


def enforce_vertical_ordering(seg: np.ndarray, si_axis: int) -> np.ndarray:
    """Drop labels whose SI centroid violates monotonic L1->L5 ordering.

    The dominant direction (ascending vs descending SI index) is inferred from
    the present labels' centroids. Any label that breaks the monotonic chain is
    removed.
    """
    out = seg.copy()
    centroids: Dict[int, float] = {}
    for lbl in VERTEBRA_LABELS:
        coords = np.argwhere(out == lbl)
        if coords.size == 0:
            continue
        centroids[lbl] = float(coords[:, si_axis].mean())

    if len(centroids) < 2:
        return out

    present = sorted(centroids.keys())  # sorted by label value (L1..L5)
    si_values = [centroids[l] for l in present]

    # Decide direction: if labels are ascending in label-order, SI should
    # consistently increase or decrease. Pick whichever majority agrees with.
    asc_ok = sum(si_values[i] < si_values[i + 1] for i in range(len(si_values) - 1))
    desc_ok = sum(si_values[i] > si_values[i + 1] for i in range(len(si_values) - 1))
    descending = desc_ok >= asc_ok
    log.debug("Ordering: descending=%s (asc_ok=%d, desc_ok=%d)", descending, asc_ok, desc_ok)

    # Greedy monotonic chain: keep labels that respect direction wrt previous kept.
    kept: list[int] = [present[0]]
    for lbl in present[1:]:
        prev_lbl = kept[-1]
        prev_val = centroids[prev_lbl]
        cur_val = centroids[lbl]
        ok = (cur_val < prev_val) if descending else (cur_val > prev_val)
        if ok:
            kept.append(lbl)
        else:
            log.debug("Ordering violation: label %d centroid=%.1f vs prev %d centroid=%.1f -> dropping %d",
                      lbl, cur_val, prev_lbl, prev_val, lbl)

    dropped = [l for l in present if l not in kept]
    for lbl in dropped:
        out[out == lbl] = 0
    return out


def postprocess(
    seg: np.ndarray,
    cfg: PostProcessConfig | None = None,
) -> np.ndarray:
    """Run all 3 PDF post-processing steps in order, return cleaned array."""
    cfg = cfg or PostProcessConfig()
    lr_axis, si_axis = find_lr_si_axes(seg)
    log.info("Post-processing: LR axis=%d, SI axis=%d", lr_axis, si_axis)

    out = sagittal_strip_constraint(seg, lr_axis, fraction=cfg.sagittal_strip_fraction)
    out = largest_connected_component_per_label(
        out, min_voxels=cfg.min_component_voxels, connectivity=cfg.cc_connectivity
    )
    out = enforce_vertical_ordering(out, si_axis=si_axis)
    return out
