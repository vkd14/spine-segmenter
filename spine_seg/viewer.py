"""Server-side viewer figures: 3-pane MPR (matplotlib) + 3D mesh (plotly).

Replaces the previous NiiVue-in-iframe approach. Both figures render purely
server-side via Streamlit's st.pyplot / st.plotly_chart — no CDN, no JS, no
WebGL context juggling. Works reliably inside the Streamlit iframe sandbox.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import trimesh
from matplotlib.colors import ListedColormap

from .config import LABEL_COLORS, LABELS, VERTEBRA_LABELS

log = logging.getLogger(__name__)


# Discrete colormap for the 6 labels (0..5).
def _seg_colormap() -> ListedColormap:
    colors = [(0, 0, 0, 0)]  # background = transparent
    for lbl in VERTEBRA_LABELS:
        r, g, b = LABEL_COLORS[lbl]
        colors.append((r / 255.0, g / 255.0, b / 255.0, 1.0))
    return ListedColormap(colors)


def _normalize_for_display(slc: np.ndarray) -> np.ndarray:
    if slc.size == 0:
        return slc
    lo, hi = np.percentile(slc, [1, 99])
    if hi <= lo:
        return np.zeros_like(slc, dtype=np.float32)
    out = np.clip((slc - lo) / (hi - lo), 0, 1).astype(np.float32)
    return out


def _to_canonical(img):
    """Reorient a NIfTI to closest canonical RAS, returning (data, affine)."""
    can = nib.as_closest_canonical(img)
    return np.asarray(can.dataobj), can.affine


def _ras_slice(arr: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """Slice an RAS-oriented array (axes: 0=L-R, 1=P-A, 2=I-S) and return a
    2D image oriented so superior is up.

    plane: 'sagittal' (slice along axis 0), 'coronal' (axis 1), 'axial' (axis 2).
    """
    if plane == "sagittal":
        sl = arr[idx, :, :]          # shape (AP, IS)
        return np.flipud(sl.T)        # rows go S->I, cols go P->A
    if plane == "coronal":
        sl = arr[:, idx, :]          # shape (LR, IS)
        return np.flipud(sl.T)        # rows S->I, cols L->R
    # axial
    sl = arr[:, :, idx]              # shape (LR, AP)
    return np.flipud(sl.T)            # rows A->P, cols L->R


def load_canonical(volume_nifti: Path, segmentation_nifti: Optional[Path] = None):
    """Load and reorient volume + optional segmentation to canonical RAS.

    Returns (vol_array, seg_array_or_None, spacing_mm, default_centroid).
    """
    img = nib.load(str(volume_nifti))
    vol, vol_affine = _to_canonical(img)
    vol = vol.astype(np.float32)
    spacing = nib.affines.voxel_sizes(vol_affine)

    seg = None
    if segmentation_nifti is not None and Path(segmentation_nifti).exists():
        seg_img = nib.load(str(segmentation_nifti))
        seg, _ = _to_canonical(seg_img)
        seg = seg.astype(np.uint8)
        if seg.shape != vol.shape:
            log.warning("Seg %s and volume %s shape mismatch - skipping overlay",
                        seg.shape, vol.shape)
            seg = None

    ref = seg if (seg is not None and seg.any()) else (vol > 0).astype(np.uint8)
    nz = np.argwhere(ref > 0)
    if nz.size == 0:
        center = [s // 2 for s in vol.shape]
    else:
        center = nz.mean(axis=0).astype(int).tolist()
    return vol, seg, tuple(spacing), tuple(int(c) for c in center)


def build_mpr_figure(
    volume_nifti: Optional[Path] = None,
    segmentation_nifti: Optional[Path] = None,
    *,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    seg_alpha: float = 0.5,
    figsize: Tuple[float, float] = (14.0, 6.0),
    preloaded: Optional[Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float], Tuple[int, int, int]]] = None,
):
    """3-panel MPR (sagittal / coronal / axial) with optional segmentation overlay.

    If `preloaded` is given (output of load_canonical), skips disk I/O — useful
    when sliders cause many re-renders.

    `slice_indices` is (sagittal_idx, coronal_idx, axial_idx) along canonical
    RAS axes (0=LR, 1=PA, 2=IS). Falls back to centroid of labeled voxels.

    Returns a matplotlib Figure ready for st.pyplot(fig).
    """
    if preloaded is not None:
        vol, seg, spacing, default_center = preloaded
    else:
        vol, seg, spacing, default_center = load_canonical(volume_nifti, segmentation_nifti)

    if slice_indices is None:
        slice_indices = default_center
    sag_i, cor_i, ax_i = slice_indices
    # Clamp to volume bounds.
    sag_i = max(0, min(int(sag_i), vol.shape[0] - 1))
    cor_i = max(0, min(int(cor_i), vol.shape[1] - 1))
    ax_i  = max(0, min(int(ax_i),  vol.shape[2] - 1))

    cmap = _seg_colormap()
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#111111")
    # (title, plane, idx, horizontal_mm, vertical_mm)
    panel_specs = [
        ("Sagittal", "sagittal", sag_i, spacing[1], spacing[2]),
        ("Coronal",  "coronal",  cor_i, spacing[0], spacing[2]),
        ("Axial",    "axial",    ax_i,  spacing[0], spacing[1]),
    ]

    for ax, (title, plane, idx, h_mm, v_mm) in zip(axes, panel_specs):
        vol_slc = _normalize_for_display(_ras_slice(vol, plane, idx))
        aspect = (v_mm / h_mm) if h_mm > 0 else 1.0

        # Bilinear for grayscale = smooth; nearest for labels = crisp boundaries.
        ax.imshow(vol_slc, cmap="gray", interpolation="bilinear", aspect=aspect)
        if seg is not None:
            seg_slc = _ras_slice(seg, plane, idx)
            mask = seg_slc > 0
            if mask.any():
                masked = np.ma.masked_where(~mask, seg_slc)
                ax.imshow(
                    masked,
                    cmap=cmap, vmin=0, vmax=5,
                    interpolation="nearest",
                    alpha=seg_alpha,
                    aspect=aspect,
                )
        ax.set_title(f"{title} (slice {idx})", color="#dddddd", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.set_facecolor("#111111")

    fig.tight_layout()
    return fig


def build_mesh_figure(
    mesh_stls: Iterable,
    height_px: int = 600,
):
    """Plotly 3D scene with one Mesh3d trace per STL.

    `mesh_stls` is an iterable of (label_int, Path) tuples. Returns a Plotly
    Figure for st.plotly_chart(fig, use_container_width=True).
    """
    fig = go.Figure()
    have_any = False
    for item in mesh_stls:
        if isinstance(item, tuple):
            lbl, path = item
        else:
            lbl, path = None, item
        path = Path(path)
        if not path.exists():
            continue
        try:
            mesh = trimesh.load(str(path), force="mesh")
        except Exception as e:
            log.warning("Failed to load mesh %s: %s", path, e)
            continue
        if mesh is None or len(mesh.vertices) == 0:
            continue
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        color = LABEL_COLORS.get(lbl, (200, 200, 200))
        rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        name = LABELS.get(lbl, path.stem)
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=rgb,
            opacity=1.0,
            name=name,
            showlegend=True,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.7, specular=0.3, roughness=0.6),
            lightposition=dict(x=100, y=100, z=200),
        ))
        have_any = True

    if not have_any:
        # Empty placeholder so st.plotly_chart still has something to render.
        fig.add_annotation(
            text="No meshes available",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="#aaaaaa", size=14),
        )

    fig.update_layout(
        height=height_px,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#111111",
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor="#111111",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(font=dict(color="#dddddd"),
                    bgcolor="rgba(20,20,20,0.6)"),
    )
    return fig
