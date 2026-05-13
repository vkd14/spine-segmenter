"""Render all figures the presentation needs from the real pipeline outputs."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spine_seg.config import LABEL_COLORS, LABELS, VERTEBRA_LABELS
from spine_seg.viewer import build_mpr_figure, load_canonical
from spine_seg.postprocess import (
    find_lr_si_axes, label_summary, largest_connected_component_per_label,
    postprocess, sagittal_strip_constraint,
)

FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

VOL = Path("/Volumes/thev_ssd/BMD_T1_Flair-main/Sagittal_T1_FLAIR_inference_output/_nnunet_workdir/nnunet_input/542_20240612_Lumbar_Spine_Sagittal_T1_FLAIR_s3_0000.nii.gz")
SEG = ROOT / "results/smoke_test_542/segmentation.nii.gz"
RAW = ROOT / "results/smoke_test_542/raw_prediction/542_20240612_Lumbar_Spine_Sagittal_T1_FLAIR_s3.nii.gz"
MESH_DIR = ROOT / "results/smoke_test_542/meshes"


def fig_mpr_overlay():
    fig = build_mpr_figure(volume_nifti=VOL, segmentation_nifti=SEG, seg_alpha=0.55,
                            figsize=(14, 5))
    out = FIGS / "mpr_overlay.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_mri_only():
    """The same panels without segmentation, for a 'before' visual."""
    fig = build_mpr_figure(volume_nifti=VOL, segmentation_nifti=None,
                            figsize=(14, 5))
    out = FIGS / "mri_only.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_postprocess_steps():
    """Visualize the 3-step post-processing on a single sagittal slice."""
    raw = np.asarray(nib.load(str(RAW)).dataobj).astype(np.uint8)

    # Apply each step in turn (mirroring spine_seg.postprocess.postprocess).
    lr_axis, si_axis = find_lr_si_axes(raw)
    after_strip = sagittal_strip_constraint(raw, lr_axis, fraction=0.20)
    after_cc = largest_connected_component_per_label(after_strip, min_voxels=50)
    final = postprocess(raw)  # full pipeline applied to the same raw

    img = nib.load(str(VOL))
    vol_can = nib.as_closest_canonical(img)
    vol = np.asarray(vol_can.dataobj).astype(np.float32)

    raw_can = nib.as_closest_canonical(nib.Nifti1Image(raw, nib.load(str(RAW)).affine))
    raw_arr = np.asarray(raw_can.dataobj).astype(np.uint8)
    strip_can = nib.as_closest_canonical(nib.Nifti1Image(after_strip, nib.load(str(RAW)).affine))
    strip_arr = np.asarray(strip_can.dataobj).astype(np.uint8)
    cc_can = nib.as_closest_canonical(nib.Nifti1Image(after_cc, nib.load(str(RAW)).affine))
    cc_arr = np.asarray(cc_can.dataobj).astype(np.uint8)
    final_can = nib.as_closest_canonical(nib.Nifti1Image(final, nib.load(str(RAW)).affine))
    final_arr = np.asarray(final_can.dataobj).astype(np.uint8)

    # Mid-sagittal slice (axis 0)
    sx = vol.shape[0]
    sag_idx = sx // 2

    def slc(a, idx):
        return np.flipud(a[idx, :, :].T)

    cmap = matplotlib.colors.ListedColormap(
        [(0, 0, 0, 0)] +
        [(LABEL_COLORS[l][0]/255, LABEL_COLORS[l][1]/255, LABEL_COLORS[l][2]/255, 1.0)
         for l in VERTEBRA_LABELS]
    )

    panels = [
        ("Raw nnU-Net output", raw_arr),
        ("After Step 1: Sagittal strip", strip_arr),
        ("After Step 2: Largest CC", cc_arr),
        ("After Step 3: SI ordering", final_arr),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor="#111111")
    for ax, (title, seg) in zip(axes, panels):
        bg = slc(vol, sag_idx)
        lo, hi = np.percentile(bg, [1, 99])
        bg_n = np.clip((bg - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(bg_n, cmap="gray", interpolation="bilinear", aspect="equal")
        seg_slc = slc(seg, sag_idx)
        mask = seg_slc > 0
        if mask.any():
            masked = np.ma.masked_where(~mask, seg_slc)
            ax.imshow(masked, cmap=cmap, vmin=0, vmax=5,
                      interpolation="nearest", alpha=0.55, aspect="equal")
        ax.set_title(title, color="#dddddd", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#444444")
        ax.set_facecolor("#111111")

    fig.tight_layout()
    out = FIGS / "postprocess_steps.png"
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_3d_meshes():
    """Render all 5 vertebra meshes in 3D using matplotlib's mplot3d."""
    fig = plt.figure(figsize=(10, 8), facecolor="#111111")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")

    for lbl in VERTEBRA_LABELS:
        path = MESH_DIR / f"L{lbl}.stl"
        if not path.exists():
            continue
        m = trimesh.load(str(path), force="mesh")
        verts = np.asarray(m.vertices)
        faces = np.asarray(m.faces)
        tris = verts[faces]
        color = tuple(c/255 for c in LABEL_COLORS[lbl])
        coll = Poly3DCollection(tris, alpha=0.95, facecolor=color, edgecolor="none",
                                 linewidth=0)
        ax.add_collection3d(coll)

    # Compute combined bounding box
    all_pts = []
    for lbl in VERTEBRA_LABELS:
        path = MESH_DIR / f"L{lbl}.stl"
        if path.exists():
            all_pts.append(trimesh.load(str(path), force="mesh").vertices)
    pts = np.vstack(all_pts)
    pad = 5
    ax.set_xlim(pts[:, 0].min()-pad, pts[:, 0].max()+pad)
    ax.set_ylim(pts[:, 1].min()-pad, pts[:, 1].max()+pad)
    ax.set_zlim(pts[:, 2].min()-pad, pts[:, 2].max()+pad)
    ax.set_box_aspect([np.ptp(pts[:, i]) for i in range(3)])

    ax.view_init(elev=15, azim=110)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for spine in ("x", "y", "z"):
        ax.xaxis.pane.set_facecolor("#111111")
        ax.yaxis.pane.set_facecolor("#111111")
        ax.zaxis.pane.set_facecolor("#111111")
        ax.xaxis.pane.set_edgecolor("#222222")
        ax.yaxis.pane.set_edgecolor("#222222")
        ax.zaxis.pane.set_edgecolor("#222222")

    # Color legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=tuple(c/255 for c in LABEL_COLORS[l]),
                     edgecolor="none", label=f"{LABELS[l]}")
               for l in VERTEBRA_LABELS]
    leg = ax.legend(handles=handles, loc="upper left", facecolor="#1a1a1a",
                    edgecolor="#444444", labelcolor="white")

    fig.tight_layout()
    out = FIGS / "mesh_3d.png"
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_voxel_counts_bar():
    """Bar chart of per-vertebra voxel counts in the smoke-test result."""
    counts = label_summary(np.asarray(nib.load(str(SEG)).dataobj).astype(np.uint8))
    labels = [LABELS[l] for l in VERTEBRA_LABELS]
    values = [counts[l] for l in VERTEBRA_LABELS]
    colors = [tuple(c/255 for c in LABEL_COLORS[l]) for l in VERTEBRA_LABELS]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111111")
    ax.set_facecolor("#1a1a1a")
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,}", ha="center", color="#dddddd", fontsize=11)
    ax.set_ylabel("Voxels", color="#dddddd", fontsize=12)
    ax.set_title("Per-vertebra voxel counts (cleaned segmentation)",
                 color="#dddddd", fontsize=13)
    ax.tick_params(colors="#dddddd")
    for s in ax.spines.values():
        s.set_color("#444444")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#333333", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(values) * 1.18)

    fig.tight_layout()
    out = FIGS / "voxel_counts.png"
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    print("Generating presentation figures into", FIGS)
    fig_mri_only()
    fig_mpr_overlay()
    fig_postprocess_steps()
    fig_3d_meshes()
    fig_voxel_counts_bar()
    print("Done.")


if __name__ == "__main__":
    main()
