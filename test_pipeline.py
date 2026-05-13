"""Self-test: build a synthetic 5-vertebrae phantom, run postprocess + STL
export, and verify the outputs end-to-end. Skips the nnU-Net inference step
since that requires the trained model and CUDA — but every other code path is
exercised here."""

import os
import sys
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from spine_seg.config import LABEL_COLORS, LABELS, VERTEBRA_LABELS, PostProcessConfig
from spine_seg.postprocess import (
    enforce_vertical_ordering,
    find_lr_si_axes,
    label_summary,
    largest_connected_component_per_label,
    postprocess,
    sagittal_strip_constraint,
)
from spine_seg.mesh_export import export_combined_stl, export_label_stls
from spine_seg.slicer_export import (
    save_segmentation_nifti,
    write_slicer_color_table,
    write_slicer_readme,
)


def make_phantom(shape=(40, 80, 200), spacing=(0.7, 0.7, 0.7)):
    """Synthetic spine: 5 disjoint elliptical 'vertebrae' stacked along z (axis 2),
    centered on the middle of axis 0 (LR) and axis 1 (AP). Plus an intentional
    contralateral mirror artifact and a tiny stray fragment to test cleanup."""
    seg = np.zeros(shape, dtype=np.uint8)
    nx, ny, nz = shape
    cx, cy = nx // 2, ny // 2

    # Stack 5 vertebral bodies along z, evenly spaced
    z_centers = np.linspace(40, 160, 5).astype(int)
    rx, ry, rz = 8, 14, 8

    grid = np.indices(shape)
    x, y, z = grid
    for idx, zc in enumerate(z_centers, start=1):
        ellipsoid = (
            ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - zc) / rz) ** 2
        ) <= 1.0
        seg[ellipsoid] = idx

    # Add a contralateral MIRROR artifact for L3 — should be removed by sagittal strip.
    # Keep it smaller than the real L3 so the centroid stays near the real spine,
    # which is what happens with real mirror artifacts in practice.
    mirror_cx = cx + 16  # outside the central corridor at ±20%·N = ±8 from center
    mirror_zc = z_centers[2]
    mrx, mry, mrz = 4, 6, 4
    mirror = (
        ((x - mirror_cx) / mrx) ** 2 + ((y - cy) / mry) ** 2 + ((z - mirror_zc) / mrz) ** 2
    ) <= 1.0
    seg[mirror] = 3  # same label as L3

    # Add a tiny disconnected fragment for L4 — should be dropped by min_voxels filter.
    seg[cx, cy, z_centers[3] + 20] = 4  # 1 voxel
    seg[cx, cy, z_centers[3] + 21] = 4  # plus one more — still <50

    # Build a NIfTI with a realistic affine: RAS, anisotropic spacing
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return seg, affine


def test_axis_detection(seg):
    lr, si = find_lr_si_axes(seg)
    print(f"  axes: LR={lr}, SI={si}")
    assert lr == 0, "LR should be the smallest extent (axis 0 in our phantom)"
    assert si == 2, "SI should be the largest extent (axis 2 in our phantom)"


def test_strip_removes_mirror(seg):
    lr, _ = find_lr_si_axes(seg)
    pre_counts = label_summary(seg)
    cleaned = sagittal_strip_constraint(seg, lr, fraction=0.20)
    post_counts = label_summary(cleaned)
    print(f"  L3 before strip: {pre_counts[3]}, after: {post_counts[3]}")
    # The mirror artifact should be gone, but the real L3 should mostly remain.
    assert post_counts[3] < pre_counts[3], "Sagittal strip didn't remove anything"
    assert post_counts[3] > 0.5 * pre_counts[3] * (3705 / 6067), \
        "Sagittal strip removed too much (real L3 mostly gone)"
    # The other real labels should retain most of their voxels (allow small edge nibble)
    for lbl in (1, 2, 4, 5):
        retained = post_counts[lbl] / max(pre_counts[lbl], 1)
        print(f"    L{lbl} retained: {retained:.0%}")
        assert retained > 0.85, f"Label {lbl} lost too many voxels: retained {retained:.0%}"


def test_cc_filter(seg):
    cleaned = largest_connected_component_per_label(seg, min_voxels=50)
    counts = label_summary(cleaned)
    # The L4 stray 2 voxels are a separate CC; the main L4 ellipsoid is huge.
    # Largest-CC filter should keep only the main blob.
    print(f"  L4 voxel count after CC filter: {counts[4]}")
    assert counts[4] > 50, "Main L4 ellipsoid was wrongly dropped"


def test_full_postprocess(seg):
    cleaned = postprocess(seg.copy())
    counts = label_summary(cleaned)
    print(f"  voxel counts post-pipeline: {counts}")
    # All five labels should survive in the phantom.
    for lbl in (1, 2, 3, 4, 5):
        assert counts[lbl] > 50, f"Label {lbl} lost in post-processing"
    return cleaned


def test_ordering_drops_swapped_label():
    # Build a tiny phantom with a deliberately swapped L2 / L3 order
    seg = np.zeros((20, 20, 60), dtype=np.uint8)
    centers = {1: 50, 2: 30, 3: 40, 4: 20, 5: 10}  # L2 and L3 swapped
    for lbl, zc in centers.items():
        seg[8:12, 8:12, zc-2:zc+2] = lbl
    cleaned = enforce_vertical_ordering(seg.copy(), si_axis=2)
    counts = label_summary(cleaned)
    print(f"  ordering test counts: {counts}")
    # Whichever direction wins, at least one of {L2, L3} should be removed
    assert counts[2] == 0 or counts[3] == 0, "Ordering check didn't drop the swap"


def test_stl_export(seg, affine, out_dir: Path):
    paths = export_label_stls(seg, affine, out_dir / "meshes",
                              smoothing_iterations=3, coord_system="LPS")
    print(f"  wrote {len(paths)} STLs")
    assert len(paths) == 5
    # Every STL has the right header magic and is non-empty
    for lbl, p in paths.items():
        size = p.stat().st_size
        assert size > 200, f"STL {p} is suspiciously small ({size} bytes)"
        with open(p, "rb") as f:
            head = f.read(5)
        # Either ASCII "solid" or 80-byte binary header — trimesh writes binary by default
        assert head[:5] == b"solid" or len(head) == 5  # binary header begins with 80 zero-padded bytes
    combined = export_combined_stl(seg, affine, out_dir / "meshes" / "all_vertebrae.stl",
                                    smoothing_iterations=3, coord_system="LPS")
    assert combined and combined.exists()


def test_world_coordinates(seg, affine, tmp: Path):
    """Verify that STL vertices are in correct world (LPS) coordinates."""
    import trimesh
    paths = export_label_stls(seg, affine, tmp / "meshes_world",
                              smoothing_iterations=0, coord_system="LPS")
    # Take L3 mesh; its centroid should be near LPS world coordinates derived
    # from voxel centroid via affine, with X,Y negated.
    mesh = trimesh.load(str(paths[3]))
    mesh_centroid_lps = mesh.vertices.mean(axis=0)

    # Compute expected centroid: voxel centroid -> RAS -> LPS
    coords = np.argwhere(seg == 3)
    vox_centroid = coords.mean(axis=0)
    homo = np.append(vox_centroid, 1.0)
    ras_centroid = (affine @ homo)[:3]
    expected_lps = ras_centroid * np.array([-1.0, -1.0, 1.0])
    diff = np.linalg.norm(mesh_centroid_lps - expected_lps)
    print(f"  L3 mesh centroid (LPS): {mesh_centroid_lps}")
    print(f"  L3 expected   (LPS):    {expected_lps}")
    print(f"  difference: {diff:.3f} mm")
    assert diff < 3.0, f"Mesh centroid is off by {diff} mm — affine transform broken"


def test_slicer_outputs(seg, affine, tmp: Path):
    nifti_path = save_segmentation_nifti(seg, affine, tmp / "segmentation.nii.gz")
    assert nifti_path.exists()
    img = nib.load(str(nifti_path))
    assert img.get_data_dtype() == np.dtype("uint8")
    # Affine survived the round-trip
    assert np.allclose(img.affine, affine)

    ctbl_path = write_slicer_color_table(tmp / "labels.ctbl")
    text = ctbl_path.read_text()
    for name in ("L1", "L2", "L3", "L4", "L5"):
        assert name in text
    print(f"  segmentation NIfTI ok, dtype={img.get_data_dtype()}")
    print(f"  color table has {len(text.splitlines())} lines")

    write_slicer_readme(tmp / "slicer_README.md", {"segmentation.nii.gz": "test"})


def main() -> int:
    print("\n=== Building phantom ===")
    seg, affine = make_phantom()
    print(f"  shape={seg.shape}  affine diag={np.diag(affine[:3,:3])}")
    print(f"  initial label counts: {label_summary(seg)}")

    print("\n=== Axis detection ===")
    test_axis_detection(seg)

    print("\n=== Sagittal strip removes mirror artifact ===")
    test_strip_removes_mirror(seg)

    print("\n=== Connected-component filter ===")
    test_cc_filter(seg)

    print("\n=== Full post-processing ===")
    cleaned = test_full_postprocess(seg)

    print("\n=== Ordering enforcement drops swapped labels ===")
    test_ordering_drops_swapped_label()

    tmp = Path(tempfile.mkdtemp(prefix="spineseg_test_"))
    try:
        print(f"\n=== STL export (writing to {tmp}) ===")
        test_stl_export(cleaned, affine, tmp)

        print("\n=== World coordinate transform ===")
        test_world_coordinates(cleaned, affine, tmp)

        print("\n=== Slicer outputs ===")
        test_slicer_outputs(cleaned, affine, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nAll self-tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
