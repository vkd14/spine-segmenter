"""Streamlit GUI for the lumbar L1-L5 vertebral body segmentation app.

Run with:
    streamlit run app.py

Workflow:
  1. Drop ONE or MANY .nii / .nii.gz sagittal T1 FLAIR volumes.
  2. Pick fold mode (1-fold fast preview vs 5-fold ensemble).
  3. Click "Run inference" -> for each case, the pipeline runs end-to-end and
     writes results into ./results/<case_id>/.
  4. Inspect outputs in the in-page NiiVue 3D viewer (MPR + 3D + STL surfaces).
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List

# Make spine_seg importable regardless of CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import streamlit as st

from spine_seg.config import LABEL_COLORS, LABELS, ModelConfig, PostProcessConfig
from spine_seg.pipeline import PipelineResult, run_pipeline
from spine_seg.viewer import build_mesh_figure, build_mpr_figure, load_canonical


@st.cache_data(show_spinner=False)
def _cached_canonical(vol_path: str, seg_path: str, vol_mtime: float, seg_mtime: float):
    """Cache the reorientation + load so slider changes don't re-read NIfTIs.

    mtime args are part of the cache key so the cache invalidates if files change.
    """
    return load_canonical(Path(vol_path), Path(seg_path))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# -----------------------------------------------------------------------------
# Project-relative defaults
# -----------------------------------------------------------------------------
DEFAULT_MODEL_DIR = _HERE / "nnUNet_results"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Lumbar Spine Segmentation",
    page_icon="🦴",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Sidebar: configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Model")
default_model_dir = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() else os.environ.get("nnUNet_results", "")
model_dir = st.sidebar.text_input(
    "nnUNet_results path",
    value=default_model_dir,
    help="Parent directory containing Dataset100_SpineL1L5/.",
)

mode = st.sidebar.radio(
    "Inference mode",
    options=["1-fold (fast preview)", "5-fold ensemble (best quality)"],
    index=0,
    help="On Apple Silicon (M1 Max), 1-fold takes ~3-6 min/case; 5-fold takes ~15-25 min/case.",
)
folds = (0,) if mode.startswith("1-fold") else (0, 1, 2, 3, 4)

device_pref = st.sidebar.selectbox("Device", ["auto", "mps", "cpu", "cuda"], index=0)

st.sidebar.header("Post-processing")
sagittal_fraction = st.sidebar.slider(
    "Sagittal strip fraction (± N · LR-axis)",
    min_value=0.05, max_value=0.40,
    value=PostProcessConfig.sagittal_strip_fraction, step=0.01,
)
min_voxels = st.sidebar.number_input(
    "Min component voxels",
    min_value=10, max_value=5000,
    value=PostProcessConfig.min_component_voxels, step=10,
)

st.sidebar.header("Mesh export")
coord_system = st.sidebar.selectbox("Coordinate system", ["LPS", "RAS"], index=0)
smoothing = st.sidebar.slider("Taubin smoothing iterations", 0, 30, 10)
combined_stl = st.sidebar.checkbox("Also write combined all_vertebrae.stl", value=True)

st.sidebar.header("Output")
custom_out_root = st.sidebar.text_input(
    "Output root directory",
    value=str(RESULTS_DIR),
    help="One subfolder per case is created here.",
)

# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
st.title("🦴 Lumbar L1-L5 Vertebral Body Segmentation")
st.caption(
    "Sagittal T1 FLAIR -> nnU-Net ResEncUNet -> 3-step anatomical post-processing -> "
    "multi-label NIfTI + per-vertebra STLs + in-app 3D viewer."
)

uploaded_files = st.file_uploader(
    "Upload one or more sagittal T1 FLAIR volumes (.nii or .nii.gz)",
    type=["nii", "gz"],
    accept_multiple_files=True,
    help="Drag and drop multiple files at once for batch processing.",
)

run_clicked = st.button(
    "Run inference",
    disabled=not uploaded_files or not model_dir,
    type="primary",
)


def _legend() -> None:
    cols = st.columns(5)
    for col, lbl in zip(cols, (1, 2, 3, 4, 5)):
        r, g, b = LABEL_COLORS[lbl]
        col.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:18px;height:18px;background:rgb({r},{g},{b});"
            f"border-radius:3px;'></div><strong>{LABELS[lbl]}</strong></div>",
            unsafe_allow_html=True,
        )


_legend()
st.warning(
    "**Modality note:** the model is trained on **sagittal T1 FLAIR** only. "
    "Other contrasts (T2, STIR) will produce poor results."
)


def _save_upload_to_tempfile(uploaded_file) -> Path:
    """Persist the uploaded bytes to a real .nii.gz so nnU-Net can read it."""
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    name = Path(uploaded_file.name).name
    # Preserve the user's filename so case_id matches.
    work = Path(tempfile.mkdtemp(prefix="spine_seg_in_"))
    out = work / name
    out.write_bytes(uploaded_file.getbuffer())
    return out


def _zip_directory(src: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(src))
    return buf.getvalue()


def _render_result(input_volume: Path, result: PipelineResult) -> None:
    st.success(
        f"**{result.case_id}** -> `{result.output_dir}`  "
        f"(folds={list(result.folds_used)}, device={result.device})"
    )

    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("**Multi-planar view** (sagittal / coronal / axial — drag sliders to scrub through slices)")
        try:
            preloaded = _cached_canonical(
                str(input_volume),
                str(result.cleaned_segmentation),
                input_volume.stat().st_mtime,
                result.cleaned_segmentation.stat().st_mtime,
            )
            vol, seg, spacing, default_center = preloaded
            sx, sy, sz = vol.shape  # canonical RAS: (LR, PA, IS)
            slider_cols = st.columns(3)
            with slider_cols[0]:
                sag_i = st.slider(
                    "Sagittal slice (L↔R)",
                    min_value=0, max_value=sx - 1,
                    value=int(default_center[0]),
                    key=f"sag_{result.case_id}",
                )
            with slider_cols[1]:
                cor_i = st.slider(
                    "Coronal slice (P↔A)",
                    min_value=0, max_value=sy - 1,
                    value=int(default_center[1]),
                    key=f"cor_{result.case_id}",
                )
            with slider_cols[2]:
                ax_i = st.slider(
                    "Axial slice (I↔S)",
                    min_value=0, max_value=sz - 1,
                    value=int(default_center[2]),
                    key=f"ax_{result.case_id}",
                )

            mpr_fig = build_mpr_figure(
                preloaded=preloaded,
                slice_indices=(sag_i, cor_i, ax_i),
                seg_alpha=0.5,
            )
            st.pyplot(mpr_fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"MPR view unavailable: {exc}")
            st.exception(exc)

        st.markdown("**3D mesh view** (drag = rotate, scroll = zoom, click legend to toggle vertebrae)")
        try:
            stl_pairs = [(lbl, p) for lbl, p in result.stl_per_label.items()]
            mesh_fig = build_mesh_figure(stl_pairs, height_px=600)
            st.plotly_chart(mesh_fig, use_container_width=True, key=f"mesh_{result.case_id}")
        except Exception as exc:
            st.warning(f"3D mesh viewer unavailable: {exc}")
            st.exception(exc)

    with cols[1]:
        st.markdown("**Per-label voxel counts**")
        st.json(result.label_voxel_counts)

        st.markdown("**Output bundle**")
        zip_bytes = _zip_directory(result.output_dir)
        st.download_button(
            "Download all outputs (.zip)",
            data=zip_bytes,
            file_name=f"{result.case_id}_spine_seg.zip",
            mime="application/zip",
            key=f"zip_{result.case_id}",
        )
        with open(result.cleaned_segmentation, "rb") as f:
            st.download_button(
                "segmentation.nii.gz only",
                data=f.read(),
                file_name=f"{result.case_id}_segmentation.nii.gz",
                mime="application/gzip",
                key=f"seg_{result.case_id}",
            )
        for lbl, stl_path in result.stl_per_label.items():
            with open(stl_path, "rb") as f:
                st.download_button(
                    f"{LABELS[lbl]}.stl",
                    data=f.read(),
                    file_name=f"{result.case_id}_{LABELS[lbl]}.stl",
                    mime="model/stl",
                    key=f"stl_{result.case_id}_{lbl}",
                )


# -----------------------------------------------------------------------------
# Run pipeline
# -----------------------------------------------------------------------------
# Persist results across reruns: every slider / button / file-upload click
# reruns the entire script, so we can't render results inline inside the
# `if run_clicked:` block — they'd vanish on the next interaction. Instead we
# store completed runs in session_state and only clear when the user clicks
# Run inference again.
if "completed_results" not in st.session_state:
    st.session_state.completed_results = []  # list of (input_path_str, PipelineResult, error_str_or_None)
if "last_run_summary" not in st.session_state:
    st.session_state.last_run_summary = None

if run_clicked and uploaded_files and model_dir:
    # User explicitly asked for a new run — wipe previous results.
    st.session_state.completed_results = []
    st.session_state.last_run_summary = None

    pp_cfg = PostProcessConfig(
        sagittal_strip_fraction=float(sagittal_fraction),
        min_component_voxels=int(min_voxels),
    )
    model_cfg = ModelConfig()
    out_root = Path(custom_out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n = len(uploaded_files)
    overall = st.progress(0.0, text=f"Starting (0/{n})")
    case_progress = st.progress(0.0, text="—")
    log_box = st.empty()

    def _make_progress_callbacks(case_idx: int, case_label: str):
        """Build (tqdm_callback, stage_callback) bound to the current case progress bar.

        nnU-Net's tqdm bars dominate the runtime; stage_callback handles the
        post-processing / mesh / export tail. We map both to a single 0..1 bar.
        """
        # Inference takes ~85% of runtime on M1; the remaining 15% is
        # post-processing + STL extraction + Slicer files.
        infer_max = 0.83

        def tqdm_cb(cur: int, total: int, desc: str) -> None:
            if total <= 0:
                return
            frac = min(cur / total, 1.0) * infer_max
            label = desc.strip() or "predicting"
            text = f"[{case_idx}/{n}] {case_label} - {label} {cur}/{total} ({int(frac*100)}%)"
            try:
                case_progress.progress(frac, text=text)
            except Exception:
                pass

        def stage_cb(name: str, frac: float) -> None:
            text = f"[{case_idx}/{n}] {case_label} - {name} ({int(frac*100)}%)"
            try:
                case_progress.progress(min(max(frac, 0.0), 1.0), text=text)
            except Exception:
                pass

        return tqdm_cb, stage_cb

    for i, uploaded in enumerate(uploaded_files, start=1):
        log_box.info(f"[{i}/{n}] Saving {uploaded.name}…")
        input_path = _save_upload_to_tempfile(uploaded)
        case_id = Path(uploaded.name).name.replace(".nii.gz", "").replace(".nii", "")
        case_out = out_root / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        overall.progress(
            (i - 1) / n,
            text=f"[{i}/{n}] Processing {uploaded.name} (folds={list(folds)})",
        )
        tqdm_cb, stage_cb = _make_progress_callbacks(i, uploaded.name)
        case_progress.progress(0.01, text=f"[{i}/{n}] {uploaded.name} - starting…")

        try:
            result = run_pipeline(
                input_nifti=input_path,
                model_root=Path(model_dir),
                output_dir=case_out,
                model_cfg=model_cfg,
                pp_cfg=pp_cfg,
                coord_system=coord_system,
                smoothing_iterations=smoothing,
                write_combined_stl=combined_stl,
                keep_raw_prediction=True,
                device_pref=None if device_pref == "auto" else device_pref,
                folds=folds,
                progress_callback=tqdm_cb,
                stage_callback=stage_cb,
            )
            st.session_state.completed_results.append(
                (str(input_path), result, None)
            )
            case_progress.progress(1.0, text=f"[{i}/{n}] {uploaded.name} - done (100%)")
        except Exception as exc:
            import traceback
            st.session_state.completed_results.append(
                (str(input_path), None,
                 f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
            )
            case_progress.progress(1.0, text=f"[{i}/{n}] {uploaded.name} - FAILED")

        overall.progress(i / n, text=f"[{i}/{n}] Done")

    st.session_state.last_run_summary = (
        f"Finished {n} case(s). Outputs saved under `{out_root}`."
    )
    log_box.success(st.session_state.last_run_summary)

# -----------------------------------------------------------------------------
# Always render persisted results (survives slider / button reruns)
# -----------------------------------------------------------------------------
if st.session_state.completed_results:
    if st.session_state.last_run_summary and not run_clicked:
        # Show the summary as a quiet reminder, not a fresh-from-this-rerun toast.
        st.success(st.session_state.last_run_summary)

    st.divider()
    st.subheader("Results")
    for input_path_str, result, error in st.session_state.completed_results:
        if error is not None:
            st.error(f"`{Path(input_path_str).name}`: pipeline failed")
            with st.expander("Error details"):
                st.code(error)
        else:
            _render_result(Path(input_path_str), result)

    st.info(
        "Open in 3D Slicer (optional): drag `segmentation.nii.gz` onto the Slicer "
        "window and pick **Segmentation**. Drag any STL to load as **Model** (LPS)."
    )
